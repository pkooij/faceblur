"""Keyframe-only face scan: find which files (and which frames) need blurring, without decoding every frame.

Supports hardware NVDEC keyframe-only decoding (`nvdec > 0`, fastest on H100/B300: separate fetcher processes stream
and pack Annex-B IDR keyframes into shared memory with zero GIL contention, and NVDEC decodes directly into GPU NV12
VRAM), CPU libavcodec decoding (`decoders > 0`), or both in parallel (hybrid mode) sharing one node-wide unit queue.

Work units come from `faceblur units` (faceblur/units.py): ranges of at most 2,000 keyframes, largest first, shared
by every GPU of the node through one counter file, so all GPUs finish within seconds of each other.

Output (`--report`): one JSON line per file with its frame and keyframe counts and every detection as
[presentation frame, x1, y1, x2, y2, detector score, verifier score]. Detections are kept down to the detector
threshold, so the verifier threshold can be chosen afterwards.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory

import numpy as np

SLOT = 1024 * 1024 * 3 // 2  # bytes per uncompressed frame slot: yuv420p up to 1024x1024
NV_SLOT = 64 << 20           # bytes per compressed Annex-B unit slot (64 MiB)
DECODER_BOXES = ((b"avcC", "h264"), (b"hvcC", "hevc"), (b"av1C", "av1"), (b"vpcC", "vp9"))


def decoder_config(stsd: bytes) -> tuple[str, bytes]:
    """Codec name and decoder extradata from the payload of an MP4 `stsd` box."""
    for tag, codec in DECODER_BOXES:
        p = stsd.find(tag)
        if p >= 4:
            n = int.from_bytes(stsd[p - 4:p], "big")
            return codec, bytes(stsd[p + 4:p - 4 + n])
    raise ValueError("no avcC/hvcC/av1C/vpcC decoder config in the stsd box")


def load_unit(u: dict, parts: int = 4):
    """Read one unit's MP4 index and the byte span holding its keyframes. Returns (index, decode-order sample ids,
    offsets relative to the span, sizes, span bytes)."""
    from .units import index, sync_samples
    rr, idx = index(u["path"], parts)
    sync = sync_samples(idx)
    step = max(1, int(u.get("step", 1)))
    sel = sync[u["lo"]:len(sync) if u["hi"] < 0 else u["hi"]:step]
    offs = np.asarray(idx.sample_offsets, np.int64)[sel]
    sizes = np.asarray(idx.sample_sizes, np.int64)[sel]
    lo, hi = int(offs.min()), int((offs + sizes).max())
    return idx, sel, offs - lo, sizes, rr(u["path"], lo, hi - lo)


def pres_map(idx) -> np.ndarray:
    """Decode-order sample id -> presentation-order frame index."""
    n = len(idx.sample_sizes)
    pres = np.empty(n, np.int64)
    pres[np.argsort(np.asarray(idx.sample_pts), kind="stable")] = np.arange(n)
    return pres


def decode_unit(idx, sel, roffs, sizes, data, skip_loop_filter: bool = False) -> Iterator[tuple[int, np.ndarray]]:
    """Decode the selected keyframes on the CPU. Yields (presentation frame index, yuv420p array [1.5h, w])."""
    import av
    codec, extra = decoder_config(idx.stsd_body)
    cc = av.CodecContext.create(codec, "r")
    cc.extradata = extra
    cc.thread_count = 1
    if skip_loop_filter:
        cc.options = {"skip_loop_filter": "all"}
    pres = pres_map(idx)
    for m in range(len(sel)):
        o = int(roffs[m])
        pk = av.Packet(data[o:o + int(sizes[m])])
        pk.pts = int(sel[m])  # carries the decode-order sample id through the decoder
        for fr in cc.decode(pk):
            yield int(pres[int(fr.pts)]), fr.to_ndarray(format="yuv420p")
    for fr in cc.decode(None):
        yield int(pres[int(fr.pts)]), fr.to_ndarray(format="yuv420p")


def avcc_unit_to_annexb(stsd_body: bytes, sel, roffs, sizes, data) -> list[tuple[int, bytes]]:
    """Convert an H.264 unit's AVCC IDR keyframe packets into Annex-B (SPS/PPS prepended to the first packet)."""
    from .splice import avcc_nals, parse_avcc_extradata
    codec, extra = decoder_config(stsd_body)
    if codec != "h264":
        raise ValueError(f"nvdec keyframe path supports h264 only (got {codec})")
    ls, sps, pps = parse_avcc_extradata(extra)
    hdr = b"".join(b"\x00\x00\x00\x01" + x for x in sps + pps)
    out = []
    for m in range(len(sel)):
        o = int(roffs[m])
        nals = avcc_nals(data[o:o + int(sizes[m])], ls)
        body = b"".join(b"\x00\x00\x00\x01" + n for n in nals)
        out.append((int(sel[m]), (hdr + body) if m == 0 else body))
    return out


def avcc_unit_to_annexb_flat(stsd_body: bytes, sel, roffs, sizes, data) -> tuple[bytearray, np.ndarray]:
    """Pack an H.264 unit's AVCC IDR keyframes into one contiguous Annex-B bytearray + int32 packet lengths."""
    from .splice import parse_avcc_extradata
    codec, extra = decoder_config(stsd_body)
    if codec != "h264":
        raise ValueError(f"nvdec keyframe path supports h264 only (got {codec})")
    ls, sps, pps = parse_avcc_extradata(extra)
    hdr = b"".join(b"\x00\x00\x00\x01" + x for x in sps + pps)
    mv = memoryview(data)
    out = bytearray()
    lengths = np.empty(len(sel), dtype=np.int32)
    sc = b"\x00\x00\x00\x01"
    for m in range(len(sel)):
        start = len(out)
        if m == 0:
            out += hdr
        o = int(roffs[m])
        end = o + int(sizes[m])
        i = o
        while i + ls <= end:
            n = int.from_bytes(mv[i:i + ls], "big")
            i += ls
            out += sc
            out += mv[i:i + n]
            i += n
        lengths[m] = len(out) - start
    return out, lengths


def claim(path: str, n: int) -> int | None:
    """Next unit index from a node-wide counter file (fcntl lock), or None when all n units are taken."""
    import fcntl
    with open(path, "r+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        v = int(f.read().strip() or 0)
        f.seek(0)
        f.write(str(v + 1).encode())
        f.truncate()
    return v if v < n else None


def _nv_fetcher(cfg: dict, freeq, readyq, shm_name: str) -> None:
    """Separate OS process: fetches Hub/local units and packs Annex-B keyframes into shared memory with zero GPU GIL."""
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((cfg["nv_slots"], NV_SLOT), np.uint8, buffer=shm.buf)
    units = [json.loads(line) for line in open(cfg["units"])]

    def run_thread():
        while (uid := claim(cfg["claim"], len(units))) is not None:
            t, err = time.time(), None
            for attempt in range(3):
                try:
                    loaded = load_unit(units[uid], cfg["parts"])
                    break
                except Exception as e:  # noqa: BLE001 - retried, then reported
                    err = e
                    time.sleep(2 * (attempt + 1))
            else:
                readyq.put(("err", uid, repr(err)[:300]))
                continue
            fetch_s = round(time.time() - t, 2)
            idx, sel, roffs, sizes, data = loaded
            n_frames = len(idx.sample_sizes)
            span_bytes = len(data)
            try:
                raw, lengths = avcc_unit_to_annexb_flat(idx.stsd_body, sel, roffs, sizes, data)
                pres_sel = pres_map(idx)[sel].astype(np.int32)
            except Exception:  # noqa: BLE001 - non-H264 falls back to CPU decode
                readyq.put(("cpu", uid, fetch_s))
                del loaded, data
                continue
            nbytes = len(raw)
            if nbytes <= NV_SLOT:
                slot = freeq.get()
                buf[slot, :nbytes] = np.frombuffer(raw, dtype=np.uint8)
                readyq.put(("nv", uid, slot, nbytes, lengths, pres_sel, n_frames, span_bytes, fetch_s))
            else:
                readyq.put(("nv_inline", uid, bytes(raw), lengths, pres_sel, n_frames, span_bytes, fetch_s))
            del loaded, data, raw

    ts = [threading.Thread(target=run_thread, daemon=True) for _ in range(cfg["fetchers"])]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    readyq.put(None)
    shm.close()


def _decoder(w: int, cfg: dict, relq, outq, shm_name: str) -> None:
    """CPU decoder process: prefetch threads claim and read units, main thread decodes into shared-memory slots."""
    shm = shared_memory.SharedMemory(name=shm_name)
    k = cfg["slots"]
    buf = np.ndarray((cfg["decoders"] * k, SLOT), np.uint8, buffer=shm.buf)
    free = deque(range(w * k, (w + 1) * k))
    units = [json.loads(line) for line in open(cfg["units"])]
    pq: queue.Queue = queue.Queue(maxsize=cfg["prefetch"])
    pending: list = []

    def prefetch():
        while (uid := claim(cfg["claim"], len(units))) is not None:
            t, err = time.time(), None
            for attempt in range(3):
                try:
                    pq.put((uid, load_unit(units[uid], cfg["parts"]), time.time() - t))
                    break
                except Exception as e:  # noqa: BLE001 - retried, then reported
                    err = e
                    time.sleep(2 * (attempt + 1))
            else:
                pq.put((uid, None, repr(err)[:300]))
        pq.put(None)

    def slot() -> int:
        while not free:
            free.extend(relq.get())
        return free.popleft()

    def push(force=False):
        if pending and (force or len(pending) >= 32):
            outq.put(list(pending))
            pending.clear()

    for _ in range(cfg["fetchers"]):
        threading.Thread(target=prefetch, daemon=True).start()
    ends = 0
    while ends < cfg["fetchers"]:
        job = pq.get()
        if job is None:
            ends += 1
            continue
        uid, loaded, info = job
        u = units[uid]
        if loaded is None:
            outq.put(("err", uid, info))
            continue
        t0, nkey = time.time(), 0
        idx, sel, roffs, sizes, data = loaded
        try:
            for pres, arr in decode_unit(idx, sel, roffs, sizes, data, skip_loop_filter=cfg.get("skip_loop_filter", True)):
                if arr.size > SLOT:
                    raise ValueError(f"frame {arr.shape} larger than a {SLOT}-byte slot (max 1024x1024)")
                s = slot()
                buf[s, :arr.size] = arr.reshape(-1)
                pending.append((u["fid"], pres, arr.shape, s))
                nkey += 1
                push()
            push(True)
            n = len(idx.sample_sizes)
            outq.put(("done", uid, {"fid": u["fid"], "frames": n if u["first"] else 0, "keyframes": nkey,
                                    "bytes": len(data), "fetch_s": round(info, 2), "dec_s": round(time.time() - t0, 2)}))
        except Exception as e:  # noqa: BLE001 - reported per unit
            push(True)
            outq.put(("err", uid, repr(e)[:300]))
        del data, loaded
    outq.put(None)


def scan(units_path: str, report: str, claim_path: str, decoders: int = 0, slots: int = 64, gpu_threads: int = 2,
         nvdec: int = 8, parts: int = 4, fetchers: int = 2, prefetch: int = 2, det_thr: float = 0.4, ver_thr: float = 0.8,
         skip_loop_filter: bool = True, log: Callable[[dict], None] | None = None) -> dict:
    """Scan the units in `units_path` on the current CUDA device. Several processes (one per GPU) may share one
    `claim_path`; each writes its own `report`.

    `nvdec` sets hardware NVDEC keyframe worker threads per GPU (frames stay in VRAM, fed by separate fetcher OS
    processes so HTTP/Annex-B work never contends on the GPU process GIL); `decoders` sets parallel CPU decoder
    processes per GPU. Either or both may be non-zero.
    """
    import multiprocessing as mp

    import torch

    from . import gpu
    from .models import Detector, Verifier

    if nvdec <= 0 and decoders <= 0:
        raise ValueError("at least one of nvdec or decoders must be > 0")

    units = [json.loads(line) for line in open(units_path)]
    files = {u["fid"]: u["path"] for u in units}
    t0 = time.time()

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.cuda.current_device()
    det, ver = Detector(), Verifier()
    B = 128

    vq: queue.Queue = queue.Queue()
    dets: dict[int, list] = {}
    done: dict[int, dict] = {}
    errs: list = []
    lock = threading.Lock()
    st = {"keyframes": 0, "crops": 0, "batches": 0, "det_busy_s": 0.0, "ver_busy_s": 0.0, "nvdec_keyframes": 0, "cpu_keyframes": 0}
    t_first = [None]

    def verify(n_producers: int):
        torch.cuda.set_device(dev)
        stream = torch.cuda.Stream()
        pend: list = []
        npend, ends = 0, 0

        def run():
            nonlocal pend, npend
            if not pend:
                return
            tv = time.time()
            with torch.cuda.stream(stream):
                for c, _, ev in pend:
                    stream.wait_event(ev)
                    c.record_stream(stream)
                scores = ver(torch.cat([p[0] for p in pend])).tolist()
            meta = [m for p in pend for m in p[1]]
            with lock:
                for (fid, pr, b, s), v in zip(meta, scores):
                    dets.setdefault(fid, []).append([pr, *b, s, round(v, 3)])
                st["crops"] += len(meta)
                st["ver_busy_s"] += time.time() - tv
            pend, npend = [], 0

        while True:
            try:
                m = vq.get(timeout=0.2)
            except queue.Empty:
                run()
                continue
            if m is None:
                ends += 1
                if ends == n_producers:
                    run()
                    return
                continue
            pend.append(m)
            npend += len(m[1])
            if npend >= 512:
                run()

    def run_nv12_batch(nv: torch.Tensor, h: int, w: int, meta_items: list[tuple[int, int]], dctx, stream, is_nvdec: bool):
        td = time.time()
        with torch.cuda.stream(stream):
            y = dctx(gpu.detector_input(nv, h), stream)
            fb, bx, sc = gpu.yolo_postproc(y, det_thr, h, w)
            if len(fb):
                crops = gpu.verifier_crops(nv, h, fb, bx, ver.size)
                ev = torch.cuda.Event()
                ev.record(stream)
                meta = [(meta_items[f][0], meta_items[f][1], [round(c, 1) for c in b], round(s, 3))
                        for f, b, s in zip(fb.tolist(), bx.tolist(), sc.tolist())]
                vq.put((crops, meta, ev))
        with lock:
            st["det_busy_s"] += time.time() - td
            st["keyframes"] += len(meta_items)
            st["batches"] += 1
            st["nvdec_keyframes" if is_nvdec else "cpu_keyframes"] += len(meta_items)

    # ---------------------------------------------------------------- NVDEC worker threads (GIL-free MP fetchers)
    nv_shm = None
    nv_procs: list = []
    nvdec_threads: list[threading.Thread] = []
    nv_det_threads: list[threading.Thread] = []
    if nvdec > 0:
        import PyNvVideoCodec as nvc

        nv_slots = max(nvdec * prefetch + 8, 24)
        n_mp_fetch = max(2, min(4, nvdec // 2))
        nv_ctx = mp.get_context("spawn")
        nv_shm = shared_memory.SharedMemory(create=True, size=nv_slots * NV_SLOT)
        nv_buf = np.ndarray((nv_slots, NV_SLOT), np.uint8, buffer=nv_shm.buf)
        nv_freeq = nv_ctx.Queue()
        for s_i in range(nv_slots):
            nv_freeq.put(s_i)
        nv_readyq = nv_ctx.Queue()
        nv_cfg = {"units": units_path, "claim": claim_path, "parts": parts, "fetchers": max(fetchers * 2, 4),
                  "nv_slots": nv_slots}
        nv_procs = [nv_ctx.Process(target=_nv_fetcher, args=(nv_cfg, nv_freeq, nv_readyq, nv_shm.name), daemon=True)
                    for _ in range(n_mp_fetch)]
        for p in nv_procs:
            p.start()

        nv_wq: queue.Queue = queue.Queue(maxsize=max(nvdec * 2, 8))
        nv_bq: queue.Queue = queue.Queue(maxsize=max(gpu_threads * 4, 8))

        def nv_dispatch():
            ends = 0
            while ends < n_mp_fetch:
                item = nv_readyq.get()
                if item is None:
                    ends += 1
                    continue
                nv_wq.put(item)
            for _ in range(nvdec):
                nv_wq.put(None)

        def nv_detect():
            torch.cuda.set_device(dev)
            dctx, stream = det.context(), torch.cuda.Stream()
            while (job := nv_bq.get()) is not None:
                nv, h, w, meta_items, ev, is_nvdec = job
                with torch.cuda.stream(stream):
                    if ev is not None:
                        stream.wait_event(ev)
                        nv.record_stream(stream)
                run_nv12_batch(nv, h, w, meta_items, dctx, stream, is_nvdec=is_nvdec)
            vq.put(None)

        def _push_cpu_fallback_batch(frames_list, h, w, meta_items, stream):
            n = len(frames_list)
            with torch.cuda.stream(stream):
                x = torch.from_numpy(np.stack(frames_list)).to("cuda", non_blocking=True)
                q = (h // 2) * (w // 2)
                Y = x[:, :h * w].view(n, h, w)
                U = x[:, h * w:h * w + q].view(n, h // 2, w // 2)
                V = x[:, h * w + q:h * w + 2 * q].view(n, h // 2, w // 2)
                nv = torch.cat([Y, torch.stack([U, V], -1).view(n, h // 2, w)], 1)
                ev = torch.cuda.Event()
                ev.record(stream)
            nv_bq.put((nv, h, w, meta_items, ev, False))

        def _cpu_fallback_unit(uid: int, fetch_s: float, stream):
            u = units[uid]
            t_u = time.time()
            idx, sel, roffs, sizes, data = load_unit(u, parts)
            cpu_frames, cpu_meta = [], []
            bh = bw = nkey = 0
            for pr, arr in decode_unit(idx, sel, roffs, sizes, data, skip_loop_filter=skip_loop_filter):
                r, w = arr.shape
                h = r * 2 // 3
                if cpu_frames and (h, w) != (bh, bw):
                    _push_cpu_fallback_batch(cpu_frames, bh, bw, cpu_meta, stream)
                    cpu_frames, cpu_meta = [], []
                bh, bw = h, w
                cpu_frames.append(arr.reshape(-1))
                cpu_meta.append((u["fid"], pr))
                nkey += 1
                if t_first[0] is None:
                    t_first[0] = time.time()
                if len(cpu_frames) >= B:
                    _push_cpu_fallback_batch(cpu_frames, bh, bw, cpu_meta, stream)
                    cpu_frames, cpu_meta = [], []
            if cpu_frames:
                _push_cpu_fallback_batch(cpu_frames, bh, bw, cpu_meta, stream)
            with lock:
                done[uid] = {"fid": u["fid"], "frames": len(idx.sample_sizes) if u["first"] else 0, "keyframes": nkey,
                             "bytes": len(data), "fetch_s": fetch_s, "dec_s": round(time.time() - t_u, 2)}

        def nvdec_worker():
            torch.cuda.set_device(dev)
            stream = torch.cuda.Stream()
            while (job := nv_wq.get()) is not None:
                kind = job[0]
                if kind == "err":
                    _, uid, msg = job
                    with lock:
                        errs.append({"src": units[uid]["path"], "unit": uid, "error": msg})
                    continue
                if kind == "cpu":
                    _, uid, fetch_s = job
                    try:
                        _cpu_fallback_unit(uid, fetch_s, stream)
                    except Exception as e:  # noqa: BLE001
                        with lock:
                            errs.append({"src": units[uid]["path"], "unit": uid, "error": repr(e)[:300]})
                    continue

                slot = None
                if kind == "nv":
                    _, uid, slot, nbytes, lengths, pres_sel, n_frames, span_bytes, fetch_s = job
                    raw_arr = nv_buf[slot, :nbytes]
                else:
                    _, uid, raw_bytes, lengths, pres_sel, n_frames, span_bytes, fetch_s = job
                    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                u = units[uid]
                t_u = time.time()
                nkey = 0
                try:
                    try:
                        dec = nvc.CreateDecoder(gpuid=dev, codec=nvc.cudaVideoCodec.H264, usedevicememory=True)
                        batch_buf = None
                        batch_meta: list[tuple[int, int]] = []
                        bh = bw = out_idx = 0
                        n_sel = len(pres_sel)
                        pres_list = pres_sel.tolist()

                        def flush_batch():
                            nonlocal batch_buf, batch_meta
                            if not batch_meta:
                                return
                            ev = torch.cuda.Event()
                            ev.record(stream)
                            nv_bq.put((batch_buf[:len(batch_meta)], bh, bw, batch_meta, ev, True))
                            batch_buf = None
                            batch_meta = []

                        def push_frame(f):
                            nonlocal batch_buf, bh, bw, out_idx, nkey
                            t = torch.from_dlpack(f)
                            r, w = t.shape
                            h = r * 2 // 3
                            if batch_buf is None or (h, w) != (bh, bw):
                                flush_batch()
                                bh, bw = h, w
                                batch_buf = torch.empty((B, r, w), dtype=torch.uint8, device="cuda")
                            batch_buf[len(batch_meta)].copy_(t)
                            pr = pres_list[out_idx] if out_idx < n_sel else pres_list[-1]
                            out_idx += 1
                            nkey += 1
                            batch_meta.append((u["fid"], pr))
                            if t_first[0] is None:
                                t_first[0] = time.time()
                            if len(batch_meta) >= B:
                                flush_batch()

                        base_ptr = int(raw_arr.ctypes.data)
                        ptr = base_ptr
                        p = nvc.PacketData()
                        with torch.cuda.stream(stream):
                            for pkt_len in lengths.tolist():
                                p.bsl = pkt_len
                                p.bsl_data = ptr
                                ptr += pkt_len
                                for f in dec.Decode(p):
                                    push_frame(f)
                            p.bsl = 0
                            p.bsl_data = 0
                            for f in dec.Decode(p):
                                push_frame(f)
                            flush_batch()
                        dec = None
                        with lock:
                            done[uid] = {"fid": u["fid"], "frames": n_frames if u["first"] else 0, "keyframes": nkey,
                                         "bytes": span_bytes, "fetch_s": fetch_s, "dec_s": round(time.time() - t_u, 2)}
                    except Exception:  # noqa: BLE001 - fallback to CPU libavcodec on unsupported H264 profile
                        if nkey > 0:
                            raise
                        _cpu_fallback_unit(uid, fetch_s, stream)
                except Exception as e:  # noqa: BLE001 - reported per unit
                    with lock:
                        errs.append({"src": u["path"], "unit": uid, "error": repr(e)[:300]})
                finally:
                    if slot is not None:
                        nv_freeq.put(slot)

        threading.Thread(target=nv_dispatch, daemon=True).start()
        nvdec_threads = [threading.Thread(target=nvdec_worker, daemon=True) for _ in range(nvdec)]
        nv_det_threads = [threading.Thread(target=nv_detect, daemon=True) for _ in range(gpu_threads)]

    # ---------------------------------------------------------------- CPU decoder processes (optional / hybrid)
    shm = None
    buf = None
    cpu_gts: list[threading.Thread] = []
    cpu_procs: list = []
    cpu_feeder: threading.Thread | None = None
    if decoders > 0:
        cfg = {"units": units_path, "claim": claim_path, "decoders": decoders, "slots": slots, "parts": parts,
               "fetchers": fetchers, "prefetch": prefetch, "skip_loop_filter": skip_loop_filter}
        nslot = decoders * slots
        ctx = mp.get_context("spawn")
        shm = shared_memory.SharedMemory(create=True, size=nslot * SLOT)
        buf = np.ndarray((nslot, SLOT), np.uint8, buffer=shm.buf)
        outq = ctx.Queue()
        relq = [ctx.Queue() for _ in range(decoders)]
        cpu_procs = [ctx.Process(target=_decoder, args=(w, cfg, relq[w], outq, shm.name), daemon=True) for w in range(decoders)]
        for p in cpu_procs:
            p.start()
        rc = torch.cuda.cudart().cudaHostRegister(buf.ctypes.data, buf.nbytes, 0)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {rc}")
        host = torch.from_numpy(buf)
        bq: queue.Queue = queue.Queue(maxsize=8)

        def release(items):
            rel: dict[int, list] = {}
            for it in items:
                rel.setdefault(it[3] // slots, []).append(it[3])
            for wk, sl in rel.items():
                relq[wk].put(sl)

        def detect_cpu():
            torch.cuda.set_device(dev)
            dctx, stream = det.context(), torch.cuda.Stream()
            while (job := bq.get()) is not None:
                (r, w), items = job
                n, h, size = len(items), r * 2 // 3, r * w
                with torch.cuda.stream(stream):
                    idx_t = torch.tensor([it[3] for it in items], dtype=torch.long)
                    x = host.index_select(0, idx_t)[:, :size].to("cuda", non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record(stream)
                    q = (h // 2) * (w // 2)
                    Y = x[:, :h * w].view(n, h, w)
                    U = x[:, h * w:h * w + q].view(n, h // 2, w // 2)
                    V = x[:, h * w + q:h * w + 2 * q].view(n, h // 2, w // 2)
                    nv = torch.cat([Y, torch.stack([U, V], -1).view(n, h // 2, w)], 1)
                    copied.synchronize()
                    release(items)
                run_nv12_batch(nv, h, w, [(it[0], it[1]) for it in items], dctx, stream, is_nvdec=False)
            vq.put(None)

        def feed_cpu():
            pend: dict = {}
            born: dict = {}
            held = 0

            def dispatch(shape):
                nonlocal held
                items = pend.pop(shape)
                born.pop(shape, None)
                held -= len(items)
                bq.put((shape, items))

            alive = decoders
            while alive:
                try:
                    m = outq.get(timeout=0.05)
                except queue.Empty:
                    m = False
                if m is None:
                    alive -= 1
                elif isinstance(m, tuple):
                    with lock:
                        if m[0] == "done":
                            done[m[1]] = m[2]
                        else:
                            errs.append({"src": units[m[1]]["path"], "unit": m[1], "error": m[2]})
                elif m:
                    t_first[0] = t_first[0] or time.time()
                    for it in m:
                        s = it[2]
                        pend.setdefault(s, []).append(it)
                        born.setdefault(s, time.time())
                        held += 1
                        if len(pend[s]) >= B:
                            dispatch(s)
                now = time.time()
                for s in [s for s, t in born.items() if now - t > 0.25]:
                    dispatch(s)
                while held > 0.5 * nslot:
                    dispatch(max(pend, key=lambda s: len(pend[s])))
            for s in list(pend):
                dispatch(s)
            for _ in cpu_gts:
                bq.put(None)

        cpu_gts = [threading.Thread(target=detect_cpu, daemon=True) for _ in range(gpu_threads)]
        cpu_feeder = threading.Thread(target=feed_cpu, daemon=True)

    # Start verifier and all worker threads
    n_ver_producers = len(nv_det_threads) + len(cpu_gts)
    vt = threading.Thread(target=verify, args=(n_ver_producers,), daemon=True)
    vt.start()
    for t in (*nvdec_threads, *nv_det_threads, *cpu_gts):
        t.start()
    if cpu_feeder is not None:
        cpu_feeder.start()

    for t in nvdec_threads:
        t.join()
    if nvdec > 0:
        for _ in nv_det_threads:
            nv_bq.put(None)
    if cpu_feeder is not None:
        cpu_feeder.join()
    for t in (*nv_det_threads, *cpu_gts, vt):
        t.join()
    for p in (*nv_procs, *cpu_procs):
        p.join(timeout=5)
    torch.cuda.synchronize()
    dt = time.time() - t0

    per: dict[int, dict] = {}
    for d in done.values():
        q = per.setdefault(d["fid"], {"frames": 0, "keyframes": 0, "units": 0})
        q["frames"] += d["frames"]
        q["keyframes"] += d["keyframes"]
        q["units"] += 1
    with open(report + ".tmp", "w") as fh:
        for fid, q in per.items():
            fh.write(json.dumps({"src": files[fid], **q, "dets": sorted(dets.get(fid, []))}) + "\n")
        for e in errs:
            fh.write(json.dumps(e) + "\n")
    os.replace(report + ".tmp", report)
    frames = sum(d["frames"] for d in done.values())
    out = {"gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "units": len(done), "errors": len(errs), "seconds": round(dt, 1),
           "startup_s": round((t_first[0] or t0) - t0, 1), "frames": frames, "video_frames_per_s": round(frames / dt),
           "fetched_GB": round(sum(d["bytes"] for d in done.values()) / 1e9, 2),
           **{k: round(v, 1) if isinstance(v, float) else v for k, v in st.items()},
           "keyframes_per_s": round(st["keyframes"] / dt),
           "faces": sum(d[-1] >= ver_thr for v in dets.values() for d in v)}
    if log:
        log(out)
    if nv_shm is not None:
        nv_shm.close()
        nv_shm.unlink()
    if shm is not None:
        torch.cuda.cudart().cudaHostUnregister(buf.ctypes.data)
        shm.close()
        shm.unlink()
    return out


def merge_reports(parts: list[str], out: str, ver_thr: float = 0.8) -> dict:
    """Join per-GPU reports (a file's units may be split across GPUs) into one line per file."""
    files: dict[str, dict] = {}
    errors = []
    for p in parts:
        for line in open(p):
            r = json.loads(line)
            if "error" in r:
                errors.append(r)
                continue
            f = files.setdefault(r["src"], {"src": r["src"], "frames": 0, "keyframes": 0, "units": 0, "dets": []})
            for k in ("frames", "keyframes", "units"):
                f[k] += r[k]
            f["dets"] += r["dets"]
    with open(out + ".tmp", "w") as fh:
        for f in files.values():
            f["dets"].sort()
            f["faces"] = sum(d[-1] >= ver_thr for d in f["dets"])
            fh.write(json.dumps(f) + "\n")
        for e in errors:
            fh.write(json.dumps(e) + "\n")
    os.replace(out + ".tmp", out)
    return {"files": len(files), "frames": sum(f["frames"] for f in files.values()),
            "keyframes": sum(f["keyframes"] for f in files.values()), "errors": len(errors),
            "faces": sum(f["faces"] for f in files.values()), "files_with_faces": sum(f["faces"] > 0 for f in files.values())}
