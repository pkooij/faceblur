"""Keyframe-only face scan: find which files (and which frames) need blurring, without decoding every frame.

The house profile has closed 8-frame GOPs, so looking at the IDR keyframes only samples about every 8th frame for an
eighth of the decode work. Decoding is done on the CPU, where a node has far more decode capacity than in its NVDECs:

  CPU decoder processes (about 10 per GPU)              one GPU process
  ─────────────────────────────────────────             ─────────────────────────────────────────────────────
  claim the next work unit (node-wide counter)          2 threads: H2D copy from page-locked shared memory,
  read its keyframes' byte span (local or Hub range) ─▶   YOLOv11n-face TRT on batches of 128 keyframes
  decode them with libavcodec, write yuv420p frames     1 thread: SigLIP2 verifier on batches of >= 512 crops
  into shared-memory slots

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

SLOT = 1024 * 1024 * 3 // 2  # bytes per frame slot: yuv420p up to 1024x1024
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
    sel = sync[u["lo"]:len(sync) if u["hi"] < 0 else u["hi"]]
    offs = np.asarray(idx.sample_offsets, np.int64)[sel]
    sizes = np.asarray(idx.sample_sizes, np.int64)[sel]
    lo, hi = int(offs.min()), int((offs + sizes).max())
    return idx, sel, offs - lo, sizes, rr(u["path"], lo, hi - lo)


def decode_unit(idx, sel, roffs, sizes, data) -> Iterator[tuple[int, np.ndarray]]:
    """Decode the selected keyframes on the CPU. Yields (presentation frame index, yuv420p array [1.5h, w])."""
    import av
    codec, extra = decoder_config(idx.stsd_body)
    cc = av.CodecContext.create(codec, "r")
    cc.extradata = extra
    cc.thread_count = 1
    n = len(idx.sample_sizes)
    pres = np.empty(n, np.int64)
    pres[np.argsort(np.asarray(idx.sample_pts), kind="stable")] = np.arange(n)
    for m in range(len(sel)):
        o = int(roffs[m])
        pk = av.Packet(data[o:o + int(sizes[m])])
        pk.pts = int(sel[m])  # carries the decode-order sample id through the decoder
        for fr in cc.decode(pk):
            yield int(pres[int(fr.pts)]), fr.to_ndarray(format="yuv420p")
    for fr in cc.decode(None):
        yield int(pres[int(fr.pts)]), fr.to_ndarray(format="yuv420p")


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


def _decoder(w: int, cfg: dict, relq, outq, shm_name: str) -> None:
    """Decoder process: prefetch threads claim and read units, the main thread decodes into this worker's slots."""
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
            for pres, arr in decode_unit(idx, sel, roffs, sizes, data):
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


def scan(units_path: str, report: str, claim_path: str, decoders: int = 10, slots: int = 64, gpu_threads: int = 2,
         parts: int = 4, fetchers: int = 2, prefetch: int = 2, det_thr: float = 0.4, ver_thr: float = 0.8,
         log: Callable[[dict], None] | None = None) -> dict:
    """Scan the units in `units_path` on the current CUDA device. Several processes (one per GPU) may share one
    `claim_path`; each writes its own `report`."""
    import multiprocessing as mp

    import torch

    from . import gpu
    from .models import Detector, Verifier

    cfg = {"units": units_path, "claim": claim_path, "decoders": decoders, "slots": slots, "parts": parts,
           "fetchers": fetchers, "prefetch": prefetch}
    nslot = decoders * slots
    units = [json.loads(line) for line in open(units_path)]
    files = {u["fid"]: u["path"] for u in units}
    ctx = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=nslot * SLOT)
    buf = np.ndarray((nslot, SLOT), np.uint8, buffer=shm.buf)
    outq = ctx.Queue()
    relq = [ctx.Queue() for _ in range(decoders)]
    t0 = time.time()
    procs = [ctx.Process(target=_decoder, args=(w, cfg, relq[w], outq, shm.name), daemon=True) for w in range(decoders)]
    for p in procs:
        p.start()

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.cuda.current_device()
    # page-lock the shared slots so host-to-device copies read them directly
    rc = torch.cuda.cudart().cudaHostRegister(buf.ctypes.data, buf.nbytes, 0)
    if int(rc) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {rc}")
    host = torch.from_numpy(buf)
    det, ver = Detector(), Verifier()
    B = 128
    bq: queue.Queue = queue.Queue(maxsize=8)
    vq: queue.Queue = queue.Queue()
    dets: dict[int, list] = {}
    done: dict[int, dict] = {}
    errs: list = []
    lock = threading.Lock()
    st = {"keyframes": 0, "crops": 0, "batches": 0, "det_busy_s": 0.0, "ver_busy_s": 0.0}
    t_first = [None]

    def release(items):
        rel: dict[int, list] = {}
        for it in items:
            rel.setdefault(it[3] // slots, []).append(it[3])
        for wk, sl in rel.items():
            relq[wk].put(sl)

    def detect():
        torch.cuda.set_device(dev)
        dctx, stream = det.context(), torch.cuda.Stream()
        while (job := bq.get()) is not None:
            (r, w), items = job
            n, h, size = len(items), r * 2 // 3, r * w
            td = time.time()
            with torch.cuda.stream(stream):
                x = torch.empty((n, size), dtype=torch.uint8, device="cuda")
                for i, it in enumerate(items):
                    x[i].copy_(host[it[3], :size], non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(stream)
                q = (h // 2) * (w // 2)
                Y = x[:, :h * w].view(n, h, w)
                U = x[:, h * w:h * w + q].view(n, h // 2, w // 2)
                V = x[:, h * w + q:h * w + 2 * q].view(n, h // 2, w // 2)
                nv = torch.cat([Y, torch.stack([U, V], -1).view(n, h // 2, w)], 1)
                y = dctx(gpu.detector_input(nv, h), stream)
                copied.synchronize()
                release(items)
                fb, bx, sc = gpu.yolo_postproc(y, det_thr, h, w)
                if len(fb):
                    crops = gpu.verifier_crops(nv, h, fb, bx, ver.size)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                    meta = [(items[f][0], items[f][1], [round(c, 1) for c in b], round(s, 3))
                            for f, b, s in zip(fb.tolist(), bx.tolist(), sc.tolist())]
                    vq.put((crops, meta, ev))
            with lock:
                st["det_busy_s"] += time.time() - td
                st["keyframes"] += n
                st["batches"] += 1
        vq.put(None)

    def verify():
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
                if ends == gpu_threads:
                    run()
                    return
                continue
            pend.append(m)
            npend += len(m[1])
            if npend >= 512:
                run()

    gts = [threading.Thread(target=detect, daemon=True) for _ in range(gpu_threads)]
    vt = threading.Thread(target=verify, daemon=True)
    for t in (*gts, vt):
        t.start()

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
        # flush partial batches that are old, or when the decoders risk running out of free slots
        for s in [s for s, t in born.items() if now - t > 0.25]:
            dispatch(s)
        while held > 0.5 * nslot:
            dispatch(max(pend, key=lambda s: len(pend[s])))
    for s in list(pend):
        dispatch(s)
    for _ in gts:
        bq.put(None)
    for t in (*gts, vt):
        t.join()
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
