"""Targeted Pass 2: from a `faceblur scan` report, download/open ONLY files with verified faces, decode ONLY the GOPs
within ±pad_gops of each keyframe hit (~1% of frames), detect + verify + blur every frame in those GOPs on GPU, and
splice only the blurred GOPs back into the original MP4 bitstream.

Every file without verified faces is untouched (no download, no write, no re-upload). Every frame outside a blurred GOP
in a modified file stays bit-exact.
"""
from __future__ import annotations

import bisect
import json
import os
import queue
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import av
import numpy as np

from . import splice
from .pipeline import Recipe, _by_frame, _splice_or_reencode
from .units import opener, split_hub


def gop_windows(gop_starts: list[int], n_frames: int, hit_frames: list[int], pad_gops: int = 1) -> list[tuple[int, int]]:
    """Return sorted, merged [start_frame, end_frame) presentation-frame intervals covering every GOP within ±pad_gops
    of any frame in `hit_frames`."""
    if not gop_starts or not hit_frames:
        return []
    n_gops = len(gop_starts)
    marked: set[int] = set()
    for f in hit_frames:
        g = max(0, bisect.bisect_right(gop_starts, f) - 1)
        for j in range(max(0, g - pad_gops), min(n_gops, g + pad_gops + 1)):
            marked.add(j)
    runs: list[tuple[int, int]] = []
    for g in sorted(marked):
        a = gop_starts[g]
        b = gop_starts[g + 1] if g + 1 < n_gops else n_frames
        if runs and runs[-1][1] == a:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return runs


def decode_gop_run(pkts: list, order: list[int], vs, a: int, b: int) -> tuple[bool, list[tuple[int, np.ndarray]]]:
    """Decode only the closed-GOP presentation interval [a, b) from already-demuxed `pkts`.
    Returns (contiguous_ok, [(pres_frame, yuv420p array), ...])."""
    dec = sorted(order[a:b])
    if dec != list(range(dec[0], dec[0] + len(dec))):
        return False, []
    dec_to_pres = {order[p]: p for p in range(a, b)}
    cc = av.CodecContext.create(vs.codec_context.name, "r")
    cc.extradata = bytes(vs.codec_context.extradata or b"")
    cc.thread_count = 1
    out: list[tuple[int, np.ndarray]] = []
    for d in dec:
        p = pkts[d]
        pk = av.Packet(bytes(p))
        pk.pts = dec_to_pres[d]
        for fr in cc.decode(pk):
            out.append((int(fr.pts), fr.to_ndarray(format="yuv420p")))
    for fr in cc.decode(None):
        out.append((int(fr.pts), fr.to_ndarray(format="yuv420p")))
    out.sort(key=lambda x: x[0])
    return len(out) == (b - a), out


def dst_path(src: str, dst_root: Path, src_root: Path | None = None) -> Path:
    """Map a local path or hf://datasets/<org>/<repo>/<rel> URI to an output path under `dst_root`."""
    if src.startswith("hf://datasets/"):
        repo, rel = split_hub(src)
        return dst_root / repo / rel
    p = Path(src)
    if src_root is not None:
        try:
            return dst_root / p.resolve().relative_to(src_root.resolve())
        except ValueError:
            pass
    return dst_root / p.name


def fetch_to_local(src: str, cache_dir: Path | None = None, parts: int = 4) -> tuple[str, bool]:
    """Ensure `src` is a local file. Returns (local_path, is_temp)."""
    if not src.startswith("hf://"):
        return src, False
    rr, size = opener(src, parts=parts)
    if cache_dir is not None:
        repo, rel = split_hub(src)
        local = cache_dir / repo / rel
        if local.exists() and local.stat().st_size == size:
            return str(local), False
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(local) + ".tmp"
        data = rr(src, 0, size)
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, local)
        return str(local), False
    fd, tmp = tempfile.mkstemp(prefix="faceblur-src-", suffix=".mp4")
    try:
        data = rr(src, 0, size)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return tmp, True
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _load_hits(scan_report: str, ver_thr: float, shard: str | None) -> tuple[list[dict], int]:
    """Read scan JSONL and return (hit_records, total_files_in_report)."""
    hits, total = [], 0
    for line in open(scan_report):
        r = json.loads(line)
        if "error" in r:
            continue
        total += 1
        hf = sorted({int(d[0]) for d in r.get("dets", []) if d[-1] >= ver_thr})
        if hf:
            hits.append({"src": r["src"], "frames": r.get("frames", 0), "hit_frames": hf})
    hits.sort(key=lambda x: x["src"])
    if shard:
        i, n = map(int, shard.split("/"))
        hits = hits[i::n]
    return hits, total


def apply_scan(scan_report: str, dst_dir: str, recipe: Recipe | None = None, pad_gops: int = 1,
               src_root: str | None = None, cache_dir: str | None = None, workers: int = 4,
               encoders: int | None = None, parts: int = 4, batch: int = 128, shard: str | None = None,
               overwrite: bool = False, report: str | None = None) -> dict:
    """Apply targeted ±pad_gops blurring to every file with verified faces in `scan_report`."""
    import multiprocessing as mp

    import torch

    from . import gpu
    from .models import Detector, Verifier

    recipe = recipe or Recipe()
    dst_root = Path(dst_dir).resolve()
    sroot = Path(src_root).resolve() if src_root else None
    cdir = Path(cache_dir).resolve() if cache_dir else None

    hits, total_scanned = _load_hits(scan_report, recipe.ver_thr, shard)
    todo = []
    for h in hits:
        d = dst_path(h["src"], dst_root, sroot)
        if overwrite or not d.exists():
            todo.append({**h, "dst": str(d)})

    t0 = time.time()
    if not todo:
        return {"scanned_files": total_scanned, "hit_files": len(hits), "files": 0, "skipped_existing": len(hits),
                "frames_total": 0, "frames_decoded": 0, "frames_blurred": 0, "frames_reencoded": 0,
                "faces": 0, "errors": 0, "seconds": 0.0, "modes": {"copy": 0, "splice": 0, "reencode": 0, "noop": 0}}

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.cuda.current_device()
    det, ver = Detector(), Verifier()
    encoders = encoders or max(2, min(16, (len(os.sched_getaffinity(0)) - workers - 1)))
    pool = ProcessPoolExecutor(encoders, mp_context=mp.get_context("spawn"))

    # Prefetch Hub downloads concurrently so GPU workers don't wait on network
    fetch_q: queue.Queue = queue.Queue(maxsize=max(workers * 2, 8))
    results: queue.Queue = queue.Queue()
    inflight = threading.Semaphore(encoders * 4)

    def _prefetch_one(item):
        t_dl = time.time()
        try:
            local_src, is_tmp = fetch_to_local(item["src"], cdir, parts=parts)
            return item, local_src, is_tmp, round(time.time() - t_dl, 2), None
        except Exception:  # noqa: BLE001
            return item, None, False, round(time.time() - t_dl, 2), traceback.format_exc()[-1000:]

    def feeder():
        with ThreadPoolExecutor(max(workers * 2, 4)) as ex:
            for res in ex.map(_prefetch_one, todo):
                fetch_q.put(res)
        for _ in range(workers):
            fetch_q.put(None)

    def _finish(local_src: str, is_tmp: bool, info: dict, fut):
        try:
            res = {**info, **fut.result()}
        except Exception:  # noqa: BLE001
            res = {**info, "error": traceback.format_exc()[-2000:]}
        finally:
            if is_tmp:
                try:
                    os.remove(local_src)
                except OSError:
                    pass
            inflight.release()
        results.put(res)

    def worker():
        torch.cuda.set_device(dev)
        dctx, stream = det.context(), torch.cuda.Stream()
        while (job := fetch_q.get()) is not None:
            item, local_src, is_tmp, dl_s, err = job
            if err is not None:
                results.put({"src": item["src"], "dst": item["dst"], "download_s": dl_s, "error": err})
                continue
            try:
                t_g = time.time()
                with av.open(local_src) as inp:
                    vs = inp.streams.video[0]
                    w, h = vs.codec_context.width, vs.codec_context.height
                    pkts = [p for p in inp.demux(vs) if p.size and p.pts is not None]
                    order = sorted(range(len(pkts)), key=lambda i: pkts[i].pts)
                    gstarts = [p for p, d in enumerate(order) if pkts[d].is_keyframe]
                    n_pk = len(pkts)
                    wins = gop_windows(gstarts, n_pk, item["hit_frames"], pad_gops=pad_gops)
                    decoded_runs: list[tuple[int, int, list[tuple[int, np.ndarray]]]] = []
                    splice_ok = len(inp.streams) == 1 and bool(gstarts) and gstarts[0] == 0
                    for a, b in wins:
                        ok, frames = decode_gop_run(pkts, order, vs, a, b)
                        if not ok:
                            splice_ok = False
                            break
                        decoded_runs.append((a, b, frames))

                if not splice_ok:
                    # Fallback to full NVDEC/CPU reencode via pipeline on this single file
                    from .pipeline import run as full_run
                    Path(item["dst"]).parent.mkdir(parents=True, exist_ok=True)
                    tot_one = full_run([(local_src, item["dst"])], recipe, workers=1, encoders=1, batch=64)
                    if is_tmp:
                        os.remove(local_src)
                    results.put({"src": local_src, "orig_src": item["src"], "dst": item["dst"], "frames": n_pk,
                                 "height": h, "width": w, "faces": tot_one["faces"],
                                 "frames_blurred": tot_one["frames_blurred"],
                                 "frames_reencoded": tot_one["frames_reencoded"], "frames_decoded": n_pk,
                                 "download_s": dl_s, "gpu_s": round(time.time() - t_g, 2),
                                 "mode": "reencode", "boxes": []})
                    continue

                # Run GPU detect + verify + blur on EVERY frame in the candidate ±pad_gops runs
                all_boxes: list = []
                segments: list[tuple[int, np.ndarray]] = []
                n_dec = 0
                seg_frames = 0
                q = (h // 2) * (w // 2)
                for a, b, frames in decoded_runs:
                    n_dec += len(frames)
                    f_ids = [p for p, _ in frames]
                    arrs = [arr for _, arr in frames]
                    blurred_nv_chunks = []
                    run_flagged: set[int] = set()
                    for off in range(0, len(arrs), batch):
                        chunk = arrs[off:off + batch]
                        n = len(chunk)
                        with torch.cuda.stream(stream):
                            x = torch.from_numpy(np.stack([c.reshape(-1) for c in chunk])).to("cuda", non_blocking=True)
                            Y = x[:, :h * w].view(n, h, w)
                            U = x[:, h * w:h * w + q].view(n, h // 2, w // 2)
                            V = x[:, h * w + q:h * w + 2 * q].view(n, h // 2, w // 2)
                            nv = torch.cat([Y, torch.stack([U, V], -1).view(n, h // 2, w)], 1)
                            y = dctx(gpu.detector_input(nv, h), stream)
                            fb, bx, sc = gpu.yolo_postproc(y, recipe.det_thr, h, w)
                            if len(fb):
                                vs_scores = ver(gpu.verifier_crops(nv, h, fb, bx, ver.size))
                                keep = vs_scores >= recipe.ver_thr
                                if keep.any():
                                    gpu.blur_nv12_(nv, h, fb[keep], bx[keep], recipe.grow)
                                    for f_, b_, s_, v_ in zip(fb[keep].tolist(), bx[keep].tolist(),
                                                              sc[keep].tolist(), vs_scores[keep].tolist()):
                                        pf = f_ids[off + f_]
                                        run_flagged.add(pf)
                                        all_boxes.append([pf, *[round(c, 1) for c in b_], round(s_, 3), round(v_, 3)])
                            yuv = gpu.nv12_to_yuv420p(nv, h).cpu().numpy()
                        stream.synchronize()
                        blurred_nv_chunks.append(yuv)

                    if not run_flagged:
                        continue
                    full_yuv = np.concatenate(blurred_nv_chunks, axis=0)
                    # Trim to ONLY the GOPs inside [a, b) that actually got a blurred frame
                    gs = gstarts[bisect.bisect_left(gstarts, a):bisect.bisect_left(gstarts, b)] or [a]
                    bounds = gs + [b]
                    for s, e in zip(bounds[:-1], bounds[1:]):
                        if any(s <= pf < e for pf in run_flagged):
                            sub = full_yuv[s - a:e - a]
                            if segments and segments[-1][0] + len(segments[-1][1]) == s:
                                segments[-1] = (segments[-1][0], np.concatenate([segments[-1][1], sub], axis=0))
                            else:
                                segments.append((s, sub))
                            seg_frames += e - s

                gpu_s = round(time.time() - t_g, 2)
                info = {"src": local_src, "orig_src": item["src"], "dst": item["dst"], "frames": n_pk,
                        "height": h, "width": w, "faces": len(all_boxes),
                        "frames_blurred": len({b[0] for b in all_boxes}), "frames_decoded": n_dec,
                        "download_s": dl_s, "gpu_s": gpu_s, "boxes": all_boxes}
                if not all_boxes:
                    if is_tmp:
                        os.remove(local_src)
                    results.put({**info, "mode": "noop", "frames_reencoded": 0})
                    continue

                Path(item["dst"]).parent.mkdir(parents=True, exist_ok=True)
                inflight.acquire()
                if seg_frames <= recipe.max_reencode_frac * n_pk:
                    fut = pool.submit(_splice_or_reencode, local_src, item["dst"], segments, _by_frame(all_boxes))
                else:
                    fut = pool.submit(splice.reencode, local_src, item["dst"], _by_frame(all_boxes))
                fut.add_done_callback(lambda f, ls=local_src, it=is_tmp, inf=info: _finish(ls, it, inf, f))
            except Exception:  # noqa: BLE001
                if is_tmp:
                    try:
                        os.remove(local_src)
                    except OSError:
                        pass
                results.put({"src": item["src"], "dst": item["dst"], "download_s": dl_s,
                             "error": traceback.format_exc()[-2000:]})

    ft = threading.Thread(target=feeder, daemon=True)
    ws = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    ft.start()
    for w_ in ws:
        w_.start()

    tot = {"scanned_files": total_scanned, "hit_files": len(hits), "files": 0, "skipped_existing": len(hits) - len(todo),
           "frames_total": 0, "frames_decoded": 0, "frames_blurred": 0, "frames_reencoded": 0,
           "faces": 0, "errors": 0, "modes": {"copy": 0, "splice": 0, "reencode": 0, "noop": 0}}
    rep = open(report, "a") if report else None
    while tot["files"] + tot["errors"] < len(todo):
        r = results.get()
        if "error" in r:
            tot["errors"] += 1
        else:
            tot["files"] += 1
            tot["frames_total"] += r["frames"]
            tot["frames_decoded"] += r["frames_decoded"]
            tot["frames_blurred"] += r["frames_blurred"]
            tot["frames_reencoded"] += r.get("frames_reencoded", 0)
            tot["faces"] += r["faces"]
            tot["modes"][r["mode"]] = tot["modes"].get(r["mode"], 0) + 1
        if rep:
            rep.write(json.dumps(r) + "\n")
            rep.flush()
    ft.join()
    for w_ in ws:
        w_.join()
    pool.shutdown()
    if rep:
        rep.close()
    dt = time.time() - t0
    tot.update({"seconds": round(dt, 1),
                "decoded_pct": round(100 * tot["frames_decoded"] / max(tot["frames_total"], 1), 2),
                "reencoded_pct": round(100 * tot["frames_reencoded"] / max(tot["frames_total"], 1), 2),
                "effective_hit_fps": round(tot["frames_total"] / max(dt, 0.01)),
                "recipe": asdict(recipe), "pad_gops": pad_gops, "gpu": torch.cuda.get_device_name(dev)})
    return tot
