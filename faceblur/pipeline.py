"""GPU pipeline: NVDEC decode -> YOLOv11n-face (TensorRT) -> SigLIP2 verifier -> in-place blur -> minimal re-encode.

One process drives one GPU. Inside it, `workers` threads each own an NVDEC decoder, a CUDA stream and a TensorRT
execution context and process whole files; the H100 has 7 NVDEC engines, so 7 workers saturate decode. Only the
frames of GOPs that contain a blurred face leave the GPU; they go to a pool of CPU processes that splice the
re-encoded GOPs into the original bitstream (see splice.py).
"""
from __future__ import annotations

import bisect
import json
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
import torch

from . import gpu, splice
from .models import Detector, Verifier


@dataclass
class Recipe:
    """Default: high precision. Missing a face is acceptable; blurring things that are not faces is not."""
    det_thr: float = 0.4       # YOLOv11n-face score
    ver_thr: float = 0.8       # SigLIP2-so400m probe P(face)
    grow: float = 0.25         # box growth on each side before blurring
    max_reencode_frac: float = 0.5  # above this share of re-encoded frames a full re-encode is used instead


def gop_starts(path: str) -> tuple[list[int], int]:
    """Presentation indices of keyframes and the frame count, from a packet scan (no decoding)."""
    with av.open(path) as c:
        vs = c.streams.video[0]
        pk = [(p.pts, p.is_keyframe) for p in c.demux(vs) if p.size and p.pts is not None]
    pk.sort()
    return [i for i, (_, k) in enumerate(pk) if k], len(pk)


class Worker(threading.Thread):
    def __init__(self, wid, jobs, results, det: Detector, ver: Verifier, pool, recipe: Recipe, batch: int, inflight):
        super().__init__(daemon=True)
        self.wid, self.jobs, self.results, self.pool, self.r, self.batch = wid, jobs, results, pool, recipe, batch
        self.det, self.ver, self.inflight = det.context(), ver, inflight
        self.stream = torch.cuda.Stream()

    def run(self):
        torch.cuda.set_device(0)
        while True:
            job = self.jobs.get()
            if job is None:
                return
            src, dst = job
            try:
                self.process(src, dst)
            except Exception:  # noqa: BLE001
                self.results.put({"src": src, "dst": dst, "error": traceback.format_exc()[-2000:]})

    def process(self, src: str, dst: str):
        import PyNvVideoCodec as nvc
        t0 = time.time()
        keys, n_pk = gop_starts(src)
        key_set = set(keys)
        dec = nvc.SimpleDecoder(src, gpu_id=0, use_device_memory=True)
        md = dec.get_stream_metadata()
        h, w = int(md.height), int(md.width)
        boxes, n_raw, idx = [], 0, 0
        segments: list[list] = []        # [start, [host yuv420p chunks]]
        carry, carry0 = None, 0
        splice_ok, seg_frames = True, 0
        with torch.cuda.stream(self.stream):
            while True:
                fr = dec.get_batch_frames(self.batch)
                eof = not fr
                if not eof:
                    x = torch.stack([torch.from_dlpack(f) for f in fr])  # own copy of the NV12 surfaces
                    n = len(x)
                    y = self.det(gpu.detector_input(x, h), self.stream)
                    fb, bx, sc = gpu.yolo_postproc(y, self.r.det_thr, h, w)
                    n_raw += len(fb)
                    if len(fb):
                        vs = self.ver(gpu.verifier_crops(x, h, fb, bx, self.ver.size))
                        keep = vs >= self.r.ver_thr
                        if keep.any():
                            gpu.blur_nv12_(x, h, fb[keep], bx[keep], self.r.grow)
                            for f_, b_, s_, v_ in zip(fb[keep].tolist(), bx[keep].tolist(), sc[keep].tolist(), vs[keep].tolist()):
                                boxes.append([idx + f_, *[round(c, 1) for c in b_], round(s_, 3), round(v_, 3)])
                    buf = x if carry is None else torch.cat([carry, x])
                    buf0 = carry0 if carry is not None else idx
                    idx += n
                else:
                    if carry is None:
                        break
                    buf, buf0 = carry, carry0
                end = buf0 + len(buf)
                # frames [buf0, cut) belong to complete GOPs; the rest waits for the next batch
                if eof:
                    cut = end
                else:
                    j = bisect.bisect_right(keys, end) - 1
                    cut = keys[j] if j >= 0 and keys[j] > buf0 else buf0
                if splice_ok and cut > buf0 and boxes:
                    flagged = {b[0] for b in boxes if buf0 <= b[0] < cut}
                    if flagged:
                        gs = keys[bisect.bisect_left(keys, buf0):bisect.bisect_left(keys, cut)] or [buf0]
                        bounds = gs + [cut]
                        for s, e in zip(bounds[:-1], bounds[1:]):
                            if any(s <= f < e for f in flagged):
                                host = gpu.nv12_to_yuv420p(buf[s - buf0:e - buf0], h).cpu().numpy()
                                if segments and segments[-1][0] + sum(len(c) for c in segments[-1][1]) == s:
                                    segments[-1][1].append(host)
                                else:
                                    segments.append([s, [host]])
                                seg_frames += e - s
                        if seg_frames > self.r.max_reencode_frac * n_pk:
                            splice_ok, segments = False, []
                carry, carry0 = (buf[cut - buf0:], cut) if cut < end else (None, 0)
                if eof:
                    break
        dec = None
        t_gpu = time.time() - t0
        if idx != n_pk or (keys and keys[0] != 0):
            splice_ok = False  # decoder and demuxer disagree, or no leading keyframe: do not splice
        info = {"src": src, "dst": dst, "frames": idx, "height": h, "width": w, "raw_detections": n_raw,
                "faces": len(boxes), "frames_blurred": len({b[0] for b in boxes}), "gpu_s": round(t_gpu, 2), "boxes": boxes}
        self.inflight.acquire()  # backpressure: bounded number of files waiting for the CPU pool
        if not boxes:
            fut = self.pool.submit(splice.copy, src, dst)
        elif splice_ok and all(s in key_set for s, _ in segments):
            segs = [(s, np.concatenate(c)) for s, c in segments]
            fut = self.pool.submit(_splice_or_reencode, src, dst, segs, _by_frame(boxes))
        else:
            fut = self.pool.submit(splice.reencode, src, dst, _by_frame(boxes))
        fut.add_done_callback(lambda f, info=info: (self.inflight.release(), self.results.put(_merge(info, f))))


def _by_frame(boxes):
    out: dict[int, list] = {}
    for b in boxes:
        out.setdefault(b[0], []).append(b[1:5])
    return out


def _splice_or_reencode(src, dst, segs, by_frame):
    try:
        return splice.splice(src, dst, segs)
    except splice.SpliceError as e:
        r = splice.reencode(src, dst, by_frame)
        r["splice_error"] = str(e)
        return r


def _merge(info, fut):
    try:
        return {**info, **fut.result()}
    except Exception:  # noqa: BLE001
        return {**info, "error": traceback.format_exc()[-2000:]}


def run(pairs: list[tuple[str, str]], recipe: Recipe | None = None, workers: int = 7, encoders: int | None = None,
        batch: int = 64, report: str | None = None, log_every: float = 30.0) -> dict:
    """Blur faces in every (src, dst) mp4 pair. Returns totals; one JSON line per file goes to `report`."""
    recipe = recipe or Recipe()
    torch.backends.cuda.matmul.allow_tf32 = True
    det, ver = Detector(), Verifier()
    encoders = encoders or max(2, min(16, (len(os.sched_getaffinity(0)) - workers - 1)))
    import multiprocessing as mp
    pool = ProcessPoolExecutor(encoders, mp_context=mp.get_context("spawn"))
    jobs, results = queue.Queue(), queue.Queue()
    for s, d in pairs:
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        jobs.put((s, d))
    for _ in range(workers):
        jobs.put(None)
    inflight = threading.Semaphore(encoders * 4)
    ws = [Worker(i, jobs, results, det, ver, pool, recipe, batch, inflight) for i in range(workers)]
    t0 = time.time()
    [w_.start() for w_ in ws]
    tot = {"files": 0, "frames": 0, "faces": 0, "frames_blurred": 0, "frames_reencoded": 0, "errors": 0,
           "modes": {"copy": 0, "splice": 0, "reencode": 0}}
    rep = open(report, "a") if report else None
    last = t0
    while tot["files"] + tot["errors"] < len(pairs):
        r = results.get()
        if "error" in r:
            tot["errors"] += 1
        else:
            tot["files"] += 1
            for k in ("frames", "faces", "frames_blurred"):
                tot[k] += r[k]
            tot["frames_reencoded"] += r.get("frames_reencoded", 0)
            tot["modes"][r["mode"]] += 1
        if rep:
            rep.write(json.dumps(r) + "\n")
            rep.flush()
        if time.time() - last > log_every:
            last = time.time()
            print(json.dumps({"progress": f'{tot["files"] + tot["errors"]}/{len(pairs)}',
                              "fps": round(tot["frames"] / (last - t0))}), flush=True)
    pool.shutdown()
    dt = time.time() - t0
    tot.update({"seconds": round(dt, 1), "fps": round(tot["frames"] / dt), "recipe": asdict(recipe), "workers": workers,
                "encoders": encoders, "gpu": torch.cuda.get_device_name()})
    if rep:
        rep.close()
    return tot
