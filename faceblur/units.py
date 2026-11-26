"""Index pass for `faceblur scan`: read only each file's MP4 sample table and split the file into keyframe-range work
units, largest first, so long files are decoded by many CPU workers at once and every GPU finishes at about the same
time.

Paths are local files or Hub URIs `hf://datasets/<org>/<repo>/<path in repo>`, which are read with HTTP range requests
(no download).
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._vendor.lerobot_streaming.mp4 import fetch_mp4_index

_FETCHERS: dict = {}
_LOCK = threading.Lock()


def split_hub(path: str) -> tuple[str, str]:
    """hf://datasets/org/repo/rel/path.mp4 -> ("org/repo", "rel/path.mp4")."""
    parts = path[len("hf://datasets/"):].split("/", 2)
    if len(parts) < 3:
        raise ValueError(f"not a Hub dataset file URI: {path}")
    return f"{parts[0]}/{parts[1]}", parts[2]


def opener(path: str, parts: int = 4):
    """(read_range(path, offset, n) -> bytes, file size) for a local path or an hf://datasets/... URI."""
    if path.startswith("hf://"):
        from ._vendor.lerobot_streaming.range_fetch import NativeHTTPRangeFetcher
        repo, rel = split_hub(path)
        with _LOCK:
            fe = _FETCHERS.get((repo, parts))
            if fe is None:
                fe = _FETCHERS[(repo, parts)] = NativeHTTPRangeFetcher(
                    f"hf://datasets/{repo}", max_connections=max(8, 4 * parts), subrange_parts=parts,
                    subrange_min_bytes=4 << 20, timeout=30.0)
        return (lambda p, o, n: fe.read_range(rel, o, n)), fe.info_size(rel)

    def rr(p, o, n):
        with open(p, "rb") as fh:
            fh.seek(o)
            return fh.read(n)
    return rr, os.path.getsize(path)


def index(path: str, parts: int = 4):
    rr, size = opener(path, parts)
    return rr, fetch_mp4_index(path, rr, file_size=size)


def sync_samples(idx) -> np.ndarray:
    """Decode-order indices of the keyframes (every sample if the track has no sync-sample table)."""
    s = np.asarray(idx.sync_samples, np.int64)
    return s if len(s) else np.arange(len(idx.sample_sizes))


def build_units(paths: list[str], chunk: int = 2000, threads: int = 64) -> tuple[list[dict], dict]:
    """Split every file into units of at most `chunk` keyframes. A file whose index cannot be read becomes one
    whole-file unit (hi = -1) so the scan reports its error instead of silently skipping it."""
    t0 = time.time()

    def info(fp):
        fid, path = fp
        try:
            return fid, path, len(sync_samples(index(path, parts=1)[1])), None
        except Exception as e:  # noqa: BLE001 - reported, not fatal
            return fid, path, -1, repr(e)[:200]

    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(info, enumerate(paths)))
    units, errors = [], []
    for fid, path, nk, err in res:
        if nk < 0:
            errors.append({"src": path, "error": err})
            units.append({"fid": fid, "path": path, "lo": 0, "hi": -1, "first": True})
            continue
        for lo in range(0, max(nk, 1), chunk):
            units.append({"fid": fid, "path": path, "lo": lo, "hi": min(nk, lo + chunk), "first": lo == 0})
    units.sort(key=lambda u: -(u["hi"] - u["lo"]) if u["hi"] >= 0 else -10**9)
    nks = sorted(r[2] for r in res if r[2] >= 0) or [0]
    stats = {"files": len(paths), "units": len(units), "index_errors": len(errors), "index_s": round(time.time() - t0, 1),
             "keyframes": int(sum(nks)), "keyframes_max": nks[-1], "keyframes_median": nks[len(nks) // 2]}
    return units, {**stats, "errors": errors[:20]}


def write_units(units: list[dict], out: str) -> None:
    with open(out, "w") as fh:
        for u in units:
            fh.write(json.dumps(u) + "\n")
