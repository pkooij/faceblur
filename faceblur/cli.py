"""faceblur command line.

  faceblur run SRC DST            blur every mp4 under SRC into the same tree under DST (resumable)
  faceblur bench SRC [-n FILES]   same pipeline into a temp dir, prints throughput, deletes the output
  faceblur check SRC DST          verify outputs: frame counts, untouched frames bit-exact, blurred frames changed
  faceblur prepare                fetch and convert the models into $FACEBLUR_MODELS
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _pairs(src: Path, dst: Path, pattern: str, files: str | None, shard: str | None, overwrite: bool):
    rels = [Path(x.strip()) for x in open(files)] if files else sorted(p.relative_to(src) for p in src.glob(pattern))
    if shard:
        i, n = map(int, shard.split("/"))
        rels = rels[i::n]
    out = [(str(src / r), str(dst / r)) for r in rels]
    return out if overwrite else [(s, d) for s, d in out if not os.path.exists(d)]


def _recipe(a):
    from .pipeline import Recipe
    return Recipe(det_thr=a.det_thr, ver_thr=a.ver_thr)


def _multi_gpu(a) -> int:
    """Re-launch this command once per GPU with CUDA_VISIBLE_DEVICES and a shard of the file list."""
    gpus = a.gpus.split(",")
    procs = []
    for k, g in enumerate(gpus):
        argv = [x for x in sys.argv[1:]]
        i = argv.index("--gpus")
        del argv[i:i + 2]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g}
        rep = ["--report", f"{a.report}.gpu{g}"] if a.report else []
        procs.append(subprocess.Popen([sys.executable, "-m", "faceblur", *argv, "--shard", f"{k}/{len(gpus)}", *rep], env=env))
    return max(p.wait() for p in procs)


def cmd_run(a, bench=False):
    if a.gpus:
        return _multi_gpu(a)
    from .pipeline import run
    src = Path(a.src).resolve()
    dst = Path(tempfile.mkdtemp(prefix="faceblur-bench-")) if bench else Path(a.dst).resolve()
    pairs = _pairs(src, dst, a.glob, a.files, a.shard, a.overwrite or bench)
    if bench:
        pairs = pairs[: a.n]
    print(json.dumps({"files": len(pairs), "src": str(src), "dst": str(dst)}), flush=True)
    try:
        tot = run(pairs, _recipe(a), workers=a.workers, encoders=a.encoders, batch=a.batch, report=a.report)
    finally:
        if bench:
            shutil.rmtree(dst, ignore_errors=True)
    print(json.dumps(tot, indent=1), flush=True)
    return 1 if tot["errors"] else 0


def cmd_check(a):
    from .verify import check_tree
    res = check_tree(Path(a.src), Path(a.dst), a.report, limit=a.n)
    print(json.dumps(res, indent=1))
    return 0 if res["failed"] == 0 else 1


def cmd_prepare(a):
    from .prepare import prepare
    prepare(Path(a.models) if a.models else None)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="faceblur", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bench"):
        p = sub.add_parser(name)
        p.add_argument("src")
        if name == "run":
            p.add_argument("dst")
        else:
            p.add_argument("-n", type=int, default=32, help="number of files")
        p.add_argument("--glob", default="**/*.mp4")
        p.add_argument("--files", help="text file with paths relative to SRC (one per line)")
        p.add_argument("--det-thr", type=float, default=0.4)
        p.add_argument("--ver-thr", type=float, default=0.8)
        p.add_argument("--workers", type=int, default=7, help="decode/GPU threads per GPU (H100: 7 NVDEC engines)")
        p.add_argument("--encoders", type=int, default=None, help="CPU processes for copy/splice/re-encode")
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--report", help="JSONL file, one line per video (boxes, scores, mode)")
        p.add_argument("--gpus", help="comma-separated GPU ids: one process per GPU, files sharded")
        p.add_argument("--shard", help=argparse.SUPPRESS)
        p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--report", required=True, help="the JSONL written by `faceblur run`")
    p.add_argument("-n", type=int, default=None, help="check at most n files")
    p = sub.add_parser("prepare")
    p.add_argument("--models", help="target dir (default $FACEBLUR_MODELS)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "bench":
        a.dst = None
        return cmd_run(a, bench=True)
    if a.cmd == "check":
        return cmd_check(a)
    return cmd_prepare(a)


if __name__ == "__main__":
    sys.exit(main())
