"""faceblur command line.

  faceblur run SRC DST            blur every mp4 under SRC into the same tree under DST (resumable)
  faceblur bench SRC [-n FILES]   same pipeline into a temp dir, prints throughput, deletes the output
  faceblur check SRC DST          verify outputs: frame counts, untouched frames bit-exact, blurred frames changed
  faceblur prepare                fetch and convert the models into $FACEBLUR_MODELS
  faceblur units LIST OUT         index pass for `scan`: split files (local or hf://datasets/...) into keyframe units
  faceblur scan UNITS --report R  keyframe-only face scan with CPU decoders, one process per GPU (finds what to blur)
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


def cmd_units(a):
    from .units import build_units, write_units
    paths = [x.strip() for x in open(a.list) if x.strip()]
    units, stats = build_units(paths, chunk=a.chunk, threads=a.threads)
    write_units(units, a.out)
    print(json.dumps(stats), flush=True)
    return 1 if stats["index_errors"] else 0


def _scan_opts(a) -> list[str]:
    return ["--decoders", str(a.decoders), "--gpu-threads", str(a.gpu_threads), "--parts", str(a.parts),
            "--fetchers", str(a.fetchers), "--prefetch", str(a.prefetch), "--det-thr", str(a.det_thr), "--ver-thr", str(a.ver_thr)]


def cmd_scan(a):
    claim = a.claim
    own_claim = claim is None
    if own_claim:
        d = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        claim = os.path.join(d, f"faceblur-claim-{os.getpid()}")
        with open(claim, "w") as f:
            f.write("0")
    try:
        if a.gpus:  # one process per GPU, all claiming units from the same counter
            gpus = a.gpus.split(",")
            parts = [f"{a.report}.gpu{g}" for g in gpus]
            procs = [subprocess.Popen([sys.executable, "-m", "faceblur", "scan", a.units, "--report", r, "--claim", claim, *_scan_opts(a)],
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": g}) for g, r in zip(gpus, parts)]
            rc = max(p.wait() for p in procs)
            from .scan import merge_reports
            tot = merge_reports([r for r in parts if os.path.exists(r)], a.report, a.ver_thr)
            print(json.dumps({"merged": a.report, **tot}), flush=True)
            return rc or (1 if tot["errors"] else 0)
        from .scan import scan
        out = scan(a.units, a.report, claim, decoders=a.decoders, gpu_threads=a.gpu_threads, parts=a.parts, fetchers=a.fetchers,
                   prefetch=a.prefetch, det_thr=a.det_thr, ver_thr=a.ver_thr, log=lambda d: print(json.dumps(d), flush=True))
        return 1 if out["errors"] else 0
    finally:
        if own_claim:
            os.remove(claim)


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
    p = sub.add_parser("units", help="split files into keyframe-range work units for `scan`")
    p.add_argument("list", help="text file: one local path or hf://datasets/<org>/<repo>/<file> per line")
    p.add_argument("out", help="units JSONL")
    p.add_argument("--chunk", type=int, default=2000, help="keyframes per unit")
    p.add_argument("--threads", type=int, default=64, help="parallel index reads")
    p = sub.add_parser("scan", help="keyframe-only face scan (CPU decode, GPU detect + verify)")
    p.add_argument("units", help="JSONL from `faceblur units`")
    p.add_argument("--report", required=True, help="JSONL, one line per file with every detection")
    p.add_argument("--gpus", help="comma-separated GPU ids: one process per GPU sharing one work queue")
    p.add_argument("--decoders", type=int, default=10, help="CPU decoder processes per GPU (about CPUs / GPUs)")
    p.add_argument("--gpu-threads", type=int, default=2, help="detector threads per GPU")
    p.add_argument("--parts", type=int, default=4, help="parallel HTTP range requests per Hub read")
    p.add_argument("--fetchers", type=int, default=2, help="prefetch threads per decoder")
    p.add_argument("--prefetch", type=int, default=2, help="units read ahead per decoder")
    p.add_argument("--det-thr", type=float, default=0.4)
    p.add_argument("--ver-thr", type=float, default=0.8, help="only used for the face counts; all scores are reported")
    p.add_argument("--claim", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cmd == "units":
        return cmd_units(a)
    if a.cmd == "scan":
        return cmd_scan(a)
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
