"""Output checks: same frame count as the source, frames outside re-encoded GOPs decode bit-exact, every frame with
a reported face differs from the source inside the face box. Uses CPU decoding (PyAV), independent of the GPU path."""
from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np


def _frames(path: str):
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            yield f.to_ndarray(format="yuv420p")


def check_file(rec: dict) -> dict:
    src, dst, h = rec["src"], rec["dst"], rec["height"]
    boxes: dict[int, list] = {}
    for b in rec["boxes"]:
        boxes.setdefault(b[0], []).append(b[1:5])
    n = same = changed_outside = 0
    blurred_ok = blurred_total = 0
    for k, (a, b) in enumerate(zip(_frames(src), _frames(dst))):
        n += 1
        if k in boxes:
            blurred_total += 1
            diff = [np.abs(a[int(y1):int(y2) + 1, int(x1):int(x2) + 1].astype(int) - b[int(y1):int(y2) + 1, int(x1):int(x2) + 1]).mean()
                    for x1, y1, x2, y2 in boxes[k]]
            blurred_ok += int(min(diff) > 2.0)
        elif np.array_equal(a, b):
            same += 1
        else:
            changed_outside += 1
    n_dst = sum(1 for _ in _frames(dst)) if n else 0
    out = {"src": src, "mode": rec["mode"], "frames": n, "frames_dst": n_dst, "identical": same,
           "reencoded_unblurred": changed_outside, "blurred_frames_changed": f"{blurred_ok}/{blurred_total}"}
    expect_reenc = rec.get("frames_reencoded", 0) - blurred_total
    out["ok"] = (n == n_dst == rec["frames"] and blurred_ok == blurred_total
                 and (rec["mode"] != "copy" or same == n) and changed_outside <= max(expect_reenc, 0))
    return out


def check_tree(src: Path, dst: Path, report: str, limit: int | None = None) -> dict:
    recs = [json.loads(x) for x in open(report) if '"error"' not in x]
    recs = sorted(recs, key=lambda r: -r.get("faces", 0))[:limit] if limit else recs
    res = [check_file(r) for r in recs]
    bad = [r for r in res if not r["ok"]]
    return {"checked": len(res), "failed": len(bad), "failures": bad[:20],
            "modes": {m: sum(r["mode"] == m for r in res) for m in ("copy", "splice", "reencode")}}
