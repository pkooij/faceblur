"""CPU-only tests of the keyframe scan's index pass and decoder: work units cover every keyframe exactly once and the
decoded keyframes match a full decode bit for bit."""
import json

import numpy as np
from test_splice import N, _clip, _decode

from faceblur import scan, units


def test_units_cover_every_keyframe(tmp_path):
    src = str(tmp_path / "src.mp4")
    _clip(src)
    us, stats = units.build_units([src], chunk=3, threads=2)
    assert stats["index_errors"] == 0 and stats["keyframes"] == N // 8
    assert [u["hi"] - u["lo"] for u in us] == sorted((u["hi"] - u["lo"] for u in us), reverse=True)  # largest first
    assert sum(u["first"] for u in us) == 1
    frames = _decode(src)
    seen = []
    for u in us:
        for pres, arr in scan.decode_unit(*scan.load_unit(u)):
            assert np.array_equal(arr, frames[pres]), f"keyframe {pres} differs from a full decode"
            seen.append(pres)
    assert sorted(seen) == list(range(0, N, 8))  # closed 8-frame GOPs: one IDR per GOP


def test_bad_file_becomes_whole_file_unit(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not an mp4")
    us, stats = units.build_units([str(bad)], chunk=3, threads=1)
    assert stats["index_errors"] == 1 and us == [{"fid": 0, "path": str(bad), "lo": 0, "hi": -1, "step": 1, "first": True}]


def test_key_stride_and_annexb(tmp_path):
    src = str(tmp_path / "src.mp4")
    _clip(src)
    us, stats = units.build_units([src], chunk=2, threads=1, key_stride=2, stride_max_gop=12.0)
    assert stats["index_errors"] == 0 and stats["keyframes"] == (N // 8) // 2
    seen = []
    for u in us:
        idx, sel, roffs, sizes, data = scan.load_unit(u)
        pkts = scan.avcc_unit_to_annexb(idx.stsd_body, sel, roffs, sizes, data)
        assert len(pkts) == len(sel) and pkts[0][1].startswith(b"\x00\x00\x00\x01")
        for pres, _ in scan.decode_unit(idx, sel, roffs, sizes, data):
            seen.append(pres)
    assert sorted(seen) == list(range(0, N, 16))


def test_claim_counter(tmp_path):
    c = tmp_path / "claim"
    c.write_text("0")
    assert [scan.claim(str(c), 3) for _ in range(5)] == [0, 1, 2, None, None]


def test_merge_reports(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    a.write_text(json.dumps({"src": "x.mp4", "frames": 64, "keyframes": 4, "units": 1, "dets": [[8, 1, 2, 3, 4, 0.5, 0.9]]}) + "\n")
    b.write_text(json.dumps({"src": "x.mp4", "frames": 0, "keyframes": 4, "units": 1, "dets": [[0, 1, 2, 3, 4, 0.6, 0.2]]}) + "\n"
                 + json.dumps({"src": "y.mp4", "unit": 3, "error": "boom"}) + "\n")
    tot = scan.merge_reports([str(a), str(b)], str(out))
    assert tot == {"files": 1, "frames": 64, "keyframes": 8, "errors": 1, "faces": 1, "files_with_faces": 1}
    rows = [json.loads(line) for line in open(out)]
    assert rows[0]["dets"][0][0] == 0 and rows[0]["units"] == 2


def test_split_hub():
    assert units.split_hub("hf://datasets/org/repo/videos/a/b.mp4") == ("org/repo", "videos/a/b.mp4")
