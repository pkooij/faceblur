"""CPU-only tests of targeted Pass 2 (`faceblur apply` helpers): ±pad GOP windowing, partial closed-GOP decoding
bit-exactness, and output path mapping."""
import av
import numpy as np
from test_splice import N, _clip, _decode

from faceblur import apply, splice


def test_gop_windows_pad_and_merge():
    gstarts = list(range(0, 64, 8))  # 8 GOPs of 8 frames
    assert apply.gop_windows(gstarts, 64, [0], pad_gops=1) == [(0, 16)]
    assert apply.gop_windows(gstarts, 64, [24], pad_gops=1) == [(16, 40)]
    assert apply.gop_windows(gstarts, 64, [16, 32], pad_gops=1) == [(8, 48)]  # adjacent/overlapping merged
    assert apply.gop_windows(gstarts, 64, [8, 56], pad_gops=1) == [(0, 24), (48, 64)]


def test_decode_gop_run_bit_exact_and_splice(tmp_path):
    src, dst = str(tmp_path / "src.mp4"), str(tmp_path / "dst.mp4")
    _clip(src)
    full = _decode(src)
    with av.open(src) as inp:
        vs = inp.streams.video[0]
        pkts = [p for p in inp.demux(vs) if p.size and p.pts is not None]
        order = sorted(range(len(pkts)), key=lambda i: pkts[i].pts)
        gstarts = [p for p, d in enumerate(order) if pkts[d].is_keyframe]
        wins = apply.gop_windows(gstarts, len(pkts), [24], pad_gops=1)
        assert wins == [(16, 40)]
        ok, decoded = apply.decode_gop_run(pkts, order, vs, 16, 40)
    assert ok and len(decoded) == 24
    for p, arr in decoded:
        assert np.array_equal(arr, full[p]), f"targeted decode of frame {p} differs from full decode"
    # Blur only middle GOP [24, 32) and splice back
    seg = np.stack([arr for p, arr in decoded if 24 <= p < 32]).copy()
    seg[:, 30:70, 50:110] = 0
    r = splice.splice(src, dst, [(24, seg)])
    assert r["mode"] == "splice" and r["frames_reencoded"] == 8
    out = _decode(dst)
    assert len(out) == N
    for k in range(N):
        if 24 <= k < 32:
            assert out[k][30:70, 50:110].mean() < 20
        else:
            assert np.array_equal(out[k], full[k])


def test_dst_path_mapping(tmp_path):
    d = apply.dst_path("hf://datasets/finerobotics/abc-130k/videos/cam/f.mp4", tmp_path)
    assert d == tmp_path / "finerobotics/abc-130k/videos/cam/f.mp4"
    src_root = tmp_path / "in"
    d2 = apply.dst_path(str(src_root / "sub/a.mp4"), tmp_path / "out", src_root=src_root)
    assert d2 == tmp_path / "out/sub/a.mp4"
