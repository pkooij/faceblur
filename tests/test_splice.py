"""CPU-only tests of the GOP splice: no GPU, no models. A synthetic clip is encoded with the house settings, one GOP
is changed and spliced back; every frame outside that GOP must decode bit-exact and the stream must stay valid."""
from fractions import Fraction

import av
import numpy as np
import pytest

from faceblur import splice

W, H, N = 256, 160, 64


def _clip(path):
    rng = np.random.default_rng(0)
    out = av.open(str(path), "w", format="mp4")
    st = out.add_stream("libx264", rate=30, options=splice.ENC_OPTS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    for k in range(N):
        y = ((np.arange(W)[None, :] + 3 * k + np.arange(H)[:, None]) % 256).astype(np.uint8)
        y = np.clip(y.astype(int) + rng.integers(-3, 4, y.shape), 0, 255).astype(np.uint8)
        f = np.concatenate([y, np.full((H // 2, W), 128, np.uint8)])
        vf = av.VideoFrame.from_ndarray(f, format="yuv420p")
        vf.pts, vf.time_base = k, Fraction(1, 30)
        for p in st.encode(vf):
            out.mux(p)
    for p in st.encode(None):
        out.mux(p)
    out.close()


def _decode(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="yuv420p") for f in c.decode(c.streams.video[0])]


def test_annexb_roundtrip():
    nals = [b"\x67abc", b"\x68de", b"\x65" + bytes(range(50))]
    annexb = b"".join(b"\x00\x00\x00\x01" + n for n in nals)
    assert splice.annexb_nals(annexb) == nals
    assert splice.avcc_nals(splice.to_avcc(nals, 4), 4) == nals


@pytest.mark.parametrize("gop_start", [0, 16, 56])
def test_splice_one_gop(tmp_path, gop_start):
    src, dst = tmp_path / "src.mp4", tmp_path / "dst.mp4"
    _clip(src)
    frames = _decode(src)
    seg = np.stack(frames[gop_start:gop_start + 8]).copy()
    seg[:, 40:80, 60:120] = 0  # the "blur"
    r = splice.splice(str(src), str(dst), [(gop_start, seg)])
    assert r["mode"] == "splice" and r["frames_reencoded"] == 8
    out = _decode(dst)
    assert len(out) == N
    for k in range(N):
        if gop_start <= k < gop_start + 8:
            assert out[k][40:80, 60:120].mean() < 20  # edit survived the re-encode
        else:
            assert np.array_equal(out[k], frames[k]), f"frame {k} changed outside the spliced GOP"


def test_reencode_and_copy(tmp_path):
    src = tmp_path / "src.mp4"
    _clip(src)
    r = splice.reencode(str(src), str(tmp_path / "re.mp4"), {3: [[10, 10, 50, 50]]})
    assert r["frames"] == N and len(_decode(tmp_path / "re.mp4")) == N
    splice.copy(str(src), str(tmp_path / "cp.mp4"))
    assert (tmp_path / "cp.mp4").read_bytes() == src.read_bytes()
