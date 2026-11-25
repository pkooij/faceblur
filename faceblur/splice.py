"""Write the blurred video with as little re-encoding as possible (runs in CPU worker processes).

copy      No verified face in the file: the input bytes are copied unchanged.
splice    Only the GOPs that contain a blurred frame are re-encoded (x264 with the house settings and the source's
          colour tags); their packets replace the originals and every other packet is copied bit-exact. Needs closed
          GOPs that start with an IDR frame, as the house profile writes them. If the new SPS/PPS differ from the
          source's, the new ones are sent in-band at the start of each replaced run and the source's right after it.
reencode  Fallback for anything splice cannot do safely: full decode, CPU blur of the recorded boxes, full x264.

Every mode writes `<dst>.tmp` and renames it, so a crash never leaves a partial file at the final path.
"""
from __future__ import annotations

import os
import shutil
from fractions import Fraction

import av
import numpy as np

# Encoder settings for re-encoded GOPs. They match the source encoder (x264 veryfast-like, CRF 23, keyint 8, 3 B-frames,
# closed GOPs) so spliced GOPs share the source's SPS/PPS; change them to match your own videos.
ENC_OPTS = {"crf": "23", "preset": "veryfast", "g": "8", "bf": "3", "keyint_min": "1", "sc_threshold": "40"}


class SpliceError(Exception):
    pass


# ---------------------------------------------------------------- H.264 bitstream helpers
def annexb_nals(b: bytes) -> list[bytes]:
    """Split an Annex B byte stream (start codes) into NAL units."""
    starts, i, n = [], 0, len(b)
    while i < n - 2:
        if b[i] == 0 and b[i + 1] == 0 and b[i + 2] == 1:
            starts.append((i - 1 if i > 0 and b[i - 1] == 0 else i, i + 3))
            i += 3
        else:
            i += 1
    out = []
    for k, (_, e) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else n
        if e < end:
            out.append(b[e:end])
    return out


def avcc_nals(b: bytes, ls: int) -> list[bytes]:
    out, i = [], 0
    while i + ls <= len(b):
        n = int.from_bytes(b[i:i + ls], "big")
        out.append(b[i + ls:i + ls + n])
        i += ls + n
    return out


def to_avcc(nals: list[bytes], ls: int) -> bytes:
    return b"".join(len(x).to_bytes(ls, "big") + x for x in nals)


def parse_avcc_extradata(ex: bytes) -> tuple[int, list[bytes], list[bytes]]:
    """avcC box payload -> (NAL length size, [SPS], [PPS])."""
    if not ex or ex[0] != 1:
        raise SpliceError("stream extradata is not avcC (not an MP4/H.264 track)")
    ls = (ex[4] & 3) + 1
    i, sps, pps = 6, [], []
    for _ in range(ex[5] & 0x1F):
        n = int.from_bytes(ex[i:i + 2], "big")
        sps.append(ex[i + 2:i + 2 + n])
        i += 2 + n
    for _ in range(ex[i]):
        n = int.from_bytes(ex[i + 1:i + 3], "big")
        pps.append(ex[i + 3:i + 3 + n])
        i += 2 + n
    return ls, sps, pps


def _colour_opts(cc) -> dict:
    opts = {}
    for attr in ("color_primaries", "color_trc", "colorspace", "color_range"):
        v = getattr(cc, attr, None)
        if v is not None and int(v) not in (0, 2):  # unknown / unspecified
            opts[attr] = str(int(v))
    return opts


def _encoder(w: int, h: int, rate, src_cc, threads: int):
    cc = av.CodecContext.create("libx264", "w")
    cc.width, cc.height, cc.pix_fmt = w, h, "yuv420p"
    cc.time_base = Fraction(1, 1000)
    cc.framerate = rate
    cc.thread_count = threads
    cc.options = {**ENC_OPTS, **_colour_opts(src_cc)}
    return cc


# ---------------------------------------------------------------- modes
def copy(src: str, dst: str) -> dict:
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return {"mode": "copy"}


def splice(src: str, dst: str, segments: list[tuple[int, np.ndarray]], threads: int = 1) -> dict:
    """segments: sorted (first presentation frame index, yuv420p uint8 [n, 1.5h, w]) runs covering whole GOPs."""
    inp = av.open(src)
    try:
        if len(inp.streams) != 1:
            raise SpliceError("container has more than one stream")
        vs = inp.streams.video[0]
        pkts = [p for p in inp.demux(vs) if p.size and p.pts is not None]
        ls, sps0, pps0 = parse_avcc_extradata(bytes(vs.codec_context.extradata or b""))
        order = sorted(range(len(pkts)), key=lambda i: pkts[i].pts)  # presentation index -> decode index
        pres_pts = [pkts[i].pts for i in order]
        keys = {p for p, d in enumerate(order) if pkts[d].is_keyframe}
        w, h = vs.codec_context.width, vs.codec_context.height
        rate = vs.average_rate or vs.guessed_rate or Fraction(30)
        replace: dict[int, tuple[int, list]] = {}
        for a, frames in segments:
            b = a + len(frames)
            if a not in keys or b > len(order) or (b < len(order) and b not in keys):
                raise SpliceError(f"segment [{a},{b}) is not GOP-aligned")
            dec = sorted(order[a:b])
            if dec != list(range(dec[0], dec[0] + len(dec))):
                raise SpliceError("segment packets are not contiguous in decode order (open GOP?)")
            cc = _encoder(w, h, rate, vs.codec_context, threads)
            new = []
            for k, f in enumerate(frames):
                vf = av.VideoFrame.from_ndarray(f, format="yuv420p")
                vf.pts, vf.time_base = k, cc.time_base
                new += cc.encode(vf)
            new += cc.encode(None)
            if len(new) != len(frames):
                raise SpliceError(f"encoder returned {len(new)} packets for {len(frames)} frames")
            dts = sorted(pkts[i].dts for i in dec)
            run = []
            for j, p in enumerate(new):
                k = int(p.pts)
                run.append({"nals": annexb_nals(bytes(p)), "pts": pres_pts[a + k], "dts": dts[j], "key": bool(p.is_keyframe),
                            "dur": pkts[order[a + k]].duration})
            if not run[0]["key"]:
                raise SpliceError("re-encoded run does not start with a keyframe")
            if any(o["pts"] < o["dts"] for o in run):
                raise SpliceError("pts < dts after timestamp remap")
            replace[dec[0]] = (len(dec), run)

        new_ps = [n for r in replace.values() for o in r[1] for n in o["nals"] if n[0] & 0x1F in (7, 8)]
        new_sps = next((n for n in new_ps if n[0] & 0x1F == 7), None)
        new_pps = next((n for n in new_ps if n[0] & 0x1F == 8), None)
        same_ps = new_sps in sps0 and new_pps in pps0

        tmp = dst + ".tmp"
        outc = av.open(tmp, "w", format="mp4", options={"movflags": "faststart"})
        ost = outc.add_stream_from_template(vs)
        i, resend, n_rep = 0, False, 0
        while i < len(pkts):
            if i in replace:
                run_len, run = replace[i]
                for q, o in enumerate(run):
                    nals = [n for n in o["nals"] if n[0] & 0x1F not in (6, 7, 8, 9)]  # drop SEI/AUD/param sets
                    if not same_ps and q == 0:
                        nals = [new_sps, new_pps] + nals
                    pk = av.Packet(to_avcc(nals, ls))
                    pk.pts, pk.dts, pk.time_base, pk.is_keyframe, pk.duration = o["pts"], o["dts"], vs.time_base, o["key"], o["dur"]
                    pk.stream = ost
                    outc.mux(pk)
                i += run_len
                n_rep += run_len
                resend = not same_ps
                continue
            p = pkts[i]
            data = bytes(p)
            if resend:
                data = to_avcc(sps0 + pps0, ls) + data
                resend = False
            pk = av.Packet(data)
            pk.pts, pk.dts, pk.time_base, pk.is_keyframe, pk.duration = p.pts, p.dts, vs.time_base, p.is_keyframe, p.duration
            pk.stream = ost
            outc.mux(pk)
            i += 1
        outc.close()
    finally:
        inp.close()
    os.replace(tmp, dst)
    return {"mode": "splice", "frames": len(pkts), "frames_reencoded": n_rep, "same_param_sets": bool(same_ps)}


def blur_yuv_cpu(f: np.ndarray, h: int, boxes, grow: float = 0.25) -> np.ndarray:
    """CPU twin of gpu.blur_nv12_ on a yuv420p frame [1.5h, w] (fallback path only)."""
    import cv2
    w = f.shape[1]
    planes = ((f[:h], 1, 8, 3.0), (f[h:h + h // 4].reshape(h // 2, w // 2), 2, 4, 1.5),
              (f[h + h // 4:].reshape(h // 2, w // 2), 2, 4, 1.5))
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        a, b = max(0, x1 - grow * bw), max(0, y1 - grow * bh)
        c, d = min(w, x2 + grow * bw), min(h, y2 + grow * bh)
        for plane, s, blk, sig in planes:
            pa, pb = int(a / s), int(b / s)
            pc, pd = max(int(np.ceil(c / s)), pa + 1), max(int(np.ceil(d / s)), pb + 1)
            roi = plane[pb:pd, pa:pc]
            small = cv2.resize(roi, (max(1, roi.shape[1] // blk), max(1, roi.shape[0] // blk)), interpolation=cv2.INTER_AREA)
            up = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            plane[pb:pd, pa:pc] = cv2.GaussianBlur(up, (0, 0), sig)
    return f


def reencode(src: str, dst: str, boxes_by_frame: dict[int, list], threads: int = 0) -> dict:
    """Full re-encode fallback: decode every frame, blur the recorded boxes on CPU, encode with the house settings."""
    inp = av.open(src)
    vs = inp.streams.video[0]
    vs.thread_type = "AUTO"
    tmp = dst + ".tmp"
    outc = av.open(tmp, "w", format="mp4", options={"movflags": "faststart"})
    rate = vs.average_rate or vs.guessed_rate or Fraction(30)
    st = outc.add_stream("libx264", rate=rate, options={**ENC_OPTS, **_colour_opts(vs.codec_context)})
    st.width, st.height, st.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv420p"
    st.codec_context.thread_count = threads
    h, n = vs.codec_context.height, 0
    for k, fr in enumerate(inp.decode(vs)):
        a = fr.to_ndarray(format="yuv420p")
        if k in boxes_by_frame:
            a = blur_yuv_cpu(a.copy(), h, boxes_by_frame[k])
        vf = av.VideoFrame.from_ndarray(a, format="yuv420p")
        vf.pts, vf.time_base = fr.pts, fr.time_base
        for p in st.encode(vf):
            outc.mux(p)
        n += 1
    for p in st.encode(None):
        outc.mux(p)
    outc.close()
    inp.close()
    os.replace(tmp, dst)
    return {"mode": "reencode", "frames": n, "frames_reencoded": n}
