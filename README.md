# faceblur

**High-precision face blurring for large egocentric / robot video datasets, at thousands of frames per second on one GPU.**

`faceblur` finds faces with a fast detector, confirms each one with an image-encoder verifier, blurs it on the GPU and
writes the result while re-encoding as little video as possible. It is tuned for datasets where blurring *non-faces*
(hands, tools, wheels…) is worse than occasionally missing a face, as in manipulation data where the hands are the
signal.

```
         ┌─────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────────────────┐
 mp4 ──▶ │  NVDEC  │──▶│ YOLOv11n-face   │──▶│ SigLIP2 verifier │──▶│ GPU blur │──▶│ copy │ GOP splice │   │──▶ mp4
         │ (NV12)  │   │ TensorRT fp16   │   │ P(face) ≥ 0.8    │   │ on NV12  │   │ full re-encode (x264)│
         └─────────┘   │ score ≥ 0.4     │   │ on context crops │   └──────────┘   └──────────────────────┘
                       └─────────────────┘   └──────────────────┘
          frames stay in GPU memory; only GOPs that need re-encoding are copied to the host
```

## Why it is fast

* **Hardware decode.** NVDEC writes NV12 frames straight into GPU memory. One worker thread per NVDEC engine
  (7 on an H100) keeps them all busy.
* **Detector as a native TensorRT fp16 engine** with dynamic batch and resolution, built once per GPU type and cached.
  Colour conversion, padding, thresholding and NMS all run on the GPU.
* **Verifier only on candidates.** SigLIP2 sees one 256 px context crop per detection, cut with `roi_align` from the
  decoded frame, never whole frames.
* **Blur in place on the YUV planes** (8 px pixelation + Gaussian on luma, matching chroma), vectorised over the batch.
* **Minimal re-encoding.** A video without faces is copied byte for byte. Otherwise only the GOPs that contain a
  blurred frame are re-encoded and spliced into the original bitstream, and every other packet is copied bit-exact.
  Untouched frames keep their exact pixels and the CPU encoders see only a tiny share of the frames. Anything that
  cannot be spliced safely falls back to a full re-encode.

## Throughput

One H100 80GB, 512×288 H.264 egocentric video (closed 8-frame GOPs), default recipe:

| stage | frames / s |
|---|---|
| detector engine alone (batch 256, CUDA graph + GPU NMS) | 31,000 |
| NVDEC decode (7 workers) + detect + blur, frames stay on the GPU | 7,300 – 7,700 |
| **full `faceblur run`**: decode → detect → verify → blur → write (96 videos, 2.6 M frames) | **4,860** |

That is about **160 hours of 30 fps video per GPU-hour**. With `--gpus 0,1,…,7` an 8-GPU node scales with the number
of GPUs, since each GPU brings its own decoders. In the benchmark run 95 of 96 videos had no verified face and were
copied byte for byte; the other one had a single 8-frame GOP re-encoded and spliced.

## Accuracy

Measured on 10,875 hand-reviewed egocentric frames (head- and wrist-mounted cameras, 512 px), verifier scores
out-of-fold. Precision counts detections that hit a reviewed face or a reviewed non-face.

| setting | frames flagged | detections / 1k frames | precision |
|---|---|---|---|
| YOLOv11n-face ≥ 0.3, no verifier | 11.9 % | 129 | 6 % |
| YOLOv11n-face ≥ 0.3 + SigLIP2-base ≥ 0.5 | 0.74 % | 9 | 71 % |
| **default:** YOLOv11n-face ≥ 0.4 + SigLIP2-so400m ≥ 0.8 | **0.23 %** | **2.6** | **96 %** |

The default recipe trades recall for precision. It removes about 98 % of the detector's false positives (hands, tools,
wheels, toys) but also misses many small or partly visible faces. Lower `--ver-thr` and `--det-thr` if recall matters
more for your data, and spot-check results with `faceblur check`.

## Keyframe scan: find what to blur across a whole dataset

`faceblur run` decodes every frame with NVDEC. To find *which* files and frames contain faces across a large
collection, `faceblur scan` looks at keyframes only and decodes them on the CPU, where a node has far more decode
capacity than in its NVDECs:

```
 faceblur units                                faceblur scan --gpus 0,…,7
 ──────────────                                ──────────────────────────────────────────────────────────────
 read each file's MP4 sample table      ──▶    ~10 CPU decoder processes per GPU: claim the next unit from a
 (a few KB, local or Hub range read),          node-wide counter, range-read its keyframes, decode with
 split into ≤ 2,000-keyframe units,            libavcodec into page-locked shared memory  ──▶  per GPU:
 largest first                                 YOLOv11n-face TRT (batch 128) + SigLIP2 verifier (batch ≥ 512)
```

With closed 8-frame GOPs that samples every 8th frame for an eighth of the decode work. Files can be local paths or
Hub files, `hf://datasets/<org>/<repo>/<path>`, streamed with HTTP range requests (no download). Splitting long files
into units keeps every decoder busy until the end, so all GPUs finish within seconds of each other.

```bash
faceblur units files.txt units.jsonl                   # index pass: 701 Hub files in 41 s
faceblur scan units.jsonl --report scan.jsonl --gpus 0,1,2,3,4,5,6,7
```

The report has one line per file with every detection above the detector threshold as
`[frame, x1, y1, x2, y2, detector score, verifier score]`, so the verifier threshold can be chosen afterwards.

One 8×H100 node (92 CPU cores, 10 decoders per GPU), mixed robot datasets streamed from the Hub:

| data | video frames / s per node |
|---|---|
| 701 files from 108 datasets, 15.6 M frames, keyframe every 4.9 frames on average | 99,000 (157 s wall, incl. 53 s startup) |
| egocentric video, 512×288, keyframe every 8 frames | 271,000 |

The scan is bound by CPU decoding (about 330–480 keyframes per second per core), not by the GPUs. A face that is
visible only between two keyframes (under about a quarter of a second at 30 fps) is not seen by the scan.

## Install

Needs an NVIDIA GPU with NVDEC (tested on H100), a CUDA 13 driver and Python ≥ 3.10.

```bash
pip install "faceblur[export] @ git+https://github.com/pkooij/faceblur"
faceblur prepare   # downloads and converts the models into ~/.cache/faceblur (or $FACEBLUR_MODELS)
```

The `export` extra (Ultralytics) is only needed once, for `faceblur prepare`.

## Usage

```bash
# blur every mp4 under SRC into the same tree under DST (outputs that already exist are skipped)
faceblur run SRC DST --report report.jsonl

# every GPU of a node: one process per GPU, files sharded between them
faceblur run SRC DST --gpus 0,1,2,3,4,5,6,7 --report report.jsonl

# throughput on 32 files, output discarded
faceblur bench SRC -n 32

# verify outputs: same frame count, untouched frames bit-exact, blurred frames changed inside the box
faceblur check SRC DST --report report.jsonl
```

From Python:

```python
from faceblur.pipeline import Recipe, run

totals = run([("in/a.mp4", "out/a.mp4")], Recipe(det_thr=0.4, ver_thr=0.8), report="report.jsonl")
```

The report has one JSON line per video: frame count, output mode (`copy`, `splice` or `reencode`), frames re-encoded,
and each blurred box as `[frame, x1, y1, x2, y2, detector score, verifier score]`.

## Options

| flag | default | meaning |
|---|---|---|
| `--det-thr` | 0.4 | YOLOv11n-face score threshold |
| `--ver-thr` | 0.8 | verifier P(face) threshold |
| `--workers` | 7 | decode + GPU threads per GPU (about the number of NVDEC engines) |
| `--encoders` | CPUs − workers | CPU processes for copy / splice / re-encode |
| `--batch` | 64 | frames per GPU batch |
| `--gpus` | | one process per listed GPU |
| `--files` | | text file with paths relative to SRC, instead of `--glob` |

## How the splice works

Inputs must be MP4/H.264 with closed GOPs (each GOP starts with an IDR frame). For every run of GOPs that holds a
blurred frame, `faceblur` encodes the blurred frames with x264 using the source's settings and colour tags, converts
the packets to the track's AVCC framing, gives them the original decode and presentation timestamps and puts them in
place of the original packets. If x264's SPS/PPS differ from the source's, the new parameter sets are sent in-band at
the start of the run and the original ones right after it. Timestamps that would not line up, open GOPs, extra
streams or a decoder/demuxer frame-count mismatch fall back to a full re-encode.

Set `ENC_OPTS` in `faceblur/splice.py` to the encoder settings of your own videos so spliced GOPs match them.

## Layout

```
faceblur/
  cli.py        command line (run, bench, check, prepare)
  pipeline.py   per-GPU orchestration: NVDEC workers, recipe, CPU writer pool, report
  models.py     TensorRT detector engine, SigLIP2 verifier
  gpu.py        CUDA-side kernels: NV12 conversion, YOLO post-processing, verifier crops, blur
  splice.py     copy / GOP splice / full re-encode writers and H.264 bitstream helpers
  verify.py     output checks used by `faceblur check`
  prepare.py    model download and conversion
  units.py      index pass for the keyframe scan: MP4 sample tables → keyframe-range work units
  scan.py       keyframe scan: CPU decoder processes, shared-memory hand-off, GPU detector + verifier
  _vendor/      MP4 index reader and HTTP range fetcher from LeRobot's streaming module (Apache-2.0)
  assets/       verifier probe weights
tests/          CPU-only tests of the bitstream, splice and keyframe-scan paths
```

## Models and licences

The code in this repository is Apache-2.0. The models it downloads have their own licences:

| model | source | licence |
|---|---|---|
| YOLOv11n-face | [akanametov/yolo-face](https://github.com/akanametov/yolo-face) release 1.0.0 (trained with Ultralytics) | GPL-3.0 (Ultralytics: AGPL-3.0) |
| SigLIP2 so400m/16 @ 256 | [google/siglip2-so400m-patch16-256](https://huggingface.co/google/siglip2-so400m-patch16-256) | Apache-2.0 |
| verifier probe | `faceblur/assets/verifier-so400m.npz`: logistic regression on SigLIP2 features, trained on reviewed egocentric crops and WIDER FACE faces | Apache-2.0 |

Check that the detector's licence fits your use before deploying it. WIDER FACE (used only to train the probe) is
distributed under CC BY-NC-ND.

## Tests

```bash
pip install "faceblur[test] @ git+https://github.com/pkooij/faceblur"
pytest tests   # CPU only: bitstream helpers, GOP splice bit-exactness, re-encode and copy paths
```
