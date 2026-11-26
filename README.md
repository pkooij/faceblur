# faceblur

**High-precision face blurring for large egocentric and robot video datasets, at thousands of frames per second per GPU.**

`faceblur` detects faces with a TensorRT YOLOv11n-face engine, confirms each candidate with a SigLIP2 image-encoder
verifier, blurs verified boxes in place on the GPU, and writes output videos while re-encoding as little as possible. It
is tuned for datasets where blurring *non-faces* (hands, tools, grippers, wheels) is worse than occasionally missing a
small face.

```
         ┌─────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────────────────┐
 mp4 ──▶ │  NVDEC  │──▶│ YOLOv11n-face   │──▶│ SigLIP2 verifier │──▶│ GPU blur │──▶│ copy │ GOP splice │   │──▶ mp4
         │ (NV12)  │   │ TensorRT fp16   │   │ P(face) ≥ 0.8    │   │ on NV12  │   │ full re-encode (x264)│
         └─────────┘   │ score ≥ 0.4     │   │ on context crops │   └──────────┘   └──────────────────────┘
                       └─────────────────┘   └──────────────────┘
          frames stay in GPU memory; only GOPs that need re-encoding are copied to the host
```

## Why it is fast

* **Hardware decode.** NVDEC writes NV12 frames straight into GPU memory; multiple worker threads per GPU keep all
  NVDEC engines busy (`--workers 7` on H100, `--workers 10` on B300).
* **Native TensorRT fp16 detector** with dynamic batch and resolution, built once per GPU architecture and cached.
  Colour conversion, padding, thresholding, and NMS all run on the GPU.
* **Verifier on crops only.** SigLIP2 runs on 256 px context crops cut with `roi_align` from decoded GPU frames, never
  on full frames.
* **In-place GPU blur** on YUV planes (8 px pixelation + Gaussian on luma, matching chroma), vectorised over the batch.
* **Minimal re-encoding.** Videos without verified faces are copied byte for byte. Otherwise only GOPs containing a
  blurred frame are re-encoded and spliced back into the original bitstream, keeping untouched frames bit-exact. Inputs
  that cannot be spliced safely fall back to a full re-encode.

## Throughput

### Full-frame blur pipeline (`faceblur run` / `faceblur bench`)

Measured on local 512×288 H.264 egocentric video (closed 8-frame GOPs), default recipe (`--det-thr 0.4 --ver-thr 0.8`):

| GPU | stage / setting | frames / s | 30 fps video h / GPU-h |
|---|---|---:|---:|
| 1× H100 80GB | detector engine alone (batch 256, CUDA graph + GPU NMS) | 31,000 | — |
| 1× H100 80GB | NVDEC decode (`--workers 7`) + detect + blur (GPU only) | 7,300 – 7,700 | — |
| 1× H100 80GB | **full `faceblur run`** (`--workers 7`, 96 videos, 2.6 M frames) | **4,860** | ~160 |
| 1× B300 SXM6 | **full `faceblur bench`** (`--workers 7`, 64 videos, 1.72 M frames) | 5,206 | ~174 |
| 1× B300 SXM6 | **full `faceblur bench`** (`--workers 10`, 32 videos, 0.86 M frames) | **6,228** | **~208** |

With `--gpus 0,1,…,7`, an 8-GPU node scales linearly with GPU count since each GPU brings its own NVDEC engines (~50k
frames/s on an 8×B300 node). In the H100 run 95 of 96 videos had no verified face and were copied byte for byte; the
other had a single 8-frame GOP re-encoded and spliced.

### Keyframe scan (`faceblur units` + `faceblur scan`)

To find *which* files and frames contain faces across a large collection without decoding every frame, `faceblur scan`
decodes keyframes only on the CPU (where a node has far more decode capacity than in NVDEC) and hands them to the GPUs
via page-locked shared memory:

```
 faceblur units                                faceblur scan --gpus 0,…,7
 ──────────────                                ──────────────────────────────────────────────────────────────
 read each file's MP4 sample table      ──▶    ~10 CPU decoder processes per GPU: claim the next unit from a
 (a few KB, local or Hub range read),          node-wide counter, range-read its keyframes, decode with
 split into ≤ 2,000-keyframe units,            libavcodec into page-locked shared memory  ──▶  per GPU:
 largest first                                 YOLOv11n-face TRT (batch 128) + SigLIP2 verifier (batch ≥ 512)
```

With closed 8-frame GOPs this samples every 8th frame for ~1/8 of the decode work. Inputs can be local paths or
Hugging Face Hub URIs (`hf://datasets/<org>/<repo>/<path>`) streamed via HTTP range requests (no download). Splitting
long files into ≤ 2,000-keyframe units keeps all GPUs balanced to within a few seconds.

| hardware | dataset | measured video fps | full-node (92 CPU) video fps | notes |
|---|---|---:|---:|---|
| 8× H100 (92 CPUs, 80 dec) | 701 mixed Hub files, 15.6 M frames (stride 4.9) | 99,000 wall (150,000 steady) | 99,000 wall / 150,000 steady | 41 s index; 157 s scan (53 s startup) |
| 8× H100 (92 CPUs, 80 dec) | local HubEgo 512×288, 106.7 M frames (stride 8.0) | 271,000 | 271,000 | whole-file decode |
| 6× B300 (60 CPUs, 60 dec) | same 701 mixed Hub files, 15.6 M frames (stride 4.9) | 122,000 steady (98,500 wall) | ~163,000 – 188,000 steady | 25.7 s index; 127 s active (17.8 s startup); 0 errors |
| 6× B300 (60 CPUs, 54 dec) | 2,697 catalog-weighted Hub files (78 repos), 59.6 M frames (stride 6.9) | **142,000 steady** (132,000 wall) | **~210,000 steady** (~170,000 incl. index) | 72.6 s index; 420 s active (1 s skew across 6 GPUs); 537 MB/s Hub; 0 errors |
| 6× B300 (60 CPUs, 60 dec) | local HubEgo 512×288, 33.5 M frames (stride 8.0) | 200,000 steady (181,000 wall) | ~266,000 – 306,000 steady | 25.7 s index; 167.5 s active; 793 MB/s `/fsx` |

The scan is **CPU-decode bound** (~380–420 keyframes/s per Xeon Platinum 8559C core on the B300 nodes), leaving B300
GPUs ~75–85 % idle. Full-node B300 figures scale linearly from the 60-CPU measurements to a 92-CPU (8×B300) node. A face
visible only between two keyframes (under ~0.25 s at 30 fps with GOP 8) is not seen by the scan.

## Accuracy

Measured on 10,875 hand-reviewed egocentric frames (head- and wrist-mounted cameras, 512 px), verifier scores
out-of-fold. Precision counts detections that hit a reviewed face or a reviewed non-face.

| setting | frames flagged | detections / 1k frames | precision |
|---|---:|---:|---:|
| YOLOv11n-face ≥ 0.3, no verifier | 11.9 % | 129 | 6 % |
| YOLOv11n-face ≥ 0.3 + SigLIP2-base ≥ 0.5 | 0.74 % | 9 | 71 % |
| **default:** YOLOv11n-face ≥ 0.4 + SigLIP2-so400m ≥ 0.8 | **0.23 %** | **2.6** | **96 %** |

The default recipe trades recall for precision: it removes ~98 % of detector false positives (hands, tools, wheels,
toys) while keeping >90 % of verified detections on real faces/people in cross-dataset spot checks. Lower `--ver-thr` and
`--det-thr` if recall matters more for your data, and spot-check results with `faceblur check`.

## Install

Needs an NVIDIA GPU with NVDEC (tested on H100 and B300), a CUDA 13 driver, and Python ≥ 3.10.

```bash
pip install "faceblur[export] @ git+https://github.com/pkooij/faceblur"
faceblur prepare   # downloads and converts the models into ~/.cache/faceblur (or $FACEBLUR_MODELS)
```

The `export` extra (Ultralytics) is only needed once, for `faceblur prepare`.

## Usage

### Blur videos (`run`, `bench`, `check`)

```bash
# blur every mp4 under SRC into the same tree under DST (existing outputs are skipped)
faceblur run SRC DST --report report.jsonl

# every GPU of a node: one process per GPU, files sharded between them
faceblur run SRC DST --gpus 0,1,2,3,4,5,6,7 --report report.jsonl

# throughput check on 32 files (output discarded)
faceblur bench SRC -n 32 --workers 10

# verify outputs: same frame count, untouched frames bit-exact, blurred frames changed inside each box
faceblur check SRC DST --report report.jsonl
```

From Python:

```python
from faceblur.pipeline import Recipe, run

totals = run([("in/a.mp4", "out/a.mp4")], Recipe(det_thr=0.4, ver_thr=0.8), report="report.jsonl")
```

The `run` report has one JSON line per video: frame count, output mode (`copy`, `splice`, or `reencode`), frames
re-encoded, and each blurred box as `[frame, x1, y1, x2, y2, detector score, verifier score]`.

### Keyframe scan (`units`, `scan`)

```bash
# 1. index pass: local paths or hf://datasets/<org>/<repo>/<path> in files.txt
faceblur units files.txt units.jsonl --threads 64

# 2. scan keyframes across all GPUs of a node
faceblur scan units.jsonl --report scan.jsonl --gpus 0,1,2,3,4,5,6,7 --decoders 10
```

The `scan` report has one JSON line per file with frame/keyframe counts and every detection above `--det-thr` as
`[frame, x1, y1, x2, y2, detector score, verifier score]`, so `--ver-thr` can be tuned afterwards without re-scanning.

## Options

### `faceblur run` / `faceblur bench`

| flag | default | meaning |
|---|---|---|
| `--det-thr` | `0.4` | YOLOv11n-face score threshold |
| `--ver-thr` | `0.8` | SigLIP2 verifier P(face) threshold |
| `--workers` | `7` | NVDEC decode + GPU threads per GPU (`7` on H100, `10` on B300) |
| `--encoders` | `CPUs − workers` | CPU processes for copy / splice / re-encode |
| `--batch` | `64` | frames per GPU batch |
| `--gpus` | | comma-separated GPU ids (one process per GPU, sharded) |
| `--files` | | text file with paths relative to `SRC`, instead of `--glob` |

### `faceblur units` / `faceblur scan`

| flag | default | meaning |
|---|---|---|
| `--chunk` (`units`) | `2000` | max keyframes per work unit |
| `--threads` (`units`) | `64` | parallel MP4 index readers |
| `--gpus` (`scan`) | | comma-separated GPU ids sharing one work queue |
| `--decoders` (`scan`) | `10` | CPU decoder processes per GPU (~`CPUs / GPUs`) |
| `--gpu-threads` (`scan`) | `2` | detector threads per GPU |
| `--parts` (`scan`) | `4` | parallel HTTP range sub-requests per Hub read |
| `--fetchers` / `--prefetch` (`scan`) | `2` / `2` | prefetch threads and queued units per decoder |
| `--det-thr` / `--ver-thr` (`scan`) | `0.4` / `0.8` | detector threshold (kept in report) and verifier threshold (used for counts) |

## How the splice works

Inputs must be MP4/H.264 with closed GOPs (each GOP starts with an IDR frame). For every run of GOPs containing a
blurred frame, `faceblur` encodes the blurred frames with x264 using the source's settings and colour tags, converts the
packets to the track's AVCC framing, restores the original decode and presentation timestamps, and splices them in place
of the original packets. If x264's SPS/PPS differ from the source, the new parameter sets are sent in-band at the start
of the run and the originals right after it. Misaligned timestamps, open GOPs, extra streams, or a decoder/demuxer
frame-count mismatch fall back to a full re-encode.

Set `ENC_OPTS` in `faceblur/splice.py` to match the encoder settings of your own videos.

## Layout

```
faceblur/
  cli.py        command line (run, bench, check, prepare, units, scan)
  pipeline.py   per-GPU orchestration: NVDEC workers, recipe, CPU writer pool, report
  models.py     TensorRT detector engine, SigLIP2 verifier
  gpu.py        CUDA kernels: NV12 conversion, YOLO post-processing, verifier crops, blur
  splice.py     copy / GOP splice / full re-encode writers and H.264 bitstream helpers
  verify.py     output checks used by `faceblur check`
  prepare.py    model download and conversion
  units.py      index pass for `scan`: MP4 sample tables → keyframe-range work units
  scan.py       keyframe scan: CPU decoder processes, shared-memory hand-off, GPU detect + verify
  _vendor/      MP4 index reader and HTTP range fetcher from LeRobot's streaming module (Apache-2.0)
  assets/       verifier probe weights
tests/          CPU-only tests of the bitstream, splice, and keyframe-scan paths
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
pytest tests   # CPU only: bitstream helpers, GOP splice bit-exactness, re-encode, copy, and keyframe-scan paths
```
