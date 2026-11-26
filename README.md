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
decodes keyframes only. On GPUs with NVDEC (`--nvdec 8`, default), separate fetcher processes stream and pack Annex-B
IDR keyframes into shared memory (avoiding Python GIL contention) and hardware NVDEC decodes them straight into GPU
NV12 VRAM. A CPU `libavcodec` decoder pool (`--decoders`) can be used instead or alongside NVDEC.

```
 faceblur units                                faceblur scan --gpus 0,…,7
 ──────────────                                ──────────────────────────────────────────────────────────────
 read each file's MP4 sample table      ──▶    fetcher processes: claim units from a node-wide counter,
 (a few KB, local or Hub range read),          range-read keyframes, pack Annex-B into shared memory  ──▶
 split into ≤ 2,000-keyframe units,            per GPU: 8× NVDEC decode into VRAM ──▶ 2× YOLOv11n-face TRT
 largest first (optional --key-stride)         (batch 128) ──▶ SigLIP2 verifier (batch ≥ 512)
```

With closed 8-frame GOPs this samples every 8th frame for ~1/8 of the decode work. Inputs can be local paths or
Hugging Face Hub URIs (`hf://datasets/<org>/<repo>/<path>`) streamed via HTTP range requests (no download). Splitting
long files into ≤ 2,000-keyframe units keeps all GPUs balanced to within a second. Run `faceblur units` ahead of time
on CPU nodes so GPUs stay 100 % utilized.

| hardware / mode | dataset | measured video fps | full 8-GPU node (scaled) | notes |
|---|---|---:|---:|---|
| 8× H100, CPU (`80 dec`) | 701 mixed Hub files, 15.6 M frames (stride 4.9) | 99,000 wall (150,000 steady) | 99,000 wall / 150,000 steady | 41 s index; 157 s scan (53 s startup) |
| 8× H100, CPU (`80 dec`) | local HubEgo 512×288, 106.7 M frames (stride 8.0) | 271,000 | 271,000 | whole-file decode |
| 6× B300, CPU (`54 dec`) | 2,697 catalog-weighted Hub files (78 repos), 59.6 M frames (stride 6.9) | 142,000 steady (132,000 wall) | ~210,000 steady | 72.6 s index; 420 s active (1 s skew); 537 MB/s Hub; 0 errors |
| 2× B300, CPU (`20 dec`) | 420 `abc-130k` Hub files, 10.5 M frames (stride 7.9) | 40,400 active (36,600 wall) | ~161,000 steady | baseline CPU decode (`259.6 s` active, 425 faces / 45 files) |
| **2× B300, NVDEC (`--nvdec 8`)** | **same 420 `abc-130k` Hub files, 10.5 M frames (stride 1)** | **99,800 active (79,600 wall)** | **~399,000 steady** | **2.47× faster**, `0.0 s` skew, **exact 425/425 faces parity**, 0 errors |
| **2× B300, NVDEC (`--nvdec 8`)** | **same 420 `abc-130k` Hub files (`--key-stride 2`)** | **189,500 active (148,800 wall)** | **~758,000 steady** | **4.70× faster**, 586 MB/s Hub on 2 GPUs; ~97.5 % sustained-face recall |
| **2× B300, NVDEC (`--nvdec 8`)** | **local HubEgo 512×288, 33.5 M frames (stride 1)** | **91,900 active (88,700 wall)** | **~368,000 steady** | `0.0 s` skew across GPUs, exact 15/15 faces parity, 0 errors |

A face visible only between two keyframes (under ~0.25 s at 30 fps with GOP 8, or ~0.5 s with `--key-stride 2`) is not
seen by the scan.

### Targeted Pass 2 (`faceblur apply`)

Instead of running full `faceblur run` across the entire dataset (or even across every frame of hit files),
`faceblur apply` reads the `scan` report, opens/downloads **only** files with verified faces, decodes **only** the
closed GOPs within `±1` GOP of each keyframe hit (~0.5–1 % of frames), runs the detector + verifier + GPU blur on
every frame in those GOPs, and splices only the blurred GOPs back bit-exact.

Measured on the 10.5 M-frame `abc-130k` Hub test above (45 hit files / 1.11 M frames inside hit files, `2× B300`):
* **45 / 45 files spliced cleanly** (`0` full re-encodes, `0` errors) in **40.2 s** (including Hub download, targeted
  decode, GPU blur, and x264 splice — equivalent to **~1.0 M dataset video fps** on an 8×B300 node).
* Decoded only **0.53 %** of hit-file frames (`0.056 %` of all 10.5 M frames), blurred **2,867 frames** (`2,875` faces),
  and re-encoded **0.40 %** of hit-file frames.
* `faceblur check` passed **5 / 5** (`0` failures: exact frame counts, untouched frames bit-exact, all blurred boxes
  verified).

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

### Two-pass dataset workflow (`units`, `scan`, `apply`)

```bash
# 1. index pass (CPU only): local paths or hf://datasets/<org>/<repo>/<path> in files.txt
faceblur units files.txt units.jsonl --threads 64          # add --key-stride 2 for 2x faster scan on GOP <= 12

# 2. scan keyframes across all GPUs of a node (NVDEC default)
faceblur scan units.jsonl --report scan.jsonl --gpus 0,1,2,3,4,5,6,7 --nvdec 8

# 3. targeted Pass 2: blur only ±1 GOP around verified hits, splice bit-exact
faceblur apply scan.jsonl DST --gpus 0,1,2,3,4,5,6,7 --pad-gops 1 --report apply.jsonl
```

The `scan` report has one JSON line per file with frame/keyframe counts and every detection above `--det-thr` as
`[frame, x1, y1, x2, y2, detector score, verifier score]`, so `--ver-thr` can be tuned afterwards without re-scanning.
Pass `--cache-dir DIR` to `faceblur apply` to retain downloaded Hub sources if you want to verify outputs with
`faceblur check DIR DST --report apply.jsonl`.

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

### `faceblur units` / `faceblur scan` / `faceblur apply`

| flag | default | meaning |
|---|---|---|
| `--chunk` (`units`) | `2000` | max decoded keyframes per work unit |
| `--threads` (`units`) | `64` | parallel MP4 index readers |
| `--key-stride` (`units`) | `1` | decode every Nth keyframe when file GOP ≤ `--stride-max-gop` (`12`) |
| `--gpus` (`scan`, `apply`) | | comma-separated GPU ids |
| `--nvdec` (`scan`) | `8` | hardware NVDEC keyframe worker threads per GPU (`0` disables NVDEC) |
| `--decoders` (`scan`) | `0` | parallel CPU decoder processes per GPU (`0` = NVDEC only) |
| `--gpu-threads` (`scan`) | `2` | detector threads per GPU |
| `--parts` (`scan`, `apply`) | `4` | parallel HTTP range sub-requests per Hub read |
| `--pad-gops` (`apply`) | `1` | neighboring GOPs to decode and check on each side of each hit |
| `--cache-dir` (`apply`) | | optional dir to keep downloaded Hub sources (e.g. for `faceblur check`) |

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
  cli.py        command line (run, bench, check, prepare, units, scan, apply)
  pipeline.py   per-GPU orchestration: NVDEC workers, recipe, CPU writer pool, report
  models.py     TensorRT detector engine, SigLIP2 verifier
  gpu.py        CUDA kernels: NV12 conversion, YOLO post-processing, verifier crops, blur
  splice.py     copy / GOP splice / full re-encode writers and H.264 bitstream helpers
  verify.py     output checks used by `faceblur check`
  prepare.py    model download and conversion
  units.py      index pass for `scan`: MP4 sample tables → keyframe-range work units
  scan.py       keyframe scan: MP fetcher processes, NVDEC/CPU decode, GPU detect + verify
  apply.py      targeted Pass 2: ±pad GOP decode around verified hits, GPU blur, bit-exact splice
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
