# faceblur

**High-precision, GPU-accelerated face blurring for massive egocentric and robot video datasets.**

`faceblur` detects faces with a TensorRT YOLOv11n-face engine, confirms each candidate with a SigLIP2 crop verifier,
blurs verified faces in place on the GPU, and splices only the modified GOPs back into the MP4 bitstream so untouched
frames remain **bit-exact**. It is tuned for manipulation datasets where blurring hands, tools, or grippers is far worse
than missing an occasional tiny face.

---

## How it works

`faceblur` offers two workflows built on the same detector (`YOLOv11n-face` TRT fp16, default `≥ 0.4`), verifier
(`SigLIP2-so400m` on 256 px context crops, default `≥ 0.8`), GPU NV12 blur (8 px pixelation + Gaussian), and bit-exact
GOP splicer:

### 1. Two-pass dataset pipeline (`units` → `scan` → `apply`) — *recommended for large datasets*

Instead of decoding every frame of every video, split the job into a fast keyframe scan followed by targeted blurring
of only the GOPs around verified hits:

```
Pass 0 · CPU only (faceblur units)
  Read each MP4 sample table (local or hf://datasets/... via HTTP range reads, no download)
  Split into ≤ 2,000-keyframe work units, largest first

Pass 1 · GPU keyframe scan (faceblur scan)
  Fetcher OS processes claim units from a node-wide counter, range-read keyframes, pack Annex-B into shared memory
  ──▶ Per GPU: 8× NVDEC decode straight into VRAM ──▶ 2× YOLOv11n-face TRT (batch 128) ──▶ SigLIP2 verifier
  Writes JSONL with every detection ≥ --det-thr (so --ver-thr can be tuned later without re-scanning)

Pass 2 · Targeted ±1 GOP blur & splice (faceblur apply)
  Opens/downloads ONLY files with verified faces (~15–20% of files)
  Decodes ONLY closed GOPs within ±1 GOP of each keyframe hit (~0.5–1% of frames)
  Runs per-frame YOLO + SigLIP2 + GPU blur ──▶ x264 re-encodes ONLY blurred GOPs ──▶ splices bit-exact
```

### 2. Single-pass full-frame pipeline (`faceblur run`) — *for smaller local directories*

```
         ┌─────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────────────────┐
 mp4 ──▶ │  NVDEC  │──▶│ YOLOv11n-face   │──▶│ SigLIP2 verifier │──▶│ GPU blur │──▶│ copy │ GOP splice │   │──▶ mp4
         │ (NV12)  │   │ TensorRT fp16   │   │ P(face) ≥ 0.8    │   │ on NV12  │   │ full re-encode (x264)│
         └─────────┘   │ score ≥ 0.4     │   │ on context crops │   └──────────┘   └──────────────────────┘
                       └─────────────────┘   └──────────────────┘
          frames stay in GPU memory; only GOPs that need re-encoding are copied to the host
```

### Bit-exact GOP splicing

Inputs must be MP4/H.264 with closed GOPs (each GOP starts with an IDR frame). For each run of GOPs containing a blurred
frame, `faceblur` re-encodes only those frames with x264 matching the source colour tags and settings (`ENC_OPTS` in
`faceblur/splice.py`), converts packets back to AVCC framing with the original PTS/DTS, and copies every other packet
byte-for-byte. Clean files are untouched (in `apply`) or byte-copied (in `run`). Open GOPs, extra streams, or timestamp
mismatches fall back automatically to a full re-encode.

---

## Install

Requires an NVIDIA GPU with NVDEC (tested on H100 and B300), a CUDA 13 driver, and Python ≥ 3.10.

```bash
pip install "faceblur[export] @ git+https://github.com/pkooij/faceblur"
faceblur prepare   # downloads & converts models into ~/.cache/faceblur (or $FACEBLUR_MODELS)
```

The `export` extra (`ultralytics`) is only needed once, for `faceblur prepare`. TensorRT engines are built once per GPU
architecture on first run (~1 min on B300) and cached automatically under `$FACEBLUR_MODELS/engines/`.

---

## How to run it

### Large dataset two-pass workflow (local files or Hugging Face Hub URIs)

Put one local path or `hf://datasets/<org>/<repo>/<path>` URI per line in `files.txt`:

```bash
# 1. Index pass (CPU only — run on CPU nodes so GPUs stay 100% utilized)
faceblur units files.txt units.jsonl --threads 64

# 2. Keyframe scan on GPUs (NVDEC default; streams Hub files via HTTP range reads, no download)
faceblur scan units.jsonl --report scan.jsonl --gpus 0,1,2,3,4,5,6,7 --nvdec 8

# 3. Targeted Pass 2: download/open only hit files, blur ±1 GOP around hits, splice bit-exact
faceblur apply scan.jsonl DST --gpus 0,1,2,3,4,5,6,7 --pad-gops 1 --report apply.jsonl

# 4. Optional: verify outputs frame-by-frame (pass --cache-dir SRC_CACHE to `apply` for Hub sources)
faceblur check SRC_CACHE DST --report apply.jsonl -n 10
```

> **Cluster tip:** `faceblur scan` with NVDEC only needs ~8–12 CPUs per GPU and achieves its highest per-GPU Hub
> throughput in **2-GPU shards across nodes** (~50k video fps/GPU) without waiting for exclusive 8-GPU nodes. Add
> `--key-stride 2` to `faceblur units` for ~2× faster scanning on dense GOP-8 data if missing brief `< 0.5 s` face
> glimpses is acceptable (default `--key-stride 1` is recommended for strict PII removal).

### Single-pass local directory (`run`, `bench`, `check`)

```bash
# Blur every mp4 under SRC into the same tree under DST (resumable: skips existing outputs)
faceblur run SRC DST --gpus 0,1,2,3,4,5,6,7 --report report.jsonl

# Benchmark single-GPU full-frame throughput on 32 files (output discarded)
faceblur bench SRC -n 32 --workers 10   # 7 on H100, 10 on B300

# Verify outputs: identical frame count, untouched frames bit-exact, blurred boxes modified
faceblur check SRC DST --report report.jsonl
```

From Python:

```python
from faceblur.pipeline import Recipe, run

totals = run([("in/a.mp4", "out/a.mp4")], Recipe(det_thr=0.4, ver_thr=0.8), report="report.jsonl")
```

---

## Throughput

All measurements use the default recipe (`--det-thr 0.4 --ver-thr 0.8`) with zero errors. B300 benchmarks ran on
`NVIDIA B300 SXM6 AC` (`cc 10.3`) with dual-socket `Intel Xeon Platinum 8559C` (92 usable CPUs/node).

### Pass 1: Keyframe scan (`faceblur scan`)

| hardware & mode | dataset (GOP stride) | measured active fps | per-GPU fps | notes |
|---|---|---:|---:|---|
| 8× H100, CPU (`80 dec`) | 701 mixed Hub files, 15.6 M frames (`4.9`) | 150,000 *(99k wall)* | 18,800 | `41 s` index, `157 s` scan (`53 s` startup) |
| 8× H100, CPU (`80 dec`) | local HubEgo 512×288, 106.7 M frames (`8.0`) | 271,000 | 33,900 | whole-file CPU decode |
| 6× B300, CPU (`54 dec`) | 2,697 catalog-weighted Hub files / 78 repos, 59.6 M frames (`6.9`) | 142,000 *(132k wall)* | 23,600 | `73 s` index, `420 s` active (`1 s` skew), `537 MB/s` Hub |
| 2× B300, CPU (`20 dec`) | 420 `abc-130k` Hub files, 10.5 M frames (`7.9`) | 40,400 *(36.6k wall)* | 20,200 | CPU baseline (`260 s` active, 425 faces / 45 files) |
| **2× B300, NVDEC (`--nvdec 8`)** | **same 420 `abc-130k` Hub files (`stride 1`)** | **99,800** *(79.6k wall)* | **49,900** | **2.47× faster**, `0.0 s` skew, **exact 425/425 parity** |
| **6× B300, NVDEC (`--nvdec 8`)** | **same 420 `abc-130k` Hub files (`stride 1`)** | **150,100** *(117k wall)* | 25,000 | **2.06× vs 60-CPU baseline**, `0.3 s` skew, `464 MB/s` Hub |
| **2× B300, NVDEC (`--nvdec 8`)** | **same 420 `abc-130k` Hub files (`--key-stride 2`)** | **189,500** *(149k wall)* | **94,800** | **4.70× faster**, `586 MB/s` Hub on 2 GPUs |
| **6× B300, NVDEC (`--nvdec 8`)** | **same 420 `abc-130k` Hub files (`--key-stride 2`)** | **294,300** *(189k wall)* | 49,100 | `35.6 s` active, **`910 MB/s` Hub** on 1 node |
| **2× B300, NVDEC (`--nvdec 8`)** | **local HubEgo 512×288, 33.5 M frames (`stride 1`)** | **91,900** *(88.7k wall)* | **46,000** | `0.0 s` skew, **exact 15/15 parity** (~368k on 8× B300) |

### Pass 2: Targeted ±1 GOP blur & splice (`faceblur apply`)

Measured on the 10.5 M-frame `abc-130k` Hub run above (`45` hit files / `1.11 M` hit-file frames):

* **`2× B300`:** `40.2 s` wall (**including Hub download, targeted decode, GPU blur, and x264 splice**) → **~261k effective dataset fps** (~130k dataset fps/GPU, ~1 M/node).
* **45 / 45 files spliced cleanly** (`0` full re-encodes, `0` errors); decoded only **0.53 %** of hit-file frames (`0.056 %` of all 10.5 M frames), blurred **2,867 frames** (`2,875` faces), and re-encoded **0.40 %** of hit-file frames.
* **`faceblur check`:** **10 / 10 passed** (`0` failures: exact frame count, untouched frames bit-exact, blurred boxes verified).

### Single-pass full-frame pipeline (`faceblur run` / `faceblur bench`)

Measured on local 512×288 H.264 egocentric video (closed 8-frame GOPs):

| GPU | stage / setting | frames / s | 30 fps video h / GPU-h |
|---|---|---:|---:|
| 1× H100 80GB | detector engine alone (batch 256, CUDA graph + GPU NMS) | 31,000 | — |
| 1× H100 80GB | NVDEC decode (`--workers 7`) + detect + blur (GPU only) | 7,300 – 7,700 | — |
| 1× H100 80GB | **full `faceblur run`** (`--workers 7`, 96 videos, 2.6 M frames) | **4,860** | ~160 |
| 1× B300 SXM6 | **full `faceblur bench`** (`--workers 7`, 64 videos, 1.72 M frames) | 5,206 | ~174 |
| 1× B300 SXM6 | **full `faceblur bench`** (`--workers 10`, 32 videos, 0.86 M frames) | **6,228** | **~208** |

---

## Accuracy

Measured on 10,875 hand-reviewed egocentric frames (head- and wrist-mounted cameras, 512 px), verifier scores out-of-fold:

| setting | frames flagged | detections / 1k frames | precision |
|---|---:|---:|---:|
| YOLOv11n-face ≥ 0.3, no verifier | 11.9 % | 129 | 6 % |
| YOLOv11n-face ≥ 0.3 + SigLIP2-base ≥ 0.5 | 0.74 % | 9 | 71 % |
| **default:** YOLOv11n-face ≥ 0.4 + SigLIP2-so400m ≥ 0.8 | **0.23 %** | **2.6** | **96 %** |

The default recipe removes ~98 % of detector false positives (hands, tools, wheels, toys) while keeping >90 % of verified
detections on real faces/people in cross-dataset reviews. Lower `--ver-thr` and `--det-thr` if recall matters more for
your data.

---

## Options

### `faceblur units` / `faceblur scan` / `faceblur apply`

| flag | default | meaning |
|---|---|---|
| `--chunk` (`units`) | `2000` | max decoded keyframes per work unit |
| `--threads` (`units`) | `64` | parallel MP4 index readers |
| `--key-stride` (`units`) | `1` | decode every Nth keyframe when file GOP ≤ `--stride-max-gop` (`12`) |
| `--gpus` (`scan`, `apply`) | | comma-separated GPU ids |
| `--nvdec` (`scan`) | `8` | hardware NVDEC keyframe worker threads per GPU (`0` disables NVDEC) |
| `--decoders` (`scan`) | `0` | parallel CPU decoder processes per GPU (`0` = NVDEC only) |
| `--gpu-threads` (`scan`) | `2` | GPU detector threads per GPU |
| `--parts` (`scan`, `apply`) | `4` | parallel HTTP range sub-requests per Hub read |
| `--pad-gops` (`apply`) | `1` | neighboring GOPs decoded & checked on each side of each hit |
| `--cache-dir` (`apply`) | | optional dir to retain downloaded Hub sources (e.g. for `faceblur check`) |

### `faceblur run` / `faceblur bench`

| flag | default | meaning |
|---|---|---|
| `--det-thr` / `--ver-thr` | `0.4` / `0.8` | YOLOv11n-face score and SigLIP2 verifier P(face) thresholds |
| `--workers` | `7` | NVDEC decode + GPU threads per GPU (`7` on H100, `10` on B300) |
| `--encoders` | `CPUs − workers` | CPU processes for copy / splice / re-encode |
| `--batch` | `64` | frames per GPU batch |
| `--gpus` | | comma-separated GPU ids (one process per GPU, sharded) |
| `--files` | | text file with paths relative to `SRC`, instead of `--glob` |

---

## Layout & Tests

```
faceblur/
  cli.py        command line (run, bench, check, prepare, units, scan, apply)
  units.py      index pass: MP4 sample tables → keyframe-range work units
  scan.py       keyframe scan: MP fetcher processes, NVDEC/CPU decode, GPU detect + verify
  apply.py      targeted Pass 2: ±pad GOP decode around hits, GPU blur, bit-exact splice
  pipeline.py   single-pass full-frame NVDEC pipeline
  models.py     TensorRT detector engine, SigLIP2 verifier
  gpu.py        CUDA kernels: NV12 conversion, YOLO post-processing, verifier crops, blur
  splice.py     copy / GOP splice / full re-encode writers and H.264 bitstream helpers
  verify.py     output checks used by `faceblur check`
  prepare.py    model download and conversion
  _vendor/      MP4 index reader and HTTP range fetcher from LeRobot (Apache-2.0)
  assets/       verifier probe weights
tests/          CPU-only tests (bitstream, GOP splice bit-exactness, units/scan, targeted apply)
```

```bash
pip install "faceblur[test] @ git+https://github.com/pkooij/faceblur"
pytest tests
```

## Models and licences

Code in this repository is Apache-2.0. Downloaded models have their own licences:

| model | source | licence |
|---|---|---|
| YOLOv11n-face | [akanametov/yolo-face](https://github.com/akanametov/yolo-face) release 1.0.0 (trained with Ultralytics) | GPL-3.0 (Ultralytics: AGPL-3.0) |
| SigLIP2 so400m/16 @ 256 | [google/siglip2-so400m-patch16-256](https://huggingface.co/google/siglip2-so400m-patch16-256) | Apache-2.0 |
| verifier probe | `faceblur/assets/verifier-so400m.npz`: logistic regression on SigLIP2 features (egocentric crops + WIDER FACE) | Apache-2.0 |

Check that the detector's licence fits your use before deploying it. WIDER FACE (used only to train the probe) is
distributed under CC BY-NC-ND.
