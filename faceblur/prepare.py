"""Fetch and convert the models into $FACEBLUR_MODELS (one-off, needs a GPU for the fp16 export):

  yolov11n-face.fp16.onnx          akanametov/yolo-face release 1.0.0, exported by Ultralytics with half=True, dynamic
  siglip2-so400m-patch16-256/      google/siglip2-so400m-patch16-256 from the Hugging Face Hub
  verifier-so400m.npz              logistic probe shipped in this package (faceblur/assets)
  engines/                         TensorRT plans, built on first use per GPU type
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

YOLO_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
YOLO_SHA256 = "9420a9d4933940d2202f511d45df1750e3c1546dd9efc7a3985caae3bbbf7ed2"
SIGLIP = "google/siglip2-so400m-patch16-256"


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(models: Path | None = None) -> None:
    from .models import MODELS
    models = models or MODELS
    models.mkdir(parents=True, exist_ok=True)
    onnx = models / "yolov11n-face.fp16.onnx"
    if not onnx.exists():
        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "yolov11n-face.pt"
            urllib.request.urlretrieve(YOLO_URL, pt)  # noqa: S310 (fixed https URL, checksum verified below)
            if _sha256(pt) != YOLO_SHA256:
                raise RuntimeError("yolov11n-face.pt checksum mismatch")
            code = ("from ultralytics import YOLO; import sys; "
                    "YOLO(sys.argv[1]).export(format='onnx', half=True, dynamic=True, device=0, simplify=True, opset=17)")
            subprocess.run([sys.executable, "-c", code, str(pt)], check=True)
            shutil.move(str(pt.with_suffix(".onnx")), onnx)
        print("wrote", onnx)
    enc = models / "siglip2-so400m-patch16-256"
    if not enc.exists():
        from huggingface_hub import snapshot_download
        snapshot_download(SIGLIP, local_dir=enc, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
        print("wrote", enc)
    probe = models / "verifier-so400m.npz"
    if not probe.exists():
        shutil.copy(Path(__file__).parent / "assets" / "verifier-so400m.npz", probe)
        print("wrote", probe)
