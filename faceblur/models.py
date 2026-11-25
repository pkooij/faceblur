"""Model wrappers: YOLOv11n-face as a TensorRT fp16 engine (built once per GPU type and cached) and the SigLIP2 crop
verifier (vision tower in fp16 + logistic-regression probe)."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

MODELS = Path(os.environ.get("FACEBLUR_MODELS", Path.home() / ".cache" / "faceblur"))


def _log(**kw):
    print(json.dumps({"t": round(time.time(), 1), **kw}), flush=True)


class Detector:
    """YOLOv11n-face TRT engine: dynamic batch 1..maxb, H/W 64..1024 (multiples of 32), fp16 in/out."""

    def __init__(self, onnx_path: Path | None = None, maxb: int = 128, opt_hw=(288, 512)):
        import tensorrt as trt
        self.trt = trt
        self.log = trt.Logger(trt.Logger.WARNING)
        onnx_path = Path(onnx_path or MODELS / "yolov11n-face.fp16.onnx")
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        key = hashlib.sha1(onnx_path.read_bytes() + f"{trt.__version__}|{maxb}|{opt_hw}".encode()).hexdigest()[:12]
        plan = MODELS / "engines" / f"yolov11n-face-{gpu}-{key}.plan"
        if not plan.exists():
            plan.parent.mkdir(parents=True, exist_ok=True)
            self._build(onnx_path, plan, maxb, opt_hw)
        self.engine = trt.Runtime(self.log).deserialize_cuda_engine(plan.read_bytes())
        self.inp, self.out = self.engine.get_tensor_name(0), self.engine.get_tensor_name(1)
        self.maxb = maxb

    def _build(self, onnx_path, plan, maxb, opt_hw):
        trt = self.trt
        _log(event="building_trt_engine", plan=str(plan), note="one-off, a few minutes")
        b = trt.Builder(self.log)
        net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(net, self.log)
        if not parser.parse(onnx_path.read_bytes()):
            raise RuntimeError([parser.get_error(i).desc() for i in range(parser.num_errors)])
        cfg = b.create_builder_config()
        cfg.builder_optimization_level = 4
        prof = b.create_optimization_profile()
        prof.set_shape(net.get_input(0).name, (1, 3, 64, 64), (maxb, 3, *opt_hw), (maxb, 3, 1024, 1024))
        cfg.add_optimization_profile(prof)
        ser = b.build_serialized_network(net, cfg)
        tmp = plan.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(bytes(ser))
        tmp.rename(plan)

    def context(self) -> "DetCtx":
        return DetCtx(self)


class DetCtx:
    """One execution context per worker thread (contexts are not thread-safe; the engine is)."""

    def __init__(self, d: Detector):
        self.d, self.ctx, self.shape = d, d.engine.create_execution_context(), None

    def __call__(self, x: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        if tuple(x.shape) != self.shape:
            self.ctx.set_input_shape(self.d.inp, tuple(x.shape))
            self.y = torch.empty(tuple(self.ctx.get_tensor_shape(self.d.out)), dtype=torch.float16, device="cuda")
            self.ctx.set_tensor_address(self.d.out, self.y.data_ptr())
            self.shape = tuple(x.shape)
        self.ctx.set_tensor_address(self.d.inp, x.data_ptr())
        self.ctx.execute_async_v3(stream.cuda_stream)
        return self.y


class Verifier:
    """SigLIP2 image features of a context crop + logistic probe -> P(real face). Shared by all worker threads."""

    def __init__(self, encoder_dir: Path | None = None, probe: Path | None = None):
        from transformers import AutoModel
        encoder_dir = Path(encoder_dir or MODELS / "siglip2-so400m-patch16-256")
        m = AutoModel.from_pretrained(encoder_dir, dtype=torch.float16, attn_implementation="sdpa")
        self.size = int(m.config.vision_config.image_size)
        self.model = m.cuda().eval()
        if hasattr(self.model, "text_model"):
            del self.model.text_model  # vision tower only
        p = np.load(probe or MODELS / "verifier-so400m.npz")
        self.w = torch.from_numpy(p["coef"].reshape(-1)).float().cuda()
        self.b = float(p["intercept"].reshape(-1)[0])
        self.lock = threading.Lock()

    @torch.no_grad()
    def __call__(self, crops: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, len(crops), 256):
            f = self.model.get_image_features(pixel_values=crops[i:i + 256])
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            out.append(torch.sigmoid(f @ self.w + self.b))
        return torch.cat(out) if out else torch.empty(0, device=crops.device)
