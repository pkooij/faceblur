"""GPU kernels: NV12 colour conversion, YOLO post-processing, verifier crops and in-place face blur.

All functions take/return CUDA tensors and run on the caller's current stream. Frames are NV12 as NVDEC returns them:
uint8 [b, 1.5*h, w] with the Y plane in rows [0, h) and interleaved UV at half resolution in rows [h, 1.5h).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision


def nv12_to_rgb(x: torch.Tensor, h: int) -> torch.Tensor:
    """NV12 uint8 [b,1.5h,w] -> float RGB [b,3,h,w] in 0..255 (BT.709 limited range, as the house encodes)."""
    b, _, w = x.shape
    y = (x[:, :h].float() - 16.0) * 1.164
    uv = x[:, h:].reshape(b, h // 2, w // 2, 2).permute(0, 3, 1, 2).float() - 128.0
    uv = F.interpolate(uv, scale_factor=2, mode="nearest")
    u, v = uv[:, 0], uv[:, 1]
    return torch.stack([y + 1.793 * v, y - 0.213 * u - 0.533 * v, y + 2.112 * u], 1).clamp_(0, 255)


def detector_input(x: torch.Tensor, h: int) -> torch.Tensor:
    """NV12 -> fp16 RGB /255, zero-padded right/bottom to multiples of 32 (YOLO strides)."""
    rgb = nv12_to_rgb(x, h).mul_(1 / 255)
    ph, pw = (-rgb.shape[2]) % 32, (-rgb.shape[3]) % 32
    if ph or pw:
        rgb = F.pad(rgb, (0, pw, 0, ph))
    return rgb.half().contiguous()


def yolo_postproc(y: torch.Tensor, thr: float, h: int, w: int, iou: float = 0.45):
    """YOLO head [b, 4+1(+kps), N] (cx,cy,w,h,score) -> (frame idx [n], boxes xyxy [n,4] float, scores [n])."""
    s = y[:, 4]
    fb, a = torch.nonzero(s >= thr, as_tuple=True)
    if not len(fb):
        return fb, torch.empty(0, 4, device=y.device), torch.empty(0, device=y.device)
    p = y[fb, :, a].float()
    xyxy = torch.cat([p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2], 1)
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
    keep = torchvision.ops.batched_nms(xyxy, p[:, 4], fb, iou)
    return fb[keep], xyxy[keep], p[keep, 4]


def context_rois(fb: torch.Tensor, bx: torch.Tensor, k: float = 3.0, minside: float = 96.0) -> torch.Tensor:
    """Square context windows (k x the box's longer side, at least `minside` px) as roi_align rois [n,5]."""
    c = (bx[:, :2] + bx[:, 2:]) / 2
    s = torch.clamp((bx[:, 2:] - bx[:, :2]).max(1).values * k, min=minside)
    half = (s / 2)[:, None]
    return torch.cat([fb[:, None].float(), c - half, c + half], 1)


def verifier_crops(x: torch.Tensor, h: int, fb: torch.Tensor, bx: torch.Tensor, size: int) -> torch.Tensor:
    """Context crops for the verifier, [n,3,size,size] normalised to [-1,1] (SigLIP), zero outside the frame."""
    frames, inv = torch.unique(fb, return_inverse=True)
    rgb = nv12_to_rgb(x.index_select(0, frames), h)
    rois = context_rois(inv, bx)
    crops = torchvision.ops.roi_align(rgb, rois, output_size=size, spatial_scale=1.0, sampling_ratio=0, aligned=True)
    return ((crops / 255.0 - 0.5) / 0.5).half()


_G: dict = {}


def _gauss(sigma: float, device) -> tuple[torch.Tensor, int]:
    key = (sigma, str(device))
    if key not in _G:
        r = int(3 * sigma)
        k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
        _G[key] = ((k / k.sum()).to(device), r)
    return _G[key]


def _pixelate_blur(p: torch.Tensor, block: int, sigma: float) -> torch.Tensor:
    k, r = _gauss(sigma, p.device)
    q = F.avg_pool2d(p, block, ceil_mode=True)
    q = F.interpolate(q, scale_factor=block, mode="nearest")[..., : p.shape[2], : p.shape[3]]
    c = q.shape[1]
    q = F.conv2d(F.pad(q, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(q, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def blur_nv12_(x: torch.Tensor, h: int, fb: torch.Tensor, bx: torch.Tensor, grow: float = 0.25) -> torch.Tensor:
    """Blur (8 px pixelate + gaussian) every box grown by `grow` on each side, in place on NV12 frames.
    Returns the sorted indices of the frames that were modified."""
    if not len(fb):
        return fb
    w = x.shape[2]
    frames, inv = torch.unique(fb, return_inverse=True)
    wh = bx[:, 2:] - bx[:, :2]
    b = torch.cat([bx[:, :2] - grow * wh, bx[:, 2:] + grow * wh], 1)
    b[:, 0::2] = b[:, 0::2].clamp(0, w)
    b[:, 1::2] = b[:, 1::2].clamp(0, h)
    ys = torch.arange(h, device=x.device, dtype=torch.float32).view(1, h, 1) + 0.5
    xs = torch.arange(w, device=x.device, dtype=torch.float32).view(1, 1, w) + 0.5
    m = (xs >= b[:, 0, None, None]) & (xs < b[:, 2, None, None]) & (ys >= b[:, 1, None, None]) & (ys < b[:, 3, None, None])
    mask = torch.zeros(len(frames), h, w, device=x.device).index_add_(0, inv, m.float()) > 0
    sub = x.index_select(0, frames)
    yp = sub[:, :h].float().unsqueeze(1)
    uv = sub[:, h:].reshape(len(frames), h // 2, w // 2, 2).permute(0, 3, 1, 2).float()
    mc = F.max_pool2d(mask.float().unsqueeze(1), 2) > 0
    yo = torch.where(mask.unsqueeze(1), _pixelate_blur(yp, 8, 3.0), yp)
    uvo = torch.where(mc, _pixelate_blur(uv, 4, 1.5), uv)
    sub[:, :h] = yo.squeeze(1).round_().clamp_(0, 255).to(torch.uint8)
    sub[:, h:] = uvo.permute(0, 2, 3, 1).reshape(len(frames), h // 2, w).round_().clamp_(0, 255).to(torch.uint8)
    x.index_copy_(0, frames, sub)
    return frames


def nv12_to_yuv420p(x: torch.Tensor, h: int) -> torch.Tensor:
    """NV12 [b,1.5h,w] -> planar yuv420p with the same [b,1.5h,w] layout PyAV's from_ndarray expects."""
    b, _, w = x.shape
    uv = x[:, h:].reshape(b, h // 2, w // 2, 2)
    return torch.cat([x[:, :h].reshape(b, -1), uv[..., 0].reshape(b, -1), uv[..., 1].reshape(b, -1)], 1).reshape(b, h * 3 // 2, w)
