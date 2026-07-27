# -*- coding: utf-8 -*-
# Small generic helpers: IO, metrics, bbox/ROI geometry, meta (de)normalization.

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def psnr_from_mse(mse: float, data_range: float = 1.0, eps: float = 1e-12) -> float:
    mse = float(max(mse, eps))
    return float(10.0 * math.log10((data_range * data_range) / mse))


def safe_arctan2(y: float, x: float) -> float:
    return float(math.atan2(float(y), float(x)))


def angle_from_sincos(sin_t: float, cos_t: float) -> float:
    return float(math.atan2(float(sin_t), float(cos_t)))


def tensor_to_uint8_img(x: torch.Tensor) -> np.ndarray:
    # (3,H,W) or (B,3,H,W) in [0,1] -> uint8
    if x.dim() == 3:
        y = x.detach().cpu().clamp(0.0, 1.0)
        return (y.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if x.dim() == 4:
        y = x.detach().cpu().clamp(0.0, 1.0)
        return (y.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def _gaussian_window(window_size: int, sigma: float, channel: int, dev, dtype):
    coords = torch.arange(window_size, device=dev, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / torch.sum(g)
    w = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    w = w.repeat(channel, 1, 1, 1)
    return w


def ssim_torch(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    dev = pred.device
    dtype = pred.dtype
    C = pred.shape[1]
    w = _gaussian_window(window_size, sigma, C, dev, dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, w, padding=pad, groups=C)
    mu2 = F.conv2d(gt, w, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, w, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, w, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * gt, w, padding=pad, groups=C) - mu12

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2.0 * mu12 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return torch.mean(ssim_map)


def masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    err = (pred - gt) ** 2
    mask = mask.to(dtype=err.dtype, device=err.device)
    denom = mask.sum() * err.shape[1] + eps
    return (err * mask).sum() / denom


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    err = torch.abs(pred - gt)
    mask = mask.to(dtype=err.dtype, device=err.device)
    denom = mask.sum() * err.shape[1] + eps
    return (err * mask).sum() / denom


def bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return float(inter / union)


def ema_meta_split(
    prev: np.ndarray | None,
    cur: np.ndarray,
    alpha_phys: float = 0.55,
    alpha_bbox: float = 0.78,
    alpha_range: float | None = None,
    alpha_vel: float | None = None,
    alpha_angle: float | None = None,
) -> np.ndarray:
    # display-only EMA so the shown numbers don't jitter frame to frame
    cur = np.asarray(cur, dtype=np.float32).copy()
    if prev is None:
        out = cur
    else:
        prev = np.asarray(prev, dtype=np.float32)
        out = cur.copy()
        a_r = float(alpha_phys if alpha_range is None else alpha_range)
        a_v = float(alpha_phys if alpha_vel is None else alpha_vel)
        a_a = float(alpha_phys if alpha_angle is None else alpha_angle)
        a_b = float(alpha_bbox)
        out[0] = a_r * prev[0] + (1.0 - a_r) * cur[0]
        out[1] = a_v * prev[1] + (1.0 - a_v) * cur[1]
        out[2] = a_a * prev[2] + (1.0 - a_a) * cur[2]
        out[3] = a_a * prev[3] + (1.0 - a_a) * cur[3]
        out[4:] = a_b * prev[4:] + (1.0 - a_b) * cur[4:]
    if out.shape[0] >= 4:
        s, c = float(out[2]), float(out[3])
        r = math.sqrt(s * s + c * c) + 1e-12
        out[2] = s / r
        out[3] = c / r
    return np.clip(out, -1.0, 1.0)


# ================================================================
# Bbox / ROI geometry
# ================================================================
def bbox_xyxy_to_cxcywh(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    return (cx, cy, w, h)


def bbox_cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return (x1, y1, x2, y2)


def clip_bbox_xyxy(bbox_xyxy: Tuple[float, float, float, float], img_wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
    W, H = img_wh
    x1, y1, x2, y2 = bbox_xyxy
    x1i = int(max(0, min(W - 1, round(x1))))
    y1i = int(max(0, min(H - 1, round(y1))))
    x2i = int(max(0, min(W, round(x2))))
    y2i = int(max(0, min(H, round(y2))))
    if x2i <= x1i:
        x2i = min(W, x1i + 1)
    if y2i <= y1i:
        y2i = min(H, y1i + 1)
    return x1i, y1i, x2i, y2i


def crop_bbox_xyxy(img: Image.Image, bbox_xyxy: Tuple[float, float, float, float]) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = clip_bbox_xyxy(bbox_xyxy, (W, H))
    return img.crop((x1, y1, x2, y2))


def fit_to_canvas_keep_ar(
    roi: Image.Image,
    canvas_hw: Tuple[int, int],
    pad_value: int = 0,
    max_scale: float = 1.0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    # paste the ROI into a fixed canvas, keep aspect ratio, remember the placement
    Hc, Wc = int(canvas_hw[0]), int(canvas_hw[1])
    w, h = roi.size
    if w <= 0 or h <= 0:
        canvas = Image.new("RGB", (Wc, Hc), (pad_value, pad_value, pad_value))
        return canvas, {"scale": 0.0, "new_w": 0, "new_h": 0, "pad_left": 0, "pad_top": 0, "canvas_hw": (Hc, Wc), "max_scale": float(max_scale)}

    if max_scale is None or float(max_scale) <= 0.0:
        max_scale = max(float(Wc) / max(float(w), 1.0), float(Hc) / max(float(h), 1.0))
    scale = min(Wc / w, Hc / h, float(max_scale))
    new_w = max(1, min(Wc, int(round(w * scale))))
    new_h = max(1, min(Hc, int(round(h * scale))))

    roi_rs = roi.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (Wc, Hc), (pad_value, pad_value, pad_value))
    pad_left = (Wc - new_w) // 2
    pad_top = (Hc - new_h) // 2
    canvas.paste(roi_rs, (pad_left, pad_top))

    info = {
        "scale": float(scale),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "canvas_hw": (Hc, Wc),
        "roi_wh": (int(w), int(h)),
        "max_scale": float(max_scale),
    }
    return canvas, info


def crop_content_from_canvas(
    canvas_img: np.ndarray,
    roi_wh: Tuple[int, int],
    canvas_hw: Tuple[int, int],
    fit_info: Dict[str, Any] | None = None,
) -> np.ndarray:
    Hc, Wc = int(canvas_hw[0]), int(canvas_hw[1])

    if fit_info is not None:
        pad_left = int(fit_info["pad_left"])
        pad_top = int(fit_info["pad_top"])
        new_w = int(fit_info["new_w"])
        new_h = int(fit_info["new_h"])
        pad_left = max(0, min(Wc - 1, pad_left))
        pad_top = max(0, min(Hc - 1, pad_top))
        new_w = max(1, min(Wc - pad_left, new_w))
        new_h = max(1, min(Hc - pad_top, new_h))
        return canvas_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :]

    w, h = int(roi_wh[0]), int(roi_wh[1])
    if w <= 0 or h <= 0:
        return canvas_img.copy()

    scale = min(Wc / w, Hc / h, 1.0)
    new_w = max(1, min(Wc, int(round(w * scale))))
    new_h = max(1, min(Hc, int(round(h * scale))))
    pad_left = (Wc - new_w) // 2
    pad_top = (Hc - new_h) // 2
    return canvas_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :]


# ================================================================
# Meta (de)normalization
# meta = [rn, vn, sin(theta), cos(theta), cx, cy, w, h] in [-1,1]
# ================================================================
def normalize_meta(
    range_m: float,
    v_rad_mps: float,
    theta_rad: float,
    bbox_xyxy: Tuple[float, float, float, float],
    img_wh: Tuple[int, int],
    range_max: float = 80.0,
    vel_max: float = 20.0,
) -> np.ndarray:
    W, H = float(img_wh[0]), float(img_wh[1])
    cx, cy, w, h = bbox_xyxy_to_cxcywh(bbox_xyxy)

    cx01 = float(np.clip(cx / max(W, 1.0), 0.0, 1.0))
    cy01 = float(np.clip(cy / max(H, 1.0), 0.0, 1.0))
    w01 = float(np.clip(w / max(W, 1.0), 0.0, 1.0))
    h01 = float(np.clip(h / max(H, 1.0), 0.0, 1.0))

    rn = 2.0 * (float(np.clip(range_m, 0.0, range_max)) / max(range_max, 1e-6)) - 1.0
    vn = float(np.clip(v_rad_mps, -vel_max, vel_max)) / max(vel_max, 1e-6)
    sin_t = float(math.sin(theta_rad))
    cos_t = float(math.cos(theta_rad))
    cxn = 2.0 * cx01 - 1.0
    cyn = 2.0 * cy01 - 1.0
    wn = 2.0 * w01 - 1.0
    hn = 2.0 * h01 - 1.0
    return np.array([rn, vn, sin_t, cos_t, cxn, cyn, wn, hn], dtype=np.float32)


def denormalize_meta(
    meta_norm: np.ndarray,
    img_wh: Tuple[int, int],
    range_max: float = 80.0,
    vel_max: float = 20.0,
) -> Dict[str, Any]:
    W, H = float(img_wh[0]), float(img_wh[1])
    meta_norm = np.asarray(meta_norm, dtype=np.float32).reshape(-1)
    assert meta_norm.shape[0] == 8

    rn, vn, sin_t, cos_t, cxn, cyn, wn, hn = [float(x) for x in meta_norm.tolist()]

    range_m = 0.5 * (rn + 1.0) * range_max
    v_rad = float(np.clip(vn, -1.0, 1.0)) * vel_max
    theta = angle_from_sincos(sin_t, cos_t)

    cx01 = 0.5 * (cxn + 1.0)
    cy01 = 0.5 * (cyn + 1.0)
    w01 = 0.5 * (wn + 1.0)
    h01 = 0.5 * (hn + 1.0)

    cx = float(np.clip(cx01, 0.0, 1.0)) * W
    cy = float(np.clip(cy01, 0.0, 1.0)) * H
    bw = float(np.clip(w01, 0.0, 1.0)) * W
    bh = float(np.clip(h01, 0.0, 1.0)) * H
    bbox_xyxy = bbox_cxcywh_to_xyxy(cx, cy, bw, bh)

    return {
        "range_m": float(range_m),
        "v_rad_mps": float(v_rad),
        "theta_rad": float(theta),
        "theta_deg": float(theta * 180.0 / math.pi),
        "bbox_cxcywh": (float(cx), float(cy), float(bw), float(bh)),
        "bbox_xyxy": bbox_xyxy,
    }
