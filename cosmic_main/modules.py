# -*- coding: utf-8 -*-
# CoSMIC core: waveform, covert projection, channel, receiver, networks, radar.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


@dataclass
class CovertISACConfig:
    canvas_hw: Tuple[int, int] = (192, 320)
    latent_down: int = 4
    pair_ch: int = 2

    # one frame = num_chirps chirps, spc samples per chirp
    num_chirps: int = 64
    spc: int = 240
    chirp_mu: float = 40.0

    # phase map gain and AR shaping coefficient
    alpha: float = 0.26
    ar_rho: float = 0.0

    rx_use_map: bool = True
    rx_map_iters: int = 2
    rx_map_u_clip: float = 6.0
    rx_sigma_phi_min: float = 1e-4
    rx_sigma_phi_max: float = 4.0

    # covert budget: sum chi(tau) <= 2*eps^2
    covert_eps: float = 0.10
    covert_delta_override: float = -1.0
    tau_proj_iters: int = 28
    tau_floor: float = 1e-8

    enc_base: int = 80
    enc_blocks: int = 6
    dec_base: int = 224
    dec_blocks_lat: int = 6
    dec_blocks_img: int = 4
    dec_res_scale: float = 0.30
    tau_hidden: int = 128
    dec_use_rel: bool = True
    dec_use_slot_mask: bool = True
    dec_rel_scale_base: float = 1.00
    dec_rel_scale_rf: float = 0.15

    # Bob-side classification head
    cls_enable: bool = True
    num_classes: int = 13
    cls_hidden: int = 256
    cls_dropout: float = 0.20
    cls_blocks: int = 2
    cls_use_meta: bool = True
    cls_detach_feat: bool = False
    cls_input_dropout: float = 0.10
    cls_feat_noise: float = 0.05
    cls_slot_enable: bool = False
    cls_slot_repeats: int = 96
    cls_slot_theta_scale: float = 1.50
    cls_slot_temp: float = 8.0

    grad_checkpoint: bool = True

    meta_dim: int = 8
    meta_slot_repeats: Tuple[int, ...] = (192, 224, 320, 320, 96, 96, 48, 48)
    meta_theta_max: float = 1.70
    meta_theta_scales: Tuple[float, ...] = (1.55, 1.85, 1.65, 1.65, 1.25, 1.25, 1.10, 1.10)
    meta_refine_scale: float = 0.018
    vel_compand_beta: float = 1.60
    theta_compand_beta: float = 1.80
    slot_refine_hidden: int = 192
    slot_refine_scale: float = 0.10
    meta_balanced_slots: bool = True

    rel_use_logden: bool = True
    rel_use_mag: bool = True
    rel_use_logsnr: bool = True
    rel_use_phi_resid: bool = True
    rel_use_logtau: bool = True

    # RFlow refiner
    rf_enable: bool = True
    rf_steps: int = 2
    rf_ch: int = 96
    rf_blocks: int = 4
    rf_td: int = 96
    rf_endpoint_weight: float = 0.12
    rf_gate_floor: float = 0.10
    rf_train_steps: int = 5
    rf_path_gamma: float = 1.00
    rf_step_scale: float = 1.00
    rf_step_loss_weight: float = 1.00

    # radar (ULA matched filter)
    radar_rx_ant: int = 8
    radar_angle_max_deg: float = 60.0
    radar_angle_grid: int = 121
    radar_range_os: int = 4
    radar_dop_os: int = 4
    radar_delay_min: int = 8
    radar_delay_max: int = 84
    radar_dop_min: int = -16
    radar_dop_max: int = 16
    radar_nms_delay: int = 3
    radar_nms_dop: int = 2
    radar_det_tol_delay: int = 1
    radar_det_tol_dop: int = 1
    radar_det_tol_angle: float = 3.0
    radar_amp_min: float = 0.55
    radar_amp_max: float = 1.0

    seed: int = 520

    @property
    def latent_h(self) -> int:
        return int(self.canvas_hw[0] // self.latent_down)

    @property
    def latent_w(self) -> int:
        return int(self.canvas_hw[1] // self.latent_down)

    @property
    def k_pairs(self) -> int:
        return int(self.pair_ch * self.latent_h * self.latent_w)

    @property
    def D(self) -> int:
        return int(self.num_chirps * self.spc)

    @property
    def covert_delta(self) -> float:
        if self.covert_delta_override is not None and self.covert_delta_override > 0.0:
            return float(self.covert_delta_override)
        return 2.0 * float(self.covert_eps) * float(self.covert_eps)

    @property
    def sem_ch(self) -> int:
        return 2 * int(self.pair_ch)

    @property
    def rel_ch(self) -> int:
        n = 0
        if self.rel_use_phi_resid:
            n += self.pair_ch
        if self.rel_use_logden:
            n += self.pair_ch
        if self.rel_use_mag:
            n += self.pair_ch
        if self.rel_use_logtau:
            n += self.pair_ch
        if self.rel_use_logsnr:
            n += self.pair_ch
        return int(n)

    @property
    def dec_in_ch(self) -> int:
        ch = self.sem_ch
        if self.dec_use_rel:
            ch += self.rel_ch
        if self.dec_use_slot_mask:
            ch += 1
        return int(ch)

    @property
    def meta_slot_total(self) -> int:
        return int(sum(int(x) for x in self.meta_slot_repeats))

    def meta_scales(self) -> Tuple[float, ...]:
        vals = tuple(float(x) for x in self.meta_theta_scales)
        if len(vals) != self.meta_dim:
            vals = tuple(float(self.meta_theta_max) for _ in range(self.meta_dim))
        return vals

    def check(self) -> None:
        assert self.canvas_hw[0] % self.latent_down == 0 and self.canvas_hw[1] % self.latent_down == 0
        assert self.spc % 2 == 0
        assert self.D == 2 * self.k_pairs, (
            f"Need D=2*k_pairs. Got D={self.D}, k_pairs={self.k_pairs}."
        )
        assert len(tuple(self.meta_slot_repeats)) == self.meta_dim
        assert self.meta_slot_total <= self.k_pairs
        if self.cls_slot_enable:
            assert int(self.cls_slot_repeats) > 0
            assert int(self.num_classes) >= 2
            assert self.meta_slot_total + int(self.cls_slot_repeats) <= self.k_pairs
        assert int(self.rf_train_steps) >= int(self.rf_steps)
        assert self.radar_delay_max < self.spc
        assert self.radar_angle_grid >= 31
        assert int(self.radar_range_os) >= 1
        assert int(self.radar_dop_os) >= 1
        assert self.radar_rx_ant >= 2


# ================================================================
# Basic blocks
# ================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(ch))
        self.b = nn.Parameter(torch.zeros(ch))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.w[:, None, None] * x + self.b[:, None, None]


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, ch: int, expansion: int = 4, kernel: int = 7) -> None:
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(ch, ch, kernel, 1, pad, groups=ch)
        self.norm = LayerNorm2d(ch)
        self.pw1 = nn.Conv2d(ch, expansion * ch, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(expansion * ch, ch, 1)
        self.gamma = nn.Parameter(1e-4 * torch.ones(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw2(self.act(self.pw1(h)))
        return x + self.gamma[:, None, None] * h


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        ang = t[:, None].float() * freq[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class FlowBlock(nn.Module):
    def __init__(self, ci: int, co: int, td: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ci, co, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, co), co),
            nn.SiLU(),
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(td, co))
        self.conv2 = nn.Sequential(
            nn.Conv2d(co, co, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, co), co),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x) + self.time(te)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


# ================================================================
# Waveform helpers
# ================================================================
def chirp_torch(N: int, mu: float, dev: torch.device) -> torch.Tensor:
    t = torch.arange(N, device=dev, dtype=torch.float32) / float(N)
    return torch.exp(1j * math.pi * mu * t * t).to(torch.complex64)


def sample_b0(B: int, cfg: CovertISACConfig, dev: torch.device) -> torch.Tensor:
    # Gaussian reference pairs shared between Alice and Bob
    return torch.randn(B, cfg.num_chirps, cfg.spc // 2, 2, device=dev, dtype=torch.float32)


def maps_to_ms(x: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    B = x.shape[0]
    return x.reshape(B, -1).reshape(B, cfg.num_chirps, cfg.spc // 2)


def ms_to_maps(x: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    B = x.shape[0]
    return x.reshape(B, cfg.pair_ch, cfg.latent_h, cfg.latent_w)


def wrap_phase(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def normalize_cs(c_raw: torch.Tensor, s_raw: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    r2 = c_raw * c_raw + s_raw * s_raw
    inv_r = torch.rsqrt(torch.clamp(r2, min=eps))
    c = c_raw * inv_r
    s = s_raw * inv_r
    mask = r2 < eps
    c = torch.where(mask, torch.ones_like(c), c)
    s = torch.where(mask, torch.zeros_like(s), s)
    return c, s


def theta_to_cs(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cos(theta), torch.sin(theta)


def theta_to_u(theta: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    c, s = theta_to_cs(theta)
    return torch.cat([c, s], dim=1)


def cs_maps_to_theta(u_sem: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    c = u_sem[:, :cfg.pair_ch]
    s = u_sem[:, cfg.pair_ch: 2 * cfg.pair_ch]
    return torch.atan2(s, c)


def phase_delta_u(u_cur: torch.Tensor, u_base: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    theta_cur = cs_maps_to_theta(u_cur, cfg)
    theta_base = cs_maps_to_theta(u_base, cfg)
    return theta_to_u(wrap_phase(theta_cur - theta_base), cfg)


# ================================================================
# Covert projection: chi(t) = t - ln(t) - 1, keep sum chi(tau) <= 2*eps^2
# ================================================================
def kl_g_tau(tau: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    tau = torch.clamp(tau, min=eps)
    return tau - torch.log(tau) - 1.0


def project_tau_budget_flat(
    tau0: torch.Tensor,
    delta: float,
    iters: int = 28,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # bisection on the contraction lambda: tau = 1 + lambda*(tau0 - 1)
    B = tau0.shape[0]
    g0 = kl_g_tau(tau0, eps=eps).sum(dim=1)
    if float(torch.max(g0).item()) <= float(delta) + 1e-12:
        return tau0, g0

    lo = torch.zeros(B, 1, device=tau0.device, dtype=tau0.dtype)
    hi = torch.ones(B, 1, device=tau0.device, dtype=tau0.dtype)
    target = torch.full((B, 1), float(delta), device=tau0.device, dtype=tau0.dtype)

    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        tau = torch.clamp(1.0 + mid * (tau0 - 1.0), min=eps)
        g = kl_g_tau(tau, eps=eps).sum(dim=1, keepdim=True)
        hi = torch.where(g > target, mid, hi)
        lo = torch.where(g <= target, mid, lo)

    lam = lo
    tau = torch.clamp(1.0 + lam * (tau0 - 1.0), min=eps)
    g = kl_g_tau(tau, eps=eps).sum(dim=1)
    return tau, g


def project_tau_budget_map(tau_raw_map: torch.Tensor, cfg: CovertISACConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    B = tau_raw_map.shape[0]
    tau_flat = tau_raw_map.reshape(B, -1)
    tau_proj, kl = project_tau_budget_flat(
        tau_flat,
        delta=cfg.covert_delta,
        iters=cfg.tau_proj_iters,
        eps=cfg.tau_floor,
    )
    return tau_proj.reshape_as(tau_raw_map), kl


# ================================================================
# Meta slot layout and coding
# ================================================================
_META_SLOT_CACHE: Dict[Tuple[Any, ...], Tuple[torch.Tensor, ...]] = {}


def meta_slot_repeats_list(cfg: CovertISACConfig) -> List[int]:
    reps = [int(x) for x in tuple(cfg.meta_slot_repeats)]
    if len(reps) != int(cfg.meta_dim):
        raise ValueError(f"meta_slot_repeats length mismatch: {len(reps)} vs meta_dim={cfg.meta_dim}")
    if min(reps) <= 0:
        raise ValueError("All meta_slot_repeats entries must be positive")
    return reps


_SLOT_ORDER_CACHE: Dict[Tuple[Any, ...], np.ndarray] = {}


def _slot_order(cfg: CovertISACConfig) -> np.ndarray:
    key = (
        cfg.num_chirps,
        cfg.spc,
        cfg.pair_ch,
        cfg.latent_h,
        cfg.latent_w,
        int(cfg.seed),
        bool(cfg.meta_balanced_slots),
    )
    if key in _SLOT_ORDER_CACHE:
        return _SLOT_ORDER_CACHE[key]

    K = cfg.k_pairs
    rng = np.random.default_rng(int(cfg.seed) + 977)
    if cfg.meta_balanced_slots:
        # spread the slots over all chirps so no single chirp carries too much
        grid = np.arange(K, dtype=np.int64).reshape(cfg.num_chirps, cfg.spc // 2)
        col_perm = rng.permutation(cfg.spc // 2)
        order_cols = []
        for col in col_perm:
            order_cols.append(np.roll(grid[:, col], int(rng.integers(0, cfg.num_chirps))))
        order = np.concatenate(order_cols, axis=0)
    else:
        order = rng.permutation(K)
    _SLOT_ORDER_CACHE[key] = order
    return order


def build_meta_slot_indices(cfg: CovertISACConfig) -> Tuple[torch.Tensor, ...]:
    key = (
        cfg.num_chirps,
        cfg.spc,
        cfg.pair_ch,
        cfg.latent_h,
        cfg.latent_w,
        tuple(int(x) for x in cfg.meta_slot_repeats),
        int(cfg.seed),
        bool(cfg.meta_balanced_slots),
    )
    if key in _META_SLOT_CACHE:
        return _META_SLOT_CACHE[key]

    reps = meta_slot_repeats_list(cfg)
    if sum(reps) > cfg.k_pairs:
        raise ValueError("Not enough latent pairs for meta slot allocation")

    order = _slot_order(cfg)
    out: List[torch.Tensor] = []
    start = 0
    for r in reps:
        out.append(torch.from_numpy(order[start:start + r].astype(np.int64)))
        start += r
    _META_SLOT_CACHE[key] = tuple(out)
    return tuple(out)


_CLS_SLOT_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def build_cls_slot_indices(cfg: CovertISACConfig) -> torch.Tensor:
    key = (
        cfg.num_chirps,
        cfg.spc,
        cfg.pair_ch,
        cfg.latent_h,
        cfg.latent_w,
        tuple(int(x) for x in cfg.meta_slot_repeats),
        int(cfg.cls_slot_repeats),
        int(cfg.seed),
        bool(cfg.meta_balanced_slots),
    )
    if key in _CLS_SLOT_CACHE:
        return _CLS_SLOT_CACHE[key]

    start = int(sum(int(x) for x in cfg.meta_slot_repeats))
    r = int(cfg.cls_slot_repeats)
    if start + r > cfg.k_pairs:
        raise ValueError("Not enough latent pairs for class slot allocation")

    order = _slot_order(cfg)
    out = torch.from_numpy(order[start:start + r].astype(np.int64))
    _CLS_SLOT_CACHE[key] = out
    return out


def meta_slot_mask_map(cfg: CovertISACConfig, device: torch.device) -> torch.Tensor:
    idx_list = build_meta_slot_indices(cfg)
    mask = torch.zeros(cfg.k_pairs, device=device, dtype=torch.bool)
    for idx in idx_list:
        mask[idx.to(device)] = True
    if bool(getattr(cfg, "cls_slot_enable", False)):
        mask[build_cls_slot_indices(cfg).to(device)] = True
    return mask.reshape(1, cfg.pair_ch, cfg.latent_h, cfg.latent_w)


def slot_mask_union_map(cfg: CovertISACConfig, device: torch.device) -> torch.Tensor:
    mask = meta_slot_mask_map(cfg, device)
    return mask.any(dim=1, keepdim=True)


def _theta_phys_max_rad(cfg: CovertISACConfig) -> float:
    return float(cfg.radar_angle_max_deg) * math.pi / 180.0


def _signed_compand(x: torch.Tensor, beta: float) -> torch.Tensor:
    beta = float(beta)
    if beta <= 1e-6:
        return torch.clamp(x, -1.0, 1.0)
    s = math.tanh(beta)
    return torch.clamp(torch.tanh(beta * torch.clamp(x, -1.0, 1.0)) / max(s, 1e-6), -1.0, 1.0)


def _signed_decompand(u: torch.Tensor, beta: float) -> torch.Tensor:
    beta = float(beta)
    if beta <= 1e-6:
        return torch.clamp(u, -1.0, 1.0)
    s = math.tanh(beta)
    z = torch.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6) * s
    return torch.clamp(torch.atanh(torch.clamp(z, -0.999999, 0.999999)) / beta, -1.0, 1.0)


def public_meta_to_internal_slots(meta: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    # public meta = [rn, vn, sin, cos, cx, cy, w, h] in [-1,1]
    rn = torch.clamp(meta[:, 0], -1.0, 1.0)
    vn = _signed_compand(meta[:, 1], float(cfg.vel_compand_beta))
    theta = torch.atan2(meta[:, 2], meta[:, 3])
    theta_n = torch.clamp(theta / max(_theta_phys_max_rad(cfg), 1e-6), -1.0, 1.0)
    theta_c = _signed_compand(theta_n, float(cfg.theta_compand_beta))
    cx = torch.clamp(meta[:, 4], -1.0, 1.0)
    cy = torch.clamp(meta[:, 5], -1.0, 1.0)
    w = torch.clamp(meta[:, 6], -1.0, 1.0)
    h = torch.clamp(meta[:, 7], -1.0, 1.0)
    return torch.stack([rn, vn, theta_c, theta_c, cx, cy, w, h], dim=1)


def internal_slots_to_public(internal: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    rn = torch.clamp(internal[:, 0], -1.0, 1.0)
    vn = _signed_decompand(internal[:, 1], float(cfg.vel_compand_beta))
    theta_c = torch.clamp(0.5 * (internal[:, 2] + internal[:, 3]), -1.0, 1.0)
    theta_n = _signed_decompand(theta_c, float(cfg.theta_compand_beta))
    theta = theta_n * _theta_phys_max_rad(cfg)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    cx = torch.clamp(internal[:, 4], -1.0, 1.0)
    cy = torch.clamp(internal[:, 5], -1.0, 1.0)
    w = torch.clamp(internal[:, 6], -1.0, 1.0)
    h = torch.clamp(internal[:, 7], -1.0, 1.0)
    return torch.stack([rn, vn, sin_t, cos_t, cx, cy, w, h], dim=1)


def inject_internal_slots(c_map: torch.Tensor, s_map: torch.Tensor, internal: torch.Tensor, cfg: CovertISACConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    B = c_map.shape[0]
    idx_list = [idx.to(c_map.device) for idx in build_meta_slot_indices(cfg)]
    scales = list(cfg.meta_scales())
    flat_c = c_map.reshape(B, -1).clone()
    flat_s = s_map.reshape(B, -1).clone()
    for d, idx in enumerate(idx_list):
        r = idx.numel()
        ang = float(scales[d]) * torch.clamp(internal[:, d:d + 1], -1.0, 1.0)
        code_c = torch.cos(ang).expand(B, r).to(c_map.dtype)
        code_s = torch.sin(ang).expand(B, r).to(s_map.dtype)
        flat_c.scatter_(1, idx.reshape(1, -1).expand(B, -1), code_c)
        flat_s.scatter_(1, idx.reshape(1, -1).expand(B, -1), code_s)
    return flat_c.reshape_as(c_map), flat_s.reshape_as(s_map)


def inject_meta_slots(c_map: torch.Tensor, s_map: torch.Tensor, meta: torch.Tensor, cfg: CovertISACConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    return inject_internal_slots(c_map, s_map, public_meta_to_internal_slots(meta, cfg), cfg)


def force_tau_neutral_on_slots(tau_map: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    # keep tau = 1 on meta slots so they never spend covert budget
    idx_list = [idx.to(tau_map.device) for idx in build_meta_slot_indices(cfg)]
    if bool(getattr(cfg, "cls_slot_enable", False)):
        idx_list.append(build_cls_slot_indices(cfg).to(tau_map.device))
    B = tau_map.shape[0]
    flat = tau_map.reshape(B, -1).clone()
    for idx in idx_list:
        flat.scatter_(1, idx.reshape(1, -1).expand(B, -1), torch.ones(B, idx.numel(), device=tau_map.device, dtype=tau_map.dtype))
    return flat.reshape_as(tau_map)


def _circular_resultant(angle: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    re = torch.sum(weight * torch.cos(angle), dim=-1)
    im = torch.sum(weight * torch.sin(angle), dim=-1)
    mu = torch.atan2(im, re)
    R = torch.sqrt(re * re + im * im + 1e-8)
    return mu, torch.clamp(R, max=1.0)


def _robust_circular_decode(angle: torch.Tensor, weight_raw: torch.Tensor, ang_scale: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # weighted circular mean with a couple of reweighting rounds
    eps = 1e-8
    w = torch.clamp(weight_raw, min=eps)
    w = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=eps)
    mu, R = _circular_resultant(angle, w)
    kappa = max(0.20 * float(ang_scale), 0.10)
    for _ in range(2):
        err = wrap_phase(angle - mu.unsqueeze(-1))
        robust = 1.0 / torch.sqrt(1.0 + (err / kappa) ** 2)
        w = torch.clamp(weight_raw * robust, min=eps)
        w = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=eps)
        mu, R = _circular_resultant(angle, w)

    n = angle.shape[-1]
    k = max(6, int(round(0.60 * n)))
    k = min(k, n)
    topv, topi = torch.topk(weight_raw, k=k, dim=-1)
    ang_top = torch.gather(angle, 1, topi)
    w_top = topv / torch.clamp(topv.sum(dim=-1, keepdim=True), min=eps)
    mu_top, R_top = _circular_resultant(ang_top, w_top)

    err_fin = wrap_phase(angle - mu.unsqueeze(-1))
    circ_std = torch.sqrt(torch.sum(w * err_fin * err_fin, dim=-1) + eps)
    stats = {
        "R": R,
        "R_top": R_top,
        "mu_top": mu_top,
        "circ_std": circ_std,
        "logw_mean": torch.log(torch.clamp(weight_raw.mean(dim=-1), min=eps)),
        "logw_max": torch.log(torch.clamp(weight_raw.max(dim=-1).values, min=eps)),
        "dmu": wrap_phase(mu_top - mu),
    }
    return mu, stats


def cls_slot_levels(cfg: CovertISACConfig, device: torch.device) -> torch.Tensor:
    G = int(cfg.num_classes)
    g = torch.arange(G, device=device, dtype=torch.float32)
    gn = 2.0 * g / max(float(G - 1), 1.0) - 1.0
    return float(cfg.cls_slot_theta_scale) * gn


def inject_cls_slot_angle(c_map: torch.Tensor, s_map: torch.Tensor, ang: torch.Tensor, cfg: CovertISACConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    B = c_map.shape[0]
    idx = build_cls_slot_indices(cfg).to(c_map.device)
    ang = ang.reshape(B, 1).to(c_map.dtype)
    flat_c = c_map.reshape(B, -1).clone()
    flat_s = s_map.reshape(B, -1).clone()
    code_c = torch.cos(ang).expand(B, idx.numel())
    code_s = torch.sin(ang).expand(B, idx.numel())
    flat_c.scatter_(1, idx.reshape(1, -1).expand(B, -1), code_c)
    flat_s.scatter_(1, idx.reshape(1, -1).expand(B, -1), code_s)
    return flat_c.reshape_as(c_map), flat_s.reshape_as(s_map)


def inject_cls_slots(c_map: torch.Tensor, s_map: torch.Tensor, cls_idx: Optional[torch.Tensor], cfg: CovertISACConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    B = c_map.shape[0]
    if cls_idx is None:
        ang = torch.zeros(B, device=c_map.device, dtype=c_map.dtype)
    else:
        levels = cls_slot_levels(cfg, c_map.device)
        ang = levels[cls_idx.reshape(-1).long()]
    return inject_cls_slot_angle(c_map, s_map, ang, cfg)


def decode_cls_slots(
    u_sem: torch.Tensor,
    tau_hat_map: Optional[torch.Tensor],
    slot_conf_map: Optional[torch.Tensor],
    den_map: Optional[torch.Tensor],
    cfg: CovertISACConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-8
    B = u_sem.shape[0]
    idx = build_cls_slot_indices(cfg).to(u_sem.device)
    flat_c = u_sem[:, :cfg.pair_ch].reshape(B, -1)
    flat_s = u_sem[:, cfg.pair_ch: 2 * cfg.pair_ch].reshape(B, -1)
    ang = torch.atan2(flat_s[:, idx], flat_c[:, idx])
    if slot_conf_map is not None:
        w = torch.clamp(slot_conf_map.reshape(B, -1)[:, idx], min=eps)
    elif den_map is not None:
        w = torch.clamp(den_map.reshape(B, -1)[:, idx], min=eps)
    else:
        w = torch.ones_like(ang)
    if tau_hat_map is not None:
        w = w * torch.exp(-0.35 * torch.abs(torch.log(torch.clamp(tau_hat_map.reshape(B, -1)[:, idx], min=eps))))
    mu, stats = _robust_circular_decode(ang, w, ang_scale=float(cfg.cls_slot_theta_scale))
    levels = cls_slot_levels(cfg, u_sem.device)
    d = wrap_phase(mu.unsqueeze(1) - levels.unsqueeze(0))
    logits = -float(cfg.cls_slot_temp) * d * d
    feat = torch.stack([
        torch.clamp(mu / max(float(cfg.cls_slot_theta_scale), 1e-6), -1.0, 1.0),
        stats["R"],
        stats["R_top"],
        stats["circ_std"],
    ], dim=1)
    return mu, logits, feat


# ================================================================
# Networks
# ================================================================
class PhaseEncoderLite(nn.Module):
    # ConvNeXt encoder, outputs the phase field for the direction head
    def __init__(self, cfg: CovertISACConfig) -> None:
        super().__init__()
        n_down = int(round(math.log2(cfg.latent_down)))
        assert (2 ** n_down) == cfg.latent_down
        layers: List[nn.Module] = []
        in_ch = 3
        ch = cfg.enc_base
        for _ in range(n_down):
            layers += [
                nn.Conv2d(in_ch, ch, 5, 2, 2, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.GELU(),
            ]
            in_ch = ch
            ch = min(ch * 2, 256)
        self.stem = nn.Sequential(*layers)
        self.blocks = nn.Sequential(*[ConvNeXtLiteBlock(in_ch) for _ in range(cfg.enc_blocks)])
        self.theta_head = nn.Conv2d(in_ch, cfg.pair_ch, 3, 1, 1)
        nn.init.normal_(self.theta_head.weight, mean=0.0, std=0.01)
        if self.theta_head.bias is not None:
            nn.init.zeros_(self.theta_head.bias)
        self.feat_ch = in_ch
        self.use_ckpt = bool(getattr(cfg, "grad_checkpoint", False))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        if self.use_ckpt and self.training and feat.requires_grad and len(self.blocks) > 0:
            feat = checkpoint_sequential(self.blocks, len(self.blocks), feat, use_reentrant=False)
        else:
            feat = self.blocks(feat)
        theta_raw = self.theta_head(feat)
        return theta_raw, feat


class TauNetLite(nn.Module):
    # covert scale head, predicts raw tau per pair
    def __init__(self, feat_ch: int, cfg: CovertISACConfig) -> None:
        super().__init__()
        hidden = max(64, int(cfg.tau_hidden))
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
            ConvNeXtLiteBlock(hidden),
            ConvNeXtLiteBlock(hidden),
            nn.Conv2d(hidden, cfg.pair_ch, 3, 1, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        eta = self.net(feat)
        tau_raw = torch.exp(1.10 * torch.tanh(eta))
        return tau_raw


class SemanticDecoderLite(nn.Module):
    # latent ConvNeXt blocks -> coarse image, then upsample + residual blocks
    def __init__(self, in_ch: int, cfg: CovertISACConfig) -> None:
        super().__init__()
        base = int(cfg.dec_base)
        self.out_h = int(cfg.canvas_hw[0])
        self.out_w = int(cfg.canvas_hw[1])
        self.res_scale = float(cfg.dec_res_scale)
        self.inp = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, base), base),
            nn.GELU(),
        )
        self.lat_blocks = nn.Sequential(*[ConvNeXtLiteBlock(base) for _ in range(cfg.dec_blocks_lat)])
        self.base_head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, base // 2), base // 2),
            nn.GELU(),
            nn.Conv2d(base // 2, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        up_layers: List[nn.Module] = []
        n_up = int(round(math.log2(cfg.latent_down)))
        ch = base
        for _ in range(n_up):
            up_layers += [
                nn.Conv2d(ch, ch * 4, 3, 1, 1, bias=False),
                nn.PixelShuffle(2),
                nn.GroupNorm(min(8, ch), ch),
                nn.GELU(),
            ]
        self.up = nn.Sequential(*up_layers)
        self.img_blocks = nn.Sequential(*[ConvNeXtLiteBlock(base) for _ in range(cfg.dec_blocks_img)])
        self.res_head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, base // 2), base // 2),
            nn.GELU(),
            nn.Conv2d(base // 2, 3, 3, 1, 1),
        )
        self.use_ckpt = bool(getattr(cfg, "grad_checkpoint", False))

    def _run_blocks(self, blocks: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if self.use_ckpt and self.training and x.requires_grad and len(blocks) > 0:
            return checkpoint_sequential(blocks, len(blocks), x, use_reentrant=False)
        return blocks(x)

    def forward(self, u_in: torch.Tensor, return_latent: bool = False):
        h = self._run_blocks(self.lat_blocks, self.inp(u_in))
        base_lat = self.base_head(h)
        base_img = F.interpolate(base_lat, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        feat = self._run_blocks(self.img_blocks, self.up(h))
        res = torch.tanh(self.res_head(feat))
        img = torch.clamp(base_img + self.res_scale * res, 0.0, 1.0)
        if return_latent:
            return img, h
        return img


class ClassifierHead(nn.Module):
    # reads the recovered semantic field, predicts the target class
    def __init__(self, in_ch: int, cfg: CovertISACConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.cls_hidden)
        n_blocks = int(max(1, cfg.cls_blocks))
        self.in_drop = nn.Dropout2d(float(getattr(cfg, "cls_input_dropout", 0.0)))
        self.feat_noise = float(getattr(cfg, "cls_feat_noise", 0.0))
        self.pre = nn.Sequential(*[ConvNeXtLiteBlock(in_ch) for _ in range(n_blocks)])
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 2, 1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            ConvNeXtLiteBlock(hidden),
            nn.Conv2d(hidden, hidden, 3, 2, 1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            ConvNeXtLiteBlock(hidden),
        )
        extra = int(cfg.meta_dim) if bool(cfg.cls_use_meta) else 0
        if bool(getattr(cfg, "cls_slot_enable", False)):
            extra += 4
        self.extra_dim = int(extra)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + extra, hidden),
            nn.SiLU(),
            nn.Dropout(float(cfg.cls_dropout)),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(float(cfg.cls_dropout)),
            nn.Linear(hidden // 2, int(cfg.num_classes)),
        )

    def pooled(self, feat: torch.Tensor, meta_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training and self.feat_noise > 0:
            feat = feat + torch.randn_like(feat) * self.feat_noise
        feat = self.in_drop(feat)
        h = self.down(self.pre(feat))
        avg = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        z = torch.cat([avg, mx], dim=1)
        if self.extra_dim > 0 and meta_extra is not None:
            z = torch.cat([z, meta_extra], dim=1)
        return z

    def classify_pooled(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

    def forward(self, feat: torch.Tensor, meta_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.pooled(feat, meta_extra))


class MetaHeadAnalytic(nn.Module):
    # analytic decode of the protected meta slots (circular mean per slot group)
    def __init__(self, cfg: CovertISACConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scales = list(cfg.meta_scales())

    def analytic_decode_internal(
        self,
        u_sem: torch.Tensor,
        tau_hat_map: Optional[torch.Tensor] = None,
        slot_conf_map: Optional[torch.Tensor] = None,
        den_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-8
        B = u_sem.shape[0]
        c_map = u_sem[:, :self.cfg.pair_ch]
        s_map = u_sem[:, self.cfg.pair_ch: 2 * self.cfg.pair_ch]
        flat_c = c_map.reshape(B, -1)
        flat_s = s_map.reshape(B, -1)
        flat_tau = tau_hat_map.reshape(B, -1) if tau_hat_map is not None else None
        flat_conf = slot_conf_map.reshape(B, -1) if slot_conf_map is not None else None
        flat_den = den_map.reshape(B, -1) if den_map is not None else None

        internal_parts: List[torch.Tensor] = []
        feat_parts: List[torch.Tensor] = []

        for d, idx in enumerate(build_meta_slot_indices(self.cfg)):
            idx = idx.to(u_sem.device)
            ang = torch.atan2(flat_s[:, idx], flat_c[:, idx])
            if flat_conf is not None:
                w = torch.clamp(flat_conf[:, idx], min=eps)
            elif flat_den is not None:
                w = torch.clamp(flat_den[:, idx], min=eps)
            else:
                w = torch.ones_like(ang)
            if flat_tau is not None:
                w = w * torch.exp(-0.35 * torch.abs(torch.log(torch.clamp(flat_tau[:, idx], min=eps))))
            mu, stats = _robust_circular_decode(ang, w, ang_scale=self.scales[d])
            internal_d = torch.clamp(mu / max(self.scales[d], 1e-6), -1.0, 1.0)
            internal_parts.append(internal_d.unsqueeze(1))
            tau_mean = flat_tau[:, idx].mean(dim=-1) if flat_tau is not None else torch.ones(B, device=u_sem.device, dtype=u_sem.dtype)
            feat_parts.extend([
                stats["R"].unsqueeze(1),
                stats["R_top"].unsqueeze(1),
                torch.clamp(stats["mu_top"] / max(self.scales[d], 1e-6), -1.0, 1.0).unsqueeze(1),
                torch.clamp(stats["dmu"] / max(self.scales[d], 1e-6), -1.0, 1.0).unsqueeze(1),
                stats["circ_std"].unsqueeze(1),
                torch.log(torch.clamp(tau_mean, min=eps)).unsqueeze(1),
                stats["logw_mean"].unsqueeze(1),
                stats["logw_max"].unsqueeze(1),
            ])

        internal = torch.cat(internal_parts, dim=1)
        th2 = internal[:, 2] * self.scales[2]
        th3 = internal[:, 3] * self.scales[3]
        re = torch.cos(th2) + torch.cos(th3)
        im = torch.sin(th2) + torch.sin(th3)
        theta_code = torch.atan2(im, re)
        theta_c = torch.clamp(theta_code / max(0.5 * (self.scales[2] + self.scales[3]), 1e-6), -1.0, 1.0)
        internal[:, 2] = theta_c
        internal[:, 3] = theta_c
        feat = torch.cat(feat_parts, dim=1)
        return internal, feat

    def public_from_internal(self, internal: torch.Tensor) -> torch.Tensor:
        return internal_slots_to_public(internal, self.cfg)

    def forward(
        self,
        u_sem: torch.Tensor,
        tau_hat_map: Optional[torch.Tensor] = None,
        slot_conf_map: Optional[torch.Tensor] = None,
        den_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        internal, _ = self.analytic_decode_internal(u_sem, tau_hat_map, slot_conf_map, den_map)
        return self.public_from_internal(internal)


class SlotDomainRefiner(nn.Module):
    # small MLP that nudges the analytic meta estimate
    def __init__(self, cfg: CovertISACConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.slot_refine_hidden)
        in_dim = int(cfg.meta_dim) + 8 * int(cfg.meta_dim)
        self.scale = float(cfg.slot_refine_scale)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(cfg.meta_dim)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, internal_ana: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        z = torch.cat([internal_ana, feat], dim=1)
        delta = self.scale * torch.tanh(self.net(z))
        out = torch.clamp(internal_ana + delta, -1.0, 1.0)
        theta_c = torch.clamp(0.5 * (out[:, 2] + out[:, 3]), -1.0, 1.0)
        out[:, 2] = theta_c
        out[:, 3] = theta_c
        return out


class SemanticFlowRefiner(nn.Module):
    # RFlow velocity field, small UNet conditioned on time + reliability
    def __init__(self, rel_ch: int, cfg: CovertISACConfig) -> None:
        super().__init__()
        self.cfg = cfg
        td = int(cfg.rf_td)
        ch = int(cfg.rf_ch)
        in_ch = 3 * cfg.sem_ch + rel_ch + 1
        self.temb = nn.Sequential(
            SinusoidalTimeEmbedding(td),
            nn.Linear(td, td),
            nn.SiLU(),
            nn.Linear(td, td),
        )
        self.e1 = FlowBlock(in_ch, ch, td)
        self.d1 = nn.Conv2d(ch, ch, 4, 2, 1)
        self.e2 = FlowBlock(ch, 2 * ch, td)
        self.mid = FlowBlock(2 * ch, 2 * ch, td)
        self.u1 = nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1)
        self.dec = FlowBlock(2 * ch, ch, td)
        self.out_vel = nn.Conv2d(ch, cfg.pair_ch, 3, 1, 1)
        self.out_gate = nn.Conv2d(ch, cfg.pair_ch, 3, 1, 1)
        nn.init.zeros_(self.out_vel.weight)
        nn.init.zeros_(self.out_gate.weight)
        if self.out_vel.bias is not None:
            nn.init.zeros_(self.out_vel.bias)
        if self.out_gate.bias is not None:
            nn.init.zeros_(self.out_gate.bias)

    def forward(
        self,
        u_cur: torch.Tensor,
        t: torch.Tensor,
        u_base: torch.Tensor,
        rel: torch.Tensor,
        slot_union: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        te = self.temb(t)
        u_delta = phase_delta_u(u_cur, u_base, self.cfg)
        inp = torch.cat([u_cur, u_base, u_delta, rel, slot_union], dim=1)
        e1 = self.e1(inp, te)
        e2 = self.e2(self.d1(e1), te)
        m = self.mid(e2, te)
        d = self.dec(torch.cat([self.u1(m), e1], dim=1), te)
        return self.out_vel(d), self.out_gate(d)


# ================================================================
# RFlow helpers
# ================================================================
def masked_mse(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.sum(mask * x * x)
    den = torch.sum(mask) + eps
    return num / den


def non_slot_sem_mse(u_pred: torch.Tensor, u_true: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    theta_p = cs_maps_to_theta(u_pred, cfg)
    theta_t = cs_maps_to_theta(u_true, cfg)
    diff = wrap_phase(theta_p - theta_t)
    keep = (~meta_slot_mask_map(cfg, u_pred.device)).float()
    return masked_mse(diff, keep)


def reliability_gate_from_rel(rel: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    if rel.numel() == 0:
        return torch.ones(rel.shape[0], cfg.pair_ch, cfg.latent_h, cfg.latent_w, device=rel.device, dtype=rel.dtype)
    dev = rel.device
    gate_raw = torch.ones(rel.shape[0], cfg.pair_ch, cfg.latent_h, cfg.latent_w, device=dev, dtype=rel.dtype)
    off = 0
    if cfg.rel_use_phi_resid:
        phi = torch.clamp(rel[:, off:off + cfg.pair_ch], min=0.0, max=math.pi)
        gate_raw = gate_raw * torch.exp(-0.42 * phi)
        off += cfg.pair_ch
    if cfg.rel_use_logden:
        logden = rel[:, off:off + cfg.pair_ch]
        gate_raw = gate_raw * (0.55 + 0.45 * torch.sigmoid(0.48 * (logden + 0.20)))
        off += cfg.pair_ch
    if cfg.rel_use_mag:
        mag_dev = rel[:, off:off + cfg.pair_ch]
        gate_raw = gate_raw * torch.exp(-0.78 * torch.abs(mag_dev))
        off += cfg.pair_ch
    if cfg.rel_use_logtau:
        logtau = rel[:, off:off + cfg.pair_ch]
        gate_raw = gate_raw * torch.exp(-0.35 * torch.abs(logtau))
        off += cfg.pair_ch
    if cfg.rel_use_logsnr:
        logsnr = rel[:, off:off + cfg.pair_ch]
        gate_raw = gate_raw * (0.55 + 0.45 * torch.sigmoid(0.24 * (logsnr - math.log(1.4))))
        off += cfg.pair_ch
    floor = float(getattr(cfg, 'rf_gate_floor', 0.12))
    gate = floor + (1.0 - floor) * torch.clamp(gate_raw, 0.0, 1.0)
    return torch.clamp(gate, min=floor, max=1.0)


def _rf_progress_points(n_steps: int, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n_steps = int(max(1, n_steps))
    u = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    gamma = float(max(1.0, gamma))
    if abs(gamma - 1.0) < 1e-6:
        p = u
    else:
        p = 1.0 - torch.pow(torch.clamp(1.0 - u, min=0.0), gamma)
    p[0] = 0.0
    p[-1] = 1.0
    return p


def _masked_smooth_l1_phase(diff: torch.Tensor, mask: torch.Tensor, beta: float = 0.06, eps: float = 1e-8) -> torch.Tensor:
    err = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=float(beta), reduction='none')
    return torch.sum(err * mask) / (torch.sum(mask) + eps)


def rf_refine_phase_eval(net: Optional[SemanticFlowRefiner], u_base: torch.Tensor, rel: torch.Tensor, cfg: CovertISACConfig, steps: int) -> torch.Tensor:
    # Q-step ODE rollout on the phase; slots stay analytic, pairs stay on the unit circle
    if net is None or int(steps) <= 0:
        return u_base
    theta0 = cs_maps_to_theta(u_base, cfg)
    theta = theta0
    slot_mask = meta_slot_mask_map(cfg, u_base.device)
    slot_union = slot_mask.any(dim=1, keepdim=True).float().expand(u_base.shape[0], -1, -1, -1)
    rel_gate = reliability_gate_from_rel(rel, cfg)
    n_steps = int(max(1, steps))
    prog = _rf_progress_points(n_steps, float(getattr(cfg, 'rf_path_gamma', 1.00)), u_base.device, u_base.dtype)
    step_scale = float(getattr(cfg, 'rf_step_scale', 1.00))
    for k in range(n_steps):
        dt = torch.clamp(prog[k + 1] - prog[k], min=1e-4)
        t = torch.full((u_base.shape[0],), float(prog[k].item()), device=u_base.device, dtype=torch.float32)
        u_cur = theta_to_u(theta, cfg)
        vel_raw, gate_logit = net(u_cur, t, u_base, rel, slot_union)
        step_gate = rel_gate * torch.sigmoid(gate_logit)
        vel = step_gate * (math.pi * step_scale * torch.tanh(vel_raw))
        theta_prop = wrap_phase(theta + dt * vel)
        theta = torch.where(slot_mask, theta0, theta_prop)
    return theta_to_u(theta, cfg)


def rf_semantic_train_objective(
    net: Optional[SemanticFlowRefiner],
    u_base: torch.Tensor,
    rel: torch.Tensor,
    u_true: torch.Tensor,
    cfg: CovertISACConfig,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # flow matching: predicted velocity vs the straight-bridge target
    if net is None or int(steps) <= 0:
        z = torch.tensor(0.0, device=u_base.device)
        return z, z

    with torch.no_grad():
        u0 = u_base.detach()
        u1 = u_true.detach()
        rel0 = rel.detach()

    theta0 = cs_maps_to_theta(u0, cfg)
    theta1 = cs_maps_to_theta(u1, cfg)
    slot_mask = meta_slot_mask_map(cfg, u_base.device)
    keep = (~slot_mask).float()
    slot_union = slot_mask.any(dim=1, keepdim=True).float().expand(u_base.shape[0], -1, -1, -1)
    rel_gate = reliability_gate_from_rel(rel0, cfg)
    n_steps = int(max(1, steps))
    prog = _rf_progress_points(n_steps, float(getattr(cfg, 'rf_path_gamma', 1.00)), u_base.device, u_base.dtype)
    step_scale = float(getattr(cfg, 'rf_step_scale', 1.00))

    theta_est = theta0
    loss_vel = torch.tensor(0.0, device=u_base.device, dtype=u_base.dtype)
    loss_roll = torch.tensor(0.0, device=u_base.device, dtype=u_base.dtype)
    weights = torch.linspace(0.70, 1.00, n_steps, device=u_base.device, dtype=u_base.dtype)
    weights = weights / torch.clamp(weights.sum(), min=1e-8)

    for k in range(n_steps):
        pk = prog[k]
        dt = torch.clamp(prog[k + 1] - prog[k], min=1e-4)
        t = torch.full((u_base.shape[0],), float(pk.item()), device=u_base.device, dtype=torch.float32)
        u_cur = theta_to_u(theta_est, cfg)
        vel_raw, gate_logit = net(u_cur, t, u0, rel0, slot_union)
        step_gate = rel_gate * torch.sigmoid(gate_logit)
        vel_pred = step_gate * (math.pi * step_scale * torch.tanh(vel_raw))

        remain = torch.clamp(1.0 - pk, min=1e-3)
        vel_tgt = wrap_phase(theta1 - theta_est) / remain
        diff_vel = wrap_phase(vel_pred - vel_tgt)
        loss_vel = loss_vel + weights[k] * _masked_smooth_l1_phase(diff_vel, keep, beta=0.06)

        theta_prop = wrap_phase(theta_est + dt * vel_pred)
        theta_est = torch.where(slot_mask, theta0, theta_prop)
        loss_roll = loss_roll + weights[k] * masked_mse(wrap_phase(theta_est - theta1), keep)

    endpoint = masked_mse(wrap_phase(theta_est - theta1), keep)
    loss = float(getattr(cfg, 'rf_step_loss_weight', 1.0)) * loss_vel + 0.35 * loss_roll + float(cfg.rf_endpoint_weight) * endpoint
    return loss, endpoint


# ================================================================
# Modulation / channel / receiver
# ================================================================
def modulate_rotation(c_ms: torch.Tensor, s_ms: torch.Tensor, b0: torch.Tensor, cfg: CovertISACConfig) -> torch.Tensor:
    # rotate each Gaussian reference pair by (c, s)
    u1 = b0[:, :, :, 0]
    u2 = b0[:, :, :, 1]
    v1 = c_ms * u1 - s_ms * u2
    v2 = s_ms * u1 + c_ms * u2
    b = torch.empty(c_ms.shape[0], cfg.num_chirps, cfg.spc, device=c_ms.device, dtype=torch.float32)
    b[:, :, 0::2] = v1
    b[:, :, 1::2] = v2
    return b


def demodulate_rotation_ls(b_hat: torch.Tensor, b0: torch.Tensor, cfg: CovertISACConfig, eps: float = 1e-8):
    # least squares per pair against the shared reference
    v1 = b_hat[:, :, 0::2]
    v2 = b_hat[:, :, 1::2]
    u1 = b0[:, :, :, 0]
    u2 = b0[:, :, :, 1]
    den = u1 * u1 + u2 * u2 + eps
    c_ls = (u1 * v1 + u2 * v2) / den
    s_ls = (u1 * v2 - u2 * v1) / den
    mag = torch.sqrt(c_ls * c_ls + s_ls * s_ls + eps)
    c_norm = c_ls / mag
    s_norm = s_ls / mag
    return c_norm, s_norm, den, mag, c_ls, s_ls


def ar1_forward(b: torch.Tensor, rho: float) -> torch.Tensor:
    if rho <= 0.0:
        return b
    B, M, N = b.shape
    u = torch.empty_like(b)
    u[:, :, 0] = b[:, :, 0]
    scale = math.sqrt(max(1e-12, 1.0 - rho * rho))
    for n in range(1, N):
        u[:, :, n] = rho * u[:, :, n - 1] + scale * b[:, :, n]
    return u


def ar1_inverse(u: torch.Tensor, rho: float) -> torch.Tensor:
    if rho <= 0.0:
        return u
    B, M, N = u.shape
    b = torch.empty_like(u)
    b[:, :, 0] = u[:, :, 0]
    scale = math.sqrt(max(1e-12, 1.0 - rho * rho))
    for n in range(1, N):
        b[:, :, n] = (u[:, :, n] - rho * u[:, :, n - 1]) / scale
    return b


def phase_forward(u: torch.Tensor, alpha: float) -> torch.Tensor:
    return math.pi * torch.tanh(alpha * u)


def phase_inverse_naive(phi: torch.Tensor, alpha: float, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(phi / math.pi, -1.0 + eps, 1.0 - eps)
    return torch.atanh(x) / alpha


def phase_inverse_map(
    phi: torch.Tensor,
    alpha: float,
    sigma2_phi: torch.Tensor,
    iters: int,
    u_clip: float,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    # a few Newton steps on the MAP objective for the tanh phase map
    eps = 1e-8
    s2 = torch.clamp(sigma2_phi, sigma_min, sigma_max)
    u = torch.clamp(phi / (math.pi * alpha + eps), -u_clip, u_clip)
    for _ in range(int(iters)):
        t = torch.tanh(alpha * u)
        f = math.pi * t
        fp = math.pi * alpha * (1.0 - t * t)
        fpp = -2.0 * math.pi * (alpha * alpha) * t * (1.0 - t * t)
        g = (fp / (s2 + eps)) * (f - phi) + u
        gp = (1.0 / (s2 + eps)) * (fpp * (f - phi) + fp * fp) + 1.0
        u = u - g / (gp + eps)
    return torch.clamp(u, -u_clip, u_clip)


def mmse_eq(y: torch.Tensor, h: torch.Tensor, nvar: torch.Tensor) -> torch.Tensor:
    denom = (h.real * h.real + h.imag * h.imag) + nvar
    w = torch.conj(h) / (denom + 1e-12)
    return w * y


def _snr_to_nvar(snr_db: Any, B: int, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(snr_db):
        s = snr_db.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if s.numel() == 1:
            s = s.repeat(B)
        elif s.numel() != B:
            raise ValueError(f"snr_db tensor must have 1 or B elements, got {s.numel()} for B={B}")
    elif isinstance(snr_db, (list, tuple, np.ndarray)):
        arr = np.asarray(snr_db, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, B)
        elif arr.size != B:
            raise ValueError(f"snr_db array must have 1 or B elements, got {arr.size} for B={B}")
        s = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    else:
        s = torch.full((B,), float(snr_db), device=device, dtype=torch.float32)
    snr_lin = 10.0 ** (s / 10.0)
    return (1.0 / snr_lin).reshape(B, 1, 1)


def bob_channel(x: torch.Tensor, snr_db: Any, channel_type: str = "rayleigh") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = x.device
    B, M, N = x.shape
    nvar = _snr_to_nvar(snr_db, B, dev)
    if str(channel_type).lower() == "awgn":
        h = torch.ones(B, M, 1, device=dev, dtype=torch.complex64)
        nstd = torch.sqrt(nvar / 2.0)
        noise = (torch.randn(B, M, N, device=dev) + 1j * torch.randn(B, M, N, device=dev)) * nstd
        y = x + noise.to(torch.complex64)
    elif str(channel_type).lower() == "rayleigh":
        h = ((torch.randn(B, M, 1, device=dev) + 1j * torch.randn(B, M, 1, device=dev)) / math.sqrt(2.0)).to(torch.complex64)
        nstd = torch.sqrt(nvar / 2.0)
        noise = (torch.randn(B, M, N, device=dev) + 1j * torch.randn(B, M, N, device=dev)) * nstd
        y = h * x + noise.to(torch.complex64)
    else:
        raise ValueError(f"Unsupported channel_type: {channel_type}")
    return y.to(torch.complex64), h.to(torch.complex64), nvar


def bob_receive_features(
    y: torch.Tensor,
    h: torch.Tensor,
    nvar: torch.Tensor,
    b0: torch.Tensor,
    chirp: torch.Tensor,
    cfg: CovertISACConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # MMSE eq -> dechirp -> MAP phase inversion -> AR inverse -> LS demod
    xhat = mmse_eq(y, h, nvar)
    zc = xhat * torch.conj(chirp)[None, None, :]
    phi_hat = torch.angle(zc)

    h2 = (h.real * h.real + h.imag * h.imag) + 1e-12
    sigma2_phi = (nvar / (2.0 * h2)).expand_as(phi_hat)
    if cfg.rx_use_map:
        u_hat = phase_inverse_map(
            phi_hat,
            cfg.alpha,
            sigma2_phi,
            iters=cfg.rx_map_iters,
            u_clip=cfg.rx_map_u_clip,
            sigma_min=cfg.rx_sigma_phi_min,
            sigma_max=cfg.rx_sigma_phi_max,
        )
    else:
        u_hat = phase_inverse_naive(phi_hat, cfg.alpha)

    rel_maps: List[torch.Tensor] = []
    resid_pair = None
    if cfg.rel_use_phi_resid:
        phi_pred = phase_forward(u_hat, cfg.alpha)
        d = phi_hat - phi_pred
        d = torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi
        resid = torch.abs(d)
        resid_pair = 0.5 * (resid[:, :, 0::2] + resid[:, :, 1::2])
        rel_maps.append(ms_to_maps(resid_pair, cfg))

    b_hat = ar1_inverse(u_hat, cfg.ar_rho)
    c_norm, s_norm, den, mag, c_ls, s_ls = demodulate_rotation_ls(b_hat, b0, cfg)

    u_sem = torch.cat([ms_to_maps(c_norm, cfg), ms_to_maps(s_norm, cfg)], dim=1)
    tau_hat = torch.clamp(c_ls * c_ls + s_ls * s_ls, min=cfg.tau_floor)
    tau_hat_map = ms_to_maps(tau_hat, cfg)
    den_map = ms_to_maps(den, cfg)

    sigma2_phi_pair = 0.5 * (sigma2_phi[:, :, 0::2] + sigma2_phi[:, :, 1::2])
    conf = den / (sigma2_phi_pair + 1e-8)
    if resid_pair is not None:
        conf = conf / (1.0 + 1.4 * resid_pair)
    conf = conf * torch.exp(-0.35 * torch.abs(torch.log(tau_hat + 1e-8))) * torch.exp(-0.60 * torch.abs(mag - 1.0))
    slot_conf_map = ms_to_maps(torch.clamp(conf, min=1e-8), cfg)

    if cfg.rel_use_logden:
        rel_maps.append(ms_to_maps(torch.log(den + 1e-12) - math.log(2.0), cfg))
    if cfg.rel_use_mag:
        rel_maps.append(ms_to_maps(mag - 1.0, cfg))
    if cfg.rel_use_logtau:
        rel_maps.append(ms_to_maps(torch.log(tau_hat + 1e-12), cfg))
    if cfg.rel_use_logsnr:
        snr_ch = (h2 / (nvar + 1e-12)).expand(y.shape[0], cfg.num_chirps, cfg.spc // 2)
        rel_maps.append(ms_to_maps(torch.log(snr_ch + 1e-12), cfg))

    rel = torch.cat(rel_maps, dim=1) if len(rel_maps) > 0 else torch.zeros_like(u_sem[:, :1])
    return u_sem, rel, tau_hat_map, den_map, slot_conf_map


def make_h0_waveform(b0: torch.Tensor, chirp: torch.Tensor, cfg: CovertISACConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # sensing-only mode: identity rotation on the reference pairs
    c0 = torch.ones(b0.shape[0], cfg.num_chirps, cfg.spc // 2, device=b0.device, dtype=torch.float32)
    s0 = torch.zeros_like(c0)
    b = modulate_rotation(c0, s0, b0, cfg)
    u = ar1_forward(b, cfg.ar_rho)
    phi = phase_forward(u, cfg.alpha)
    x = (torch.exp(1j * phi) * chirp[None, None, :]).to(torch.complex64)
    return x, phi


def make_waveform_from_fields(
    c_dir: torch.Tensor,
    s_dir: torch.Tensor,
    tau_map: torch.Tensor,
    b0: torch.Tensor,
    chirp: torch.Tensor,
    cfg: CovertISACConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    amp_map = torch.sqrt(torch.clamp(tau_map, min=cfg.tau_floor))
    c_ms = maps_to_ms(amp_map * c_dir, cfg)
    s_ms = maps_to_ms(amp_map * s_dir, cfg)
    b = modulate_rotation(c_ms, s_ms, b0, cfg)
    u = ar1_forward(b, cfg.ar_rho)
    phi = phase_forward(u, cfg.alpha)
    x = (torch.exp(1j * phi) * chirp[None, None, :]).to(torch.complex64)
    return x, phi


# ================================================================
# Radar: scene, ULA echo, matched filtering, estimation
# ================================================================
def meta_to_radar_scene(meta: torch.Tensor, cfg: CovertISACConfig, range_max_m: float, vel_max_mps: float) -> Dict[str, torch.Tensor]:
    B = meta.shape[0]
    rn = meta[:, 0]
    vn = meta[:, 1]
    sin_t = meta[:, 2]
    cos_t = meta[:, 3]

    range_m = 0.5 * (rn + 1.0) * float(range_max_m)
    vel_mps = vn * float(vel_max_mps)
    theta_rad = torch.atan2(sin_t, cos_t)
    theta_deg = theta_rad * (180.0 / math.pi)

    delay = float(cfg.radar_delay_min) + (range_m / max(float(range_max_m), 1e-6)) * float(cfg.radar_delay_max - cfg.radar_delay_min)
    dop = ((vel_mps + float(vel_max_mps)) / max(2.0 * float(vel_max_mps), 1e-6)) * float(cfg.radar_dop_max - cfg.radar_dop_min) + float(cfg.radar_dop_min)

    delay = torch.clamp(delay, float(cfg.radar_delay_min), float(cfg.radar_delay_max)).unsqueeze(1)
    dop = torch.clamp(dop, float(cfg.radar_dop_min), float(cfg.radar_dop_max)).unsqueeze(1)
    theta_deg = torch.clamp(theta_deg, -cfg.radar_angle_max_deg, cfg.radar_angle_max_deg).unsqueeze(1)
    amp = torch.full((B, 1), 0.85, device=meta.device, dtype=torch.float32)
    phase = torch.zeros(B, 1, device=meta.device, dtype=torch.float32)
    return {"delay": delay, "doppler": dop, "angle_deg": theta_deg, "amp": amp, "phase": phase}


def fractional_delay_batch_lastdim(x: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
    B, M, N = x.shape
    k = torch.arange(N, device=x.device, dtype=torch.float32)
    X = torch.fft.fft(x, dim=-1)
    phase = torch.exp((-1j * 2.0 * math.pi / float(N)) * delays[:, None] * k[None, :]).to(x.dtype)
    return torch.fft.ifft(X * phase[:, None, :], dim=-1).to(x.dtype)


def radar_echo_ula_torch(x: torch.Tensor, scene: Dict[str, torch.Tensor], snr_db: float, cfg: CovertISACConfig) -> torch.Tensor:
    B, M, N = x.shape
    A = cfg.radar_rx_ant
    dev = x.device
    dtype = x.dtype
    y = torch.zeros(B, A, M, N, device=dev, dtype=dtype)

    ant = torch.arange(A, device=dev, dtype=torch.float32)
    m_idx = torch.arange(M, device=dev, dtype=torch.float32)
    T = scene["delay"].shape[1]

    for t in range(T):
        dl = scene["delay"][:, t].float()
        dp = scene["doppler"][:, t].float()
        ang = scene["angle_deg"][:, t].float() * math.pi / 180.0
        amp = scene["amp"][:, t].float()
        ph = scene["phase"][:, t].float()

        steer = torch.exp(1j * math.pi * ant[None, :] * torch.sin(ang)[:, None]).to(dtype)
        dop_phase = torch.exp(1j * 2.0 * math.pi * dp[:, None] * m_idx[None, :] / float(M)).to(dtype)
        amp_c = (amp * torch.exp(1j * ph)).to(dtype)
        shifted = fractional_delay_batch_lastdim(x, dl)
        contrib = amp_c[:, None, None, None] * steer[:, :, None, None] * dop_phase[:, None, :, None] * shifted[:, None, :, :]
        y = y + contrib

    sp2 = torch.mean(torch.abs(y) ** 2, dim=(1, 2, 3), keepdim=True)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    nvar = sp2 / max(snr_lin, 1e-12)
    nstd = torch.sqrt(nvar / 2.0)
    noise = (torch.randn_like(y) + 1j * torch.randn_like(y)) * nstd
    return (y + noise).to(dtype)


def rd_cube_torch(y: torch.Tensor, xref: torch.Tensor, cfg: Optional[CovertISACConfig] = None) -> torch.Tensor:
    # matched filter in range then FFT over chirps for Doppler
    N = y.shape[-1]
    M = y.shape[-2]
    range_os = int(getattr(cfg, 'radar_range_os', 1)) if cfg is not None else 1
    dop_os = int(getattr(cfg, 'radar_dop_os', 1)) if cfg is not None else 1
    Nr = int(max(1, range_os) * N)
    Md = int(max(1, dop_os) * M)
    Y = torch.fft.fft(y, dim=-1)
    X = torch.fft.fft(xref, dim=-1).unsqueeze(1)
    rng_comp = torch.fft.ifft(Y * torch.conj(X), n=Nr, dim=-1)
    rd = torch.fft.fft(rng_comp, n=Md, dim=-2)
    rd = torch.fft.fftshift(rd, dim=-2)
    return rd


def _parabolic_offset_1d(vm1: torch.Tensor, v0: torch.Tensor, vp1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = (vm1 - 2.0 * v0 + vp1)
    denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)
    off = 0.5 * (vm1 - vp1) / denom
    return torch.clamp(off, -0.5, 0.5)


def estimate_scene_from_rd(rd_cube: torch.Tensor, cfg: CovertISACConfig, return_first_spectrum: bool = False):
    B, A, Mrd, Nrd = rd_cube.shape
    dev = rd_cube.device
    power = torch.sum(torch.abs(rd_cube) ** 2, dim=1)

    range_os = int(max(1, getattr(cfg, 'radar_range_os', 1)))
    dop_os = int(max(1, getattr(cfg, 'radar_dop_os', 1)))

    dmin = int(cfg.radar_delay_min) * range_os
    dmax = int(cfg.radar_delay_max) * range_os
    dop_min = int(cfg.radar_dop_min) * dop_os + (Mrd // 2)
    dop_max = int(cfg.radar_dop_max) * dop_os + (Mrd // 2)
    dmin = max(0, min(Nrd - 1, dmin))
    dmax = max(0, min(Nrd - 1, dmax))
    dop_min = max(0, min(Mrd - 1, dop_min))
    dop_max = max(0, min(Mrd - 1, dop_max))

    ang_grid = torch.linspace(-cfg.radar_angle_max_deg, cfg.radar_angle_max_deg, cfg.radar_angle_grid, device=dev)
    ant = torch.arange(A, device=dev, dtype=torch.float32)
    steer = torch.exp(1j * math.pi * ant[:, None] * torch.sin(ang_grid[None, :] * math.pi / 180.0)).to(rd_cube.dtype)

    est_delay = torch.zeros(B, 1, device=dev, dtype=torch.float32)
    est_dop = torch.zeros(B, 1, device=dev, dtype=torch.float32)
    est_ang = torch.zeros(B, 1, device=dev, dtype=torch.float32)
    est_amp = torch.zeros(B, 1, device=dev, dtype=torch.float32)
    first_spec = None

    for b in range(B):
        pw = power[b, dop_min:dop_max + 1, dmin:dmax + 1]
        idx = int(torch.argmax(pw).item())
        local_db = idx // pw.shape[1]
        local_dl = idx % pw.shape[1]
        db = local_db + dop_min
        dl = local_dl + dmin

        p_center = power[b, db, dl]
        if 1 <= dl < Nrd - 1:
            d_off = _parabolic_offset_1d(power[b, db, dl - 1], p_center, power[b, db, dl + 1])
        else:
            d_off = torch.tensor(0.0, device=dev)
        if 1 <= db < Mrd - 1:
            v_off = _parabolic_offset_1d(power[b, db - 1, dl], p_center, power[b, db + 1, dl])
        else:
            v_off = torch.tensor(0.0, device=dev)

        snap = rd_cube[b, :, db, dl]
        if A >= 2:
            r = torch.sum(torch.conj(snap[:-1]) * snap[1:])
            phase_inc = torch.atan2(r.imag, r.real)
            sin_est = torch.clamp(phase_inc / math.pi, -1.0, 1.0)
            ang_est = torch.asin(sin_est) * (180.0 / math.pi)
        else:
            ang_est = torch.tensor(0.0, device=dev)
        scores = torch.abs(torch.conj(steer).T @ snap) ** 2 / float(max(A * A, 1))
        if not torch.isfinite(ang_est):
            gi = int(torch.argmax(scores).item())
            ang_est = ang_grid[gi]
        if return_first_spectrum and b == 0:
            first_spec = (ang_grid.detach().cpu().numpy(), scores.detach().cpu().numpy())

        est_delay[b, 0] = (float(dl) + float(d_off.item())) / float(range_os)
        est_dop[b, 0] = (float(db - (Mrd // 2)) + float(v_off.item())) / float(dop_os)
        est_ang[b, 0] = float(torch.clamp(ang_est, -cfg.radar_angle_max_deg, cfg.radar_angle_max_deg).item())
        est_amp[b, 0] = float(torch.sqrt(torch.clamp(p_center / max(1, A), min=1e-12)).item())

    est_scene = {"delay": est_delay, "doppler": est_dop, "angle_deg": est_ang, "amp": est_amp}
    return est_scene, first_spec


# ================================================================
# Full system
# ================================================================
class CovertISACVoDSystem(nn.Module):
    def __init__(self, cfg: CovertISACConfig) -> None:
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.encoder = PhaseEncoderLite(cfg)
        self.tau_net = TauNetLite(self.encoder.feat_ch, cfg)
        self.decoder = SemanticDecoderLite(cfg.dec_in_ch, cfg)
        self.meta_head = MetaHeadAnalytic(cfg)
        self.slot_refiner = SlotDomainRefiner(cfg) if cfg.rf_enable else None
        self.rf_net = SemanticFlowRefiner(cfg.rel_ch, cfg) if cfg.rf_enable else None
        self.classifier = ClassifierHead(int(cfg.dec_in_ch), cfg) if getattr(cfg, "cls_enable", False) else None
        self.register_buffer("chirp", chirp_torch(cfg.spc, cfg.chirp_mu, torch.device("cpu")), persistent=False)

    def _chirp(self, device: torch.device) -> torch.Tensor:
        if self.chirp.device != device:
            return chirp_torch(self.cfg.spc, self.cfg.chirp_mu, device)
        return self.chirp

    def _decoder_input(self, u_sem: torch.Tensor, rel: torch.Tensor, rel_scale: float = 1.0) -> torch.Tensor:
        xs = [u_sem]
        if self.cfg.dec_use_rel:
            xs.append(float(rel_scale) * rel)
        if self.cfg.dec_use_slot_mask:
            xs.append(slot_mask_union_map(self.cfg, u_sem.device).float().expand(u_sem.shape[0], -1, -1, -1))
        return torch.cat(xs, dim=1)

    @torch.no_grad()
    def model_info(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "n_params": int(n_params),
            "n_trainable": int(n_trainable),
            "canvas_hw": tuple(self.cfg.canvas_hw),
            "latent_hw": (self.cfg.latent_h, self.cfg.latent_w),
            "pair_ch": int(self.cfg.pair_ch),
            "num_chirps": int(self.cfg.num_chirps),
            "spc": int(self.cfg.spc),
            "k_pairs": int(self.cfg.k_pairs),
            "covert_delta": float(self.cfg.covert_delta),
            "rf_steps": int(self.cfg.rf_steps),
            "cls_enable": bool(getattr(self.cfg, "cls_enable", False)),
            "num_classes": int(getattr(self.cfg, "num_classes", 0)),
        }

    def transmitter(self, img: torch.Tensor, meta: torch.Tensor, b0: Optional[torch.Tensor] = None, cls_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if b0 is None:
            b0 = sample_b0(img.shape[0], self.cfg, img.device)
        chirp = self._chirp(img.device)
        theta_raw, feat = self.encoder(img)
        theta_img = math.pi * torch.tanh(theta_raw)
        c_dir, s_dir = theta_to_cs(theta_img)
        c_dir, s_dir = inject_meta_slots(c_dir, s_dir, meta, self.cfg)
        if bool(getattr(self.cfg, "cls_slot_enable", False)):
            c_dir, s_dir = inject_cls_slots(c_dir, s_dir, cls_idx, self.cfg)
        c_dir, s_dir = normalize_cs(c_dir, s_dir)
        u_true = torch.cat([c_dir, s_dir], dim=1)

        tau_raw = self.tau_net(feat)
        tau_raw = force_tau_neutral_on_slots(tau_raw, self.cfg)
        tau_map, kl = project_tau_budget_map(tau_raw, self.cfg)
        x, phi = make_waveform_from_fields(c_dir, s_dir, tau_map, b0, chirp, self.cfg)
        return {
            "x": x,
            "phi": phi,
            "b0": b0,
            "u_true": u_true,
            "tau_map": tau_map,
            "kl": kl,
            "c_dir": c_dir,
            "s_dir": s_dir,
        }

    def receiver(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        nvar: torch.Tensor,
        b0: torch.Tensor,
        rf_steps_override: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        chirp = self._chirp(y.device)
        u_sem, rel, tau_hat_map, den_map, slot_conf_map = bob_receive_features(y, h, nvar, b0, chirp, self.cfg)
        internal_ana, slot_feat = self.meta_head.analytic_decode_internal(u_sem, tau_hat_map, slot_conf_map, den_map)
        cls_slot_mu = None
        cls_slot_logits = None
        cls_slot_feat = None
        if bool(getattr(self.cfg, "cls_slot_enable", False)):
            cls_slot_mu, cls_slot_logits, cls_slot_feat = decode_cls_slots(u_sem, tau_hat_map, slot_conf_map, den_map, self.cfg)
        internal_base = internal_ana
        meta_base = self.meta_head.public_from_internal(internal_base)
        dec_in_base = self._decoder_input(u_sem, rel, rel_scale=float(self.cfg.dec_rel_scale_base))
        img_base = self.decoder(dec_in_base)

        steps_eff = int(self.cfg.rf_steps if rf_steps_override is None else rf_steps_override)
        if self.rf_net is not None and steps_eff > 0:
            u_rf_non = rf_refine_phase_eval(self.rf_net, u_sem, rel, self.cfg, steps=steps_eff)
            internal_rf = self.slot_refiner(internal_ana, slot_feat) if self.slot_refiner is not None else internal_ana
            c_non = u_rf_non[:, :self.cfg.pair_ch]
            s_non = u_rf_non[:, self.cfg.pair_ch: 2 * self.cfg.pair_ch]
            c_rf, s_rf = inject_internal_slots(c_non, s_non, internal_rf, self.cfg)
            if cls_slot_mu is not None:
                c_rf, s_rf = inject_cls_slot_angle(c_rf, s_rf, cls_slot_mu, self.cfg)
            c_rf, s_rf = normalize_cs(c_rf, s_rf)
            u_rf = torch.cat([c_rf, s_rf], dim=1)
            dec_in_rf = self._decoder_input(u_rf, rel, rel_scale=float(self.cfg.dec_rel_scale_rf))
            img_rf = self.decoder(dec_in_rf)
            meta_rf = self.meta_head.public_from_internal(internal_rf)
        else:
            internal_rf = internal_base
            u_rf = u_sem
            img_rf = img_base
            dec_in_rf = dec_in_base
            meta_rf = meta_base
            steps_eff = 0

        # classifier has its own branch on the recovered field, so its gradients
        # never touch the image decoder
        cls_logits = None
        cls_logits_base = None
        cls_extra_rf = None
        if self.classifier is not None:
            cls_in_rf = dec_in_rf.detach() if bool(getattr(self.cfg, "cls_detach_feat", False)) else dec_in_rf
            cls_in_base = dec_in_base.detach() if bool(getattr(self.cfg, "cls_detach_feat", False)) else dec_in_base
            parts_rf: List[torch.Tensor] = []
            parts_base: List[torch.Tensor] = []
            if bool(self.cfg.cls_use_meta):
                parts_rf.append(internal_rf.detach())
                parts_base.append(internal_base.detach())
            if cls_slot_feat is not None:
                parts_rf.append(cls_slot_feat.detach())
                parts_base.append(cls_slot_feat.detach())
            cls_extra_rf = torch.cat(parts_rf, dim=1) if len(parts_rf) > 0 else None
            cls_extra_base = torch.cat(parts_base, dim=1) if len(parts_base) > 0 else None
            cls_logits = self.classifier(cls_in_rf, cls_extra_rf)
            cls_logits_base = self.classifier(cls_in_base, cls_extra_base)
            if cls_slot_logits is not None:
                cls_logits = cls_logits + cls_slot_logits
                cls_logits_base = cls_logits_base + cls_slot_logits

        return {
            "u_sem": u_sem,
            "rel": rel,
            "tau_hat": tau_hat_map,
            "den_map": den_map,
            "slot_conf_map": slot_conf_map,
            "slot_feat": slot_feat,
            "internal_base": internal_base,
            "img_base": img_base,
            "meta_base": meta_base,
            "u_rf": u_rf,
            "internal_rf": internal_rf,
            "img_rf": img_rf,
            "meta_rf": meta_rf,
            "cls_logits": cls_logits,
            "cls_logits_base": cls_logits_base,
            "cls_feat_rf": dec_in_rf,
            "cls_extra_rf": cls_extra_rf,
            "cls_slot_logits": cls_slot_logits,
            "rf_steps_used": torch.tensor(float(steps_eff), device=y.device),
        }

    def rf_semantic_loss(self, u_base: torch.Tensor, rel: torch.Tensor, u_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return rf_semantic_train_objective(self.rf_net, u_base, rel, u_true, self.cfg, steps=int(getattr(self.cfg, 'rf_train_steps', self.cfg.rf_steps)))

    def forward_batch(
        self,
        img: torch.Tensor,
        meta: torch.Tensor,
        snr_db: Any,
        channel_type: str = "rayleigh",
        rf_steps_override: Optional[int] = None,
        cls_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tx = self.transmitter(img, meta, cls_idx=cls_idx)
        y, h, nvar = bob_channel(tx["x"], snr_db=snr_db, channel_type=channel_type)
        rx = self.receiver(y, h, nvar, tx["b0"], rf_steps_override=rf_steps_override)
        return {**tx, **rx, "y": y, "h": h, "nvar": nvar}

    @torch.no_grad()
    def make_h0(self, B: int, device: torch.device) -> Dict[str, torch.Tensor]:
        b0 = sample_b0(B, self.cfg, device)
        chirp = self._chirp(device)
        x0, phi0 = make_h0_waveform(b0, chirp, self.cfg)
        return {"x": x0, "phi": phi0, "b0": b0}
