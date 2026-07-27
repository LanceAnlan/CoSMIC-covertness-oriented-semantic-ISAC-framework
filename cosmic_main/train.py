# -*- coding: utf-8 -*-
# End-to-end training (Algorithm 1 in the paper) and evaluation.

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import VoDCovertISACDataset, vod_collate_fn, VOD_DEFAULT_CLASSES
from modules import CovertISACConfig, CovertISACVoDSystem
from utils import (
    seed_everything,
    ensure_dir,
    save_json,
    psnr_from_mse,
    masked_mse,
    masked_l1,
    ssim_torch,
    bbox_iou_xyxy,
)


# ================================================================
# Checkpoint IO
# ================================================================
def save_checkpoint(path: str, model: CovertISACVoDSystem, optimizer: torch.optim.Optimizer, epoch: int, cfg: Dict[str, Any], best_score: float) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "cfg": cfg,
        "model_cfg": asdict(model.cfg),
        "best_score": float(best_score),
        "model_info": model.model_info(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
        return torch.load(path, map_location=map_location, weights_only=False)


def build_model_cfg(cfg: Dict[str, Any]) -> CovertISACConfig:
    mcfg = CovertISACConfig(
        canvas_hw=tuple(cfg["canvas_hw"]),
        latent_down=int(cfg["latent_down"]),
        pair_ch=int(cfg["pair_ch"]),
        num_chirps=int(cfg["num_chirps"]),
        spc=int(cfg["spc"]),
        chirp_mu=float(cfg["chirp_mu"]),
        alpha=float(cfg["alpha"]),
        ar_rho=float(cfg["ar_rho"]),
        rx_use_map=bool(cfg["rx_use_map"]),
        rx_map_iters=int(cfg["rx_map_iters"]),
        rx_map_u_clip=float(cfg["rx_map_u_clip"]),
        rx_sigma_phi_min=float(cfg["rx_sigma_phi_min"]),
        rx_sigma_phi_max=float(cfg["rx_sigma_phi_max"]),
        covert_eps=float(cfg["covert_eps"]),
        covert_delta_override=float(cfg.get("covert_delta_override", -1.0)),
        tau_proj_iters=int(cfg["tau_proj_iters"]),
        tau_floor=float(cfg["tau_floor"]),
        enc_base=int(cfg["enc_base"]),
        enc_blocks=int(cfg["enc_blocks"]),
        dec_base=int(cfg["dec_base"]),
        dec_blocks_lat=int(cfg["dec_blocks_lat"]),
        dec_blocks_img=int(cfg["dec_blocks_img"]),
        dec_res_scale=float(cfg["dec_res_scale"]),
        tau_hidden=int(cfg["tau_hidden"]),
        dec_use_rel=bool(cfg.get("dec_use_rel", True)),
        dec_use_slot_mask=bool(cfg.get("dec_use_slot_mask", True)),
        dec_rel_scale_base=float(cfg.get("dec_rel_scale_base", 1.00)),
        dec_rel_scale_rf=float(cfg.get("dec_rel_scale_rf", 0.15)),
        cls_enable=bool(cfg.get("cls_enable", True)),
        num_classes=int(cfg.get("num_classes", len(VOD_DEFAULT_CLASSES))),
        cls_hidden=int(cfg.get("cls_hidden", 256)),
        cls_dropout=float(cfg.get("cls_dropout", 0.20)),
        cls_blocks=int(cfg.get("cls_blocks", 2)),
        cls_use_meta=bool(cfg.get("cls_use_meta", True)),
        cls_detach_feat=bool(cfg.get("cls_detach_feat", False)),
        cls_input_dropout=float(cfg.get("cls_input_dropout", 0.10)),
        cls_feat_noise=float(cfg.get("cls_feat_noise", 0.05)),
        cls_slot_enable=bool(cfg.get("cls_slot_enable", False)),
        cls_slot_repeats=int(cfg.get("cls_slot_repeats", 96)),
        cls_slot_theta_scale=float(cfg.get("cls_slot_theta_scale", 1.50)),
        cls_slot_temp=float(cfg.get("cls_slot_temp", 8.0)),
        grad_checkpoint=bool(cfg.get("grad_checkpoint", True)),
        meta_dim=8,
        meta_slot_repeats=tuple(int(x) for x in cfg["meta_slot_repeats"]),
        meta_theta_max=float(cfg["meta_theta_max"]),
        meta_theta_scales=tuple(float(x) for x in cfg.get("meta_theta_scales", [cfg["meta_theta_max"]] * 8)),
        meta_refine_scale=float(cfg["meta_refine_scale"]),
        vel_compand_beta=float(cfg.get("vel_compand_beta", 1.60)),
        theta_compand_beta=float(cfg.get("theta_compand_beta", 1.80)),
        slot_refine_hidden=int(cfg.get("slot_refine_hidden", 192)),
        slot_refine_scale=float(cfg.get("slot_refine_scale", 0.10)),
        meta_balanced_slots=bool(cfg.get("meta_balanced_slots", True)),
        rel_use_logden=bool(cfg["rel_use_logden"]),
        rel_use_mag=bool(cfg["rel_use_mag"]),
        rel_use_logsnr=bool(cfg["rel_use_logsnr"]),
        rel_use_phi_resid=bool(cfg["rel_use_phi_resid"]),
        rel_use_logtau=bool(cfg["rel_use_logtau"]),
        rf_enable=bool(cfg["rf_enable"]),
        rf_steps=int(cfg["rf_steps"]),
        rf_ch=int(cfg["rf_ch"]),
        rf_blocks=int(cfg["rf_blocks"]),
        rf_td=int(cfg.get("rf_td", 96)),
        rf_endpoint_weight=float(cfg.get("rf_endpoint_weight", 0.12)),
        rf_gate_floor=float(cfg.get("rf_gate_floor", 0.10)),
        rf_train_steps=int(cfg.get("rf_train_steps", max(int(cfg.get("rf_steps", 2)), 5))),
        rf_path_gamma=float(cfg.get("rf_path_gamma", 1.00)),
        rf_step_scale=float(cfg.get("rf_step_scale", 1.00)),
        rf_step_loss_weight=float(cfg.get("rf_step_loss_weight", 1.00)),
        radar_rx_ant=int(cfg["radar_rx_ant"]),
        radar_angle_max_deg=float(cfg["radar_angle_max_deg"]),
        radar_angle_grid=int(cfg["radar_angle_grid"]),
        radar_range_os=int(cfg.get("radar_range_os", 1)),
        radar_dop_os=int(cfg.get("radar_dop_os", 1)),
        radar_delay_min=int(cfg["radar_delay_min"]),
        radar_delay_max=int(cfg["radar_delay_max"]),
        radar_dop_min=int(cfg["radar_dop_min"]),
        radar_dop_max=int(cfg["radar_dop_max"]),
        radar_nms_delay=int(cfg["radar_nms_delay"]),
        radar_nms_dop=int(cfg["radar_nms_dop"]),
        radar_det_tol_delay=int(cfg["radar_det_tol_delay"]),
        radar_det_tol_dop=int(cfg["radar_det_tol_dop"]),
        radar_det_tol_angle=float(cfg["radar_det_tol_angle"]),
        radar_amp_min=float(cfg["radar_amp_min"]),
        radar_amp_max=float(cfg["radar_amp_max"]),
        seed=int(cfg["seed"]),
    )
    mcfg.check()
    return mcfg


# ================================================================
# SNR sampling
# ================================================================
def _as_float_list(x: Any) -> List[float]:
    if isinstance(x, (list, tuple, np.ndarray)):
        vals = [float(v) for v in x]
    else:
        vals = [float(x)]
    if len(vals) == 0:
        raise ValueError("SNR list must not be empty")
    return vals


def _make_snr_probs(snrs: List[float], mode: str = "uniform", weights: Optional[List[float]] = None) -> np.ndarray:
    snrs = [float(s) for s in snrs]
    mode_l = str(mode).lower()
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != len(snrs):
            raise ValueError(f"SNR weight length mismatch: {w.size} vs {len(snrs)}")
        w = np.maximum(w, 0.0)
        if np.sum(w) <= 0:
            w = np.ones_like(w)
        return w / np.sum(w)
    if mode_l.startswith("continuous"):
        return np.ones((max(1, len(snrs)),), dtype=np.float64) / max(1, len(snrs))
    if len(snrs) == 1:
        return np.ones((1,), dtype=np.float64)
    if mode_l == "low_bias":
        s = np.asarray(snrs, dtype=np.float64)
        w = np.exp(-(s - np.min(s)) / 8.0)
    else:
        w = np.ones((len(snrs),), dtype=np.float64)
    return w / np.sum(w)


def _sample_train_snr(snrs: List[float], probs: np.ndarray, mode: str = "uniform") -> float:
    mode_l = str(mode).lower()
    if mode_l == "continuous_uniform":
        lo = float(np.min(snrs))
        hi = float(np.max(snrs))
        return float(np.random.uniform(lo, hi))
    if mode_l == "continuous_low_bias":
        lo = float(np.min(snrs))
        hi = float(np.max(snrs))
        t = np.random.beta(0.85, 1.65)
        return float(lo + (hi - lo) * t)
    if len(snrs) == 1:
        return float(snrs[0])
    idx = int(np.random.choice(len(snrs), p=probs))
    return float(snrs[idx])


def _weighted_average_dict(rows: List[Dict[str, float]], weights: np.ndarray) -> Dict[str, float]:
    if len(rows) == 0:
        return {}
    out: Dict[str, float] = {}
    keys = set(rows[0].keys())
    for r in rows[1:]:
        keys &= set(r.keys())
    for k in keys:
        vals = []
        ok = True
        for r in rows:
            v = r[k]
            if isinstance(v, (float, int, np.floating, np.integer)):
                vals.append(float(v))
            else:
                ok = False
                break
        if ok:
            out[k] = float(np.sum(weights * np.asarray(vals, dtype=np.float64)))
    return out


# ================================================================
# Losses / metrics
# ================================================================
def _image_grad_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_g = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_g = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    return masked_l1(dx_p, dx_g, mx) + masked_l1(dy_p, dy_g, my)


def image_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse = masked_mse(pred, gt, mask)
    l1 = masked_l1(pred, gt, mask)
    ssim_val = ssim_torch(pred, gt)
    grad = _image_grad_l1(pred, gt, mask)
    loss = (
        mse
        + float(cfg.get("img_l1_weight", 0.08)) * l1
        + float(cfg.get("img_ssim_weight", 0.10)) * (1.0 - ssim_val)
        + float(cfg.get("img_grad_weight", 0.03)) * grad
    )
    return loss, {"mse": float(mse.item()), "l1": float(l1.item()), "ssim": float(ssim_val.item()), "grad": float(grad.item())}


def weighted_meta_rmse(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = weights.view(1, -1).to(pred.device, dtype=pred.dtype)
    mse = torch.mean(w * (pred - gt) ** 2)
    return torch.sqrt(mse + eps)


def weighted_meta_loss_public(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.view(1, -1).to(pred.device, dtype=pred.dtype)
    diff = F.smooth_l1_loss(pred, gt, reduction="none", beta=float(0.08))
    return torch.mean(w * diff)


def meta_phys_loss(pred: torch.Tensor, gt: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    # extra loss straight in physical units so the reported RMSEs go down too
    r_p, v_p, th_p, _ = _meta_phys_from_norm_torch(pred, cfg)
    r_g, v_g, th_g, _ = _meta_phys_from_norm_torch(gt, cfg)
    rmax = max(float(cfg["meta_range_max"]), 1e-6)
    vmax = max(float(cfg["meta_vel_max"]), 1e-6)
    th_max = max(float(cfg.get("radar_angle_max_deg", 60.0)) * math.pi / 180.0, 1e-6)
    dth = torch.atan2(torch.sin(th_p - th_g), torch.cos(th_p - th_g))
    l_r = F.smooth_l1_loss(r_p / rmax, r_g / rmax, beta=0.02)
    l_v = F.smooth_l1_loss(v_p / vmax, v_g / vmax, beta=0.02)
    l_th = F.smooth_l1_loss(dth / th_max, torch.zeros_like(dth), beta=0.02)
    wr = float(cfg.get("phys_range_weight", 1.0))
    wv = float(cfg.get("phys_vel_weight", 1.0))
    wt = float(cfg.get("phys_angle_weight", 1.0))
    return wr * l_r + wv * l_v + wt * l_th


def compute_class_weights(class_counts: List[int], device: torch.device, power: float = 0.5, cap: float = 8.0) -> torch.Tensor:
    counts = np.asarray([max(int(c), 1) for c in class_counts], dtype=np.float64)
    w = (counts.sum() / (len(counts) * counts)) ** float(power)
    w = np.clip(w, 1.0 / float(cap), float(cap))
    w = w / np.mean(w)
    return torch.tensor(w, device=device, dtype=torch.float32)


def classification_loss(
    logits: Optional[torch.Tensor],
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    if logits is None:
        return torch.tensor(0.0, device=target.device)
    return F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )


def classification_acc(logits: Optional[torch.Tensor], target: torch.Tensor) -> float:
    if logits is None or target.numel() == 0:
        return 0.0
    pred = torch.argmax(logits, dim=1)
    return float((pred == target).float().mean().item())


def refinement_objective(
    model: CovertISACVoDSystem,
    out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    loss_rf, loss_rf_sem = model.rf_semantic_loss(out["u_sem"], out["rel"], out["u_true"])
    return loss_rf, loss_rf_sem


def _meta_phys_from_norm_torch(meta_norm: torch.Tensor, cfg: Dict[str, Any]):
    rn = meta_norm[:, 0]
    vn = meta_norm[:, 1]
    sin_t = meta_norm[:, 2]
    cos_t = meta_norm[:, 3]
    cxn, cyn, wn, hn = meta_norm[:, 4], meta_norm[:, 5], meta_norm[:, 6], meta_norm[:, 7]

    range_m = 0.5 * (rn + 1.0) * float(cfg["meta_range_max"])
    vel_mps = vn * float(cfg["meta_vel_max"])
    theta = torch.atan2(sin_t, cos_t)

    cx01 = torch.clamp(0.5 * (cxn + 1.0), 0.0, 1.0)
    cy01 = torch.clamp(0.5 * (cyn + 1.0), 0.0, 1.0)
    w01 = torch.clamp(0.5 * (wn + 1.0), 0.0, 1.0)
    h01 = torch.clamp(0.5 * (hn + 1.0), 0.0, 1.0)
    bbox01 = torch.stack([cx01, cy01, w01, h01], dim=-1)
    return range_m, vel_mps, theta, bbox01


def compute_metrics_single_pred(
    img_gt: torch.Tensor,
    img_hat: torch.Tensor,
    meta_gt: torch.Tensor,
    meta_hat: torch.Tensor,
    mask: torch.Tensor,
    img_wh: Optional[torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    with torch.no_grad():
        mse_img = float(masked_mse(img_hat, img_gt, mask).item())
        psnr = psnr_from_mse(mse_img)
        ssim = float(ssim_torch(img_hat, img_gt).item())
        meta_rmse = float(torch.sqrt(torch.mean((meta_hat - meta_gt) ** 2)).item())

        r_gt, v_gt, th_gt, box_gt = _meta_phys_from_norm_torch(meta_gt, cfg)
        r_hat, v_hat, th_hat, box_hat = _meta_phys_from_norm_torch(meta_hat, cfg)

        rmse_r = float(torch.sqrt(torch.mean((r_hat - r_gt) ** 2)).item())
        rmse_v = float(torch.sqrt(torch.mean((v_hat - v_gt) ** 2)).item())
        dth = torch.atan2(torch.sin(th_hat - th_gt), torch.cos(th_hat - th_gt)) * (180.0 / math.pi)
        mae_th = float(torch.mean(torch.abs(dth)).item())
        rmse_th = float(torch.sqrt(torch.mean(dth ** 2)).item())

        iou = 0.0
        if img_wh is not None:
            wh = img_wh.detach().cpu().numpy()
            bg = box_gt.detach().cpu().numpy()
            bh = box_hat.detach().cpu().numpy()
            vals = []
            for i in range(bg.shape[0]):
                W = float(wh[i, 0])
                H = float(wh[i, 1])
                cgx, cgy, gw, gh = bg[i]
                chx, chy, hw, hh = bh[i]
                gt_xyxy = ((cgx - 0.5 * gw) * W, (cgy - 0.5 * gh) * H, (cgx + 0.5 * gw) * W, (cgy + 0.5 * gh) * H)
                ht_xyxy = ((chx - 0.5 * hw) * W, (chy - 0.5 * hh) * H, (chx + 0.5 * hw) * W, (chy + 0.5 * hh) * H)
                vals.append(bbox_iou_xyxy(gt_xyxy, ht_xyxy))
            iou = float(np.mean(vals)) if len(vals) > 0 else 0.0

        return {
            "mse_img": mse_img,
            "psnr": float(psnr),
            "ssim": float(ssim),
            "meta_rmse": meta_rmse,
            "range_rmse_m": rmse_r,
            "vel_rmse_mps": rmse_v,
            "angle_mae_deg": mae_th,
            "angle_rmse_deg": rmse_th,
            "bbox_iou": iou,
        }


# ================================================================
# Evaluation
# ================================================================
def evaluate(
    model: CovertISACVoDSystem,
    loader: DataLoader,
    device: torch.device,
    snr_db: float,
    bob_channel_type: str,
    cfg: Dict[str, Any],
    max_batches: Optional[int] = None,
    desc: str = "Eval",
    rf_steps_override: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    weights = torch.tensor(cfg["meta_loss_weights"], device=device, dtype=torch.float32)

    sums: Dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for bidx, batch in enumerate(pbar):
            if max_batches is not None and bidx >= int(max_batches):
                break
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            meta = batch["meta"].to(device)
            img_wh = batch["img_wh"].to(device) if "img_wh" in batch else None
            cls_idx = batch["cls_idx"].to(device) if "cls_idx" in batch else None

            out = model.forward_batch(img, meta, snr_db=snr_db, channel_type=bob_channel_type, rf_steps_override=rf_steps_override, cls_idx=cls_idx)
            acc_rf = classification_acc(out.get("cls_logits"), cls_idx) if cls_idx is not None else 0.0
            acc_base = classification_acc(out.get("cls_logits_base"), cls_idx) if cls_idx is not None else 0.0

            loss_img_base, _ = image_loss(out["img_base"], img, mask, cfg)
            loss_img_rf, _ = image_loss(out["img_rf"], img, mask, cfg)
            loss_meta_base = weighted_meta_rmse(out["meta_base"], meta, weights)
            loss_meta_rf = weighted_meta_rmse(out["meta_rf"], meta, weights)
            if model.rf_net is not None:
                loss_rf, loss_rf_sem = refinement_objective(model, out)
            else:
                z = torch.tensor(0.0, device=device)
                loss_rf, loss_rf_sem = z, z
            total = (
                loss_img_rf
                + float(cfg["lambda_meta"]) * weighted_meta_loss_public(out["meta_rf"], meta, weights)
                + float(cfg["lambda_rf"]) * loss_rf
                + float(cfg.get("base_loss_weight", 0.0)) * loss_img_base
            )

            mb = compute_metrics_single_pred(img, out["img_base"], meta, out["meta_base"], mask, img_wh, cfg)
            mr = compute_metrics_single_pred(img, out["img_rf"], meta, out["meta_rf"], mask, img_wh, cfg)

            vals = {
                "loss": float(total.item()),
                "loss_img_base": float(loss_img_base.item()),
                "loss_img_rf": float(loss_img_rf.item()),
                "loss_meta_base": float(loss_meta_base.item()),
                "loss_meta_rf": float(loss_meta_rf.item()),
                "loss_rf_sem": float(loss_rf_sem.item()),
                "loss_rf_total": float(loss_rf.item()),
                "cert_kl": float(out["kl"].mean().item()),
                "tau_mean": float(out["tau_map"].mean().item()),
                "base_cls_acc": float(acc_base),
                "rf_cls_acc": float(acc_rf),
                **{f"base_{k}": float(v) for k, v in mb.items()},
                **{f"rf_{k}": float(v) for k, v in mr.items()},
            }
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n_batches += 1

            pbar.set_postfix({
                "PSNR": f"{sums['rf_psnr']/n_batches:.2f}",
                "SSIM": f"{sums['rf_ssim']/n_batches:.3f}",
                "mRMSE": f"{sums['rf_meta_rmse']/n_batches:.3f}",
                "Acc": f"{sums['rf_cls_acc']/n_batches:.3f}",
                "KL": f"{sums['cert_kl']/n_batches:.3e}",
            })

    if n_batches == 0:
        return {}
    outm = {k: v / n_batches for k, v in sums.items()}
    outm["joint_score"] = (
        outm["rf_psnr"]
        + 10.0 * outm["rf_ssim"]
        - float(cfg["model_select_meta_weight"]) * outm["rf_meta_rmse"]
        + float(cfg.get("model_select_cls_weight", 15.0)) * outm.get("rf_cls_acc", 0.0)
    )
    return outm


def evaluate_multi_snr(
    model: CovertISACVoDSystem,
    loader: DataLoader,
    device: torch.device,
    snr_list: List[float],
    snr_probs: np.ndarray,
    bob_channel_type: str,
    cfg: Dict[str, Any],
    max_batches: Optional[int] = None,
    desc_prefix: str = "Eval",
    rf_steps_override: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    for snr_db in snr_list:
        row = evaluate(
            model,
            loader,
            device=device,
            snr_db=float(snr_db),
            bob_channel_type=bob_channel_type,
            cfg=cfg,
            max_batches=max_batches,
            desc=f"{desc_prefix}@{snr_db}dB",
            rf_steps_override=rf_steps_override,
        )
        row["snr_db"] = float(snr_db)
        rows.append(row)
    agg = _weighted_average_dict(rows, snr_probs.astype(np.float64))
    if len(agg) > 0:
        agg["snr_list"] = [float(s) for s in snr_list]
        agg["snr_probs"] = [float(x) for x in snr_probs.tolist()]
    return agg, rows


def plot_history(history: List[Dict[str, float]], out_dir: str) -> None:
    ensure_dir(out_dir)
    if len(history) == 0:
        return
    epochs = [h["epoch"] for h in history]

    def series(k: str) -> List[float]:
        return [float(h.get(k, 0.0)) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    axes[0].plot(epochs, series("train_loss"), "-o", ms=3, label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(epochs, series("val_rf_psnr"), "-o", ms=3, label="val PSNR")
    axes[1].plot(epochs, series("val_base_psnr"), "--s", ms=3, label="val PSNR (no RFlow)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("PSNR (dB)"); axes[1].grid(alpha=0.3); axes[1].legend()
    axes[2].plot(epochs, series("val_rf_meta_rmse"), "-o", ms=3, label="val meta RMSE")
    axes[2].plot(epochs, series("val_rf_cls_acc"), "--s", ms=3, label="val cls acc")
    axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3); axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_curve.png"), dpi=150)
    plt.close(fig)


# ================================================================
# Train
# ================================================================
def train(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg["seed"]))
    device = torch.device(cfg["device"])

    train_snr_list = _as_float_list(cfg["train_snr_dbs"])
    val_snr_list = _as_float_list(cfg["val_snr_dbs"])
    train_snr_probs = _make_snr_probs(train_snr_list, mode=str(cfg.get("train_snr_sampling", "uniform")), weights=cfg.get("train_snr_weights", None))
    val_snr_probs = _make_snr_probs(val_snr_list, mode=str(cfg.get("val_snr_sampling", "uniform")), weights=cfg.get("val_snr_weights", None))

    train_set = VoDCovertISACDataset(
        vod_root=cfg["vod_root"],
        split="train",
        canvas_hw=tuple(cfg["canvas_hw"]),
        meta_range_max=cfg["meta_range_max"],
        meta_vel_max=cfg["meta_vel_max"],
        frame_rate=cfg["frame_rate"],
        min_range=cfg["min_range"],
        max_range=cfg["max_range"],
        min_bbox_size=cfg["min_bbox_size"],
        wanted_classes=cfg["wanted_classes"],
        max_samples=cfg.get("max_train_samples", None),
        min_track_len=cfg.get("min_track_len", 3),
        min_quality_score=cfg.get("min_quality_score", 0.0),
        sort_by_quality=True,
        image_cache_size=int(cfg.get("image_cache_size", 256)),
        bbox_mode=str(cfg.get("bbox_mode", "tx")),
        roi_max_scale=float(cfg.get("roi_max_scale", 8.0)),
        class_list=cfg.get("class_list", None),
        class_balanced_selection=bool(cfg.get("class_balanced_train", True)),
        augment=bool(cfg.get("augment_train", True)),
        aug_hflip=bool(cfg.get("aug_hflip", True)),
        aug_photometric=bool(cfg.get("aug_photometric", True)),
        aug_strength=float(cfg.get("aug_strength", 0.20)),
        aug_noise_std=float(cfg.get("aug_noise_std", 0.02)),
    )

    # the class vocabulary travels with the checkpoint
    cfg["class_list"] = list(train_set.class_list)
    cfg["num_classes"] = int(train_set.num_classes)
    _train_counts = [0 for _ in range(train_set.num_classes)]
    for _s in train_set.samples:
        _train_counts[train_set.class_to_idx[_s.cls]] += 1
    cfg["class_counts_train"] = list(_train_counts)
    class_weights = (
        compute_class_weights(
            _train_counts, device,
            power=float(cfg.get("cls_weight_power", 0.5)),
            cap=float(cfg.get("cls_weight_cap", 8.0)),
        )
        if bool(cfg.get("cls_use_class_weights", False))
        else None
    )
    print("Class list:", cfg["class_list"])
    print("Train class counts:", dict(zip(cfg["class_list"], _train_counts)))

    val_set = VoDCovertISACDataset(
        vod_root=cfg["vod_root"],
        split="val",
        canvas_hw=tuple(cfg["canvas_hw"]),
        meta_range_max=cfg["meta_range_max"],
        meta_vel_max=cfg["meta_vel_max"],
        frame_rate=cfg["frame_rate"],
        min_range=cfg["min_range"],
        max_range=cfg["max_range"],
        min_bbox_size=cfg["min_bbox_size"],
        wanted_classes=cfg["wanted_classes"],
        max_samples=cfg.get("max_val_samples", None),
        min_track_len=cfg.get("min_track_len", 3),
        min_quality_score=cfg.get("min_quality_score", 0.0),
        sort_by_quality=True,
        image_cache_size=int(cfg.get("image_cache_size", 256)),
        bbox_mode=str(cfg.get("bbox_mode", "tx")),
        roi_max_scale=float(cfg.get("roi_max_scale", 8.0)),
        class_list=cfg["class_list"],
        class_balanced_selection=bool(cfg.get("class_balanced_eval", True)),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=vod_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=vod_collate_fn,
    )

    model_cfg = build_model_cfg(cfg)
    model = CovertISACVoDSystem(model_cfg).to(device)
    # classifier head gets its own lr / weight decay group
    cls_lr_mult = float(cfg.get("cls_lr_mult", 1.0))
    cls_weight_decay = float(cfg.get("cls_weight_decay", cfg["weight_decay"]))
    if model.classifier is not None:
        cls_ids = {id(p) for p in model.classifier.parameters()}
        base_params = [p for p in model.parameters() if id(p) not in cls_ids]
        cls_params = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": float(cfg["lr"]), "weight_decay": float(cfg["weight_decay"])},
                {"params": cls_params, "lr": float(cfg["lr"]) * cls_lr_mult, "weight_decay": cls_weight_decay},
            ],
            betas=tuple(cfg["betas"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            betas=tuple(cfg["betas"]),
        )
    meta_weights = torch.tensor(cfg["meta_loss_weights"], device=device, dtype=torch.float32)

    exp_dir = os.path.join(cfg["models_dir"], cfg["exp_name"])
    ensure_dir(exp_dir)
    save_json(cfg, os.path.join(exp_dir, "config.json"))
    save_json(model.model_info(), os.path.join(exp_dir, "model_info.json"))

    print("Model info:", model.model_info())
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    print(f"Bob channel: {cfg['bob_channel_type']} | Train SNRs: {train_snr_list} | Val SNRs: {val_snr_list}")
    print(f"Covert budget: KL <= {model.cfg.covert_delta:.3e} (epsilon={model.cfg.covert_eps:.3f})")

    best_score = -1e18
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} Train", leave=True)

        sums = {k: 0.0 for k in ["loss", "img_rf", "meta_rf", "rf_total", "kl", "psnr", "mrmse", "cls", "acc"]}
        n_steps = 0
        accum_steps = max(1, int(cfg.get("accum_steps", 1)))
        n_batches_total = len(train_loader)
        optimizer.zero_grad(set_to_none=True)

        for bi, batch in enumerate(pbar):
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            meta = batch["meta"].to(device)
            img_wh = batch["img_wh"].to(device)
            cls_idx = batch["cls_idx"].to(device) if "cls_idx" in batch else None

            batch_snr = _sample_train_snr(train_snr_list, train_snr_probs, mode=str(cfg.get("train_snr_sampling", "uniform")))
            out = model.forward_batch(img, meta, snr_db=batch_snr, channel_type=str(cfg["bob_channel_type"]), cls_idx=cls_idx)

            loss_img_base, _ = image_loss(out["img_base"], img, mask, cfg)
            loss_img_rf, _ = image_loss(out["img_rf"], img, mask, cfg)
            loss_meta_phys = meta_phys_loss(out["meta_rf"], meta, cfg)
            if model.rf_net is not None:
                loss_rf, _ = refinement_objective(model, out)
            else:
                loss_rf = torch.tensor(0.0, device=device)

            ls = float(cfg.get("cls_label_smoothing", 0.05))
            mixup_alpha = float(cfg.get("cls_mixup_alpha", 0.0))
            if cls_idx is not None and out.get("cls_logits") is not None:
                acc_rf = classification_acc(out["cls_logits"], cls_idx)
                if mixup_alpha > 0.0 and model.classifier is not None and out.get("cls_feat_rf") is not None and cls_idx.numel() > 1 and not bool(cfg.get("cls_slot_enable", False)):
                    # manifold mixup on the pooled classifier feature
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(cls_idx.shape[0], device=device)
                    z = model.classifier.pooled(out["cls_feat_rf"], out.get("cls_extra_rf"))
                    z_mix = lam * z + (1.0 - lam) * z[perm]
                    logits_mix = model.classifier.classify_pooled(z_mix)
                    loss_cls_rf = (lam * classification_loss(logits_mix, cls_idx, class_weights, ls)
                                   + (1.0 - lam) * classification_loss(logits_mix, cls_idx[perm], class_weights, ls))
                else:
                    loss_cls_rf = classification_loss(out["cls_logits"], cls_idx, class_weights, ls)
                loss_cls_base = classification_loss(out.get("cls_logits_base"), cls_idx, class_weights, ls)
                loss_cls = loss_cls_rf + float(cfg.get("cls_aux_base_weight", 0.3)) * loss_cls_base
            else:
                loss_cls = torch.tensor(0.0, device=device)
                acc_rf = 0.0

            loss = (
                loss_img_rf
                + float(cfg["lambda_meta"]) * weighted_meta_loss_public(out["meta_rf"], meta, meta_weights)
                + float(cfg.get("lambda_meta_phys", 0.4)) * loss_meta_phys
                + float(cfg["lambda_rf"]) * loss_rf
                + float(cfg.get("base_loss_weight", 0.0)) * loss_img_base
                + float(cfg.get("lambda_cls", 0.7)) * loss_cls
            )
            loss_value = float(loss.item())
            (loss / float(accum_steps)).backward()
            do_step = ((bi + 1) % accum_steps == 0) or ((bi + 1) == n_batches_total)
            if do_step:
                if cfg.get("grad_clip_norm", None) is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            mr = compute_metrics_single_pred(img, out["img_rf"].detach(), meta, out["meta_rf"].detach(), mask, img_wh, cfg)
            sums["loss"] += loss_value
            sums["img_rf"] += float(loss_img_rf.item())
            sums["meta_rf"] += float(weighted_meta_rmse(out["meta_rf"], meta, meta_weights).item())
            sums["rf_total"] += float(loss_rf.item())
            sums["kl"] += float(out["kl"].mean().item())
            sums["psnr"] += float(mr["psnr"])
            sums["mrmse"] += float(mr["meta_rmse"])
            sums["cls"] += float(loss_cls.item())
            sums["acc"] += float(acc_rf)
            n_steps += 1

            pbar.set_postfix({
                "snr": f"{batch_snr:.1f}",
                "L": f"{sums['loss']/n_steps:.4f}",
                "PSNR": f"{sums['psnr']/n_steps:.2f}",
                "mRMSE": f"{sums['mrmse']/n_steps:.3f}",
                "Acc": f"{sums['acc']/n_steps:.3f}",
                "KL": f"{sums['kl']/n_steps:.3e}",
            })

        train_summary = {
            "loss": sums["loss"] / max(1, n_steps),
            "psnr": sums["psnr"] / max(1, n_steps),
            "meta_rmse": sums["mrmse"] / max(1, n_steps),
            "cert_kl": sums["kl"] / max(1, n_steps),
            "cls_loss": sums["cls"] / max(1, n_steps),
            "cls_acc": sums["acc"] / max(1, n_steps),
            "epoch_time_sec": time.time() - t0,
        }

        if epoch % int(cfg.get("val_every", 1)) == 0:
            val_agg, val_rows = evaluate_multi_snr(
                model,
                val_loader,
                device=device,
                snr_list=val_snr_list,
                snr_probs=val_snr_probs,
                bob_channel_type=str(cfg["bob_channel_type"]),
                cfg=cfg,
                max_batches=cfg.get("eval_max_batches", None),
                desc_prefix="Val",
            )
        else:
            val_agg = history[-1]["_val_agg"] if len(history) > 0 and "_val_agg" in history[-1] else {}
            val_rows = history[-1].get("_val_rows", []) if len(history) > 0 else []

        hist_row = {
            "epoch": epoch,
            "train_loss": train_summary["loss"],
            "train_psnr": train_summary["psnr"],
            "train_cls_acc": train_summary["cls_acc"],
            "val_base_psnr": val_agg.get("base_psnr", 0.0),
            "val_rf_psnr": val_agg.get("rf_psnr", 0.0),
            "val_rf_ssim": val_agg.get("rf_ssim", 0.0),
            "val_rf_meta_rmse": val_agg.get("rf_meta_rmse", 0.0),
            "val_rf_cls_acc": val_agg.get("rf_cls_acc", 0.0),
            "val_joint": val_agg.get("joint_score", -1e18),
            "_val_agg": val_agg,
            "_val_rows": val_rows,
        }
        history.append(hist_row)

        print(
            f"[Epoch {epoch}] loss={train_summary['loss']:.5f} | "
            f"val PSNR={val_agg.get('base_psnr', 0.0):.3f}/{val_agg.get('rf_psnr', 0.0):.3f} dB (no-RF/RF) | "
            f"SSIM={val_agg.get('rf_ssim', 0.0):.4f} | metaRMSE={val_agg.get('rf_meta_rmse', 0.0):.4f} | "
            f"clsAcc={val_agg.get('rf_cls_acc', 0.0):.3f} | KL={val_agg.get('cert_kl', 0.0):.3e} | "
            f"joint={val_agg.get('joint_score', -1e18):.4f}"
        )

        save_json({"train": train_summary, "val": val_agg, "val_rows": val_rows,
                   "history": [{k: v for k, v in h.items() if not k.startswith("_")} for h in history]},
                  os.path.join(exp_dir, "metrics_last.json"))

        if val_agg.get("joint_score", -1e18) > best_score:
            best_score = float(val_agg["joint_score"])
            save_checkpoint(os.path.join(exp_dir, "best.pt"), model, optimizer, epoch, cfg, best_score)
            save_json({"train": train_summary, "val": val_agg, "val_rows": val_rows},
                      os.path.join(exp_dir, "metrics_best.json"))
            print(f"[Epoch {epoch}] new best checkpoint (joint={best_score:.4f})")

        save_checkpoint(os.path.join(exp_dir, "last.pt"), model, optimizer, epoch, cfg, best_score)
        if bool(cfg.get("save_every_epoch_ckpt", False)):
            save_checkpoint(os.path.join(exp_dir, f"epoch_{epoch:03d}.pt"), model, optimizer, epoch, cfg, best_score)

        plot_history([{k: v for k, v in h.items() if not k.startswith("_")} for h in history], exp_dir)
