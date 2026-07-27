# -*- coding: utf-8 -*-
# Test: PSNR / SSIM / sensing RMSE (+ cls acc) at each requested SNR.
# Optionally saves a few closed-loop demo sequences as png + gif.

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
from torch.utils.data import DataLoader

from dataset import VoDCovertISACDataset, vod_collate_fn
from modules import (
    CovertISACConfig,
    CovertISACVoDSystem,
    meta_to_radar_scene,
    radar_echo_ula_torch,
    rd_cube_torch,
    estimate_scene_from_rd,
)
from train import (
    load_checkpoint,
    build_model_cfg,
    evaluate,
    _as_float_list,
    compute_metrics_single_pred,
)
from utils import (
    seed_everything,
    ensure_dir,
    save_json,
    psnr_from_mse,
    denormalize_meta,
    crop_content_from_canvas,
    tensor_to_uint8_img,
    ssim_torch,
    ema_meta_split,
)


def build_system_from_ckpt(ckpt: Dict[str, Any], cfg: Dict[str, Any], device: torch.device) -> CovertISACVoDSystem:
    md = ckpt.get("model_cfg", None)
    if md is None:
        model_cfg = build_model_cfg(cfg)
    else:
        valid = {f.name for f in fields(CovertISACConfig)}
        md = {k: v for k, v in dict(md).items() if k in valid}
        md.setdefault("cls_enable", False)
        model_cfg = CovertISACConfig(**md)
        model_cfg.check()
    model = CovertISACVoDSystem(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


# ================================================================
# Closed loop helpers: probe the scene with radar, then transmit the
# estimated state as the semantic message of the next frame.
# ================================================================
def _neutral_probe_meta(meta: torch.Tensor) -> torch.Tensor:
    m = meta.clone()
    m[:, 0] = 0.0
    m[:, 1] = 0.0
    m[:, 2] = 0.0
    m[:, 3] = 1.0
    return m


def _scene_to_meta4(scene: Dict[str, torch.Tensor], cfg_m: CovertISACConfig, cfg: Dict[str, Any]) -> torch.Tensor:
    delay = scene["delay"].float()[:, 0]
    dop = scene["doppler"].float()[:, 0]
    ang_deg = scene["angle_deg"].float()[:, 0]

    range_m = (delay - float(cfg_m.radar_delay_min)) / max(1.0, float(cfg_m.radar_delay_max - cfg_m.radar_delay_min)) * float(cfg["meta_range_max"])
    vel_mps = (dop - float(cfg_m.radar_dop_min)) / max(1.0, float(cfg_m.radar_dop_max - cfg_m.radar_dop_min)) * (2.0 * float(cfg["meta_vel_max"])) - float(cfg["meta_vel_max"])
    ang_rad = ang_deg * math.pi / 180.0

    rn = 2.0 * (range_m / max(float(cfg["meta_range_max"]), 1e-6)) - 1.0
    vn = vel_mps / max(float(cfg["meta_vel_max"]), 1e-6)
    sin_t = torch.sin(ang_rad)
    cos_t = torch.cos(ang_rad)
    return torch.stack([torch.clamp(rn, -1.0, 1.0), torch.clamp(vn, -1.0, 1.0), sin_t, cos_t], dim=-1)


@torch.no_grad()
def closed_loop_frame(
    model: CovertISACVoDSystem,
    item: Dict[str, Any],
    device: torch.device,
    cfg: Dict[str, Any],
    snr_db: float,
) -> Dict[str, Any]:
    img = item["img"].unsqueeze(0).to(device)
    meta_gt = item["meta"].unsqueeze(0).to(device)

    # step 1: sensing-side probe at high radar SNR gives the tx meta
    meta_probe_seed = _neutral_probe_meta(meta_gt)
    tx_probe = model.transmitter(img, meta_probe_seed)
    scene_gt = meta_to_radar_scene(meta_gt, model.cfg, cfg["meta_range_max"], cfg["meta_vel_max"])
    y_probe = radar_echo_ula_torch(tx_probe["x"], scene_gt, float(cfg["radar_probe_snr_db"]), model.cfg)
    rd_probe = rd_cube_torch(y_probe, tx_probe["x"], model.cfg)
    est_scene, _ = estimate_scene_from_rd(rd_probe, model.cfg)
    meta_tx = meta_gt.clone()
    meta_tx[:, :4] = _scene_to_meta4(est_scene, model.cfg, cfg)

    # step 2: covert transmission of image + estimated state to Bob
    cls_idx = item.get("cls_idx", None)
    if cls_idx is not None:
        cls_idx = cls_idx.reshape(1).to(device)
    out = model.forward_batch(img, meta_tx, snr_db=float(snr_db), channel_type=str(cfg["bob_channel_type"]), cls_idx=cls_idx)
    return {"out": out, "meta_tx": meta_tx, "meta_gt": meta_gt, "img": img}


def _pick_window(dataset: VoDCovertISACDataset, seq: List[int], num_keep: int) -> List[int]:
    # contiguous window of consecutive frames, placed at the best quality stretch
    if num_keep is None or num_keep <= 0 or num_keep >= len(seq):
        return list(seq)
    quals = np.asarray([float(dataset.samples[si].quality) for si in seq], dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(quals)])
    means = (csum[num_keep:] - csum[:-num_keep]) / float(num_keep)
    start = int(np.argmax(means))
    return list(seq[start:start + num_keep])


def _clip_box(box, W, H):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(W), x2))
    y2 = max(y1 + 1.0, min(float(H), y2))
    return x1, y1, x2, y2


def _paste_roi(bg_uint8: np.ndarray, roi_uint8: np.ndarray, bbox_xyxy) -> np.ndarray:
    out = bg_uint8.copy()
    H, W = out.shape[0], out.shape[1]
    x1, y1, x2, y2 = _clip_box(bbox_xyxy, W, H)
    x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if x2i <= x1i + 1 or y2i <= y1i + 1:
        return out
    roi_rs = np.asarray(Image.fromarray(roi_uint8).resize((x2i - x1i, y2i - y1i), Image.BILINEAR))
    out[y1i:y2i, x1i:x2i, :] = roi_rs
    return out


def _recover_roi(canvas_hat_uint8: np.ndarray, item: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    roi_wh = tuple(int(x) for x in item["roi_wh"].tolist())
    content = crop_content_from_canvas(canvas_hat_uint8, roi_wh=roi_wh, canvas_hw=tuple(cfg["canvas_hw"]), fit_info=item["fit_info"])
    return np.asarray(Image.fromarray(content).resize(roi_wh, Image.BILINEAR))


def _render_demo_frame(
    save_path: str,
    tx_full: np.ndarray,
    roi_gt: np.ndarray,
    roi_hat: np.ndarray,
    tx_bbox,
    bbox_hat,
    cls_name: str,
    cls_bob: str,
    frame_str: str,
    gt_info: Dict[str, Any],
    hat_info: Dict[str, Any],
    psnr_roi: float,
    ssim_roi: float,
    snr_db: float,
) -> None:
    ensure_dir(os.path.dirname(save_path))
    bob_full = _paste_roi(tx_full.copy(), roi_hat, bbox_hat)
    tx1, ty1, tx2, ty2 = [float(v) for v in tx_bbox]
    bx1, by1, bx2, by2 = [float(v) for v in bbox_hat]

    fig = plt.figure(figsize=(14.2, 7.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.12, 0.66], height_ratios=[1.0, 0.80], wspace=0.02, hspace=0.05)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_bob = fig.add_subplot(gs[0, 1])
    ax_rg = fig.add_subplot(gs[1, 0])
    ax_rh = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[:, 2])
    for ax in [ax_gt, ax_bob, ax_rg, ax_rh, ax_txt]:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax_gt.imshow(tx_full)
    ax_gt.add_patch(patches.Rectangle((tx1, ty1), tx2 - tx1, ty2 - ty1, lw=2.0, edgecolor="#2e8b57", facecolor="none"))
    ax_gt.set_title("Ground truth", fontsize=12, pad=5)

    ax_bob.imshow(bob_full)
    ax_bob.add_patch(patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, lw=2.0, edgecolor="#1f77b4", facecolor="none"))
    ax_bob.set_title(f"Bob recovery @ {snr_db:.0f} dB", fontsize=12, pad=5)

    ax_rg.imshow(roi_gt)
    ax_rg.set_title("Target (GT)", fontsize=11, pad=4)
    ax_rh.imshow(roi_hat)
    ax_rh.set_title(f"Target (Bob)  PSNR {psnr_roi:.2f} dB  SSIM {ssim_roi:.3f}", fontsize=11, pad=4)

    dth = (hat_info["theta_deg"] - gt_info["theta_deg"] + 180.0) % 360.0 - 180.0
    lines = [
        f"class  GT  {cls_name}",
        f"       Bob {cls_bob}",
        f"frame  {frame_str}",
        "",
        "         GT        Bob",
        f"range  {gt_info['range_m']:7.2f} m {hat_info['range_m']:7.2f} m",
        f"speed  {gt_info['v_rad_mps']:7.2f}   {hat_info['v_rad_mps']:7.2f} m/s",
        f"angle  {gt_info['theta_deg']:7.2f}   {hat_info['theta_deg']:7.2f} deg",
        "",
        "err",
        f"  dR   {hat_info['range_m'] - gt_info['range_m']:+.2f} m",
        f"  dV   {hat_info['v_rad_mps'] - gt_info['v_rad_mps']:+.2f} m/s",
        f"  dA   {dth:+.2f} deg",
    ]
    ax_txt.text(0.05, 0.97, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")

    fig.subplots_adjust(left=0.003, right=0.997, top=0.99, bottom=0.006)
    fig.savefig(save_path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _compile_gif(seq_dir: str, duration_ms: int = 140) -> None:
    frame_paths = [os.path.join(seq_dir, n) for n in sorted(os.listdir(seq_dir)) if n.lower().endswith(".png")]
    if len(frame_paths) == 0:
        return
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    base = frames[0].convert("P", palette=Image.ADAPTIVE, colors=255)
    pal_frames = [base] + [f.quantize(palette=base) for f in frames[1:]]
    pal_frames[0].save(
        os.path.join(seq_dir, "sequence.gif"),
        save_all=True,
        append_images=pal_frames[1:],
        duration=int(duration_ms),
        loop=0,
    )


@torch.no_grad()
def save_demo_sequences(
    model: CovertISACVoDSystem,
    dataset: VoDCovertISACDataset,
    device: torch.device,
    cfg: Dict[str, Any],
    out_dir: str,
    snr_db: float,
    num_sequences: int,
) -> None:
    ensure_dir(out_dir)
    model.eval()
    seqs = dataset.get_sequences(min_len=int(cfg.get("demo_min_seq_len", 12)), top_k=int(cfg.get("demo_pool_topk", 40)))
    if len(seqs) == 0:
        print("No sequences long enough for the demo.")
        return
    chosen = seqs[: int(num_sequences)]
    frames_keep = int(cfg.get("demo_frames_per_sequence", 30))

    for rank, seq in enumerate(chosen):
        first = dataset[seq[0]]
        seq_dir = os.path.join(out_dir, f"seq{rank:02d}_{first['cls']}_tid{first['track_id']}")
        ensure_dir(seq_dir)
        kept = _pick_window(dataset, seq, frames_keep)
        ema_pred = None
        for j, sample_idx in enumerate(kept):
            item = dataset[sample_idx]
            res = closed_loop_frame(model, item, device, cfg, snr_db)
            out = res["out"]
            class_list = cfg.get("class_list", None)
            if out.get("cls_logits") is not None and class_list:
                cls_bob = str(class_list[int(torch.argmax(out["cls_logits"][0]).item())])
            else:
                cls_bob = str(item["cls"])
            meta_hat = out["meta_rf"][0].detach().cpu().numpy()
            ema_pred = ema_meta_split(
                ema_pred, meta_hat,
                alpha_range=float(cfg.get("vis_alpha_range", 0.58)),
                alpha_vel=float(cfg.get("vis_alpha_vel", 0.82)),
                alpha_angle=float(cfg.get("vis_alpha_angle", 0.62)),
                alpha_bbox=float(cfg.get("vis_alpha_bbox", 0.90)),
            )
            img_wh = tuple(int(x) for x in item["img_wh"].tolist())
            gt_info = denormalize_meta(item["meta"].numpy(), img_wh=img_wh, range_max=cfg["meta_range_max"], vel_max=cfg["meta_vel_max"])
            hat_info = denormalize_meta(ema_pred, img_wh=img_wh, range_max=cfg["meta_range_max"], vel_max=cfg["meta_vel_max"])

            canvas_hat = tensor_to_uint8_img(out["img_rf"][0])
            roi_gt = item["roi_uint8"]
            roi_hat = _recover_roi(canvas_hat, item, cfg)
            gt_f = roi_gt.astype(np.float32) / 255.0
            hat_f = roi_hat.astype(np.float32) / 255.0
            psnr_roi = psnr_from_mse(float(np.mean((gt_f - hat_f) ** 2)))
            ssim_roi = float(ssim_torch(
                torch.from_numpy(hat_f.transpose(2, 0, 1)).unsqueeze(0),
                torch.from_numpy(gt_f.transpose(2, 0, 1)).unsqueeze(0),
            ).item())

            _render_demo_frame(
                os.path.join(seq_dir, f"{j:04d}_tx{item['tx_frame_str']}.png"),
                item["tx_full_img_uint8"],
                roi_gt,
                roi_hat,
                tuple(float(x) for x in item["tx_bbox_xyxy"].tolist()),
                hat_info["bbox_xyxy"],
                item["cls"],
                cls_bob,
                item["tx_frame_str"],
                gt_info,
                hat_info,
                psnr_roi,
                ssim_roi,
                float(snr_db),
            )
        _compile_gif(seq_dir)
        print(f"Saved demo sequence: {seq_dir}")


# ================================================================
# Main test entry
# ================================================================
def test(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    ckpt_path = cfg["test_ckpt_path"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_system_from_ckpt(ckpt, cfg, device)

    # use the class vocabulary the checkpoint was trained with
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg", {}), dict) else {}
    class_list = ckpt_cfg.get("class_list", cfg.get("class_list", None))
    cfg["class_list"] = class_list
    cfg["num_classes"] = int(getattr(model.cfg, "num_classes", len(class_list) if class_list else 0))

    demo_n = int(cfg.get("num_demo_sequences", 0))
    test_set = VoDCovertISACDataset(
        vod_root=cfg["vod_root"],
        split=cfg["test_split"],
        canvas_hw=tuple(cfg["canvas_hw"]),
        meta_range_max=cfg["meta_range_max"],
        meta_vel_max=cfg["meta_vel_max"],
        frame_rate=cfg["frame_rate"],
        min_range=cfg["min_range"],
        max_range=cfg["max_range"],
        min_bbox_size=cfg["min_bbox_size"],
        wanted_classes=cfg["wanted_classes"],
        max_samples=cfg.get("max_test_samples", None),
        min_track_len=cfg.get("min_track_len", 3),
        min_quality_score=cfg.get("min_quality_score", 0.0),
        sort_by_quality=True,
        image_cache_size=int(cfg.get("image_cache_size", 256)),
        return_full_image=(demo_n > 0),
        return_tx_image=(demo_n > 0),
        return_raw_roi=(demo_n > 0),
        bbox_mode=str(cfg.get("bbox_mode", "tx")),
        roi_max_scale=float(cfg.get("roi_max_scale", 8.0)),
        class_list=class_list,
        class_balanced_selection=bool(cfg.get("class_balanced_test", True)),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=vod_collate_fn,
    )

    out_root = os.path.join(cfg["results_dir"], cfg["exp_name"], f"test_{os.path.splitext(os.path.basename(ckpt_path))[0]}")
    ensure_dir(out_root)
    save_json({"cfg": {k: v for k, v in cfg.items() if not callable(v)}, "model_cfg": asdict(model.cfg), "ckpt": ckpt_path},
              os.path.join(out_root, "run_info.json"))

    rows: List[Dict[str, float]] = []
    for snr_db in _as_float_list(cfg["test_snr_dbs"]):
        m = evaluate(
            model,
            test_loader,
            device=device,
            snr_db=float(snr_db),
            bob_channel_type=str(cfg["bob_channel_type"]),
            cfg=cfg,
            max_batches=cfg.get("test_max_batches", None),
            desc=f"Test@{snr_db}dB",
        )
        row = {
            "snr_db": float(snr_db),
            "psnr": m["rf_psnr"],
            "ssim": m["rf_ssim"],
            "range_rmse_m": m["rf_range_rmse_m"],
            "vel_rmse_mps": m["rf_vel_rmse_mps"],
            "angle_rmse_deg": m["rf_angle_rmse_deg"],
            "cls_acc": m.get("rf_cls_acc", 0.0),
            "psnr_no_rflow": m["base_psnr"],
            "ssim_no_rflow": m["base_ssim"],
            "cert_kl": m["cert_kl"],
        }
        rows.append(row)
        print(
            f"[Test {snr_db:>5.1f} dB] PSNR={row['psnr']:.2f} dB | SSIM={row['ssim']:.4f} | "
            f"RMSE range/vel/angle = {row['range_rmse_m']:.2f} m / {row['vel_rmse_mps']:.2f} m/s / {row['angle_rmse_deg']:.2f} deg | "
            f"clsAcc={row['cls_acc']:.3f} | KL={row['cert_kl']:.3e}"
        )

    save_json(rows, os.path.join(out_root, "results_vs_snr.json"))
    csv_path = os.path.join(out_root, "results_vs_snr.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {out_root}")

    if demo_n > 0:
        save_demo_sequences(
            model,
            test_set,
            device,
            cfg,
            os.path.join(out_root, "demo_sequences"),
            snr_db=float(cfg.get("demo_snr_db", 15.0)),
            num_sequences=demo_n,
        )
