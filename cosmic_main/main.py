# -*- coding: utf-8 -*-
# Entry point. `python main.py --mode train` or `python main.py --mode test`.

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch


def build_default_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "mode": "train",
        "seed": 520,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exp_name": "cosmic",

        # paths
        "vod_root": os.path.join(".", "dataset", "VoD", "view_of_delft_PUBLIC"),
        "models_dir": os.path.join(".", "models"),
        "results_dir": os.path.join(".", "results"),

        # dataset
        "frame_rate": 10.0,
        "min_range": 2.0,
        "max_range": 80.0,
        "min_bbox_size": 12,
        "min_track_len": 4,
        "min_quality_score": 0.10,
        "wanted_classes": None,          # None = all VoD classes
        "canvas_hw": (192, 320),
        "meta_range_max": 80.0,
        "meta_vel_max": 20.0,
        "test_split": "val",
        "max_train_samples": 4000,       # raise for better accuracy, cost scales linearly
        "max_val_samples": 800,
        "max_test_samples": 1000,
        "eval_max_batches": None,
        "test_max_batches": None,
        "image_cache_size": 256,
        "bbox_mode": "tx",
        "roi_max_scale": 8.0,
        "class_balanced_train": True,
        "class_balanced_eval": True,
        "class_balanced_test": True,
        "augment_train": True,
        "aug_hflip": True,
        "aug_photometric": True,
        "aug_strength": 0.20,
        "aug_noise_std": 0.02,

        # classification head (kept to match the network in the paper;
        # accuracy is a side metric, not the focus)
        "cls_enable": True,
        "num_classes": 13,
        "class_list": None,
        "cls_hidden": 320,
        "cls_dropout": 0.35,
        "cls_blocks": 2,
        "cls_input_dropout": 0.10,
        "cls_feat_noise": 0.05,
        "cls_slot_enable": True,
        "cls_slot_repeats": 96,
        "cls_slot_theta_scale": 1.50,
        "cls_slot_temp": 8.0,
        "cls_mixup_alpha": 0.20,
        "cls_use_meta": True,
        "cls_detach_feat": False,
        "lambda_cls": 0.70,
        "cls_lr_mult": 1.0,
        "cls_weight_decay": 1e-3,
        "cls_aux_base_weight": 0.30,
        "cls_label_smoothing": 0.10,
        "cls_use_class_weights": False,
        "cls_weight_power": 0.50,
        "cls_weight_cap": 8.0,
        "model_select_cls_weight": 15.0,

        # waveform / covert budget
        "latent_down": 4,
        "pair_ch": 2,
        "num_chirps": 64,
        "spc": 240,
        "chirp_mu": 40.0,
        "alpha": 0.26,
        "ar_rho": 0.0,
        "covert_eps": 0.10,
        "covert_delta_override": -1.0,
        "tau_proj_iters": 28,
        "tau_floor": 1e-8,

        # receiver MAP inversion
        "rx_use_map": True,
        "rx_map_iters": 2,
        "rx_map_u_clip": 6.0,
        "rx_sigma_phi_min": 1e-4,
        "rx_sigma_phi_max": 4.0,

        # networks
        "enc_base": 80,
        "enc_blocks": 6,
        "dec_base": 224,
        "dec_blocks_lat": 6,
        "dec_blocks_img": 4,
        "dec_res_scale": 0.30,
        "tau_hidden": 128,
        "dec_use_rel": True,
        "dec_use_slot_mask": True,
        "dec_rel_scale_base": 1.00,
        "dec_rel_scale_rf": 0.12,

        # protected meta slots
        "meta_slot_repeats": [192, 224, 320, 320, 96, 96, 48, 48],
        "meta_theta_max": 1.70,
        "meta_theta_scales": [1.55, 1.85, 1.65, 1.65, 1.25, 1.25, 1.10, 1.10],
        "meta_refine_scale": 0.018,
        "vel_compand_beta": 1.60,
        "theta_compand_beta": 1.80,
        "meta_balanced_slots": True,

        # reliability channels
        "rel_use_logden": True,
        "rel_use_mag": True,
        "rel_use_logsnr": True,
        "rel_use_phi_resid": True,
        "rel_use_logtau": True,

        # RFlow refiner
        "rf_enable": True,
        "rf_steps": 2,
        "rf_ch": 96,
        "rf_blocks": 4,
        "rf_td": 96,
        "rf_train_steps": 5,
        "rf_path_gamma": 1.00,
        "rf_step_scale": 1.00,
        "rf_step_loss_weight": 1.00,
        "rf_endpoint_weight": 0.12,
        "rf_gate_floor": 0.10,
        "slot_refine_hidden": 256,
        "slot_refine_scale": 0.14,

        # Alice -> Bob channel
        "bob_channel_type": "rayleigh",
        "train_snr_dbs": [0.0, 20.0],
        "train_snr_sampling": "continuous_uniform",
        "train_snr_weights": None,
        "val_snr_dbs": [0.0, 10.0, 20.0],
        "val_snr_sampling": "uniform",
        "val_snr_weights": None,
        "test_snr_dbs": [0.0, 5.0, 10.0, 15.0, 20.0],

        # losses
        "img_l1_weight": 0.08,
        "img_ssim_weight": 0.06,
        "img_grad_weight": 0.02,
        "lambda_meta": 2.80,
        "lambda_meta_phys": 0.40,
        "phys_range_weight": 1.0,
        "phys_vel_weight": 1.0,
        "phys_angle_weight": 1.0,
        "lambda_rf": 0.36,
        "base_loss_weight": 0.01,
        "meta_loss_weights": [1.4, 2.4, 2.8, 2.8, 0.65, 0.65, 0.40, 0.40],
        "model_select_meta_weight": 4.0,

        # training
        "epochs": 30,
        "grad_checkpoint": True,
        "batch_size": 4,
        "accum_steps": 4,                # effective batch = 16
        "eval_batch_size": 4,
        "num_workers": 8,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "betas": (0.9, 0.999),
        "grad_clip_norm": 1.0,
        "val_every": 1,
        "save_every_epoch_ckpt": False,

        # radar (Alice side sensing, LoS at high SNR)
        "radar_probe_snr_db": 30.0,
        "radar_rx_ant": 8,
        "radar_angle_max_deg": 60.0,
        "radar_angle_grid": 241,
        "radar_range_os": 4,
        "radar_dop_os": 4,
        "radar_delay_min": 8,
        "radar_delay_max": 84,
        "radar_dop_min": -16,
        "radar_dop_max": 16,
        "radar_nms_delay": 3,
        "radar_nms_dop": 2,
        "radar_det_tol_delay": 1,
        "radar_det_tol_dop": 1,
        "radar_det_tol_angle": 3.0,
        "radar_amp_min": 0.55,
        "radar_amp_max": 1.0,

        # optional demo sequences saved during test (0 = off)
        "num_demo_sequences": 0,
        "demo_snr_db": 15.0,
        "demo_frames_per_sequence": 30,
        "demo_min_seq_len": 12,
        "demo_pool_topk": 40,
        "vis_alpha_range": 0.58,
        "vis_alpha_vel": 0.82,
        "vis_alpha_angle": 0.62,
        "vis_alpha_bbox": 0.90,
    }
    return cfg


def finalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from utils import ensure_dir

    exp_name = cfg["exp_name"]
    cfg["model_exp_dir"] = os.path.join(cfg["models_dir"], exp_name)
    cfg["result_exp_dir"] = os.path.join(cfg["results_dir"], exp_name)
    ensure_dir(cfg["models_dir"])
    ensure_dir(cfg["results_dir"])
    ensure_dir(cfg["model_exp_dir"])
    ensure_dir(cfg["result_exp_dir"])
    if not cfg.get("test_ckpt_path"):
        cfg["test_ckpt_path"] = os.path.join(cfg["model_exp_dir"], "best.pt")

    if cfg["mode"] not in ["train", "test"]:
        raise ValueError(f"mode must be 'train' or 'test', got {cfg['mode']}")
    if cfg["bob_channel_type"] not in ["rayleigh", "awgn"]:
        raise ValueError(f"bob_channel_type must be 'rayleigh' or 'awgn', got {cfg['bob_channel_type']}")

    reps = cfg.get("meta_slot_repeats", None)
    if reps is None or len(reps) != 8:
        raise ValueError("meta_slot_repeats must be a list of 8 ints")
    cfg["meta_slot_repeats"] = [int(r) for r in reps]
    ths = cfg.get("meta_theta_scales", None)
    if ths is None:
        cfg["meta_theta_scales"] = [float(cfg["meta_theta_max"])] * 8
    elif len(ths) != 8:
        raise ValueError("meta_theta_scales must have length 8")
    else:
        cfg["meta_theta_scales"] = [float(x) for x in ths]
    return cfg


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args_into(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="CoSMIC: covert semantic ISAC on View-of-Delft")
    p.add_argument("--mode", choices=["train", "test"], default=cfg["mode"])
    p.add_argument("--exp", type=str, default=cfg["exp_name"], help="experiment name")
    p.add_argument("--vod-root", type=str, default=cfg["vod_root"])
    p.add_argument("--epochs", type=int, default=cfg["epochs"])
    p.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    p.add_argument("--max-train-samples", type=int, default=cfg["max_train_samples"])
    p.add_argument("--snrs", type=float, nargs="+", default=None, help="test SNR list in dB")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint for test mode")
    p.add_argument("--demo", type=int, default=cfg["num_demo_sequences"], help="number of demo sequences to save in test mode")
    p.add_argument("--demo-snr", type=float, default=cfg["demo_snr_db"])
    args = p.parse_args()

    cfg["mode"] = args.mode
    cfg["exp_name"] = args.exp
    cfg["vod_root"] = args.vod_root
    cfg["epochs"] = int(args.epochs)
    cfg["batch_size"] = int(args.batch_size)
    cfg["max_train_samples"] = int(args.max_train_samples)
    if args.snrs is not None:
        cfg["test_snr_dbs"] = [float(s) for s in args.snrs]
    if args.ckpt is not None:
        cfg["test_ckpt_path"] = args.ckpt
    cfg["num_demo_sequences"] = int(args.demo)
    cfg["demo_snr_db"] = float(args.demo_snr)
    return cfg


if __name__ == "__main__":
    cfg = build_default_config()
    cfg = parse_args_into(cfg)
    cfg = finalize_config(cfg)
    set_global_seed(cfg["seed"])

    print("=" * 60)
    print("CoSMIC | covert semantic ISAC")
    print(f"mode={cfg['mode']} exp={cfg['exp_name']} device={cfg['device']}")
    print(f"channel={cfg['bob_channel_type']} eps={cfg['covert_eps']} test_snrs={cfg['test_snr_dbs']}")
    print("=" * 60)

    if cfg["mode"] == "train":
        from train import train
        train(cfg)
    else:
        from test import test
        test(cfg)
