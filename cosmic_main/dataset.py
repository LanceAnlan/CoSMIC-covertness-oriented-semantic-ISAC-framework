# -*- coding: utf-8 -*-
# View-of-Delft loader. Each sample follows the sense-then-transmit frame logic:
# the target state sensed at frame k-1 becomes the semantic message of frame k.

from __future__ import annotations

import glob
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils import crop_bbox_xyxy, fit_to_canvas_keep_ar, normalize_meta, safe_arctan2


# VoD object classes (DontCare excluded), ordered by frequency. The index order
# is stored in the checkpoint, so don't reorder without retraining.
VOD_DEFAULT_CLASSES: List[str] = [
    "Pedestrian",
    "Car",
    "bicycle",
    "rider",
    "Cyclist",
    "bicycle_rack",
    "moped_scooter",
    "ride_other",
    "motor",
    "human_depiction",
    "vehicle_other",
    "truck",
    "ride_uncertain",
]


def build_class_to_idx(class_list: Optional[List[str]] = None) -> Dict[str, int]:
    cls = list(class_list) if class_list is not None else list(VOD_DEFAULT_CLASSES)
    return {c: i for i, c in enumerate(cls)}


@dataclass(frozen=True)
class TrackObject:
    cls: str
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    pos_cam: Tuple[float, float, float]
    truncation: float = 0.0
    occlusion: float = 0.0


@dataclass
class SampleItem:
    seq_id: int
    sample_pos: int
    tx_frame_id: int
    tx_frame_str: str
    src_frame_id: int
    src_frame_str: str
    prv_frame_id: int
    prv_frame_str: str
    track_id: int
    cls: str

    bbox_src_xyxy: Tuple[float, float, float, float]
    bbox_tx_xyxy: Tuple[float, float, float, float]

    pos_prv: Tuple[float, float, float]
    pos_src: Tuple[float, float, float]

    range_m: float
    v_rad_mps: float
    theta_rad: float

    src_image_path: str
    tx_image_path: str

    truncation: float
    occlusion: float
    bbox_area: float
    quality: float
    track_len: int


def _read_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def read_split_ids(vod_root: str, split: str) -> List[str]:
    split_path = os.path.join(vod_root, "lidar", "ImageSets", f"{split}.txt")
    return _read_lines(split_path)


def frame_str_to_int(frame_str: str) -> int:
    return int(frame_str)


def group_consecutive(frame_ids: List[int]) -> List[List[int]]:
    if len(frame_ids) == 0:
        return []
    frame_ids = sorted(frame_ids)
    groups: List[List[int]] = []
    cur = [frame_ids[0]]
    for fid in frame_ids[1:]:
        if fid == cur[-1] + 1:
            cur.append(fid)
        else:
            groups.append(cur)
            cur = [fid]
    groups.append(cur)
    return groups


def parse_label_file(label_path: str) -> List[TrackObject]:
    # KITTI-style label_2 files; bbox at tokens 4:8, camera position at 11:14
    objects: List[TrackObject] = []
    if not os.path.isfile(label_path):
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 15:
                continue
            cls = toks[0]
            if cls.lower() == "dontcare":
                continue
            try:
                track_id = int(float(toks[1]))
            except Exception:
                continue
            bbox = tuple(float(x) for x in toks[4:8])
            pos = tuple(float(x) for x in toks[11:14])
            trunc = 0.0
            occ = 0.0
            if len(toks) > 2:
                try:
                    trunc = float(toks[2])
                except Exception:
                    trunc = 0.0
            if len(toks) > 3:
                try:
                    occ = float(toks[3])
                except Exception:
                    occ = 0.0
            objects.append(
                TrackObject(
                    cls=cls,
                    track_id=track_id,
                    bbox_xyxy=bbox,
                    pos_cam=pos,
                    truncation=float(trunc),
                    occlusion=float(occ),
                )
            )
    return objects


def find_image_path(image_dir: str, frame_str: str) -> Optional[str]:
    jpg = os.path.join(image_dir, frame_str + ".jpg")
    png = os.path.join(image_dir, frame_str + ".png")
    if os.path.isfile(jpg):
        return jpg
    if os.path.isfile(png):
        return png
    cand = glob.glob(os.path.join(image_dir, frame_str + ".*"))
    return cand[0] if len(cand) > 0 else None


class VoDCovertISACDataset(Dataset):
    def __init__(
        self,
        vod_root: str,
        split: str = "train",
        canvas_hw: Tuple[int, int] = (192, 320),
        meta_range_max: float = 80.0,
        meta_vel_max: float = 20.0,
        frame_rate: float = 10.0,
        min_range: float = 2.0,
        max_range: float = 80.0,
        min_bbox_size: int = 12,
        wanted_classes: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        min_track_len: int = 3,
        min_quality_score: float = 0.0,
        sort_by_quality: bool = True,
        image_cache_size: int = 256,
        return_full_image: bool = False,
        return_tx_image: bool = False,
        return_raw_roi: bool = False,
        bbox_mode: str = "tx",
        roi_max_scale: float = 8.0,
        class_list: Optional[List[str]] = None,
        class_balanced_selection: bool = False,
        augment: bool = False,
        aug_hflip: bool = True,
        aug_photometric: bool = True,
        aug_strength: float = 0.20,
        aug_noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.vod_root = vod_root
        self.split = split
        self.canvas_hw = tuple(int(x) for x in canvas_hw)
        self.meta_range_max = float(meta_range_max)
        self.meta_vel_max = float(meta_vel_max)
        self.frame_rate = float(frame_rate)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.min_bbox_size = int(min_bbox_size)
        self.min_track_len = int(min_track_len)
        self.min_quality_score = float(min_quality_score)
        self.sort_by_quality = bool(sort_by_quality)
        self.image_cache_size = int(max(0, image_cache_size))
        self.return_full_image = bool(return_full_image)
        self.return_tx_image = bool(return_tx_image)
        self.return_raw_roi = bool(return_raw_roi)
        self.bbox_mode = str(bbox_mode).lower().strip()
        if self.bbox_mode not in {"src", "tx"}:
            raise ValueError(f"bbox_mode must be 'src' or 'tx', got {bbox_mode}")
        self.roi_max_scale = float(roi_max_scale)
        self.wanted_classes = None if wanted_classes is None else set(wanted_classes)
        self.class_balanced_selection = bool(class_balanced_selection)
        self.augment = bool(augment)
        self.aug_hflip = bool(aug_hflip)
        self.aug_photometric = bool(aug_photometric)
        self.aug_strength = float(aug_strength)
        self.aug_noise_std = float(aug_noise_std)
        # class_list, when given, is the checkpoint's vocabulary and wins
        self.class_list: List[str] = list(class_list) if class_list is not None else list(VOD_DEFAULT_CLASSES)
        self.class_to_idx: Dict[str, int] = build_class_to_idx(self.class_list)
        self.num_classes: int = len(self.class_list)
        self._image_cache: OrderedDict[str, Image.Image] = OrderedDict()

        self.image_dir = os.path.join(self.vod_root, "lidar", "training", "image_2")
        self.label_dir = os.path.join(self.vod_root, "lidar", "training", "label_2")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"image_2 directory not found: {self.image_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"label_2 directory not found: {self.label_dir}")

        frame_strs = read_split_ids(self.vod_root, split)
        frame_ids = [frame_str_to_int(s) for s in frame_strs]
        self._frame_id_to_str = {frame_str_to_int(s): s for s in frame_strs}
        groups = group_consecutive(frame_ids)

        self._frame_objects: Dict[int, Dict[int, TrackObject]] = {}
        self._frame_img_paths: Dict[int, str] = {}
        self.full_img_wh: Tuple[int, int] = (1, 1)
        for fid in frame_ids:
            fstr = self._frame_id_to_str[fid]
            label_path = os.path.join(self.label_dir, fstr + ".txt")
            img_path = find_image_path(self.image_dir, fstr)
            if img_path is None:
                continue
            self._frame_img_paths[fid] = img_path
            if self.full_img_wh == (1, 1):
                with Image.open(img_path) as im0:
                    self.full_img_wh = im0.size
            objs = parse_label_file(label_path)
            obj_map: Dict[int, TrackObject] = {}
            for obj in objs:
                if self.wanted_classes is not None and obj.cls not in self.wanted_classes:
                    continue
                if obj.track_id not in obj_map:
                    obj_map[obj.track_id] = obj
            self._frame_objects[fid] = obj_map

        temp_samples: List[SampleItem] = []
        dt = 1.0 / self.frame_rate

        # need three consecutive frames: prv/src give the sensed state, tx is transmitted
        for seq_id, grp in enumerate(groups):
            if len(grp) < 3:
                continue
            for i in range(2, len(grp)):
                fid_prv = grp[i - 2]
                fid_src = grp[i - 1]
                fid_tx = grp[i]
                if fid_prv not in self._frame_img_paths or fid_src not in self._frame_img_paths or fid_tx not in self._frame_img_paths:
                    continue

                common_tracks = (
                    set(self._frame_objects.get(fid_prv, {}).keys())
                    & set(self._frame_objects.get(fid_src, {}).keys())
                    & set(self._frame_objects.get(fid_tx, {}).keys())
                )
                if not common_tracks:
                    continue

                for tid in common_tracks:
                    obj_prv = self._frame_objects[fid_prv][tid]
                    obj_src = self._frame_objects[fid_src][tid]
                    obj_tx = self._frame_objects[fid_tx][tid]

                    x1, y1, x2, y2 = obj_src.bbox_xyxy
                    bw = max(float(x2 - x1), 0.0)
                    bh = max(float(y2 - y1), 0.0)
                    if bw < self.min_bbox_size or bh < self.min_bbox_size:
                        continue

                    pos_prv = np.array(obj_prv.pos_cam, dtype=np.float64)
                    pos_src = np.array(obj_src.pos_cam, dtype=np.float64)
                    range_m = float(np.linalg.norm(pos_src))
                    if not (self.min_range <= range_m <= self.max_range):
                        continue

                    v_vec = (pos_src - pos_prv) / dt
                    los = pos_src / (range_m + 1e-12)
                    v_rad = float(np.dot(v_vec, los))
                    theta = float(safe_arctan2(pos_src[0], pos_src[2]))

                    temp_samples.append(
                        SampleItem(
                            seq_id=seq_id,
                            sample_pos=len(temp_samples),
                            tx_frame_id=fid_tx,
                            tx_frame_str=self._frame_id_to_str[fid_tx],
                            src_frame_id=fid_src,
                            src_frame_str=self._frame_id_to_str[fid_src],
                            prv_frame_id=fid_prv,
                            prv_frame_str=self._frame_id_to_str[fid_prv],
                            track_id=tid,
                            cls=obj_src.cls,
                            bbox_src_xyxy=obj_src.bbox_xyxy,
                            bbox_tx_xyxy=obj_tx.bbox_xyxy,
                            pos_prv=tuple(pos_prv.tolist()),
                            pos_src=tuple(pos_src.tolist()),
                            range_m=range_m,
                            v_rad_mps=v_rad,
                            theta_rad=theta,
                            src_image_path=self._frame_img_paths[fid_src],
                            tx_image_path=self._frame_img_paths[fid_tx],
                            truncation=float(obj_src.truncation),
                            occlusion=float(obj_src.occlusion),
                            bbox_area=float(bw * bh),
                            quality=0.0,
                            track_len=0,
                        )
                    )

        if len(temp_samples) == 0:
            raise RuntimeError(
                f"No valid VoD covert-ISAC samples for split={split}. "
                f"Check ImageSets, labels, classes, and filtering thresholds."
            )

        by_track: Dict[Tuple[int, int], List[int]] = {}
        for idx, s in enumerate(temp_samples):
            by_track.setdefault((s.seq_id, s.track_id), []).append(idx)
        track_len_map = {k: len(v) for k, v in by_track.items()}

        samples: List[SampleItem] = []
        W, H = float(self.full_img_wh[0]), float(self.full_img_wh[1])
        for s in temp_samples:
            if s.cls not in self.class_to_idx:
                continue
            tlen = int(track_len_map[(s.seq_id, s.track_id)])
            if tlen < self.min_track_len:
                continue
            qual = self._quality_score(s, img_wh=(W, H), track_len=tlen)
            if qual < self.min_quality_score:
                continue
            s.track_len = tlen
            s.quality = qual
            samples.append(s)

        if len(samples) == 0:
            raise RuntimeError(
                f"All samples were filtered out for split={split}. "
                f"Relax min_track_len/min_quality_score/min_bbox_size or range filters."
            )

        if self.sort_by_quality:
            samples.sort(key=lambda z: (z.quality, z.bbox_area, z.track_len), reverse=True)
        if max_samples is not None and len(samples) > int(max_samples):
            if self.class_balanced_selection:
                samples = self._class_balanced_pick(samples, int(max_samples))
            else:
                samples = samples[: int(max_samples)]

        for i, s in enumerate(samples):
            s.sample_pos = i
        self.samples = samples
        self._sequences = self._build_track_sequences(min_len=max(5, self.min_track_len))

    def _class_balanced_pick(self, samples: List[SampleItem], max_samples: int) -> List[SampleItem]:
        # round-robin over classes so the rare ones survive the truncation
        by_cls: "OrderedDict[str, List[SampleItem]]" = OrderedDict()
        for s in samples:
            by_cls.setdefault(s.cls, []).append(s)
        order = list(by_cls.keys())
        ptr = {c: 0 for c in order}
        picked: List[SampleItem] = []
        while len(picked) < max_samples:
            progressed = False
            for c in order:
                if ptr[c] < len(by_cls[c]):
                    picked.append(by_cls[c][ptr[c]])
                    ptr[c] += 1
                    progressed = True
                    if len(picked) >= max_samples:
                        break
            if not progressed:
                break
        picked.sort(key=lambda z: (z.quality, z.bbox_area, z.track_len), reverse=True)
        return picked

    def _quality_score(self, s: SampleItem, img_wh: Tuple[float, float], track_len: int) -> float:
        W, H = img_wh
        x1, y1, x2, y2 = s.bbox_src_xyxy
        bw = max(float(x2 - x1), 1.0)
        bh = max(float(y2 - y1), 1.0)
        area = bw * bh
        cx = 0.5 * (x1 + x2) / max(W, 1.0)
        cy = 0.5 * (y1 + y2) / max(H, 1.0)
        center_dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

        area_term = np.clip((math.sqrt(area) - float(self.min_bbox_size)) / 120.0, 0.0, 1.0)
        range_term = math.exp(-float(s.range_m) / max(self.meta_range_max * 0.75, 1.0))
        center_term = float(np.clip(1.0 - center_dist / 0.75, 0.0, 1.0))
        track_term = float(np.clip((track_len - self.min_track_len) / 10.0, 0.0, 1.0))
        trunc_pen = math.exp(-1.25 * max(0.0, float(s.truncation)))
        occ_pen = math.exp(-0.40 * max(0.0, float(s.occlusion)))

        quality = (0.48 * area_term + 0.24 * range_term + 0.18 * center_term + 0.10 * track_term) * trunc_pen * occ_pen
        return float(quality)

    def _build_track_sequences(self, min_len: int = 5) -> List[List[int]]:
        by_key: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for idx, s in enumerate(self.samples):
            key = (s.seq_id, s.track_id)
            by_key.setdefault(key, []).append((s.tx_frame_id, idx))

        seqs: List[List[int]] = []
        for _, lst in by_key.items():
            lst = sorted(lst, key=lambda z: z[0])
            cur = [lst[0][1]]
            for (fid, idx), (fid_prev, _) in zip(lst[1:], lst[:-1]):
                if fid == fid_prev + 1:
                    cur.append(idx)
                else:
                    if len(cur) >= min_len:
                        seqs.append(cur)
                    cur = [idx]
            if len(cur) >= min_len:
                seqs.append(cur)

        def seq_score(seq: List[int]) -> Tuple[float, float, float]:
            qs = [self.samples[i].quality for i in seq]
            areas = [self.samples[i].bbox_area for i in seq]
            return (float(np.mean(qs)), float(np.mean(areas)), float(len(seq)))

        seqs.sort(key=seq_score, reverse=True)
        return seqs

    def get_sequences(self, min_len: int = 5, top_k: Optional[int] = None) -> List[List[int]]:
        if min_len <= max(5, self.min_track_len):
            seqs = self._sequences
        else:
            seqs = self._build_track_sequences(min_len=min_len)
        if top_k is not None and len(seqs) > int(top_k):
            return seqs[: int(top_k)]
        return seqs

    def __len__(self) -> int:
        return len(self.samples)

    def _load_img(self, path: str) -> Image.Image:
        if self.image_cache_size > 0 and path in self._image_cache:
            img = self._image_cache.pop(path)
            self._image_cache[path] = img
            return img.copy()
        img = Image.open(path).convert("RGB")
        if self.image_cache_size > 0:
            self._image_cache[path] = img.copy()
            while len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return img

    def _augment(self, canvas_t: torch.Tensor, mask_t: torch.Tensor, meta_t: torch.Tensor):
        # photometric jitter keeps labels; hflip negates sin(theta) and centre-x
        def u(a: float, b: float) -> float:
            return a + (b - a) * float(torch.rand(1).item())

        if self.aug_photometric:
            s = self.aug_strength
            canvas_t = canvas_t * (1.0 + u(-s, s))
            mean = canvas_t.mean(dim=(1, 2), keepdim=True)
            canvas_t = (canvas_t - mean) * (1.0 + u(-s, s)) + mean
            gray = canvas_t.mean(dim=0, keepdim=True)
            canvas_t = (canvas_t - gray) * (1.0 + u(-s, s)) + gray
            if float(torch.rand(1).item()) < 0.5 and self.aug_noise_std > 0:
                canvas_t = canvas_t + torch.randn_like(canvas_t) * self.aug_noise_std
            canvas_t = torch.clamp(canvas_t, 0.0, 1.0) * mask_t

        if self.aug_hflip and float(torch.rand(1).item()) < 0.5:
            canvas_t = torch.flip(canvas_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[2])
            meta_t = meta_t.clone()
            meta_t[2] = -meta_t[2]
            meta_t[4] = -meta_t[4]
        return canvas_t, mask_t, meta_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        src_img = self._load_img(s.src_image_path)
        tx_img = self._load_img(s.tx_image_path)
        W, H = src_img.size

        bbox_src_xyxy = s.bbox_src_xyxy
        bbox_tx_xyxy = s.bbox_tx_xyxy
        bbox_meta_xyxy = bbox_tx_xyxy if self.bbox_mode == "tx" else bbox_src_xyxy
        roi = crop_bbox_xyxy(src_img, bbox_src_xyxy)
        roi_w, roi_h = roi.size

        canvas_img, fit_info = fit_to_canvas_keep_ar(roi, self.canvas_hw, max_scale=self.roi_max_scale)
        Hc, Wc = self.canvas_hw
        mask = np.zeros((Hc, Wc), dtype=np.float32)
        pl = int(fit_info["pad_left"])
        pt = int(fit_info["pad_top"])
        nw = int(fit_info["new_w"])
        nh = int(fit_info["new_h"])
        mask[pt:pt + nh, pl:pl + nw] = 1.0

        canvas_np = np.asarray(canvas_img, dtype=np.float32) / 255.0
        canvas_t = torch.from_numpy(canvas_np).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        meta = normalize_meta(
            range_m=s.range_m,
            v_rad_mps=s.v_rad_mps,
            theta_rad=s.theta_rad,
            bbox_xyxy=bbox_meta_xyxy,
            img_wh=(W, H),
            range_max=self.meta_range_max,
            vel_max=self.meta_vel_max,
        )
        meta_t = torch.from_numpy(meta.astype(np.float32))

        if self.augment:
            canvas_t, mask_t, meta_t = self._augment(canvas_t, mask_t, meta_t)

        out: Dict[str, Any] = {
            "img": canvas_t,
            "mask": mask_t,
            "meta": meta_t,
            "cls": s.cls,
            "cls_idx": torch.tensor(int(self.class_to_idx[s.cls]), dtype=torch.long),
            "track_id": s.track_id,
            "seq_id": s.seq_id,
            "sample_pos": s.sample_pos,
            "src_frame_id": s.src_frame_id,
            "src_frame_str": s.src_frame_str,
            "tx_frame_id": s.tx_frame_id,
            "tx_frame_str": s.tx_frame_str,
            "prv_frame_id": s.prv_frame_id,
            "prv_frame_str": s.prv_frame_str,
            "bbox_xyxy": torch.tensor(list(bbox_meta_xyxy), dtype=torch.float32),
            "src_bbox_xyxy": torch.tensor(list(bbox_src_xyxy), dtype=torch.float32),
            "tx_bbox_xyxy": torch.tensor(list(bbox_tx_xyxy), dtype=torch.float32),
            "img_wh": torch.tensor([W, H], dtype=torch.int32),
            "fit_info": fit_info,
            "src_image_path": s.src_image_path,
            "tx_image_path": s.tx_image_path,
            "quality": torch.tensor(float(s.quality), dtype=torch.float32),
            "bbox_area": torch.tensor(float(s.bbox_area), dtype=torch.float32),
            "track_len": torch.tensor(int(s.track_len), dtype=torch.int32),
        }

        if self.return_full_image:
            out["full_img_uint8"] = np.asarray(src_img).astype(np.uint8)
        if self.return_tx_image:
            out["tx_full_img_uint8"] = np.asarray(tx_img).astype(np.uint8)
        if self.return_raw_roi:
            out["roi_uint8"] = np.asarray(roi).astype(np.uint8)
            out["roi_wh"] = torch.tensor([roi_w, roi_h], dtype=torch.int32)
        return out


def vod_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0:
        return {}
    out: Dict[str, Any] = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
