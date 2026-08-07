"""
Datasets for restoration training / evaluation.

Two modes:
  * synthetic : point at a folder of clean grayscale images; degradations are
                generated on the fly by the Degrader (best for large clean sets).
  * paired    : point at aligned lq_dir / gt_dir folders (best when the
                organiser provides real degraded/clean pairs).

Images are read as single-channel grayscale, scaled to [0, 1]. Training crops
random patches with flip/rotate augmentation; validation returns whole images.
"""

from __future__ import annotations

import os
import random
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degradations import Degrader

_EXTS = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")


def _list_images(root: str) -> List[str]:
    files: List[str] = []
    for ext in _EXTS:
        files += glob(os.path.join(root, f"**/*{ext}"), recursive=True)
    return sorted(files)


def _load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _augment(img: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        img = img[:, ::-1]
    if random.random() < 0.5:
        img = img[::-1, :]
    if random.random() < 0.5:
        img = img.transpose(1, 0)
    return np.ascontiguousarray(img)


class SyntheticRestorationDataset(Dataset):
    def __init__(self, root: str, degrader: Degrader, patch_size: int = 128,
                 scale: int = 1, train: bool = True, repeat: int = 1):
        self.files = _list_images(root)
        if not self.files:
            raise FileNotFoundError(f"No images found under {root}")
        self.degrader = degrader
        self.patch = patch_size
        self.scale = scale
        self.train = train
        self.repeat = repeat if train else 1

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        path = self.files[idx % len(self.files)]
        hr = _load_gray(path)

        if self.train:
            ph = pw = self.patch
            h, w = hr.shape
            if h < ph or w < pw:  # pad small images
                hr = np.pad(hr, ((0, max(0, ph - h)), (0, max(0, pw - w))), mode="reflect")
                h, w = hr.shape
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            hr = hr[top:top + ph, left:left + pw]
            hr = _augment(hr)

        hr_t = torch.from_numpy(hr).unsqueeze(0)          # (1,H,W)
        lq_t = self.degrader(hr_t)                        # (1,h,w) possibly downsized
        return {"lq": lq_t, "gt": hr_t, "path": path}


class PairedRestorationDataset(Dataset):
    def __init__(self, lq_dir: str, gt_dir: str, patch_size: int = 128,
                 scale: int = 1, train: bool = True):
        self.lq_files = _list_images(lq_dir)
        self.gt_files = _list_images(gt_dir)
        assert len(self.lq_files) == len(self.gt_files) and self.lq_files, \
            "lq/gt folders must contain the same number of images"
        self.patch = patch_size
        self.scale = scale
        self.train = train

    def __len__(self):
        return len(self.lq_files)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        lq = _load_gray(self.lq_files[idx])
        gt = _load_gray(self.gt_files[idx])
        if self.train:
            ps = self.patch
            h, w = lq.shape
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
            lq = lq[top:top + ps, left:left + ps]
            gt = gt[top * self.scale:(top + ps) * self.scale,
                    left * self.scale:(left + ps) * self.scale]
            if random.random() < 0.5:
                lq, gt = lq[:, ::-1], gt[:, ::-1]
            if random.random() < 0.5:
                lq, gt = lq[::-1], gt[::-1]
            lq, gt = np.ascontiguousarray(lq), np.ascontiguousarray(gt)
        return {
            "lq": torch.from_numpy(lq).unsqueeze(0),
            "gt": torch.from_numpy(gt).unsqueeze(0),
            "path": self.lq_files[idx],
        }


def build_dataset(cfg: dict, split: str, degrader: Optional[Degrader] = None) -> Dataset:
    d = cfg["data"]
    scale = cfg["model"].get("upscale", 1)
    train = split == "train"
    mode = d.get("mode", "synthetic")
    if mode == "paired":
        key = "train" if train else "val"
        return PairedRestorationDataset(
            d[f"{key}_lq_dir"], d[f"{key}_gt_dir"],
            patch_size=d.get("patch_size", 128), scale=scale, train=train,
        )
    root = d["train_dir"] if train else d.get("val_dir", d["train_dir"])
    assert degrader is not None
    return SyntheticRestorationDataset(
        root, degrader, patch_size=d.get("patch_size", 128), scale=scale,
        train=train, repeat=d.get("repeat", 1),
    )
