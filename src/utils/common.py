"""Shared utilities: config loading, seeding, EMA, checkpoint IO, logging."""

from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "kla", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
class EMA:
    """Exponential Moving Average of model weights (improves eval PSNR/SSIM)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)

    def state_dict(self):
        return self.shadow.state_dict()


# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    ema: EMA | None = None, step: int = 0, best: float = 0.0,
                    extra: Dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "step": step,
        "best": best,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    ema: EMA | None = None, map_location="cpu", use_ema: bool = False):
    ckpt = torch.load(path, map_location=map_location)
    weights = ckpt["ema"] if (use_ema and "ema" in ckpt) else ckpt["model"]
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.shadow.load_state_dict(ckpt["ema"])
    return ckpt.get("step", 0), ckpt.get("best", 0.0)


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
