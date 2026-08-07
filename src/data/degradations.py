"""
Degradation pipeline for grayscale semiconductor inspection images.

Given a clean patch in [0, 1], synthesise a realistic degraded observation by
composing (any subset of):

  * Gaussian blur          - defocus / optical PSF
  * Bicubic down-sampling  - for the super-resolution task (scale > 1)
  * Poisson (shot) noise   - electron/photon counting statistics of SEM imaging
  * Gaussian read noise     - high-frequency sensor noise (the main target)
  * Optional quantisation   - detector bit-depth effects

All ops run on torch tensors so they can execute on GPU inside the DataLoader's
collate if desired. Ranges are drawn per-sample for robustness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F


def _gaussian_blur_kernel(sigma: float, ksize: int) -> torch.Tensor:
    coords = torch.arange(ksize).float() - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] @ g[None, :])
    return k.unsqueeze(0).unsqueeze(0)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    k = _gaussian_blur_kernel(sigma, ksize).to(x.dtype).to(x.device)
    c = x.shape[0]
    k = k.repeat(c, 1, 1, 1)
    x = F.pad(x.unsqueeze(0), (ksize // 2,) * 4, mode="reflect")
    return F.conv2d(x, k, groups=c).squeeze(0)


@dataclass
class DegradationConfig:
    scale: int = 1
    blur_sigma: Tuple[float, float] = (0.0, 1.2)
    noise_sigma: Tuple[float, float] = (2.0, 25.0)     # in 0-255 units
    poisson_scale: Tuple[float, float] = (0.0, 3.0)    # 0 disables
    gray_noise_prob: float = 1.0
    downsample_prob: float = 1.0
    quantize_bits: int = 0                              # 0 disables


class Degrader:
    """Callable that maps a clean tensor (C,H,W) in [0,1] -> degraded tensor."""

    def __init__(self, cfg: DegradationConfig):
        self.cfg = cfg

    def __call__(self, hr: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = hr.clone()

        # 1) blur
        sigma = random.uniform(*cfg.blur_sigma)
        x = gaussian_blur(x, sigma)

        # 2) downsample for SR
        if cfg.scale > 1 and random.random() < cfg.downsample_prob:
            _, h, w = x.shape
            x = F.interpolate(
                x.unsqueeze(0), size=(h // cfg.scale, w // cfg.scale),
                mode="bicubic", align_corners=False,
            ).squeeze(0).clamp(0, 1)

        # 3) Poisson (shot) noise
        pscale = random.uniform(*cfg.poisson_scale)
        if pscale > 0:
            vals = 10 ** pscale
            noisy = torch.poisson(x * vals) / vals
            x = noisy

        # 4) Gaussian read noise
        nsigma = random.uniform(*cfg.noise_sigma) / 255.0
        x = x + torch.randn_like(x) * nsigma

        # 5) optional quantisation
        if cfg.quantize_bits > 0:
            levels = 2 ** cfg.quantize_bits - 1
            x = torch.round(x.clamp(0, 1) * levels) / levels

        return x.clamp(0, 1)


def build_degrader(cfg: dict) -> Degrader:
    d = cfg.get("degradation", {})
    return Degrader(DegradationConfig(
        scale=cfg["model"].get("upscale", 1),
        blur_sigma=tuple(d.get("blur_sigma", [0.0, 1.2])),
        noise_sigma=tuple(d.get("noise_sigma", [2.0, 25.0])),
        poisson_scale=tuple(d.get("poisson_scale", [0.0, 3.0])),
        gray_noise_prob=d.get("gray_noise_prob", 1.0),
        downsample_prob=d.get("downsample_prob", 1.0),
        quantize_bits=d.get("quantize_bits", 0),
    ))
