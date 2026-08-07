"""Evaluation metrics: PSNR and SSIM (competition-style, data in [0, 1])."""

from __future__ import annotations

import torch

from .losses.losses import SSIM


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
         crop_border: int = 0) -> float:
    if crop_border:
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-12)
    return (10.0 * torch.log10(data_range ** 2 / mse)).mean().item()


class SSIMMetric:
    """Reusable SSIM metric wrapper (keeps the Gaussian window on device)."""

    def __init__(self, data_range: float = 1.0, device="cpu"):
        self.ssim = SSIM(data_range=data_range).to(device)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor,
                 crop_border: int = 0) -> float:
        if crop_border:
            pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
            target = target[..., crop_border:-crop_border, crop_border:-crop_border]
        return self.ssim(pred, target).item()
