"""Sanity tests. Run: pytest -q  (CPU-friendly, uses a tiny model config)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses.losses import CombinedLoss, SSIM
from src.metrics import psnr
from src.models.wavelet import DWT, IDWT
from src.models.wavelet_swinir import WaveletSwinIR


def _tiny(upscale=1):
    return WaveletSwinIR(
        in_channels=1, upscale=upscale, embed_dim=24,
        depths=(2, 2), num_heads=(2, 2), window_size=4, mlp_ratio=2.0,
        drop_path_rate=0.0,
    )


def test_dwt_idwt_roundtrip():
    x = torch.randn(2, 1, 32, 32)
    rec = IDWT(1)(DWT(1)(x))
    assert rec.shape == x.shape
    assert (rec - x).abs().max() < 1e-5


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_model_shapes(scale):
    m = _tiny(scale).eval()
    x = torch.rand(1, 1, 40, 56)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 1, 40 * scale, 56 * scale), y.shape
    assert (y >= 0).all() and (y <= 1).all()


def test_odd_size_padding():
    m = _tiny(1).eval()
    x = torch.rand(1, 1, 37, 45)  # not a multiple of 2*window_size
    with torch.no_grad():
        y = m(x)
    assert y.shape[-2:] == (37, 45)


def test_combined_loss_backward():
    m = _tiny(1)
    x = torch.rand(2, 1, 32, 32)
    gt = torch.rand(2, 1, 32, 32)
    crit = CombinedLoss()
    y = m(x)
    loss, parts = crit(y, gt)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) >= {"charbonnier", "edge", "ssim"}
    g = next(p.grad for p in m.parameters() if p.grad is not None)
    assert torch.isfinite(g).all()


def test_ssim_identity():
    x = torch.rand(1, 1, 64, 64)
    assert SSIM()(x, x).item() == pytest.approx(1.0, abs=1e-4)


def test_psnr_identity():
    x = torch.rand(1, 1, 64, 64)
    assert psnr(x, x) > 100  # clamped mse -> very high
