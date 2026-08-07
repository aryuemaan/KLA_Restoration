"""
Discrete Wavelet Transform (DWT) and its inverse (IDWT), implemented as fixed
(non-trainable) convolutions using the orthonormal Haar basis.

Why a conv-based Haar DWT instead of `pytorch_wavelets`?
  * `Conv2d` / `ConvTranspose2d` are first-class ONNX operators and are fully
    supported by TensorRT, so the whole restoration graph exports cleanly to
    FP16 engines with no custom plugins.
  * The Haar basis is orthonormal, so `IDWT(DWT(x)) == x` up to floating point,
    which keeps the frequency-decoupling step loss-less.

The DWT decouples an image into four sub-bands at half resolution:
    LL  -> low/low   : smooth structural silicon-wafer patterns
    LH  -> low/high  : horizontal detail  (horizontal edges)
    HL  -> high/low  : vertical detail    (vertical edges)
    HH  -> high/high : diagonal detail + most of the high-frequency noise
This lets the network reason about structure and noise in separate channels,
exactly the "feature decoupling" step of the winning architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_kernels() -> torch.Tensor:
    """Return the 4 orthonormal 2x2 Haar analysis filters, shape (4, 1, 2, 2)."""
    h = 0.5
    ll = torch.tensor([[h, h], [h, h]])       # low-low
    lh = torch.tensor([[h, h], [-h, -h]])      # low-high  (horizontal edges)
    hl = torch.tensor([[h, -h], [h, -h]])      # high-low  (vertical edges)
    hh = torch.tensor([[h, -h], [-h, h]])      # high-high (diagonal / noise)
    return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)


class DWT(nn.Module):
    """2D Haar DWT. (B, C, H, W) -> (B, 4C, H/2, W/2).

    Output channel layout is grouped per input channel:
        [c0_LL, c0_LH, c0_HL, c0_HH, c1_LL, ...]
    For the standard single-channel grayscale input this is simply
    [LL, LH, HL, HH].
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        kernels = _haar_kernels()                       # (4,1,2,2)
        weight = kernels.repeat(in_channels, 1, 1, 1)   # (4C,1,2,2)
        # groups == in_channels -> each input channel produces its own 4 bands.
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input spatial dims must be even; the model guarantees this via padding.
        return F.conv2d(x, self.weight, stride=2, groups=self.in_channels)


class IDWT(nn.Module):
    """Inverse 2D Haar DWT. (B, 4C, H/2, W/2) -> (B, C, H, W).

    Uses a transposed convolution with the same orthonormal Haar filters; for an
    orthonormal basis the synthesis filters equal the analysis filters, so this
    perfectly inverts :class:`DWT`.
    """

    def __init__(self, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        kernels = _haar_kernels()                        # (4,1,2,2)
        weight = kernels.repeat(out_channels, 1, 1, 1)   # (4C,1,2,2)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv_transpose2d with groups=C reconstructs each channel from its 4 bands.
        return F.conv_transpose2d(x, self.weight, stride=2, groups=self.out_channels)


if __name__ == "__main__":  # tiny self-check
    dwt, idwt = DWT(1), IDWT(1)
    x = torch.randn(2, 1, 64, 64)
    bands = dwt(x)
    rec = idwt(bands)
    assert bands.shape == (2, 4, 32, 32), bands.shape
    assert rec.shape == x.shape, rec.shape
    err = (rec - x).abs().max().item()
    print(f"DWT/IDWT round-trip max abs error: {err:.2e}")
    assert err < 1e-5
