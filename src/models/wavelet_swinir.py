"""
WaveletSwinIR - the "winning idea" restoration network.

Pipeline
--------
    Degraded (B,1,H,W)
        |  DWT  (frequency decoupling: LL / LH / HL / HH at H/2 x W/2)
        v
    conv_first (4 -> C)  ->  shallow features  f0
        |
        v
    N x RSTB (shifted-window Swin attention over the wavelet sub-bands)
        |
    conv_after_body(.) + f0        (deep-feature global residual)
        |
        v
    Reconstruction:
        upsampler (PixelShuffle, factor 2*upscale)  ->  conv_last (-> 1 ch)
        |
        + bicubic(input, scale=upscale)             (global image residual)
        v
    Restored (B,1, upscale*H, upscale*W)

Operating on the wavelet sub-bands means the transformer runs at H/2 x W/2,
giving a ~4x reduction in attention cost versus pixel-space SwinIR while the DWT
hands the network pre-separated structure vs. high-frequency-noise channels.

`upscale == 1` performs pure denoising / feature-recovery at native resolution;
`upscale in {2, 4}` additionally super-resolves.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_modules import RSTB, to_image, to_tokens
from .wavelet import DWT


class _Upsample(nn.Sequential):
    """PixelShuffle upsampler for factors that are powers of two."""

    def __init__(self, factor: int, num_feat: int):
        layers = []
        assert (factor & (factor - 1)) == 0, "upsample factor must be a power of 2"
        for _ in range(int(torch.log2(torch.tensor(float(factor))).item())):
            layers.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            layers.append(nn.PixelShuffle(2))
        super().__init__(*layers)


class WaveletSwinIR(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        upscale: int = 1,
        embed_dim: int = 120,
        depths: Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads: Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        img_range: float = 1.0,
    ):
        super().__init__()
        assert len(depths) == len(num_heads)
        self.in_channels = in_channels
        self.upscale = upscale
        self.window_size = window_size
        self.img_range = img_range
        # DWT halves resolution, so pad the *input* to a multiple of 2*window_size.
        self.pad_multiple = window_size * 2

        # 1) frequency decoupling
        self.dwt = DWT(in_channels)
        wave_channels = in_channels * 4

        # 2) shallow feature extraction (on the 4 sub-bands)
        self.conv_first = nn.Conv2d(wave_channels, embed_dim, 3, 1, 1)

        # 3) deep feature extraction
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        cur = 0
        for i, depth in enumerate(depths):
            self.layers.append(
                RSTB(
                    dim=embed_dim, depth=depth, num_heads=num_heads[i],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    drop_path=dpr[cur:cur + depth],
                )
            )
            cur += depth
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 4) reconstruction.  Backbone is at H/2, so total pixel-shuffle factor is
        #    2 * upscale to reach (upscale * H).
        recon_factor = 2 * upscale
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = _Upsample(recon_factor, num_feat)
        self.conv_last = nn.Conv2d(num_feat, in_channels, 3, 1, 1)

        self.apply(self._init_weights)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        m = self.pad_multiple
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        x = to_tokens(x)
        for layer in self.layers:
            x = layer(x, (h, w))
        x = self.norm(x)
        return to_image(x, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x, pad_h, pad_w = self._pad(x)
        H, W = x.shape[2], x.shape[3]

        bands = self.dwt(x)                       # (B, 4C, H/2, W/2)
        f0 = self.conv_first(bands)               # shallow features
        f = self.conv_after_body(self.forward_features(f0)) + f0

        f = self.conv_before_upsample(f)
        f = self.upsample(f)                      # -> (B, num_feat, upscale*H, upscale*W)
        res = self.conv_last(f)

        base = F.interpolate(
            x, scale_factor=self.upscale, mode="bicubic", align_corners=False
        ) if self.upscale != 1 else x
        out = res + base

        # crop away the padding (scaled by the upscale factor)
        oh = (H - pad_h) * self.upscale
        ow = (W - pad_w) * self.upscale
        out = out[:, :, :oh, :ow]
        return out.clamp(0.0, self.img_range)


def build_model(cfg: dict) -> WaveletSwinIR:
    """Instantiate the model from a config dict (see configs/base.yaml)."""
    m = cfg["model"]
    return WaveletSwinIR(
        in_channels=m.get("in_channels", 1),
        upscale=m.get("upscale", 1),
        embed_dim=m.get("embed_dim", 120),
        depths=tuple(m.get("depths", [6, 6, 6, 6, 6, 6])),
        num_heads=tuple(m.get("num_heads", [6, 6, 6, 6, 6, 6])),
        window_size=m.get("window_size", 8),
        mlp_ratio=m.get("mlp_ratio", 2.0),
        drop_path_rate=m.get("drop_path_rate", 0.1),
        img_range=m.get("img_range", 1.0),
    )
