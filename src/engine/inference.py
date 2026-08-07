"""
Inference / prediction utilities.

`Predictor` restores a full-resolution grayscale image. For images larger than
the training patch it performs overlapped tiled inference with a smooth
(raised-cosine) blend to avoid visible seams, and supports FP16 autocast.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.wavelet_swinir import build_model
from ..utils.common import load_checkpoint


def _cosine_window(h: int, w: int, device, border: int) -> torch.Tensor:
    """2D raised-cosine weight map that tapers to ~0 over `border` px on each edge."""
    def ramp(n):
        r = torch.ones(n, device=device)
        if border > 0:
            t = torch.linspace(0, math.pi, steps=2 * border, device=device)
            edge = 0.5 - 0.5 * torch.cos(t[:border])
            r[:border] = edge
            r[-border:] = edge.flip(0)
        return r
    wy, wx = ramp(h), ramp(w)
    return (wy[:, None] * wx[None, :]).clamp_min(1e-4)


class Predictor:
    def __init__(self, cfg: dict, ckpt_path: str, device: str = "cuda",
                 use_ema: bool = True, fp16: bool = True):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.upscale = cfg["model"].get("upscale", 1)
        self.model = build_model(cfg).to(self.device).eval()
        load_checkpoint(ckpt_path, self.model, map_location=self.device, use_ema=use_ema)
        self.fp16 = fp16 and self.device.type == "cuda"
        if self.fp16:
            self.model.half()

    @torch.no_grad()
    def _run(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp16:
            x = x.half()
        out = self.model(x)
        return out.float()

    @torch.no_grad()
    def restore_tensor(self, x: torch.Tensor, tile: int = 0, overlap: int = 32) -> torch.Tensor:
        """x: (1,1,H,W) in [0,1]. Returns (1,1,upscale*H,upscale*W)."""
        x = x.to(self.device)
        _, _, h, w = x.shape
        if tile <= 0 or (h <= tile and w <= tile):
            return self._run(x).clamp(0, 1)

        s = self.upscale
        stride = tile - overlap
        out = torch.zeros((1, 1, h * s, w * s), device=self.device)
        weight = torch.zeros_like(out)
        ys = list(range(0, max(1, h - overlap), stride))
        xs = list(range(0, max(1, w - overlap), stride))
        for y in ys:
            for xx in xs:
                y2, x2 = min(y + tile, h), min(xx + tile, w)
                y1, x1 = max(0, y2 - tile), max(0, x2 - tile)
                patch = x[:, :, y1:y2, x1:x2]
                res = self._run(patch)
                ph, pw = res.shape[-2], res.shape[-1]
                win = _cosine_window(ph, pw, self.device, max(1, overlap * s))[None, None]
                out[:, :, y1 * s:y1 * s + ph, x1 * s:x1 * s + pw] += res * win
                weight[:, :, y1 * s:y1 * s + ph, x1 * s:x1 * s + pw] += win
        return (out / weight.clamp_min(1e-6)).clamp(0, 1)

    @torch.no_grad()
    def restore_image(self, path: str, out_path: Optional[str] = None,
                      tile: int = 512, overlap: int = 32) -> np.ndarray:
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(img)[None, None]
        out = self.restore_tensor(x, tile=tile, overlap=overlap)
        arr = (out.squeeze().cpu().numpy() * 255.0).round().astype(np.uint8)
        if out_path:
            Image.fromarray(arr).save(out_path)
        return arr
