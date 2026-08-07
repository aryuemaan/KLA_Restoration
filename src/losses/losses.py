"""
Loss functions for physics-informed, edge-preserving restoration.

    L_total = alpha * L_Charbonnier  +  beta * L_Edge  +  gamma * L_SSIM

* Charbonnier  - a differentiable, robust L1 that avoids the over-smoothing of MSE.
* Edge         - Scharr (or Sobel) gradient loss that forces nanometer-scale chip
                 boundaries to stay sharp (the "physics-informed" term).
* SSIM         - structural similarity, aligning with the competition metric and
                 preserving local contrast / texture statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Charbonnier
# --------------------------------------------------------------------------- #
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


# --------------------------------------------------------------------------- #
#  Edge (gradient) loss
# --------------------------------------------------------------------------- #
def _edge_kernels(kind: str) -> torch.Tensor:
    if kind == "scharr":
        kx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]) / 16.0
    elif kind == "sobel":
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
    else:
        raise ValueError(f"unknown edge kind: {kind}")
    ky = kx.t().contiguous()
    return torch.stack([kx, ky], dim=0).unsqueeze(1)  # (2,1,3,3)


class EdgeLoss(nn.Module):
    """Gradient-domain loss using Scharr/Sobel filters.

    Penalises differences in both the directional gradients (gx, gy) and the
    gradient magnitude, so edges keep their location *and* their strength.
    """

    def __init__(self, kind: str = "scharr", eps: float = 1e-3):
        super().__init__()
        self.register_buffer("kernels", _edge_kernels(kind), persistent=False)
        self.eps2 = eps * eps

    def _grad(self, x):
        c = x.shape[1]
        k = self.kernels.repeat(c, 1, 1, 1)  # (2C,1,3,3), depthwise per channel
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        g = F.conv2d(x, k, groups=c)         # (B, 2C, H, W): interleaved gx,gy per ch
        gx = g[:, 0::2]
        gy = g[:, 1::2]
        return gx, gy

    def forward(self, pred, target):
        px, py = self._grad(pred)
        tx, ty = self._grad(target)
        dir_loss = torch.sqrt((px - tx) ** 2 + self.eps2).mean() \
            + torch.sqrt((py - ty) ** 2 + self.eps2).mean()
        mag_p = torch.sqrt(px ** 2 + py ** 2 + self.eps2)
        mag_t = torch.sqrt(tx ** 2 + ty ** 2 + self.eps2)
        mag_loss = torch.sqrt((mag_p - mag_t) ** 2 + self.eps2).mean()
        return dir_loss + mag_loss


# --------------------------------------------------------------------------- #
#  SSIM
# --------------------------------------------------------------------------- #
def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = g[:, None] @ g[None, :]
    return win.unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)


class SSIM(nn.Module):
    """Windowed SSIM, returns the mean SSIM in [0, 1] (for data in [0, 1])."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("window", _gaussian_window(window_size, sigma), persistent=False)
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def forward(self, x, y):
        c = x.shape[1]
        w = self.window.repeat(c, 1, 1, 1).to(x.dtype)
        pad = self.window_size // 2
        mu_x = F.conv2d(x, w, padding=pad, groups=c)
        mu_y = F.conv2d(y, w, padding=pad, groups=c)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sigma_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
        sigma_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x2 + mu_y2 + self.C1) * (sigma_x + sigma_y + self.C2)
        )
        return ssim_map.mean()


class SSIMLoss(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.ssim = SSIM(**kw)

    def forward(self, pred, target):
        return 1.0 - self.ssim(pred, target)


# --------------------------------------------------------------------------- #
#  Combined loss
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.2, edge_kind="scharr",
                 charbonnier_eps=1e-3, data_range=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.charb = CharbonnierLoss(charbonnier_eps)
        self.edge = EdgeLoss(edge_kind, charbonnier_eps)
        self.ssim = SSIMLoss(data_range=data_range)

    def forward(self, pred, target):
        l_char = self.charb(pred, target)
        l_edge = self.edge(pred, target)
        l_ssim = self.ssim(pred, target)
        total = self.alpha * l_char + self.beta * l_edge + self.gamma * l_ssim
        return total, {
            "loss": total.detach(),
            "charbonnier": l_char.detach(),
            "edge": l_edge.detach(),
            "ssim": l_ssim.detach(),
        }


def build_loss(cfg: dict) -> CombinedLoss:
    l = cfg.get("loss", {})
    return CombinedLoss(
        alpha=l.get("alpha", 1.0),
        beta=l.get("beta", 0.1),
        gamma=l.get("gamma", 0.2),
        edge_kind=l.get("edge_kind", "scharr"),
        charbonnier_eps=l.get("charbonnier_eps", 1e-3),
        data_range=l.get("data_range", 1.0),
    )
