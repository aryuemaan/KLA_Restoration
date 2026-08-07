"""
Swin Transformer building blocks used by the WaveletSwinIR backbone.

This is a clean, self-contained implementation of the components introduced in
SwinIR (Liang et al., 2021), adapted so the whole graph is ONNX/TensorRT
exportable:

    WindowAttention        : (shifted) window multi-head self-attention with a
                             learned relative-position bias.
    SwinTransformerLayer   : LN -> (S)W-MSA -> residual -> LN -> MLP -> residual.
    BasicLayer             : a stack of SwinTransformerLayers.
    RSTB                    : Residual Swin Transformer Block = BasicLayer + a
                             conv, wrapped in a residual connection.

Tokens flow as (B, L, C) with L = H*W; helper `to_tokens` / `to_image` convert
between the (B, L, C) and (B, C, H, W) layouts.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Layout helpers
# --------------------------------------------------------------------------- #
def to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)."""
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


def to_image(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, H, W)."""
    b, l, c = x.shape
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# --------------------------------------------------------------------------- #
#  Window partition / reverse
# --------------------------------------------------------------------------- #
def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, ws, ws, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(windows: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    """(num_windows*B, ws, ws, C) -> (B, H, W, C)."""
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


# --------------------------------------------------------------------------- #
#  Window attention
# --------------------------------------------------------------------------- #
class WindowAttention(nn.Module):
    """Window multi-head self attention with a learned relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # (2*ws-1)*(2*ws-1) distinct relative positions, one bias per head.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flat = torch.flatten(coords, 1)                                   # (2, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]                  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()                                  # (N, N, 2)
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_index = rel.sum(-1)                                                   # (N, N)
        self.register_buffer("relative_position_index", rel_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (num_windows*B, N, C)
        bn, n, c = x.shape
        qkv = self.qkv(x).reshape(bn, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # each (bn, heads, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (bn, heads, N, N)

        n_win = self.window_size * self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n_win, n_win, -1).permute(2, 0, 1).contiguous()  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(bn // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.attn_drop(self.softmax(attn))
        out = (attn @ v).transpose(1, 2).reshape(bn, n, c)
        return self.proj_drop(self.proj(out))


# --------------------------------------------------------------------------- #
#  Swin Transformer layer
# --------------------------------------------------------------------------- #
class SwinTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 2.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias,
                                    attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def _attn_mask(self, h: int, w: int, device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for wsl in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, hs, wsl, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, c)
        attn_windows = self.attn(x_windows, self._attn_mask(h, w, x.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor()


# --------------------------------------------------------------------------- #
#  BasicLayer + RSTB
# --------------------------------------------------------------------------- #
class BasicLayer(nn.Module):
    """A stack of Swin layers alternating regular / shifted windows."""

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        dpr = drop_path if isinstance(drop_path, (list, tuple)) else [drop_path] * depth
        self.blocks = nn.ModuleList([
            SwinTransformerLayer(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop, drop_path=dpr[i],
            )
            for i in range(depth)
        ])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: BasicLayer + conv, with a residual add."""

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.residual_group = BasicLayer(
            dim, depth, num_heads, window_size, mlp_ratio, qkv_bias,
            drop, attn_drop, drop_path,
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        h, w = x_size
        out = self.residual_group(x, x_size)          # (B, L, C)
        out = to_image(out, h, w)
        out = self.conv(out)
        out = to_tokens(out)
        return out + x
