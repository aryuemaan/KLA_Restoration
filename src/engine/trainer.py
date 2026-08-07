"""
Training engine.

Features
  * AMP mixed-precision (torch.cuda.amp) for speed + memory
  * AdamW + cosine-annealing-with-warmup schedule
  * EMA weights for a free eval-metric boost
  * gradient clipping, periodic validation (PSNR/SSIM), best-checkpoint tracking
  * TensorBoard logging (optional), full resume support
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..losses.losses import build_loss
from ..metrics import SSIMMetric, psnr
from ..models.wavelet_swinir import build_model
from ..utils.common import (EMA, count_params, get_logger, load_checkpoint,
                            save_checkpoint)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard optional
    SummaryWriter = None


def cosine_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * prog))


class Trainer:
    def __init__(self, cfg: Dict, train_ds, val_ds):
        self.cfg = cfg
        tcfg = cfg["train"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and tcfg.get("device", "cuda") == "cuda" else "cpu"
        )
        self.out_dir = tcfg.get("out_dir", "experiments/default")
        os.makedirs(self.out_dir, exist_ok=True)
        self.logger = get_logger("kla", os.path.join(self.out_dir, "train.log"))

        self.model = build_model(cfg).to(self.device)
        self.logger.info(f"Model params: {count_params(self.model):.2f} M")
        self.criterion = build_loss(cfg).to(self.device)

        self.total_steps = tcfg.get("total_steps", 300000)
        self.warmup = tcfg.get("warmup_steps", 2000)
        self.base_lr = tcfg.get("lr", 2e-4)
        self.clip = tcfg.get("grad_clip", 1.0)
        self.val_interval = tcfg.get("val_interval", 5000)
        self.save_interval = tcfg.get("save_interval", 5000)
        self.log_interval = tcfg.get("log_interval", 100)
        self.crop_border = tcfg.get("crop_border", cfg["model"].get("upscale", 1))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.base_lr,
            betas=(0.9, 0.99), weight_decay=tcfg.get("weight_decay", 0.0),
        )
        self.use_amp = tcfg.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = EMA(self.model, tcfg.get("ema_decay", 0.999)) if tcfg.get("ema", True) else None

        self.train_loader = DataLoader(
            train_ds, batch_size=tcfg.get("batch_size", 16), shuffle=True,
            num_workers=tcfg.get("num_workers", 8), pin_memory=True, drop_last=True,
            persistent_workers=tcfg.get("num_workers", 8) > 0,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=max(1, tcfg.get("num_workers", 8) // 2),
        )
        self.ssim_metric = SSIMMetric(device=self.device)
        self.writer = SummaryWriter(self.out_dir) if SummaryWriter else None
        self.step, self.best = 0, 0.0

    # ------------------------------------------------------------------ #
    def _set_lr(self):
        lr = self.base_lr * cosine_warmup(self.step, self.warmup, self.total_steps)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def resume(self, path: str):
        self.step, self.best = load_checkpoint(
            path, self.model, self.optimizer, scaler=self.scaler, ema=self.ema,
            map_location=self.device,
        )
        self.logger.info(f"Resumed from {path} @ step {self.step} (best={self.best:.3f})")

    # ------------------------------------------------------------------ #
    def train(self):
        self.model.train()
        data_iter = iter(self.train_loader)
        t0 = time.time()
        running = {}
        while self.step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            lq = batch["lq"].to(self.device, non_blocking=True)
            gt = batch["gt"].to(self.device, non_blocking=True)
            lr = self._set_lr()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(lq)
                loss, parts = self.criterion(pred, gt)

            self.scaler.scale(loss).backward()
            if self.clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema:
                self.ema.update(self.model)

            self.step += 1
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item()

            if self.step % self.log_interval == 0:
                speed = self.log_interval / (time.time() - t0)
                msg = " ".join(f"{k}={running[k] / self.log_interval:.4f}" for k in running)
                self.logger.info(
                    f"step {self.step}/{self.total_steps} lr={lr:.2e} {msg} "
                    f"({speed:.1f} it/s)"
                )
                if self.writer:
                    for k in running:
                        self.writer.add_scalar(f"train/{k}", running[k] / self.log_interval, self.step)
                    self.writer.add_scalar("train/lr", lr, self.step)
                running, t0 = {}, time.time()

            if self.step % self.val_interval == 0:
                self.validate()
                self.model.train()

            if self.step % self.save_interval == 0:
                save_checkpoint(
                    os.path.join(self.out_dir, "last.pth"), self.model,
                    self.optimizer, scaler=self.scaler, ema=self.ema,
                    step=self.step, best=self.best,
                )
        self.logger.info("Training complete.")
        if self.writer:
            self.writer.close()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self):
        eval_model = self.ema.shadow if self.ema else self.model
        eval_model.eval()
        ps, ss, n = 0.0, 0.0, 0
        for batch in self.val_loader:
            lq = batch["lq"].to(self.device)
            gt = batch["gt"].to(self.device)
            pred = eval_model(lq)
            # guard against off-by-one from odd sizes
            pred = pred[..., :gt.shape[-2], :gt.shape[-1]]
            ps += psnr(pred, gt, crop_border=self.crop_border)
            ss += self.ssim_metric(pred, gt, crop_border=self.crop_border)
            n += 1
        ps, ss = ps / max(1, n), ss / max(1, n)
        self.logger.info(f"[val] step {self.step}: PSNR={ps:.3f} dB  SSIM={ss:.4f}")
        if self.writer:
            self.writer.add_scalar("val/psnr", ps, self.step)
            self.writer.add_scalar("val/ssim", ss, self.step)
        score = ps + 20 * ss  # combined selection metric
        if score > self.best:
            self.best = score
            save_checkpoint(
                os.path.join(self.out_dir, "best.pth"), self.model,
                self.optimizer, scaler=self.scaler, ema=self.ema,
                step=self.step, best=self.best,
                extra={"val_psnr": ps, "val_ssim": ss},
            )
            self.logger.info(f"  -> new best (PSNR={ps:.3f}, SSIM={ss:.4f}) saved.")
        return ps, ss
