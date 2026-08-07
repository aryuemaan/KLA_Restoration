#!/usr/bin/env python3
"""Evaluate a checkpoint on a validation set. Reports mean PSNR / SSIM."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.degradations import build_degrader
from src.data.dataset import build_dataset
from src.engine.inference import Predictor
from src.metrics import SSIMMetric, psnr
from src.utils.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    degrader = build_degrader(cfg)
    val_ds = build_dataset(cfg, "val", degrader)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    pred = Predictor(cfg, args.ckpt, device=args.device,
                     use_ema=not args.no_ema, fp16=args.fp16)
    ssim_metric = SSIMMetric(device=pred.device)
    cb = cfg["model"].get("upscale", 1)

    ps = ss = 0.0
    n = 0
    for batch in loader:
        out = pred.restore_tensor(batch["lq"], tile=0)
        gt = batch["gt"].to(pred.device)
        out = out[..., :gt.shape[-2], :gt.shape[-1]]
        ps += psnr(out, gt, crop_border=cb)
        ss += ssim_metric(out, gt, crop_border=cb)
        n += 1
    print(f"Evaluated {n} images | PSNR={ps/n:.3f} dB | SSIM={ss/n:.4f}")


if __name__ == "__main__":
    main()
