#!/usr/bin/env python3
"""Restore image(s). Usage: python scripts/infer.py --ckpt best.pth -i in/ -o out/"""
import argparse
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.inference import Predictor
from src.utils.common import load_config

EXTS = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-i", "--input", required=True, help="image file or folder")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pred = Predictor(cfg, args.ckpt, device=args.device,
                     use_ema=not args.no_ema, fp16=args.fp16)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        files = []
        for e in EXTS:
            files += glob(os.path.join(args.input, f"*{e}"))
    else:
        files = [args.input]

    for f in sorted(files):
        name = os.path.splitext(os.path.basename(f))[0] + ".png"
        out_path = os.path.join(args.output, name)
        pred.restore_image(f, out_path, tile=args.tile, overlap=args.overlap)
        print(f"restored {f} -> {out_path}")


if __name__ == "__main__":
    main()
