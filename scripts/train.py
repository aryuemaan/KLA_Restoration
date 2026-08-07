#!/usr/bin/env python3
"""Train WaveletSwinIR. Usage: python scripts/train.py -c configs/base.yaml [--resume ckpt]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.degradations import build_degrader
from src.data.dataset import build_dataset
from src.engine.trainer import Trainer
from src.utils.common import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    degrader = build_degrader(cfg)
    train_ds = build_dataset(cfg, "train", degrader)
    val_ds = build_dataset(cfg, "val", degrader)

    trainer = Trainer(cfg, train_ds, val_ds)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
