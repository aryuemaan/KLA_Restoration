#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX (FP32), then optionally simplify.

    python scripts/export_onnx.py --ckpt best.pth -o model.onnx --height 512 --width 512

Notes
  * Uses opset 17 so torch.roll / relative-position gather export cleanly.
  * Dynamic batch axis is enabled; keep H,W fixed for best TensorRT performance
    (semiconductor tiles are usually a fixed size). Use --dynamic-hw to allow
    variable spatial dims (slightly slower engines).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.wavelet_swinir import build_model
from src.utils.common import load_checkpoint, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-o", "--output", default="model.onnx")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--dynamic-hw", action="store_true")
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg).eval()
    load_checkpoint(args.ckpt, model, map_location="cpu", use_ema=not args.no_ema)

    dummy = torch.randn(1, cfg["model"].get("in_channels", 1), args.height, args.width)
    dyn = {"input": {0: "batch"}, "output": {0: "batch"}}
    if args.dynamic_hw:
        dyn["input"].update({2: "height", 3: "width"})
        dyn["output"].update({2: "out_h", 3: "out_w"})

    export_kwargs = dict(
        input_names=["input"], output_names=["output"],
        opset_version=args.opset, dynamic_axes=dyn, do_constant_folding=True,
    )
    import inspect
    # Prefer the stable TorchScript exporter (dynamo=False); it produces the
    # cleanest graph for TensorRT. Fall back gracefully on older torch.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, args.output, **export_kwargs)
    print(f"exported ONNX -> {args.output}")

    if not args.no_simplify:
        try:
            import onnx
            from onnxsim import simplify
            m = onnx.load(args.output)
            m_sim, ok = simplify(m)
            if ok:
                onnx.save(m_sim, args.output)
                print("simplified ONNX graph.")
        except Exception as e:  # onnxsim optional
            print(f"(skip simplify: {e})")


if __name__ == "__main__":
    main()
