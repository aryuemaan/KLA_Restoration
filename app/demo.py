#!/usr/bin/env python3
"""Interactive Gradio demo: upload a degraded grayscale image and restore it.

    python app/demo.py --ckpt experiments/wswinir_x1/best.pth
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.inference import Predictor
from src.utils.common import load_config


def build_app(predictor: Predictor, tile: int):
    import gradio as gr

    def restore(image):
        if image is None:
            return None
        if image.ndim == 3:
            image = image[..., 0]
        x = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
        out = predictor.restore_tensor(x, tile=tile)
        return (out.squeeze().cpu().numpy() * 255.0).round().astype(np.uint8)

    with gr.Blocks(title="KLA WaveletSwinIR Restoration") as demo:
        gr.Markdown("# WaveletSwinIR - Semiconductor Image Restoration\n"
                    "Upload a degraded grayscale inspection image.")
        with gr.Row():
            inp = gr.Image(image_mode="L", label="Degraded", type="numpy")
            out = gr.Image(image_mode="L", label="Restored")
        gr.Button("Restore", variant="primary").click(restore, inp, out)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(cfg, args.ckpt, device=args.device, fp16=False)
    build_app(predictor, args.tile).launch(share=args.share)


if __name__ == "__main__":
    main()
