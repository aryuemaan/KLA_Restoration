#!/usr/bin/env python3
"""Benchmark latency / FPS across PyTorch (FP32/FP16), ONNX Runtime and TensorRT.

    python scripts/benchmark.py --ckpt best.pth --onnx model.onnx \
        --engine model.engine --height 512 --width 512 --iters 100
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.wavelet_swinir import build_model
from src.utils.common import count_params, load_checkpoint, load_config


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_torch(model, x, iters, warmup, half=False, label="torch"):
    model.eval()
    if half:
        model = model.half()
        x = x.half()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync()
        t0 = time.time()
        for _ in range(iters):
            model(x)
        _sync()
    dt = (time.time() - t0) / iters
    print(f"{label:16s}: {dt*1000:7.2f} ms/img  |  {1/dt:7.1f} FPS")
    return dt


def bench_onnx(onnx_path, x_np, iters, warmup):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x_np})
    t0 = time.time()
    for _ in range(iters):
        sess.run(None, {name: x_np})
    dt = (time.time() - t0) / iters
    print(f"{'onnxruntime':16s}: {dt*1000:7.2f} ms/img  |  {1/dt:7.1f} FPS")
    return dt


def bench_trt(engine_path, x_np, iters, warmup):
    import tensorrt as trt
    import pycuda.autoinit  # noqa
    import pycuda.driver as cuda

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    ctx.set_input_shape(engine.get_tensor_name(0), x_np.shape)

    out_shape = tuple(ctx.get_tensor_shape(engine.get_tensor_name(1)))
    d_in = cuda.mem_alloc(x_np.nbytes)
    out = np.empty(out_shape, dtype=np.float32)
    d_out = cuda.mem_alloc(out.nbytes)
    stream = cuda.Stream()

    def run():
        cuda.memcpy_htod_async(d_in, x_np, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(out, d_out, stream)
        stream.synchronize()

    ctx.set_tensor_address(engine.get_tensor_name(0), int(d_in))
    ctx.set_tensor_address(engine.get_tensor_name(1), int(d_out))
    for _ in range(warmup):
        run()
    t0 = time.time()
    for _ in range(iters):
        run()
    dt = (time.time() - t0) / iters
    print(f"{'tensorrt-fp16':16s}: {dt*1000:7.2f} ms/img  |  {1/dt:7.1f} FPS")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = cfg["model"].get("in_channels", 1)
    x = torch.randn(1, c, args.height, args.width, device=device)
    x_np = x.cpu().numpy().astype(np.float32)

    print(f"Input {tuple(x.shape)} on {device}\n" + "-" * 48)
    if args.ckpt:
        model = build_model(cfg).to(device)
        load_checkpoint(args.ckpt, model, map_location=device, use_ema=True)
        print(f"params: {count_params(model):.2f} M")
        bench_torch(model, x, args.iters, args.warmup, half=False, label="torch-fp32")
        if device == "cuda":
            bench_torch(build_model(cfg).to(device), x, args.iters, args.warmup,
                        half=True, label="torch-fp16")
    if args.onnx:
        try:
            bench_onnx(args.onnx, x_np, args.iters, args.warmup)
        except Exception as e:
            print(f"onnx bench skipped: {e}")
    if args.engine:
        try:
            bench_trt(args.engine, x_np, args.iters, args.warmup)
        except Exception as e:
            print(f"tensorrt bench skipped: {e}")


if __name__ == "__main__":
    main()
