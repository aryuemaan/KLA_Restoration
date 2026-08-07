#!/usr/bin/env python3
"""Build an FP16 TensorRT engine from an ONNX file (TensorRT Python API).

    python scripts/export_tensorrt.py --onnx model.onnx -o model.engine --fp16

If you don't have the TensorRT Python bindings, the equivalent trtexec command is:

    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 \
            --memPoolSize=workspace:4096 \
            --minShapes=input:1x1x256x256 \
            --optShapes=input:1x1x512x512 \
            --maxShapes=input:1x1x1024x1024
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("-o", "--output", default="model.engine")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--workspace", type=int, default=4096, help="MiB")
    ap.add_argument("--min-hw", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--opt-hw", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--max-hw", type=int, nargs=2, default=[1024, 1024])
    args = ap.parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 20)
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled.")

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    c = inp.shape[1] if inp.shape[1] > 0 else 1
    profile.set_shape(
        inp.name,
        (1, c, *args.min_hw), (1, c, *args.opt_hw), (1, c, *args.max_hw),
    )
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("Engine build failed.")
        sys.exit(1)
    with open(args.output, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine saved -> {args.output}")


if __name__ == "__main__":
    main()
