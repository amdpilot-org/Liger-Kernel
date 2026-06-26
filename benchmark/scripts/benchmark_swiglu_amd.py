#!/usr/bin/env python3
"""Benchmark Liger SwiGLU AMD kernel vs PyTorch eager baseline on MI300X/gfx942."""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP, _swiglu_dispatch
from liger_kernel.transformers.amd.swiglu import LigerSiLUMulFunctionAMD


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_eager(batch_tokens: int, intermediate: int, warmup: int, iters: int):
    device = torch.device("cuda")
    gate = torch.randn((batch_tokens, intermediate), device=device, dtype=torch.bfloat16)
    up = torch.randn((batch_tokens, intermediate), device=device, dtype=torch.bfloat16)

    ref = F.silu(gate[:4, : min(128, intermediate)].float()) * up[:4, : min(128, intermediate)].float()
    got = (F.silu(gate[:4, : min(128, intermediate)]) * up[:4, : min(128, intermediate)]).float()
    max_abs_diff = (got - ref).abs().max().item()
    if not math.isfinite(max_abs_diff):
        raise RuntimeError("baseline produced non-finite parity diff")

    for _ in range(warmup):
        out = F.silu(gate) * up
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        out = F.silu(gate) * up
    _sync()
    elapsed = time.perf_counter() - start
    checksum = float(out.float().mean().item())
    return elapsed * 1000.0 / iters, max_abs_diff, checksum


def time_liger(batch_tokens: int, intermediate: int, warmup: int, iters: int):
    device = torch.device("cuda")
    gate = torch.randn((batch_tokens, intermediate), device=device, dtype=torch.bfloat16)
    up = torch.randn((batch_tokens, intermediate), device=device, dtype=torch.bfloat16)

    # parity sanity check
    with torch.no_grad():
        eager_slice = (F.silu(gate[:4, : min(128, intermediate)]) * up[:4, : min(128, intermediate)]).float()
        liger_slice = _swiglu_dispatch(gate[:4, : min(128, intermediate)], up[:4, : min(128, intermediate)]).float()
    max_abs_diff = (liger_slice - eager_slice).abs().max().item()
    if not math.isfinite(max_abs_diff):
        raise RuntimeError("liger produced non-finite parity diff")

    for _ in range(warmup):
        out = _swiglu_dispatch(gate, up)
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        out = _swiglu_dispatch(gate, up)
    _sync()
    elapsed = time.perf_counter() - start
    checksum = float(out.float().mean().item())
    return elapsed * 1000.0 / iters, max_abs_diff, checksum


def time_liger_training(batch_tokens: int, intermediate: int, warmup: int, iters: int):
    device = torch.device("cuda")
    gate = torch.randn((batch_tokens, intermediate), device=device, dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn((batch_tokens, intermediate), device=device, dtype=torch.bfloat16, requires_grad=True)

    for _ in range(warmup):
        out = LigerSiLUMulFunctionAMD.apply(gate, up)
        out.sum().backward()
        gate.grad = None
        up.grad = None
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        out = LigerSiLUMulFunctionAMD.apply(gate, up)
        out.sum().backward()
        gate.grad = None
        up.grad = None
    _sync()
    elapsed = time.perf_counter() - start
    checksum = float(out.detach().float().mean().item())
    return elapsed * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No HIP/CUDA GPU visible to PyTorch")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    name = torch.cuda.get_device_name(0)
    if "gfx942" not in arch:
        raise SystemExit(f"Expected MI300X gfx942, got device={name!r} arch={arch!r}")

    results = []
    for intermediate in (2048, 4096):
        eager_ms, eager_diff, eager_checksum = time_eager(args.batch_tokens, intermediate, args.warmup, args.iters)
        liger_ms, liger_diff, liger_checksum = time_liger(args.batch_tokens, intermediate, args.warmup, args.iters)
        liger_train_ms = time_liger_training(args.batch_tokens, intermediate, args.warmup, args.iters)

        results.append({
            "intermediate": intermediate,
            "eager_ms": eager_ms,
            "liger_ms": liger_ms,
            "liger_train_ms": liger_train_ms,
            "speedup_vs_eager": eager_ms / liger_ms,
            "max_abs_diff": liger_diff,
        })
        print(
            f"intermediate={intermediate} eager_ms={eager_ms:.6f} liger_ms={liger_ms:.6f} "
            f"liger_train_ms={liger_train_ms:.6f} speedup={eager_ms/liger_ms:.3f} diff={liger_diff:.6g}",
            flush=True,
        )

    metric = next(r["liger_ms"] for r in results if r["intermediate"] == 4096)
    target = 0.75 * next(r["eager_ms"] for r in results if r["intermediate"] == 4096)
    if args.json:
        print(json.dumps({"device": name, "arch": arch, "results": results, "metric_value": metric, "target_ms": target}, indent=2))
    print("===== AMDPILOT_METRIC v1 =====")
    print(f"metric_value: {metric:.6f}")
    print(f"target_ms: {target:.6f}")
    print(f"pass: {metric <= target}")
    print("===== END AMDPILOT_METRIC =====")


if __name__ == "__main__":
    main()
