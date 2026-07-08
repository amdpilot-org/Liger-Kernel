#!/usr/bin/env python3
"""Per-kernel benchmark for Liger LayerNorm on MI355X (gfx950).

Measures forward+backward latency (wall-clock via cuda.Event) and kernel-only
GPU time (via torch.profiler) for both the Liger gfx950 dispatch arm and the
PyTorch eager baseline (``torch.nn.functional.layer_norm``), at the Qwen3-32B
(5120) and Llama-3.3 (8192) hidden sizes in bf16.

Emits the canonical metric line:
    liger_ms / baseline_ms: <kernel-only ratio>

Usage:
    /opt/venv/bin/python3 benchmark/scripts/benchmark_layernorm_mi355x.py
"""
import sys
import os

REPO = os.environ.get("REPO_DIR", "/workspace/Liger-Kernel")
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from liger_kernel.transformers.layer_norm import (
    LIGER_AMD_LAYERNORM_ACTIVE,
    LigerLayerNorm,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-5
BATCH = 2048
ITERS = 50
WARMUP = 10


def time_wallclock(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def kernel_only_time(fn, iters=20):
    """Return per-iter GPU kernel time (us) for layernorm-relevant kernels."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    total_us = 0.0
    for ev in prof.key_averages():
        k = ev.key.lower()
        if ev.self_device_time_total > 0 and (
            "layer_norm" in k
            or "layernorm" in k
            or "liger" in k
        ):
            total_us += ev.self_device_time_total
    return total_us / iters / 1000.0  # ms


def make_liger_fn(hidden):
    x = torch.randn(BATCH, hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)
    layer = LigerLayerNorm(hidden, eps=EPS).to(DEVICE).to(DTYPE)

    def fn():
        out = layer(x)
        out.sum().backward()
        # zero grads to prevent bf16 overflow from accumulation across iters
        x.grad = None
        layer.weight.grad = None
        layer.bias.grad = None
    return fn


def make_native_fn(hidden):
    x = torch.randn(BATCH, hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)
    w = torch.ones(hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)
    b = torch.zeros(hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)

    def fn():
        out = F.layer_norm(x, (hidden,), weight=w, bias=b, eps=EPS)
        out.sum().backward()
        x.grad = None
        w.grad = None
        b.grad = None
    return fn


def main():
    print("=" * 64)
    print("Liger LayerNorm MI355X (gfx950) benchmark")
    print(f"  AMD dispatch arm active: {LIGER_AMD_LAYERNORM_ACTIVE}")
    print(f"  arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"  batch={BATCH}  dtype={DTYPE}  iters={ITERS}")
    print("=" * 64)

    results = {}
    for hidden in (5120, 8192):
        lig_fn = make_liger_fn(hidden)
        nat_fn = make_native_fn(hidden)

        lig_wall = time_wallclock(lig_fn)
        nat_wall = time_wallclock(nat_fn)
        lig_kern = kernel_only_time(lig_fn)
        nat_kern = kernel_only_time(nat_fn)

        wall_ratio = lig_wall / nat_wall
        kern_ratio = lig_kern / nat_kern

        print(f"\n--- hidden={hidden} ---")
        print(f"  wall-clock:  liger={lig_wall:.4f} ms  native={nat_wall:.4f} ms  ratio={wall_ratio:.3f}")
        print(f"  kernel-only: liger={lig_kern:.4f} ms  native={nat_kern:.4f} ms  ratio={kern_ratio:.3f}")
        results[hidden] = {
            "liger_wall_ms": lig_wall,
            "native_wall_ms": nat_wall,
            "wall_ratio": wall_ratio,
            "liger_kernel_ms": lig_kern,
            "native_kernel_ms": nat_kern,
            "kernel_ratio": kern_ratio,
        }

    # Primary metric: kernel-only ratio (averaged over both hidden sizes)
    avg_kern_ratio = sum(r["kernel_ratio"] for r in results.values()) / len(results)
    print()
    print(f"liger_ms / baseline_ms: {avg_kern_ratio:.6f}")
    return avg_kern_ratio


if __name__ == "__main__":
    main()
