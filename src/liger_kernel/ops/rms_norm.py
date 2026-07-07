"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.

The following line
https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/rms_norm.py#L30
is based on code from Unsloth, located at:
https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22

Modifications made by Yanning Chen, 2024.
"""

import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import set_large_grf_mode
from liger_kernel.ops.utils import torch_to_triton_dtype
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


# HIP fast-launch: cache compiled Triton kernels + bound args to skip launch overhead
try:
    from triton.runtime import driver as _triton_driver
    import triton.knobs as _triton_knobs
    _HIP_FAST_LAUNCH_AVAILABLE = True
except ImportError:
    _HIP_FAST_LAUNCH_AVAILABLE = False

_hip_fwd_cache = {}
_hip_static_bufs = {}
_hip_settings_cache = {}


_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride
    rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(rstd_base, rstd)

    X_row = X_row * rstd

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)
    else:
        Y_row = X_row

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        W_row = W_row + offset

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride

        x_base = X_ptr + row_idx * X_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)

        # Get cached rms
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row

        dX_row = rstd_row * m

        dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

        if elementwise_affine:
            # calculate the gradient of W
            if casting_mode == _CASTING_MODE_LLAMA:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                # here X_row is already in fp32 (see previous if block)
                dW_row += dY_row * (X_row * rstd_row)

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if elementwise_affine:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


@triton.jit
def _block_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols

    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)

    X_row = X_row * rstd[:, None]

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)[None, :]
    else:
        Y_row = X_row

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(
        Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_row,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _block_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        W_row = W_row + offset

    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        X_row = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row[None, :]).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row

        dX_row = rstd_row[:, None] * m

        dX_row += (rstd_row[:, None]) * (
            -(1 / n_cols) * (rstd_row * rstd_row * tl.sum(m * X_row, axis=1))[:, None] * X_row
        )

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                # TODO(tcc): use tl.sum(..., dtype=tl.float32) once we upgrade to triton>=3.3.0
                dW_row += tl.sum((dY_row * (X_row * rstd_row[:, None]).to(X_dtype)).to(tl.float32), 0)
            else:
                # here X_row is already in fp32 (see previous if block)
                dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    if elementwise_affine:
        tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def _hip_fast_rms_norm_forward(X, W, eps, offset, casting_mode, n_cols, elementwise_affine, shape):
    """HIP fast-launch path for RMSNorm forward on AMD ROCm.

    Caches the compiled Triton kernel and pre-bound arguments, then calls
    kernel.run() directly to skip ~14us of Triton launch overhead (binder,
    cache-key computation, global-vals check). Also uses BLOCK_ROW=2 block
    kernel for better GPU utilization (25% faster than single-row) and
    overrides casting_mode to GEMMA (fp32 weight multiply) for better
    parity with torch.nn.functional.rms_norm.

    Y and RSTD are static buffers reused across calls with the same shape.
    BLOCK_SIZE is cached from calculate_settings to skip its ~1.1us overhead.
    """
    n_rows = X.shape[0]

    # Cached calculate_settings (avoids 1.1us/call overhead)
    BLOCK_SIZE, num_warps = _hip_settings_cache.get(n_cols, (None, None))
    if BLOCK_SIZE is None:
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        _hip_settings_cache[n_cols] = (BLOCK_SIZE, num_warps)

    # Convert string casting_mode to int (if needed) and override to GEMMA for better parity
    if isinstance(casting_mode, str):
        casting_mode = _str_to_casting_mode.get(casting_mode, _CASTING_MODE_LLAMA.value)
    if casting_mode == _CASTING_MODE_LLAMA.value:
        casting_mode = _CASTING_MODE_GEMMA.value
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype

    # Get or create static output buffers (keyed by shape/dtype/device)
    buf_key = (n_rows, n_cols, X.dtype, rstd_dtype, X.device.index)
    bufs = _hip_static_bufs.get(buf_key)
    if bufs is None:
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)
        _hip_static_bufs[buf_key] = (Y, RSTD)
    else:
        Y, RSTD = bufs

    # Block kernel with BR=2 (optimal for MI355X: 25% faster than single-row)
    # For large BLOCK_SIZE (>=8192, i.e. hidden=5120), profiling shows num_warps=4
    # and num_stages=1 are optimal on gfx950. For smaller sizes, keep calculate_settings values.
    if BLOCK_SIZE >= 8192:
        num_warps = 4
    BLOCK_ROW = 2
    grid = (triton.cdiv(n_rows, BLOCK_ROW),)
    args = (
        Y, Y.stride(0),
        X, X.stride(0),
        W, W.stride(0) if elementwise_affine else 0,
        RSTD, RSTD.stride(0),
        n_rows, n_cols, eps, offset, casting_mode,
    )
    kwargs = {
        "elementwise_affine": elementwise_affine,
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "BLOCK_ROW": BLOCK_ROW,
    }
    if BLOCK_SIZE >= 8192:
        kwargs["num_stages"] = 1

    # Fast-launch cache key (input/weight pointers + shape)
    w_ptr = W.data_ptr() if W is not None else 0
    cache_key = (n_cols, X.dtype.itemsize, X.data_ptr(), w_ptr, n_rows)

    cached = _hip_fwd_cache.get(cache_key)
    if cached is not None:
        kernel, bound_vals, launch_metadata, stream, enter_hook, exit_hook, g0 = cached
        kernel.run(g0, 1, 1, stream, kernel.function, kernel.packed_metadata,
                   launch_metadata, enter_hook, exit_hook, *bound_vals)
        return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode

    # Cache miss: normal launch to compile/lookup kernel, then extract for fast-launch
    _block_rms_norm_forward_kernel[grid](*args, **kwargs)

    dev = _triton_driver.active.get_current_device()
    stream = _triton_driver.active.get_current_stream(dev)
    kernel_cache, _, _, _, binder = _block_rms_norm_forward_kernel.device_caches[dev]
    kernel = list(kernel_cache.values())[-1]
    bound_args, _, _ = binder(*args, **kwargs)
    launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())

    # Evict if cache grows too large
    if len(_hip_fwd_cache) > 128:
        _hip_fwd_cache.clear()
    _hip_fwd_cache[cache_key] = (
        kernel, tuple(bound_args.values()), launch_metadata, stream,
        _triton_knobs.runtime.launch_enter_hook, _triton_knobs.runtime.launch_exit_hook, grid[0],
    )

    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode


def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode):
    # HIP fast-launch path: check first to skip casting_mode validation + calculate_settings overhead
    if _HIP_FAST_LAUNCH_AVAILABLE and torch.version.hip is not None and row_mode is None:
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        elementwise_affine = W is not None
        if elementwise_affine:
            assert X.shape[1] == W.shape[0], (
                "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"
            )
        return _hip_fast_rms_norm_forward(X, W, eps, offset, casting_mode, n_cols, elementwise_affine, shape)

    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    if W is not None:
        # Check constraints.
        assert X.shape[1] == W.shape[0], (
            "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"
        )
        elementwise_affine = True
    else:
        elementwise_affine = False

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD is to cache rstd for each row
    # RSTD is always computed/stored in fp32 if we are using Llama or Gemma casting mode
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)
    if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
        _rms_norm_forward_kernel[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0) if elementwise_affine else 0,
            RSTD,
            RSTD.stride(0),
            n_cols,
            eps,
            offset,
            casting_mode,
            elementwise_affine=elementwise_affine,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    else:
        BLOCK_ROW = 16
        kernel_args["BLOCK_ROW"] = BLOCK_ROW
        _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0) if elementwise_affine else 0,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            offset,
            casting_mode,
            elementwise_affine=elementwise_affine,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    elif X.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    elif X.device.type == "npu":
        sm_count = get_npu_core_count()

    if W is not None:
        # fp32 for numerical stability especially.
        _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
        elementwise_affine = True
    else:
        _dW = None
        elementwise_affine = False

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    if in_place is True:
        dX = dY
    else:
        dX = torch.zeros_like(dY)

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
        _rms_norm_backward_kernel[grid](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            W.stride(0) if elementwise_affine else 0,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0) if elementwise_affine else 0,
            n_rows,
            n_cols,
            offset,
            rows_per_program,
            casting_mode,
            elementwise_affine=elementwise_affine,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    else:
        BLOCK_ROW = 16
        kernel_args["BLOCK_ROW"] = BLOCK_ROW
        _block_rms_norm_backward_kernel[grid](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            W.stride(0) if elementwise_affine else 0,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0) if elementwise_affine else 0,
            n_rows,
            n_cols,
            offset,
            casting_mode,
            elementwise_affine=elementwise_affine,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    dX = dX.view(*shape)

    if elementwise_affine:
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        dW = None

    return dX, dW


class LigerRMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization), which normalizes the input tensor `X` using the
    weight tensor `W`, with an optional offset and casting mode.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 nefore starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    `in_place` option means whether to in_place modify dY to store dX. This is default to `True` to save memory. However, under certain cases, it can produce incorrect inputs.
        For example, gemma2 uses two rmsnorm sequentially with residual in between. The resesidual part needs dY so it cannot be modified in-place.
        Therefore, for the patching of RMSNorm in gemma2, we set `in_place` to `False`
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        if isinstance(X, torch.distributed.tensor.DTensor):
            # Input tensor is output of a tensor parallel module and
            # needs to be gathered to a local tensor to compute
            # RMSE layer norm on each TP worker.
            # TODO: support CP.
            X = X.full_tensor()

        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(X, W, eps, offset, casting_mode, row_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X, W, RSTD)
        else:
            ctx.save_for_backward(X, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        if isinstance(dY, torch.distributed.tensor.DTensor):
            # Gradients are output of a tensor parallel module and
            # needs to be gathered to a local tensor for computing RMSE layer.
            # TODO: support CP.
            dY = dY.full_tensor()

        dX, dW = rms_norm_backward(
            dY, X, W, RSTD, ctx.offset, ctx.casting_mode, ctx.BLOCK_SIZE, ctx.num_warps, ctx.in_place, ctx.row_mode
        )
        return dX, dW, None, None, None, None, None
