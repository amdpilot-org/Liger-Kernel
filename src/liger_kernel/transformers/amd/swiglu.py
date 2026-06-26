import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_forward_kernel_flat(a_ptr, b_ptr, c_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    sig_a = tl.sigmoid(a_val)
    silu_a = a_val * sig_a
    res = silu_a.cast(b_val.dtype) * b_val
    tl.store(c_ptr + offsets, res, mask=mask)


@triton.jit
def _swiglu_backward_kernel_flat(dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    term1 = silu_a * (1.0 - sig_a) + sig_a
    db = dc * silu_a
    da = dc * b * term1
    tl.store(da_ptr + offsets, da, mask=mask)
    tl.store(db_ptr + offsets, db, mask=mask)


def _get_config(total_elements: int, is_backward: bool = False):
    """Tuned heuristics for MI300X gfx942 on bf16 elementwise SwiGLU.
    Forward favours smaller blocks with fewer warps to minimise launch overhead.
    Backward is more memory-bound; slightly larger blocks amortise load/store.
    """
    if is_backward:
        # Backward reads 3x and writes 2x; use moderate block size.
        if total_elements <= 8_388_608:
            return 2048, 2
        else:
            return 4096, 4
    else:
        # Forward is heavily launch-overhead bound on gfx942.
        if total_elements <= 8_388_608:
            return 1024, 1
        else:
            return 2048, 2


def swiglu_forward_amd(a, b):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    total_elements = a.numel()
    c = torch.empty_like(a)
    block_size, num_warps = _get_config(total_elements, is_backward=False)
    grid = (max(1, (total_elements + block_size - 1) // block_size),)
    _swiglu_forward_kernel_flat[grid](a, b, c, total_elements, BLOCK_SIZE=block_size, num_warps=num_warps)
    return c


def swiglu_backward_amd(a, b, dc):
    if not dc.is_contiguous():
        dc = dc.contiguous()
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    total_elements = dc.numel()
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    block_size, num_warps = _get_config(total_elements, is_backward=True)
    grid = (max(1, (total_elements + block_size - 1) // block_size),)
    _swiglu_backward_kernel_flat[grid](dc, a, b, grad_a, grad_b, total_elements, BLOCK_SIZE=block_size, num_warps=num_warps)
    return grad_a, grad_b


class LigerSiLUMulFunctionAMD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, gate_multiplier: float = 1.0, down_multiplier: float = 1.0):
        gate_multiplier = float(gate_multiplier)
        down_multiplier = float(down_multiplier)
        ctx.gate_multiplier = gate_multiplier
        ctx.down_multiplier = down_multiplier

        a_in = a if gate_multiplier == 1.0 else a * gate_multiplier
        c = swiglu_forward_amd(a_in, b)
        if down_multiplier != 1.0:
            c = c * down_multiplier
        ctx.save_for_backward(a_in, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        a_in, b = ctx.saved_tensors
        gate_multiplier = ctx.gate_multiplier
        down_multiplier = ctx.down_multiplier
        if down_multiplier != 1.0:
            dc = dc * down_multiplier
        grad_a_in, grad_b = swiglu_backward_amd(a_in, b, dc)
        grad_a = grad_a_in if gate_multiplier == 1.0 else grad_a_in * gate_multiplier
        return grad_a, grad_b, None, None
