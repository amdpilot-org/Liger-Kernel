"""Per-kernel parity test for the gfx950 (MI355X) LayerNorm dispatch arm.

Verifies that ``LigerLayerNorm`` dispatched through the AMD gfx950 arm produces
outputs and gradients that match ``torch.nn.functional.layer_norm`` within
bf16 tolerance, for the Qwen3-32B (5120) and Llama-3.3 (8192) hidden sizes.
"""
import pytest
import torch
import torch.nn.functional as F

from liger_kernel.transformers.layer_norm import (
    LIGER_AMD_LAYERNORM_ACTIVE,
    LigerLayerNorm,
)


@pytest.mark.skipif(
    not LIGER_AMD_LAYERNORM_ACTIVE,
    reason="gfx950 AMD dispatch arm not active on this device",
)
@pytest.mark.parametrize("hidden", [5120, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_layernorm_amd_forward_parity(hidden, dtype):
    """Forward output matches F.layer_norm within bf16 tolerance (identity weights)."""
    device = "cuda"
    eps = 1e-5
    batch = 2048
    torch.manual_seed(0)
    x = torch.randn(batch, hidden, device=device, dtype=dtype)
    w = torch.ones(hidden, device=device, dtype=dtype)
    b = torch.zeros(hidden, device=device, dtype=dtype)

    ref = F.layer_norm(x.float(), (hidden,), weight=w.float(), bias=b.float(), eps=eps).to(dtype)
    layer = LigerLayerNorm(hidden, eps=eps).to(device).to(dtype)
    out = layer(x)

    max_diff = (out.float() - ref.float()).abs().max().item()
    assert not torch.isnan(out).any(), "NaN in forward output"
    assert not torch.isinf(out).any(), "Inf in forward output"
    assert max_diff < 0.02, f"forward parity max_diff={max_diff} exceeds 0.02 (hidden={hidden})"


@pytest.mark.skipif(
    not LIGER_AMD_LAYERNORM_ACTIVE,
    reason="gfx950 AMD dispatch arm not active on this device",
)
@pytest.mark.parametrize("hidden", [5120, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_layernorm_amd_backward_parity(hidden, dtype):
    """Gradients (dX, dW, dB) match F.layer_norm within bf16 tolerance.

    bf16 backward accumulation is noisier than forward; dW is summed over the
    batch dimension so absolute tolerance is scaled to the gradient magnitude.
    """
    device = "cuda"
    eps = 1e-5
    batch = 512
    torch.manual_seed(42)
    x = torch.randn(batch, hidden, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    go = torch.randn(batch, hidden, device=device, dtype=dtype)

    # reference
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = F.layer_norm(x_ref, (hidden,), weight=w_ref, bias=b_ref, eps=eps)
    out_ref.backward(go)

    # liger
    layer = LigerLayerNorm(hidden, eps=eps).to(device).to(dtype)
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.bias.copy_(b)
    x_lig = x.detach().clone().requires_grad_(True)
    out = layer(x_lig)
    out.backward(go)

    dx_diff = (x_lig.grad.float() - x_ref.grad.float()).abs().max().item()
    dw_diff = (layer.weight.grad.float() - w_ref.grad.float()).abs().max().item()
    db_diff = (layer.bias.grad.float() - b_ref.grad.float()).abs().max().item()
    dw_scale = w_ref.grad.float().abs().max().item()

    assert not torch.isnan(x_lig.grad).any(), "NaN in dX"
    assert not torch.isinf(x_lig.grad).any(), "Inf in dX"
    assert dx_diff < 0.08, f"dX parity max_diff={dx_diff} (hidden={hidden})"
    # dW absolute tolerance scales with gradient magnitude (bf16 reduction noise)
    assert dw_diff < 0.02 * dw_scale + 0.5, f"dW parity max_diff={dw_diff} scale={dw_scale} (hidden={hidden})"
    assert db_diff < 0.04, f"dB parity max_diff={db_diff} (hidden={hidden})"
