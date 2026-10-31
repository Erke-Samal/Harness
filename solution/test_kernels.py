"""Numerical correctness tests for all custom Triton kernels in ``solution/``.

Run on a CUDA machine (e.g., Kaggle Tesla T4):
    pytest solution/test_kernels.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from solution.attention_triton import triton_causal_gqa_attention
from solution.cross_entropy_triton import triton_chunked_causal_lm_loss
from solution.nf4_triton import triton_dequantize_nf4
from solution.rmsnorm_triton import triton_rmsnorm
from solution.rope_triton import _pytorch_rope_fallback, triton_rope_qk
from solution.swiglu import swiglu
from solution.token_packing import TokenPacking, pack_tokens, unpack_tokens

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOL = {
    torch.float16: (2e-3, 2e-3),
    torch.float32: (1e-4, 1e-4),
}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 64, 1536), (4, 128, 8960), (7, 19, 512)])
def test_swiglu_2d_matches_pytorch(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)

    out = swiglu(gate, up)
    out_ref = F.silu(gate_ref) * up_ref

    atol, rtol = TOL[dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    # The backward state is int8: a step of the largest tile value, times |grad_out| <= about 5.
    torch.testing.assert_close(gate.grad, gate_ref.grad, atol=0.15, rtol=0.0)
    torch.testing.assert_close(up.grad, up_ref.grad, atol=0.15, rtol=0.0)
    assert (gate.grad - gate_ref.grad).abs().mean() < 0.02


@pytest.mark.parametrize("x_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("w_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 128, 1536), (2, 63, 1536), (3, 17, 512)])
def test_rmsnorm_matches_qwen2_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(1)
    eps = 1e-6
    hidden_dim = shape[-1]

    x = torch.randn(shape, device="cuda", dtype=x_dtype, requires_grad=True)
    w = torch.randn(hidden_dim, device="cuda", dtype=w_dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out = triton_rmsnorm(x, w, eps=eps)
    assert out.dtype == x_dtype

    # Reference in fp32 cast back to x_dtype
    x_f32 = x_ref.to(torch.float32)
    var = x_f32.pow(2).mean(-1, keepdim=True)
    out_ref = (w_ref.to(torch.float32) * (x_f32 * torch.rsqrt(var + eps))).to(x_dtype)

    atol, rtol = TOL[x_dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(w.grad, w_ref.grad, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("broadcast_batch", [True, False])
def test_rope_qk_matches_pytorch(dtype: torch.dtype, broadcast_batch: bool) -> None:
    torch.manual_seed(2)
    batch, n_qh, n_kh, seq_len, head_dim = 4, 12, 2, 128, 128

    # Create non-contiguous transposed views just like Qwen2Attention.forward
    q_raw = torch.randn((batch, seq_len, n_qh, head_dim), device="cuda", dtype=dtype, requires_grad=True)
    k_raw = torch.randn((batch, seq_len, n_kh, head_dim), device="cuda", dtype=dtype, requires_grad=True)
    q_ref_raw = q_raw.detach().clone().requires_grad_(True)
    k_ref_raw = k_raw.detach().clone().requires_grad_(True)

    cos_b = 1 if broadcast_batch else batch
    cos = torch.randn((cos_b, seq_len, head_dim), device="cuda", dtype=dtype)
    sin = torch.randn((cos_b, seq_len, head_dim), device="cuda", dtype=dtype)

    q_out, k_out = triton_rope_qk(q_raw.transpose(1, 2), k_raw.transpose(1, 2), cos, sin)
    q_ref, k_ref = _pytorch_rope_fallback(q_ref_raw.transpose(1, 2), k_ref_raw.transpose(1, 2), cos, sin)

    atol, rtol = (5e-3, 5e-3) if dtype == torch.float16 else TOL[dtype]
    torch.testing.assert_close(q_out, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, k_ref, atol=atol, rtol=rtol)

    dq = torch.randn_like(q_out)
    dk = torch.randn_like(k_out)
    (q_out * dq).sum().add_((k_out * dk).sum()).backward()
    (q_ref * dq).sum().add_((k_ref * dk).sum()).backward()

    torch.testing.assert_close(q_raw.grad, q_ref_raw.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_raw.grad, k_ref_raw.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("compress_statistics", [True, False])
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (256, 1536)])
def test_nf4_dequant_matches_bitsandbytes(
    compress_statistics: bool,
    quant_type: str,
    shape: tuple[int, int],
) -> None:
    import bitsandbytes.functional as bnb_F
    from solution.nf4_triton import _orig_dequantize_4bit

    torch.manual_seed(3)
    w = torch.randn(shape, device="cuda", dtype=torch.float16)
    qweight, quant_state = bnb_F.quantize_4bit(
        w,
        blocksize=64,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )

    ref_fn = _orig_dequantize_4bit if _orig_dequantize_4bit is not None else bnb_F.dequantize_4bit
    w_ref = ref_fn(qweight, quant_state=quant_state)
    w_tri = triton_dequantize_nf4(qweight, quant_state=quant_state)

    torch.testing.assert_close(w_tri, w_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunked_cross_entropy_and_entropy_match_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(4)
    batch, seq_len, hidden_dim, vocab_size = 2, 33, 256, 4099
    ignore_index = -100

    hidden = (torch.randn((batch, seq_len, hidden_dim), device="cuda", dtype=dtype) * 0.2).requires_grad_(True)
    weight = (torch.randn((vocab_size, hidden_dim), device="cuda", dtype=dtype) * 0.2).requires_grad_(True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")
    labels[:, :5] = ignore_index
    labels[0, -3:] = ignore_index

    loss, tok_losses, tok_entropies, correct_toks, valid_toks = triton_chunked_causal_lm_loss(
        hidden,
        weight,
        labels,
        chunk_size=16,
        ignore_index=ignore_index,
    )

    # Reference computation
    logits_all = F.linear(hidden_ref.float(), weight_ref.float())
    logits_ref = logits_all[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    tok_losses_ref = F.cross_entropy(
        logits_ref.reshape(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(batch, seq_len - 1)
    valid_mask = shift_labels.ne(ignore_index)
    loss_ref = tok_losses_ref.sum() / valid_mask.sum().clamp_min(1)

    log_probs = F.log_softmax(logits_ref, dim=-1)
    tok_entropies_ref = -(log_probs.exp() * log_probs).sum(dim=-1)
    preds_ref = logits_ref.argmax(dim=-1)
    correct_ref = (preds_ref.eq(shift_labels) & valid_mask).sum()

    atol, rtol = (2e-2, 2e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    assert tok_losses.shape == labels.shape
    torch.testing.assert_close(tok_losses[:, :-1], tok_losses_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(tok_entropies[:, :-1], tok_entropies_ref, atol=atol, rtol=rtol)
    assert int(correct_toks.item()) == int(correct_ref.item())
    assert int(valid_toks.item()) == int(valid_mask.sum().item())

    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(hidden.grad, hidden_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(weight.grad, weight_ref.grad.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("seq_len", [2, 64, 129])
def test_causal_gqa_attention_matches_sdpa(dtype: torch.dtype, seq_len: int) -> None:
    batch, n_heads, n_kv_heads, head_dim = 2, 12, 2, 32
    scaling = head_dim**-0.5
    q = torch.randn((batch, n_heads, seq_len, head_dim), device="cuda", dtype=dtype).requires_grad_(True)
    k = torch.randn((batch, n_kv_heads, seq_len, head_dim), device="cuda", dtype=dtype).requires_grad_(True)
    v = torch.randn((batch, n_kv_heads, seq_len, head_dim), device="cuda", dtype=dtype).requires_grad_(True)
    q_ref, k_ref, v_ref = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    weight = torch.randn((batch, seq_len, n_heads * head_dim), device="cuda")

    out = triton_causal_gqa_attention(q, k, v, scaling)
    repeats = n_heads // n_kv_heads
    out_ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref.repeat_interleave(repeats, dim=1),
        v_ref.repeat_interleave(repeats, dim=1),
        is_causal=True,
        scale=scaling,
    )
    out_ref = out_ref.transpose(1, 2).reshape(batch, seq_len, -1)

    atol, rtol = (2e-2, 2e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), out_ref, atol=atol, rtol=rtol)

    (out.float() * weight).sum().backward()
    (out_ref * weight).sum().backward()
    torch.testing.assert_close(q.grad.float(), q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad.float(), k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad.float(), v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("width", [256, 1536, 4000])
def test_token_packing_round_trip_and_gradients(width: int) -> None:
    mask = torch.zeros((3, 9), dtype=torch.bool, device="cuda")
    for row, length in enumerate((9, 4, 6)):
        mask[row, :length] = True
    packing = TokenPacking(mask.view(-1).nonzero().squeeze(1), 3, 9)
    packed = torch.randn((1, packing.index.numel(), width), device="cuda", dtype=torch.float16).requires_grad_(True)

    grid = unpack_tokens(packed, packing)
    assert torch.equal(grid[mask], packed[0])
    assert not grid[~mask].any()
    assert torch.equal(pack_tokens(grid, packing), packed)

    weight = torch.randn((3, 9, width), device="cuda", dtype=torch.float16)
    (grid.float() * weight.float()).sum().backward()
    assert torch.equal(packed.grad[0], weight[mask])
