"""Custom Triton causal GQA self-attention in fp16, with attention weights rebuilt in backward.

PyTorch's ``scaled_dot_product_attention`` on a T4 runs its math path: it upcasts fp16 Q, K and V to
fp32, repeats K and V for every query head, and autograd keeps the ``[B, H, S, S]`` attention
weights of all 28 layers until backward. This module replaces that path:

* Query heads that share one key head are laid out as one ``[r * S, D]`` matrix, so
  ``[B, 2, r * S, D] @ [B, 2, D, S]`` needs no repeated K or V and runs on fp16 Tensor Cores.
* ``triton_causal_softmax_kernel`` turns the scores into causal weights in place, in fp32 registers.
* Autograd saves only Q, K and V. Backward rebuilds the weights for one layer at a time, and
  ``triton_causal_softmax_vjp_kernel`` applies the softmax vector-Jacobian product in place.

Both kernels touch only ``col <= row``, and the online softmax follows Milakov and Gimelshein (2018).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_BLOCK = 64
_MAX_FP16_GRAD = 16.0


@triton.jit
def triton_causal_softmax_kernel(
    P_ptr,
    seq_len,
    BLOCK: tl.constexpr,
):
    """In-place causal softmax over one ``[seq_len]`` row of scores per program."""
    row = tl.program_id(0)
    row_ptr = P_ptr + row * seq_len
    n_valid = row % seq_len + 1

    m_i = -float("inf")
    d_i = 0.0
    for start in range(0, n_valid, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_valid
        x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0), axis=0)
        m_i = m_new

    inv_d = 1.0 / d_i
    for start in range(0, seq_len, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < seq_len
        x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        prob = tl.where(cols < n_valid, tl.exp(x - m_i) * inv_d, 0.0)
        tl.store(row_ptr + cols, prob.to(P_ptr.dtype.element_ty), mask=mask)


@triton.jit
def triton_causal_softmax_vjp_kernel(
    P_ptr,
    G_ptr,
    seq_len,
    BLOCK: tl.constexpr,
):
    """In-place ``g = p * (g - sum(p * g))`` over one causal row per program."""
    row = tl.program_id(0)
    p_ptr = P_ptr + row * seq_len
    g_ptr = G_ptr + row * seq_len
    n_valid = row % seq_len + 1

    dot = 0.0
    for start in range(0, n_valid, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_valid
        p = tl.load(p_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(p * g, axis=0)

    for start in range(0, seq_len, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < seq_len
        p = tl.load(p_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(g_ptr + cols, (p * (g - dot)).to(G_ptr.dtype.element_ty), mask=mask)


def _causal_weights(q_groups: torch.Tensor, k_scaled: torch.Tensor) -> torch.Tensor:
    """Causal attention weights ``[B, Hk, r * S, S]`` from grouped Q and scaled K."""
    weights = torch.matmul(q_groups, k_scaled.transpose(-1, -2))
    seq_len = weights.shape[-1]
    triton_causal_softmax_kernel[(weights.numel() // seq_len,)](
        weights, seq_len, BLOCK=_BLOCK, num_warps=2
    )
    return weights


def _attention_vjp(
    q_groups: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients for grouped Q, K and V, given the gradient of the grouped output."""
    k_scaled = k * scaling
    weights = _causal_weights(q_groups, k_scaled)
    dv = torch.matmul(weights.transpose(-1, -2), dy)

    grad_weights = torch.matmul(dy, v.transpose(-1, -2))
    seq_len = weights.shape[-1]
    triton_causal_softmax_vjp_kernel[(weights.numel() // seq_len,)](
        weights, grad_weights, seq_len, BLOCK=_BLOCK, num_warps=2
    )
    del weights

    dq = torch.matmul(grad_weights, k_scaled)
    dk = torch.matmul(grad_weights.transpose(-1, -2), q_groups) * scaling
    return dq, dk, dv


def _is_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t.sum(dtype=torch.float32)) for t in tensors)


class _TritonCausalGQAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scaling):
        # q: [B, H, S, D], k and v: [B, Hk, S, D], all contiguous, H divisible by Hk
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]
        q_groups = q.view(batch, n_kv_heads, n_heads // n_kv_heads * seq_len, head_dim)

        weights = _causal_weights(q_groups, k * scaling)
        out = torch.matmul(weights, v)

        ctx.save_for_backward(q, k, v)
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, dy):
        q, k, v = ctx.saved_tensors
        q_groups = q.view(*dy.shape)
        dy = dy.contiguous()

        # GradScaler gradients are large. A power of two keeps the fp16 sums in range without
        # rounding, and a rare overflow is redone in fp32 instead of skipping the optimizer step.
        peak = float(dy.abs().amax())
        rescale = 1.0
        if _MAX_FP16_GRAD < peak < math.inf:
            rescale = 2.0 ** math.ceil(math.log2(peak / _MAX_FP16_GRAD))
        dq, dk, dv = _attention_vjp(q_groups, k, v, dy / rescale, ctx.scaling)
        if not _is_finite(dq, dk, dv):
            dq, dk, dv = _attention_vjp(
                q_groups.float(), k.float(), v.float(), dy.float() / rescale, ctx.scaling
            )
        dq, dk, dv = (g.to(q.dtype) * rescale for g in (dq, dk, dv))
        return dq.view_as(q), dk, dv, None


def triton_causal_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Causal GQA attention over ``[B, H, S, D]`` Q and ``[B, Hk, S, D]`` K and V: ``[B, S, H * D]``."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, n_heads, seq_len, head_dim = q.shape
    out = _TritonCausalGQAFunction.apply(q, k, v, scaling)
    return out.view(batch, n_heads, seq_len, head_dim).transpose(1, 2).reshape(batch, seq_len, -1)


_PATCHED_ATTENTION = False


def patch_qwen2_attention() -> None:
    """Route plain causal fp16 CUDA training steps of ``Qwen2Attention`` through the kernels."""
    global _PATCHED_ATTENTION
    if _PATCHED_ATTENTION:
        return

    from transformers.models.qwen2 import modeling_qwen2

    orig_forward = modeling_qwen2.Qwen2Attention.forward

    def _fast_attention_forward(
        self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs
    ):
        seq_len = hidden_states.shape[1]
        if (
            attention_mask is not None
            or not hidden_states.is_cuda
            or not torch.is_grad_enabled()
            or hidden_states.dtype != torch.float16
            or seq_len < 2
            or self.sliding_window is not None
            or (self.training and self.attention_dropout > 0.0)
            or self.config._attn_implementation != "sdpa"
            or self.num_key_value_groups < 2
            or (past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0)
        ):
            return orig_forward(
                self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
            )

        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        out = triton_causal_gqa_attention(q, k, v, self.scaling)
        return self.o_proj(out), None

    modeling_qwen2.Qwen2Attention.forward = _fast_attention_forward
    _PATCHED_ATTENTION = True
