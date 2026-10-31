"""Custom Triton fused Rotary Position Embedding (RoPE) kernels for Q and K.

Replaces HuggingFace Qwen2's ``apply_rotary_pos_emb`` (which allocates 6
intermediate tensors per attention layer via ``rotate_half`` slicing, negation,
``torch.cat``, multiplication, and addition on both ``q`` and ``k``) with a
single 2D-grid ``@triton.jit`` kernel launch ``triton_rope_qk_fwd_kernel`` over
``(B * S, H_q + H_k)`` and its exact inverse ``triton_rope_qk_bwd_kernel`` in
backward.

Reads directly from non-contiguous ``transpose(1, 2)`` views of ``q`` and ``k``
and writes contiguous ``[B, H, S, D]`` tensors ready for SDPA while saving only
``cos`` and ``sin`` for the backward pass (0 bytes of extra activation memory).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def triton_rope_qk_fwd_kernel(
    Q_ptr,
    K_ptr,
    COS_ptr,
    SIN_ptr,
    QO_ptr,
    KO_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_cos_b,
    stride_cos_s,
    stride_qob,
    stride_qoh,
    stride_qos,
    stride_kob,
    stride_koh,
    stride_kos,
    seq_len,
    n_qh,
    HALF_D: tl.constexpr,
):
    """Fused Q+K Rotary Position Embedding forward kernel."""
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_idx = pid_bs // seq_len
    s_idx = pid_bs % seq_len

    offs = tl.arange(0, HALF_D)
    cos_base = COS_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_base = SIN_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s

    c1 = tl.load(cos_base + offs).to(tl.float32)
    c2 = tl.load(cos_base + HALF_D + offs).to(tl.float32)
    s1 = tl.load(sin_base + offs).to(tl.float32)
    s2 = tl.load(sin_base + HALF_D + offs).to(tl.float32)

    if pid_h < n_qh:
        in_base = Q_ptr + b_idx * stride_qb + pid_h * stride_qh + s_idx * stride_qs
        out_base = QO_ptr + b_idx * stride_qob + pid_h * stride_qoh + s_idx * stride_qos
        x1 = tl.load(in_base + offs).to(tl.float32)
        x2 = tl.load(in_base + HALF_D + offs).to(tl.float32)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        tl.store(out_base + offs, y1.to(QO_ptr.dtype.element_ty))
        tl.store(out_base + HALF_D + offs, y2.to(QO_ptr.dtype.element_ty))
    else:
        kh = pid_h - n_qh
        in_base = K_ptr + b_idx * stride_kb + kh * stride_kh + s_idx * stride_ks
        out_base = KO_ptr + b_idx * stride_kob + kh * stride_koh + s_idx * stride_kos
        x1 = tl.load(in_base + offs).to(tl.float32)
        x2 = tl.load(in_base + HALF_D + offs).to(tl.float32)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        tl.store(out_base + offs, y1.to(KO_ptr.dtype.element_ty))
        tl.store(out_base + HALF_D + offs, y2.to(KO_ptr.dtype.element_ty))


@triton.jit
def triton_rope_qk_bwd_kernel(
    DQO_ptr,
    DKO_ptr,
    COS_ptr,
    SIN_ptr,
    DQ_ptr,
    DK_ptr,
    stride_dqob,
    stride_dqoh,
    stride_dqos,
    stride_dkob,
    stride_dkoh,
    stride_dkos,
    stride_cos_b,
    stride_cos_s,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dkb,
    stride_dkh,
    stride_dks,
    seq_len,
    n_qh,
    HALF_D: tl.constexpr,
):
    """Fused Q+K Rotary Position Embedding backward kernel (transpose orthogonal rotation)."""
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_idx = pid_bs // seq_len
    s_idx = pid_bs % seq_len

    offs = tl.arange(0, HALF_D)
    cos_base = COS_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_base = SIN_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s

    c1 = tl.load(cos_base + offs).to(tl.float32)
    c2 = tl.load(cos_base + HALF_D + offs).to(tl.float32)
    s1 = tl.load(sin_base + offs).to(tl.float32)
    s2 = tl.load(sin_base + HALF_D + offs).to(tl.float32)

    if pid_h < n_qh:
        in_base = DQO_ptr + b_idx * stride_dqob + pid_h * stride_dqoh + s_idx * stride_dqos
        out_base = DQ_ptr + b_idx * stride_dqb + pid_h * stride_dqh + s_idx * stride_dqs
        dy1 = tl.load(in_base + offs).to(tl.float32)
        dy2 = tl.load(in_base + HALF_D + offs).to(tl.float32)
        dx1 = dy1 * c1 + dy2 * s2
        dx2 = dy2 * c2 - dy1 * s1
        tl.store(out_base + offs, dx1.to(DQ_ptr.dtype.element_ty))
        tl.store(out_base + HALF_D + offs, dx2.to(DQ_ptr.dtype.element_ty))
    else:
        kh = pid_h - n_qh
        in_base = DKO_ptr + b_idx * stride_dkob + kh * stride_dkoh + s_idx * stride_dkos
        out_base = DK_ptr + b_idx * stride_dkb + kh * stride_dkh + s_idx * stride_dks
        dy1 = tl.load(in_base + offs).to(tl.float32)
        dy2 = tl.load(in_base + HALF_D + offs).to(tl.float32)
        dx1 = dy1 * c1 + dy2 * s2
        dx2 = dy2 * c2 - dy1 * s1
        tl.store(out_base + offs, dx1.to(DK_ptr.dtype.element_ty))
        tl.store(out_base + HALF_D + offs, dx2.to(DK_ptr.dtype.element_ty))


class _TritonRoPEQKFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q: [B, H_q, S, D], k: [B, H_k, S, D] with D contiguous (stride(-1) == 1)
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()

        batch, n_qh, seq_len, head_dim = q.shape
        n_kh = k.shape[1]
        half_d = head_dim // 2

        cos_c = cos.contiguous()
        sin_c = sin.contiguous()
        stride_cos_b = 0 if cos_c.shape[0] == 1 else cos_c.stride(0)
        stride_cos_s = cos_c.stride(1)

        q_out = torch.empty((batch, n_qh, seq_len, head_dim), dtype=q.dtype, device=q.device)
        k_out = torch.empty((batch, n_kh, seq_len, head_dim), dtype=k.dtype, device=k.device)

        grid = (batch * seq_len, n_qh + n_kh)
        triton_rope_qk_fwd_kernel[grid](
            q,
            k,
            cos_c,
            sin_c,
            q_out,
            k_out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            stride_cos_b,
            stride_cos_s,
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            seq_len,
            n_qh,
            HALF_D=half_d,
            num_warps=2,
        )

        ctx.save_for_backward(cos_c, sin_c)
        ctx.stride_cos_b = stride_cos_b
        ctx.stride_cos_s = stride_cos_s
        ctx.half_d = half_d
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out: torch.Tensor, dk_out: torch.Tensor):
        cos_c, sin_c = ctx.saved_tensors
        if dq_out.stride(-1) != 1:
            dq_out = dq_out.contiguous()
        if dk_out.stride(-1) != 1:
            dk_out = dk_out.contiguous()

        batch, n_qh, seq_len, head_dim = dq_out.shape
        n_kh = dk_out.shape[1]

        dq = torch.empty((batch, n_qh, seq_len, head_dim), dtype=dq_out.dtype, device=dq_out.device)
        dk = torch.empty((batch, n_kh, seq_len, head_dim), dtype=dk_out.dtype, device=dk_out.device)

        grid = (batch * seq_len, n_qh + n_kh)
        triton_rope_qk_bwd_kernel[grid](
            dq_out,
            dk_out,
            cos_c,
            sin_c,
            dq,
            dk,
            dq_out.stride(0),
            dq_out.stride(1),
            dq_out.stride(2),
            dk_out.stride(0),
            dk_out.stride(1),
            dk_out.stride(2),
            ctx.stride_cos_b,
            ctx.stride_cos_s,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            seq_len,
            n_qh,
            HALF_D=ctx.half_d,
            num_warps=2,
        )
        return dq, dk, None, None


def _pytorch_rope_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    half_q = q.shape[-1] // 2
    half_k = k.shape[-1] // 2
    q_rot = torch.cat((-q[..., half_q:], q[..., :half_q]), dim=-1)
    k_rot = torch.cat((-k[..., half_k:], k[..., :half_k]), dim=-1)
    return (q * cos_u) + (q_rot * sin_u), (k * cos_u) + (k_rot * sin_u)


def triton_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused Triton Rotary Position Embedding to ``q`` and ``k``."""
    head_dim = q.shape[-1]
    half_d = head_dim // 2
    if (
        not q.is_cuda
        or q.ndim != 4
        or k.ndim != 4
        or head_dim != k.shape[-1]
        or (head_dim % 2 != 0)
        or (half_d & (half_d - 1) != 0)
    ):
        return _pytorch_rope_fallback(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    if unsqueeze_dim == 2:
        # [B, S, H, D] -> transpose to [B, H, S, D], apply kernel, transpose back
        q_t, k_t = _TritonRoPEQKFunction.apply(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        return q_t.transpose(1, 2), k_t.transpose(1, 2)
    if unsqueeze_dim != 1:
        return _pytorch_rope_fallback(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    return _TritonRoPEQKFunction.apply(q, k, cos, sin)


_PATCHED_ROPE = False


def patch_qwen2_rope() -> None:
    """Monkey-patch ``modeling_qwen2.apply_rotary_pos_emb`` with ``triton_rope_qk``."""
    global _PATCHED_ROPE
    if _PATCHED_ROPE:
        return

    from transformers.models.qwen2 import modeling_qwen2

    modeling_qwen2.apply_rotary_pos_emb = triton_rope_qk
    _PATCHED_ROPE = True
