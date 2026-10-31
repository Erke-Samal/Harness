"""Custom Triton RMSNorm forward and backward kernels for Qwen2.

Fixes a critical precision/memory pitfall in HuggingFace Qwen2 under QLoRA:
``prepare_model_for_kbit_training`` casts ``Qwen2RMSNorm.weight`` to ``float32``,
and ``Qwen2RMSNorm.forward`` computes ``self.weight * hidden_states.to(input_dtype)``,
which silently promotes ``float16`` hidden states to ``float32`` on every RMSNorm
call (57 times per forward pass).

Our ``triton_rmsnorm_fwd_kernel`` and ``triton_rmsnorm_bwd_kernel`` compute the
normalization and affine scale in ``float32`` registers for full numerical
precision, and write the output in ``x.dtype`` (``float16``). Backward needs ``x / rms``,
which is ``y / weight``, so autograd keeps ``y`` instead of ``x``: the input norm's ``y``
is kept by the LoRA layers anyway, and ``x`` can be freed with the residual. Only the 1D
inverse-RMS vector ``rstd`` (``[M]`` in ``float32``) is added.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def triton_rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    RSTD_ptr,
    stride_x_row,
    stride_y_row,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Single-pass row-parallel RMSNorm forward kernel in fp32 registers."""
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x_ptrs = X_ptr + row_idx * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(mean_sq + eps)
    tl.store(RSTD_ptr + row_idx, rstd)

    y = (x * rstd) * w
    y_ptrs = Y_ptr + row_idx * stride_y_row + cols
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def triton_rmsnorm_bwd_kernel(
    DY_ptr,
    Y_ptr,
    W_ptr,
    RSTD_ptr,
    DX_ptr,
    DW_ptr,
    stride_dy_row,
    stride_y_row,
    stride_dx_row,
    n_cols,
    HAS_DW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Row-parallel RMSNorm backward kernel computing dx (and optional dw via fp32 atomic_add)."""
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    dy = tl.load(DY_ptr + row_idx * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row_idx * stride_y_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_idx).to(tl.float32)

    x_hat = tl.where(w != 0.0, y / w, 0.0)
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
    dx = (wdy - x_hat * c1) * rstd

    tl.store(DX_ptr + row_idx * stride_dx_row + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)

    if HAS_DW:
        dw_row = dy * x_hat
        tl.atomic_add(DW_ptr + cols, dw_row, mask=mask)


def _pick_rmsnorm_launch_params(n_cols: int) -> tuple[int, int]:
    block_n = triton.next_power_of_2(n_cols)
    if block_n <= 1024:
        num_warps = 4
    elif block_n <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    return block_n, num_warps


class _TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x_2d = x.contiguous().view(-1, n_cols)
        w_1d = weight.contiguous().view(-1)
        n_rows = x_2d.shape[0]

        y_2d = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        block_n, num_warps = _pick_rmsnorm_launch_params(n_cols)
        triton_rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            w_1d,
            y_2d,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            n_cols,
            float(eps),
            BLOCK_N=block_n,
            num_warps=num_warps,
        )

        ctx.save_for_backward(y_2d, w_1d, rstd)
        ctx.orig_shape = orig_shape
        ctx.n_cols = n_cols
        ctx.block_n = block_n
        ctx.num_warps = num_warps
        ctx.weight_dtype = weight.dtype
        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        y_2d, w_1d, rstd = ctx.saved_tensors
        n_cols = ctx.n_cols
        n_rows = y_2d.shape[0]

        dy_2d = grad_output.contiguous().view(-1, n_cols)
        dx_2d = torch.empty_like(y_2d)

        need_dw = bool(ctx.needs_input_grad[1])
        if need_dw:
            dw_f32 = torch.zeros(n_cols, dtype=torch.float32, device=y_2d.device)
        else:
            dw_f32 = rstd  # dummy pointer, unused when HAS_DW=False

        triton_rmsnorm_bwd_kernel[(n_rows,)](
            dy_2d,
            y_2d,
            w_1d,
            rstd,
            dx_2d,
            dw_f32,
            dy_2d.stride(0),
            y_2d.stride(0),
            dx_2d.stride(0),
            n_cols,
            HAS_DW=need_dw,
            BLOCK_N=ctx.block_n,
            num_warps=ctx.num_warps,
        )

        grad_w = dw_f32.to(ctx.weight_dtype) if need_dw else None
        return dx_2d.view(ctx.orig_shape), grad_w, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply fused Triton RMSNorm, preserving ``x.dtype`` on output."""
    if not x.is_cuda or x.shape[-1] > 16384:
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(var + eps)
        return (weight.to(torch.float32) * x_norm).to(in_dtype)
    return _TritonRMSNormFunction.apply(x, weight, float(eps))


_PATCHED_RMSNORM = False


def patch_qwen2_rmsnorm() -> None:
    """Monkey-patch ``Qwen2RMSNorm.forward`` to use ``triton_rmsnorm``."""
    global _PATCHED_RMSNORM
    if _PATCHED_RMSNORM:
        return

    from transformers.models.qwen2 import modeling_qwen2

    def _fast_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = _fast_rmsnorm_forward
    _PATCHED_RMSNORM = True
