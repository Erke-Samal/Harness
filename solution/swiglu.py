"""SwiGLU with compressed backward state, in one fused forward and one fused backward kernel.

Backward of ``silu(gate) * up`` needs two values per element: ``a = silu(gate)`` and
``b = silu'(gate) * up``. Autograd would keep ``gate`` and ``up`` in fp16, 35.8 KiB per token for
each of the 28 layers. ``triton_swiglu_compress_fwd_kernel`` keeps ``a`` and ``b`` instead, as int8
with one float scale per tile of ``BLOCK`` columns, in the spirit of block floating point. The
output is exact; only the gradient is rounded, by at most half a step of its own tile.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 256
_INT8_MAX = tl.constexpr(127.0)


@triton.jit
def triton_swiglu_compress_fwd_kernel(
    Gate_ptr,
    Up_ptr,
    Out_ptr,
    A_ptr,
    B_ptr,
    AScale_ptr,
    BScale_ptr,
    n_cols,
    n_tiles,
    BLOCK: tl.constexpr,
):
    """One tile of one row: exact output, plus int8 ``a`` and ``b`` with their tile scales."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row * n_cols + cols

    gate = tl.load(Gate_ptr + base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up_ptr + base, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    a = gate * sig
    b = (sig + a * (1.0 - sig)) * up
    tl.store(Out_ptr + base, (a * up).to(Out_ptr.dtype.element_ty), mask=mask)

    a_scale = tl.max(tl.abs(a), axis=0)
    b_scale = tl.max(tl.abs(b), axis=0)
    tl.store(AScale_ptr + row * n_tiles + tile, a_scale)
    tl.store(BScale_ptr + row * n_tiles + tile, b_scale)

    a_step = tl.where(a_scale > 0.0, _INT8_MAX / a_scale, 0.0)
    b_step = tl.where(b_scale > 0.0, _INT8_MAX / b_scale, 0.0)
    tl.store(A_ptr + base, tl.floor(a * a_step + 0.5).to(tl.int8), mask=mask)
    tl.store(B_ptr + base, tl.floor(b * b_step + 0.5).to(tl.int8), mask=mask)


@triton.jit
def triton_swiglu_decompress_bwd_kernel(
    DOut_ptr,
    A_ptr,
    B_ptr,
    AScale_ptr,
    BScale_ptr,
    DGate_ptr,
    DUp_ptr,
    n_cols,
    n_tiles,
    BLOCK: tl.constexpr,
):
    """One tile of one row: ``dgate = dout * b`` and ``dup = dout * a``."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row * n_cols + cols

    dout = tl.load(DOut_ptr + base, mask=mask, other=0.0).to(tl.float32)
    a_step = tl.load(AScale_ptr + row * n_tiles + tile) / _INT8_MAX
    b_step = tl.load(BScale_ptr + row * n_tiles + tile) / _INT8_MAX
    a = tl.load(A_ptr + base, mask=mask, other=0).to(tl.float32) * a_step
    b = tl.load(B_ptr + base, mask=mask, other=0).to(tl.float32) * b_step

    tl.store(DGate_ptr + base, (dout * b).to(DGate_ptr.dtype.element_ty), mask=mask)
    tl.store(DUp_ptr + base, (dout * a).to(DUp_ptr.dtype.element_ty), mask=mask)


class _TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        n_cols = gate.shape[-1]
        gate_2d = gate.contiguous().view(-1, n_cols)
        up_2d = up.contiguous().view(-1, n_cols)
        n_rows = gate_2d.shape[0]
        n_tiles = triton.cdiv(n_cols, _BLOCK)
        device = gate.device

        out_2d = torch.empty_like(gate_2d)
        a = torch.empty((n_rows, n_cols), dtype=torch.int8, device=device)
        b = torch.empty((n_rows, n_cols), dtype=torch.int8, device=device)
        a_scale = torch.empty((n_rows, n_tiles), dtype=torch.float32, device=device)
        b_scale = torch.empty((n_rows, n_tiles), dtype=torch.float32, device=device)

        triton_swiglu_compress_fwd_kernel[(n_rows, n_tiles)](
            gate_2d, up_2d, out_2d, a, b, a_scale, b_scale, n_cols, n_tiles, BLOCK=_BLOCK, num_warps=2
        )

        ctx.compressed = (a, b, a_scale, b_scale)
        ctx.shape = gate.shape
        ctx.dtype = gate.dtype
        return out_2d.view(gate.shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b, a_scale, b_scale = ctx.compressed
        n_rows, n_cols = a.shape
        n_tiles = a_scale.shape[1]

        dout_2d = grad_out.contiguous().view(n_rows, n_cols)
        dgate = torch.empty_like(dout_2d)
        dup = torch.empty_like(dout_2d)

        triton_swiglu_decompress_bwd_kernel[(n_rows, n_tiles)](
            dout_2d, a, b, a_scale, b_scale, dgate, dup, n_cols, n_tiles, BLOCK=_BLOCK, num_warps=2
        )
        return dgate.view(ctx.shape), dup.view(ctx.shape)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute ``F.silu(gate) * up`` with int8 backward state on CUDA."""
    if not gate.is_cuda:
        return torch.nn.functional.silu(gate) * up
    return _TritonSwiGLUFunction.apply(gate, up)


_PATCHED_SWIGLU = False


def patch_qwen2_mlp() -> None:
    """Monkey-patch ``Qwen2MLP.forward`` to use the fused SwiGLU kernels."""
    global _PATCHED_SWIGLU
    if _PATCHED_SWIGLU:
        return

    from transformers.models.qwen2 import modeling_qwen2

    def _fast_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = _fast_mlp_forward
    _PATCHED_SWIGLU = True
