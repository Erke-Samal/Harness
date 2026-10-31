"""Pad-free training steps: layers see only the real tokens of a right-padded batch.

About 43% of the positions of a batch are padding. In causal self-attention a real token never
reads a later pad, so a pad cannot change anything the loss sees. Each batch keeps its own rows and
order: its N real tokens go through norms and linear layers as ``[1, N, W]``, and only attention
copies them to a ``[B, S, W]`` grid. ``triton_token_copy_kernel`` does that copy in one launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import triton
import triton.language as tl

_BLOCK = 256


@dataclass(frozen=True)
class TokenPacking:
    """Where the real tokens of a batch live on its ``[B, S]`` grid."""

    index: torch.Tensor  # [N], flat grid position of each real token
    lengths: torch.Tensor  # [B], int32 count of real tokens per row
    batch: int
    seq_len: int


@triton.jit
def triton_token_copy_kernel(
    Packed_ptr,
    Grid_ptr,
    Index_ptr,
    width,
    TO_GRID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one tile of one token between the packed ``[N, W]`` and the ``[B * S, W]`` layouts."""
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    packed = Packed_ptr + token * width + cols
    grid = Grid_ptr + tl.load(Index_ptr + token) * width + cols
    if TO_GRID:
        tl.store(grid, tl.load(packed, mask=mask, other=0.0), mask=mask)
    else:
        tl.store(packed, tl.load(grid, mask=mask, other=0.0), mask=mask)


def _copy_tokens(packed: torch.Tensor, grid: torch.Tensor, index: torch.Tensor, to_grid: bool) -> None:
    width = packed.shape[-1]
    triton_token_copy_kernel[(index.numel(), triton.cdiv(width, _BLOCK))](
        packed, grid, index, width, TO_GRID=to_grid, BLOCK=_BLOCK, num_warps=2
    )


def _to_grid(packed: torch.Tensor, packing: TokenPacking) -> torch.Tensor:
    grid = packed.new_zeros((packing.batch, packing.seq_len, packed.shape[-1]))
    _copy_tokens(packed, grid, packing.index, True)
    return grid


def _from_grid(grid: torch.Tensor, packing: TokenPacking) -> torch.Tensor:
    packed = torch.empty((1, packing.index.numel(), grid.shape[-1]), dtype=grid.dtype, device=grid.device)
    _copy_tokens(packed, grid.contiguous(), packing.index, False)
    return packed


class _UnpackTokensFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, packing):
        ctx.packing = packing
        return _to_grid(packed, packing)

    @staticmethod
    def backward(ctx, grad_grid):
        return _from_grid(grad_grid, ctx.packing), None


class _PackTokensFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, packing):
        ctx.packing = packing
        return _from_grid(grid, packing)

    @staticmethod
    def backward(ctx, grad_packed):
        return _to_grid(grad_packed.contiguous(), ctx.packing), None


def unpack_tokens(packed: torch.Tensor, packing: TokenPacking) -> torch.Tensor:
    """``[1, N, W]`` to ``[B, S, W]``, with zeros on the pad positions."""
    return _UnpackTokensFunction.apply(packed.contiguous(), packing)


def pack_tokens(grid: torch.Tensor, packing: TokenPacking) -> torch.Tensor:
    """``[B, S, W]`` to ``[1, N, W]``."""
    return _PackTokensFunction.apply(grid.contiguous(), packing)


def plan_token_packing(
    model: Any,
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> Optional[TokenPacking]:
    """A packing for a right-padded CUDA batch that the fp16 attention path can run, else None."""
    config = model.config
    if (
        not torch.is_tensor(input_ids)
        or not torch.is_tensor(attention_mask)
        or attention_mask.ndim != 2
        or attention_mask.shape[1] < 2
        or not attention_mask.is_cuda
        or model.get_input_embeddings().weight.dtype != torch.float16
        or config._attn_implementation != "sdpa"
        or config.attention_dropout > 0.0
        or config.num_attention_heads == config.num_key_value_heads
        or model.has_sliding_layers
    ):
        return None
    mask = attention_mask.bool()
    if not bool(mask[:, 0].all() and (mask[:, :-1] >= mask[:, 1:]).all()):
        return None
    index = mask.view(-1).nonzero().squeeze(1)
    return TokenPacking(index, mask.sum(1, dtype=torch.int32), attention_mask.shape[0], attention_mask.shape[1])
