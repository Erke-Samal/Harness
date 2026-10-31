"""Compile the kernels while Soup loads the model, instead of during the first training steps.

Triton compiles a kernel at its first launch, so the first steps otherwise pay for it one kernel at
a time. Each launch here uses the dtypes and shape constants of a real step on tiny tensors, which
hits the same compiled variant. Nothing is cached between runs, and the thread lives inside the
measured ``soup train`` process. If it fails, the real launches simply compile as usual.
"""

from __future__ import annotations

import threading

import torch

from solution.attention_triton import triton_causal_gqa_attention
from solution.cross_entropy_triton import CE_CHUNK, triton_chunked_causal_lm_loss
from solution.nf4_triton import triton_nf4_dequant_kernel
from solution.rmsnorm_triton import triton_rmsnorm
from solution.rope_triton import triton_rope_qk
from solution.swiglu import swiglu
from solution.token_packing import TokenPacking, pack_tokens, unpack_tokens

_HIDDEN, _HEAD_DIM, _MLP_WIDTH = 1536, 128, 8960
_N_HEADS, _N_KV_HEADS = 12, 2
_NF4_BLOCK_SHIFT, _NF4_STATE2_SHIFT = 6, 8  # bitsandbytes defaults: blocks of 64 and 256


def _tensor(shape: tuple[int, ...], device: torch.device, grad: bool = True) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.float16, device=device, requires_grad=grad)


def compile_kernels(device: torch.device) -> None:
    """Launch every kernel once, forward and backward, on tiny fp16 tensors."""
    seq_len = 2
    index = torch.arange(seq_len, device=device)
    packing = TokenPacking(index, torch.tensor([seq_len], dtype=torch.int32, device=device), 1, seq_len)

    x = _tensor((1, seq_len, _HIDDEN), device)
    weight = torch.ones(_HIDDEN, dtype=torch.float32, device=device)  # norms stay in fp32
    pack_tokens(unpack_tokens(triton_rmsnorm(x, weight), packing), packing).sum().backward()

    gate, up = _tensor((1, seq_len, _MLP_WIDTH), device), _tensor((1, seq_len, _MLP_WIDTH), device)
    swiglu(gate, up).sum().backward()

    q = _tensor((1, _N_HEADS, seq_len, _HEAD_DIM), device)
    k = _tensor((1, _N_KV_HEADS, seq_len, _HEAD_DIM), device)
    v = _tensor((1, _N_KV_HEADS, seq_len, _HEAD_DIM), device)
    cos = _tensor((1, seq_len, _HEAD_DIM), device, grad=False)
    q_rot, k_rot = triton_rope_qk(q, k, cos, cos)
    triton_causal_gqa_attention(q_rot, k_rot, v, _HEAD_DIM**-0.5).float().sum().backward()

    hidden = _tensor((1, seq_len, 16), device)
    head = _tensor((16, 16), device, grad=False)
    labels = torch.zeros((1, seq_len), dtype=torch.int64, device=device)
    triton_chunked_causal_lm_loss(hidden, head, labels, chunk_size=CE_CHUNK)[0].backward()

    qweight = torch.zeros(4, dtype=torch.uint8, device=device)
    float_table = torch.zeros(16, dtype=torch.float32, device=device)
    out = torch.empty(4, dtype=torch.int32, device=device)
    triton_nf4_dequant_kernel[(1,)](
        qweight, float_table, qweight, float_table, float_table, float_table, out, 0.0, 4,
        SHIFT_B1=_NF4_BLOCK_SHIFT, SHIFT_B2=_NF4_STATE2_SHIFT, NESTED=True, BLOCK_BYTES=512, num_warps=4,
    )


def start_compiling() -> None:
    """Compile in a daemon thread of this process when a GPU is present."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        threading.Thread(target=compile_kernels, args=(device,), daemon=True).start()
