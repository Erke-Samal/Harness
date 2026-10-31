"""Custom Triton NF4/FP4 double-dequantization kernel and fused QLoRA linear ops.

Replaces bitsandbytes' multi-kernel 4-bit dequantization fallback:
  1. ``_dequantize_blockwise_impl`` (dequantize nested absmax uint8 -> fp32)
  2. ``absmax += offset`` (elementwise add kernel)
  3. ``_dequantize_4bit_impl`` (unpack 4-bit nibbles and scale)
with a single fused ``@triton.jit`` kernel ``triton_nf4_dequant_kernel`` that:
  - Performs nested double-dequantization of ``absmax`` in registers.
  - Unpacks two 4-bit indices per byte (high nibble = even element, low nibble = odd).
  - Looks up the 16-entry NF4/FP4 codebook in registers, multiplies by the block scale
    in fp32, and writes pairs of fp16 values via coalesced 32-bit stores.

Also provides custom autograd functions for ``bnb.nn.Linear4bit`` and
``peft.tuners.lora.bnb.Linear4bit`` that eliminate redundant fp32<->fp16 round-trip
casts, remove the unnecessary ``result.clone()`` allocation in PEFT QLoRA forward,
and redraw peft's LoRA dropout mask in backward instead of storing it.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from solution.token_packing import TokenPacking, active_packing

_SHIFT_MAP = {
    32: 5,
    64: 6,
    128: 7,
    256: 8,
    512: 9,
    1024: 10,
    2048: 11,
    4096: 12,
}

_BLOCK_BYTES = 512
_ARGS_ATTR = "_skyxplorers_launch_args"
_HOST_ATTR = "_skyxplorers_host_weight"

# Cache for per-device NF4 / FP4 float32 codebooks and temporary dequant buffer.
_CODEBOOK_CACHE: dict[tuple[torch.device, str], torch.Tensor] = {}
_DEQUANT_SCRATCH: dict[torch.device, torch.Tensor] = {}
_SCRATCH_VIEWS: dict[tuple[torch.device, tuple[int, ...]], torch.Tensor] = {}


def _get_codebook(quant_type: str, device: torch.device, fallback_code: Optional[torch.Tensor]) -> torch.Tensor:
    key = (device, quant_type)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached
    if fallback_code is not None:
        cb = fallback_code.to(device=device, dtype=torch.float32).contiguous()
        _CODEBOOK_CACHE[key] = cb
        return cb
    import bitsandbytes.functional as bnb_F

    cb = bnb_F.get_4bit_type(quant_type, device=device).to(dtype=torch.float32).contiguous()
    _CODEBOOK_CACHE[key] = cb
    return cb


def _get_dequant_scratch(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    view = _SCRATCH_VIEWS.get((device, shape))
    if view is not None:
        return view
    n_elems = math.prod(shape)
    buf = _DEQUANT_SCRATCH.get(device)
    if buf is None or buf.numel() < n_elems:
        # Allocate at least 8960 * 1536 elements (26.25 MiB) to avoid reallocations.
        alloc_elems = max(n_elems, 8960 * 1536)
        buf = torch.empty(alloc_elems, dtype=torch.float16, device=device)
        _DEQUANT_SCRATCH[device] = buf
        _SCRATCH_VIEWS.clear()
    view = buf[:n_elems].view(shape)
    _SCRATCH_VIEWS[(device, shape)] = view
    return view


def _weight_device(quant_state: Any) -> torch.device:
    """Where the weight is unpacked: its state stays on the GPU when the packed bytes move to the host."""
    return quant_state.absmax.device


class WeightStream:
    """The packed bytes of the decoder layers, one pinned slab on the host and two slots on the GPU.

    The layer in use holds one slot; the next one, in the direction of travel, arrives in the other
    on its own CUDA stream while the kernels run. A copy starts only after the kernels queued so far
    have finished, and the kernels of a layer start only after its copy has. The scheme of keeping
    parameters on the host and prefetching the next block on a separate stream is from MegaTrain
    (arXiv:2604.05091) and SuperNeurons (arXiv:1801.04380).
    """

    def __init__(self, slab: torch.Tensor, spans: list[tuple[int, int]], device: torch.device) -> None:
        self.slab = slab
        self.spans = spans
        slot_bytes = max(end - start for start, end in spans)
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8, device=device) for _ in range(2)]
        self.owner: list[Optional[int]] = [None, None]
        self.ready: list[Any] = [None, None]
        self.stream = torch.cuda.Stream(device)
        self.layer: Optional[int] = None
        self.slot = 0

    def _copy(self, layer: int, slot: int) -> None:
        queued = torch.cuda.Event()
        queued.record(torch.cuda.current_stream())
        start, end = self.spans[layer]
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(queued)
            self.slots[slot][: end - start].copy_(self.slab[start:end], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        self.owner[slot] = layer
        self.ready[slot] = ready

    def _enter(self, layer: int) -> None:
        going_up = self.layer is None or layer > self.layer
        if layer in self.owner:
            self.slot = self.owner.index(layer)
        else:
            self.slot = 0 if self.layer is None else 1 - self.slot
            self._copy(layer, self.slot)
        torch.cuda.current_stream().wait_event(self.ready[self.slot])
        self.layer = layer
        following = layer + 1 if going_up else layer - 1
        if 0 <= following < len(self.spans) and self.owner[1 - self.slot] != following:
            self._copy(following, 1 - self.slot)

    def packed(self, layer: int, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
        if layer != self.layer:
            self._enter(layer)
        return self.slots[self.slot][offset : offset + math.prod(shape)].view(shape)


def move_packed_weight_to_host(weight: torch.Tensor) -> None:
    """Right after quantization, before the slab can be sized."""
    weight.data = weight.data.to("cpu")
    setattr(weight.quant_state, _HOST_ATTR, True)


def build_weight_stream(model: torch.nn.Module, device: torch.device) -> None:
    """Gather the packed weights of the decoder layers into one pinned slab; values do not change."""
    layers: dict[int, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if getattr(getattr(param, "quant_state", None), _HOST_ATTR, None) is None:
            continue
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None:
            param.data = param.data.to(device)
            setattr(param.quant_state, _HOST_ATTR, None)
            continue
        layers.setdefault(int(match.group(1)), []).append(param)

    total = sum(p.data.nbytes for params in layers.values() for p in params)
    slab = torch.empty(total, dtype=torch.uint8).pin_memory()
    spans = []
    used = 0
    places = []
    for number, layer in enumerate(sorted(layers)):
        start = used
        for param in layers[layer]:
            n_bytes = param.data.nbytes
            slab[used : used + n_bytes].copy_(param.data.view(-1).view(torch.uint8))
            places.append((param, number, used - start, used, n_bytes))
            used += n_bytes
        spans.append((start, used))
    stream = WeightStream(slab, spans, device)
    for param, number, offset, start, n_bytes in places:
        shape = tuple(param.data.shape)
        param.data = slab[start : start + n_bytes].view(param.data.dtype).view(shape)
        setattr(param.quant_state, _HOST_ATTR, (stream, number, offset, shape))


def restore_packed_weight(weight: torch.Tensor) -> None:
    weight.data = weight.data.to(_weight_device(weight.quant_state))
    setattr(weight.quant_state, _HOST_ATTR, None)


@triton.jit(do_not_specialize=["n_bytes"])
def triton_nf4_dequant_kernel(
    qweight_ptr,
    absmax_ptr,
    qabsmax_ptr,
    state2_absmax_ptr,
    state2_code_ptr,
    nf4_code_ptr,
    out_i32_ptr,
    offset,
    n_bytes,
    SHIFT_B1: tl.constexpr,
    SHIFT_B2: tl.constexpr,
    NESTED: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Fused 4-bit NF4/FP4 (nested or flat) dequantization kernel writing fp16x2 as int32."""
    pid = tl.program_id(0)
    byte_offs = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    mask = byte_offs < n_bytes

    packed = tl.load(qweight_ptr + byte_offs, mask=mask, other=0).to(tl.int32)
    hi_idx = (packed >> 4) & 0xF
    lo_idx = packed & 0xF

    # Each packed byte holds 2 elements -> 1st-level block of B1 elements spans (B1 >> 1) bytes.
    block_idx = byte_offs >> (SHIFT_B1 - 1)

    if NESTED:
        qabs = tl.load(qabsmax_ptr + block_idx, mask=mask, other=0).to(tl.int32)
        code_val = tl.load(state2_code_ptr + qabs, mask=mask, other=0.0).to(tl.float32)
        super_idx = block_idx >> SHIFT_B2
        super_abs = tl.load(state2_absmax_ptr + super_idx, mask=mask, other=0.0).to(tl.float32)
        scale = code_val * super_abs + offset
    else:
        scale = tl.load(absmax_ptr + block_idx, mask=mask, other=0.0).to(tl.float32)

    val_even = tl.load(nf4_code_ptr + hi_idx, mask=mask, other=0.0).to(tl.float32) * scale
    val_odd = tl.load(nf4_code_ptr + lo_idx, mask=mask, other=0.0).to(tl.float32) * scale

    even_f16 = val_even.to(tl.float16)
    odd_f16 = val_odd.to(tl.float16)

    even_bits = tl.cast(even_f16, tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    odd_bits = tl.cast(odd_f16, tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    packed_f16x2 = even_bits | (odd_bits << 16)

    tl.store(out_i32_ptr + byte_offs, packed_f16x2, mask=mask)


def _launch_args(A: torch.Tensor, quant_state: Any) -> Optional[tuple]:
    """Kernel arguments of one weight, or None if the kernel cannot unpack it.

    A weight is unpacked hundreds of times per step and its state never changes, so they are built
    once. Above all this reads ``offset`` from the GPU once: ``.item()`` waits for the whole queue.
    """
    cached = getattr(quant_state, _ARGS_ATTR, None)
    if cached is None:
        cached = {}
        setattr(quant_state, _ARGS_ATTR, cached)
    known = cached.get(A.data_ptr())
    if known is not None:
        return known

    b1 = int(quant_state.blocksize)
    nested = bool(quant_state.nested)
    b2 = int(quant_state.state2.blocksize) if nested else 64
    if not A.is_cuda or b1 not in _SHIFT_MAP or (nested and b2 not in _SHIFT_MAP):
        return None

    qweight = A.contiguous().view(-1)
    n_bytes = qweight.numel()
    nf4_code = _get_codebook(quant_state.quant_type, A.device, getattr(quant_state, "code", None))
    if nested:
        state2_absmax = quant_state.state2.absmax.contiguous().view(-1)
        state2_code = quant_state.state2.code.contiguous().view(-1)
        qabsmax = quant_state.absmax.contiguous().view(-1)
        absmax = state2_absmax
        offset = quant_state.offset
        offset = float(offset.item() if torch.is_tensor(offset) else offset or 0.0)
    else:
        absmax = state2_absmax = state2_code = quant_state.absmax.contiguous().view(-1)
        qabsmax = qweight
        offset = 0.0

    constants = {
        "SHIFT_B1": _SHIFT_MAP[b1],
        "SHIFT_B2": _SHIFT_MAP[b2],
        "NESTED": nested,
        "BLOCK_BYTES": _BLOCK_BYTES,
        "num_warps": 4,
    }
    grid = (triton.cdiv(n_bytes, _BLOCK_BYTES),)
    args = (A.data_ptr(), grid, qweight, absmax, qabsmax, state2_absmax, state2_code, nf4_code, offset, n_bytes, constants)
    cached[A.data_ptr()] = args
    return args


def triton_dequantize_nf4(
    A: torch.Tensor,
    quant_state: Any = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type: str = "fp4",
) -> torch.Tensor:
    """Dequantize a bitsandbytes 4-bit tensor using ``triton_nf4_dequant_kernel``.

    Same signature as ``bitsandbytes.functional.dequantize_4bit``, which it replaces.
    """
    if quant_state is None:
        return _orig_dequantize_4bit(A, None, absmax, out, blocksize, quant_type)

    host = getattr(quant_state, _HOST_ATTR, None)
    if host is not None:
        stream, layer, offset, shape = host
        A = stream.packed(layer, offset, shape)

    launch = _launch_args(A, quant_state)
    if launch is None:
        return _orig_dequantize_4bit(A, quant_state, absmax, out, blocksize, quant_type)

    target_dtype = quant_state.dtype if out is None else out.dtype

    if out is not None and out.dtype == torch.float16 and out.is_contiguous():
        out_fp16 = out
    else:
        out_fp16 = torch.empty(tuple(quant_state.shape), dtype=torch.float16, device=A.device)

    _, grid, qweight, absmax_f32, qabsmax, state2_absmax, state2_code, nf4_code, offset, n_bytes, constants = launch
    triton_nf4_dequant_kernel[grid](
        qweight,
        absmax_f32,
        qabsmax,
        state2_absmax,
        state2_code,
        nf4_code,
        out_fp16.view(-1).view(torch.int32),
        offset,
        n_bytes,
        **constants,
    )

    if out is not None and out.data_ptr() != out_fp16.data_ptr():
        out.copy_(out_fp16.to(out.dtype))
        return out
    if target_dtype != torch.float16:
        return out_fp16.to(target_dtype)
    return out_fp16


def _default_generator(device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        return torch.cuda.default_generators[device.index or 0]
    return torch.default_generator


def _dropout_factor(dropout: torch.nn.Module, x: torch.Tensor, dtype: torch.dtype, packing: Optional[TokenPacking]) -> torch.Tensor:
    """What peft's own LoRA dropout multiplies ``x`` by, drawn by it from the default generator.

    The base draws on the ``[B, S, W]`` grid in the LoRA dtype, so this does too, and a packed step
    then keeps its real tokens: the masks and the generator's state stay those of the base run.
    """
    width = x.shape[-1]
    shape = (packing.batch, packing.seq_len, width) if packing is not None else x.shape
    noise = dropout(torch.ones(shape, dtype=dtype, device=x.device)).view(-1, width)
    if packing is not None:
        noise = noise[packing.index]
    return noise.to(x.dtype)


def _redraw_dropout_factor(ctx: Any, x: torch.Tensor) -> torch.Tensor:
    """The same mask again for backward, from the state saved in forward, so no mask is stored."""
    generator = _default_generator(x.device)
    current = generator.get_state()
    generator.set_state(ctx.rng_state)
    factor = _dropout_factor(ctx.dropout, x, ctx.lora_dtype, ctx.packing)
    generator.set_state(current)
    return factor


class _TritonBNBLinearFunction(torch.autograd.Function):
    """Autograd function for non-LoRA 4-bit Linear layers (k_proj, o_proj, gate_proj, up_proj, down_proj)."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qweight: torch.Tensor,
        bias: Optional[torch.Tensor],
        quant_state: Any,
    ) -> torch.Tensor:
        compute_dtype = torch.float16
        x_fp16 = x.to(compute_dtype) if x.dtype != compute_dtype else x
        x_2d = x_fp16.reshape(-1, x_fp16.shape[-1]).contiguous()

        w_scratch = _get_dequant_scratch(tuple(quant_state.shape), _weight_device(quant_state))
        w_deq = triton_dequantize_nf4(qweight, quant_state, out=w_scratch)
        out_2d = torch.mm(x_2d, w_deq.t())

        if bias is not None:
            bias_cast = bias.to(out_2d.dtype) if bias.dtype != out_2d.dtype else bias
            out_2d = out_2d + bias_cast

        ctx.save_for_backward(qweight)
        ctx.quant_state = quant_state
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        ctx.has_bias = bias is not None
        ctx.bias_requires_grad = bool(bias is not None and bias.requires_grad)
        ctx.bias_dtype = bias.dtype if bias is not None else None

        out = out_2d.view(*x.shape[:-1], out_2d.shape[-1])
        return out.to(x.dtype) if out.dtype != x.dtype else out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (qweight,) = ctx.saved_tensors
        quant_state = ctx.quant_state
        compute_dtype = ctx.compute_dtype

        grad_fp16 = grad_output.to(compute_dtype) if grad_output.dtype != compute_dtype else grad_output
        grad_2d = grad_fp16.reshape(-1, grad_fp16.shape[-1]).contiguous()

        grad_x = None
        if ctx.needs_input_grad[0]:
            w_scratch = _get_dequant_scratch(tuple(quant_state.shape), _weight_device(quant_state))
            w_deq = triton_dequantize_nf4(qweight, quant_state, out=w_scratch)
            grad_x_2d = torch.mm(grad_2d, w_deq)
            grad_x = grad_x_2d.view(*grad_output.shape[:-1], grad_x_2d.shape[-1])
            if grad_x.dtype != ctx.input_dtype:
                grad_x = grad_x.to(ctx.input_dtype)

        grad_bias = None
        if ctx.has_bias and ctx.bias_requires_grad and ctx.needs_input_grad[2]:
            grad_bias = grad_2d.sum(dim=0).to(ctx.bias_dtype)

        return grad_x, None, grad_bias, None


class _TritonQLoRALinearFunction(torch.autograd.Function):
    """Fused autograd function for PEFT QLoRA Linear4bit (base 4-bit + LoRA branch)."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qweight: torch.Tensor,
        bias: Optional[torch.Tensor],
        quant_state: Any,
        lora_A_weight: torch.Tensor,
        lora_B_weight: torch.Tensor,
        scaling: float,
        dropout: Optional[torch.nn.Module],
        packing: Optional[TokenPacking],
    ) -> torch.Tensor:
        compute_dtype = torch.float16
        x_fp16 = x.to(compute_dtype) if x.dtype != compute_dtype else x
        x_2d = x_fp16.reshape(-1, x_fp16.shape[-1]).contiguous()

        # 1. Base 4-bit projection
        w_scratch = _get_dequant_scratch(tuple(quant_state.shape), _weight_device(quant_state))
        w_deq = triton_dequantize_nf4(qweight, quant_state, out=w_scratch)
        out_2d = torch.mm(x_2d, w_deq.t())
        if bias is not None:
            bias_cast = bias.to(out_2d.dtype) if bias.dtype != out_2d.dtype else bias
            out_2d = out_2d + bias_cast

        # 2. LoRA branch (runs on fp16 Tensor Cores while keeping master lora_A/B in fp32)
        ctx.dropout = dropout
        if dropout is not None:
            ctx.rng_state = _default_generator(x.device).get_state()
            ctx.lora_dtype = lora_A_weight.dtype
            ctx.packing = packing
            x_dropped = x_2d * _dropout_factor(dropout, x_2d, lora_A_weight.dtype, packing)
        else:
            x_dropped = x_2d

        lora_A_fp16 = lora_A_weight.to(compute_dtype)
        lora_B_fp16 = lora_B_weight.to(compute_dtype)

        # Save h_A = x_dropped @ A.T ([M, r=16], only 64 KiB) to avoid recomputing in backward
        h_a = torch.mm(x_dropped, lora_A_fp16.t())
        out_2d.addmm_(h_a, lora_B_fp16.t(), alpha=float(scaling))

        ctx.save_for_backward(x_2d, h_a, qweight, lora_A_weight, lora_B_weight)
        ctx.quant_state = quant_state
        ctx.scaling = float(scaling)
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        ctx.has_bias = bias is not None
        ctx.bias_requires_grad = bool(bias is not None and bias.requires_grad)
        ctx.bias_dtype = bias.dtype if bias is not None else None

        out = out_2d.view(*x.shape[:-1], out_2d.shape[-1])
        return out.to(x.dtype) if out.dtype != x.dtype else out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, h_a, qweight, lora_A_weight, lora_B_weight = ctx.saved_tensors
        quant_state = ctx.quant_state
        scaling = ctx.scaling
        compute_dtype = ctx.compute_dtype

        grad_fp16 = grad_output.to(compute_dtype) if grad_output.dtype != compute_dtype else grad_output
        grad_2d = grad_fp16.reshape(-1, grad_fp16.shape[-1]).contiguous()

        lora_A_fp16 = lora_A_weight.to(compute_dtype)
        lora_B_fp16 = lora_B_weight.to(compute_dtype)

        if ctx.dropout is not None:
            drop_factor = _redraw_dropout_factor(ctx, x_2d)
            x_dropped = x_2d * drop_factor
        else:
            drop_factor = None
            x_dropped = x_2d

        # grad_lora_B = scaling * (grad_2d.T @ h_a) -> cast to master dtype (fp32)
        grad_lora_B = None
        if ctx.needs_input_grad[5]:
            grad_b_fp16 = torch.mm(grad_2d.t(), h_a)
            grad_lora_B = grad_b_fp16.to(lora_B_weight.dtype) * scaling

        # grad_h_a = scaling * (grad_2d @ lora_B) -> [M, r=16]
        grad_h_a = torch.mm(grad_2d, lora_B_fp16) * scaling

        # grad_lora_A = grad_h_a.T @ x_dropped -> cast to master dtype (fp32)
        grad_lora_A = None
        if ctx.needs_input_grad[4]:
            grad_a_fp16 = torch.mm(grad_h_a.t(), x_dropped)
            grad_lora_A = grad_a_fp16.to(lora_A_weight.dtype)

        # grad_x = grad_2d @ W_deq + (grad_h_a @ lora_A) * drop_factor
        grad_x = None
        if ctx.needs_input_grad[0]:
            w_scratch = _get_dequant_scratch(tuple(quant_state.shape), _weight_device(quant_state))
            w_deq = triton_dequantize_nf4(qweight, quant_state, out=w_scratch)
            grad_x_2d = torch.mm(grad_2d, w_deq)
            grad_x_lora = torch.mm(grad_h_a, lora_A_fp16)
            if drop_factor is not None:
                grad_x_lora = grad_x_lora * drop_factor
            grad_x_2d.add_(grad_x_lora)
            grad_x = grad_x_2d.view(*grad_output.shape[:-1], grad_x_2d.shape[-1])
            if grad_x.dtype != ctx.input_dtype:
                grad_x = grad_x.to(ctx.input_dtype)

        grad_bias = None
        if ctx.has_bias and ctx.bias_requires_grad and ctx.needs_input_grad[2]:
            grad_bias = grad_2d.sum(dim=0).to(ctx.bias_dtype)

        return grad_x, None, grad_bias, None, grad_lora_A, grad_lora_B, None, None, None


_orig_dequantize_4bit = None
_orig_bnb_linear4bit_forward = None
_orig_peft_linear4bit_forward = None
_PATCHED_NF4 = False


def _fast_bnb_linear4bit_forward(self, x: torch.Tensor) -> torch.Tensor:
    import bitsandbytes as bnb

    bnb.matmul_4bit._is_called = True
    if getattr(self.weight, "quant_state", None) is None:
        return _orig_bnb_linear4bit_forward(self, x)
    if not x.is_cuda or self.compute_dtype not in (None, torch.float16):
        return _orig_bnb_linear4bit_forward(self, x)
    return _TritonBNBLinearFunction.apply(x, self.weight, self.bias, self.weight.quant_state)


def _fast_peft_linear4bit_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)

    if (
        self.disable_adapters
        or kwargs.get("adapter_names") is not None
        or self.merged
        or len(self.active_adapters) != 1
    ):
        return _orig_peft_linear4bit_forward(self, x, *args, **kwargs)

    active_adapter = self.active_adapters[0]
    if active_adapter not in self.lora_A.keys() or active_adapter in self.lora_variant:
        return _orig_peft_linear4bit_forward(self, x, *args, **kwargs)

    base_layer = self.get_base_layer()
    quant_state = getattr(base_layer.weight, "quant_state", None)
    if quant_state is None or not x.is_cuda or base_layer.compute_dtype not in (None, torch.float16):
        return _orig_peft_linear4bit_forward(self, x, *args, **kwargs)

    import bitsandbytes as bnb

    bnb.matmul_4bit._is_called = True

    lora_A_mod = self.lora_A[active_adapter]
    lora_B_mod = self.lora_B[active_adapter]
    dropout_mod = self.lora_dropout[active_adapter]
    dropping = isinstance(dropout_mod, torch.nn.Dropout) and dropout_mod.p > 0.0 and self.training and dropout_mod.training
    scaling = float(self.scaling[active_adapter])

    return _TritonQLoRALinearFunction.apply(
        x,
        base_layer.weight,
        base_layer.bias,
        quant_state,
        lora_A_mod.weight,
        lora_B_mod.weight,
        scaling,
        dropout_mod if dropping else None,
        active_packing() if dropping else None,
    )


def patch_nf4_and_lora() -> None:
    """Install Triton NF4 dequantization and fused QLoRA linear patches."""
    global _orig_dequantize_4bit, _orig_bnb_linear4bit_forward, _orig_peft_linear4bit_forward, _PATCHED_NF4
    if _PATCHED_NF4:
        return

    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit

    _orig_dequantize_4bit = bnb_F.dequantize_4bit
    bnb_F.dequantize_4bit = triton_dequantize_nf4

    _orig_bnb_linear4bit_forward = bnb.nn.Linear4bit.forward
    bnb.nn.Linear4bit.forward = _fast_bnb_linear4bit_forward

    _orig_peft_linear4bit_forward = PeftLinear4bit.forward
    PeftLinear4bit.forward = _fast_peft_linear4bit_forward

    _PATCHED_NF4 = True
