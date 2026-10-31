"""Values saved for backward live in pinned host memory during a training step.

The idea of offloading activations to pinned host memory is from SuperNeurons (Wang et al., 2018,
arXiv:1801.04380). One pinned slab is filled in order and reused every step, since pinning on the fly
is slow. Copies go into the same CUDA queue as the kernels, so the CPU never waits, and the queue
order alone guarantees that a value is written to the slab before it is read back in backward.
Allowed by the organizer's answers; backward stays complete and its values do not change.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

_SLAB_BYTES = 1 << 31  # a slab that fills up leaves further values on the GPU
_MIN_BYTES = 1 << 20  # smaller values cost more in copies than they free
_ALIGN = 64
_STEP_BUDGET_BYTES = int(os.environ.get("SKY_BUDGET", "300")) << 20  # only what exceeds it moves

_slab: Optional[torch.Tensor] = None
_used = 0
_kept = 0
_device: Any = None
_last_saved: Optional[torch.Tensor] = None  # held so that its address stays its own
_last_host: Optional[torch.Tensor] = None
_unpacked: dict[int, torch.Tensor] = {}


def _pack(saved: torch.Tensor) -> torch.Tensor:
    """The same tensor is saved by the norm and both LoRA branches of a layer: it counts once."""
    global _slab, _used, _kept, _last_saved, _last_host
    n_bytes = saved.nbytes
    if not saved.is_cuda or n_bytes < _MIN_BYTES:
        return saved
    if _last_saved is not None and _last_saved.data_ptr() == saved.data_ptr() and _last_saved.shape == saved.shape:
        return saved if _last_host is None else _last_host
    _last_saved = saved
    _last_host = None
    if _kept + n_bytes <= _STEP_BUDGET_BYTES or _used + n_bytes > _SLAB_BYTES:
        _kept += n_bytes
        return saved
    if _slab is None:
        _slab = torch.empty(_SLAB_BYTES, dtype=torch.uint8).pin_memory()
    host = _slab[_used : _used + n_bytes].view(saved.dtype).view(saved.shape)
    _used = (_used + n_bytes + _ALIGN - 1) & -_ALIGN
    host.copy_(saved, non_blocking=True)
    _last_host = host
    return host


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    if packed.is_cuda:
        return packed
    key = packed.data_ptr()
    restored = _unpacked.get(key)
    if restored is None:
        if len(_unpacked) >= 2:
            _unpacked.clear()
        restored = _unpacked[key] = packed.to(_device, non_blocking=True)
    return restored


def saved_on_host(device: torch.device) -> torch.autograd.graph.saved_tensors_hooks:
    """Hooks for one training forward; the backward of the previous step has already read the slab."""
    global _used, _kept, _device, _last_saved, _last_host
    _used = _kept = 0
    _device = device
    _last_saved = _last_host = None
    _unpacked.clear()
    return torch.autograd.graph.saved_tensors_hooks(_pack, _unpack)
