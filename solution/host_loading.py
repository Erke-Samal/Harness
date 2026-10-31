"""Load the model without its GPU peak.

The peak of the run is not training but ``from_pretrained``: the warm-up reserves the size of the
model, and ``prepare_model_for_kbit_training`` widens the 445 MiB embedding to fp32 on the GPU while
both copies exist. So during Soup's setup the embedding stays on the host and goes to the GPU once,
already in fp16, and the packed bytes of each 4-bit weight move to pinned host memory as soon as
they are quantized. Values, the embedding tie with ``lm_head`` and ``QuantState`` do not change, and
all copies are inside the measured time. Allowed by the organizer's answers, see README.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from solution.nf4_triton import _HOST_ATTR, build_weight_stream, move_packed_weight_to_host, restore_packed_weight

_EMBEDDING = "embed_tokens.weight"
_loading = False
_PATCHED = False


@contextmanager
def loading_to_host() -> Iterator[None]:
    """Only Soup's own model setup loads this way; any other ``from_pretrained`` stays as it was."""
    global _loading
    _loading = True
    try:
        yield
    finally:
        _loading = False


def patch_model_loading() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import bitsandbytes as bnb
    from transformers import core_model_loading, modeling_utils

    orig_get_device = core_model_loading.get_device
    orig_set_param = core_model_loading.set_param_for_module
    orig_warmup = modeling_utils.caching_allocator_warmup

    def _get_device(device_map: Any, param_name: str, *args: Any, **kwargs: Any) -> Any:
        if _loading and param_name.endswith(_EMBEDDING):
            return "cpu"
        return orig_get_device(device_map, param_name, *args, **kwargs)

    def _set_param(model: Any, target_name: str, param_value: Any, *args: Any, **kwargs: Any) -> None:
        orig_set_param(model, target_name, param_value, *args, **kwargs)
        if _loading:
            module_path, _, param_name = target_name.rpartition(".")
            weight = getattr(model.get_submodule(module_path), param_name, None)
            if getattr(weight, "quant_state", None) is not None:
                move_packed_weight_to_host(weight)

    def _warmup(*args: Any, **kwargs: Any) -> None:
        if not _loading:
            orig_warmup(*args, **kwargs)

    orig_param_to = bnb.nn.Params4bit.to

    def _param_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Preparing the model moves it to the GPU; the host copy is already the only one."""
        if getattr(getattr(self, "quant_state", None), _HOST_ATTR, None) is not None:
            return self
        return orig_param_to(self, *args, **kwargs)

    bnb.nn.Params4bit.to = _param_to
    core_model_loading.get_device = _get_device
    core_model_loading.set_param_for_module = _set_param
    modeling_utils.caching_allocator_warmup = _warmup
    _PATCHED = True


def place_loaded_model(model: torch.nn.Module, device: torch.device, fits_kernels: bool) -> None:
    """Send what peft and the embedding left on the host to the GPU, once and in its final dtype.

    A model the kernels do not take gets its 4-bit weights back and trains the ordinary way.
    """
    for param in model.parameters():
        if getattr(param, "quant_state", None) is not None:
            if not fits_kernels and not param.is_cuda:
                restore_packed_weight(param)
        elif not param.is_cuda:
            param.data = param.data.to(device)
    if fits_kernels:
        build_weight_stream(model, device)
