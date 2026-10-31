"""Entry point for the custom Triton solution package (team ``skyxplorers``).

``solution.apply()`` installs five custom ``@triton.jit`` kernel modules and the
supporting QLoRA / Soup SFT wiring:
  1. ``solution.nf4_triton``: Fused single-launch NF4/FP4 double-dequantization
     (``triton_nf4_dequant_kernel``) + fused QLoRA autograd functions.
  2. ``solution.cross_entropy_triton``: Chunked vocabulary cross-entropy + online
     Shannon entropy + top-1 accuracy + in-place backward
     (``triton_vocab_ce_entropy_bwd_kernel``) wired into Soup's ``ChunkedHeadLoss``.
  3. ``solution.rmsnorm_triton``: Fused fp32-register RMSNorm forward/backward
     (``triton_rmsnorm_fwd_kernel``, ``triton_rmsnorm_bwd_kernel``) preserving
     ``fp16`` hidden state dtype.
  4. ``solution.rope_triton``: Fused 2D Q+K Rotary Position Embedding forward/backward
     (``triton_rope_qk_fwd_kernel``, ``triton_rope_qk_bwd_kernel``).
  5. ``solution.swiglu``: 2D multi-row tiled SwiGLU forward/backward with backward
     activation recomputation (``triton_swiglu_2d_fwd_kernel``,
     ``triton_swiglu_2d_bwd_kernel``).
"""

from __future__ import annotations

import gc

import torch

from solution.cross_entropy_triton import arm_chunked_ce, patch_chunked_ce_loss
from solution.nf4_triton import patch_nf4_and_lora
from solution.rmsnorm_triton import patch_qwen2_rmsnorm
from solution.rope_triton import patch_qwen2_rope
from solution.swiglu import patch_qwen2_mlp

_APPLIED = False


def _patch_qwen2_causal_attention_mask() -> None:
    """Enable native causal GQA SDPA when 2D attention_mask is purely right-padded.

    In causal self-attention, non-padding tokens at positions ``t < L_b`` cannot
    attend to right-padding tokens at ``s >= L_b > t``. Passing a 2D right-padded
    mask into ``Qwen2Model.forward`` causes ``create_causal_mask`` to materialize
    a 4D ``[B, 1, S, S]`` mask, which disables ``enable_gqa=True`` (forcing 6x
    ``repeat_kv`` copies across all 28 layers) and disables ``is_causal=True`` in
    PyTorch SDPA. Replacing a verified right-padded 2D ``attention_mask`` with
    ``None`` inside ``Qwen2Model.forward`` produces identical representations on
    all valid tokens while unlocking fused causal GQA SDPA on Tesla T4.
    """
    from transformers.models.qwen2 import modeling_qwen2

    orig_qwen2_model_forward = modeling_qwen2.Qwen2Model.forward
    if getattr(orig_qwen2_model_forward, "_skyxplorers_patched", False):
        return

    def _fast_qwen2_model_forward(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        if (
            not args
            and torch.is_tensor(attention_mask)
            and attention_mask.ndim == 2
            and attention_mask.is_cuda
            and attention_mask.shape[1] > 1
            and kwargs.get("past_key_values") is None
        ):
            # Verify right-padding (monotonic 1s followed by 0s, starting with 1)
            is_right_padded = bool(
                (attention_mask[:, 0] == 1).all()
                and (attention_mask[:, :-1] >= attention_mask[:, 1:]).all()
            )
            if is_right_padded:
                kwargs["attention_mask"] = None

        return orig_qwen2_model_forward(self, *args, **kwargs)

    _fast_qwen2_model_forward._skyxplorers_patched = True
    modeling_qwen2.Qwen2Model.forward = _fast_qwen2_model_forward


def _patch_soup_sft_setup() -> None:
    """Hook ``SFTTrainerWrapper._setup_transformers``: restore frozen fp16 weights on real
    CUDA models and arm the chunked cross-entropy.
    """
    from soup_cli.trainer import sft as soup_sft
    from transformers.models.qwen2 import modeling_qwen2

    orig_setup = soup_sft.SFTTrainerWrapper._setup_transformers
    if getattr(orig_setup, "_skyxplorers_patched", False):
        return

    def _wrapped_setup_transformers(self, *args, **kwargs):
        result = orig_setup(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if not isinstance(model, torch.nn.Module):
            return result
        if not any(p.is_cuda for p in model.parameters()):
            return result

        # 1. Restore frozen non-norm fp32 weights (embed_tokens, lm_head, frozen biases)
        #    back to fp16 after prepare_model_for_kbit_training upcast them to fp32.
        #    Trainable LoRA weights (requires_grad=True) and RMSNorm weights stay in fp32.
        cast_any = False
        for name, param in model.named_parameters():
            if (
                not param.requires_grad
                and param.dtype == torch.float32
                and "norm" not in name.lower()
            ):
                param.data = param.data.to(torch.float16)
                cast_any = True

        # 2. Full [B, S, V] logits are never materialized on a labelled training step.
        causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
        if isinstance(causal_lm, modeling_qwen2.Qwen2ForCausalLM):
            arm_chunked_ce(causal_lm)

        if cast_any:
            gc.collect()
            torch.cuda.empty_cache()

        return result

    _wrapped_setup_transformers._skyxplorers_patched = True
    soup_sft.SFTTrainerWrapper._setup_transformers = _wrapped_setup_transformers


def apply() -> None:
    """Install all custom Triton kernel patches for ``soup train``."""
    global _APPLIED
    if _APPLIED:
        return

    patch_qwen2_mlp()
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_nf4_and_lora()
    patch_chunked_ce_loss()
    _patch_qwen2_causal_attention_mask()
    _patch_soup_sft_setup()

    _APPLIED = True
