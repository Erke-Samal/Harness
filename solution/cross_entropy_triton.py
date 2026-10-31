"""Custom Triton chunked vocabulary Cross-Entropy + Shannon Entropy + Top-1 Accuracy kernel.

Eliminates the single largest memory and bandwidth bottleneck in Qwen2.5-1.5B
(``vocab_size = 151936``) training under TRL ``SFTTrainer`` + Soup ``RewindTrainer``:
materializing the full ``[B=4, S=512, V=151936]`` logits tensor 3+ times per step
(``lm_head`` output, ``F.cross_entropy``, ``entropy_from_logits``, and
``outputs.logits[..., :-1, :].contiguous()``), which consumes over 3,100 MiB of peak
GPU memory and performs ~10 full-vocabulary HBM passes per step.

Our ``@triton.jit`` kernel ``triton_vocab_ce_entropy_bwd_kernel`` processes token rows
in chunks of ``CE_CHUNK`` (reusing a single workspace buffer across all chunks), skips
padding when the step is packed (``solution.token_packing``), and in a single fused vocabulary kernel computes:
  1. Exact per-token cross-entropy loss (for both scalar loss and ``rewind.jsonl``).
  2. Exact per-token Shannon entropy ``H(p) = log(Z) - sum_j exp(x_j - m) * (x_j - m) / Z``
     via a shift-invariant online softmax recurrence (for TRL ``entropy`` logging).
  3. Exact per-token top-1 accuracy ``argmax_j(x_j) == target`` (for TRL
     ``mean_token_accuracy`` logging).
  4. In-place unscaled logit gradient ``dlogits = softmax(x) - one_hot(target)`` in
     ``[-1, 1]`` (optimal for fp16 dynamic range), immediately projected onto
     ``grad_hidden`` and ``grad_lm_head_weight`` via fp16 Tensor Core GEMMs.

The wiring is our own: ``Qwen2ForCausalLM.forward`` skips ``lm_head`` on a labelled
training step and returns the loss without logits, and ``SFTTrainer.compute_loss``
logs TRL's metrics from the per-token results. Soup's rewind recorder reads the
per-token losses from the trainer attribute named by ``_STEP_ATTR``. Soup's optional
``stream_loss_chunk`` mode stays off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from solution.host_saved import saved_on_host
from solution.nf4_triton import drop_gpu_copies
from solution.token_packing import TokenPacking, plan_token_packing, set_active_packing

# The attribute under which Soup's rewind recorder looks for per-token losses.
_STEP_ATTR = "_soup_chunked_step"
_ARMED_ATTR = "_skyxplorers_chunked_ce"
_LAST_STEP_ATTR = "_skyxplorers_last_step"

CE_CHUNK = 32  # the workspace costs CE_CHUNK * 297 KiB at the peak
_LOGITS_WORKSPACE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_logits_workspace(n_rows: int, vocab_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (device, dtype)
    needed = n_rows * vocab_size
    buf = _LOGITS_WORKSPACE.get(key)
    if buf is None or buf.numel() < needed:
        buf = torch.empty(needed, dtype=dtype, device=device)
        _LOGITS_WORKSPACE[key] = buf
    return buf[:needed].view(n_rows, vocab_size)


@triton.jit
def triton_vocab_ce_entropy_bwd_kernel(
    Logits_ptr,
    Targets_ptr,
    Losses_ptr,
    Entropies_ptr,
    Correct_ptr,
    stride_logits_row,
    vocab_size,
    ignore_index,
    WITH_GRAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fused online-softmax Cross-Entropy + Shannon Entropy + Argmax + In-Place Backward."""
    row = tl.program_id(0)
    row_ptr = Logits_ptr + row * stride_logits_row
    target = tl.load(Targets_ptr + row).to(tl.int64)
    valid_target = (target != ignore_index) & (target >= 0) & (target < vocab_size)

    # Pass 1: Online softmax (m_i, d_i), shift-invariant entropy numerator (u_i), and argmax.
    m_i = -float("inf")
    d_i = 0.0
    u_i = 0.0
    best_val = -float("inf")
    best_idx = 0

    for start_v in range(0, vocab_size, BLOCK_V):
        cols = start_v + tl.arange(0, BLOCK_V)
        mask = cols < vocab_size
        x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)

        blk_max, blk_arg = tl.max(x, axis=0, return_indices=True)
        if blk_max > best_val:
            best_val = blk_max
            best_idx = start_v + blk_arg

        m_new = tl.maximum(m_i, blk_max)
        # NaN-safe shift for initial empty state (d_i == 0 and m_i == -inf)
        delta_old = tl.where(d_i > 0.0, m_i - m_new, 0.0)
        alpha = tl.exp(delta_old)

        x_shifted = x - m_new
        exp_x = tl.where(mask, tl.exp(x_shifted), 0.0)
        sum_exp = tl.sum(exp_x, axis=0)

        # Avoid 0.0 * (-inf) = NaN on out-of-bounds lanes
        x_shifted_safe = tl.where(mask, x_shifted, 0.0)
        sum_x_exp = tl.sum(exp_x * x_shifted_safe, axis=0)

        u_i = u_i * alpha + d_i * alpha * delta_old + sum_x_exp
        d_i = d_i * alpha + sum_exp
        m_i = m_new

    log_z = tl.log(d_i)
    entropy = log_z - (u_i / d_i)
    tl.store(Entropies_ptr + row, entropy)

    if valid_target:
        target_logit = tl.load(row_ptr + target).to(tl.float32)
        loss_val = (m_i + log_z) - target_logit
        is_correct = 1 if (best_idx == target) else 0
    else:
        loss_val = 0.0
        is_correct = 0

    tl.store(Losses_ptr + row, loss_val)
    tl.store(Correct_ptr + row, is_correct)

    # Pass 2: In-place unscaled gradient dlogits = softmax(x) - 1_{j == target}
    if WITH_GRAD:
        inv_d = 1.0 / d_i
        for start_v in range(0, vocab_size, BLOCK_V):
            cols = start_v + tl.arange(0, BLOCK_V)
            mask = cols < vocab_size
            if valid_target:
                x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
                prob = tl.exp(x - m_i) * inv_d
                grad = tl.where(cols == target, prob - 1.0, prob)
            else:
                grad = tl.zeros([BLOCK_V], dtype=tl.float32)
            tl.store(row_ptr + cols, grad.to(Logits_ptr.dtype.element_ty), mask=mask)


class _TritonChunkedHeadLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: Optional[torch.Tensor],
        targets: torch.Tensor,
        chunk_size: int,
        num_items_in_batch: Optional[int],
        ignore_index: int,
    ):
        # hidden: [R, D], targets: [R]; each row predicts its own target.
        n_total, hidden_dim = hidden.shape
        vocab_size = lm_head_weight.shape[0]
        device = hidden.device

        valid_count = targets.ne(ignore_index).sum()

        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                denom = num_items_in_batch.to(device=device, dtype=torch.float32).clamp_min(1.0)
            else:
                denom = torch.tensor(max(float(num_items_in_batch), 1.0), dtype=torch.float32, device=device)
        else:
            denom = valid_count.to(torch.float32).clamp_min(1.0)

        need_h_grad = bool(hidden.requires_grad)
        need_w_grad = bool(lm_head_weight.requires_grad)
        need_b_grad = bool(lm_head_bias is not None and lm_head_bias.requires_grad)
        with_grad = need_h_grad or need_w_grad or need_b_grad

        # Perform chunked GEMMs in fp16 for Tensor Core throughput on T4
        compute_dtype = torch.float16 if hidden.dtype in (torch.float16, torch.float32) else hidden.dtype
        w_comp = lm_head_weight.to(compute_dtype) if lm_head_weight.dtype != compute_dtype else lm_head_weight
        b_comp = (
            lm_head_bias.to(compute_dtype)
            if (lm_head_bias is not None and lm_head_bias.dtype != compute_dtype)
            else lm_head_bias
        )

        token_losses = torch.empty(n_total, dtype=torch.float32, device=device)
        token_entropies = torch.empty(n_total, dtype=torch.float32, device=device)
        token_correct = torch.empty(n_total, dtype=torch.int32, device=device)

        if need_h_grad:
            grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
        else:
            grad_hidden = torch.empty(0, dtype=torch.float32, device=device)

        if need_w_grad:
            grad_w_accum = torch.zeros_like(lm_head_weight, dtype=torch.float32)
        else:
            grad_w_accum = torch.empty(0, dtype=torch.float32, device=device)

        if need_b_grad:
            grad_b_accum = torch.zeros_like(lm_head_bias, dtype=torch.float32)
        else:
            grad_b_accum = torch.empty(0, dtype=torch.float32, device=device)

        logits_ws = _get_logits_workspace(min(chunk_size, n_total), vocab_size, compute_dtype, device)

        BLOCK_V = 4096
        num_warps = 8

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            n_rows = end - start

            h_2d = hidden[start:end].to(compute_dtype)
            logits_chunk = logits_ws[:n_rows]
            torch.mm(h_2d, w_comp.t(), out=logits_chunk)
            if b_comp is not None:
                logits_chunk.add_(b_comp)

            triton_vocab_ce_entropy_bwd_kernel[(n_rows,)](
                logits_chunk,
                targets[start:end],
                token_losses[start:end],
                token_entropies[start:end],
                token_correct[start:end],
                logits_chunk.stride(0),
                vocab_size,
                int(ignore_index),
                WITH_GRAD=with_grad,
                BLOCK_V=BLOCK_V,
                num_warps=num_warps,
            )

            if with_grad:
                # logits_chunk now holds unscaled dlogits in [-1, 1]
                if need_h_grad:
                    grad_hidden[start:end] = torch.mm(logits_chunk, w_comp).to(torch.float32)
                if need_w_grad:
                    gw_chunk = torch.mm(logits_chunk.t(), h_2d)
                    grad_w_accum.add_(gw_chunk.to(torch.float32))
                if need_b_grad:
                    gb_chunk = logits_chunk.to(torch.float32).sum(dim=0)
                    grad_b_accum.add_(gb_chunk)

        inv_denom = 1.0 / denom
        if need_h_grad:
            grad_hidden.mul_(inv_denom)
        if need_w_grad:
            grad_w_accum.mul_(inv_denom)
        if need_b_grad:
            grad_b_accum.mul_(inv_denom)

        # Zero out grad_hidden if valid_count == 0
        valid_any = valid_count > 0
        if need_h_grad:
            grad_hidden = torch.where(valid_any, grad_hidden, torch.zeros_like(grad_hidden))

        total_loss = torch.where(valid_any, token_losses.sum() * inv_denom, token_losses.sum() * 0.0)
        correct_total = token_correct.to(torch.int64).sum()
        valid_total = valid_count.to(torch.int64)

        ctx.save_for_backward(grad_hidden, grad_w_accum, grad_b_accum)
        ctx.hidden_dtype = hidden.dtype
        ctx.w_dtype = lm_head_weight.dtype
        ctx.b_dtype = lm_head_bias.dtype if lm_head_bias is not None else None
        ctx.has_h_grad = need_h_grad
        ctx.has_w_grad = need_w_grad
        ctx.has_b_grad = need_b_grad
        ctx.mark_non_differentiable(token_losses, token_entropies, correct_total, valid_total)

        return total_loss, token_losses, token_entropies, correct_total, valid_total

    @staticmethod
    def backward(ctx, grad_loss, _grad_tl, _grad_te, _grad_ct, _grad_vt):
        grad_hidden, grad_w_accum, grad_b_accum = ctx.saved_tensors
        scale = grad_loss.to(torch.float32)

        d_hidden = (grad_hidden * scale).to(ctx.hidden_dtype) if ctx.has_h_grad else None
        d_weight = (grad_w_accum * scale).to(ctx.w_dtype) if ctx.has_w_grad else None
        d_bias = (grad_b_accum * scale).to(ctx.b_dtype) if ctx.has_b_grad else None

        return d_hidden, d_weight, d_bias, None, None, None, None


def triton_chunked_causal_lm_loss(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    packing: Optional[TokenPacking] = None,
):
    """Run chunked causal LM loss + entropy + top-1 accuracy via Triton.

    ``hidden`` is ``[B, S, D]``, or ``[1, N, D]`` with a ``packing``. Token results are ``[B, S]``,
    and position t of a row holds the loss of label t + 1, with 0 on padding.
    """
    batch, seq_len = labels.shape
    # Causal LM shift: position t predicts label t + 1, the last position nothing.
    targets = F.pad(labels, (0, 1), value=ignore_index)[:, 1:].reshape(-1)
    if packing is not None:
        targets = targets[packing.index]

    loss, token_losses, token_entropies, correct, valid = _TritonChunkedHeadLossFunction.apply(
        hidden.reshape(-1, hidden.shape[-1]),
        lm_head_weight,
        lm_head_bias,
        targets,
        int(chunk_size),
        num_items_in_batch,
        int(ignore_index),
    )
    if packing is not None:
        index = packing.index
        token_losses = token_losses.new_zeros(batch * seq_len).index_copy_(0, index, token_losses)
        token_entropies = token_entropies.new_zeros(batch * seq_len).index_copy_(0, index, token_entropies)
    return loss, token_losses.view(batch, seq_len), token_entropies.view(batch, seq_len), correct, valid


@dataclass(frozen=True)
class ChunkStep:
    """Per-token results of a chunked step, in the layout Soup's rewind recorder reads."""

    token_entropy: torch.Tensor  # [B, S]
    token_loss: torch.Tensor  # [B, S], position t holds the loss of label t + 1, else 0
    correct: torch.Tensor
    labelled: torch.Tensor


def arm_chunked_ce(causal_lm: torch.nn.Module) -> None:
    """Turn the chunked path on for a model whose ``lm_head`` is frozen."""
    if any(param.requires_grad for param in causal_lm.lm_head.parameters()):
        raise ValueError("the chunked loss takes gradients on the hidden states: lm_head must be frozen")
    setattr(causal_lm, _ARMED_ATTR, True)


def _armed_causal_lm(model: Any) -> Optional[torch.nn.Module]:
    get_base_model = getattr(model, "get_base_model", None)
    causal_lm = get_base_model() if callable(get_base_model) else model
    return causal_lm if getattr(causal_lm, _ARMED_ATTR, False) is True else None


def _log_trl_metrics(trainer: Any, inputs: dict[str, Any], step: ChunkStep) -> None:
    """``entropy``, ``num_tokens`` and ``mean_token_accuracy``, by the formulas of TRL."""
    gather = trainer.accelerator.gather_for_metrics
    metrics = trainer._metrics["train"]
    attention_mask = inputs["attention_mask"]
    entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
    metrics["entropy"].append(gather(entropy).mean().item())
    trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    metrics["num_tokens"] = [trainer._total_train_tokens]
    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics["mean_token_accuracy"].append(accuracy)


_PATCHED_CE = False
_FIRST_STEP_DONE_ATTR = "_skyxplorers_first_step_done"


def patch_chunked_ce_loss() -> None:
    """Route labelled training steps of an armed Qwen2 model through the chunked kernel."""
    global _PATCHED_CE
    if _PATCHED_CE:
        return

    import trl
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    orig_forward = modeling_qwen2.Qwen2ForCausalLM.forward
    orig_compute_loss = trl.SFTTrainer.compute_loss

    def _fast_causal_forward(self, *args, **kwargs):
        if (
            args
            or kwargs.get("labels") is None
            or not self.training
            or getattr(self, _ARMED_ATTR, False) is not True
            or "shift_labels" in kwargs
            or kwargs.get("logits_to_keep", 0) != 0
        ):
            return orig_forward(self, *args, **kwargs)

        if not getattr(self, _FIRST_STEP_DONE_ATTR, False):
            setattr(self, _FIRST_STEP_DONE_ATTR, True)
            before = torch.cuda.memory_allocated()
            drop_gpu_copies(self)
            torch.cuda.empty_cache()
            print("DIAG", before >> 20, torch.cuda.memory_allocated() >> 20, torch.cuda.memory_reserved() >> 20, torch.cuda.max_memory_allocated() >> 20, flush=True)
            torch.cuda.reset_peak_memory_stats()

        labels = kwargs.pop("labels")
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        kwargs.pop("logits_to_keep", None)

        packing = None
        if kwargs.get("past_key_values") is None and kwargs.get("position_ids") is None:
            packing = plan_token_packing(self.model, kwargs.get("input_ids"), kwargs.get("attention_mask"))
        if packing is None:
            outputs = self.model(**kwargs)
        else:
            input_ids = kwargs.pop("input_ids")
            device = input_ids.device
            kwargs.update(
                attention_mask=None,
                use_cache=False,
                inputs_embeds=self.model.embed_tokens(input_ids.view(-1)[packing.index].unsqueeze(0)),
                position_ids=torch.arange(packing.seq_len, device=device).unsqueeze(0),
                token_packing=packing,
            )
            set_active_packing(packing)
            try:
                with saved_on_host(device):
                    outputs = self.model(**kwargs)
            finally:
                set_active_packing(None)

        lm_head = self.lm_head
        loss, token_losses, token_entropies, correct, valid = triton_chunked_causal_lm_loss(
            outputs.last_hidden_state,
            lm_head.weight,
            labels,
            lm_head_bias=lm_head.bias,
            chunk_size=CE_CHUNK,
            num_items_in_batch=num_items_in_batch,
            packing=packing,
        )
        setattr(self, _LAST_STEP_ATTR, ChunkStep(token_entropies, token_losses, correct, valid))
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _fast_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal_lm = _armed_causal_lm(self.model)
        if (
            causal_lm is None
            or not self.model.training
            or "labels" not in inputs
            or "shift_labels" in inputs
            or "attention_mask" not in inputs
            or self.label_smoother is not None
            or self.compute_loss_func is not None
        ):
            return orig_compute_loss(
                self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        loss, outputs = Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        step = getattr(causal_lm, _LAST_STEP_ATTR)
        setattr(self, _STEP_ATTR, step)
        _log_trl_metrics(self, inputs, step)
        return (loss, outputs) if return_outputs else loss

    modeling_qwen2.Qwen2ForCausalLM.forward = _fast_causal_forward
    trl.SFTTrainer.compute_loss = _fast_compute_loss
    _PATCHED_CE = True
