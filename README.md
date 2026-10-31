# Soup Kernel Challenge: team `skyxplorers`

Our own `@triton.jit` kernels for `soup train` (`soup-cli==0.77.0`) on a free Kaggle Tesla T4:
Qwen2.5-1.5B-Instruct, 4-bit NF4 weights, LoRA r=16, batch 4, up to 512 tokens, 64 steps.

- **Public Kaggle notebook:** `<paste the saved version link after Save & Run All>`
- **Final commit:** `<paste the commit link>`

## 1. What we did and why

The organizers' profile of base Soup: matrix multiplies about 36% of GPU time, element-wise ops
and dtype casts about 44%, the softmax over the 151,936-token vocabulary about 7%, attention about
5%, NF4 unpacking about 5%; peak memory about 14,350 MiB. We read the training path in
transformers, peft, trl, bitsandbytes and Soup and targeted these causes:

1. **The fp32 cascade.** `prepare_model_for_kbit_training` casts the tied embedding, `lm_head`, the
   norms and biases to fp32, and `Qwen2RMSNorm.forward` multiplies by an fp32 weight, so activations
   between the layers stay in fp32 although the weights were loaded in fp16.
2. **Full logits.** `[B, S, 151936]` logits are built and copied for the loss, entropy, accuracy
   and the rewind log.
3. **Separate NF4 unpacking.** With double quantization, each dequantization of a weight is several
   kernel launches, in forward and again in backward.
4. **The 4D padding mask.** A right-padded 2D mask becomes a 4D mask, which turns off native GQA
   and `is_causal` in SDPA.

| Module | Kernels | What it does |
|---|---|---|
| `solution/nf4_triton.py` | `triton_nf4_dequant_kernel` | One launch for double-quantized NF4/FP4 unpacking: nested `absmax`, both nibbles, the codebook, pairs of fp16 written as one 32-bit word, into one reusable buffer. Autograd functions for `bnb.nn.Linear4bit` and the peft LoRA layer avoid `result.clone()` and keep the dropout mask in `uint8`. |
| `solution/cross_entropy_triton.py` | `triton_vocab_ce_entropy_bwd_kernel` | Chunks of 64 positions in one reused workspace. One row pass gives cross-entropy, Shannon entropy and argmax by an online softmax; a second pass overwrites the chunk with `softmax - one_hot`. Full logits are never built on a training step. |
| `solution/rmsnorm_triton.py` | `triton_rmsnorm_fwd_kernel`, `triton_rmsnorm_bwd_kernel` | RMSNorm in fp32 registers with output in the input's dtype, which stops the fp32 cascade. Saves `x` and a 1D `rstd`. |
| `solution/rope_triton.py` | `triton_rope_qk_fwd_kernel`, `triton_rope_qk_bwd_kernel` | Q and K rotary embedding in one launch, reading the non-contiguous views directly. Backward saves only `cos` and `sin`. |
| `solution/swiglu.py` | `triton_swiglu_2d_fwd_kernel`, `triton_swiglu_2d_bwd_kernel` | SwiGLU over 2D rows, `silu(gate)` recomputed in backward. **This is our rewrite of the kit's example, so we do not claim it as our own kernel**: the rules say the example does not count however it is rewritten. |

Other patches, in `solution/__init__.py`:

- After `SFTTrainerWrapper._setup_transformers`, frozen fp32 weights other than the norms go back
  to fp16. LoRA weights stay in fp32. The saved adapter is unchanged.
- A 2D `attention_mask` verified as right-padded is passed to `Qwen2Model` as `None`. Valid tokens
  never attend to later padding, `position_ids` do not depend on the mask, and TRL still reads the
  mask from `inputs` for its metrics.

**How the chunked loss plugs in.** We do not turn on Soup's optional `stream_loss_chunk` mode: the
config is untouched. Our own patch of `Qwen2ForCausalLM.forward` holds back `lm_head` on a labelled
training step, and our patch of `trl.SFTTrainer.compute_loss` logs `entropy`, `num_tokens` and
`mean_token_accuracy` by TRL's formulas. Per-token losses are left in the trainer attribute that
Soup's rewind recorder already reads for steps without logits.

**Sources.** The online softmax follows Milakov and Gimelshein, *Online normalizer calculation for
softmax* (2018); the chunked loss idea and its data layout come from Soup's own
`soup_cli/utils/stream_loss_chunk.py`; the NF4 format comes from Dettmers et al., *QLoRA* (2023) and
the `bitsandbytes` `QuantState`. The fused-kernel path follows the kit's `solution/swiglu.py`.

## 2. How to run it

On Kaggle (the way the jury runs it): import `templates/kaggle_notebook.ipynb`, set the accelerator
to GPU T4 x2 and Internet on, set `REPO_URL` and `COMMIT` in the first code cell, press Run All.
The repository is private: add a Kaggle secret `GITHUB_TOKEN` with a read-only token.

On a Linux machine with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py
python harness/check_rules.py
python harness/score.py runs/result.json
```

## 3. Results and limits

**T4 results: not measured yet.** The table is filled from `runs/result.json` of the notebook
above. Base numbers in the kit are the organizers' measurement, not ours.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | `<from run>` | `<from run>` | `<speedup>`x |
| Peak GPU memory (nvidia-smi), MiB | `<from run>` | `<from run>` | `<memory base/solution>`x |
| Eval loss (public rows) | `<from run>` | `<from run>` | `<delta>`% |

What has been checked without a T4:

- all kernels compile for `sm_75` on Triton 3.6.0 with torch 2.11;
- the kernels match plain PyTorch in the Triton interpreter on CPU, forward and backward; the
  autograd functions for 4-bit and LoRA layers match in fp16 with a stubbed dequantization;
- 79 of 79 Soup tests pass with `SOUP_APPLY_SOLUTION=1`;
- a real `soup train` on a tiny random Qwen2 on CPU, with the chunked loss on: tokens, accuracy and
  rewind rows equal to the base, loss and entropy within 1e-5.

Limits:

1. Nothing above measures the speed, the memory or the 4-bit path of a real T4 run.
2. Gradients now flow in fp16 under the `GradScaler` where the base keeps fp32 activations. An
   overflow would skip a step, which checks 6 and 7 would show.
3. The chunked loss works on training steps of a single process; other paths use the original code.
4. Each row of the chunked loss passes over the vocabulary twice, because it does not fit in
   registers.
5. The fast LoRA path covers one vanilla adapter; DoRA, several adapters and merged layers use
   peft's original forward. Unusual masks (left padding, gaps) keep the 2D mask.
6. LoRA dropout keeps its probability but draws its mask by our own code, so masks differ from the
   base run.
