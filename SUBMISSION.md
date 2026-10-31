# Submission: skyxplorers

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-d3f63101-skyxplorers/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `triton_nf4_dequant_kernel` | `solution/nf4_triton.py` | the 3 bitsandbytes kernels of double-quantized NF4 unpacking | % |
| `triton_vocab_ce_entropy_bwd_kernel` | `solution/cross_entropy_triton.py` | `lm_head`, softmax, cross-entropy, entropy and accuracy over the full logits | % |
| `triton_rmsnorm_fwd_kernel`, `triton_rmsnorm_bwd_kernel` | `solution/rmsnorm_triton.py` | `Qwen2RMSNorm.forward` and its backward | % |
| `triton_rope_qk_fwd_kernel`, `triton_rope_qk_bwd_kernel` | `solution/rope_triton.py` | `apply_rotary_pos_emb` and its backward | % |

`triton_swiglu_2d_fwd_kernel` and `triton_swiglu_2d_bwd_kernel` in `solution/swiglu.py` are a rewrite
of the kit's example and are not claimed as our own kernels.

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Online softmax: Milakov and Gimelshein (2018), https://arxiv.org/abs/1805.02867
- Chunked head and loss, and the per-token data layout: Soup's `soup_cli/utils/stream_loss_chunk.py`,
  https://github.com/MakazhanAlpamys/Soup
- NF4 and double quantization: Dettmers et al. (2023), https://arxiv.org/abs/2305.14314, and
  `bitsandbytes`

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | | | speedup x |
| Peak GPU memory, MiB | | | base/ours x |
| Eval loss (public rows) | | | % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [ ] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
See section 3 of the README: T4 numbers, fp16 gradients under the `GradScaler`, the two vocabulary
passes, the vanilla single-adapter LoRA path, and our own LoRA dropout masks.
