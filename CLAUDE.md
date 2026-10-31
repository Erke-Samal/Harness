# Soup Kernel Challenge: project rules

Team repository for Decentrathon 6.0, client Soup. Our own Triton kernels in `solution/` (entry
point `apply()`) make `soup train` faster or lighter on a free Kaggle T4 at the same quality.
Source of truth: the technical specification PDF and the kit's README. The jury's decision is final,
and failing any item of "Must have" means the solution is not admitted and not ranked.

## Task

- Soup 0.77.0 (`soup-cli[train]==0.77.0`), Qwen2.5-1.5B-Instruct, 4-bit NF4 weights, LoRA r=16,
  alpha=32 on `q_proj` and `v_proj`, batch 4, up to 512 tokens, 256 rows, one epoch (64 steps).
- Environment: Kaggle T4 (`sm_75`, one of the two cards, fp16, no bf16), Python 3.13, torch 2.11,
  Triton 3.6.0. Only numbers from a Kaggle T4 count.
- Base on a T4: about 80 s, about 14,350 MiB. Unsloth trains a similar LoRA in about 2.3 GB.
- Base step profile: matmul 36%, element-wise ops and dtype casts 44%, softmax over the 152k
  vocabulary 7%, attention 5%, NF4 unpacking 5%.
- Data: `AmanMussa/kazakh-instruction-v2`. Train on rows 0-255, self-check on rows 40000-40255.
  The final check trains and evaluates on hidden rows with another seed.

## Must have

1. Runs in a clean Kaggle notebook on a T4 with "Run All", without edits.
2. Our `@triton.jit` kernels from `solution/` run in every measured run and together take at least
   5% of the training time, counted from step 3. Not counted: `torch.compile` kernels, the example's
   kernels however rewritten, copied kernels, kernels whose result training does not use.
3. Eval loss on hidden rows at most 0.5% above the base; time and peak memory at most 10% above it.
4. Same training settings, adapter and Soup outputs. Tokens per step match; entropy, accuracy and
   `rewind.jsonl` per-row losses within 3%. Loss curve: first step within 1%, mean within 3%. Clean
   measurement: one GPU, idle before the run, the harness allocator. All 79 Soup tests pass.
5. README: what was done, how to run it, the results, a link to the Kaggle notebook.

## Forbidden

- Ready-made kernel libraries: unsloth, liger-kernel, xformers, flash-attn, torchao and everything
  in `BANNED_MODULES` of `harness/common.py`, including Soup's `backend: unsloth` and `use_liger`.
- Passing off `torch.compile` kernels as our own.
- Changing the task: model, data, content and order of batches, number of steps, LoRA settings
  (dropout included), optimizer, lr schedule.
- Switching Soup's built-in modes: `gradient_checkpointing`, `activation_offloading`, `fast_lora`,
  `use_cut_ce`, `packing`, `stream_loss_chunk` and the like.
- Adding trainable parameters; turning off Soup's outputs or changing their values (log metrics,
  `rewind.jsonl`).
- Mixing rows of different batches. Removing pad tokens inside a batch with our own code is allowed.
- Changing `harness/`, `tests/`, `task/` or the settings they set, such as the PyTorch allocator.
- Getting around the measurement: another GPU, background processes, special behaviour under
  measurement or in tests, swapping the saved adapter, ready-made compiled caches.
- Searching for or guessing the hidden data; reading files outside the repository and the task data.
- Copying other people's kernel code. Ideas are allowed with the source always credited.
- bf16 and any Ampere+ feature. Allowed packages: torch, triton, transformers, peft, trl,
  bitsandbytes, soup-cli 0.77.0 from PyPI. AI agents are allowed.
- Closed code, or more than one result per team, or more than 4 people.

## Scoring (100 points)

| Criterion | Points |
|---|---|
| Works and correct: all five "Must have" items, all or nothing | 20 |
| Speed: `35 * ln(s) / ln(s_best)`, s = base time / our time | 35 |
| Memory: `30 * ln(m) / ln(m_best)`, m = base peak / our peak | 30 |
| README: what was done, how to run it, results and limits, up to 5 each | 15 |

- The jury runs base, solution, solution, base in one session on hidden rows and averages.
- Time is the whole `soup train` command, model loading and Triton compilation included.
- Memory is the nvidia-smi peak sampled every 0.1 s over the whole run, not `torch` reserved.
- A ratio under 1.05 is a tie and gives 0 points. There is no ceiling. Repeats vary by 1-4%.
- On equal totals the better memory ratio ranks higher.

## Submission

Deadline: October 31, 23:59 Astana time (18:59 UTC). Submit on the hackathon platform:

1. The team repository https://github.com/BAITC-Hacks/hack-d3f63101-skyxplorers: `solution/`, a
   README and a filled `SUBMISSION.md` with links to the final commit and the Kaggle notebook. Work
   in other repositories is not checked. It must be public or open to the jury by the deadline.
2. A public Kaggle notebook from `templates/kaggle_notebook.ipynb`, a saved version with results,
   `COMMIT` set to the final commit. Its link goes in both the README and `SUBMISSION.md`.
3. A demo video of up to 3 minutes: what we sped up and what came out of it.
4. A presentation: which operation and why, how the kernel works, results on speed, memory and
   quality against the base, limits and next steps.

## Working here

- Code, comments, README and SUBMISSION.md are in English. Talk to the user in Russian.
- Never write a number in the README that does not come from a real `runs/result.json`.
- Admission first, then optimizations. Keep the last commit that passed all 10 checks.
- `harness/check_rules.py` needs `runs/result.json`; `python -m pytest` needs a GPU for
  `solution/test_kernels.py`, otherwise the tests skip.
- No local T4. CPU checks: `TRITON_INTERPRET=1` for kernel math, pinned packages in a virtual
  environment. Machine: CachyOS, fish shell, system packages through pacman, 7 GB RAM.
- Guard patches with `isinstance(..., torch.nn.Module)`: `apply()` also runs before tests that use
  fake modules.
