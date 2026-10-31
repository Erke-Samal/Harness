# Soup Kernel Challenge: project rules

Team repository for Decentrathon 6.0. Our own Triton kernels in `solution/` (entry point `apply()`)
make `soup train` faster or lighter on a Kaggle T4. Breaking a rule below disqualifies the solution.

Task: Soup 0.77.0, Qwen2.5-1.5B NF4, LoRA r=16 on `q_proj` and `v_proj`, batch 4, seq 512, 64 steps.
Target: Kaggle T4 (`sm_75`, fp16, no bf16), Python 3.13, torch 2.11, Triton 3.6.0.
Submission deadline: October 31, 23:59 Astana time (18:59 UTC).
Team repository: https://github.com/BAITC-Hacks/hack-d3f63101-skyxplorers

## Forbidden

- Changing `harness/`, `tests/`, `task/`, or the settings they set, including the PyTorch allocator.
- Ready-made kernel libraries: unsloth, liger-kernel, xformers, flash-attn, torchao and everything
  in `BANNED_MODULES` of `harness/common.py`; `backend: unsloth`, `use_liger`.
- Passing off `torch.compile` kernels as our own.
- Changing the task: model, data, content and order of batches, number of steps, LoRA settings
  (dropout included), optimizer, lr schedule.
- Switching Soup's built-in modes: `gradient_checkpointing`, `activation_offloading`, `fast_lora`,
  `use_cut_ce`, `packing`, `stream_loss_chunk` and the like.
- Adding trainable parameters; turning off or changing Soup's outputs (log metrics, `rewind.jsonl`).
- Mixing rows of different batches. Removing pad tokens inside a batch with our own code is allowed.
- Special behaviour under measurement or in tests.
- Copying other people's kernel code. Ideas are allowed with the source named in the README.
- bf16 and any Ampere+ feature.
- Reading or guessing hidden rows. Locally use only rows 0-255 and 40000-40255.

## Required

- Our `@triton.jit` kernels from `solution/` run in every measured run, take at least 5% of the
  training time from step 3 on, and training uses their results. The kit's example kernels do not
  count, however rewritten.
- Eval loss at most 0.5% above the base; time and peak memory at most 10% above the base.
- Tokens per step equal; entropy, accuracy and rewind per-row losses within 3%.
- Loss curve: first step within 1%, mean of all steps within 3%.
- All 79 Soup tests pass: `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`.
- README: what was done, how to run it, results from a real run, limits, a link to the Kaggle
  notebook. Never write a number that does not come from `runs/result.json`.

## Working here

- Code, comments, README and SUBMISSION.md are in English.
- Only numbers from a Kaggle T4 count, through `templates/kaggle_notebook.ipynb`. No local T4.
- Local machine: CachyOS, fish shell; system packages through pacman.
- CPU checks: Triton interpreter (`TRITON_INTERPRET=1`) for kernel math, and pinned packages from
  the kit's README in a virtual environment.
- Guard patches with `isinstance(..., torch.nn.Module)`: `apply()` also runs before tests that use
  fake modules.
