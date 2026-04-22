# GRPO Training with MiniTorch CUDA Kernels

GRPO (Group Relative Policy Optimization) training pipeline for GSM8K math reasoning, with custom fused CUDA kernels implemented via MiniTorch.

**Base model:** `Qwen/Qwen3.5-2B`

## Project Structure

```
src/
  grpo_trainer.py      # GRPO trainer with mini-batch grad accum and minitorch toggle
  rlhf_trainer.py      # VERLPolicyWrapper (shared by PPO and GRPO)
  gsm8k_reward.py      # Rule-based reward: extract #### answer, compare to ground truth
  config.py            # All configuration dataclasses
  dpo_trainer.py       # DPO baseline trainer
  reward_model.py      # Neural reward model (for PPO)
  policy_io.py         # Pluggable model load/save hooks

minitorch/
  ops.py               # 3 fused ops with CUDA kernel + PyTorch fallback
  cuda_kernels/
    grpo_kernels.cu    # CUDA kernel implementations
    Makefile           # Compile with: make

baselines/
  run_trl.py           # HuggingFace TRL GRPOTrainer wrapper
  run_verl.py          # VeRL integration (stub)
  run_simple_grpo.py   # simple_GRPO baseline wrapper (stub)

scripts/
  run_grpo_gsm8k.py    # Main GRPO training entrypoint
  evaluate_grpo.py     # Greedy evaluation on val/test splits
  benchmark_all.py     # Unified comparison harness across all methods
```

## MiniTorch CUDA Kernels

Three fused CUDA kernels replace multi-step PyTorch operations in the GRPO training loop:

| Kernel | What it fuses | Where used |
|--------|--------------|------------|
| `fused_log_prob_gather` | gather + logsumexp over 151k vocab in one pass | Log prob computation in `train_step` |
| `group_advantage_norm` | mean + std + normalize per group via shared memory | `_compute_group_advantages()` |
| `fused_grpo_objective` | clipped surrogate + KL divergence in one grid-stride loop | `_compute_policy_objective()` |

Each op has a **PyTorch fallback** that runs when the compiled `.so` is unavailable, so everything works without CUDA compilation.

### Compile kernels (on cluster)

```bash
cd minitorch/cuda_kernels
make
```

### Toggle minitorch

```bash
# PyTorch-only (default)
python scripts/run_grpo_gsm8k.py --model_name Qwen/Qwen3.5-2B

# With minitorch CUDA kernels
python scripts/run_grpo_gsm8k.py --model_name Qwen/Qwen3.5-2B --use_minitorch
```

## Quick Start

### 1. Prepare GSM8K data

```bash
python scripts/prepare_gsm8k.py
```

### 2. Train GRPO on GSM8K

```bash
python scripts/run_grpo_gsm8k.py \
    --model_name Qwen/Qwen3.5-2B \
    --num_epochs 2 \
    --batch_size 2 \
    --group_size 16 \
    --learning_rate 1e-6 \
    --kl_penalty 0.04 \
    --rollout_max_length 512 \
    --use_minitorch
```

### 3. Evaluate checkpoint

```bash
python scripts/evaluate_grpo.py \
    --checkpoint results/<run_dir>/checkpoints/best_model \
    --val_data data/gsm8k_val.json \
    --test_data data/gsm8k_test.json
```

### 4. Benchmark all methods

```bash
python scripts/benchmark_all.py \
    --model_name Qwen/Qwen3.5-2B \
    --methods ours ours_minitorch trl simple_grpo
```

## Baseline Comparisons

Five variants for comparison:

| Method | Script | Description |
|--------|--------|-------------|
| **Ours (PyTorch)** | `scripts/run_grpo_gsm8k.py` | Our GRPO implementation, PyTorch ops |
| **Ours (MiniTorch)** | `scripts/run_grpo_gsm8k.py --use_minitorch` | Same implementation, fused CUDA kernels |
| **TRL** | `baselines/run_trl.py` | HuggingFace TRL GRPOTrainer |
| **VeRL** | `baselines/run_verl.py` | VeRL framework (stub) |
| **Simple_GRPO** | `baselines/run_simple_grpo.py` | [simple_GRPO](https://github.com/lsdefine/simple_GRPO) (stub) |

## Memory Optimizations

- **Log prob computation:** `gather + logsumexp` instead of full `log_softmax` over 151k vocab (~18GB VRAM savings)
- **Mini-batched forward passes:** Splits sequences into chunks for forward/backward to control peak memory
- **Mini-batch gradient accumulation:** Configurable `mini_batch_size` in `train_step`

## PPO and DPO Baselines

The repo also includes PPO and DPO trainers from prior work:

- `scripts/run_ppo.py` — PPO with learned reward model
- `scripts/run_dpo.py` — DPO with preference pairs
- `scripts/evaluate.py` — General evaluation script

These use GPT-2 as the base model and are independent of the GSM8K GRPO pipeline.

## Bridges-2 Cluster

```bash
export HF_HOME=/ocean/projects/cis260009p/atewari1/huggingface
export HF_HUB_CACHE=/ocean/projects/cis260009p/atewari1/huggingface_cache
```

Slurm scripts in `slurm/`:

```bash
sbatch slurm/run_grpo.sh          # GRPO training
sbatch slurm/run_eval_grpo.sh     # GRPO evaluation
sbatch slurm/run_ppo.sh           # PPO baseline
sbatch slurm/run_dpo.sh           # DPO baseline
```

## Tests

```bash
python -m pytest tests/test_gsm8k_reward.py minitorch/tests/test_ops.py -v
```

- `tests/test_gsm8k_reward.py` — Reward function: answer extraction, normalization, scoring
- `minitorch/tests/test_ops.py` — MiniTorch ops: each kernel output vs PyTorch reference
- `tests/test_grpo_trainer.py` — GRPO trainer unit tests
- `tests/test_rlhf_eos_pad.py` — EOS/pad masking for GPT-2 and Qwen3 paths
