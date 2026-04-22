# GRPO Post-Training with PPO and DPO Baselines

This repository is organized around one main goal:

- implement GRPO for post-training a language model
- train a comparable PPO baseline
- train a comparable DPO baseline
- evaluate all three methods on the same prompts so GRPO can be compared against PPO and DPO

The paper runs in this repo currently use Hugging Face `gpt2` as the base policy. GPT-2 itself is not reimplemented here. The work in this repo is the post-training harnessing:

- GRPO is implemented as a new trainer in `src/grpo_trainer.py`
- PPO uses the existing trainer logic in `src/rlhf_trainer.py`, exposed through an explicit PPO harness
- DPO is implemented as a new trainer in `src/dpo_trainer.py`

PPO and GRPO optimize against the learned reward model. DPO trains directly on preference pairs and uses the reward model only for evaluation and model selection.

## MiniTorch Integration

The training and evaluation harnesses are now integrated at the model I/O boundary so a MiniTorch-backed causal LM can be plugged in without changing the PPO, GRPO, or DPO learning logic.

The integration points are:

- `config.model.policy_loader`
- `config.model.policy_saver`

or the equivalent CLI flags on the main harness scripts:

- `--model_loader module:function`
- `--model_saver module:function`

The expected contract is:

- loader: return `(policy_model, tokenizer)` for a model identifier or checkpoint path
- saver: persist `(policy_model, tokenizer)` to a checkpoint directory

If these hooks are not set, the repo falls back to Hugging Face loading and `save_pretrained`.

This means:

- the core GRPO trainer already works with a ready model and tokenizer
- the DPO trainer already works with a ready model and tokenizer
- the PPO harness now also goes through the same external loader path instead of hard-coding Hugging Face model creation
- the evaluator can load base and candidate models through the same external loader path

The current catalyst experiment results were produced with the default Hugging Face GPT-2 path, not with a MiniTorch GPT-2 implementation. The harness is now prepared for MiniTorch integration, but this repository still does not contain a MiniTorch GPT-2 model implementation by itself.

## What To Run

Primary training scripts:

- `scripts/run_grpo.py`
- `scripts/run_ppo.py`
- `scripts/run_dpo.py`

Primary evaluation script:

- `scripts/evaluate.py`

Legacy compatibility entrypoints still exist, but they are not the primary interface:

- `scripts/run_rlhf.py`
- `slurm/run_rlhf.sh`
- `create_rlhf_trainer(...)`

Treat those as PPO aliases only.

## Method Summary

### GRPO

GRPO is the main method under study. The trainer:

- samples a group of completions per prompt
- scores those completions with the reward model
- normalizes rewards within each prompt group
- applies a clipped policy objective on generated tokens only
- keeps a frozen reference policy for optional KL regularization

Core API:

```python
create_grpo_trainer(policy_model, tokenizer, reward_model, config, device)
```

Main output root:

- `outputs/grpo_model`

### PPO

PPO is the reward-model baseline. The harness reuses the existing policy/value implementation and exposes it as a first-class comparison script.

Core API:

```python
create_ppo_trainer(model_name, reward_model, config, device)
```

Main output root:

- `outputs/ppo_model`

### DPO

DPO is the preference-pair baseline. It does not optimize the reward model directly. Instead, it:

- trains on `prompt`, `chosen`, `rejected` triples
- keeps a frozen reference policy
- computes DPO loss on completion tokens only
- uses reward-model evaluation to choose the best checkpoint for comparison parity with PPO and GRPO

Core API:

```python
create_dpo_trainer(policy_model, tokenizer, config, device)
```

Main output root:

- `outputs/dpo_model`

## Recommended Experiment Flow

The intended comparison is:

1. Train the reward model.
2. Post-train GPT-2 with GRPO.
3. Post-train GPT-2 with PPO.
4. Post-train GPT-2 with DPO.
5. Evaluate each resulting model on the same held-out prompts.
6. Compare GRPO against PPO and DPO using the saved evaluation summaries and reward plots.

### 1. Prepare data

```bash
python3 scripts/prepare_data.py \
  --dataset Anthropic/hh-rlhf \
  --output_dir data \
  --max_samples 10000
```

This produces the prompt and preference data used by the downstream scripts.

### 2. Train the reward model

```bash
python3 scripts/train_reward_model.py
```

This creates the reward model used by:

- PPO training
- GRPO training
- reward-based evaluation for PPO, GRPO, and DPO

### 3. Train GRPO

This is the primary method.

Smoke run:

```bash
python3 scripts/run_grpo.py \
  --model_name gpt2 \
  --num_epochs 1 \
  --batch_size 2 \
  --group_size 2 \
  --update_epochs 1 \
  --rollout_max_length 64 \
  --max_train_prompts 16 \
  --max_eval_prompts 4
```

Saved model directories:

- `outputs/grpo_model/best_grpo_model`
- `outputs/grpo_model/final_grpo_model`

### 4. Train PPO

This is the closest reward-model baseline to GRPO.

Smoke run:

```bash
python3 scripts/run_ppo.py \
  --model_name gpt2 \
  --num_epochs 1 \
  --batch_size 2 \
  --max_train_prompts 16 \
  --max_eval_prompts 4 \
  --rollout_max_length 64 \
  --ppo_update_epochs 1
```

Saved model directories:

- `outputs/ppo_model/best_ppo_model`
- `outputs/ppo_model/final_ppo_model`

### 5. Train DPO

This is the preference-learning baseline.

Smoke run:

```bash
python3 scripts/run_dpo.py \
  --model_name gpt2 \
  --num_epochs 1 \
  --batch_size 2 \
  --max_train_pairs 16 \
  --max_val_pairs 4 \
  --max_test_pairs 4 \
  --max_eval_prompts 4
```

Saved model directories:

- `outputs/dpo_model/best_dpo_model`
- `outputs/dpo_model/final_dpo_model`

If `data/preference_data_val.json` and `data/preference_data_test.json` do not exist, the DPO harness deterministically splits `data/preference_data.json` using the configured split ratios.

## How To Compare GRPO Against PPO and DPO

The evaluator compares the base model against one candidate method at a time. That means the comparison workflow is:

1. evaluate GRPO against base GPT-2
2. evaluate PPO against base GPT-2
3. evaluate DPO against base GPT-2
4. compare the resulting summaries side by side

There is no separate direct `grpo vs ppo` evaluator. Instead, all methods are measured against the same base model and prompt set, which gives a clean comparison axis.

### Evaluate GRPO

```bash
python3 scripts/evaluate.py \
  --base_model gpt2 \
  --candidate_model outputs/grpo_model \
  --candidate_label grpo
```

### Evaluate PPO

```bash
python3 scripts/evaluate.py \
  --base_model gpt2 \
  --candidate_model outputs/ppo_model \
  --candidate_label ppo
```

### Evaluate DPO

```bash
python3 scripts/evaluate.py \
  --base_model gpt2 \
  --candidate_model outputs/dpo_model \
  --candidate_label dpo
```

### What To Compare

For the final GRPO vs PPO vs DPO comparison, look at:

- `evaluation_results/evaluation_summary_grpo.json`
- `evaluation_results/evaluation_summary_ppo.json`
- `evaluation_results/evaluation_summary_dpo.json`

and the method-specific plots:

- `plots/reward_comparison_grpo.png`
- `plots/reward_comparison_ppo.png`
- `plots/reward_comparison_dpo.png`

and the training summaries:

- `logs/grpo_training_summary.json`
- `logs/ppo_training_summary.json`
- `logs/dpo_training_summary.json`

In practice, the most important comparison is:

- reward statistics on the held-out evaluation prompts
- sample outputs saved by each harness
- whether GRPO outperforms PPO and DPO under the same evaluation setup

## Output Layout

### GRPO artifacts

- model root: `outputs/grpo_model`
- checkpoints:
  - `best_grpo_model`
  - `final_grpo_model`
  - `grpo_checkpoint_epoch_{n}.pt`
- metrics:
  - `logs/grpo_training_metrics.json`
  - `logs/grpo_training_summary.json`
- plots:
  - `plots/grpo_training_curves.png`
  - `plots/grpo_reward_distribution.png`
- sample outputs:
  - `logs/baseline_outputs_grpo.json`
  - `logs/grpo_outputs.json`
  - `logs/grpo_model_comparison.json`

### PPO artifacts

- model root: `outputs/ppo_model`
- checkpoints:
  - `best_ppo_model`
  - `final_ppo_model`
  - `ppo_checkpoint_epoch_{n}.pt`
- metrics:
  - `logs/ppo_training_metrics.json`
  - `logs/ppo_training_summary.json`
- plots:
  - `plots/ppo_training_curves.png`
  - `plots/ppo_reward_distribution.png`
- sample outputs:
  - `logs/baseline_outputs_ppo.json`
  - `logs/ppo_outputs.json`
  - `logs/ppo_model_comparison.json`

### DPO artifacts

- model root: `outputs/dpo_model`
- checkpoints:
  - `best_dpo_model`
  - `final_dpo_model`
  - `dpo_checkpoint_epoch_{n}.pt`
- metrics:
  - `logs/dpo_training_metrics.json`
  - `logs/dpo_training_summary.json`
- plots:
  - `plots/dpo_training_curves.png`
  - `plots/dpo_reward_distribution.png`
- sample outputs:
  - `logs/baseline_outputs_dpo.json`
  - `logs/dpo_outputs.json`
  - `logs/dpo_model_comparison.json`

## Bridges-2 Usage

Use `/ocean` for Hugging Face cache directories so the home directory does not fill up:

```bash
mkdir -p /ocean/projects/cis260009p/atewari1/huggingface
mkdir -p /ocean/projects/cis260009p/atewari1/huggingface_cache
export HF_HOME=/ocean/projects/cis260009p/atewari1/huggingface
export HF_HUB_CACHE=/ocean/projects/cis260009p/atewari1/huggingface_cache
```

Available Slurm launchers:

- `slurm/train_reward_model.sh`
- `slurm/run_grpo.sh`
- `slurm/run_ppo.sh`
- `slurm/run_dpo.sh`
- `slurm/evaluate.sh`

Submit jobs with:

```bash
sbatch slurm/train_reward_model.sh
sbatch slurm/run_grpo.sh
sbatch slurm/run_ppo.sh
sbatch slurm/run_dpo.sh
sbatch slurm/evaluate.sh
```

The intended cluster workflow is the same as local execution:

1. train reward model
2. run GRPO
3. run PPO
4. run DPO
5. run evaluation for each method
6. compare GRPO against PPO and DPO from the saved summaries

## Tests

Relevant regression coverage:

- `tests/test_grpo_trainer.py`
- `tests/test_ppo_harness.py`
- `tests/test_rlhf_eos_pad.py`
- `tests/test_dpo_trainer.py`
- `tests/test_dpo_harness.py`
- `tests/test_run_dpo_smoke.py`
- `tests/test_evaluate_candidate.py`

Run all tests with:

```bash
python3 -m pytest tests -q
```

## Short Summary

If you only need the essential workflow:

1. `python3 scripts/train_reward_model.py`
2. `python3 scripts/run_grpo.py ...`
3. `python3 scripts/run_ppo.py ...`
4. `python3 scripts/run_dpo.py ...`
5. `python3 scripts/evaluate.py --candidate_model outputs/grpo_model --candidate_label grpo`
6. `python3 scripts/evaluate.py --candidate_model outputs/ppo_model --candidate_label ppo`
7. `python3 scripts/evaluate.py --candidate_model outputs/dpo_model --candidate_label dpo`

Then compare:

- `evaluation_results/evaluation_summary_grpo.json`
- `evaluation_results/evaluation_summary_ppo.json`
- `evaluation_results/evaluation_summary_dpo.json`

That is the comparison path for measuring whether the GRPO implementation improves over PPO and DPO on the same base GPT-2 model and evaluation setup.
