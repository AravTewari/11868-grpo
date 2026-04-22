#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/catalyst_minimal.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME="${HF_HOME:-/raid/user_data/aravt/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/raid/user_data/aravt/datasets}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

python scripts/prepare_data.py \
  --output_dir data \
  --max_samples "${MAX_SAMPLES}"

python scripts/train_reward_model.py \
  --config "${CONFIG_PATH}" \
  --num_epochs 1 \
  --batch_size 1

python scripts/run_grpo.py \
  --config "${CONFIG_PATH}" \
  --model_name gpt2 \
  --num_epochs 1 \
  --batch_size 1 \
  --group_size 2 \
  --update_epochs 1 \
  --rollout_max_length 128 \
  --max_train_prompts 16 \
  --max_eval_prompts 8 \
  --eval_steps 100000

python scripts/run_ppo.py \
  --config "${CONFIG_PATH}" \
  --model_name gpt2 \
  --num_epochs 1 \
  --batch_size 1 \
  --max_train_prompts 16 \
  --max_eval_prompts 8 \
  --eval_steps 100000 \
  --rollout_max_length 128 \
  --ppo_update_epochs 1

python scripts/run_dpo.py \
  --config "${CONFIG_PATH}" \
  --model_name gpt2 \
  --num_epochs 1 \
  --batch_size 1 \
  --max_train_pairs 16 \
  --max_val_pairs 8 \
  --max_test_pairs 8 \
  --max_eval_prompts 8

python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model outputs/grpo_model \
  --candidate_label grpo \
  --num_samples 2 \
  --max_prompts 8

python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model outputs/ppo_model \
  --candidate_label ppo \
  --num_samples 2 \
  --max_prompts 8

python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model outputs/dpo_model \
  --candidate_label dpo \
  --num_samples 2 \
  --max_prompts 8

python scripts/build_comparison_table.py \
  --methods grpo ppo dpo \
  --evaluation-dir evaluation_results \
  --logs-dir logs \
  --output-prefix evaluation_results/catalyst_minimal
