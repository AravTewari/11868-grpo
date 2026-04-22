#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/catalyst_compare.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-800}"
PREP_MAX_TRAIN_PROMPTS="${PREP_MAX_TRAIN_PROMPTS:-128}"
PREP_MAX_TRAIN_PAIRS="${PREP_MAX_TRAIN_PAIRS:-128}"
PREP_MAX_EVAL_PROMPTS="${PREP_MAX_EVAL_PROMPTS:-32}"
PREP_MAX_VAL_PAIRS="${PREP_MAX_VAL_PAIRS:-32}"
PREP_MAX_TEST_PAIRS="${PREP_MAX_TEST_PAIRS:-32}"

export HF_HOME="${HF_HOME:-/raid/user_data/aravt/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/raid/user_data/aravt/datasets}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

mkdir -p logs_compare evaluation_results outputs_compare

python scripts/prepare_data.py \
  --output_dir data \
  --max_samples "${MAX_SAMPLES}"

CUDA_VISIBLE_DEVICES=0 python scripts/train_reward_model.py \
  --config "${CONFIG_PATH}" \
  --num_epochs 2 \
  --batch_size 2

CUDA_VISIBLE_DEVICES=0 python scripts/run_grpo.py \
  --config "${CONFIG_PATH}" \
  --model_name gpt2 \
  --learning_rate 2e-6 \
  --num_epochs 2 \
  --batch_size 1 \
  --group_size 4 \
  --update_epochs 2 \
  --rollout_max_length 128 \
  --max_train_prompts "${PREP_MAX_TRAIN_PROMPTS}" \
  --max_eval_prompts "${PREP_MAX_EVAL_PROMPTS}" \
  --eval_steps 100000 \
  > logs_compare/grpo_compare.out 2>&1 &
GRPO_PID=$!

CUDA_VISIBLE_DEVICES=1 python scripts/run_ppo.py \
  --config "${CONFIG_PATH}" \
  --model_name gpt2 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --batch_size 1 \
  --max_train_prompts "${PREP_MAX_TRAIN_PROMPTS}" \
  --max_eval_prompts "${PREP_MAX_EVAL_PROMPTS}" \
  --eval_steps 100000 \
  --rollout_max_length 128 \
  --ppo_update_epochs 1 \
  > logs_compare/ppo_compare.out 2>&1 &
PPO_PID=$!

CUDA_VISIBLE_DEVICES=2 python scripts/run_dpo.py \
  --config "${CONFIG_PATH}" \
  --model_name gpt2 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --batch_size 1 \
  --max_train_pairs "${PREP_MAX_TRAIN_PAIRS}" \
  --max_val_pairs "${PREP_MAX_VAL_PAIRS}" \
  --max_test_pairs "${PREP_MAX_TEST_PAIRS}" \
  --max_eval_prompts "${PREP_MAX_EVAL_PROMPTS}" \
  > logs_compare/dpo_compare.out 2>&1 &
DPO_PID=$!

wait "${GRPO_PID}"
wait "${PPO_PID}"
wait "${DPO_PID}"

CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model outputs_compare/grpo_model \
  --candidate_label grpo \
  --num_samples 2 \
  --max_prompts "${PREP_MAX_EVAL_PROMPTS}"

CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model outputs_compare/ppo_model \
  --candidate_label ppo \
  --num_samples 2 \
  --max_prompts "${PREP_MAX_EVAL_PROMPTS}"

CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model outputs_compare/dpo_model \
  --candidate_label dpo \
  --num_samples 2 \
  --max_prompts "${PREP_MAX_EVAL_PROMPTS}"

python scripts/build_comparison_table.py \
  --methods grpo ppo dpo \
  --evaluation-dir evaluation_results \
  --logs-dir logs_compare \
  --output-prefix evaluation_results/catalyst_compare
