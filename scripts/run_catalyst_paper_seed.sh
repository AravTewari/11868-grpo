#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_DIR}/configs/catalyst_compare.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-paper_results}"
SHARED_REWARD_MODEL_DIR="${SHARED_REWARD_MODEL_DIR:-outputs_compare/reward_model}"
SEED="${SEED:?Set SEED to the experiment seed to run}"

GRPO_GPU="${GRPO_GPU:-0}"
PPO_GPU="${PPO_GPU:-1}"
DPO_GPU="${DPO_GPU:-2}"
EVAL_GPU="${EVAL_GPU:-3}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-40}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-3}"

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

if [[ ! -f data/train_prompts.json || ! -f data/eval_prompts.json ]]; then
  echo "Prepared data files are missing under data/. Run scripts/run_catalyst_compare.sh first." >&2
  exit 1
fi

if [[ ! -f "${SHARED_REWARD_MODEL_DIR}/best_reward_model.pt" && ! -f "${SHARED_REWARD_MODEL_DIR}/final_reward_model.pt" ]]; then
  echo "Shared reward model not found in ${SHARED_REWARD_MODEL_DIR}." >&2
  exit 1
fi

RUN_DIR="${RESULTS_ROOT}/seed_${SEED}"
RUN_OUTPUT_DIR="${RUN_DIR}/outputs"
RUN_LOGS_DIR="${RUN_DIR}/logs"
RUN_EVAL_DIR="${RUN_DIR}/evaluation_results"
RUN_PLOTS_DIR="${RUN_DIR}/plots"
GENERATED_CONFIG_DIR="${REPO_DIR}/configs/generated"
GENERATED_CONFIG_PATH="${GENERATED_CONFIG_DIR}/catalyst_paper_seed_${SEED}.yaml"

mkdir -p "${RUN_OUTPUT_DIR}" "${RUN_LOGS_DIR}" "${RUN_EVAL_DIR}" "${RUN_PLOTS_DIR}" "${GENERATED_CONFIG_DIR}"

python - "${BASE_CONFIG}" "${GENERATED_CONFIG_PATH}" "${SEED}" "${RUN_OUTPUT_DIR}" "${RUN_LOGS_DIR}" "${SHARED_REWARD_MODEL_DIR}" <<'PY'
import sys
from pathlib import Path

import yaml

base_config_path = Path(sys.argv[1])
generated_config_path = Path(sys.argv[2])
seed = int(sys.argv[3])
run_output_dir = Path(sys.argv[4])
run_logs_dir = Path(sys.argv[5])
shared_reward_model_dir = sys.argv[6]

with base_config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config.setdefault("experiment", {})
config.setdefault("system", {})
config.setdefault("model", {})

config["experiment"]["seed"] = seed
config["experiment"]["experiment_name"] = f"catalyst_paper_seed_{seed}"

config["system"]["output_dir"] = str(run_output_dir)
config["system"]["reward_model_dir"] = shared_reward_model_dir
config["system"]["ppo_model_dir"] = str(run_output_dir / "ppo_model")
config["system"]["rlhf_model_dir"] = str(run_output_dir / "ppo_model")
config["system"]["grpo_model_dir"] = str(run_output_dir / "grpo_model")
config["system"]["dpo_model_dir"] = str(run_output_dir / "dpo_model")
config["system"]["logs_dir"] = str(run_logs_dir)

generated_config_path.parent.mkdir(parents=True, exist_ok=True)
with generated_config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

cp "${GENERATED_CONFIG_PATH}" "${RUN_DIR}/config.yaml"

capture_eval_artifacts() {
  local method="$1"
  shopt -s nullglob
  for f in evaluation_results/*_"${method}".json; do
    cp "${f}" "${RUN_EVAL_DIR}/"
  done
  for f in plots/*"${method}"*.png; do
    cp "${f}" "${RUN_PLOTS_DIR}/"
  done
  shopt -u nullglob
}

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${GRPO_GPU}" python scripts/run_grpo.py \
  --config "${GENERATED_CONFIG_PATH}" \
  > "${RUN_LOGS_DIR}/grpo_seed_${SEED}.out" 2>&1 &
PIDS+=("$!")

CUDA_VISIBLE_DEVICES="${PPO_GPU}" python scripts/run_ppo.py \
  --config "${GENERATED_CONFIG_PATH}" \
  > "${RUN_LOGS_DIR}/ppo_seed_${SEED}.out" 2>&1 &
PIDS+=("$!")

CUDA_VISIBLE_DEVICES="${DPO_GPU}" python scripts/run_dpo.py \
  --config "${GENERATED_CONFIG_PATH}" \
  > "${RUN_LOGS_DIR}/dpo_seed_${SEED}.out" 2>&1 &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done
PIDS=()

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python scripts/evaluate.py \
  --config "${GENERATED_CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model "${RUN_OUTPUT_DIR}/grpo_model" \
  --candidate_label grpo \
  --num_samples "${EVAL_NUM_SAMPLES}" \
  --max_prompts "${EVAL_MAX_PROMPTS}" \
  > "${RUN_LOGS_DIR}/evaluate_grpo_seed_${SEED}.out" 2>&1
capture_eval_artifacts grpo

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python scripts/evaluate.py \
  --config "${GENERATED_CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model "${RUN_OUTPUT_DIR}/ppo_model" \
  --candidate_label ppo \
  --num_samples "${EVAL_NUM_SAMPLES}" \
  --max_prompts "${EVAL_MAX_PROMPTS}" \
  > "${RUN_LOGS_DIR}/evaluate_ppo_seed_${SEED}.out" 2>&1
capture_eval_artifacts ppo

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python scripts/evaluate.py \
  --config "${GENERATED_CONFIG_PATH}" \
  --base_model gpt2 \
  --candidate_model "${RUN_OUTPUT_DIR}/dpo_model" \
  --candidate_label dpo \
  --num_samples "${EVAL_NUM_SAMPLES}" \
  --max_prompts "${EVAL_MAX_PROMPTS}" \
  > "${RUN_LOGS_DIR}/evaluate_dpo_seed_${SEED}.out" 2>&1
capture_eval_artifacts dpo

echo "Completed seed ${SEED}. Results stored in ${RUN_DIR}"
