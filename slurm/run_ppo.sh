#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 02:00:00
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gpus=v100-32:1
#SBATCH --output=logs/slurm_ppo_%j.out
#SBATCH --error=logs/slurm_ppo_%j.err

set -euo pipefail

REPO_DIR="${HOME}/llmsys_hw7"
HF_ROOT="/ocean/projects/cis260009p/atewari1"
PYTHON_BIN=""

if [ -x "${HOME}/.conda/envs/progen/bin/python3" ]; then
    PYTHON_BIN="${HOME}/.conda/envs/progen/bin/python3"
elif [ -x "${HOME}/.conda/envs/llmsys/bin/python3" ]; then
    PYTHON_BIN="${HOME}/.conda/envs/llmsys/bin/python3"
else
    echo "Could not find a usable python environment (expected progen or llmsys)." >&2
    exit 1
fi

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${HF_ROOT}/huggingface"
mkdir -p "${HF_ROOT}/huggingface_cache"
mkdir -p "${HF_ROOT}/huggingface_datasets"

export HF_HOME="${HF_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_ROOT}/huggingface_cache"
export HF_DATASETS_CACHE="${HF_ROOT}/huggingface_datasets"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TQDM_DISABLE=1

cd "${REPO_DIR}"

nvidia-smi
"${PYTHON_BIN}" -m pytest tests/test_ppo_harness.py tests/test_evaluate_candidate.py tests/test_rlhf_eos_pad.py -q
"${PYTHON_BIN}" scripts/run_ppo.py \
  --model_name gpt2 \
  --num_epochs 1 \
  --batch_size 2 \
  --max_train_prompts 32 \
  --max_eval_prompts 8 \
  --eval_steps 100000 \
  --rollout_max_length 64 \
  --ppo_update_epochs 1
