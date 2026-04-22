#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 08:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=logs/slurm_reward_%j.out
#SBATCH --error=logs/slurm_reward_%j.err

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

module load anaconda3/2024.10-1
cd "${REPO_DIR}"

nvidia-smi
"${PYTHON_BIN}" -m pytest tests/test_reward_model.py -q
"${PYTHON_BIN}" scripts/prepare_data.py --dataset Anthropic/hh-rlhf --output_dir data --max_samples 10000
"${PYTHON_BIN}" scripts/train_reward_model.py
