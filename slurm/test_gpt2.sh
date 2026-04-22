#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 02:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=logs/slurm_test_gpt2_%j.out
#SBATCH --error=logs/slurm_test_gpt2_%j.err

# ============================================================
# GSM8K GRPO pipeline test with GPT-2
#
# Full pipeline: tests → data prep → SFT warmup → GRPO training
#
# SFT warmup teaches the model the "#### answer" output format
# so GRPO has nonzero reward signal to learn from.
#
# Also runs locally without SLURM:
#   bash slurm/test_gpt2.sh
# ============================================================

set -euo pipefail

REPO_DIR="${HOME}/llmsys_hw7"
HF_ROOT="/ocean/projects/cis260009p/atewari1"

REPO_DIR="${GRPO_REPO_DIR:-${REPO_DIR}}"
HF_ROOT="${GRPO_HF_ROOT:-${HF_ROOT}}"

PYTHON_BIN=""
if [ -x "${HOME}/.conda/envs/progen/bin/python3" ]; then
    PYTHON_BIN="${HOME}/.conda/envs/progen/bin/python3"
elif [ -x "${HOME}/.conda/envs/llmsys/bin/python3" ]; then
    PYTHON_BIN="${HOME}/.conda/envs/llmsys/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "Could not find a usable python." >&2
    exit 1
fi
echo "Using Python: ${PYTHON_BIN}"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/results"
mkdir -p "${HF_ROOT}/huggingface" 2>/dev/null || true
mkdir -p "${HF_ROOT}/huggingface_cache" 2>/dev/null || true
mkdir -p "${HF_ROOT}/huggingface_datasets" 2>/dev/null || true

export HF_HOME="${HF_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_ROOT}/huggingface_cache"
export HF_DATASETS_CACHE="${HF_ROOT}/huggingface_datasets"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

module load anaconda3/2024.10-1 2>/dev/null || true

cd "${REPO_DIR}"

nvidia-smi 2>/dev/null || echo "No GPU detected (running on CPU)"

# ============================================================
#  Step 1: Unit tests
# ============================================================
echo ""
echo "=========================================="
echo "  Step 1/4: Running unit tests"
echo "=========================================="
"${PYTHON_BIN}" -m pytest tests/test_grpo_trainer.py tests/test_gsm8k_reward.py tests/test_rlhf_eos_pad.py -v

# ============================================================
#  Step 2: Prepare GSM8K data
# ============================================================
echo ""
echo "=========================================="
echo "  Step 2/4: Preparing GSM8K data"
echo "=========================================="
if [ ! -f data/gsm8k_sft_data.json ]; then
    "${PYTHON_BIN}" scripts/prepare_gsm8k.py --output_dir data
else
    echo "GSM8K data already exists, skipping download."
fi

# ============================================================
#  Step 3: SFT warmup (teach "#### answer" format)
# ============================================================
echo ""
echo "=========================================="
echo "  Step 3/4: SFT warmup on GSM8K solutions"
echo "=========================================="
SFT_DIR="outputs/sft_gpt2"
if [ ! -d "${SFT_DIR}" ]; then
    "${PYTHON_BIN}" scripts/sft_warmup.py \
        --model_name gpt2 \
        --sft_data_path data/gsm8k_sft_data.json \
        --output_dir "${SFT_DIR}" \
        --max_samples 200 \
        --max_length 512 \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 5e-5
else
    echo "SFT model already exists at ${SFT_DIR}, skipping."
fi

# ============================================================
#  Step 4: GRPO training (using SFT-warmed model)
# ============================================================
echo ""
echo "=========================================="
echo "  Step 4/4: GRPO training (GPT-2 on GSM8K)"
echo "=========================================="
"${PYTHON_BIN}" scripts/run_grpo_gsm8k.py \
    --model_name gpt2 \
    --sft_model_path "${SFT_DIR}" \
    --num_epochs 1 \
    --batch_size 4 \
    --group_size 4 \
    --update_epochs 2 \
    --rollout_max_length 256 \
    --max_train_prompts 50 \
    --max_eval_prompts 20 \
    --eval_steps 100000 \
    --kl_penalty 0.04 \
    --temperature 1.0 \
    --results_root results

echo ""
echo "=========================================="
echo "  Done! Check results/ for outputs."
echo "=========================================="
