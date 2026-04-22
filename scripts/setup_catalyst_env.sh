#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
HF_HOME_ROOT="${HF_HOME_ROOT:-/raid/user_data/aravt/huggingface}"
HF_DATASETS_ROOT="${HF_DATASETS_ROOT:-/raid/user_data/aravt/datasets}"

mkdir -p "${HF_HOME_ROOT}" "${HF_DATASETS_ROOT}"
export HF_HOME="${HF_HOME_ROOT}"
export HF_HUB_CACHE="${HF_HOME_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_DATASETS_ROOT}"
export PIP_NO_CACHE_DIR=1

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
python -m pip install --no-cache-dir \
  transformers datasets accelerate numpy scipy pandas scikit-learn matplotlib tqdm pyyaml

python - <<'PY'
import torch
import transformers
import datasets
import accelerate
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    print("gpu_name", torch.cuda.get_device_name(0))
PY
