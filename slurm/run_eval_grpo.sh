#!/bin/bash
set -euo pipefail

# ── Evaluate GRPO checkpoint on val + test ──────────────────────────
python scripts/evaluate_grpo.py \
    --checkpoint checkpoints/best_step30_acc0.3525 \
    --val_data data/gsm8k_val.json \
    --test_data data/gsm8k_test.json \
    --batch_size 64 \
    --max_new_tokens 512 \
    --seed 42
