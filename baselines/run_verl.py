#!/usr/bin/env python3
"""
GRPO baseline using ByteDance VeRL.

Usage:
    python baselines/run_verl.py \
        --model_name Qwen/Qwen3.5-2B \
        --dataset data/gsm8k_train.json \
        --output_dir results/verl_grpo \
        --num_epochs 1 --batch_size 16 --group_size 8
"""

import argparse
import json
import logging
import time

import torch

logger = logging.getLogger(__name__)


def build_reward_fn(data_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.gsm8k_reward import GSM8KRewardFunction

    with open(data_path) as f:
        data = json.load(f)
    reward_fn = GSM8KRewardFunction()
    reward_fn.set_prompt_answers(data)
    return reward_fn, data


def main():
    parser = argparse.ArgumentParser(description="VeRL GRPO baseline")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/verl_grpo")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    try:
        from verl.trainer.ppo import GRPOTrainer as VeRLGRPOTrainer
    except ImportError:
        logger.error(
            "verl not installed. See https://github.com/volcengine/verl for installation."
        )
        return

    reward_fn, data = build_reward_fn(args.dataset)
    prompts = [item["prompt"] for item in data]

    # VeRL uses a config-driven approach; adapt to match our hyperparams
    # This is a skeleton — exact VeRL API may require additional config files
    logger.info("Starting VeRL GRPO training")
    logger.info(f"Model: {args.model_name}, prompts: {len(prompts)}, epochs: {args.num_epochs}")

    t0 = time.time()

    # TODO: Wire up VeRL trainer once the exact API is confirmed on the cluster.
    # VeRL typically requires a YAML config and ray/distributed setup.
    # For single-GPU comparison, the key integration points are:
    #   1. verl.trainer.ppo.GRPOTrainer (or verl.single_controller)
    #   2. Custom reward function via verl's RewardManager
    #   3. Model loading via verl's ModelLoader
    logger.warning("VeRL integration is a stub — implement on cluster with verl installed")

    elapsed = time.time() - t0
    logger.info(f"VeRL GRPO training complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
