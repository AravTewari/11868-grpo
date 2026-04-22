#!/usr/bin/env python3
"""
GRPO baseline using simple_GRPO (https://github.com/lsdefine/simple_GRPO).

Usage:
    python baselines/run_simple_grpo.py \
        --model_name Qwen/Qwen3.5-2B \
        --dataset data/gsm8k_train.json \
        --output_dir results/simple_grpo \
        --num_epochs 1 --batch_size 16 --group_size 8
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

logger = logging.getLogger(__name__)

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_reward_fn(data_path):
    from src.gsm8k_reward import GSM8KRewardFunction

    with open(data_path) as f:
        data = json.load(f)
    reward_fn = GSM8KRewardFunction()
    reward_fn.set_prompt_answers(data)
    return reward_fn, data


def main():
    parser = argparse.ArgumentParser(description="simple_GRPO baseline")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/simple_grpo")
    parser.add_argument("--simple_grpo_path", type=str, default="third_party/simple_GRPO",
                        help="Path to cloned simple_GRPO repo")
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

    # Add simple_GRPO to path
    if os.path.isdir(args.simple_grpo_path):
        sys.path.insert(0, args.simple_grpo_path)
    else:
        logger.error(
            f"simple_GRPO not found at {args.simple_grpo_path}. "
            "Clone it: git clone https://github.com/lsdefine/simple_GRPO third_party/simple_GRPO"
        )
        return

    try:
        from grpo import GRPOTrainer as SimpleGRPOTrainer
    except ImportError:
        logger.error("Cannot import from simple_GRPO. Check the repo structure.")
        return

    reward_fn, data = build_reward_fn(args.dataset)
    prompts = [item["prompt"] for item in data]

    logger.info("Starting simple_GRPO training")
    logger.info(f"Model: {args.model_name}, prompts: {len(prompts)}, epochs: {args.num_epochs}")

    t0 = time.time()

    # TODO: Wire up simple_GRPO trainer with our reward function.
    # simple_GRPO has a minimal interface:
    #   trainer = SimpleGRPOTrainer(model, tokenizer, reward_fn, ...)
    #   trainer.train(prompts)
    # Adapt the reward function to match their expected signature.
    logger.warning("simple_GRPO integration is a stub — implement after cloning the repo")

    elapsed = time.time() - t0
    logger.info(f"simple_GRPO training complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
