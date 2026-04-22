#!/usr/bin/env python3
"""
GRPO baseline using HuggingFace TRL.
Wraps trl.GRPOTrainer with the same data/eval interface as our trainer.

Usage:
    python baselines/run_trl.py \
        --model_name Qwen/Qwen3.5-2B \
        --dataset data/gsm8k_train.json \
        --output_dir results/trl_grpo \
        --num_epochs 1 --batch_size 16 --group_size 8
"""

import argparse
import json
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def build_reward_fn(data_path):
    """Build a reward function compatible with TRL's GRPO interface."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.gsm8k_reward import GSM8KRewardFunction

    with open(data_path) as f:
        data = json.load(f)
    reward_fn = GSM8KRewardFunction()
    reward_fn.set_prompt_answers(data)
    return reward_fn, data


def main():
    parser = argparse.ArgumentParser(description="TRL GRPO baseline")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/trl_grpo")
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
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        logger.error("trl not installed. Run: pip install trl")
        return

    reward_fn, data = build_reward_fn(args.dataset)
    prompts = [item["prompt"] for item in data]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # TRL expects a reward function: List[str] -> List[float]
    def trl_reward_fn(completions, **kwargs):
        prompts_batch = kwargs.get("prompts", [""] * len(completions))
        texts = [f"{p} {c}" for p, c in zip(prompts_batch, completions)]
        return reward_fn.get_rewards(texts)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.group_size,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_length,
        temperature=args.temperature,
        seed=args.seed,
        logging_steps=1,
        save_strategy="epoch",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=trl_reward_fn,
        args=config,
    )

    logger.info("Starting TRL GRPO training")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"TRL GRPO training complete in {elapsed:.1f}s")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
