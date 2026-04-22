#!/usr/bin/env python3
"""
Evaluate a GRPO (or SFT) checkpoint on GSM8K val and test sets (greedy only).

Produces:
  results/grpo_eval_<ckpt_name>_<timestamp>/
    val_results.json    — per-record detail for val set
    test_results.json   — per-record detail for test set
    summary.txt         — aggregate statistics

Usage:
    python scripts/evaluate_grpo.py \
        --checkpoint checkpoints/grpo_final \
        --batch_size 64 \
        --max_new_tokens 512
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.gsm8k_reward import GSM8KRewardFunction

logger = logging.getLogger(__name__)


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_split(model, tokenizer, reward_fn, data, batch_size, max_new_tokens, device, desc="Evaluating"):
    total_correct = 0
    total_parse_success = 0
    total_records = len(data)
    results = []

    pbar = tqdm(range(0, total_records, batch_size), desc=desc, unit="batch")
    for batch_start in pbar:
        batch = data[batch_start : batch_start + batch_size]
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]

        encoded = tokenizer(
            prompts, padding=True, truncation=True,
            return_tensors="pt", max_length=max_new_tokens,
        ).to(device)
        prompt_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = outputs[:, prompt_len:]
        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        for i in range(len(batch)):
            gt_answer = GSM8KRewardFunction._normalize_answer(answers[i])
            parsed = reward_fn.extract_answer_from_response(responses[i], "####")
            parse_ok = parsed is not None
            is_correct = parse_ok and reward_fn._normalize_answer(parsed) == gt_answer

            if parse_ok:
                total_parse_success += 1
            if is_correct:
                total_correct += 1

            results.append({
                "question": prompts[i],
                "ground_truth": gt_answer,
                "is_correct": is_correct,
                "parsed_answer": parsed,
                "response": responses[i],
            })

        done = min(batch_start + batch_size, total_records)
        pbar.set_postfix({"acc": f"{total_correct / done:.3f}", "done": done})

    return results, {
        "greedy_accuracy": total_correct / total_records,
        "parse_rate": total_parse_success / total_records,
        "num_correct": total_correct,
        "num_evaluated": total_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoint on GSM8K")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="data/gsm8k_val.json")
    parser.add_argument("--test_data", type=str, default="data/gsm8k_test.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"
    model.to(device)
    model.eval()

    reward_fn = GSM8KRewardFunction()

    ckpt_name = os.path.basename(args.checkpoint.rstrip("/\\"))
    run_name = f"grpo_eval_{ckpt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary_lines = [
        "=" * 70,
        "  GRPO Checkpoint Evaluation — GSM8K val + test (greedy)",
        "=" * 70, "",
        f"  Checkpoint:        {args.checkpoint}",
        f"  Batch size:        {args.batch_size}",
        f"  Max new tokens:    {args.max_new_tokens}",
        f"  Seed:              {args.seed}",
        f"  Device:            {device}", "",
    ]

    for split_name, data_path in [("val", args.val_data), ("test", args.test_data)]:
        logger.info(f"Loading {split_name} data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"  {len(data)} examples")

        reward_fn.set_prompt_answers(data)

        start_time = time.time()
        results, stats = evaluate_split(
            model, tokenizer, reward_fn, data,
            args.batch_size, args.max_new_tokens, device,
            desc=f"Eval {split_name}",
        )
        elapsed = time.time() - start_time

        with open(os.path.join(run_dir, f"{split_name}_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary_lines += [
            f"{split_name.upper()} Set ({stats['num_evaluated']} examples)",
            "-" * 40,
            f"  Greedy accuracy:   {stats['greedy_accuracy']:.4f}  ({stats['num_correct']}/{stats['num_evaluated']})",
            f"  Parse rate:        {stats['parse_rate']:.4f}",
            f"  Time:              {elapsed:.1f}s", "",
        ]

    summary_lines.append("=" * 70)
    summary_text = "\n".join(summary_lines) + "\n"

    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
