#!/usr/bin/env python3
"""
Script to download and prepare GSM8K data for GRPO training.
Produces JSON files with prompts and ground truth answers.
"""

import os
import sys
import json
import argparse
import logging
import re

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)


def normalize_answer(answer_str: str) -> str:
    """Normalize a numeric answer: remove commas, whitespace, trailing periods."""
    answer_str = answer_str.strip().replace(",", "").rstrip(".")
    return answer_str.strip()


def extract_answer_from_solution(solution: str) -> str:
    """Extract the final numeric answer after #### from a GSM8K solution string."""
    if "####" not in solution:
        raise ValueError(f"No #### delimiter found in solution: {solution[:100]}...")
    after_marker = solution.split("####")[-1]
    return normalize_answer(after_marker)


def format_prompt(question: str, template: str) -> str:
    """Format a GSM8K question into a prompt string."""
    return template.format(question=question.strip())


def prepare_gsm8k(
    output_dir: str,
    prompt_template: str = "Question: {question}\nAnswer: Let's think step by step.\n",
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
):
    """Download GSM8K and prepare train/eval JSON files."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Downloading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")

    train_data = []
    sft_data = []
    skipped = 0
    for example in dataset["train"]:
        try:
            answer = extract_answer_from_solution(example["answer"])
            prompt = format_prompt(example["question"], prompt_template)
            train_data.append({"prompt": prompt, "answer": answer})
            solution = re.sub(r"<<.*?>>", "", example["answer"])
            sft_data.append({
                "prompt": prompt,
                "answer": answer,
                "solution": solution.strip(),
            })
        except ValueError as e:
            logger.warning(f"Skipping train example: {e}")
            skipped += 1

    if max_train_samples > 0:
        train_data = train_data[:max_train_samples]

    logger.info(f"Prepared {len(train_data)} training prompts (skipped {skipped})")

    eval_data = []
    skipped = 0
    for example in dataset["test"]:
        try:
            answer = extract_answer_from_solution(example["answer"])
            prompt = format_prompt(example["question"], prompt_template)
            eval_data.append({"prompt": prompt, "answer": answer})
        except ValueError as e:
            logger.warning(f"Skipping eval example: {e}")
            skipped += 1

    if max_eval_samples > 0:
        eval_data = eval_data[:max_eval_samples]

    logger.info(f"Prepared {len(eval_data)} evaluation prompts (skipped {skipped})")

    train_path = os.path.join(output_dir, "gsm8k_train_prompts.json")
    eval_path = os.path.join(output_dir, "gsm8k_eval_prompts.json")

    sft_path = os.path.join(output_dir, "gsm8k_sft_data.json")

    if max_train_samples > 0:
        sft_data = sft_data[:max_train_samples]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved training data to {train_path}")

    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved SFT data ({len(sft_data)} examples) to {sft_path}")

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation data to {eval_path}")

    if train_data:
        sample = train_data[0]
        logger.info(f"Sample prompt:\n{sample['prompt']}")
        logger.info(f"Sample answer: {sample['answer']}")

    return train_data, eval_data


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K data for GRPO training")
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Directory to save output files"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Question: {question}\nAnswer: Let's think step by step.\n",
        help="Prompt template with {question} placeholder",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Max training samples (0 for all)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=0,
        help="Max evaluation samples (0 for all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    prepare_gsm8k(
        output_dir=args.output_dir,
        prompt_template=args.prompt_template,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
