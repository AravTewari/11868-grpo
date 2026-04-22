#!/usr/bin/env python3
"""
Lightweight SFT (Supervised Fine-Tuning) warmup on GSM8K solutions.

Teaches the model the chain-of-thought + "#### answer" format so that
GRPO has nonzero reward signal to work with. Without this stage, a base
model (especially GPT-2) will never produce "####" and GRPO receives
all-zero rewards → zero advantages → no learning.

Usage:
    python scripts/sft_warmup.py --model_name gpt2 --max_samples 200 --num_epochs 3
"""

import os
import sys
import json
import argparse
import logging
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from utils import set_seed

logger = logging.getLogger(__name__)


class GSM8KSFTDataset(Dataset):
    """Dataset that concatenates prompt + solution for causal LM training."""

    def __init__(self, data, tokenizer, max_length=512):
        self.examples = []
        for item in data:
            text = item["prompt"] + item["solution"]
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "prompt_len": len(tokenizer(
                    item["prompt"],
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id=0):
    """Pad batch to same length."""
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for ex in batch:
        seq_len = ex["input_ids"].shape[0]
        pad_len = max_len - seq_len
        prompt_len = ex["prompt_len"]

        ids = torch.cat([
            ex["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
        ])
        mask = torch.cat([
            ex["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ])
        lab = ids.clone()
        lab[:prompt_len] = -100
        lab[seq_len:] = -100

        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def run_sft(
    model_name: str = "gpt2",
    sft_data_path: str = "data/gsm8k_sft_data.json",
    output_dir: str = "outputs/sft_model",
    max_samples: int = 0,
    max_length: int = 512,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    seed: int = 42,
    device_str: str = "auto",
):
    """Run SFT warmup training."""
    set_seed(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading SFT data from {sft_data_path}")
    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    if max_samples > 0:
        sft_data = sft_data[:max_samples]
    logger.info(f"Using {len(sft_data)} SFT examples")

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.train()

    dataset = GSM8KSFTDataset(sft_data, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    logger.info(f"Starting SFT: {num_epochs} epochs, {len(loader)} batches/epoch")
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - avg loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    logger.info(f"SFT completed in {training_time:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved SFT model to {output_dir}")

    model.eval()
    test_prompt = sft_data[0]["prompt"]
    encoded = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **encoded,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(gen[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info(f"Sample generation after SFT:\n  Prompt: {test_prompt[:80]}...\n  Response: {response[:200]}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SFT warmup on GSM8K")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--sft_data_path", type=str, default="data/gsm8k_sft_data.json")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_model")
    parser.add_argument("--max_samples", type=int, default=0, help="0 for all")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_sft(
        model_name=args.model_name,
        sft_data_path=args.sft_data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
