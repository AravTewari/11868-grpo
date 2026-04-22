#!/usr/bin/env python3
"""
Unified benchmark harness: train with each GRPO implementation, evaluate all
checkpoints on the same val/test splits, and produce a comparison table.

Usage:
    python scripts/benchmark_all.py \
        --model_name Qwen/Qwen3.5-2B \
        --train_data data/gsm8k_train.json \
        --val_data data/gsm8k_val.json \
        --test_data data/gsm8k_test.json \
        --output_dir results/benchmark \
        --methods ours ours_minitorch trl simple_grpo
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

logger = logging.getLogger(__name__)

METHODS = {
    "ours":           "scripts/run_grpo_gsm8k.py",
    "ours_minitorch": "scripts/run_grpo_gsm8k.py",
    "trl":            "baselines/run_trl.py",
    "verl":           "baselines/run_verl.py",
    "simple_grpo":    "baselines/run_simple_grpo.py",
}


@dataclass
class BenchmarkResult:
    method: str
    train_time_s: float
    val_accuracy: float
    test_accuracy: float
    peak_gpu_mb: float
    checkpoint_path: str


def run_method(method: str, args: argparse.Namespace) -> BenchmarkResult:
    script = METHODS[method]
    out_dir = os.path.join(args.output_dir, method)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, script,
        "--model_name", args.model_name,
        "--dataset", args.train_data,
        "--output_dir", out_dir,
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--group_size", str(args.group_size),
        "--learning_rate", str(args.learning_rate),
        "--seed", str(args.seed),
    ]

    if method == "ours_minitorch":
        cmd.append("--use_minitorch")

    logger.info(f"[{method}] Starting: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"[{method}] Failed:\n{result.stderr[-500:]}")
        return BenchmarkResult(method, elapsed, 0.0, 0.0, 0.0, out_dir)

    logger.info(f"[{method}] Completed in {elapsed:.1f}s")

    # TODO: run evaluate_grpo.py on checkpoint and parse accuracy
    return BenchmarkResult(
        method=method,
        train_time_s=elapsed,
        val_accuracy=0.0,
        test_accuracy=0.0,
        peak_gpu_mb=0.0,
        checkpoint_path=out_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark all GRPO implementations")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--train_data", type=str, default="data/gsm8k_train.json")
    parser.add_argument("--val_data", type=str, default="data/gsm8k_val.json")
    parser.add_argument("--test_data", type=str, default="data/gsm8k_test.json")
    parser.add_argument("--output_dir", type=str, default="results/benchmark")
    parser.add_argument("--methods", nargs="+", default=["ours", "ours_minitorch", "trl", "simple_grpo"],
                        choices=list(METHODS.keys()))
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for method in args.methods:
        results.append(run_method(method, args))

    # Write comparison table
    table_path = os.path.join(args.output_dir, "comparison.json")
    with open(table_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(f"  {'Method':<20} {'Time (s)':>10} {'Val Acc':>10} {'Test Acc':>10}")
    print("-" * 70)
    for r in results:
        print(f"  {r.method:<20} {r.train_time_s:>10.1f} {r.val_accuracy:>10.4f} {r.test_accuracy:>10.4f}")
    print("=" * 70)
    print(f"\nResults saved to {table_path}")


if __name__ == "__main__":
    main()
