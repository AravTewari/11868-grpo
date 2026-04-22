#!/usr/bin/env python3
"""
Build a compact PPO/GRPO/DPO comparison table from saved summary artifacts.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple


def load_summary(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[0] if data else None
    return data


def get_candidate_metrics(summary: Dict) -> Tuple[str, Dict]:
    for key, value in summary.items():
        if key.endswith("_model_metrics") and key != "base_model_metrics":
            return key[: -len("_model_metrics")], value
    raise ValueError("Could not find candidate model metrics in summary")


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def collect_row(method: str, evaluation_dir: str, logs_dir: str) -> Dict[str, str]:
    eval_summary = load_summary(
        os.path.join(evaluation_dir, f"evaluation_summary_{method}.json")
    )
    training_summary = load_summary(
        os.path.join(logs_dir, f"{method}_training_summary.json")
    )

    if eval_summary is None and training_summary is None:
        raise FileNotFoundError(f"No summary artifacts found for method '{method}'")

    metric_source = eval_summary or training_summary
    base_metrics = metric_source.get("base_model_metrics", {})
    candidate_key, candidate_metrics = get_candidate_metrics(metric_source)
    timing_source = training_summary or metric_source
    timing = timing_source.get("training_summary", {})
    improvements = metric_source.get("improvements", {})

    return {
        "method": candidate_key.upper(),
        "train_minutes": format_float(timing.get("training_time_minutes"), digits=2),
        "base_mean_reward": format_float(base_metrics.get("mean_reward")),
        "candidate_mean_reward": format_float(candidate_metrics.get("mean_reward")),
        "reward_improvement_abs": format_float(
            improvements.get("mean_reward_improvement_abs")
        ),
        "reward_improvement_pct": format_float(
            improvements.get("mean_reward_improvement_pct"), digits=2
        ),
        "candidate_std_reward": format_float(candidate_metrics.get("std_reward")),
        "candidate_mean_response_length": format_float(
            candidate_metrics.get("mean_response_length"), digits=2
        ),
        "num_prompts": str(candidate_metrics.get("num_prompts", "")),
        "num_samples": str(candidate_metrics.get("num_samples", "")),
    }


def write_csv(rows: List[Dict[str, str]], path: str) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], path: str) -> None:
    headers = list(rows[0].keys())
    pretty_headers = [
        "Method",
        "Train min",
        "Base reward",
        "Candidate reward",
        "Abs improve",
        "% improve",
        "Cand std reward",
        "Cand resp len",
        "Prompts",
        "Samples",
    ]
    lines = [
        "| " + " | ".join(pretty_headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a method comparison table")
    parser.add_argument("--methods", nargs="+", default=["grpo", "ppo", "dpo"])
    parser.add_argument("--evaluation-dir", default="evaluation_results")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--output-prefix",
        default="evaluation_results/method_comparison",
        help="Output path prefix without extension",
    )
    args = parser.parse_args()

    rows = [
        collect_row(method=method, evaluation_dir=args.evaluation_dir, logs_dir=args.logs_dir)
        for method in args.methods
    ]

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    csv_path = f"{args.output_prefix}.csv"
    md_path = f"{args.output_prefix}.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"Wrote comparison table to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
