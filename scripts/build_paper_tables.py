#!/usr/bin/env python3
"""
Build paper-ready PPO/GRPO/DPO comparison tables from multi-seed artifacts.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0] if data else {}
    return data


def get_candidate_metrics(summary: Dict) -> Tuple[str, Dict]:
    for key, value in summary.items():
        if key.endswith("_model_metrics") and key != "base_model_metrics":
            return key[: -len("_model_metrics")], value
    raise ValueError("Could not find candidate model metrics in summary")


def safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_pm(values: List[float], digits: int = 4) -> str:
    return f"{mean(values):.{digits}f} ± {safe_stdev(values):.{digits}f}"


def collect_seed_method_row(results_root: str, seed: int, method: str) -> Dict[str, float]:
    seed_dir = os.path.join(results_root, f"seed_{seed}")
    eval_path = os.path.join(
        seed_dir, "evaluation_results", f"evaluation_summary_{method}.json"
    )
    train_path = os.path.join(seed_dir, "logs", f"{method}_training_summary.json")

    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing evaluation summary: {eval_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training summary: {train_path}")

    eval_summary = load_summary(eval_path)
    train_summary = load_summary(train_path)

    candidate_key, candidate_metrics = get_candidate_metrics(eval_summary)
    base_metrics = eval_summary.get("base_model_metrics", {})
    improvements = eval_summary.get("improvements", {})
    training = train_summary.get("training_summary", {})

    train_minutes = float(training.get("training_time_minutes", 0.0))
    reward_gain = float(improvements.get("mean_reward_improvement_abs", 0.0))
    gain_per_minute = reward_gain / train_minutes if train_minutes > 0 else math.nan

    return {
        "seed": seed,
        "method": candidate_key.upper(),
        "base_mean_reward": float(base_metrics.get("mean_reward", 0.0)),
        "candidate_mean_reward": float(candidate_metrics.get("mean_reward", 0.0)),
        "reward_improvement_abs": reward_gain,
        "reward_improvement_pct": float(improvements.get("mean_reward_improvement_pct", 0.0)),
        "candidate_std_reward": float(candidate_metrics.get("std_reward", 0.0)),
        "candidate_mean_response_length": float(candidate_metrics.get("mean_response_length", 0.0)),
        "train_minutes": train_minutes,
        "gain_per_train_minute": gain_per_minute,
        "num_prompts": float(candidate_metrics.get("num_prompts", 0.0)),
        "num_samples": float(candidate_metrics.get("num_samples", 0.0)),
    }


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(headers: List[str], rows: List[List[str]], path: str) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_main_table(rows: List[Dict[str, float]], methods: List[str]) -> Tuple[List[Dict[str, object]], List[List[str]]]:
    by_method: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    csv_rows: List[Dict[str, object]] = []
    md_rows: List[List[str]] = []

    for method in [m.upper() for m in methods]:
        method_rows = by_method[method]
        base_rewards = [r["base_mean_reward"] for r in method_rows]
        candidate_rewards = [r["candidate_mean_reward"] for r in method_rows]
        reward_gains = [r["reward_improvement_abs"] for r in method_rows]
        reward_gains_pct = [r["reward_improvement_pct"] for r in method_rows]
        train_minutes = [r["train_minutes"] for r in method_rows]
        gain_per_minute = [r["gain_per_train_minute"] for r in method_rows]
        std_rewards = [r["candidate_std_reward"] for r in method_rows]
        response_lengths = [r["candidate_mean_response_length"] for r in method_rows]

        csv_rows.append(
            {
                "method": method,
                "base_reward_mean": mean(base_rewards),
                "base_reward_std": safe_stdev(base_rewards),
                "candidate_reward_mean": mean(candidate_rewards),
                "candidate_reward_std": safe_stdev(candidate_rewards),
                "reward_gain_mean": mean(reward_gains),
                "reward_gain_std": safe_stdev(reward_gains),
                "reward_gain_pct_mean": mean(reward_gains_pct),
                "reward_gain_pct_std": safe_stdev(reward_gains_pct),
                "train_minutes_mean": mean(train_minutes),
                "train_minutes_std": safe_stdev(train_minutes),
                "gain_per_train_minute_mean": mean(gain_per_minute),
                "gain_per_train_minute_std": safe_stdev(gain_per_minute),
                "candidate_std_reward_mean": mean(std_rewards),
                "candidate_std_reward_std": safe_stdev(std_rewards),
                "candidate_response_length_mean": mean(response_lengths),
                "candidate_response_length_std": safe_stdev(response_lengths),
                "num_seeds": len(method_rows),
            }
        )

        md_rows.append(
            [
                method,
                fmt_pm(base_rewards, digits=4),
                fmt_pm(candidate_rewards, digits=4),
                fmt_pm(reward_gains, digits=4),
                fmt_pm(reward_gains_pct, digits=2),
                fmt_pm(train_minutes, digits=2),
                fmt_pm(gain_per_minute, digits=4),
                fmt_pm(std_rewards, digits=4),
                fmt_pm(response_lengths, digits=2),
            ]
        )

    return csv_rows, md_rows


def build_pairwise_grpo_table(rows: List[Dict[str, float]]) -> Tuple[List[Dict[str, object]], List[List[str]]]:
    by_seed_method = {(int(row["seed"]), row["method"]): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})

    csv_rows: List[Dict[str, object]] = []
    md_rows: List[List[str]] = []

    for other in ("PPO", "DPO"):
        reward_gaps = []
        improvement_gaps = []
        train_minute_deltas = []
        gain_per_minute_gaps = []
        wins = 0

        for seed in seeds:
            grpo_row = by_seed_method[(seed, "GRPO")]
            other_row = by_seed_method[(seed, other)]
            reward_gap = grpo_row["candidate_mean_reward"] - other_row["candidate_mean_reward"]
            improvement_gap = grpo_row["reward_improvement_abs"] - other_row["reward_improvement_abs"]
            train_delta = grpo_row["train_minutes"] - other_row["train_minutes"]
            efficiency_gap = grpo_row["gain_per_train_minute"] - other_row["gain_per_train_minute"]

            reward_gaps.append(reward_gap)
            improvement_gaps.append(improvement_gap)
            train_minute_deltas.append(train_delta)
            gain_per_minute_gaps.append(efficiency_gap)
            if reward_gap > 0:
                wins += 1

        csv_rows.append(
            {
                "comparison": f"GRPO_vs_{other}",
                "reward_gap_mean": mean(reward_gaps),
                "reward_gap_std": safe_stdev(reward_gaps),
                "improvement_gap_mean": mean(improvement_gaps),
                "improvement_gap_std": safe_stdev(improvement_gaps),
                "train_minute_delta_mean": mean(train_minute_deltas),
                "train_minute_delta_std": safe_stdev(train_minute_deltas),
                "gain_per_minute_gap_mean": mean(gain_per_minute_gaps),
                "gain_per_minute_gap_std": safe_stdev(gain_per_minute_gaps),
                "grpo_win_count": wins,
                "num_seeds": len(seeds),
            }
        )

        md_rows.append(
            [
                f"GRPO vs {other}",
                f"{wins}/{len(seeds)}",
                fmt_pm(reward_gaps, digits=4),
                fmt_pm(improvement_gaps, digits=4),
                fmt_pm(train_minute_deltas, digits=2),
                fmt_pm(gain_per_minute_gaps, digits=4),
            ]
        )

    return csv_rows, md_rows


def build_seed_table(rows: List[Dict[str, float]]) -> Tuple[List[Dict[str, object]], List[List[str]]]:
    by_seed: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["method"]] = row

    csv_rows: List[Dict[str, object]] = []
    md_rows: List[List[str]] = []

    for seed in sorted(by_seed):
        seed_rows = by_seed[seed]
        winner = max(seed_rows.items(), key=lambda item: item[1]["candidate_mean_reward"])[0]
        csv_row = {
            "seed": seed,
            "winner": winner,
        }
        md_row = [str(seed), winner]

        for method in ("GRPO", "PPO", "DPO"):
            method_row = seed_rows[method]
            csv_row[f"{method.lower()}_reward"] = method_row["candidate_mean_reward"]
            csv_row[f"{method.lower()}_gain"] = method_row["reward_improvement_abs"]
            csv_row[f"{method.lower()}_train_minutes"] = method_row["train_minutes"]
            md_row.extend(
                [
                    fmt(method_row["candidate_mean_reward"], digits=4),
                    fmt(method_row["reward_improvement_abs"], digits=4),
                    fmt(method_row["train_minutes"], digits=2),
                ]
            )

        csv_rows.append(csv_row)
        md_rows.append(md_row)

    return csv_rows, md_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready comparison tables")
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=["grpo", "ppo", "dpo"])
    parser.add_argument(
        "--output-prefix",
        default="paper_results/grpo_vs_ppo_dpo",
        help="Prefix for output files without suffixes",
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for method in args.methods:
            rows.append(collect_seed_method_row(args.results_root, seed, method))

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    main_csv, main_md = build_main_table(rows, args.methods)
    pairwise_csv, pairwise_md = build_pairwise_grpo_table(rows)
    seed_csv, seed_md = build_seed_table(rows)

    write_csv(main_csv, f"{args.output_prefix}_main.csv")
    write_markdown(
        [
            "Method",
            "Base Reward",
            "Candidate Reward",
            "Delta Reward",
            "Delta %",
            "Train Min",
            "Gain / Min",
            "Reward Std",
            "Resp Len",
        ],
        main_md,
        f"{args.output_prefix}_main.md",
    )

    write_csv(pairwise_csv, f"{args.output_prefix}_pairwise.csv")
    write_markdown(
        [
            "Comparison",
            "GRPO Wins",
            "Reward Gap",
            "Delta Reward Gap",
            "Train Min Delta",
            "Gain / Min Gap",
        ],
        pairwise_md,
        f"{args.output_prefix}_pairwise.md",
    )

    write_csv(seed_csv, f"{args.output_prefix}_per_seed.csv")
    write_markdown(
        [
            "Seed",
            "Winner",
            "GRPO Reward",
            "GRPO Delta",
            "GRPO Min",
            "PPO Reward",
            "PPO Delta",
            "PPO Min",
            "DPO Reward",
            "DPO Delta",
            "DPO Min",
        ],
        seed_md,
        f"{args.output_prefix}_per_seed.md",
    )

    with open(f"{args.output_prefix}_raw.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(
        "Wrote paper tables to "
        f"{args.output_prefix}_main.md, "
        f"{args.output_prefix}_pairwise.md, and "
        f"{args.output_prefix}_per_seed.md"
    )


if __name__ == "__main__":
    main()
