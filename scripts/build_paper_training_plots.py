#!/usr/bin/env python3
"""
Build aggregate paper-style training plots for GRPO, PPO, and DPO.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "GRPO": "#1b9e77",
    "PPO": "#d95f02",
    "DPO": "#7570b3",
}


def safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def load_method_metrics(results_root: str, seed: int, method: str) -> List[Dict]:
    path = os.path.join(results_root, f"seed_{seed}", "logs", f"{method}_training_metrics.json")
    return load_json(path)


def split_metric_rows(method: str, rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if method in {"grpo", "ppo"}:
        train_rows = [row for row in rows if "epoch" not in row and "policy_loss" in row]
        eval_rows = [row for row in rows if "epoch" in row and "eval_mean_reward" in row]
        return train_rows, eval_rows, []

    train_rows = [
        row
        for row in rows
        if "train_loss" in row and "epoch" not in row and "test_loss" not in row
    ]
    eval_rows = [row for row in rows if "epoch" in row]
    test_rows = [row for row in rows if "test_loss" in row]
    return train_rows, eval_rows, test_rows


def step_xy(rows: List[Dict], metric: str) -> Tuple[List[int], List[float]]:
    usable = [row for row in rows if metric in row]
    usable.sort(key=lambda row: row["step"])
    return [int(row["step"]) for row in usable], [float(row[metric]) for row in usable]


def epoch_xy(rows: List[Dict], metric: str) -> Tuple[List[int], List[float]]:
    usable = [row for row in rows if metric in row and "epoch" in row]
    usable.sort(key=lambda row: row["epoch"])
    return [int(row["epoch"]) + 1 for row in usable], [float(row[metric]) for row in usable]


def aggregate_series(seed_series: Dict[int, Tuple[List[int], List[float]]]) -> Tuple[List[int], List[float], List[float]]:
    by_x: Dict[int, List[float]] = defaultdict(list)
    for _seed, (xs, ys) in seed_series.items():
        for x, y in zip(xs, ys):
            by_x[int(x)].append(float(y))

    xs = sorted(by_x)
    mean_ys = [mean(by_x[x]) for x in xs]
    std_ys = [safe_stdev(by_x[x]) for x in xs]
    return xs, mean_ys, std_ys


def plot_seed_and_mean(
    ax,
    seed_series: Dict[int, Tuple[List[int], List[float]]],
    color: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    for seed, (xs, ys) in sorted(seed_series.items()):
        if not xs:
            continue
        if len(xs) == 1:
            ax.scatter(xs, ys, color=color, alpha=0.28, s=24)
        else:
            ax.plot(xs, ys, color=color, alpha=0.22, linewidth=1.2)

    xs, mean_ys, std_ys = aggregate_series(seed_series)
    if xs:
        if len(xs) == 1:
            ax.errorbar(xs, mean_ys, yerr=std_ys, fmt="o", color=color, capsize=4, markersize=7)
        else:
            ax.plot(xs, mean_ys, color=color, linewidth=2.6, marker="o", markersize=4)
            lower = np.array(mean_ys) - np.array(std_ys)
            upper = np.array(mean_ys) + np.array(std_ys)
            ax.fill_between(xs, lower, upper, color=color, alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def load_main_table_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_method_plot(
    results_root: str,
    seeds: List[int],
    method: str,
    plots_dir: str,
) -> str:
    method_upper = method.upper()
    color = METHOD_COLORS[method_upper]
    method_rows = {seed: split_metric_rows(method, load_method_metrics(results_root, seed, method)) for seed in seeds}

    if method == "grpo":
        spec = [
            ("train", "reward_mean", "Reward Model Score", "Step", "Reward"),
            ("train", "entropy", "Token Entropy", "Step", "Entropy"),
            ("train", "kl_divergence", "KL To Reference", "Step", "KL"),
            ("train", "policy_loss", "Policy Loss", "Step", "Loss"),
            ("train", "advantage_std", "Advantage Std", "Step", "Std"),
            ("eval", "eval_mean_reward", "Held-Out Eval Reward", "Epoch", "Reward"),
        ]
    elif method == "ppo":
        spec = [
            ("train", "reward_mean", "Reward Model Score", "Step", "Reward"),
            ("train", "entropy", "Token Entropy", "Step", "Entropy"),
            ("train", "kl_divergence", "KL To Reference", "Step", "KL"),
            ("train", "policy_loss", "Policy Loss", "Step", "Loss"),
            ("train", "value_loss", "Value Loss", "Step", "Loss"),
            ("eval", "eval_mean_reward", "Held-Out Eval Reward", "Epoch", "Reward"),
        ]
    else:
        spec = [
            ("train", "train_loss", "DPO Loss", "Step", "Loss"),
            ("train", "train_accuracy", "Train Accuracy", "Step", "Accuracy"),
            ("train", "train_margin", "Train Margin", "Step", "Margin"),
            ("eval", "val_loss", "Validation Loss", "Epoch", "Loss"),
            ("eval", "val_accuracy", "Validation Accuracy", "Epoch", "Accuracy"),
            ("eval", "eval_mean_reward", "Held-Out Eval Reward", "Epoch", "Reward"),
        ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{method_upper} Training Dynamics Across Seeds", fontsize=16)

    for ax, (row_type, metric, title, xlabel, ylabel) in zip(axes.flat, spec):
        seed_series = {}
        for seed in seeds:
            train_rows, eval_rows, _test_rows = method_rows[seed]
            rows = train_rows if row_type == "train" else eval_rows
            xs, ys = (step_xy(rows, metric) if row_type == "train" else epoch_xy(rows, metric))
            seed_series[seed] = (xs, ys)
        plot_seed_and_mean(ax, seed_series, color, title, xlabel, ylabel)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path = os.path.join(plots_dir, f"{method}_aggregate_training.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_eval_reward_by_epoch_plot(results_root: str, seeds: List[int], plots_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("grpo", "ppo", "dpo"):
        seed_series = {}
        for seed in seeds:
            _train_rows, eval_rows, _test_rows = split_metric_rows(
                method, load_method_metrics(results_root, seed, method)
            )
            seed_series[seed] = epoch_xy(eval_rows, "eval_mean_reward")
            xs, ys = seed_series[seed]
            if xs:
                if len(xs) == 1:
                    ax.scatter(xs, ys, color=METHOD_COLORS[method.upper()], alpha=0.22, s=24)
                else:
                    ax.plot(xs, ys, color=METHOD_COLORS[method.upper()], alpha=0.18, linewidth=1.0)
        xs, mean_ys, std_ys = aggregate_series(seed_series)
        if xs:
            if len(xs) == 1:
                ax.errorbar(xs, mean_ys, yerr=std_ys, fmt="o", color=METHOD_COLORS[method.upper()], capsize=4, markersize=7, label=method.upper())
            else:
                ax.plot(xs, mean_ys, color=METHOD_COLORS[method.upper()], linewidth=2.5, marker="o", markersize=4, label=method.upper())
                ax.fill_between(xs, np.array(mean_ys) - np.array(std_ys), np.array(mean_ys) + np.array(std_ys), color=METHOD_COLORS[method.upper()], alpha=0.15)

    ax.set_title("Held-Out Eval Reward By Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = os.path.join(plots_dir, "eval_reward_by_epoch.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_summary_bar_plot(main_table_csv: str, plots_dir: str) -> str:
    rows = load_main_table_csv(main_table_csv)
    methods = [row["method"] for row in rows]

    def column_values(mean_key: str, std_key: str) -> Tuple[List[float], List[float]]:
        return ([float(row[mean_key]) for row in rows], [float(row[std_key]) for row in rows])

    reward_means, reward_stds = column_values("candidate_reward_mean", "candidate_reward_std")
    delta_means, delta_stds = column_values("reward_gain_mean", "reward_gain_std")
    eff_means, eff_stds = column_values("gain_per_train_minute_mean", "gain_per_train_minute_std")
    var_means, var_stds = column_values("candidate_std_reward_mean", "candidate_std_reward_std")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        (reward_means, reward_stds, "Final Reward", "Reward"),
        (delta_means, delta_stds, "Reward Gain Over Base", "Delta Reward"),
        (eff_means, eff_stds, "Reward Gain Per Train Minute", "Gain / Min"),
        (var_means, var_stds, "Final Reward Std", "Std"),
    ]

    for ax, (means, stds, title, ylabel) in zip(axes.flat, metrics):
        colors = [METHOD_COLORS[m] for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.88, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Aggregate Method Comparison Across Seeds", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path = os.path.join(plots_dir, "method_summary_bars.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper training plots")
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--main-table-csv",
        default=None,
        help="Path to the aggregate main CSV from build_paper_tables.py",
    )
    args = parser.parse_args()

    results_root = args.results_root
    plots_dir = os.path.join(results_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    output_paths = []
    for method in ("grpo", "ppo", "dpo"):
        output_paths.append(build_method_plot(results_root, args.seeds, method, plots_dir))

    output_paths.append(build_eval_reward_by_epoch_plot(results_root, args.seeds, plots_dir))

    main_table_csv = args.main_table_csv or os.path.join(results_root, "grpo_vs_ppo_dpo_main.csv")
    if os.path.exists(main_table_csv):
        output_paths.append(build_summary_bar_plot(main_table_csv, plots_dir))

    print("Wrote plots:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
