#!/usr/bin/env python3
"""
Evaluation script for Assignment 7.
This script evaluates and compares a base model against a single candidate method.
"""

import os
import sys
import argparse
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import AssignmentConfig, get_config, load_config_from_file
from policy_io import has_custom_policy_loader, load_policy_model_and_tokenizer
from reward_model import load_reward_model
from utils import (
    load_json_data,
    setup_logging,
    set_seed,
    compute_text_statistics,
    plot_reward_distribution,
    compare_models_side_by_side,
    save_model_outputs,
    create_summary_report,
    create_sample_data_files,
)

logger = logging.getLogger(__name__)


def normalize_candidate_label(candidate_label: Optional[str]) -> str:
    """Convert a user-facing method label into a filesystem-safe key."""
    if not candidate_label:
        return "ppo"
    return candidate_label.strip().lower().replace(" ", "_")


def looks_like_model_dir(path: str) -> bool:
    """Return True when the path looks like a HF model directory."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def resolve_candidate_selection(
    config: AssignmentConfig,
    candidate_model: Optional[str],
    candidate_label: Optional[str],
    legacy_rlhf_model: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve the candidate model root and normalized label."""
    if candidate_model is None and legacy_rlhf_model is not None:
        candidate_model = legacy_rlhf_model

    candidate_key = normalize_candidate_label(candidate_label)
    if candidate_model is None:
        if candidate_key == "grpo":
            candidate_model = config.system.grpo_model_dir
        elif candidate_key == "dpo":
            candidate_model = config.system.dpo_model_dir
        else:
            candidate_model = config.system.ppo_model_dir

    return candidate_model, candidate_key


def resolve_candidate_model_path(
    candidate_root: str,
    candidate_label: str,
    allow_custom_loader_dirs: bool = False,
) -> Optional[str]:
    """Resolve a candidate root into a loadable checkpoint directory."""
    candidate_dirs = [
        os.path.join(candidate_root, f"best_{candidate_label}_model"),
        os.path.join(candidate_root, f"final_{candidate_label}_model"),
        os.path.join(candidate_root, "best_dpo_model"),
        os.path.join(candidate_root, "final_dpo_model"),
        os.path.join(candidate_root, "best_ppo_model"),
        os.path.join(candidate_root, "final_ppo_model"),
        os.path.join(candidate_root, "best_grpo_model"),
        os.path.join(candidate_root, "final_grpo_model"),
        os.path.join(candidate_root, "best_rlhf_model"),
        os.path.join(candidate_root, "final_rlhf_model"),
    ]

    for candidate_dir in candidate_dirs:
        if looks_like_model_dir(candidate_dir):
            return candidate_dir
        if allow_custom_loader_dirs and os.path.isdir(candidate_dir):
            return candidate_dir

    if looks_like_model_dir(candidate_root):
        return candidate_root
    if allow_custom_loader_dirs and os.path.isdir(candidate_root):
        return candidate_root

    return None


def get_evaluation_output_paths(candidate_label: str) -> Dict[str, str]:
    """Return method-specific output paths for evaluation artifacts."""
    candidate_key = normalize_candidate_label(candidate_label)
    return {
        "base_results": os.path.join("evaluation_results", f"base_model_results_{candidate_key}.json"),
        "detailed_comparison": os.path.join("evaluation_results", f"detailed_comparison_{candidate_key}.json"),
        "base_outputs": os.path.join("evaluation_results", f"base_model_outputs_{candidate_key}.json"),
        "candidate_outputs": os.path.join("evaluation_results", f"{candidate_key}_model_outputs.json"),
        "side_by_side": os.path.join("evaluation_results", f"side_by_side_comparison_{candidate_key}.json"),
        "summary": os.path.join("evaluation_results", f"evaluation_summary_{candidate_key}.json"),
        "base_reward_plot": os.path.join("plots", f"base_model_rewards_{candidate_key}.png"),
        "candidate_reward_plot": os.path.join("plots", f"{candidate_key}_model_rewards.png"),
        "reward_comparison_plot": os.path.join("plots", f"reward_comparison_{candidate_key}.png"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a base model against one candidate method")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Name of the base model")
    parser.add_argument("--candidate_model", type=str, default=None, help="Path to candidate model directory")
    parser.add_argument("--candidate_label", type=str, default=None, help="Label for the candidate method")
    parser.add_argument("--rlhf_model", type=str, default=None, help="Compatibility alias for --candidate_model")
    parser.add_argument(
        "--model_loader",
        type=str,
        default=None,
        help="Optional module:function loader for external policy/tokenizer backends such as MiniTorch",
    )
    parser.add_argument("--reward_model", type=str, default=None, help="Path to reward model")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate per prompt")
    parser.add_argument("--max_prompts", type=int, default=None, help="Maximum number of prompts to evaluate")
    return parser


def apply_cli_overrides(config: AssignmentConfig, args: argparse.Namespace) -> AssignmentConfig:
    """Apply CLI overrides to a config object."""
    config.model.model_name = args.base_model
    if args.model_loader is not None:
        config.model.policy_loader = args.model_loader
    if args.reward_model:
        config.system.reward_model_dir = os.path.dirname(args.reward_model)
    if args.max_prompts:
        config.experiment.eval_prompts_sample_size = args.max_prompts
    return config


class ModelEvaluator:
    """Evaluate language models with reward-model scoring."""

    def __init__(self, reward_model, tokenizer, device):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.reward_model.eval()

    def evaluate_model(
        self,
        model,
        model_tokenizer,
        prompts,
        num_samples_per_prompt=5,
        max_length=256,
        temperature=0.7,
        top_p=0.9,
    ):
        """Evaluate a model on a set of prompts."""
        model.eval()

        all_prompts = []
        all_responses = []
        all_rewards = []
        all_response_lengths = []

        logger.info(f"Evaluating model on {len(prompts)} prompts...")

        for prompt in tqdm(prompts, desc="Evaluating"):
            prompt_samples = [prompt] * num_samples_per_prompt
            encoded = model_tokenizer(
                prompt_samples,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length // 2,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=model_tokenizer.pad_token_id,
                    eos_token_id=model_tokenizer.eos_token_id,
                    num_return_sequences=1,
                )

            prompt_length = encoded["input_ids"].shape[1]
            response_ids = generated[:, prompt_length:]

            responses = model_tokenizer.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            full_texts = [f"{prompt} {response}" for response in responses]
            rewards = self.reward_model.get_rewards(full_texts, self.tokenizer, self.device)
            response_lengths = [len(response.split()) for response in responses]

            all_prompts.extend([prompt] * len(responses))
            all_responses.extend(responses)
            all_rewards.extend(rewards)
            all_response_lengths.extend(response_lengths)

        results = {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "median_reward": np.median(all_rewards),
            "min_reward": np.min(all_rewards),
            "max_reward": np.max(all_rewards),
            "mean_response_length": np.mean(all_response_lengths),
            "std_response_length": np.std(all_response_lengths),
            "num_prompts": len(prompts),
            "num_samples": len(all_responses),
            "samples_per_prompt": num_samples_per_prompt,
        }

        for percentile in [10, 25, 75, 90]:
            results[f"reward_p{percentile}"] = np.percentile(all_rewards, percentile)

        text_stats = compute_text_statistics(all_responses, self.tokenizer)
        results.update({f"response_{key}": value for key, value in text_stats.items()})

        return results, all_prompts, all_responses, all_rewards

    def compare_models(
        self,
        base_model,
        base_tokenizer,
        candidate_model,
        candidate_tokenizer,
        prompts,
        num_samples_per_prompt=3,
        candidate_label="candidate",
        **generation_kwargs,
    ):
        """Compare a base model and a candidate model side by side."""
        logger.info("Evaluating base model...")
        base_results, base_prompts, base_responses, base_rewards = self.evaluate_model(
            base_model,
            base_tokenizer,
            prompts,
            num_samples_per_prompt,
            **generation_kwargs,
        )

        logger.info(f"Evaluating {candidate_label.upper()} model...")
        candidate_results, candidate_prompts, candidate_responses, candidate_rewards = self.evaluate_model(
            candidate_model,
            candidate_tokenizer,
            prompts,
            num_samples_per_prompt,
            **generation_kwargs,
        )

        improvements = {}
        for key in base_results:
            if isinstance(base_results[key], (int, float)) and key in candidate_results:
                base_val = base_results[key]
                candidate_val = candidate_results[key]

                if base_val != 0:
                    improvement_pct = ((candidate_val - base_val) / abs(base_val)) * 100
                    improvements[f"{key}_improvement_pct"] = improvement_pct

                improvements[f"{key}_improvement_abs"] = candidate_val - base_val

        return {
            "base_results": base_results,
            "candidate_results": candidate_results,
            "improvements": improvements,
            "base_data": (base_prompts, base_responses, base_rewards),
            "candidate_data": (candidate_prompts, candidate_responses, candidate_rewards),
            "candidate_label": normalize_candidate_label(candidate_label),
        }


def evaluate_models(
    config: AssignmentConfig,
    candidate_model: str,
    candidate_label: str,
    num_samples: int = 3,
):
    """Main evaluation function."""
    logger = setup_logging(config.system.logs_dir, "model_evaluation")
    set_seed(config.experiment.seed)

    device = torch.device(config.system.device)
    logger.info(f"Using device: {device}")

    os.makedirs("plots", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)

    if not os.path.exists(config.data.eval_prompts_path):
        logger.info("Evaluation data not found, creating sample data...")
        create_sample_data_files("data")

    logger.info("Loading evaluation prompts...")
    try:
        eval_prompts_data = load_json_data(config.data.eval_prompts_path)
        eval_prompts = [item["prompt"] for item in eval_prompts_data]
    except FileNotFoundError:
        logger.error(f"Could not find evaluation prompts at {config.data.eval_prompts_path}")
        return

    if config.experiment.eval_prompts_sample_size > 0:
        eval_prompts = eval_prompts[: config.experiment.eval_prompts_sample_size]

    logger.info(f"Evaluating on {len(eval_prompts)} prompts")

    reward_model_path = os.path.join(config.system.reward_model_dir, "best_reward_model.pt")
    if not os.path.exists(reward_model_path):
        reward_model_path = os.path.join(config.system.reward_model_dir, "final_reward_model.pt")

    if not os.path.exists(reward_model_path):
        logger.error(f"Could not find reward model at {reward_model_path}")
        logger.error("Please train reward model first: python scripts/train_reward_model.py")
        return

    logger.info(f"Loading reward model from {reward_model_path}")
    reward_model = load_reward_model(reward_model_path, device)

    from transformers import AutoTokenizer

    reward_tokenizer = AutoTokenizer.from_pretrained(config.model.reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    evaluator = ModelEvaluator(reward_model, reward_tokenizer, device)

    logger.info(f"Loading base model: {config.model.model_name}")
    base_model, base_tokenizer = load_policy_model_and_tokenizer(
        config.model.model_name,
        config=config,
        device=device,
    )
    base_model.to(device)

    candidate_model_path = resolve_candidate_model_path(
        candidate_model,
        candidate_label,
        allow_custom_loader_dirs=has_custom_policy_loader(config),
    )
    output_paths = get_evaluation_output_paths(candidate_label)

    if candidate_model_path:
        logger.info(f"Loading {candidate_label.upper()} model from {candidate_model_path}")
        candidate_model_instance, candidate_tokenizer = load_policy_model_and_tokenizer(
            candidate_model_path,
            config=config,
            device=device,
        )
        candidate_model_instance.to(device)

        logger.info(f"Comparing base model vs {candidate_label.upper()} model...")
        comparison_results = evaluator.compare_models(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            candidate_model=candidate_model_instance,
            candidate_tokenizer=candidate_tokenizer,
            prompts=eval_prompts,
            num_samples_per_prompt=num_samples,
            candidate_label=candidate_label,
            max_length=config.experiment.eval_generation_max_length,
            temperature=0.7,
            top_p=0.9,
        )

        logger.info("Evaluation Results:")
        logger.info("=" * 50)

        logger.info("Base Model Results:")
        for key, value in comparison_results["base_results"].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info(f"\n{candidate_label.upper()} Model Results:")
        for key, value in comparison_results["candidate_results"].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info("\nImprovements:")
        for key, value in comparison_results["improvements"].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

        serializable_results = {}
        for key, value in comparison_results.items():
            if key in ["base_data", "candidate_data"]:
                continue
            if isinstance(value, dict):
                serializable_results[key] = {
                    inner_key: float(inner_value) if isinstance(inner_value, (np.float32, np.float64)) else inner_value
                    for inner_key, inner_value in value.items()
                }
            else:
                serializable_results[key] = value

        with open(output_paths["detailed_comparison"], "w") as f:
            json.dump(serializable_results, f, indent=2)

        base_prompts, base_responses, base_rewards = comparison_results["base_data"]
        candidate_prompts, candidate_responses, candidate_rewards = comparison_results["candidate_data"]

        plot_reward_distribution(
            base_rewards,
            output_paths["base_reward_plot"],
            title="Base Model Reward Distribution",
        )
        plot_reward_distribution(
            candidate_rewards,
            output_paths["candidate_reward_plot"],
            title=f"{candidate_label.upper()} Model Reward Distribution",
        )

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(base_rewards, bins=30, alpha=0.7, label="Base Model", color="blue")
        plt.hist(candidate_rewards, bins=30, alpha=0.7, label=f"{candidate_label.upper()} Model", color="red")
        plt.axvline(np.mean(base_rewards), color="blue", linestyle="--", alpha=0.8)
        plt.axvline(np.mean(candidate_rewards), color="red", linestyle="--", alpha=0.8)
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        improvement_per_prompt = np.array(candidate_rewards) - np.array(base_rewards)
        plt.hist(improvement_per_prompt, bins=30, alpha=0.7, color="green")
        plt.axvline(np.mean(improvement_per_prompt), color="red", linestyle="--")
        plt.xlabel("Reward Improvement")
        plt.ylabel("Frequency")
        plt.title(f"Per-Sample Reward Improvement ({candidate_label.upper()})")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_paths["reward_comparison_plot"], dpi=300, bbox_inches="tight")
        plt.close()

        save_model_outputs(
            prompts=base_prompts,
            responses=base_responses,
            rewards=base_rewards,
            save_path=output_paths["base_outputs"],
            model_name="base",
        )
        save_model_outputs(
            prompts=candidate_prompts,
            responses=candidate_responses,
            rewards=candidate_rewards,
            save_path=output_paths["candidate_outputs"],
            model_name=candidate_label,
        )

        sample_size = min(10, len(eval_prompts))
        compare_models_side_by_side(
            prompts=eval_prompts[:sample_size],
            base_responses=base_responses[: sample_size * num_samples : num_samples],
            rlhf_responses=candidate_responses[: sample_size * num_samples : num_samples],
            base_rewards=base_rewards[: sample_size * num_samples : num_samples],
            rlhf_rewards=candidate_rewards[: sample_size * num_samples : num_samples],
            save_path=output_paths["side_by_side"],
            num_examples=sample_size,
            candidate_label=candidate_label,
        )

        create_summary_report(
            base_metrics=comparison_results["base_results"],
            rlhf_metrics=comparison_results["candidate_results"],
            training_time=0,
            save_path=output_paths["summary"],
            candidate_label=candidate_label,
        )

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Number of prompts evaluated: {len(eval_prompts)}")
        logger.info(f"Samples per prompt: {num_samples}")
        logger.info(f"Total samples: {len(base_rewards)}")
        logger.info("")
        logger.info(f"Base Model Average Reward: {np.mean(base_rewards):.4f} ± {np.std(base_rewards):.4f}")
        logger.info(
            f"{candidate_label.upper()} Model Average Reward: {np.mean(candidate_rewards):.4f} ± {np.std(candidate_rewards):.4f}"
        )
        logger.info(f"Average Improvement: {np.mean(candidate_rewards) - np.mean(base_rewards):.4f}")
        logger.info(
            f"Improvement Percentage: {((np.mean(candidate_rewards) - np.mean(base_rewards)) / abs(np.mean(base_rewards))) * 100:.2f}%"
        )
        logger.info("")
        logger.info(f"Percentage of samples improved: {(improvement_per_prompt > 0).mean() * 100:.1f}%")
        logger.info(f"Max improvement: {np.max(improvement_per_prompt):.4f}")
        logger.info(f"Max degradation: {np.min(improvement_per_prompt):.4f}")
    else:
        logger.info(f"{candidate_label.upper()} model not found, evaluating base model only...")
        base_results, base_prompts, base_responses, base_rewards = evaluator.evaluate_model(
            base_model,
            base_tokenizer,
            eval_prompts,
            num_samples_per_prompt=max(1, num_samples),
            max_length=config.experiment.eval_generation_max_length,
            temperature=0.7,
            top_p=0.9,
        )

        for key, value in base_results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        with open(output_paths["base_results"], "w") as f:
            serializable_results = {
                key: float(val) if isinstance(val, (np.float32, np.float64)) else val
                for key, val in base_results.items()
            }
            json.dump(serializable_results, f, indent=2)

        plot_reward_distribution(
            base_rewards,
            output_paths["base_reward_plot"],
            title="Base Model Reward Distribution",
        )
        save_model_outputs(
            prompts=base_prompts,
            responses=base_responses,
            rewards=base_rewards,
            save_path=output_paths["base_outputs"],
            model_name="base",
        )
        logger.info(
            f"Evaluation completed. To compare with a candidate model, pass --candidate_model or train a {candidate_label.upper()} model first."
        )

    logger.info("\nEvaluation completed successfully!")
    logger.info("Results saved to: evaluation_results/")
    logger.info("Plots saved to: plots/")


def main(argv: Optional[List[str]] = None):
    """CLI entrypoint for evaluation."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_config()

    config = apply_cli_overrides(config, args)
    candidate_model, candidate_label = resolve_candidate_selection(
        config=config,
        candidate_model=args.candidate_model,
        candidate_label=args.candidate_label,
        legacy_rlhf_model=args.rlhf_model,
    )
    config.validate()
    evaluate_models(config, candidate_model=candidate_model, candidate_label=candidate_label, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
