#!/usr/bin/env python3
"""
Script to run DPO training for Assignment 7.
This script loads preference pairs and optimizes a causal LM with DPO.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import AssignmentConfig, get_config, load_config_from_file
from dpo_trainer import DPOTrainer, create_dpo_trainer
from policy_io import load_policy_model_and_tokenizer, save_policy_model_and_tokenizer
from reward_model import load_reward_model
from rlhf_trainer import evaluate_policy
from utils import (
    MetricsTracker,
    compare_models_side_by_side,
    create_batch_iterator,
    create_sample_data_files,
    create_summary_report,
    load_json_data,
    plot_reward_distribution,
    plot_training_curves,
    save_model_outputs,
    set_seed,
    setup_logging,
    split_data,
)

logger = logging.getLogger(__name__)


def is_skippable_dpo_batch_error(exc: ValueError) -> bool:
    """Return True when a DPO batch can be skipped safely."""
    return "no usable examples after tokenization" in str(exc).lower()


def get_dpo_batch_size(batch, fallback_examples: List[Dict[str, str]]) -> int:
    """Infer the effective batch size from a real or mocked DPO batch."""
    if hasattr(batch, "chosen_input_ids"):
        return int(batch.chosen_input_ids.shape[0])
    if hasattr(batch, "size"):
        return int(batch.size)
    return len(fallback_examples)


def get_dpo_artifact_paths(config: AssignmentConfig, epoch: Optional[int] = None) -> Dict[str, str]:
    """Return canonical output paths for the DPO harness."""
    model_dir = config.system.dpo_model_dir
    paths = {
        "best_model": os.path.join(model_dir, "best_dpo_model"),
        "final_model": os.path.join(model_dir, "final_dpo_model"),
        "metrics": os.path.join(config.system.logs_dir, "dpo_training_metrics.json"),
        "training_curve": os.path.join("plots", "dpo_training_curves.png"),
        "reward_distribution": os.path.join("plots", "dpo_reward_distribution.png"),
        "baseline_outputs": os.path.join(config.system.logs_dir, "baseline_outputs_dpo.json"),
        "candidate_outputs": os.path.join(config.system.logs_dir, "dpo_outputs.json"),
        "comparison": os.path.join(config.system.logs_dir, "dpo_model_comparison.json"),
        "summary": os.path.join(config.system.logs_dir, "dpo_training_summary.json"),
    }
    if epoch is not None:
        paths["checkpoint"] = os.path.join(model_dir, f"dpo_checkpoint_epoch_{epoch}.pt")
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the DPO harness."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the base language model")
    parser.add_argument(
        "--model_loader",
        type=str,
        default=None,
        help="Optional module:function loader for external policy/tokenizer backends such as MiniTorch",
    )
    parser.add_argument(
        "--model_saver",
        type=str,
        default=None,
        help="Optional module:function saver for external policy/tokenizer backends such as MiniTorch",
    )
    parser.add_argument("--reward_model_path", type=str, default=None, help="Path to trained reward model")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for DPO training")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of DPO training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Preference-pair batch size")
    parser.add_argument("--beta", type=float, default=None, help="DPO beta parameter")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=None,
        help="Label smoothing for the DPO loss",
    )
    parser.add_argument(
        "--max_train_pairs",
        type=int,
        default=None,
        help="Limit the number of training preference pairs (0 uses all pairs)",
    )
    parser.add_argument(
        "--max_val_pairs",
        type=int,
        default=None,
        help="Limit the number of validation preference pairs (0 uses all pairs)",
    )
    parser.add_argument(
        "--max_test_pairs",
        type=int,
        default=None,
        help="Limit the number of test preference pairs (0 uses all pairs)",
    )
    parser.add_argument(
        "--max_eval_prompts",
        type=int,
        default=None,
        help="Limit the number of reward-eval prompts used during DPO",
    )
    return parser


def apply_cli_overrides(config: AssignmentConfig, args: argparse.Namespace) -> AssignmentConfig:
    """Apply CLI overrides to a config object."""
    if args.model_name is not None:
        config.model.model_name = args.model_name
    if args.model_loader is not None:
        config.model.policy_loader = args.model_loader
    if args.model_saver is not None:
        config.model.policy_saver = args.model_saver
    if args.learning_rate is not None:
        config.training.dpo_learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.training.dpo_num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.data.train_batch_size = args.batch_size
        config.data.eval_batch_size = args.batch_size
    if args.beta is not None:
        config.dpo.beta = args.beta
    if args.label_smoothing is not None:
        config.dpo.label_smoothing = args.label_smoothing
    if args.max_train_pairs is not None:
        config.experiment.max_train_pairs = args.max_train_pairs
    if args.max_val_pairs is not None:
        config.experiment.max_val_pairs = args.max_val_pairs
    if args.max_test_pairs is not None:
        config.experiment.max_test_pairs = args.max_test_pairs
    if args.max_eval_prompts is not None:
        config.experiment.eval_prompts_sample_size = args.max_eval_prompts
    if args.reward_model_path:
        config.system.reward_model_dir = os.path.dirname(args.reward_model_path)
    return config

def get_dpo_generation_config(config: AssignmentConfig) -> Dict[str, float]:
    """Return generation settings used for reward-model evaluation."""
    return {
        "max_length": config.experiment.eval_generation_max_length,
        "temperature": config.training.generation_temperature,
        "top_p": config.training.generation_top_p,
    }


def maybe_limit_examples(data: List[Dict[str, str]], max_examples: int) -> List[Dict[str, str]]:
    """Limit a dataset when a positive cap is provided."""
    if max_examples and max_examples > 0:
        return data[:max_examples]
    return data


def load_preference_splits(config: AssignmentConfig) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load train/val/test preference splits, falling back to a deterministic split when needed."""
    if not os.path.exists(config.data.preference_data_path):
        logger.info("Preference data not found, creating sample data...")
        create_sample_data_files("data")

    preference_data = load_json_data(config.data.preference_data_path)
    if (
        os.path.exists(config.data.preference_val_data_path)
        and os.path.exists(config.data.preference_test_data_path)
    ):
        logger.info("Using explicit preference train/val/test splits from disk")
        train_data = preference_data
        val_data = load_json_data(config.data.preference_val_data_path)
        test_data = load_json_data(config.data.preference_test_data_path)
    else:
        logger.info(
            "Validation/test preference splits not found; splitting preference_data.json with configured ratios"
        )
        train_data, val_data, test_data = split_data(
            preference_data,
            config.data.train_split_ratio,
            config.data.validation_split_ratio,
            config.data.test_split_ratio,
            shuffle=True,
        )

    train_data = maybe_limit_examples(train_data, config.experiment.max_train_pairs)
    val_data = maybe_limit_examples(val_data, config.experiment.max_val_pairs)
    test_data = maybe_limit_examples(test_data, config.experiment.max_test_pairs)

    if not train_data:
        raise ValueError("No DPO training pairs available")
    if not val_data:
        raise ValueError("No DPO validation pairs available")
    if not test_data:
        raise ValueError("No DPO test pairs available")

    return train_data, val_data, test_data


def evaluate_preference_dataset(
    trainer: DPOTrainer,
    preference_data: List[Dict[str, str]],
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate DPO loss/accuracy over a preference dataset."""
    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "margin": 0.0,
        "chosen_logp": 0.0,
        "rejected_logp": 0.0,
    }
    total_examples = 0

    for batch_examples in create_batch_iterator(preference_data, batch_size, shuffle=False):
        try:
            batch = trainer.prepare_batch(batch_examples)
        except ValueError as exc:
            if is_skippable_dpo_batch_error(exc):
                logger.warning("Skipping DPO eval batch with no usable examples after tokenization")
                continue
            raise

        metrics = trainer.evaluate_step(batch)
        batch_size_actual = get_dpo_batch_size(batch, batch_examples)
        total_examples += batch_size_actual

        totals["loss"] += metrics.loss * batch_size_actual
        totals["accuracy"] += metrics.accuracy * batch_size_actual
        totals["margin"] += metrics.margin * batch_size_actual
        totals["chosen_logp"] += metrics.chosen_logp * batch_size_actual
        totals["rejected_logp"] += metrics.rejected_logp * batch_size_actual

    if total_examples == 0:
        raise ValueError("No usable DPO evaluation examples remained after tokenization")

    return {
        key: value / total_examples
        for key, value in totals.items()
    }


def run_dpo_training(config: AssignmentConfig):
    """Main function to run DPO training."""
    logger = setup_logging(config.system.logs_dir, "dpo_training")
    set_seed(config.experiment.seed)

    device = torch.device(config.system.device)
    logger.info(f"Using device: {device}")

    artifact_paths = get_dpo_artifact_paths(config)
    os.makedirs(config.system.dpo_model_dir, exist_ok=True)
    os.makedirs(config.system.logs_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    logger.info("Loading preference data...")
    train_preferences, val_preferences, test_preferences = load_preference_splits(config)
    logger.info(f"Loaded {len(train_preferences)} training preference pairs")
    logger.info(f"Loaded {len(val_preferences)} validation preference pairs")
    logger.info(f"Loaded {len(test_preferences)} test preference pairs")

    if not os.path.exists(config.data.eval_prompts_path):
        logger.info("Evaluation prompts not found, creating sample data...")
        create_sample_data_files("data")

    eval_prompts_data = load_json_data(config.data.eval_prompts_path)
    eval_prompts = [item["prompt"] for item in eval_prompts_data]
    eval_prompts = maybe_limit_examples(
        [{"prompt": prompt} for prompt in eval_prompts],
        config.experiment.eval_prompts_sample_size,
    )
    eval_prompts = [item["prompt"] for item in eval_prompts]
    logger.info(f"Loaded {len(eval_prompts)} evaluation prompts")

    reward_model_path = os.path.join(config.system.reward_model_dir, "best_reward_model.pt")
    if not os.path.exists(reward_model_path):
        reward_model_path = os.path.join(config.system.reward_model_dir, "final_reward_model.pt")
    if not os.path.exists(reward_model_path):
        logger.error(f"Could not find trained reward model at {reward_model_path}")
        logger.error("Please run: python scripts/train_reward_model.py first")
        return

    logger.info(f"Loading reward model from {reward_model_path}")
    reward_model = load_reward_model(reward_model_path, device)

    logger.info(f"Loading policy model: {config.model.model_name}")
    policy_model, tokenizer = load_policy_model_and_tokenizer(
        config.model.model_name,
        config=config,
        device=device,
    )
    trainer = create_dpo_trainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    metrics_tracker = MetricsTracker()
    generation_config = get_dpo_generation_config(config)

    logger.info("Evaluating base model...")
    base_eval_metrics = evaluate_policy(
        trainer=trainer,
        eval_prompts=eval_prompts,
        num_samples=3,
        reward_model=reward_model,
        generation_config=generation_config,
    )

    logger.info("Base model evaluation results:")
    for key, value in base_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("Generating baseline samples...")
    baseline_responses, _, _, _, _ = trainer.policy.generate(
        prompts=eval_prompts[:5],
        max_length=generation_config["max_length"],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        return_log_probs=False,
    )
    baseline_full_texts = [
        f"{prompt} {response}" for prompt, response in zip(eval_prompts[:5], baseline_responses)
    ]
    baseline_rewards = reward_model.get_rewards(
        baseline_full_texts,
        reward_model.tokenizer,
        device,
    )

    logger.info("Starting DPO training...")
    start_time = time.time()

    best_reward = float("-inf")
    total_steps = 0

    for epoch in range(config.training.dpo_num_epochs):
        logger.info(f"DPO Epoch {epoch + 1}/{config.training.dpo_num_epochs}")
        epoch_totals = {
            "loss": 0.0,
            "accuracy": 0.0,
            "margin": 0.0,
            "chosen_logp": 0.0,
            "rejected_logp": 0.0,
        }
        seen_examples = 0

        train_batches = create_batch_iterator(
            train_preferences,
            config.data.train_batch_size,
            shuffle=True,
        )
        num_batches = (len(train_preferences) + config.data.train_batch_size - 1) // config.data.train_batch_size
        epoch_progress = tqdm(
            train_batches,
            total=num_batches,
            desc=f"Epoch {epoch + 1}",
            disable=os.environ.get("TQDM_DISABLE") == "1",
        )

        for batch_examples in epoch_progress:
            try:
                batch = trainer.prepare_batch(batch_examples)
            except ValueError as exc:
                if is_skippable_dpo_batch_error(exc):
                    logger.warning(
                        "Skipping DPO train batch with no usable examples after tokenization"
                    )
                    continue
                raise

            step_metrics = trainer.train_step(batch)
            batch_size_actual = get_dpo_batch_size(batch, batch_examples)
            total_steps += 1
            seen_examples += batch_size_actual

            epoch_totals["loss"] += step_metrics.loss * batch_size_actual
            epoch_totals["accuracy"] += step_metrics.accuracy * batch_size_actual
            epoch_totals["margin"] += step_metrics.margin * batch_size_actual
            epoch_totals["chosen_logp"] += step_metrics.chosen_logp * batch_size_actual
            epoch_totals["rejected_logp"] += step_metrics.rejected_logp * batch_size_actual

            epoch_progress.set_postfix(
                {
                    "loss": f"{step_metrics.loss:.4f}",
                    "acc": f"{step_metrics.accuracy:.3f}",
                    "margin": f"{step_metrics.margin:.3f}",
                }
            )

            if total_steps % config.system.logging_steps == 0:
                metrics_tracker.update(
                    {
                        "train_loss": step_metrics.loss,
                        "train_accuracy": step_metrics.accuracy,
                        "train_margin": step_metrics.margin,
                        "train_chosen_logp": step_metrics.chosen_logp,
                        "train_rejected_logp": step_metrics.rejected_logp,
                    },
                    step=total_steps,
                )

        if seen_examples == 0:
            raise ValueError("No usable DPO training examples remained after tokenization")

        epoch_metrics = {"epoch": epoch}
        for key, value in epoch_totals.items():
            epoch_metrics[f"train_{key}"] = value / seen_examples

        val_metrics = evaluate_preference_dataset(
            trainer=trainer,
            preference_data=val_preferences,
            batch_size=config.data.eval_batch_size,
        )
        for key, value in val_metrics.items():
            epoch_metrics[f"val_{key}"] = value

        reward_eval_metrics = evaluate_policy(
            trainer=trainer,
            eval_prompts=eval_prompts,
            num_samples=3,
            reward_model=reward_model,
            generation_config=generation_config,
        )
        for key, value in reward_eval_metrics.items():
            epoch_metrics[f"eval_{key}"] = value

        current_reward = reward_eval_metrics["mean_reward"]
        if current_reward > best_reward:
            best_reward = current_reward
            save_policy_model_and_tokenizer(
                trainer.policy.model,
                trainer.tokenizer,
                artifact_paths["best_model"],
                config=config,
            )
            logger.info(f"Saved new best DPO model with reward: {current_reward:.4f}")

        trainer.save_checkpoint(
            checkpoint_path=get_dpo_artifact_paths(config, epoch + 1)["checkpoint"],
            epoch=epoch,
            metrics=epoch_metrics,
        )
        metrics_tracker.update(epoch_metrics, step=epoch + 1)

    training_time = time.time() - start_time
    logger.info(f"DPO training completed in {training_time:.2f} seconds")

    final_preference_metrics = evaluate_preference_dataset(
        trainer=trainer,
        preference_data=test_preferences,
        batch_size=config.data.eval_batch_size,
    )
    metrics_tracker.update(
        {f"test_{key}": value for key, value in final_preference_metrics.items()},
        step=config.training.dpo_num_epochs + 1,
    )

    final_eval_metrics = evaluate_policy(
        trainer=trainer,
        eval_prompts=eval_prompts,
        num_samples=5,
        reward_model=reward_model,
        generation_config=generation_config,
    )

    logger.info("Final reward-model evaluation results:")
    for key, value in final_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("Final DPO preference metrics:")
    for key, value in final_preference_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    final_responses, _, _, _, _ = trainer.policy.generate(
        prompts=eval_prompts[:5],
        max_length=generation_config["max_length"],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        return_log_probs=False,
    )
    final_full_texts = [
        f"{prompt} {response}" for prompt, response in zip(eval_prompts[:5], final_responses)
    ]
    final_rewards = reward_model.get_rewards(
        final_full_texts,
        reward_model.tokenizer,
        device,
    )

    save_policy_model_and_tokenizer(
        trainer.policy.model,
        trainer.tokenizer,
        artifact_paths["final_model"],
        config=config,
    )

    metrics_tracker.save_metrics(artifact_paths["metrics"])
    plot_training_curves(
        metrics_tracker.metrics_history,
        artifact_paths["training_curve"],
        title="DPO Training Progress",
    )

    reward_values = [
        metrics["eval_mean_reward"]
        for metrics in metrics_tracker.metrics_history
        if "eval_mean_reward" in metrics
    ]
    if reward_values:
        plot_reward_distribution(
            reward_values,
            artifact_paths["reward_distribution"],
            title="DPO Reward Distribution During Training",
        )

    save_model_outputs(
        prompts=eval_prompts[:5],
        responses=baseline_responses,
        rewards=baseline_rewards,
        save_path=artifact_paths["baseline_outputs"],
        model_name="baseline",
    )
    save_model_outputs(
        prompts=eval_prompts[:5],
        responses=final_responses,
        rewards=final_rewards,
        save_path=artifact_paths["candidate_outputs"],
        model_name="dpo",
    )

    compare_models_side_by_side(
        prompts=eval_prompts[:5],
        base_responses=baseline_responses,
        rlhf_responses=final_responses,
        base_rewards=baseline_rewards,
        rlhf_rewards=final_rewards,
        save_path=artifact_paths["comparison"],
        candidate_label="dpo",
    )
    create_summary_report(
        base_metrics=base_eval_metrics,
        rlhf_metrics=final_eval_metrics,
        training_time=training_time,
        save_path=artifact_paths["summary"],
        candidate_label="dpo",
    )

    logger.info("DPO training completed successfully!")
    logger.info(f"Base model mean reward: {base_eval_metrics['mean_reward']:.4f}")
    logger.info(f"DPO model mean reward: {final_eval_metrics['mean_reward']:.4f}")
    logger.info(
        f"Improvement: {final_eval_metrics['mean_reward'] - base_eval_metrics['mean_reward']:.4f}"
    )
    logger.info(f"Models saved to: {config.system.dpo_model_dir}")
    logger.info(f"Logs and plots saved to: {config.system.logs_dir} and plots/")


def main(argv: Optional[List[str]] = None):
    """CLI entrypoint for DPO training."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_config()

    config = apply_cli_overrides(config, args)
    config.validate()
    run_dpo_training(config)


if __name__ == "__main__":
    main()
