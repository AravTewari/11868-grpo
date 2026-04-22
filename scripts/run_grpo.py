#!/usr/bin/env python3
"""
Script to run GRPO training for Assignment 7.
This script loads a trained reward model and optimizes a causal LM with GRPO.
"""

import os
import sys
import argparse
import torch
import time
import logging
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import AssignmentConfig, get_config, load_config_from_file
from policy_io import load_policy_model_and_tokenizer, save_policy_model_and_tokenizer
from reward_model import load_reward_model
from grpo_trainer import create_grpo_trainer
from rlhf_trainer import evaluate_policy
from utils import (
    load_json_data,
    setup_logging,
    set_seed,
    MetricsTracker,
    plot_training_curves,
    plot_reward_distribution,
    compare_models_side_by_side,
    create_summary_report,
    save_model_outputs,
    create_sample_data_files,
)

logger = logging.getLogger(__name__)


def run_grpo_training(config: AssignmentConfig):
    """Main function to run GRPO training."""
    logger = setup_logging(config.system.logs_dir, "grpo_training")
    set_seed(config.experiment.seed)

    device = torch.device(config.system.device)
    logger.info(f"Using device: {device}")

    os.makedirs(config.system.grpo_model_dir, exist_ok=True)
    os.makedirs(config.system.logs_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    if not os.path.exists(config.data.train_prompts_path):
        logger.info("Training data not found, creating sample data...")
        create_sample_data_files("data")

    logger.info("Loading training data...")
    try:
        train_prompts_data = load_json_data(config.data.train_prompts_path)
        eval_prompts_data = load_json_data(config.data.eval_prompts_path)
    except FileNotFoundError as e:
        logger.error(f"Could not find data files: {e}")
        logger.error("Please run: python src/utils.py to create sample data")
        return

    train_prompts = [item["prompt"] for item in train_prompts_data]
    eval_prompts = [item["prompt"] for item in eval_prompts_data]

    max_train_prompts = getattr(config.experiment, "max_train_prompts", 0)
    if max_train_prompts and max_train_prompts > 0:
        train_prompts = train_prompts[:max_train_prompts]

    max_eval_prompts = getattr(config.experiment, "eval_prompts_sample_size", 0)
    if max_eval_prompts and max_eval_prompts > 0:
        eval_prompts = eval_prompts[:max_eval_prompts]

    logger.info(f"Loaded {len(train_prompts)} training prompts")
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

    trainer = create_grpo_trainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=config,
        device=device,
    )

    metrics_tracker = MetricsTracker()

    logger.info("Evaluating base model...")
    base_eval_metrics = evaluate_policy(
        trainer=trainer,
        eval_prompts=eval_prompts,
        num_samples=3,
    )

    logger.info("Base model evaluation results:")
    for key, value in base_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("Generating baseline samples...")
    baseline_responses, _, _, _, _ = trainer.policy.generate(
        prompts=eval_prompts[:5],
        max_length=config.grpo.rollout_max_length,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        return_log_probs=False,
    )

    baseline_full_texts = [
        f"{prompt} {response}" for prompt, response in zip(eval_prompts[:5], baseline_responses)
    ]
    baseline_rewards = reward_model.get_rewards(
        baseline_full_texts, reward_model.tokenizer, device
    )

    logger.info("Starting GRPO training...")
    start_time = time.time()

    best_reward = float("-inf")
    total_steps = 0

    for epoch in range(config.training.grpo_num_epochs):
        logger.info(f"GRPO Epoch {epoch + 1}/{config.training.grpo_num_epochs}")

        epoch_metrics = {
            "epoch": epoch,
            "policy_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
        }

        num_batches = len(train_prompts) // config.grpo.rollout_batch_size
        epoch_progress = tqdm(
            range(num_batches),
            desc=f"Epoch {epoch + 1}",
            disable=os.environ.get("TQDM_DISABLE") == "1",
        )

        for batch_idx in epoch_progress:
            start_idx = batch_idx * config.grpo.rollout_batch_size
            end_idx = min(start_idx + config.grpo.rollout_batch_size, len(train_prompts))
            batch_prompts = train_prompts[start_idx:end_idx]

            rollout_batch = trainer.generate_rollouts(batch_prompts)
            step_metrics = trainer.train_step(rollout_batch)

            for key in epoch_metrics:
                if key != "epoch" and hasattr(step_metrics, key):
                    epoch_metrics[key] += getattr(step_metrics, key)

            total_steps += 1

            epoch_progress.set_postfix(
                {
                    "reward": f"{step_metrics.reward_mean:.3f}",
                    "policy_loss": f"{step_metrics.policy_loss:.4f}",
                    "kl": f"{step_metrics.kl_divergence:.4f}",
                }
            )

            if total_steps % config.system.logging_steps == 0:
                step_metrics_dict = {
                    "step": total_steps,
                    "policy_loss": step_metrics.policy_loss,
                    "entropy": step_metrics.entropy,
                    "kl_divergence": step_metrics.kl_divergence,
                    "reward_mean": step_metrics.reward_mean,
                    "reward_std": step_metrics.reward_std,
                    "advantage_mean": step_metrics.advantage_mean,
                    "advantage_std": step_metrics.advantage_std,
                }
                metrics_tracker.update(step_metrics_dict, step=total_steps)

            if config.system.eval_steps > 0 and (batch_idx + 1) % config.system.eval_steps == 0:
                logger.info(f"Evaluating model after batch {batch_idx + 1}...")
                eval_metrics = evaluate_policy(
                    trainer=trainer,
                    eval_prompts=eval_prompts,
                    num_samples=3,
                )
                current_reward = eval_metrics["mean_reward"]
                if current_reward > best_reward:
                    best_reward = current_reward
                    best_model_path = os.path.join(
                        config.system.grpo_model_dir, "best_grpo_model"
                    )
                    save_policy_model_and_tokenizer(
                        trainer.policy.model,
                        trainer.tokenizer,
                        best_model_path,
                        config=config,
                    )

        for key in epoch_metrics:
            if key != "epoch" and num_batches > 0:
                epoch_metrics[key] /= num_batches

        logger.info(f"Evaluating model after epoch {epoch + 1}...")
        eval_metrics = evaluate_policy(
            trainer=trainer,
            eval_prompts=eval_prompts,
            num_samples=3,
        )
        for key, value in eval_metrics.items():
            epoch_metrics[f"eval_{key}"] = value

        current_reward = eval_metrics["mean_reward"]
        if current_reward > best_reward:
            best_reward = current_reward
            best_model_path = os.path.join(config.system.grpo_model_dir, "best_grpo_model")
            save_policy_model_and_tokenizer(
                trainer.policy.model,
                trainer.tokenizer,
                best_model_path,
                config=config,
            )
            logger.info(f"Saved new best GRPO model with reward: {current_reward:.4f}")

        checkpoint_path = os.path.join(
            config.system.grpo_model_dir, f"grpo_checkpoint_epoch_{epoch + 1}.pt"
        )
        trainer.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
        metrics_tracker.update(epoch_metrics, step=epoch + 1)

    training_time = time.time() - start_time
    logger.info(f"GRPO training completed in {training_time:.2f} seconds")

    final_eval_metrics = evaluate_policy(
        trainer=trainer,
        eval_prompts=eval_prompts,
        num_samples=5,
    )

    logger.info("Final evaluation results:")
    for key, value in final_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    final_responses, _, _, _, _ = trainer.policy.generate(
        prompts=eval_prompts[:5],
        max_length=config.grpo.rollout_max_length,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        return_log_probs=False,
    )
    final_full_texts = [
        f"{prompt} {response}" for prompt, response in zip(eval_prompts[:5], final_responses)
    ]
    final_rewards = reward_model.get_rewards(final_full_texts, reward_model.tokenizer, device)

    final_model_path = os.path.join(config.system.grpo_model_dir, "final_grpo_model")
    save_policy_model_and_tokenizer(
        trainer.policy.model,
        trainer.tokenizer,
        final_model_path,
        config=config,
    )

    metrics_tracker.save_metrics(os.path.join(config.system.logs_dir, "grpo_training_metrics.json"))

    plot_training_curves(
        metrics_tracker.metrics_history,
        os.path.join("plots", "grpo_training_curves.png"),
        title="GRPO Training Progress",
    )

    if metrics_tracker.metrics_history:
        reward_values = [
            metrics["reward_mean"]
            for metrics in metrics_tracker.metrics_history
            if "reward_mean" in metrics
        ]
        if reward_values:
            plot_reward_distribution(
                reward_values,
                os.path.join("plots", "grpo_reward_distribution.png"),
                title="GRPO Reward Distribution During Training",
            )

    save_model_outputs(
        prompts=eval_prompts[:5],
        responses=baseline_responses,
        rewards=baseline_rewards,
        save_path=os.path.join(config.system.logs_dir, "baseline_outputs_grpo.json"),
        model_name="baseline",
    )
    save_model_outputs(
        prompts=eval_prompts[:5],
        responses=final_responses,
        rewards=final_rewards,
        save_path=os.path.join(config.system.logs_dir, "grpo_outputs.json"),
        model_name="grpo",
    )

    compare_models_side_by_side(
        prompts=eval_prompts[:5],
        base_responses=baseline_responses,
        rlhf_responses=final_responses,
        base_rewards=baseline_rewards,
        rlhf_rewards=final_rewards,
        save_path=os.path.join(config.system.logs_dir, "grpo_model_comparison.json"),
        candidate_label="grpo",
    )

    create_summary_report(
        base_metrics=base_eval_metrics,
        rlhf_metrics=final_eval_metrics,
        training_time=training_time,
        save_path=os.path.join(config.system.logs_dir, "grpo_training_summary.json"),
        candidate_label="grpo",
    )


def main():
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Temporary Hugging Face model name used to load the policy for GRPO",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=None,
        help="Path to trained reward model",
    )
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for GRPO training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of GRPO training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Prompt batch size before group duplication",
    )
    parser.add_argument("--group_size", type=int, default=None, help="GRPO group size")
    parser.add_argument(
        "--update_epochs",
        type=int,
        default=None,
        help="Number of policy update epochs per rollout batch",
    )
    parser.add_argument(
        "--rollout_max_length",
        type=int,
        default=None,
        help="Maximum total rollout sequence length",
    )
    parser.add_argument(
        "--max_train_prompts",
        type=int,
        default=None,
        help="Limit the number of training prompts used for GRPO (0 uses all prompts)",
    )
    parser.add_argument(
        "--max_eval_prompts",
        type=int,
        default=None,
        help="Limit the number of evaluation prompts used during GRPO",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run intermediate evaluation every N batches",
    )

    args = parser.parse_args()

    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_config()

    if args.model_name is not None:
        config.model.model_name = args.model_name
    if args.model_loader is not None:
        config.model.policy_loader = args.model_loader
    if args.model_saver is not None:
        config.model.policy_saver = args.model_saver
    if args.learning_rate is not None:
        config.training.grpo_learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.training.grpo_num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.grpo.rollout_batch_size = args.batch_size
    if args.group_size is not None:
        config.grpo.group_size = args.group_size
    if args.update_epochs is not None:
        config.grpo.update_epochs = args.update_epochs
    if args.rollout_max_length is not None:
        config.grpo.rollout_max_length = args.rollout_max_length
    if args.reward_model_path:
        config.system.reward_model_dir = os.path.dirname(args.reward_model_path)
    if args.max_train_prompts is not None:
        config.experiment.max_train_prompts = args.max_train_prompts
    if args.max_eval_prompts is not None:
        config.experiment.eval_prompts_sample_size = args.max_eval_prompts
    if args.eval_steps is not None:
        config.system.eval_steps = args.eval_steps

    config.validate()
    run_grpo_training(config)


if __name__ == "__main__":
    main()
