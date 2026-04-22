#!/usr/bin/env python3
"""
Script to run GRPO training on GSM8K math problems.
Uses a rule-based reward function (answer correctness) instead of a learned reward model.

All outputs are saved under:
  ./results/<model_name>_g<group_size>_lr<lr>_kl<kl>_<timestamp>/
    config.json        -- full training configuration
    responses.jsonl    -- detailed per-prompt responses at each eval checkpoint
    summary.txt        -- human-readable training summary
    training_metrics.json
    training_curves.png
    reward_distribution.png
    checkpoints/       -- model checkpoints
"""

import os
import sys
import json
import argparse
import torch
import time
import logging
from datetime import datetime
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import AssignmentConfig, get_config, load_config_from_file
from gsm8k_reward import GSM8KRewardFunction
from grpo_trainer import create_grpo_trainer
from utils import (
    load_json_data,
    set_seed,
    MetricsTracker,
    plot_training_curves,
)

logger = logging.getLogger(__name__)


def make_run_dir(config: AssignmentConfig, results_root: str = "results") -> str:
    """Create a structured results directory for this run."""
    model_short = config.model.model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = f"{config.training.grpo_learning_rate:.0e}".replace("+", "")
    kl_str = f"{config.grpo.kl_penalty:.0e}".replace("+", "").replace("0e00", "0")

    run_name = (
        f"{model_short}"
        f"_g{config.grpo.group_size}"
        f"_ep{config.training.grpo_num_epochs}"
        f"_lr{lr_str}"
        f"_kl{kl_str}"
        f"_len{config.grpo.rollout_max_length}"
        f"_{timestamp}"
    )
    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def save_config_json(config: AssignmentConfig, run_dir: str) -> None:
    """Save full config as JSON to the run directory."""
    path = os.path.join(run_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)


def append_responses_jsonl(
    run_dir: str,
    stage: str,
    prompts,
    responses,
    rewards,
    ground_truths=None,
):
    """Append detailed response records to responses.jsonl."""
    path = os.path.join(run_dir, "responses.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            record = {
                "stage": stage,
                "prompt": prompt,
                "response": response,
                "reward": rewards[i] if i < len(rewards) else None,
            }
            if ground_truths and i < len(ground_truths):
                record["ground_truth"] = ground_truths[i]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_txt(
    run_dir: str,
    config: AssignmentConfig,
    base_metrics: dict,
    final_metrics: dict,
    training_time: float,
    best_accuracy: float,
    metrics_history: list,
):
    """Write a human-readable summary.txt."""
    path = os.path.join(run_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  GSM8K GRPO Training Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model:             {config.model.model_name}\n")
        f.write(f"  Group size:        {config.grpo.group_size}\n")
        f.write(f"  Batch size:        {config.grpo.rollout_batch_size}\n")
        f.write(f"  Max length:        {config.grpo.rollout_max_length}\n")
        f.write(f"  Temperature:       {config.grpo.temperature}\n")
        f.write(f"  Learning rate:     {config.training.grpo_learning_rate}\n")
        f.write(f"  KL penalty:        {config.grpo.kl_penalty}\n")
        f.write(f"  Clip epsilon:      {config.grpo.clip_eps}\n")
        f.write(f"  Update epochs:     {config.grpo.update_epochs}\n")
        f.write(f"  Num epochs:        {config.training.grpo_num_epochs}\n")
        f.write(f"  Entropy coef:      {config.training.grpo_entropy_coef}\n")
        f.write(f"  Ref policy dtype:  {config.system.ref_policy_dtype}\n")
        f.write(f"  Device:            {config.system.device}\n")
        f.write(f"  Seed:              {config.experiment.seed}\n")
        f.write("\n")

        f.write("Training\n")
        f.write("-" * 40 + "\n")
        minutes = training_time / 60
        f.write(f"  Training time:     {training_time:.1f}s ({minutes:.1f}min)\n")
        f.write(f"  Total steps:       {len(metrics_history)}\n")
        f.write(f"  Best greedy acc:   {best_accuracy:.4f}\n")
        f.write("\n")

        f.write("Baseline (before training)\n")
        f.write("-" * 40 + "\n")
        for key, value in base_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key:20s}: {value:.4f}\n")
            else:
                f.write(f"  {key:20s}: {value}\n")
        f.write("\n")

        f.write("Final Evaluation (after training)\n")
        f.write("-" * 40 + "\n")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key:20s}: {value:.4f}\n")
            else:
                f.write(f"  {key:20s}: {value}\n")
        f.write("\n")

        base_acc = base_metrics.get("greedy_accuracy", 0)
        final_acc = final_metrics.get("greedy_accuracy", 0)
        f.write("Improvement\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Greedy accuracy:   {base_acc:.4f} -> {final_acc:.4f} ({final_acc - base_acc:+.4f})\n")
        base_reward = base_metrics.get("mean_reward", 0)
        final_reward = final_metrics.get("mean_reward", 0)
        f.write(f"  Mean reward:       {base_reward:.4f} -> {final_reward:.4f} ({final_reward - base_reward:+.4f})\n")
        f.write("\n")

        if metrics_history:
            f.write("Training Metrics (last 10 logged steps)\n")
            f.write("-" * 40 + "\n")
            recent = metrics_history[-10:]
            for m in recent:
                step = m.get("step", "?")
                pl = m.get("policy_loss", 0)
                rm = m.get("reward_mean", 0)
                kl = m.get("kl_divergence", 0)
                ent = m.get("entropy", 0)
                f.write(f"  step={step:>5}  loss={pl:>8.4f}  reward={rm:>6.3f}  kl={kl:>8.5f}  entropy={ent:>8.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Results directory: {run_dir}\n")


def evaluate_gsm8k_accuracy(
    trainer,
    eval_data,
    reward_fn,
    num_samples=1,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    batch_size=8,
    run_dir=None,
    stage=None,
):
    """
    Evaluate GSM8K accuracy. Optionally log detailed responses to responses.jsonl.
    """
    trainer.policy.model.eval()
    correct_greedy = 0
    correct_any = 0
    total = 0
    all_rewards = []

    with torch.no_grad():
        pbar = tqdm(
            range(0, len(eval_data), batch_size),
            desc="Evaluating GSM8K",
            unit="batch",
            disable=os.environ.get("TQDM_DISABLE") == "1",
        )
        for batch_start in pbar:
            batch = eval_data[batch_start : batch_start + batch_size]
            batch_prompts = [item["prompt"] for item in batch]
            batch_answers = [item["answer"] for item in batch]

            greedy_responses, _, _, _, _ = trainer.policy.generate(
                prompts=batch_prompts,
                max_length=max_length,
                temperature=0.01,
                top_p=1.0,
                do_sample=False,
                return_log_probs=False,
            )

            greedy_texts = [
                f"{p} {r}" for p, r in zip(batch_prompts, greedy_responses)
            ]
            greedy_rewards = reward_fn.get_rewards(greedy_texts)

            if run_dir and stage:
                append_responses_jsonl(
                    run_dir=run_dir,
                    stage=stage,
                    prompts=batch_prompts,
                    responses=greedy_responses,
                    rewards=greedy_rewards,
                    ground_truths=batch_answers,
                )

            for reward in greedy_rewards:
                total += 1
                if reward >= reward_fn.correct_reward:
                    correct_greedy += 1
                all_rewards.append(reward)

            if num_samples > 1:
                dup_prompts = batch_prompts * num_samples
                sampled_responses, _, _, _, _ = trainer.policy.generate(
                    prompts=dup_prompts,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    return_log_probs=False,
                )
                sampled_texts = [
                    f"{p} {r}" for p, r in zip(dup_prompts, sampled_responses)
                ]
                sampled_rewards = reward_fn.get_rewards(sampled_texts)

                for i in range(len(batch_prompts)):
                    sample_rewards = sampled_rewards[i :: len(batch_prompts)]
                    if any(r >= reward_fn.correct_reward for r in sample_rewards):
                        correct_any += 1

            pbar.set_postfix({
                "greedy_acc": f"{correct_greedy / max(total, 1):.3f}",
                "total": total,
            })

    trainer.policy.model.train()

    metrics = {
        "greedy_accuracy": correct_greedy / max(total, 1),
        "num_evaluated": total,
        "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
    }
    if num_samples > 1:
        metrics[f"pass@{num_samples}"] = correct_any / max(total, 1)

    return metrics


def run_gsm8k_grpo_training(
    config: AssignmentConfig,
    results_root: str = "results",
    sft_model_path: str = None,
):
    """Main function to run GRPO training on GSM8K."""

    run_dir = make_run_dir(config, results_root)
    save_config_json(config, run_dir)

    log_path = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    global logger
    logger = logging.getLogger(__name__)

    set_seed(config.experiment.seed)
    device = torch.device(config.system.device)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Using device: {device}")

    logger.info("Loading GSM8K data...")
    try:
        train_data = load_json_data(config.data.gsm8k_train_prompts_path)
        eval_data = load_json_data(config.data.gsm8k_eval_prompts_path)
    except FileNotFoundError as e:
        logger.error(f"Could not find GSM8K data files: {e}")
        logger.error("Please run: python scripts/prepare_gsm8k.py")
        return

    train_prompts = [item["prompt"] for item in train_data]

    max_train_prompts = getattr(config.experiment, "max_train_prompts", 0)
    if max_train_prompts and max_train_prompts > 0:
        train_prompts = train_prompts[:max_train_prompts]
        train_data = train_data[:max_train_prompts]

    max_eval_prompts = getattr(config.experiment, "eval_prompts_sample_size", 0)
    if max_eval_prompts and max_eval_prompts > 0:
        eval_data = eval_data[:max_eval_prompts]

    logger.info(f"Training prompts: {len(train_prompts)}")
    logger.info(f"Evaluation prompts: {len(eval_data)}")

    load_path = sft_model_path if sft_model_path else config.model.model_name
    logger.info(f"Loading policy model: {load_path}")
    if sft_model_path:
        logger.info(f"  (SFT-warmed checkpoint, base model: {config.model.model_name})")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    tokenizer.padding_side = "left"
    policy_model = AutoModelForCausalLM.from_pretrained(load_path)

    if getattr(config.gsm8k, "disable_thinking", False):
        logger.info("Disabling thinking mode via chat template (enable_thinking=False)")
        if hasattr(tokenizer, "apply_chat_template"):
            for item in train_data + eval_data:
                messages = [{"role": "user", "content": item["prompt"]}]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                item["prompt"] = formatted
            train_prompts = [item["prompt"] for item in train_data]
            logger.info(f"Sample formatted prompt:\n{train_prompts[0][:200]}...")
        else:
            logger.warning("Tokenizer has no apply_chat_template — skipping disable_thinking")

    reward_fn = GSM8KRewardFunction(
        correct_reward=config.gsm8k.correct_reward,
        format_reward=config.gsm8k.format_reward,
        incorrect_reward=config.gsm8k.incorrect_reward,
        answer_delimiter=config.gsm8k.answer_delimiter,
    )
    reward_fn.set_prompt_answers(train_data + eval_data)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.pad_token_id

    trainer = create_grpo_trainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_model=reward_fn,
        config=config,
        device=device,
        use_minitorch=config.grpo.use_minitorch,
    )

    metrics_tracker = MetricsTracker()

    logger.info("Evaluating base model on GSM8K...")
    base_eval_metrics = evaluate_gsm8k_accuracy(
        trainer=trainer,
        eval_data=eval_data,
        reward_fn=reward_fn,
        num_samples=1,
        max_length=config.grpo.rollout_max_length,
        batch_size=config.experiment.eval_batch_size,
        run_dir=run_dir,
        stage="baseline",
    )
    logger.info("Base model evaluation:")
    for key, value in base_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    logger.info("Starting GRPO training on GSM8K...")
    start_time = time.time()
    best_accuracy = 0.0
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

            epoch_progress.set_postfix({
                "reward": f"{step_metrics.reward_mean:.3f}",
                "policy_loss": f"{step_metrics.policy_loss:.4f}",
                "kl": f"{step_metrics.kl_divergence:.4f}",
            })

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
                logger.info(f"Evaluating after batch {batch_idx + 1}...")
                eval_metrics = evaluate_gsm8k_accuracy(
                    trainer=trainer,
                    eval_data=eval_data,
                    reward_fn=reward_fn,
                    max_length=config.grpo.rollout_max_length,
                    batch_size=config.experiment.eval_batch_size,
                    run_dir=run_dir,
                    stage=f"epoch{epoch+1}_batch{batch_idx+1}",
                )
                current_acc = eval_metrics["greedy_accuracy"]
                logger.info(f"  Greedy accuracy: {current_acc:.4f}")
                if current_acc > best_accuracy:
                    best_accuracy = current_acc
                    best_path = os.path.join(run_dir, "checkpoints", "best_model")
                    trainer.policy.model.save_pretrained(best_path)
                    trainer.tokenizer.save_pretrained(best_path)
                    logger.info(f"  New best model saved (accuracy: {current_acc:.4f})")

        for key in epoch_metrics:
            if key != "epoch" and num_batches > 0:
                epoch_metrics[key] /= num_batches

        logger.info(f"Evaluating after epoch {epoch + 1}...")
        eval_metrics = evaluate_gsm8k_accuracy(
            trainer=trainer,
            eval_data=eval_data,
            reward_fn=reward_fn,
            num_samples=3,
            max_length=config.grpo.rollout_max_length,
            batch_size=config.experiment.eval_batch_size,
            run_dir=run_dir,
            stage=f"epoch{epoch+1}_end",
        )
        for key, value in eval_metrics.items():
            epoch_metrics[f"eval_{key}"] = value

        current_acc = eval_metrics["greedy_accuracy"]
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_path = os.path.join(run_dir, "checkpoints", "best_model")
            trainer.policy.model.save_pretrained(best_path)
            trainer.tokenizer.save_pretrained(best_path)
            logger.info(f"Saved new best model with accuracy: {current_acc:.4f}")

        checkpoint_path = os.path.join(
            run_dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}.pt"
        )
        trainer.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
        metrics_tracker.update(epoch_metrics, step=epoch + 1)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    final_eval_metrics = evaluate_gsm8k_accuracy(
        trainer=trainer,
        eval_data=eval_data,
        reward_fn=reward_fn,
        num_samples=5,
        max_length=config.grpo.rollout_max_length,
        batch_size=config.experiment.eval_batch_size,
        run_dir=run_dir,
        stage="final",
    )
    logger.info("Final evaluation:")
    for key, value in final_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    final_path = os.path.join(run_dir, "checkpoints", "final_model")
    trainer.policy.model.save_pretrained(final_path)
    trainer.tokenizer.save_pretrained(final_path)

    metrics_tracker.save_metrics(os.path.join(run_dir, "training_metrics.json"))

    step_metrics_only = [
        m for m in metrics_tracker.metrics_history if "epoch" not in m
    ]

    plot_training_curves(
        step_metrics_only,
        os.path.join(run_dir, "training_curves.png"),
        title="GSM8K GRPO Training Progress",
    )

    write_summary_txt(
        run_dir=run_dir,
        config=config,
        base_metrics=base_eval_metrics,
        final_metrics=final_eval_metrics,
        training_time=training_time,
        best_accuracy=best_accuracy,
        metrics_history=metrics_tracker.metrics_history,
    )

    logger.info(f"Best greedy accuracy: {best_accuracy:.4f}")
    logger.info(f"All results saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run GRPO training on GSM8K")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3.5-2B",
        help="HuggingFace model name",
    )
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="Prompts per batch")
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--update_epochs", type=int, default=None)
    parser.add_argument("--rollout_max_length", type=int, default=None)
    parser.add_argument("--max_train_prompts", type=int, default=0)
    parser.add_argument("--max_eval_prompts", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--kl_penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--ref_policy_dtype", type=str, default=None,
        choices=["float32", "float16"],
    )
    parser.add_argument(
        "--results_root", type=str, default="results",
        help="Root directory for results",
    )
    parser.add_argument(
        "--sft_model_path", type=str, default=None,
        help="Path to SFT-warmed model (overrides --model_name for weights)",
    )
    parser.add_argument(
        "--disable_thinking", action="store_true",
        help="Prepend /no_think to prompts to suppress Qwen3 <think> tags",
    )
    parser.add_argument(
        "--use_minitorch", action="store_true",
        help="Use minitorch CUDA kernels for GRPO ops (advantage norm, objective)",
    )
    args = parser.parse_args()

    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_config()

    config.model.model_name = args.model_name

    if args.group_size is not None:
        config.grpo.group_size = args.group_size
    else:
        config.grpo.group_size = 16

    if args.rollout_max_length is not None:
        config.grpo.rollout_max_length = args.rollout_max_length
    else:
        config.grpo.rollout_max_length = 512

    if args.temperature is not None:
        config.grpo.temperature = args.temperature
    else:
        config.grpo.temperature = 1.0

    if args.kl_penalty is not None:
        config.grpo.kl_penalty = args.kl_penalty
    else:
        config.grpo.kl_penalty = 0.04

    config.grpo.clip_eps = 0.2

    if args.batch_size is not None:
        config.grpo.rollout_batch_size = args.batch_size
    else:
        config.grpo.rollout_batch_size = 2

    if args.learning_rate is not None:
        config.training.grpo_learning_rate = args.learning_rate
    else:
        config.training.grpo_learning_rate = 1e-6

    if args.num_epochs is not None:
        config.training.grpo_num_epochs = args.num_epochs

    if args.update_epochs is not None:
        config.grpo.update_epochs = args.update_epochs
    else:
        config.grpo.update_epochs = 1

    if args.ref_policy_dtype is not None:
        config.system.ref_policy_dtype = args.ref_policy_dtype

    config.experiment.max_train_prompts = args.max_train_prompts
    config.experiment.eval_prompts_sample_size = args.max_eval_prompts

    if args.eval_steps is not None:
        config.system.eval_steps = args.eval_steps

    if args.disable_thinking:
        config.gsm8k.disable_thinking = True

    if args.use_minitorch:
        config.grpo.use_minitorch = True

    config.validate()
    run_gsm8k_grpo_training(
        config,
        results_root=args.results_root,
        sft_model_path=args.sft_model_path,
    )


if __name__ == "__main__":
    main()
