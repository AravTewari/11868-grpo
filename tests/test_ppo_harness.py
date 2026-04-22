"""
Regression tests for the PPO harness CLI and artifact naming.
"""

import os
import sys
import copy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for path in (REPO_ROOT, SRC_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import get_config
import run_ppo


def test_ppo_cli_overrides_use_ppo_settings_and_output_dir():
    config = copy.deepcopy(get_config())
    config.system.ppo_model_dir = "/tmp/custom_ppo_output"

    args = run_ppo.build_arg_parser().parse_args(
        [
            "--model_name",
            "gpt2-medium",
            "--model_loader",
            "tests.test_policy_io:custom_loader",
            "--model_saver",
            "tests.test_policy_io:custom_saver",
            "--learning_rate",
            "5e-6",
            "--num_epochs",
            "3",
            "--batch_size",
            "7",
            "--max_train_prompts",
            "111",
            "--max_eval_prompts",
            "19",
            "--eval_steps",
            "44",
            "--rollout_max_length",
            "96",
            "--ppo_update_epochs",
            "6",
        ]
    )

    updated = run_ppo.apply_cli_overrides(config, args)
    artifact_paths = run_ppo.get_ppo_artifact_paths(updated, epoch=2)

    assert updated.model.model_name == "gpt2-medium"
    assert updated.model.policy_loader == "tests.test_policy_io:custom_loader"
    assert updated.model.policy_saver == "tests.test_policy_io:custom_saver"
    assert updated.training.ppo_learning_rate == 5e-6
    assert updated.training.ppo_num_epochs == 3
    assert updated.verl.rollout_batch_size == 7
    assert updated.experiment.max_train_prompts == 111
    assert updated.experiment.eval_prompts_sample_size == 19
    assert updated.system.eval_steps == 44
    assert updated.verl.rollout_max_length == 96
    assert updated.verl.ppo_epochs == 6
    assert updated.system.rlhf_model_dir == updated.system.ppo_model_dir

    assert artifact_paths["best_model"] == "/tmp/custom_ppo_output/best_ppo_model"
    assert artifact_paths["final_model"] == "/tmp/custom_ppo_output/final_ppo_model"
    assert artifact_paths["checkpoint"] == "/tmp/custom_ppo_output/ppo_checkpoint_epoch_2.pt"
    assert artifact_paths["metrics"].endswith("ppo_training_metrics.json")
    assert artifact_paths["candidate_outputs"].endswith("ppo_outputs.json")
    assert artifact_paths["comparison"].endswith("ppo_model_comparison.json")


def test_ppo_cli_defaults_preserve_config_limits():
    config = copy.deepcopy(get_config())
    config.experiment.max_train_prompts = 12
    config.experiment.eval_prompts_sample_size = 7

    args = run_ppo.build_arg_parser().parse_args([])
    updated = run_ppo.apply_cli_overrides(config, args)

    assert updated.experiment.max_train_prompts == 12
    assert updated.experiment.eval_prompts_sample_size == 7
