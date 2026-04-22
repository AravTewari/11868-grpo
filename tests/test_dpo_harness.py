"""
Regression tests for the DPO harness CLI, artifact naming, and data loading.
"""

import copy
import json
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for path in (REPO_ROOT, SRC_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import get_config
import run_dpo


def test_dpo_cli_overrides_use_dpo_settings_and_output_dir():
    config = copy.deepcopy(get_config())
    config.system.dpo_model_dir = "/tmp/custom_dpo_output"

    args = run_dpo.build_arg_parser().parse_args(
        [
            "--model_name",
            "gpt2-medium",
            "--model_loader",
            "tests.test_policy_io:custom_loader",
            "--model_saver",
            "tests.test_policy_io:custom_saver",
            "--learning_rate",
            "7e-6",
            "--num_epochs",
            "4",
            "--batch_size",
            "5",
            "--beta",
            "0.3",
            "--label_smoothing",
            "0.15",
            "--max_train_pairs",
            "111",
            "--max_val_pairs",
            "22",
            "--max_test_pairs",
            "33",
            "--max_eval_prompts",
            "19",
        ]
    )

    updated = run_dpo.apply_cli_overrides(config, args)
    artifact_paths = run_dpo.get_dpo_artifact_paths(updated, epoch=2)

    assert updated.model.model_name == "gpt2-medium"
    assert updated.model.policy_loader == "tests.test_policy_io:custom_loader"
    assert updated.model.policy_saver == "tests.test_policy_io:custom_saver"
    assert updated.training.dpo_learning_rate == 7e-6
    assert updated.training.dpo_num_epochs == 4
    assert updated.data.train_batch_size == 5
    assert updated.data.eval_batch_size == 5
    assert updated.dpo.beta == 0.3
    assert updated.dpo.label_smoothing == 0.15
    assert updated.experiment.max_train_pairs == 111
    assert updated.experiment.max_val_pairs == 22
    assert updated.experiment.max_test_pairs == 33
    assert updated.experiment.eval_prompts_sample_size == 19

    assert artifact_paths["best_model"] == "/tmp/custom_dpo_output/best_dpo_model"
    assert artifact_paths["final_model"] == "/tmp/custom_dpo_output/final_dpo_model"
    assert artifact_paths["checkpoint"] == "/tmp/custom_dpo_output/dpo_checkpoint_epoch_2.pt"
    assert artifact_paths["metrics"].endswith("dpo_training_metrics.json")
    assert artifact_paths["candidate_outputs"].endswith("dpo_outputs.json")
    assert artifact_paths["comparison"].endswith("dpo_model_comparison.json")


def test_dpo_cli_defaults_preserve_config_limits():
    config = copy.deepcopy(get_config())
    config.experiment.max_train_pairs = 12
    config.experiment.max_val_pairs = 5
    config.experiment.max_test_pairs = 4
    config.experiment.eval_prompts_sample_size = 6

    args = run_dpo.build_arg_parser().parse_args([])
    updated = run_dpo.apply_cli_overrides(config, args)

    assert updated.experiment.max_train_pairs == 12
    assert updated.experiment.max_val_pairs == 5
    assert updated.experiment.max_test_pairs == 4
    assert updated.experiment.eval_prompts_sample_size == 6


def test_load_preference_splits_uses_explicit_val_and_test_files(tmp_path):
    config = copy.deepcopy(get_config())
    config.data.preference_data_path = str(tmp_path / "preference_data.json")
    config.data.preference_val_data_path = str(tmp_path / "preference_data_val.json")
    config.data.preference_test_data_path = str(tmp_path / "preference_data_test.json")

    train_data = [{"prompt": "a", "chosen": "a+", "rejected": "a-"}]
    val_data = [{"prompt": "b", "chosen": "b+", "rejected": "b-"}]
    test_data = [{"prompt": "c", "chosen": "c+", "rejected": "c-"}]

    with open(config.data.preference_data_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    with open(config.data.preference_val_data_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f)
    with open(config.data.preference_test_data_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)

    loaded_train, loaded_val, loaded_test = run_dpo.load_preference_splits(config)

    assert loaded_train == train_data
    assert loaded_val == val_data
    assert loaded_test == test_data


def test_fallback_preference_split_is_deterministic(tmp_path):
    config = copy.deepcopy(get_config())
    config.data.preference_data_path = str(tmp_path / "preference_data.json")
    config.data.preference_val_data_path = str(tmp_path / "missing_val.json")
    config.data.preference_test_data_path = str(tmp_path / "missing_test.json")

    preference_data = [
        {"prompt": f"p{i}", "chosen": f"p{i}+", "rejected": f"p{i}-"}
        for i in range(10)
    ]
    with open(config.data.preference_data_path, "w", encoding="utf-8") as f:
        json.dump(preference_data, f)

    run_dpo.set_seed(config.experiment.seed)
    split_a = run_dpo.load_preference_splits(config)
    run_dpo.set_seed(config.experiment.seed)
    split_b = run_dpo.load_preference_splits(config)

    assert split_a == split_b


def test_evaluate_preference_dataset_skips_fully_truncated_batches():
    class DummyTrainer:
        def prepare_batch(self, batch_examples):
            if batch_examples[0]["prompt"] == "skip":
                raise ValueError("Preference batch has no usable examples after tokenization")
            return SimpleNamespace(chosen_input_ids=torch.ones((1, 3), dtype=torch.long))

        def evaluate_step(self, batch):
            return SimpleNamespace(
                loss=1.5,
                accuracy=0.75,
                margin=0.25,
                chosen_logp=-2.0,
                rejected_logp=-3.0,
            )

    metrics = run_dpo.evaluate_preference_dataset(
        trainer=DummyTrainer(),
        preference_data=[
            {"prompt": "skip", "chosen": "skip+", "rejected": "skip-"},
            {"prompt": "keep", "chosen": "keep+", "rejected": "keep-"},
        ],
        batch_size=1,
    )

    assert metrics == {
        "loss": 1.5,
        "accuracy": 0.75,
        "margin": 0.25,
        "chosen_logp": -2.0,
        "rejected_logp": -3.0,
    }
