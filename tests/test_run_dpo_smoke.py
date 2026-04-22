"""
Smoke test for the DPO harness entrypoint.
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


class DummyRewardModel:
    """Small reward model stub for DPO harness testing."""

    def __init__(self):
        self.tokenizer = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_rewards(self, texts, tokenizer, device=None):
        return [float(len(text) % 5) for text in texts]


class DummyHFModule:
    """Minimal save/load surface used by the DPO harness."""

    def __init__(self):
        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "dummy.bin"), "w", encoding="utf-8") as f:
            f.write("dummy")


class DummyTokenizer:
    """Tokenizer stub that only needs save_pretrained."""

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"dummy": True}, f)


class DummyPolicy:
    """Policy wrapper with the same generate surface used by DPO scripts."""

    def __init__(self):
        self.model = DummyHFModule()

    def generate(
        self,
        prompts,
        max_length=32,
        temperature=1.0,
        top_p=1.0,
        do_sample=True,
        return_log_probs=True,
        **kwargs,
    ):
        responses = [f"response-{idx}" for idx, _ in enumerate(prompts)]
        batch_size = len(prompts)
        full_seq = torch.ones((batch_size, 4), dtype=torch.long)
        full_mask = torch.ones_like(full_seq)
        log_probs = [torch.zeros(2, dtype=torch.float32) for _ in prompts]
        if not return_log_probs:
            log_probs = None
        return responses, log_probs, 2, full_seq, full_mask


class DummyTrainer:
    """Trainer stub that exercises the full DPO harness flow."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.policy = DummyPolicy()
        self.tokenizer = DummyTokenizer()
        self.prepare_calls = 0
        self.train_calls = 0
        self.eval_calls = 0

    def prepare_batch(self, preference_batch):
        self.prepare_calls += 1
        return SimpleNamespace(size=len(preference_batch))

    def train_step(self, batch):
        self.train_calls += 1
        return SimpleNamespace(
            loss=0.4,
            accuracy=0.75,
            margin=0.2,
            chosen_logp=-1.0,
            rejected_logp=-1.4,
        )

    def evaluate_step(self, batch):
        self.eval_calls += 1
        return SimpleNamespace(
            loss=0.3,
            accuracy=0.8,
            margin=0.25,
            chosen_logp=-0.9,
            rejected_logp=-1.3,
        )

    def save_checkpoint(self, checkpoint_path, epoch, metrics):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({"epoch": epoch, "metrics": metrics}, checkpoint_path)


def test_run_dpo_training_smoke(monkeypatch, tmp_path):
    """DPO harness should complete training and emit DPO-scoped artifacts."""
    config = copy.deepcopy(get_config())
    config.system.device = "cpu"
    config.system.logs_dir = str(tmp_path / "logs")
    config.system.dpo_model_dir = str(tmp_path / "outputs" / "dpo_model")
    config.system.reward_model_dir = str(tmp_path / "outputs" / "reward_model")
    config.training.dpo_num_epochs = 1
    config.data.train_batch_size = 2
    config.data.eval_batch_size = 1
    config.experiment.eval_batch_size = 1
    config.experiment.max_train_pairs = 2
    config.experiment.max_val_pairs = 1
    config.experiment.max_test_pairs = 1
    config.experiment.eval_prompts_sample_size = 1
    config.data.preference_data_path = str(tmp_path / "data" / "preference_data.json")
    config.data.preference_val_data_path = str(tmp_path / "data" / "preference_data_val.json")
    config.data.preference_test_data_path = str(tmp_path / "data" / "preference_data_test.json")
    config.data.eval_prompts_path = str(tmp_path / "data" / "eval_prompts.json")

    os.makedirs(os.path.dirname(config.data.preference_data_path), exist_ok=True)
    os.makedirs(config.system.reward_model_dir, exist_ok=True)

    train_data = [
        {"prompt": "A", "chosen": "A good", "rejected": "A bad"},
        {"prompt": "B", "chosen": "B good", "rejected": "B bad"},
    ]
    val_data = [
        {"prompt": "C", "chosen": "C good", "rejected": "C bad"},
    ]
    test_data = [
        {"prompt": "D", "chosen": "D good", "rejected": "D bad"},
    ]

    with open(config.data.preference_data_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    with open(config.data.preference_val_data_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f)
    with open(config.data.preference_test_data_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    with open(config.data.eval_prompts_path, "w", encoding="utf-8") as f:
        json.dump([{"prompt": "Eval prompt"}], f)
    with open(
        os.path.join(config.system.reward_model_dir, "best_reward_model.pt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("dummy")

    created = {}

    def create_dummy_trainer(policy_model, tokenizer, config, device):
        trainer = DummyTrainer(config=config, device=device)
        created["trainer"] = trainer
        return trainer

    monkeypatch.setattr(run_dpo, "load_reward_model", lambda path, device: DummyRewardModel())
    monkeypatch.setattr(
        run_dpo,
        "load_policy_model_and_tokenizer",
        lambda model_name, config=None, device=None: (DummyHFModule(), DummyTokenizer()),
    )
    monkeypatch.setattr(run_dpo, "create_dpo_trainer", create_dummy_trainer)

    run_dpo.run_dpo_training(config)

    trainer = created["trainer"]
    artifact_paths = run_dpo.get_dpo_artifact_paths(config, epoch=1)

    assert trainer.prepare_calls >= 3
    assert trainer.train_calls == 1
    assert trainer.eval_calls >= 2
    assert os.path.exists(artifact_paths["best_model"])
    assert os.path.exists(artifact_paths["final_model"])
    assert os.path.exists(artifact_paths["checkpoint"])
    assert os.path.exists(artifact_paths["metrics"])
    assert os.path.exists(artifact_paths["baseline_outputs"])
    assert os.path.exists(artifact_paths["candidate_outputs"])
    assert os.path.exists(artifact_paths["comparison"])
    assert os.path.exists(artifact_paths["summary"])
    assert os.path.exists(artifact_paths["training_curve"])
    assert os.path.exists(artifact_paths["reward_distribution"])
