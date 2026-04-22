"""
Smoke test for the PPO harness entrypoint.
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
import run_ppo


class DummyRewardModel:
    """Small reward model stub for PPO harness testing."""

    def __init__(self):
        self.tokenizer = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_rewards(self, texts, tokenizer, device=None):
        return [float(len(text) % 7) for text in texts]


class DummyHFModule:
    """Minimal save/load surface used by the PPO harness."""

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
    """Policy wrapper with the same generate surface used by PPO scripts."""

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
    """Trainer stub that exercises the full PPO harness flow."""

    def __init__(self, config, reward_model, device):
        self.config = config
        self.reward_model = reward_model
        self.device = device
        self.policy = DummyPolicy()
        self.tokenizer = DummyTokenizer()
        self.rollout_calls = 0
        self.train_calls = 0

    def generate_rollouts(self, prompts):
        self.rollout_calls += 1
        return SimpleNamespace(prompts=prompts)

    def train_step(self, rollout_batch):
        self.train_calls += 1
        return SimpleNamespace(
            policy_loss=0.1,
            value_loss=0.2,
            entropy=0.3,
            kl_divergence=0.01,
            reward_mean=1.5,
            reward_std=0.0,
            advantage_mean=0.0,
            advantage_std=0.0,
        )

    def save_checkpoint(self, checkpoint_path, epoch, metrics):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({"epoch": epoch, "metrics": metrics}, checkpoint_path)


def test_run_ppo_training_smoke(monkeypatch, tmp_path):
    """PPO harness should complete training and emit PPO-scoped artifacts."""
    config = copy.deepcopy(get_config())
    config.system.device = "cpu"
    config.system.logs_dir = str(tmp_path / "logs")
    config.system.ppo_model_dir = str(tmp_path / "outputs" / "ppo_model")
    config.system.rlhf_model_dir = config.system.ppo_model_dir
    config.system.reward_model_dir = str(tmp_path / "outputs" / "reward_model")
    config.system.eval_steps = 100000
    config.training.ppo_num_epochs = 1
    config.verl.rollout_batch_size = 2
    config.verl.ppo_epochs = 1
    config.verl.rollout_max_length = 16
    config.experiment.eval_batch_size = 1
    config.experiment.max_train_prompts = 1
    config.experiment.eval_prompts_sample_size = 1
    config.data.train_prompts_path = str(tmp_path / "data" / "train_prompts.json")
    config.data.eval_prompts_path = str(tmp_path / "data" / "eval_prompts.json")

    os.makedirs(os.path.dirname(config.data.train_prompts_path), exist_ok=True)
    os.makedirs(config.system.reward_model_dir, exist_ok=True)

    with open(config.data.train_prompts_path, "w", encoding="utf-8") as f:
        json.dump([{"prompt": "Train prompt"}], f)
    with open(config.data.eval_prompts_path, "w", encoding="utf-8") as f:
        json.dump([{"prompt": "Eval prompt"}], f)
    with open(
        os.path.join(config.system.reward_model_dir, "best_reward_model.pt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("dummy")

    created = {}

    def create_dummy_trainer(model_name, reward_model, config, device):
        trainer = DummyTrainer(config=config, reward_model=reward_model, device=device)
        created["trainer"] = trainer
        return trainer

    monkeypatch.setattr(run_ppo, "load_reward_model", lambda path, device: DummyRewardModel())
    monkeypatch.setattr(run_ppo, "create_ppo_trainer", create_dummy_trainer)

    run_ppo.run_ppo_training(config)

    trainer = created["trainer"]
    artifact_paths = run_ppo.get_ppo_artifact_paths(config, epoch=1)

    assert trainer.rollout_calls == 1
    assert trainer.train_calls == 1
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
