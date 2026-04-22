"""
Test cases for the GRPO trainer implementation.
"""

import os
import sys
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import get_config
from grpo_trainer import create_grpo_trainer


class DummyTokenizer:
    """Small tokenizer for GRPO unit tests."""

    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.padding_side = "right"
        letters = "abcdefghijklmnopqrstuvwxyz "
        self.vocab = {ch: idx + 1 for idx, ch in enumerate(letters)}
        self.inv_vocab = {idx: ch for ch, idx in self.vocab.items()}

    def _encode(self, text: str):
        token_ids = [self.vocab.get(ch.lower(), 1) for ch in text]
        return token_ids or [1]

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=None,
    ):
        encoded = [self._encode(text) for text in texts]
        if truncation and max_length is not None:
            encoded = [tokens[:max_length] for tokens in encoded]

        max_len = max(len(tokens) for tokens in encoded)
        input_ids = []
        attention_mask = []
        for tokens in encoded:
            pad_len = max_len - len(tokens)
            input_ids.append(tokens + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(tokens) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ):
        decoded = []
        for seq in token_ids.tolist():
            chars = []
            for token_id in seq:
                if token_id == self.eos_token_id and skip_special_tokens:
                    continue
                chars.append(self.inv_vocab.get(token_id, "a"))
            decoded.append("".join(chars).strip())
        return decoded


class DummyCausalLM(nn.Module):
    """Tiny causal LM with deterministic grouped generation."""

    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            pad_token_id=0,
            eos_token_id=0,
        )

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        hidden = torch.tanh(self.embed(input_ids))
        logits = self.lm_head(hidden)
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=[hidden])
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=16,
        temperature=1.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
        **kwargs,
    ):
        sequences = input_ids.clone()
        batch_size = sequences.shape[0]
        device = sequences.device

        max_new_tokens = max(1, min(2, max_length - sequences.shape[1]))
        scores = []

        for step in range(max_new_tokens):
            outputs = self.forward(sequences)
            next_logits = outputs.logits[:, -1, :]

            if step == 0:
                preferred = (torch.arange(batch_size, device=device) % (self.config.vocab_size - 1)) + 1
                next_token = preferred
                biased_logits = next_logits.clone()
                biased_logits.scatter_add_(
                    1,
                    preferred.unsqueeze(1),
                    torch.full((batch_size, 1), 2.0, device=device),
                )
                next_logits = biased_logits
            else:
                next_token = torch.full((batch_size,), eos_token_id, dtype=torch.long, device=device)
                eos_logits = torch.full_like(next_logits, -1e9)
                eos_logits[:, eos_token_id] = 0.0
                next_logits = eos_logits

            scores.append(next_logits)
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)

        return SimpleNamespace(sequences=sequences, scores=tuple(scores))


class DummyRewardModel(nn.Module):
    """Reward model stub that scores based on the final alphabetic character."""

    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def get_rewards(self, texts, tokenizer, device=None, batch_size=8):
        rewards = []
        for text in texts:
            last_alpha = next((ch for ch in reversed(text) if ch.isalpha()), "a").lower()
            rewards.append(float(ord(last_alpha) - ord("a") + 1))
        return rewards


@pytest.fixture
def trainer():
    """Create a GRPO trainer backed by tiny test doubles."""
    config = copy.deepcopy(get_config())
    config.system.device = "cpu"
    config.grpo.group_size = 3
    config.grpo.rollout_batch_size = 2
    config.grpo.rollout_max_length = 12
    config.grpo.update_epochs = 1
    config.grpo.clip_eps = 0.2
    config.grpo.kl_penalty = 0.1
    config.training.grpo_learning_rate = 1e-2
    config.training.grpo_entropy_coef = 0.01
    config.training.grpo_max_grad_norm = 1.0

    tokenizer = DummyTokenizer()
    policy_model = DummyCausalLM()
    reward_model = DummyRewardModel()
    return create_grpo_trainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=config,
        device=torch.device("cpu"),
    )


def test_grpo_config_validation_requires_group_size_at_least_two():
    """GRPO config should reject invalid group sizes."""
    config = copy.deepcopy(get_config())
    config.grpo.group_size = 1
    with pytest.raises(ValueError, match="group_size"):
        config.validate()


def test_compute_group_advantages_zero_variance_group_returns_zero(trainer):
    """Groups with identical rewards should get zero advantages."""
    rewards = torch.tensor([[2.0, 2.0, 2.0], [1.0, 3.0, 5.0]])
    advantages = trainer._compute_group_advantages(rewards)

    assert torch.allclose(advantages[0], torch.zeros_like(advantages[0]))
    assert torch.allclose(advantages[1].mean(), torch.tensor(0.0), atol=1e-6)


def test_generate_rollouts_returns_grouped_shapes(trainer):
    """Rollouts should preserve prompt-major grouped structure."""
    batch = trainer.generate_rollouts(["alpha", "beta"])

    assert batch.rewards.shape == (2, trainer.config.grpo.group_size)
    assert batch.advantages.shape == (2, trainer.config.grpo.group_size)
    assert len(batch.responses) == 2 * trainer.config.grpo.group_size
    assert batch.full_seq.shape[0] == 2 * trainer.config.grpo.group_size
    assert batch.old_log_probs.ndim == 1
    assert torch.allclose(batch.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_expand_token_advantages_masks_prompt_and_padding(trainer):
    """Only valid generated tokens should receive repeated sequence advantages."""
    advantages = torch.tensor([[1.0, -1.0, 0.5]])
    token_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ]
    )

    expanded = trainer._expand_token_advantages(advantages, token_mask)
    expected = torch.tensor([1.0, 1.0, -1.0, 0.5, 0.5, 0.5])
    assert torch.allclose(expanded, expected)


def test_compute_policy_objective_matches_clipped_ratio_and_ref_kl(trainer):
    """The clipped policy objective should match the GRPO formula."""
    new_log_probs = torch.log(torch.tensor([1.5, 0.5]))
    old_log_probs = torch.zeros(2)
    ref_log_probs = torch.tensor([0.0, torch.log(torch.tensor(0.8)).item()])
    token_advantages = torch.tensor([1.0, -1.0])
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    policy_loss, entropy, kl_div, total_loss = trainer._compute_policy_objective(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        token_advantages=token_advantages,
        masked_logits=logits,
    )

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - trainer.config.grpo.clip_eps,
        1.0 + trainer.config.grpo.clip_eps,
    )
    expected_policy_loss = -torch.min(
        ratio * token_advantages, clipped_ratio * token_advantages
    ).mean()
    expected_entropy = trainer._compute_entropy(logits)
    log_ratio = new_log_probs - ref_log_probs
    expected_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
    expected_total = (
        expected_policy_loss
        - trainer.config.training.grpo_entropy_coef * expected_entropy
        + trainer.config.grpo.kl_penalty * expected_kl
    )

    assert torch.allclose(policy_loss, expected_policy_loss)
    assert torch.allclose(entropy, expected_entropy)
    assert torch.allclose(kl_div, expected_kl)
    assert torch.allclose(total_loss, expected_total)


def test_train_step_updates_policy_parameters(trainer):
    """A GRPO step should update policy parameters without a value head."""
    rollout_batch = trainer.generate_rollouts(["alpha", "beta"])
    assert rollout_batch.advantages.abs().sum().item() > 0

    before = trainer.policy.model.lm_head.weight.detach().clone()
    metrics = trainer.train_step(rollout_batch)
    after = trainer.policy.model.lm_head.weight.detach()

    assert torch.isfinite(torch.tensor(metrics.policy_loss))
    assert torch.isfinite(torch.tensor(metrics.entropy))
    assert torch.isfinite(torch.tensor(metrics.kl_divergence))
    assert not torch.allclose(before, after)
    assert not hasattr(trainer, "value_model")
    assert not hasattr(trainer, "value_optimizer")
