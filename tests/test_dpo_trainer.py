"""
Test cases for the DPO trainer implementation.
"""

import copy
import os
import sys
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
from dpo_trainer import create_dpo_trainer


class DummyTokenizer:
    """Small tokenizer for DPO unit tests."""

    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.padding_side = "left"
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

        if not padding:
            assert len(encoded) == 1
            return {
                "input_ids": torch.tensor([encoded[0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1] * len(encoded[0])], dtype=torch.long),
            }

        max_len = max(len(tokens) for tokens in encoded)
        input_ids = []
        attention_mask = []
        for tokens in encoded:
            pad_len = max_len - len(tokens)
            input_ids.append([self.pad_token_id] * pad_len + tokens)
            attention_mask.append([0] * pad_len + [1] * len(tokens))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DummyCausalLM(nn.Module):
    """Tiny causal LM for DPO integration tests."""

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


@pytest.fixture
def trainer():
    """Create a DPO trainer backed by tiny test doubles."""
    config = copy.deepcopy(get_config())
    config.system.device = "cpu"
    config.data.max_prompt_length = 8
    config.data.max_response_length = 8
    config.dpo.beta = 0.3
    config.dpo.label_smoothing = 0.0
    config.training.dpo_learning_rate = 1e-2
    config.training.dpo_max_grad_norm = 1.0

    tokenizer = DummyTokenizer()
    policy_model = DummyCausalLM()
    return create_dpo_trainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        config=config,
        device=torch.device("cpu"),
    )


def test_dpo_config_validation_rejects_invalid_beta():
    """DPO config should reject non-positive beta values."""
    config = copy.deepcopy(get_config())
    config.dpo.beta = 0.0
    with pytest.raises(ValueError, match="beta"):
        config.validate()


def test_prepare_batch_rejects_invalid_prompt_prefix(trainer):
    """Chosen and rejected texts must begin with the prompt."""
    with pytest.raises(ValueError, match="begin with the prompt"):
        trainer.prepare_batch(
            [
                {
                    "prompt": "ab",
                    "chosen": "zzchosen",
                    "rejected": "abreject",
                }
            ]
        )


def test_prepare_batch_masks_prompt_tokens_and_padding(trainer):
    """Completion masks should exclude prompt tokens and left padding."""
    batch = trainer.prepare_batch(
        [
            {
                "prompt": "ab",
                "chosen": "abcd",
                "rejected": "abef",
            },
            {
                "prompt": "a",
                "chosen": "axy",
                "rejected": "azq",
            },
        ]
    )

    expected_mask = torch.tensor(
        [
            [False, False, True, True],
            [False, False, True, True],
        ]
    )

    assert torch.equal(batch.chosen_completion_mask.cpu(), expected_mask)
    assert torch.equal(batch.rejected_completion_mask.cpu(), expected_mask)


def test_compute_dpo_loss_matches_standard_sigmoid_objective(trainer):
    """The unsmoothed DPO objective should match the standard formula."""
    chosen_logratios = torch.tensor([1.0, -0.5])
    rejected_logratios = torch.tensor([0.25, -1.0])

    loss, preference_logits = trainer._compute_dpo_loss(
        chosen_logratios=chosen_logratios,
        rejected_logratios=rejected_logratios,
    )

    expected_logits = trainer.config.dpo.beta * (chosen_logratios - rejected_logratios)
    expected_loss = -torch.nn.functional.logsigmoid(expected_logits).mean()

    assert torch.allclose(preference_logits, expected_logits)
    assert torch.allclose(loss, expected_loss)


def test_label_smoothing_zero_matches_standard_dpo(trainer):
    """Zero label smoothing should reduce to the standard DPO loss."""
    trainer.config.dpo.label_smoothing = 0.0
    chosen_logratios = torch.tensor([0.5])
    rejected_logratios = torch.tensor([-0.5])

    loss, _ = trainer._compute_dpo_loss(chosen_logratios, rejected_logratios)
    expected = -torch.nn.functional.logsigmoid(
        trainer.config.dpo.beta * (chosen_logratios - rejected_logratios)
    ).mean()

    assert torch.allclose(loss, expected)


def test_train_step_updates_policy_and_leaves_ref_policy_frozen(trainer):
    """A DPO step should update the policy but not the reference policy."""
    batch = trainer.prepare_batch(
        [
            {
                "prompt": "ab",
                "chosen": "abcd",
                "rejected": "abzz",
            },
            {
                "prompt": "cd",
                "chosen": "cdef",
                "rejected": "cdxx",
            },
        ]
    )

    policy_before = trainer.policy.model.lm_head.weight.detach().clone()
    ref_before = trainer.ref_policy.model.lm_head.weight.detach().clone()
    metrics = trainer.train_step(batch)
    policy_after = trainer.policy.model.lm_head.weight.detach()
    ref_after = trainer.ref_policy.model.lm_head.weight.detach()

    assert torch.isfinite(torch.tensor(metrics.loss))
    assert torch.isfinite(torch.tensor(metrics.accuracy))
    assert torch.isfinite(torch.tensor(metrics.margin))
    assert not torch.allclose(policy_before, policy_after)
    assert torch.allclose(ref_before, ref_after)
    assert not hasattr(trainer, "reward_model")


def test_prepare_batch_skips_examples_without_completion_tokens_after_truncation(trainer):
    """Examples truncated down to prompt-only tokens should be dropped, not fatal."""
    batch = trainer.prepare_batch(
        [
            {
                "prompt": "abcdefgh",
                "chosen": "abcdefghij",
                "rejected": "abcdefghkl",
            },
            {
                "prompt": "ab",
                "chosen": "abcd",
                "rejected": "abef",
            },
        ]
    )

    assert batch.chosen_input_ids.shape[0] == 1
    assert batch.rejected_input_ids.shape[0] == 1
    assert torch.equal(
        batch.chosen_completion_mask.cpu(),
        torch.tensor([[False, False, True, True]]),
    )
    assert torch.equal(
        batch.rejected_completion_mask.cpu(),
        torch.tensor([[False, False, True, True]]),
    )


def test_prepare_batch_raises_when_all_examples_lack_completion_tokens(trainer):
    """A fully unusable batch should still fail clearly."""
    with pytest.raises(ValueError, match="no usable examples after tokenization"):
        trainer.prepare_batch(
            [
                {
                    "prompt": "abcdefgh",
                    "chosen": "abcdefghij",
                    "rejected": "abcdefghkl",
                }
            ]
        )


def test_checkpoint_round_trip_restores_policy_state(trainer, tmp_path):
    """Checkpoint save/load should restore the policy and ref-policy state."""
    batch = trainer.prepare_batch(
        [
            {
                "prompt": "ab",
                "chosen": "abcd",
                "rejected": "abzz",
            }
        ]
    )
    trainer.train_step(batch)
    checkpoint_path = tmp_path / "dpo_checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path), epoch=0, metrics={"loss": 1.0})

    saved_policy = trainer.policy.model.lm_head.weight.detach().clone()
    saved_ref = trainer.ref_policy.model.lm_head.weight.detach().clone()

    with torch.no_grad():
        trainer.policy.model.lm_head.weight.add_(1.0)
        trainer.ref_policy.model.lm_head.weight.add_(1.0)

    trainer.load_checkpoint(str(checkpoint_path))

    assert torch.allclose(trainer.policy.model.lm_head.weight.detach(), saved_policy)
    assert torch.allclose(trainer.ref_policy.model.lm_head.weight.detach(), saved_ref)
