"""
Test cases for EOS/PAD token masking in VERLPolicyWrapper.
Tests both the GPT-2 path (pad == eos) and the Qwen3 path (pad != eos).
"""

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

from rlhf_trainer import VERLPolicyWrapper


class TinyCausalLM(nn.Module):
    """Minimal causal LM for testing mask logic."""

    def __init__(self, vocab_size=100, hidden_size=16, eos_token_id=0, pad_token_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=[hidden])
        return SimpleNamespace(logits=logits)

    def generate(
        self, input_ids, attention_mask=None, max_length=16, temperature=1.0,
        top_p=1.0, do_sample=True, pad_token_id=0, eos_token_id=0,
        return_dict_in_generate=True, output_scores=True, **kwargs,
    ):
        """Generate exactly 3 new tokens: [5, eos_token_id, pad_token_id]."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        token_1 = torch.full((batch_size, 1), 5, dtype=torch.long, device=device)
        token_eos = torch.full((batch_size, 1), eos_token_id, dtype=torch.long, device=device)
        token_pad = torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device)

        sequences = torch.cat([input_ids, token_1, token_eos, token_pad], dim=1)

        scores = []
        for _ in range(3):
            logits = torch.randn(batch_size, self.config.vocab_size, device=device)
            scores.append(logits)

        return SimpleNamespace(sequences=sequences, scores=tuple(scores))


class SameEOSPadTokenizer:
    """Tokenizer where pad_token_id == eos_token_id (GPT-2 style)."""

    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.padding_side = "left"

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", max_length=None):
        # Simple encoding: each char -> its ord % 100
        encoded = [[ord(c) % 100 for c in t[:5]] for t in texts]
        max_len = max(len(e) for e in encoded)
        input_ids = []
        attention_mask = []
        for tokens in encoded:
            pad_len = max_len - len(tokens)
            input_ids.append([self.pad_token_id] * pad_len + tokens)  # left-pad
            attention_mask.append([0] * pad_len + [1] * len(tokens))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return ["resp"] * token_ids.shape[0]


class DiffEOSPadTokenizer:
    """Tokenizer where pad_token_id != eos_token_id (Qwen3 style)."""

    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.padding_side = "left"

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", max_length=None):
        encoded = [[ord(c) % 100 for c in t[:5]] for t in texts]
        max_len = max(len(e) for e in encoded)
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

    def batch_decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return ["resp"] * token_ids.shape[0]


class TestSameEOSPad:
    """Regression tests for GPT-2 path where pad_token_id == eos_token_id."""

    def test_generate_returns_valid_shapes(self):
        tokenizer = SameEOSPadTokenizer()
        model = TinyCausalLM(eos_token_id=0, pad_token_id=0)
        wrapper = VERLPolicyWrapper(model, tokenizer)

        responses, log_probs, prompt_len, full_seq, full_mask = wrapper.generate(
            prompts=["hello", "world"],
            max_length=20,
            temperature=1.0,
        )

        assert len(responses) == 2
        assert full_seq.shape[0] == 2
        assert full_mask.shape == full_seq.shape

    def test_mask_includes_first_eos(self):
        tokenizer = SameEOSPadTokenizer()
        model = TinyCausalLM(eos_token_id=0, pad_token_id=0)
        wrapper = VERLPolicyWrapper(model, tokenizer)

        _, _, prompt_len, full_seq, full_mask = wrapper.generate(
            prompts=["hello"],
            max_length=20,
        )

        # Generated tokens: [5, 0(eos), 0(pad)].
        # With pad==eos, the cumsum logic should keep token 5 and first EOS.
        gen_mask = full_mask[0, prompt_len:]
        # Token 5: kept (cumsum=0, <=1). EOS: kept (cumsum=1, <=1). PAD: masked (cumsum=2, >1).
        assert gen_mask[0].item() == 1  # token 5
        assert gen_mask[1].item() == 1  # first EOS
        assert gen_mask[2].item() == 0  # second EOS/PAD

    def test_generate_can_skip_log_prob_computation(self):
        tokenizer = SameEOSPadTokenizer()
        model = TinyCausalLM(eos_token_id=0, pad_token_id=0)
        wrapper = VERLPolicyWrapper(model, tokenizer)

        responses, log_probs, prompt_len, full_seq, full_mask = wrapper.generate(
            prompts=["hello"],
            max_length=20,
            return_log_probs=False,
        )

        assert responses == ["resp"]
        assert log_probs is None
        assert full_seq.shape == full_mask.shape
        assert prompt_len > 0


class TestDiffEOSPad:
    """Tests for Qwen3 path where pad_token_id != eos_token_id."""

    def test_generate_returns_valid_shapes(self):
        tokenizer = DiffEOSPadTokenizer()
        model = TinyCausalLM(eos_token_id=1, pad_token_id=2)
        wrapper = VERLPolicyWrapper(model, tokenizer)

        responses, log_probs, prompt_len, full_seq, full_mask = wrapper.generate(
            prompts=["hello", "world"],
            max_length=20,
        )

        assert len(responses) == 2
        assert full_seq.shape[0] == 2

    def test_mask_includes_eos_excludes_pad(self):
        tokenizer = DiffEOSPadTokenizer()
        model = TinyCausalLM(eos_token_id=1, pad_token_id=2)
        wrapper = VERLPolicyWrapper(model, tokenizer)

        _, _, prompt_len, full_seq, full_mask = wrapper.generate(
            prompts=["hello"],
            max_length=20,
        )

        # Generated tokens: [5, 1(eos), 2(pad)]
        gen_mask = full_mask[0, prompt_len:]
        assert gen_mask[0].item() == 1  # token 5: kept
        assert gen_mask[1].item() == 1  # EOS: kept (first EOS)
        assert gen_mask[2].item() == 0  # PAD: masked out

    def test_log_probs_stop_at_eos(self):
        tokenizer = DiffEOSPadTokenizer()
        model = TinyCausalLM(eos_token_id=1, pad_token_id=2)
        wrapper = VERLPolicyWrapper(model, tokenizer)

        _, log_probs, prompt_len, full_seq, full_mask = wrapper.generate(
            prompts=["hello"],
            max_length=20,
        )

        # Log probs should only cover tokens up to and including EOS
        # Generated: [5, 1(eos), 2(pad)] -> log_probs for [5, 1(eos)] = 2 tokens
        assert len(log_probs) == 1  # one sequence
        assert log_probs[0].shape[0] == 2  # token 5 + EOS
