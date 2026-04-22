"""
Correctness tests: minitorch ops vs PyTorch reference.
Run with: pytest minitorch/tests/test_ops.py -v
"""

import pytest
import torch

from minitorch.ops import (
    fused_log_prob_gather,
    group_advantage_norm,
    fused_grpo_objective,
)


def _ref_log_prob_gather(scores, token_ids):
    log_norm = torch.logsumexp(scores, dim=-1)
    gathered = scores.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
    return gathered - log_norm


def _ref_group_advantage_norm(rewards, eps):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    norm = (rewards - mean) / (std + eps)
    return torch.where(std > 0, norm, torch.zeros_like(rewards))


def _ref_grpo_objective(new_lp, old_lp, ref_lp, adv, clip_eps, kl_coeff):
    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pl = -torch.min(ratio * adv, clipped * adv).mean()
    log_r = new_lp - ref_lp
    kl = ((torch.exp(log_r) - 1) - log_r).mean()
    return pl, kl, pl + kl_coeff * kl


class TestFusedLogProbGather:
    def test_matches_reference(self):
        B, T, V = 4, 16, 1000
        scores = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        result = fused_log_prob_gather(scores, token_ids)
        expected = _ref_log_prob_gather(scores, token_ids)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_large_vocab(self):
        B, T, V = 2, 8, 151936
        scores = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        result = fused_log_prob_gather(scores, token_ids)
        expected = _ref_log_prob_gather(scores, token_ids)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


class TestGroupAdvantageNorm:
    def test_matches_reference(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
        eps = 1e-8
        result = group_advantage_norm(rewards, eps)
        expected = _ref_group_advantage_norm(rewards, eps)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_zero_std_gives_zeros(self):
        rewards = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
        result = group_advantage_norm(rewards, 1e-8)
        assert (result == 0).all()


class TestFusedGRPOObjective:
    def test_matches_reference(self):
        L = 128
        new_lp = torch.randn(L)
        old_lp = new_lp + torch.randn(L) * 0.1
        ref_lp = new_lp + torch.randn(L) * 0.1
        adv = torch.randn(L)
        clip_eps, kl_coeff = 0.2, 0.04

        pl, kl, total = fused_grpo_objective(new_lp, old_lp, ref_lp, adv, clip_eps, kl_coeff)
        e_pl, e_kl, e_total = _ref_grpo_objective(new_lp, old_lp, ref_lp, adv, clip_eps, kl_coeff)

        torch.testing.assert_close(pl, e_pl, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(kl, e_kl, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(total, e_total, atol=1e-5, rtol=1e-5)
