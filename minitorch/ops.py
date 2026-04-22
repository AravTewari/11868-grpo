"""
Fused CUDA kernels for GRPO training via ctypes.

Each function has a PyTorch reference implementation that runs when the compiled
CUDA library is unavailable. Replace the body of each function with a ctypes
call to grpo_kernels.so once the kernels are compiled on the cluster.
"""

import os
import ctypes
import torch

_lib = None
_LIB_PATH = os.path.join(os.path.dirname(__file__), "cuda_kernels", "grpo_kernels.so")


def _get_lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            return None
        _lib = ctypes.CDLL(_LIB_PATH)
    return _lib


# ── Kernel 1: fused gather + logsumexp ─────────────────────────────

def fused_log_prob_gather(scores: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute log P(token) = logit(token) - logsumexp(logits) in a single pass.

    Args:
        scores:    [B, T, V] raw logits
        token_ids: [B, T]    generated token indices

    Returns:
        [B, T] per-token log probabilities
    """
    lib = _get_lib()
    if lib is not None:
        B, T, V = scores.shape
        out = torch.empty(B, T, device=scores.device, dtype=scores.dtype)
        lib.fused_log_prob_gather(
            ctypes.c_void_p(scores.data_ptr()),
            ctypes.c_void_p(token_ids.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(B), ctypes.c_int(T), ctypes.c_int(V),
        )
        torch.cuda.synchronize()
        return out

    log_norm = torch.logsumexp(scores, dim=-1)
    gathered = scores.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
    return gathered - log_norm


# ── Kernel 2: group advantage normalization ────────────────────────

def group_advantage_norm(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize rewards within each group: (r - mean) / (std + eps).

    Args:
        rewards: [N, G] per-prompt group rewards
        eps:     stability constant

    Returns:
        [N, G] normalized advantages (zero where std < 1e-8)
    """
    lib = _get_lib()
    if lib is not None:
        N, G = rewards.shape
        out = torch.empty_like(rewards)
        lib.group_advantage_norm(
            ctypes.c_void_p(rewards.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(N), ctypes.c_int(G), ctypes.c_float(eps),
        )
        torch.cuda.synchronize()
        return out

    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    norm = (rewards - mean) / (std + eps)
    return torch.where(std > 0, norm, torch.zeros_like(rewards))


# ── Kernel 3: fused clipped surrogate + KL ─────────────────────────

def fused_grpo_objective(
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    kl_coeff: float,
) -> tuple:
    """Compute clipped policy loss and approximate KL in one pass.

    Args:
        new_lp, old_lp, ref_lp: [L] flat per-token log probs
        advantages:              [L] per-token advantages
        clip_eps:  PPO clipping range
        kl_coeff:  KL penalty weight

    Returns:
        (policy_loss, kl_div, total_loss) scalar tensors
    """
    lib = _get_lib()
    if lib is not None:
        L = new_lp.shape[0]
        out = torch.empty(3, device=new_lp.device, dtype=new_lp.dtype)
        lib.fused_grpo_objective(
            ctypes.c_void_p(new_lp.data_ptr()),
            ctypes.c_void_p(old_lp.data_ptr()),
            ctypes.c_void_p(ref_lp.data_ptr()),
            ctypes.c_void_p(advantages.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(L), ctypes.c_float(clip_eps), ctypes.c_float(kl_coeff),
        )
        torch.cuda.synchronize()
        return out[0], out[1], out[2]

    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    log_r = new_lp - ref_lp
    kl_div = ((torch.exp(log_r) - 1) - log_r).mean()
    total_loss = policy_loss + kl_coeff * kl_div
    return policy_loss, kl_div, total_loss
