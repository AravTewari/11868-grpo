"""
GRPO trainer with mini-batch support and optional minitorch CUDA kernels.

Reuses VERLPolicyWrapper from rlhf_trainer for the policy/ref-policy wrappers.
Adds gradient accumulation across rollout batches and memory-efficient
mini-batched forward/backward passes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from src.config import AssignmentConfig
    from src.rlhf_trainer import VERLPolicyWrapper
except ModuleNotFoundError:
    from config import AssignmentConfig
    from rlhf_trainer import VERLPolicyWrapper

try:
    from minitorch.ops import fused_log_prob_gather, group_advantage_norm, fused_grpo_objective
    _HAS_MINITORCH = True
except ImportError:
    _HAS_MINITORCH = False

logger = logging.getLogger(__name__)


@dataclass
class GRPORolloutBatch:
    prompts: List[str]
    responses: List[str]
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor
    full_seq: torch.Tensor
    full_attn_mask: torch.Tensor
    prompt_len: int


@dataclass
class GRPOTrainingMetrics:
    policy_loss: float
    entropy: float
    kl_divergence: float
    reward_mean: float
    reward_std: float
    advantage_mean: float
    advantage_std: float


class GRPOTrainer:
    """GRPO trainer with group-relative advantages and optional minitorch kernels."""

    def __init__(
        self,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_model: nn.Module,
        config: AssignmentConfig,
        device: torch.device,
        use_minitorch: bool = False,
    ):
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.uses_grpo = True
        self.use_minitorch = use_minitorch and _HAS_MINITORCH

        if self.use_minitorch:
            logger.info("Using minitorch CUDA kernels for GRPO ops")

        self.policy = VERLPolicyWrapper(policy_model, tokenizer, copy_model=False)
        self.ref_policy = VERLPolicyWrapper(policy_model, tokenizer)
        self.reward_model = reward_model

        self.policy.to(device)
        self.ref_policy.to(device)
        self.reward_model.to(device)

        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        if getattr(config.system, "ref_policy_dtype", "float32") == "float16":
            self.ref_policy.half()
            logger.info("Reference policy cast to float16")
        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad_(False)

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.training.grpo_learning_rate,
            weight_decay=0.01,
        )

    # ── helpers ─────────────────────────────────────────────────────

    def _duplicate_prompts(self, prompts: List[str]) -> List[str]:
        grouped = []
        for p in prompts:
            grouped.extend([p] * self.config.grpo.group_size)
        return grouped

    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.use_minitorch:
            return group_advantage_norm(rewards, self.config.grpo.advantage_eps)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        norm = (rewards - mean) / (std + self.config.grpo.advantage_eps)
        return torch.where(std > 0, norm, torch.zeros_like(rewards))

    def _expand_token_advantages(
        self, advantages: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        flat = advantages.reshape(-1)
        return flat.unsqueeze(1).expand(-1, mask.shape[1])[mask]

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=-1)
        return -(p * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _compute_policy_objective(
        self,
        new_lp: torch.Tensor,
        old_lp: torch.Tensor,
        ref_lp: torch.Tensor,
        token_adv: torch.Tensor,
        masked_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_minitorch:
            policy_loss, kl_div, total_loss = fused_grpo_objective(
                new_lp, old_lp, ref_lp, token_adv,
                self.config.grpo.clip_eps, self.config.grpo.kl_penalty,
            )
            entropy = self._compute_entropy(masked_logits)
            total_loss = total_loss - self.config.training.grpo_entropy_coef * entropy
            return policy_loss, entropy, kl_div, total_loss

        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - self.config.grpo.clip_eps, 1.0 + self.config.grpo.clip_eps)
        policy_loss = -torch.min(ratio * token_adv, clipped * token_adv).mean()

        entropy = self._compute_entropy(masked_logits)

        log_r = new_lp - ref_lp
        kl_div = ((torch.exp(log_r) - 1) - log_r).mean()

        total_loss = (
            policy_loss
            - self.config.training.grpo_entropy_coef * entropy
            + self.config.grpo.kl_penalty * kl_div
        )
        return policy_loss, entropy, kl_div, total_loss

    def _fused_log_probs(
        self, ids: torch.Tensor, scores: torch.Tensor, prompt_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log probs using fused_log_prob_gather kernel.

        Returns (flat_log_probs, mask) matching _compute_log_probs_tensor output.
        """
        gen_ids = ids[:, prompt_len:]
        all_lp = fused_log_prob_gather(scores, gen_ids)
        B, T = all_lp.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
        lp_list = []
        for i in range(B):
            length = 0
            for j in range(T):
                tid = gen_ids[i, j].item()
                mask[i, j] = True
                length += 1
                if tid == self.tokenizer.eos_token_id:
                    break
                if (self.tokenizer.pad_token_id != self.tokenizer.eos_token_id
                        and tid == self.tokenizer.pad_token_id):
                    break
            lp_list.append(all_lp[i, :length])
        return torch.cat(lp_list, dim=0), mask

    # ── generation lengths for old_log_probs slicing ───────────────

    def _compute_gen_lengths(self, ids: torch.Tensor, prompt_len: int) -> List[int]:
        bs = ids.shape[0]
        lengths = []
        for i in range(bs):
            gen = ids[i, prompt_len:]
            length = 0
            for j in range(gen.shape[0]):
                tid = gen[j].item()
                length += 1
                if tid == self.tokenizer.eos_token_id:
                    break
                if (self.tokenizer.pad_token_id != self.tokenizer.eos_token_id
                        and tid == self.tokenizer.pad_token_id):
                    break
            lengths.append(length)
        return lengths

    # ── rollout & train ────────────────────────────────────────────

    def generate_rollouts(self, prompts: List[str]) -> GRPORolloutBatch:
        grouped = self._duplicate_prompts(prompts)
        responses, old_lp_list, prompt_len, full_seq, full_mask = self.policy.generate(
            prompts=grouped,
            max_length=self.config.grpo.rollout_max_length,
            temperature=self.config.grpo.temperature,
            top_p=self.config.grpo.top_p,
            do_sample=True,
        )

        full_texts = [f"{p} {r}" for p, r in zip(grouped, responses)]
        rewards = torch.tensor(
            self.reward_model.get_rewards(full_texts, self.reward_model.tokenizer, self.device),
            device=self.device, dtype=torch.float32,
        ).view(len(prompts), self.config.grpo.group_size)

        if self.config.grpo.reward_clip > 0:
            rewards = rewards.clamp(-self.config.grpo.reward_clip, self.config.grpo.reward_clip)

        return GRPORolloutBatch(
            prompts=prompts, responses=responses,
            rewards=rewards, advantages=self._compute_group_advantages(rewards),
            old_log_probs=torch.cat(old_lp_list, dim=0),
            full_seq=full_seq, full_attn_mask=full_mask, prompt_len=prompt_len,
        )

    def train_step(
        self,
        rollout_batch: GRPORolloutBatch,
        mini_batch_size: int = 6,
        loss_scale: float = 1.0,
        step_optimizer: bool = True,
    ) -> GRPOTrainingMetrics:
        """Train on a rollout batch with gradient accumulation over mini-batches.

        Args:
            mini_batch_size: Sequences per forward/backward pass (memory control).
            loss_scale: Scale factor (use 1/grad_accum_steps for outer accumulation).
            step_optimizer: If False, skip zero_grad/clip/step (outer accumulation).
        """
        self.policy.train()
        total_pl, total_ent, total_kl = 0.0, 0.0, 0.0

        ids = rollout_batch.full_seq
        pl = rollout_batch.prompt_len
        attn = rollout_batch.full_attn_mask
        bs = ids.shape[0]
        n_mb = max(1, (bs + mini_batch_size - 1) // mini_batch_size)

        gen_lengths = self._compute_gen_lengths(ids, pl)
        offsets = [0]
        for gl in gen_lengths:
            offsets.append(offsets[-1] + gl)

        for _ in range(self.config.grpo.update_epochs):
            if step_optimizer:
                self.policy_optimizer.zero_grad()
            ep_pl, ep_ent, ep_kl = 0.0, 0.0, 0.0

            for mb_s in range(0, bs, mini_batch_size):
                mb_e = min(mb_s + mini_batch_size, bs)
                mb_ids, mb_attn = ids[mb_s:mb_e], attn[mb_s:mb_e]

                out = self.policy(input_ids=mb_ids, attention_mask=mb_attn)
                scores = out.logits[:, pl - 1:-1, :]

                if self.use_minitorch:
                    new_lp, new_lp_mask = self._fused_log_probs(mb_ids, scores, pl)
                else:
                    new_lp_list, new_lp_mask = self.policy._compute_log_probs_tensor(mb_ids, scores, pl)
                    new_lp = torch.cat(new_lp_list, dim=0)

                with torch.no_grad():
                    ref_out = self.ref_policy(input_ids=mb_ids, attention_mask=mb_attn)
                    ref_scores = ref_out.logits[:, pl - 1:-1, :]
                    if self.use_minitorch:
                        ref_lp, _ = self._fused_log_probs(mb_ids, ref_scores, pl)
                    else:
                        ref_lp_list, _ = self.ref_policy._compute_log_probs_tensor(mb_ids, ref_scores, pl)
                        ref_lp = torch.cat(ref_lp_list, dim=0)
                    del ref_out, ref_scores

                old_lp = rollout_batch.old_log_probs[offsets[mb_s]:offsets[mb_e]]
                mb_adv = rollout_batch.advantages.reshape(-1)[mb_s:mb_e]
                token_adv = self._expand_token_advantages(mb_adv, new_lp_mask)

                p_loss, ent, kl, total = self._compute_policy_objective(
                    new_lp, old_lp, ref_lp, token_adv, scores[new_lp_mask],
                )
                (total * loss_scale / n_mb).backward()

                ep_pl += p_loss.item()
                ep_ent += ent.item()
                ep_kl += kl.item()
                del out, scores, new_lp, ref_lp
                torch.cuda.empty_cache()

            if step_optimizer:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.training.grpo_max_grad_norm,
                )
                self.policy_optimizer.step()

            total_pl += ep_pl / n_mb
            total_ent += ep_ent / n_mb
            total_kl += ep_kl / n_mb

        n = self.config.grpo.update_epochs
        return GRPOTrainingMetrics(
            policy_loss=total_pl / n, entropy=total_ent / n, kl_divergence=total_kl / n,
            reward_mean=rollout_batch.rewards.mean().item(),
            reward_std=rollout_batch.rewards.std(unbiased=False).item(),
            advantage_mean=rollout_batch.advantages.mean().item(),
            advantage_std=rollout_batch.advantages.std(unbiased=False).item(),
        )

    # ── checkpointing ──────────────────────────────────────────────

    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        torch.save({
            "epoch": epoch,
            "policy_state_dict": self.policy.model.state_dict(),
            "ref_policy_state_dict": self.ref_policy.model.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.policy.model.load_state_dict(ckpt["policy_state_dict"])
        self.ref_policy.model.load_state_dict(ckpt["ref_policy_state_dict"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return ckpt


def create_grpo_trainer(
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reward_model: nn.Module,
    config: AssignmentConfig,
    device: torch.device,
    use_minitorch: bool = False,
) -> GRPOTrainer:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.pad_token_id
    trainer = GRPOTrainer(policy_model, tokenizer, reward_model, config, device, use_minitorch)
    logger.info(f"Created GRPO trainer ({sum(p.numel() for p in policy_model.parameters()):,} params)")
    return trainer
