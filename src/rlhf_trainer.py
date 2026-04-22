"""
PPO trainer implementation using a VERL-like framework for Assignment 7.
This module contains the main PPO training logic plus shared generation helpers.
"""

import copy
import os

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)

from src.reward_model import RewardModel
from src.config import AssignmentConfig
from src.policy_io import load_policy_model_and_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    """Data structure for storing rollout results."""
    prompts: List[str]
    responses: List[str]
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    full_seq: torch.Tensor
    full_attn_mask: torch.Tensor
    prompt_len: int


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring RLHF progress."""
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    reward_mean: float
    reward_std: float
    advantage_mean: float
    advantage_std: float


class VERLPolicyWrapper(nn.Module):
    """
    Wrapper class to make HuggingFace models compatible with VERL.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        copy_model: bool = True,
    ):
        """
        Initialize the policy wrapper.
        
        Args:
            model: HuggingFace causal language model
            tokenizer: Tokenizer for the model
            copy_model: Whether to deepcopy the model before wrapping it
        """
        super().__init__()
        self.model = copy.deepcopy(model) if copy_model else model
        self.tokenizer = tokenizer
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # turn off dropout due to effect described in https://arxiv.org/pdf/2202.11818
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        
        # Set model to training mode
        self.model.train()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Forward pass through the policy model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs
    
    def generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_log_probs: bool = True,
        **kwargs
    ) -> Tuple[List[str], Optional[List[torch.Tensor]], int, torch.Tensor, torch.Tensor]:
        """
        Generate responses to prompts and return log probabilities.
        
        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            return_log_probs: Whether to compute per-token log probabilities
            
        Returns:
            Tuple of (responses, log_probs)
        """
        self.model.eval()
        
        encoded_prompts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length // 2  # Leave space for generation
        )
        
        device = next(self.model.parameters()).device
        encoded_prompts = {k: v.to(device) for k, v in encoded_prompts.items()}
        
        # Generate responses
        with torch.no_grad():

            generated = self.model.generate(
                input_ids=encoded_prompts['input_ids'],
                attention_mask=encoded_prompts['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=return_log_probs,
                **kwargs
            )
        
        # Extract generated sequences
        
        # generated_ids[:,i + prompt_lengths] is the prediction using generated.scores[:,i]
        generated_ids = generated.sequences # [batch_size, seq_len]
        # seq_len is length of the longest sequence generated in the batch (including prompt)
        
        # prompt_lengths is the padded length of the longest prompt in the batch
        prompt_lengths = encoded_prompts['input_ids'].shape[1] 
        if return_log_probs:
            assert prompt_lengths + len(generated.scores) == generated_ids.shape[1]
        
        response_ids = generated_ids[:, prompt_lengths:]
        
        # Construct masking for generated tokens
        prompt_mask = encoded_prompts['attention_mask']
        generated_part_ids = generated_ids[:, prompt_lengths:]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # GPT-2 path: pad and EOS are the same token.
            # Keep tokens where we have seen <= 1 EOS/PAD token.
            is_eos = (generated_part_ids == self.tokenizer.eos_token_id)
            eos_cumsum = is_eos.cumsum(dim=1)
            generated_part_mask = (eos_cumsum <= 1).long()
        else:
            # Qwen3 path: pad and EOS are different tokens.
            # Keep tokens up to and including the first EOS, mask out padding.
            is_eos = (generated_part_ids == self.tokenizer.eos_token_id)
            is_pad = (generated_part_ids == self.tokenizer.pad_token_id)
            eos_cumsum = is_eos.cumsum(dim=1)
            generated_part_mask = ((eos_cumsum <= 1) & ~is_pad).long()
            # Ensure the first EOS token itself is kept
            first_eos_mask = (eos_cumsum == 1) & is_eos
            generated_part_mask = generated_part_mask | first_eos_mask.long()

        full_attention_mask = torch.cat([prompt_mask, generated_part_mask], dim=1)
        
        responses = self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True, # padding token etc. are stripped
            clean_up_tokenization_spaces=True # fix spacing
        )

        if not return_log_probs:
            self.model.train()
            return responses, None, prompt_lengths, generated_ids, full_attention_mask

        log_probs = []
        fwd_mb_size = 8
        batch_size = generated_ids.shape[0]
        with torch.no_grad():
            for mb_start in range(0, batch_size, fwd_mb_size):
                mb_end = min(mb_start + fwd_mb_size, batch_size)
                mb_out = self.model(
                    input_ids=generated_ids[mb_start:mb_end],
                    attention_mask=full_attention_mask[mb_start:mb_end],
                )
                mb_scores = mb_out.logits[:, prompt_lengths - 1:-1, :]
                mb_lp, _ = self._compute_log_probs_tensor(
                    generated_ids[mb_start:mb_end], mb_scores, prompt_lengths,
                )
                log_probs.extend(mb_lp)
                del mb_out, mb_scores
                torch.cuda.empty_cache()

        self.model.train()
        return responses, log_probs, prompt_lengths, generated_ids, full_attention_mask
    
    
    def _compute_log_probs_tensor(
        self,
        generated_ids: torch.Tensor, # [batch_size, seq_len]
        scores: torch.Tensor,  # [batch_size, gen_len, vocab_size]
        prompt_lengths: int, # seq_len - gen_len
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        seq_len = generated_ids.shape[1]
        gen_len = scores.shape[1]
        assert seq_len == gen_len + prompt_lengths
        # Note: pad_token_id may differ from eos_token_id (e.g. Qwen3)
        
        
        # Memory-efficient: gather + logsumexp avoids materializing full [B, T, V] log_softmax
        gen_token_ids = generated_ids[:, prompt_lengths:]
        log_normalizer = torch.logsumexp(scores, dim=-1)
        token_logits = scores.gather(2, gen_token_ids.unsqueeze(-1)).squeeze(-1)
        all_log_probs = token_logits - log_normalizer

        batch_size = generated_ids.shape[0]
        gen_len = scores.shape[1]
        mask = torch.zeros(batch_size, gen_len, dtype=torch.bool, device=scores.device)
        generated_log_probs = []

        for i in range(batch_size):
            sequence_log_probs = []
            for j in range(gen_len):
                token_id = generated_ids[i, j + prompt_lengths].item()
                mask[i, j] = True
                sequence_log_probs.append(all_log_probs[i, j])
                if token_id == self.tokenizer.eos_token_id:
                    break
                if (self.tokenizer.pad_token_id != self.tokenizer.eos_token_id
                        and token_id == self.tokenizer.pad_token_id):
                    break
            generated_log_probs.append(torch.stack(sequence_log_probs))

        return generated_log_probs, mask
        
    
class VERLValueWrapper(nn.Module):
    """
    Value model wrapper for VERL integration.
    """
    
    def __init__(self, policy_model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize value model based on policy model.
        
        Args:
            policy_model: Base policy model
            tokenizer: Tokenizer for the model
        """
        super().__init__()
        self.tokenizer = tokenizer
        
        # Create value head on top of the policy model
        self.backbone = copy.deepcopy(policy_model)
        
        self.value_head = nn.Linear(policy_model.config.hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Forward pass to compute state values."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Use last hidden state for value prediction
        last_hidden_state = outputs.hidden_states[-1]
        
        # Compute value
        values = self.value_head(last_hidden_state).squeeze(-1)
        
        return values


class VERLTrainer:
    """
    Main RLHF trainer using VERL framework.
    """
    
    def __init__(
        self,
        policy_model: PreTrainedModel,
        reward_model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        config: AssignmentConfig,
        device: torch.device
    ):
        """
        Initialize VERL RLHF trainer.
        
        Args:
            policy_model: Policy model to train
            reward_model: Trained reward model
            tokenizer: Tokenizer for text processing
            config: Training configuration
            device: Training device
        """
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        # Wrap models for VERL compatibility
        self.policy = VERLPolicyWrapper(policy_model, tokenizer, copy_model=False)
        self.ref_policy = VERLPolicyWrapper(policy_model, tokenizer)
        self.value_model = VERLValueWrapper(policy_model, tokenizer)
        self.reward_model = reward_model
        
        # Move models to device
        self.policy.to(device)
        self.ref_policy.to(device)
        self.value_model.to(device)
        self.reward_model.to(device)

        # Set reward model to eval mode
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        if getattr(config.system, "ref_policy_dtype", "float32") == "float16":
            self.ref_policy.half()
            logger.info("Reference policy cast to float16 for memory savings")
        self.reward_model.eval()
        
        # Create PPO configuration
        self.ppo_config = self._create_ppo_config()
        
        self._init_custom_optimizers()
    
    def _create_ppo_config(self) -> Dict[str, Any]:
        """Create PPO configuration for VERL."""
        return {
            'learning_rate': self.config.training.ppo_learning_rate,
            'batch_size': self.config.verl.train_batch_size,
            'mini_batch_size': self.config.verl.train_mini_batch_size,
            'ppo_epochs': self.config.verl.ppo_epochs,
            'clip_eps': self.config.verl.ppo_clip_eps,
            'target_kl': self.config.verl.ppo_target_kl,
            'gamma': self.config.training.ppo_gamma,
            'gae_lambda': self.config.training.ppo_gae_lambda,
            'value_coef': self.config.training.ppo_value_coef,
            'entropy_coef': self.config.training.ppo_entropy_coef,
            'max_grad_norm': self.config.training.ppo_max_grad_norm,
        }
    
    def _init_custom_optimizers(self):
        """Initialize custom optimizers for fallback implementation."""

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.training.ppo_learning_rate,
            weight_decay=0.01
        )
        
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=self.config.training.ppo_learning_rate * 2,  # Higher LR for value
            weight_decay=0.01
        )
    
    def generate_rollouts(self, prompts: List[str]) -> RolloutBatch:
        """
        Generate rollouts using the current policy.
        
        Args:
            prompts: List of prompts to generate responses for
            
        Returns:
            RolloutBatch containing rollout data
        """
        # Generate responses
        responses, log_probs, prompt_len, full_seq, full_attention_mask = self.policy.generate(
            prompts=prompts,
            max_length=self.config.verl.rollout_max_length,
            temperature=self.config.verl.rollout_temperature,
            top_p=self.config.verl.rollout_top_p,
            do_sample=True
        )
        
        full_texts = [f"{prompt} {response}" for prompt, response in zip(prompts, responses)]
        rewards = torch.tensor(
            self.reward_model.get_rewards(full_texts, self.reward_model.tokenizer, self.device),
            device=self.device,
            dtype=torch.float32
        )
        
        # Apply reward clipping if configured
        if hasattr(self.config.verl, 'reward_clip') and self.config.verl.reward_clip > 0:
            rewards = torch.clamp(rewards, -self.config.verl.reward_clip, self.config.verl.reward_clip)
        
        # Normalize rewards if configured
        if getattr(self.config.verl, 'reward_normalize', False):
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get values from value model        
        with torch.no_grad():
            values = self.value_model(
                input_ids=full_seq,
                attention_mask=full_attention_mask
            )
            values[~full_attention_mask.bool()] = 0.0
        
        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(
            rewards, 
            values, 
            full_attention_mask
        )
        
        return RolloutBatch(
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
            full_seq=full_seq,
            prompt_len=prompt_len,
            full_attn_mask=full_attention_mask,
        )
    
    def _compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward values
            values: State values
            
        Returns:
            Tuple of (advantages, returns)
        """
        # BEGIN ASSIGN7_2_1
        # TODO: Implement GAE computation.
        # HINT: For this assignment, think of each token as a step in a multi-step episode (like time steps in a trajectory).
        # For each sequence in the batch:
        #   - Create an expanded reward tensor (same shape as values), where only the last token position (according to attention_mask) has the sequence reward, and all earlier tokens are zero.
        #   - For each token position, compute returns as: returns[i] = reward[i + 1] + gamma * value[i + 1]
        #   - Compute advantage for each token as: advantage[i] = returns[i] - value[i].
        # Only compute these where the attention_mask is 1 (i.e., for valid tokens, not padding).
        # END ASSIGN7_2_1
        rewards_expanded = torch.zeros_like(values)
        valid_lengths = attention_mask.sum(dim=1).long()

        for batch_idx, seq_len in enumerate(valid_lengths.tolist()):
            if seq_len > 0:
                rewards_expanded[batch_idx, seq_len - 1] = rewards[batch_idx]

        next_rewards = torch.zeros_like(values)
        next_values = torch.zeros_like(values)
        next_rewards[:, :-1] = rewards_expanded[:, 1:]
        next_values[:, :-1] = values[:, 1:]

        returns = next_rewards + self.config.training.ppo_gamma * next_values
        advantages = returns - values

        valid_mask = attention_mask.bool()
        returns = returns * valid_mask
        advantages = advantages * valid_mask
        
        # END ASSIGN7_2_1
        
        return advantages, returns
    
    def train_step(self, rollout_batch: RolloutBatch) -> TrainingMetrics:
        """
        Perform one PPO training step.
        
        Args:
            rollout_batch: Batch of rollout data
            
        Returns:
            Training metrics
        """
        return self._train_step_custom(rollout_batch)
    
    def _train_step_custom(self, rollout_batch: RolloutBatch) -> TrainingMetrics:
        """Custom PPO training step implementation."""

        self.policy.train()
        self.value_model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
    
        
        input_ids = rollout_batch.full_seq
        prompt_len = rollout_batch.prompt_len
        full_attn_mask = rollout_batch.full_attn_mask
        
        # PPO training epochs
        for epoch in range(self.config.verl.ppo_epochs):
            # Forward pass through policy
            policy_outputs = self.policy(
                input_ids=input_ids,
                attention_mask=full_attn_mask,
            )
            # scores is [batch_size, gen_len, vocab_size]
            # scores[:,0,x] is the probability score of token x at 0th generated token
            scores = policy_outputs.logits[:, prompt_len - 1 : -1, :]
            
            # Compute new log probabilities
            new_log_probs, new_log_probs_mask = self.policy._compute_log_probs_tensor(
                input_ids, scores, prompt_len
            )
            new_log_probs = torch.cat(new_log_probs, dim=0)
            old_log_probs = torch.cat(rollout_batch.log_probs, dim=0)
            with torch.no_grad():
                ref_policy_outputs = self.ref_policy(
                    input_ids=input_ids,
                    attention_mask=full_attn_mask,
                )
                ref_scores = ref_policy_outputs.logits[:, prompt_len - 1 : -1, :]
                ref_log_probs, ref_log_probs_mask = self.ref_policy._compute_log_probs_tensor(
                    input_ids, ref_scores, prompt_len
                )
                ref_log_probs = torch.cat(ref_log_probs, dim=0)
            
            advantages = rollout_batch.advantages[:, prompt_len - 1 : -1][new_log_probs_mask]
            
            # BEGIN ASSIGN7_2_2
            # TODO: Compute PPO loss
            # 1. Compute probability ratio: exp(new_log_probs - old_log_probs)
            # 2. Compute surrogate losses:
            #    - surr1 = ratio * advantages
            #    - surr2 = clipped_ratio * advantages (clip ratio between 1-eps and 1+eps)
            # 3. Policy loss = -min(surr1, surr2).mean()
            # 4. Compute entropy bonus from policy logits
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.config.verl.ppo_clip_eps,
                1.0 + self.config.verl.ppo_clip_eps
            )
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = self._compute_entropy(scores[new_log_probs_mask])
            
            # END ASSIGN7_2_2

            # Compute KL divergence for monitoring
            # k3 described in http://joschu.net/blog/kl-approx.html
            log_ratio = new_log_probs - ref_log_probs
            approx_kl = (torch.exp(log_ratio) - 1) - log_ratio
            kl_div = approx_kl.mean()
            
            # Total policy loss with entropy bonus
            total_policy_loss_step = (
                policy_loss - 
                self.config.training.ppo_entropy_coef * entropy +
                self.config.training.ppo_kl_penalty * kl_div
            )
            
            # Value function training
            values = self.value_model(
                input_ids=input_ids,
                attention_mask=full_attn_mask
            )
            values = values[:, prompt_len - 1 :]
            values = values[full_attn_mask[:, prompt_len - 1 :].bool()]
            returns = rollout_batch.returns[:, prompt_len - 1 :]
            returns = returns[full_attn_mask[:, prompt_len - 1 :].bool()]
            value_loss = nn.MSELoss()(values, returns)            
            
            # Backward pass for policy
            self.policy_optimizer.zero_grad()
            total_policy_loss_step.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.config.training.ppo_max_grad_norm
            )
            self.policy_optimizer.step()
            
            # Backward pass for value function
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(),
                self.config.training.ppo_max_grad_norm
            )
            self.value_optimizer.step()
            
            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl_div.item()
        
        num_epochs = self.config.verl.ppo_epochs
        return TrainingMetrics(
            policy_loss=total_policy_loss / num_epochs,
            value_loss=total_value_loss / num_epochs,
            entropy=total_entropy / num_epochs,
            kl_divergence=total_kl / num_epochs,
            reward_mean=rollout_batch.rewards.mean().item(),
            reward_std=rollout_batch.rewards.std(unbiased=False).item(),
            advantage_mean=rollout_batch.advantages.mean().item(),
            advantage_std=rollout_batch.advantages.std(unbiased=False).item()
        )
    
    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the policy distribution.
        
        Args:
            logits: Model logits
            
        Returns:
            Entropy values
        """
        # BEGIN ASSIGN7_2_3
        # TODO: Compute entropy from logits
        # 1. Convert logits to probabilities using softmax
        # 2. Convert logits to log_probabilities using log_softmax
        # 3. Compute entropy: -(probs * log_probs).sum(dim=-1)
        # 4. Return mean over sequence length
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()
        
        # END ASSIGN7_2_3
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Training metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.model.state_dict(),
            'value_state_dict': self.value_model.state_dict(),
            'policy_optimizer_state_dict': getattr(self, 'policy_optimizer', {}).state_dict() if hasattr(self, 'policy_optimizer') else {},
            'value_optimizer_state_dict': getattr(self, 'value_optimizer', {}).state_dict() if hasattr(self, 'value_optimizer') else {},
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.policy.model.load_state_dict(checkpoint['policy_state_dict'])
        self.value_model.load_state_dict(checkpoint['value_state_dict'])
        
        # Load optimizer states if available
        if hasattr(self, 'policy_optimizer') and 'policy_optimizer_state_dict' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if hasattr(self, 'value_optimizer') and 'value_optimizer_state_dict' in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def create_ppo_trainer(
    model_name: str,
    reward_model: RewardModel,
    config: AssignmentConfig,
    device: torch.device
) -> VERLTrainer:
    """
    Create and initialize PPO trainer.
    
    Args:
        model_name: Name of the base language model
        reward_model: Trained reward model
        config: Training configuration
        device: Training device
        
    Returns:
        Initialized VERLTrainer
    """
    policy_model, tokenizer = load_policy_model_and_tokenizer(
        model_name,
        config=config,
        device=device,
    )
    
    # Create trainer
    trainer = VERLTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    
    logger.info(f"Created PPO trainer with model: {model_name}")
    logger.info(f"Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    
    return trainer


def create_rlhf_trainer(
    model_name: str,
    reward_model: RewardModel,
    config: AssignmentConfig,
    device: torch.device
) -> VERLTrainer:
    """Compatibility alias for older callers that still import create_rlhf_trainer."""
    return create_ppo_trainer(
        model_name=model_name,
        reward_model=reward_model,
        config=config,
        device=device,
    )


def evaluate_policy(
    trainer: VERLTrainer,
    eval_prompts: List[str],
    num_samples: int = 5,
    reward_model: Optional[RewardModel] = None,
    generation_config: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Evaluate the policy on a set of prompts.
    
    Args:
        trainer: RLHF trainer with trained policy
        eval_prompts: List of evaluation prompts
        num_samples: Number of samples to generate per prompt
        reward_model: Optional override for the reward model used in evaluation
        generation_config: Optional override for generation hyperparameters
        
    Returns:
        Evaluation metrics
    """
    trainer.policy.model.eval()
    eval_reward_model = reward_model if reward_model is not None else trainer.reward_model
    
    all_rewards = []
    all_response_lengths = []
    
    with torch.no_grad():
        rollout_max_length = trainer.config.verl.rollout_max_length
        rollout_temperature = trainer.config.verl.rollout_temperature
        rollout_top_p = trainer.config.verl.rollout_top_p
        if getattr(trainer, "uses_grpo", False):
            rollout_max_length = trainer.config.grpo.rollout_max_length
            rollout_temperature = trainer.config.grpo.temperature
            rollout_top_p = trainer.config.grpo.top_p
        if generation_config is not None:
            rollout_max_length = generation_config.get("max_length", rollout_max_length)
            rollout_temperature = generation_config.get("temperature", rollout_temperature)
            rollout_top_p = generation_config.get("top_p", rollout_top_p)

        pbar = tqdm(
            range(0, len(eval_prompts), trainer.config.experiment.eval_batch_size), 
            desc="Evaluating Policy", 
            unit="prompt",
            disable=os.environ.get("TQDM_DISABLE") == "1",
        )
        for batch_idx in pbar:
            start_idx = batch_idx
            end_idx = min(start_idx + trainer.config.experiment.eval_batch_size, len(eval_prompts))
            batch_prompts = eval_prompts[start_idx:end_idx]

            batch_prompts_duplicated = batch_prompts * num_samples
            
            # Generate multiple samples per prompt
            responses,_, _, _, _ = trainer.policy.generate(
                prompts=batch_prompts_duplicated,
                max_length=rollout_max_length,
                temperature=rollout_temperature,
                top_p=rollout_top_p,
                do_sample=True,
                return_log_probs=False,
            )
            
            # Get rewards for generated responses
            full_texts = [
                f"{prompt} {response}"
                for prompt, response in zip(batch_prompts_duplicated, responses)
            ]
            rewards = eval_reward_model.get_rewards(
                full_texts, 
                eval_reward_model.tokenizer, 
                trainer.device
            )
            
            all_rewards.extend(rewards)
            all_response_lengths.extend([len(response.split()) for response in responses])
            
            current_mean_reward = sum(all_rewards) / len(all_rewards)
            pbar.set_postfix({
                "avg_reward": f"{current_mean_reward:.3f}", 
                "total_samples": len(all_rewards)
            })
    
    trainer.policy.model.train()
    
    return {
        'mean_reward': sum(all_rewards) / len(all_rewards),
        'std_reward': torch.tensor(all_rewards).std(unbiased=False).item(),
        'min_reward': min(all_rewards),
        'max_reward': max(all_rewards),
        'mean_response_length': sum(all_response_lengths) / len(all_response_lengths),
        'num_evaluations': len(all_rewards)
    }
