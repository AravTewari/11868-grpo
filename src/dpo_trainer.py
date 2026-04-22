"""
DPO trainer implementation for Assignment 7.
This module contains a Direct Preference Optimization trainer that reuses the
shared policy wrapper and optimizes on preference pairs directly.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import logging

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from src.config import AssignmentConfig
    from src.rlhf_trainer import VERLPolicyWrapper
except ModuleNotFoundError:
    from config import AssignmentConfig
    from rlhf_trainer import VERLPolicyWrapper

logger = logging.getLogger(__name__)


@dataclass
class DPOBatch:
    """Tokenized preference batch for DPO updates."""

    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_completion_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_completion_mask: torch.Tensor


@dataclass
class DPOTrainingMetrics:
    """Training metrics for DPO updates."""

    loss: float
    accuracy: float
    margin: float
    chosen_logp: float
    rejected_logp: float


class DPOTrainer:
    """
    Main DPO trainer.

    DPO optimizes a trainable policy against a frozen reference policy using
    preference pairs of (prompt, chosen, rejected).
    """

    def __init__(
        self,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: AssignmentConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.uses_dpo = True
        self.max_length = self.config.data.max_prompt_length + self.config.data.max_response_length

        self.policy = VERLPolicyWrapper(policy_model, tokenizer, copy_model=False)
        self.ref_policy = VERLPolicyWrapper(policy_model, tokenizer)

        self.policy.to(device)
        self.ref_policy.to(device)

        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)
        if getattr(config.system, "ref_policy_dtype", "float32") == "float16":
            self.ref_policy.half()
            logger.info("Reference policy cast to float16 for memory savings")

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.training.dpo_learning_rate,
            weight_decay=self.config.training.dpo_weight_decay,
        )

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text without batch padding."""
        return self.tokenizer(
            [text],
            padding=False,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

    def _validate_preference_texts(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> torch.Tensor:
        """Validate prompt prefixes and return prompt token ids."""
        if not chosen.startswith(prompt):
            raise ValueError("Chosen text must begin with the prompt")
        if not rejected.startswith(prompt):
            raise ValueError("Rejected text must begin with the prompt")

        prompt_encoding = self._encode_text(prompt)
        return prompt_encoding["input_ids"][0]

    def _build_completion_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_token_ids: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a completion mask and validity flags for each example."""
        completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        valid_examples = torch.ones(input_ids.shape[0], dtype=torch.bool)

        for idx, prompt_ids in enumerate(prompt_token_ids):
            valid_positions = attention_mask[idx].nonzero(as_tuple=False).squeeze(-1)
            valid_ids = input_ids[idx, valid_positions]
            prompt_ids = prompt_ids.to(input_ids.device)
            prompt_len = prompt_ids.shape[0]

            if valid_ids.shape[0] < prompt_len:
                raise ValueError("Tokenized preference example is shorter than its prompt")

            if not torch.equal(valid_ids[:prompt_len], prompt_ids):
                raise ValueError("Chosen/rejected tokenization does not preserve the prompt prefix")

            if valid_ids.shape[0] == prompt_len:
                valid_examples[idx] = False
                continue

            completion_positions = valid_positions[prompt_len:]
            completion_mask[idx, completion_positions] = True

        return completion_mask, valid_examples

    def prepare_batch(self, preference_batch: List[Dict[str, str]]) -> DPOBatch:
        """Tokenize and validate a preference batch."""
        if not preference_batch:
            raise ValueError("Preference batch must not be empty")

        prompts = []
        chosen_texts = []
        rejected_texts = []
        prompt_token_ids = []

        for item in preference_batch:
            for key in ("prompt", "chosen", "rejected"):
                if key not in item:
                    raise ValueError(f"Preference item missing required key: {key}")

            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            prompts.append(prompt)
            chosen_texts.append(chosen)
            rejected_texts.append(rejected)
            prompt_token_ids.append(
                self._validate_preference_texts(prompt=prompt, chosen=chosen, rejected=rejected)
            )

        chosen_batch = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        rejected_batch = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        chosen_completion_mask, chosen_valid_examples = self._build_completion_mask(
            chosen_batch["input_ids"],
            chosen_batch["attention_mask"],
            prompt_token_ids,
        )
        rejected_completion_mask, rejected_valid_examples = self._build_completion_mask(
            rejected_batch["input_ids"],
            rejected_batch["attention_mask"],
            prompt_token_ids,
        )
        valid_examples = chosen_valid_examples & rejected_valid_examples
        num_skipped = int((~valid_examples).sum().item())

        if num_skipped:
            logger.warning(
                "Skipping %d DPO preference example(s) with no completion tokens after tokenization",
                num_skipped,
            )

            if not valid_examples.any():
                raise ValueError("Preference batch has no usable examples after tokenization")

            chosen_batch = {
                key: value[valid_examples]
                for key, value in chosen_batch.items()
            }
            rejected_batch = {
                key: value[valid_examples]
                for key, value in rejected_batch.items()
            }
            chosen_completion_mask = chosen_completion_mask[valid_examples]
            rejected_completion_mask = rejected_completion_mask[valid_examples]

        return DPOBatch(
            chosen_input_ids=chosen_batch["input_ids"].to(self.device),
            chosen_attention_mask=chosen_batch["attention_mask"].to(self.device),
            chosen_completion_mask=chosen_completion_mask.to(self.device),
            rejected_input_ids=rejected_batch["input_ids"].to(self.device),
            rejected_attention_mask=rejected_batch["attention_mask"].to(self.device),
            rejected_completion_mask=rejected_completion_mask.to(self.device),
        )

    def _compute_sequence_log_probs(
        self,
        model_wrapper: VERLPolicyWrapper,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute summed completion-token log probabilities for each sequence."""
        outputs = model_wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        target_mask = completion_mask[:, 1:] & attention_mask[:, 1:].bool()

        token_log_probs = torch.log_softmax(logits, dim=-1)
        gathered_log_probs = token_log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        if (target_mask.sum(dim=1) == 0).any():
            raise ValueError("DPO batch contains a sequence with no valid completion targets")

        return (gathered_log_probs * target_mask.to(gathered_log_probs.dtype)).sum(dim=1)

    def _compute_dpo_loss(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the sigmoid DPO objective with optional label smoothing."""
        preference_logits = self.config.dpo.beta * (chosen_logratios - rejected_logratios)
        positive_loss = -F.logsigmoid(preference_logits)

        if self.config.dpo.label_smoothing > 0:
            negative_loss = -F.logsigmoid(-preference_logits)
            loss_terms = (
                (1.0 - self.config.dpo.label_smoothing) * positive_loss
                + self.config.dpo.label_smoothing * negative_loss
            )
        else:
            loss_terms = positive_loss

        return loss_terms.mean(), preference_logits

    def _compute_loss_and_metrics(
        self,
        batch: DPOBatch,
    ) -> Tuple[torch.Tensor, DPOTrainingMetrics]:
        """Compute the DPO loss tensor and logging metrics for one batch."""
        chosen_policy_logp = self._compute_sequence_log_probs(
            self.policy,
            batch.chosen_input_ids,
            batch.chosen_attention_mask,
            batch.chosen_completion_mask,
        )
        rejected_policy_logp = self._compute_sequence_log_probs(
            self.policy,
            batch.rejected_input_ids,
            batch.rejected_attention_mask,
            batch.rejected_completion_mask,
        )

        with torch.no_grad():
            chosen_ref_logp = self._compute_sequence_log_probs(
                self.ref_policy,
                batch.chosen_input_ids,
                batch.chosen_attention_mask,
                batch.chosen_completion_mask,
            )
            rejected_ref_logp = self._compute_sequence_log_probs(
                self.ref_policy,
                batch.rejected_input_ids,
                batch.rejected_attention_mask,
                batch.rejected_completion_mask,
            )

        chosen_logratios = chosen_policy_logp - chosen_ref_logp
        rejected_logratios = rejected_policy_logp - rejected_ref_logp
        loss, _ = self._compute_dpo_loss(chosen_logratios, rejected_logratios)

        margin = chosen_logratios - rejected_logratios
        accuracy = (margin > 0).float().mean()
        metrics = DPOTrainingMetrics(
            loss=loss.item(),
            accuracy=accuracy.item(),
            margin=margin.mean().item(),
            chosen_logp=chosen_policy_logp.mean().item(),
            rejected_logp=rejected_policy_logp.mean().item(),
        )
        return loss, metrics

    def train_step(self, batch: DPOBatch) -> DPOTrainingMetrics:
        """Perform one DPO policy update."""
        self.policy.train()

        loss, metrics = self._compute_loss_and_metrics(batch)

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.training.dpo_max_grad_norm,
        )
        self.policy_optimizer.step()
        return metrics

    def evaluate_step(self, batch: DPOBatch) -> DPOTrainingMetrics:
        """Evaluate a preference batch without updating the policy."""
        self.policy.eval()
        with torch.no_grad():
            _, metrics = self._compute_loss_and_metrics(batch)
        self.policy.train()
        return metrics

    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        """Save a DPO checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "policy_state_dict": self.policy.model.state_dict(),
            "ref_policy_state_dict": self.ref_policy.model.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a DPO checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.model.load_state_dict(checkpoint["policy_state_dict"])
        self.ref_policy.model.load_state_dict(checkpoint["ref_policy_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def create_dpo_trainer(
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: AssignmentConfig,
    device: torch.device,
) -> DPOTrainer:
    """Create and initialize a DPO trainer from a ready policy model and tokenizer."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.pad_token_id

    trainer = DPOTrainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    logger.info("Created DPO trainer")
    logger.info(f"Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    return trainer
