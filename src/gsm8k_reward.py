"""
Rule-based reward function for GSM8K math problems.
Implements the same interface as RewardModel so it can be used as a drop-in
replacement in the GRPO trainer.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GSM8KRewardFunction(nn.Module):
    """
    Rule-based reward function for GSM8K.

    Scores responses by extracting the predicted numeric answer after "####"
    and comparing it against the ground truth.

    Rewards:
      - correct_reward (default 1.0) if the extracted answer matches ground truth
      - format_reward  (default 0.1) if "####" is present but the answer is wrong
      - incorrect_reward (default 0.0) otherwise
    """

    def __init__(
        self,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        incorrect_reward: float = 0.0,
        answer_delimiter: str = "####",
    ):
        super().__init__()
        self.correct_reward = correct_reward
        self.format_reward = format_reward
        self.incorrect_reward = incorrect_reward
        self.answer_delimiter = answer_delimiter

        self.tokenizer = None
        self._prompt_to_answer: Dict[str, str] = {}
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def set_prompt_answers(self, prompt_answer_pairs: List[Dict[str, str]]) -> None:
        """
        Load ground truth answers keyed by prompt text.

        Args:
            prompt_answer_pairs: List of dicts with "prompt" and "answer" keys.
        """
        self._prompt_to_answer = {}
        for item in prompt_answer_pairs:
            prompt_key = item["prompt"].strip()
            self._prompt_to_answer[prompt_key] = self._normalize_answer(item["answer"])
        logger.info(f"Loaded {len(self._prompt_to_answer)} prompt-answer pairs")

    @staticmethod
    def _normalize_answer(answer_str: str) -> str:
        """Normalize a numeric answer: remove commas, whitespace, trailing periods."""
        answer_str = answer_str.strip().replace(",", "").rstrip(".")
        return answer_str.strip()

    @staticmethod
    def extract_answer_from_response(response: str, delimiter: str = "####") -> Optional[str]:
        """
        Extract the numeric answer after the delimiter from a model response.

        Returns None if the delimiter is not found or no valid number follows it.
        """
        if delimiter not in response:
            return None
        after_marker = response.split(delimiter)[1].strip()
        tokens = after_marker.split()
        if not tokens:
            return None
        candidate = tokens[0].replace(",", "").rstrip(".")
        try:
            float(candidate)
            return candidate
        except ValueError:
            return None

    def _find_ground_truth(self, full_text: str) -> Optional[str]:
        """
        Match a full_text (prompt + response) back to its prompt to find ground truth.

        The GRPO trainer constructs full_text as f"{prompt} {response}", so the
        prompt is a prefix of full_text.
        """
        for prompt_key, answer in self._prompt_to_answer.items():
            if full_text.startswith(prompt_key):
                return answer
        full_stripped = full_text.strip()
        for prompt_key, answer in self._prompt_to_answer.items():
            if full_stripped.startswith(prompt_key.strip()):
                return answer
        return None

    def get_rewards(
        self,
        texts: List[str],
        tokenizer=None,
        device=None,
        batch_size: int = 8,
    ) -> List[float]:
        """
        Score a list of texts (prompt + response).

        Args:
            texts: List of full text strings (prompt + response).
            tokenizer: Unused, kept for interface compatibility.
            device: Unused, kept for interface compatibility.
            batch_size: Unused, kept for interface compatibility.

        Returns:
            List of reward scores.
        """
        rewards = []
        for text in texts:
            ground_truth = self._find_ground_truth(text)
            if ground_truth is None:
                logger.debug(f"No ground truth found for text: {text[:80]}...")
                rewards.append(self.incorrect_reward)
                continue

            predicted = self.extract_answer_from_response(text, self.answer_delimiter)
            if predicted is not None and self._normalize_answer(predicted) == ground_truth:
                rewards.append(self.correct_reward)
            elif self.answer_delimiter in text:
                rewards.append(self.format_reward)
            else:
                rewards.append(self.incorrect_reward)
        return rewards

    def forward(self, *args, **kwargs):
        """Not used — rewards are computed via get_rewards()."""
        raise NotImplementedError("Use get_rewards() instead of forward()")
