"""
Test cases for the GSM8K rule-based reward function.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from gsm8k_reward import GSM8KRewardFunction


@pytest.fixture
def reward_fn():
    """Create a GSM8KRewardFunction with sample prompt-answer pairs."""
    fn = GSM8KRewardFunction(
        correct_reward=1.0,
        format_reward=0.1,
        incorrect_reward=0.0,
    )
    fn.set_prompt_answers([
        {"prompt": "Question: What is 2+2?\nAnswer: Let's think step by step.\n", "answer": "4"},
        {"prompt": "Question: What is 10*5?\nAnswer: Let's think step by step.\n", "answer": "50"},
        {"prompt": "Question: How many apples?\nAnswer: Let's think step by step.\n", "answer": "1234"},
    ])
    return fn


class TestNormalizeAnswer:
    def test_simple_number(self):
        assert GSM8KRewardFunction._normalize_answer("42") == "42"

    def test_commas_removed(self):
        assert GSM8KRewardFunction._normalize_answer("1,234") == "1234"

    def test_trailing_period_removed(self):
        assert GSM8KRewardFunction._normalize_answer("42.") == "42"

    def test_whitespace_stripped(self):
        assert GSM8KRewardFunction._normalize_answer("  42  ") == "42"

    def test_negative_number(self):
        assert GSM8KRewardFunction._normalize_answer("-7") == "-7"

    def test_decimal(self):
        assert GSM8KRewardFunction._normalize_answer("3.14") == "3.14"

    def test_large_number_with_commas(self):
        assert GSM8KRewardFunction._normalize_answer("1,000,000") == "1000000"


class TestExtractAnswer:
    def test_standard_format(self):
        assert GSM8KRewardFunction.extract_answer_from_response("So the answer is #### 42") == "42"

    def test_no_space_after_delimiter(self):
        assert GSM8KRewardFunction.extract_answer_from_response("####42") == "42"

    def test_comma_in_answer(self):
        assert GSM8KRewardFunction.extract_answer_from_response("#### 1,234") == "1234"

    def test_negative_answer(self):
        assert GSM8KRewardFunction.extract_answer_from_response("#### -5") == "-5"

    def test_no_delimiter(self):
        assert GSM8KRewardFunction.extract_answer_from_response("The answer is 42") is None

    def test_delimiter_with_no_number(self):
        assert GSM8KRewardFunction.extract_answer_from_response("####") is None

    def test_delimiter_with_non_numeric(self):
        assert GSM8KRewardFunction.extract_answer_from_response("#### abc") is None

    def test_multiple_delimiters_takes_first(self):
        result = GSM8KRewardFunction.extract_answer_from_response("#### 10 then #### 20")
        assert result == "10"

    def test_trailing_text_after_answer(self):
        result = GSM8KRewardFunction.extract_answer_from_response("#### 42 dollars")
        assert result == "42"


class TestGetRewards:
    def test_correct_answer(self, reward_fn):
        texts = ["Question: What is 2+2?\nAnswer: Let's think step by step.\n 2+2=4 #### 4"]
        rewards = reward_fn.get_rewards(texts)
        assert rewards == [1.0]

    def test_wrong_answer_with_format(self, reward_fn):
        texts = ["Question: What is 2+2?\nAnswer: Let's think step by step.\n 2+2=5 #### 5"]
        rewards = reward_fn.get_rewards(texts)
        assert rewards == [0.1]

    def test_no_delimiter_gets_zero(self, reward_fn):
        texts = ["Question: What is 2+2?\nAnswer: Let's think step by step.\n The answer is 4"]
        rewards = reward_fn.get_rewards(texts)
        assert rewards == [0.0]  # no #### → no reward regardless of number

    def test_unknown_prompt(self, reward_fn):
        texts = ["Question: Unknown question\nAnswer: #### 42"]
        rewards = reward_fn.get_rewards(texts)
        assert rewards == [0.0]

    def test_batch_mixed_rewards(self, reward_fn):
        texts = [
            "Question: What is 2+2?\nAnswer: Let's think step by step.\n #### 4",  # correct
            "Question: What is 10*5?\nAnswer: Let's think step by step.\n #### 99",  # wrong but has ####
            "Question: How many apples?\nAnswer: Let's think step by step.\n no number here at all",  # no ####, no number
        ]
        rewards = reward_fn.get_rewards(texts)
        assert rewards == [1.0, 0.1, 0.0]

    def test_comma_answer_matches(self, reward_fn):
        texts = ["Question: How many apples?\nAnswer: Let's think step by step.\n #### 1,234"]
        rewards = reward_fn.get_rewards(texts)
        assert rewards == [1.0]


class TestModuleInterface:
    def test_to_device(self):
        fn = GSM8KRewardFunction()
        fn.to(torch.device("cpu"))  # Should not raise

    def test_eval_mode(self):
        fn = GSM8KRewardFunction()
        fn.eval()  # Should not raise

    def test_parameters_iterable(self):
        fn = GSM8KRewardFunction()
        params = list(fn.parameters())
        assert len(params) == 1  # dummy parameter

    def test_tokenizer_is_none(self):
        fn = GSM8KRewardFunction()
        assert fn.tokenizer is None

    def test_requires_grad_false(self):
        fn = GSM8KRewardFunction()
        for p in fn.parameters():
            p.requires_grad_(False)  # Should not raise
