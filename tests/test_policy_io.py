"""
Tests for configurable policy model I/O hooks.
"""

import copy
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import get_config
from policy_io import load_policy_model_and_tokenizer, save_policy_model_and_tokenizer


class DummyModel:
    pass


class DummyTokenizer:
    pass


def custom_loader(identifier, config, device=None):
    return DummyModel(), DummyTokenizer()


def custom_saver(model, tokenizer, save_path, config):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "minitorch_model.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": "minitorch"}, f)


def test_custom_policy_loader_is_used():
    config = copy.deepcopy(get_config())
    config.model.policy_loader = "tests.test_policy_io:custom_loader"

    model, tokenizer = load_policy_model_and_tokenizer("dummy-minitorch-model", config=config)

    assert type(model).__name__ == "DummyModel"
    assert type(tokenizer).__name__ == "DummyTokenizer"


def test_custom_policy_saver_is_used(tmp_path):
    config = copy.deepcopy(get_config())
    config.model.policy_saver = "tests.test_policy_io:custom_saver"

    save_path = tmp_path / "checkpoint"
    save_policy_model_and_tokenizer(
        DummyModel(),
        DummyTokenizer(),
        str(save_path),
        config=config,
    )

    assert (save_path / "minitorch_model.json").exists()
