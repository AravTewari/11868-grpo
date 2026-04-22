"""
Regression tests for candidate-method evaluation helpers.
"""

import os
import sys
import copy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for path in (REPO_ROOT, SRC_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import get_config
import evaluate


def test_candidate_output_paths_are_method_scoped():
    paths = evaluate.get_evaluation_output_paths("PPO")

    assert paths["candidate_outputs"].endswith("ppo_model_outputs.json")
    assert paths["summary"].endswith("evaluation_summary_ppo.json")
    assert paths["reward_comparison_plot"].endswith("reward_comparison_ppo.png")


def test_legacy_rlhf_flag_resolves_to_candidate_path():
    config = copy.deepcopy(get_config())
    parser = evaluate.build_arg_parser()
    args = parser.parse_args(["--rlhf_model", "/tmp/legacy_rlhf_dir"])

    candidate_model, candidate_label = evaluate.resolve_candidate_selection(
        config=config,
        candidate_model=args.candidate_model,
        candidate_label=args.candidate_label,
        legacy_rlhf_model=args.rlhf_model,
    )

    assert candidate_model == "/tmp/legacy_rlhf_dir"
    assert candidate_label == "ppo"


def test_dpo_label_defaults_to_dpo_output_dir():
    config = copy.deepcopy(get_config())
    config.system.dpo_model_dir = "/tmp/dpo_outputs"

    candidate_model, candidate_label = evaluate.resolve_candidate_selection(
        config=config,
        candidate_model=None,
        candidate_label="dpo",
    )

    assert candidate_model == "/tmp/dpo_outputs"
    assert candidate_label == "dpo"


def test_custom_policy_loader_accepts_non_hf_candidate_dir(tmp_path):
    candidate_root = tmp_path / "candidate"
    candidate_dir = candidate_root / "best_grpo_model"
    candidate_dir.mkdir(parents=True)

    resolved = evaluate.resolve_candidate_model_path(
        str(candidate_root),
        "grpo",
        allow_custom_loader_dirs=True,
    )

    assert resolved == str(candidate_dir)
