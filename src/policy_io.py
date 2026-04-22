"""
Shared policy model load/save helpers.

The default path uses Hugging Face causal LMs, but callers can provide an
external loader/saver through config.model.policy_loader and
config.model.policy_saver. This is the integration point for MiniTorch-backed
policy implementations.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_callable(spec: Optional[str]) -> Optional[Callable[..., Any]]:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(
            f"Invalid callable spec '{spec}'. Expected format 'module:function'."
        )
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if fn is None:
        raise ValueError(f"Callable '{function_name}' not found in module '{module_name}'.")
    return fn


def _call_loader(loader: Callable[..., Any], identifier: str, config: Any, device: Any):
    call_attempts = [
        lambda: loader(identifier, config, device),
        lambda: loader(identifier, config),
        lambda: loader(identifier),
        lambda: loader(model_name_or_path=identifier, config=config, device=device),
        lambda: loader(model_name_or_path=identifier, config=config),
        lambda: loader(model_name_or_path=identifier),
        lambda: loader(model_name=identifier, config=config, device=device),
        lambda: loader(model_name=identifier, config=config),
        lambda: loader(model_name=identifier),
        lambda: loader(path=identifier, config=config, device=device),
        lambda: loader(path=identifier, config=config),
        lambda: loader(path=identifier),
    ]
    last_error = None
    for attempt in call_attempts:
        try:
            result = attempt()
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "Policy loader must return a (model, tokenizer) tuple."
                )
            return result
        except TypeError as exc:
            last_error = exc
    raise TypeError(
        "Could not call the configured policy loader. "
        "Expected a callable compatible with one of: "
        "(identifier), (identifier, config), (identifier, config, device), "
        "or keyword equivalents such as model_name_or_path=... ."
    ) from last_error


def _call_saver(
    saver: Callable[..., Any],
    model: Any,
    tokenizer: Any,
    save_path: str,
    config: Any,
) -> None:
    call_attempts = [
        lambda: saver(model, tokenizer, save_path, config),
        lambda: saver(model, tokenizer, save_path),
        lambda: saver(model=model, tokenizer=tokenizer, save_path=save_path, config=config),
        lambda: saver(model=model, tokenizer=tokenizer, save_path=save_path),
        lambda: saver(policy_model=model, tokenizer=tokenizer, save_path=save_path, config=config),
        lambda: saver(policy_model=model, tokenizer=tokenizer, save_path=save_path),
        lambda: saver(model=model, tokenizer=tokenizer, path=save_path, config=config),
        lambda: saver(model=model, tokenizer=tokenizer, path=save_path),
    ]
    last_error = None
    for attempt in call_attempts:
        try:
            attempt()
            return
        except TypeError as exc:
            last_error = exc
    raise TypeError(
        "Could not call the configured policy saver. "
        "Expected a callable compatible with one of: "
        "(model, tokenizer, save_path), (model, tokenizer, save_path, config), "
        "or keyword equivalents."
    ) from last_error


def hf_load_policy_model_and_tokenizer(identifier: str, config: Any, device: Any = None):
    """Default Hugging Face policy loader."""
    tokenizer = AutoTokenizer.from_pretrained(identifier)
    tokenizer.padding_side = "left"
    policy_model = AutoModelForCausalLM.from_pretrained(identifier)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if getattr(policy_model, "config", None) is not None:
            policy_model.config.pad_token_id = tokenizer.pad_token_id

    return policy_model, tokenizer


def hf_save_policy_model_and_tokenizer(model: Any, tokenizer: Any, save_path: str, config: Any = None) -> None:
    """Default Hugging Face saver."""
    os.makedirs(save_path, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError(
            "Policy model does not implement save_pretrained(). "
            "Configure config.model.policy_saver for MiniTorch-backed models."
        )
    if not hasattr(tokenizer, "save_pretrained"):
        raise TypeError(
            "Tokenizer does not implement save_pretrained(). "
            "Configure config.model.policy_saver for MiniTorch-backed tokenizers."
        )
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def has_custom_policy_loader(config: Any, loader_spec: Optional[str] = None) -> bool:
    return bool(loader_spec or getattr(config.model, "policy_loader", None))


def load_policy_model_and_tokenizer(
    identifier: str,
    config: Any,
    device: Any = None,
    loader_spec: Optional[str] = None,
):
    """Load a policy model/tokenizer pair through the configured backend."""
    loader = _resolve_callable(loader_spec or getattr(config.model, "policy_loader", None))
    if loader is None:
        return hf_load_policy_model_and_tokenizer(identifier, config, device=device)
    return _call_loader(loader, identifier, config, device)


def save_policy_model_and_tokenizer(
    model: Any,
    tokenizer: Any,
    save_path: str,
    config: Any,
    saver_spec: Optional[str] = None,
) -> None:
    """Save a policy model/tokenizer pair through the configured backend."""
    saver = _resolve_callable(saver_spec or getattr(config.model, "policy_saver", None))
    if saver is None:
        hf_save_policy_model_and_tokenizer(model, tokenizer, save_path, config=config)
        return
    os.makedirs(save_path, exist_ok=True)
    _call_saver(saver, model, tokenizer, save_path, config)
