"""
verdict/openai_compat.py - Helpers for model-specific OpenAI request params.
"""

from __future__ import annotations


def uses_max_completion_tokens(model: str) -> bool:
    """
    Return True for model families that expect `max_completion_tokens`
    instead of `max_tokens` on chat completions.
    """
    normalized = (model or "").lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def output_token_limit_arg(model: str, limit: int) -> dict:
    """
    Return the correct output-token-limit argument for a model.
    """
    if uses_max_completion_tokens(model):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}
