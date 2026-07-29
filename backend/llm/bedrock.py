"""Bedrock model factories using langchain-aws.

`ChatBedrockConverse` is the current unified interface (uses the Bedrock
Converse API under the hood) with first-class tool-calling and structured
output. It works across providers on Bedrock, so switching between Anthropic
Claude, DeepSeek V3.2, and Z.AI GLM 4.7 Flash is just a model-ID change.

Non-Anthropic providers (e.g. `zai.*`) are not verified as streaming-capable
with ChatBedrockConverse and emit a warning + fall back to non-streaming. We
run batch workloads, so we disable streaming for those to keep logs clean.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_aws import ChatBedrockConverse

from backend.config import get_settings

# Providers on Bedrock that don't advertise streaming through Converse.
_NON_STREAMING_PREFIXES = ("zai.", "deepseek.")


def _make_llm(model_id: str, *, temperature: float, max_tokens: int) -> ChatBedrockConverse:
    s = get_settings()
    disable_streaming = model_id.startswith(_NON_STREAMING_PREFIXES)
    return ChatBedrockConverse(
        model=model_id,
        region_name=s.aws_region,
        temperature=temperature,
        max_tokens=max_tokens,
        disable_streaming=disable_streaming,
    )


@lru_cache
def get_extraction_llm() -> ChatBedrockConverse:
    """Fast, cheap model for high-volume per-transcript atom extraction."""
    s = get_settings()
    return _make_llm(s.extraction_model_id, temperature=0, max_tokens=4096)


@lru_cache
def get_synthesis_llm() -> ChatBedrockConverse:
    """Stronger model for final synthesis / conflict comparison."""
    s = get_settings()
    return _make_llm(s.synthesis_model_id, temperature=0.2, max_tokens=8192)
