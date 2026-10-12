"""Facade over the chat-provider registry.

Call sites use these four functions and never name a vendor, so switching
providers is an `LLM_PROVIDER` change and nothing else. The resolved provider is
cached because `Settings` is itself a cached singleton; `reset_provider_cache()`
exists for tests and for anything that reloads configuration.
"""
from __future__ import annotations

from typing import Any

from backend.config import get_settings
from backend.llm.providers import ChatProvider, create_chat

_PROVIDER: ChatProvider | None = None


def get_provider() -> ChatProvider:
    """Resolve (and memoise) the provider named by LLM_PROVIDER."""
    global _PROVIDER
    if _PROVIDER is None:
        settings = get_settings()
        _PROVIDER = create_chat(settings.llm_provider, settings)
    return _PROVIDER


def reset_provider_cache() -> None:
    """Forget the memoised provider.

    This alone is not enough to pick up changed configuration, because
    `get_settings()` is itself cached - use `backend.llm.reset_caches()` for
    that.
    """
    global _PROVIDER
    _PROVIDER = None


def get_extraction_llm() -> Any:
    """Fast, cheap model for high-volume per-transcript atom extraction."""
    return get_provider().extraction_llm()


def get_synthesis_llm() -> Any:
    """Stronger model for final synthesis / conflict comparison."""
    return get_provider().synthesis_llm()


def structured(llm: Any, schema: type) -> Any:
    """Bind `schema` as the model's output contract, per provider."""
    return get_provider().bind_schema(llm, schema)


def active_models() -> dict:
    """Describe the resolved provider/model pair, for /health."""
    return get_provider().describe()
