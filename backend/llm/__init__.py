"""LLM layer: provider-agnostic chat models and embeddings."""
from __future__ import annotations


def reset_caches() -> None:
    """Drop every cached provider AND the cached Settings.

    Provider resolution memoises both the provider object and, transitively,
    the `Settings` singleton. Clearing only one leaves the other stale, so
    reconfiguring at runtime needs all three cleared together.
    """
    from backend.config import get_settings
    from backend.llm import chat, embeddings

    get_settings.cache_clear()
    chat.reset_provider_cache()
    embeddings.reset_provider_cache()
