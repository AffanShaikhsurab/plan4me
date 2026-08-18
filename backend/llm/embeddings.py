"""Facade over the embedding-provider registry.

`cluster_atoms` depends only on `embed_texts` and `similarity_threshold`, so the
choice between Titan, an OpenAI-compatible endpoint, and the keyless local
vectoriser is a configuration detail.
"""
from __future__ import annotations

from backend.config import get_settings
from backend.llm.providers import (
    EmbeddingProvider,
    ProviderError,
    create_embedding,
)

_PROVIDER: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    """Resolve (and memoise) the provider named by EMBEDDING_PROVIDER."""
    global _PROVIDER
    if _PROVIDER is None:
        settings = get_settings()
        _PROVIDER = create_embedding(settings.embedding_provider, settings)
    return _PROVIDER


def reset_provider_cache() -> None:
    """Forget the memoised provider.

    This alone is not enough to pick up changed configuration, because
    `get_settings()` is itself cached - use `backend.llm.reset_caches()` for
    that.
    """
    global _PROVIDER
    _PROVIDER = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using the configured provider."""
    if not texts:
        return []
    provider = get_provider()
    vectors = provider.embed(texts)
    # EmbeddingProvider.embed promises one vector per input, in order. A
    # truncated or ragged response otherwise surfaces much later as an
    # IndexError or a numpy ragged-array error inside cluster_atoms.
    if len(vectors) != len(texts):
        raise ProviderError(
            f"Embedding provider {provider.name!r} returned {len(vectors)} "
            f"vectors for {len(texts)} inputs."
        )
    widths = {len(v) for v in vectors}
    if len(widths) > 1:
        raise ProviderError(
            f"Embedding provider {provider.name!r} returned vectors of "
            f"differing widths: {sorted(widths)}."
        )
    return vectors


def embed_text(text: str) -> list[float]:
    """Return an embedding vector for a single string."""
    return embed_texts([text])[0]


def similarity_threshold() -> float:
    """Dedupe cutoff for the active provider.

    An explicitly configured DEDUPE_SIMILARITY_THRESHOLD always wins. Otherwise
    each provider supplies the cutoff that suits its own vector space, because
    lexical and semantic cosine are not on the same scale.
    """
    settings = get_settings()
    if "dedupe_similarity_threshold" in settings.model_fields_set:
        return settings.dedupe_similarity_threshold
    return get_provider().similarity_threshold()


def describe() -> dict:
    """Resolved embedding configuration, for /health."""
    return get_provider().describe()
