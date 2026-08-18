"""Registry and abstraction contracts.

These tests guard the properties the design depends on, rather than any one
vendor: every provider is substitutable, selection is data-driven, and a bad
provider name fails loudly instead of silently falling back.
"""
from __future__ import annotations

import pytest

from backend.llm.providers import (
    ChatProvider,
    EmbeddingProvider,
    ProviderError,
    available_chat,
    available_embedding,
    create_chat,
    create_embedding,
)
from backend.llm.providers.registry import register_chat

EXPECTED_CHAT = {"anthropic", "bedrock", "gemini", "moonshot", "ollama", "openai"}
EXPECTED_EMBEDDING = {"bedrock", "local", "openai"}


def test_every_documented_chat_provider_is_registered():
    assert EXPECTED_CHAT <= set(available_chat())


def test_every_documented_embedding_provider_is_registered():
    assert EXPECTED_EMBEDDING <= set(available_embedding())


@pytest.mark.parametrize("name", sorted(EXPECTED_CHAT))
def test_chat_providers_satisfy_the_interface(name, settings):
    """Liskov: each provider is usable purely through the abstract type."""
    provider = create_chat(name, settings)
    assert isinstance(provider, ChatProvider)
    assert provider.name == name
    # Both roles resolve to a non-empty model name without touching the network.
    assert provider.model_for("extraction")
    assert provider.model_for("synthesis")
    described = provider.describe()
    assert described["provider"] == name
    assert {"extraction_model", "synthesis_model"} <= described.keys()


@pytest.mark.parametrize("name", sorted(EXPECTED_EMBEDDING))
def test_embedding_providers_satisfy_the_interface(name, settings):
    provider = create_embedding(name, settings)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.name == name
    assert 0.0 < provider.default_similarity_threshold <= 1.0


def test_unknown_chat_provider_is_rejected_with_the_options(settings):
    with pytest.raises(ProviderError) as exc:
        create_chat("not-a-provider", settings)
    message = str(exc.value)
    assert "LLM_PROVIDER" in message
    # The error must be actionable, i.e. list what is valid.
    assert "openai" in message


def test_unknown_embedding_provider_is_rejected(settings):
    with pytest.raises(ProviderError) as exc:
        create_embedding("nope", settings)
    assert "EMBEDDING_PROVIDER" in str(exc.value)


def test_provider_names_are_case_and_whitespace_insensitive(settings):
    assert create_chat("  OpenAI  ", settings).name == "openai"


def test_duplicate_registration_under_one_name_is_refused():
    """Two providers claiming one name would make selection ambiguous."""

    class _First(ChatProvider):
        name = "collision-probe"
        default_extraction_model = "a"
        default_synthesis_model = "a"

        def _build(self, spec):  # pragma: no cover - never built
            raise AssertionError

    class _Second(ChatProvider):
        name = "collision-probe"
        default_extraction_model = "b"
        default_synthesis_model = "b"

        def _build(self, spec):  # pragma: no cover - never built
            raise AssertionError

    register_chat(_First)
    try:
        with pytest.raises(ProviderError, match="collision-probe"):
            register_chat(_Second)
    finally:
        from backend.llm.providers import registry

        registry._CHAT.pop("collision-probe", None)


def test_role_parameters_differ_between_extraction_and_synthesis(settings):
    """Extraction is deterministic and small; synthesis is looser and larger."""
    provider = create_chat("openai", settings)
    extraction = provider.spec("extraction")
    synthesis = provider.spec("synthesis")
    assert extraction.temperature == 0.0
    assert synthesis.temperature > extraction.temperature
    assert synthesis.max_tokens > extraction.max_tokens


def test_whitespace_only_provider_name_is_rejected():
    """Regression: `"   "` passed the emptiness check, then normalised to `""`.

    That claimed the empty key, so `create_chat("")` selected it.
    """
    from backend.llm.providers.registry import _CHAT, register_chat

    class _Blank(ChatProvider):
        name = "   "
        default_extraction_model = "x"
        default_synthesis_model = "x"

        def _build(self, spec):  # pragma: no cover - never built
            raise AssertionError

    try:
        with pytest.raises(ProviderError, match="non-empty"):
            register_chat(_Blank)
        assert "" not in _CHAT
    finally:
        _CHAT.pop("", None)


def test_empty_provider_name_never_resolves(settings):
    with pytest.raises(ProviderError):
        create_chat("", settings)
    with pytest.raises(ProviderError):
        create_chat("   ", settings)
