"""Embedding providers, threshold resolution, and clustering behaviour.

The local provider is exercised for real (it needs no network). Bedrock and the
OpenAI-compatible path are tested through injected fakes, so the tests assert
our contract - batch order preserved, credentials demanded - without calling a
paid API.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.llm import embeddings as facade
from backend.llm.providers import ProviderError, create_embedding
from backend.llm.providers.embedding import LocalEmbeddingProvider


# --- local provider --------------------------------------------------------
def test_local_returns_one_unit_vector_per_input(settings):
    provider = create_embedding("local", settings)
    vectors = np.array(provider.embed(["alpha beta", "gamma delta", "epsilon"]))
    assert vectors.shape == (3, LocalEmbeddingProvider._DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_local_scores_paraphrases_above_unrelated_text(settings):
    provider = create_embedding("local", settings)
    corpus = [
        "Charge based on the value you deliver, not your cost",
        "Price based on the value you deliver rather than your costs",
        "Use Docker for reproducible deployments",
    ]
    v = np.array(provider.embed(corpus))
    sim = v @ v.T
    assert sim[0, 1] > sim[0, 2]
    assert sim[0, 1] > provider.default_similarity_threshold


def test_local_bucketing_is_stable_across_calls(settings):
    """crc32 rather than hash(), so vectors do not shift between processes."""
    provider = create_embedding("local", settings)
    first = provider.embed(["always network with founders"])
    second = provider.embed(["always network with founders"])
    assert first == second


def test_local_is_order_sensitive_via_bigrams(settings):
    provider = create_embedding("local", settings)
    v = np.array(provider.embed(["alpha beta gamma", "gamma beta alpha"]))
    assert float(v[0] @ v[1]) < 1.0


def test_empty_string_yields_a_finite_vector(settings):
    """A claim can be empty; that must not produce NaN and poison clustering."""
    vec = np.array(create_embedding("local", settings).embed([""]))
    assert np.isfinite(vec).all()


# --- openai-compatible provider -------------------------------------------
class _FakeOpenAI:
    """Returns embeddings out of order, to prove we re-sort by index."""

    def __init__(self, *_, **__):
        self.embeddings = self

    def create(self, model, input):  # noqa: A002 - mirrors the real signature
        class _Datum:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        data = [_Datum(i, [float(i)]) for i in range(len(input))]
        return type("Resp", (), {"data": list(reversed(data))})()


def test_openai_embeddings_are_returned_in_request_order(
    settings_factory, monkeypatch
):
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    settings = settings_factory(
        embedding_provider="openai", openai_embedding_api_key="sk-test"
    )
    result = create_embedding("openai", settings).embed(["a", "b", "c"])
    assert result == [[0.0], [1.0], [2.0]]


def test_openai_embeddings_require_a_key(settings_factory):
    settings = settings_factory(embedding_provider="openai")
    with pytest.raises(ProviderError, match="OPENAI_EMBEDDING_API_KEY"):
        create_embedding("openai", settings).embed(["a"])


# --- bedrock provider ------------------------------------------------------
def test_bedrock_embeddings_call_the_configured_model(settings_factory):
    import json

    calls = []

    class _FakeBody:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload)

    class _FakeRuntime:
        def invoke_model(self, modelId, body):  # noqa: N803 - boto3 casing
            calls.append((modelId, json.loads(body)["inputText"]))
            return {"body": _FakeBody({"embedding": [0.1, 0.2]})}

    settings = settings_factory(embedding_model_id="amazon.titan-embed-text-v2:0")
    provider = create_embedding("bedrock", settings)
    provider._client = _FakeRuntime()

    assert provider.embed(["one", "two"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert [c[0] for c in calls] == ["amazon.titan-embed-text-v2:0"] * 2
    assert [c[1] for c in calls] == ["one", "two"]


# --- threshold resolution --------------------------------------------------
def test_threshold_defaults_are_provider_specific(settings_factory, monkeypatch):
    """A lexical space and a semantic space cannot share one cutoff."""
    import backend.config as config

    for provider_name, expected in (("local", 0.35), ("bedrock", 0.82)):
        facade.reset_provider_cache()
        settings = settings_factory(embedding_provider=provider_name)
        monkeypatch.setattr(config, "get_settings", lambda s=settings: s)
        monkeypatch.setattr(facade, "get_settings", lambda s=settings: s)
        assert facade.similarity_threshold() == expected


def test_explicit_threshold_overrides_the_provider_default(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        embedding_provider="local", dedupe_similarity_threshold=0.9
    )
    monkeypatch.setattr(facade, "get_settings", lambda: settings)
    assert facade.similarity_threshold() == 0.9


# --- facade ----------------------------------------------------------------
def test_embed_texts_short_circuits_on_empty_input(settings_factory, monkeypatch):
    """No provider should be constructed, so this holds even with no creds."""
    settings = settings_factory(embedding_provider="openai")
    monkeypatch.setattr(facade, "get_settings", lambda: settings)
    assert facade.embed_texts([]) == []


def test_embed_text_returns_a_single_vector(settings_factory, monkeypatch):
    settings = settings_factory(embedding_provider="local")
    monkeypatch.setattr(facade, "get_settings", lambda: settings)
    vec = facade.embed_text("hello")
    assert isinstance(vec, list) and len(vec) == LocalEmbeddingProvider._DIM


# --- clustering integration ------------------------------------------------
def test_clustering_merges_paraphrases_across_videos(
    settings_factory, monkeypatch
):
    """The end the pipeline actually cares about: support counts by video."""
    from backend.pipeline import clustering
    from backend.schemas import AtomType, KnowledgeAtom

    settings = settings_factory(embedding_provider="local")
    monkeypatch.setattr(facade, "get_settings", lambda: settings)

    claims = [
        ("Charge based on the value you deliver, not your cost", "v1"),
        ("Price based on the value you deliver rather than your costs", "v2"),
        ("Use Docker for reproducible deployments", "v3"),
    ]
    atoms = [
        KnowledgeAtom(
            id=str(i),
            topic="t",
            type=AtomType.advice,
            claim=claim,
            confidence=0.9,
            video_id=video,
            video_title=f"Video {video}",
        )
        for i, (claim, video) in enumerate(claims)
    ]

    clusters = clustering.cluster_atoms(atoms)
    assert len(clusters) == 2
    # Sorted by support, so the merged pair leads with two distinct videos.
    assert clusters[0].support_count == 2
    assert clusters[1].support_count == 1


def test_clustering_handles_no_atoms(settings):
    from backend.pipeline import clustering

    assert clustering.cluster_atoms([]) == []


# --- regressions found by adversarial review -------------------------------
def test_local_threshold_setting_is_actually_read(settings_factory):
    """Regression: LOCAL_DEDUPE_SIMILARITY_THRESHOLD was dead config."""
    provider = create_embedding(
        "local", settings_factory(local_dedupe_similarity_threshold=0.61)
    )
    assert provider.similarity_threshold() == 0.61
    assert provider.describe()["similarity_threshold"] == 0.61


def test_local_threshold_setting_flows_through_the_facade(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        embedding_provider="local", local_dedupe_similarity_threshold=0.61
    )
    monkeypatch.setattr(facade, "get_settings", lambda: settings)
    assert facade.similarity_threshold() == 0.61


def test_truncated_embedding_response_is_rejected(settings_factory, monkeypatch):
    """The one-vector-per-input contract is enforced, not merely documented.

    Regression: a short batch surfaced as an IndexError deep inside
    cluster_atoms instead of naming the provider.
    """
    from backend.llm.providers.base import EmbeddingProvider
    from backend.llm.providers.registry import _EMBEDDING

    class _Truncating(EmbeddingProvider):
        name = "truncating-probe"
        default_similarity_threshold = 0.8

        def embed(self, texts):
            return [[1.0, 0.0]]  # one vector regardless of input size

    _EMBEDDING["truncating-probe"] = _Truncating
    try:
        settings = settings_factory(embedding_provider="truncating-probe")
        monkeypatch.setattr(facade, "get_settings", lambda: settings)
        with pytest.raises(ProviderError, match="3 inputs"):
            facade.embed_texts(["a", "b", "c"])
    finally:
        _EMBEDDING.pop("truncating-probe", None)


def test_ragged_embedding_response_is_rejected(settings_factory, monkeypatch):
    from backend.llm.providers.base import EmbeddingProvider
    from backend.llm.providers.registry import _EMBEDDING

    class _Ragged(EmbeddingProvider):
        name = "ragged-probe"
        default_similarity_threshold = 0.8

        def embed(self, texts):
            return [[1.0, 0.0], [1.0]]

    _EMBEDDING["ragged-probe"] = _Ragged
    try:
        settings = settings_factory(embedding_provider="ragged-probe")
        monkeypatch.setattr(facade, "get_settings", lambda: settings)
        with pytest.raises(ProviderError, match="differing widths"):
            facade.embed_texts(["a", "b"])
    finally:
        _EMBEDDING.pop("ragged-probe", None)
