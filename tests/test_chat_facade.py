"""The facade the pipeline actually imports, plus the health endpoints.

`extract_atoms` and `synthesize_report` must depend on nothing vendor-specific,
so these tests swap the provider for a fake and assert the pipeline still runs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.llm import chat as facade
from backend.llm.providers import ChatProvider, ModelSpec


class _FakeClient:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def invoke(self, _messages):
        return type("Resp", (), {"content": "ok"})()

    def with_structured_output(self, schema, **kwargs):
        return self


class _FakeProvider(ChatProvider):
    name = "fake"
    default_extraction_model = "fake-fast"
    default_synthesis_model = "fake-strong"

    def _build(self, spec: ModelSpec):
        return _FakeClient(spec)


@pytest.fixture
def fake_provider(settings, monkeypatch):
    provider = _FakeProvider(settings)
    monkeypatch.setattr(facade, "get_provider", lambda: provider)
    return provider


def test_facade_caches_the_provider(settings_factory, monkeypatch):
    """Providers are resolved once; Settings is itself a cached singleton."""
    settings = settings_factory(llm_provider="ollama")
    monkeypatch.setattr(facade, "get_settings", lambda: settings)
    assert facade.get_provider() is facade.get_provider()


def test_reset_clears_the_cache(settings_factory, monkeypatch):
    settings = settings_factory(llm_provider="ollama")
    monkeypatch.setattr(facade, "get_settings", lambda: settings)
    first = facade.get_provider()
    facade.reset_provider_cache()
    assert facade.get_provider() is not first


def test_facade_selects_the_configured_provider(settings_factory, monkeypatch):
    settings = settings_factory(llm_provider="anthropic")
    monkeypatch.setattr(facade, "get_settings", lambda: settings)
    assert facade.get_provider().name == "anthropic"


def test_extraction_and_synthesis_get_their_own_role_params(fake_provider):
    assert facade.get_extraction_llm().spec.temperature == 0.0
    assert facade.get_synthesis_llm().spec.temperature > 0.0


def test_active_models_reports_the_resolved_pair(fake_provider):
    described = facade.active_models()
    assert described == {
        "provider": "fake",
        "extraction_model": "fake-fast",
        "synthesis_model": "fake-strong",
    }


def test_structured_delegates_to_the_provider(fake_provider):
    client = facade.get_extraction_llm()
    assert facade.structured(client, dict) is client


def test_synthesis_runs_against_any_provider(fake_provider, monkeypatch):
    """Vendor-agnostic: synthesis only needs .invoke() and .content."""
    from backend.llm import synthesis
    from backend.schemas import AtomCluster, AtomType, KnowledgeAtom

    monkeypatch.setattr(synthesis, "get_synthesis_llm", facade.get_synthesis_llm)
    atom = KnowledgeAtom(
        id="1",
        topic="t",
        type=AtomType.advice,
        claim="Do the thing",
        confidence=0.9,
        video_id="v1",
        video_title="Video 1",
    )
    cluster = AtomCluster(
        cluster_id=0,
        representative_claim="Do the thing",
        type=AtomType.advice,
        support_count=1,
        atoms=[atom],
    )
    assert synthesis.synthesize_report("topic", [cluster]) == "ok"


def test_synthesis_short_circuits_without_clusters():
    from backend.llm import synthesis

    report = synthesis.synthesize_report("my topic", [])
    assert "my topic" in report


# --- health endpoints ------------------------------------------------------
@pytest.fixture
def client():
    from backend.api.main import app

    return TestClient(app)


def test_health_reports_both_providers(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # The frontend's HealthInfo shape must stay intact.
    assert {"region", "extraction_model", "synthesis_model", "whisper_fallback"} <= body.keys()
    assert {"llm_provider", "embedding_provider"} <= body.keys()


def test_health_embeddings_returns_real_dimensions(client, monkeypatch):
    from backend.llm import embeddings as embedding_facade

    settings = embedding_facade.get_settings()
    monkeypatch.setattr(
        embedding_facade,
        "get_settings",
        lambda: settings.model_copy(update={"embedding_provider": "local"}),
    )
    embedding_facade.reset_provider_cache()
    body = client.get("/health/embeddings").json()
    assert body["status"] == "ok"
    assert body["provider"] == "local"
    assert body["dims"] > 0


def test_health_llm_surfaces_a_missing_key_as_503(client, monkeypatch):
    """A misconfigured provider must fail loudly with a fixable message."""
    settings = facade.get_settings()
    monkeypatch.setattr(
        facade,
        "get_settings",
        lambda: settings.model_copy(
            update={"llm_provider": "anthropic", "anthropic_api_key": ""}
        ),
    )
    facade.reset_provider_cache()
    resp = client.get("/health/llm")
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_health_bedrock_remains_an_alias(client):
    """Older docs and scripts link /health/bedrock; it must not 404."""
    assert client.get("/health/bedrock").status_code in (200, 503)


def test_health_degrades_instead_of_500_on_unknown_provider(client, monkeypatch):
    """A mistyped LLM_PROVIDER must not take down the liveness probe.

    Regression: `active_models()` used to run outside any try, so a bad
    provider name surfaced as an unhandled 500.
    """
    import backend.api.main as main

    settings = facade.get_settings()
    bad = settings.model_copy(update={"llm_provider": "totally_bogus"})
    monkeypatch.setattr(main, "get_settings", lambda: bad)
    monkeypatch.setattr(facade, "get_settings", lambda: bad)
    facade.reset_provider_cache()

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["llm_provider"] == "totally_bogus"
    # The message must name the valid options.
    assert "anthropic" in body["llm_error"]


def test_health_llm_returns_503_not_500_on_unknown_provider(client, monkeypatch):
    """Configuration errors are 503, never an unhandled 500."""
    import backend.api.main as main

    settings = facade.get_settings()
    bad = settings.model_copy(update={"llm_provider": "totally_bogus"})
    monkeypatch.setattr(main, "get_settings", lambda: bad)
    monkeypatch.setattr(facade, "get_settings", lambda: bad)
    facade.reset_provider_cache()

    resp = client.get("/health/llm")
    assert resp.status_code == 503
    assert "not supported" in resp.json()["detail"]

