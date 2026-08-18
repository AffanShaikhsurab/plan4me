"""Shared fixtures.

Tests must not depend on the machine they run on. Two leaks are closed here:

* `.env` in the repo root - suppressed with `_env_file=None`.
* The real process environment - `_isolate_env` removes every variable that
  `Settings` would read, because pydantic-settings reads `os.environ` regardless
  of `_env_file`. Without this, a developer with `OPENAI_API_KEY` exported saw
  the credential tests fail.

The autouse fixtures also clear the provider caches so selection never leaks
between tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import Settings, get_settings  # noqa: E402
from backend.llm import chat as chat_facade  # noqa: E402
from backend.llm import embeddings as embedding_facade  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip every Settings-backed variable from the environment.

    Derived from `model_fields` rather than hardcoded, so a new setting is
    covered automatically.
    """
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)


@pytest.fixture
def settings_factory():
    """Build Settings from explicit kwargs only, ignoring .env and os.environ."""

    def _make(**overrides) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _make


@pytest.fixture
def settings(settings_factory) -> Settings:
    return settings_factory()


@pytest.fixture(scope="session", autouse=True)
def _disable_dotenv_for_the_session():
    """Stop every Settings construction from reading the repo `.env`.

    `settings_factory` passes `_env_file=None`, but code paths that call
    `get_settings()` directly - the FastAPI app, the facades - build Settings
    themselves and would otherwise pick up the developer's file.
    """
    original = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    yield
    Settings.model_config["env_file"] = original


@pytest.fixture(autouse=True)
def _clear_provider_caches():
    """Clear the Settings singleton too, not just the providers.

    `get_settings` is `lru_cache`d, so resetting only the provider caches would
    hand the rebuilt provider the same stale Settings.
    """
    get_settings.cache_clear()
    chat_facade.reset_provider_cache()
    embedding_facade.reset_provider_cache()
    yield
    get_settings.cache_clear()
    chat_facade.reset_provider_cache()
    embedding_facade.reset_provider_cache()
