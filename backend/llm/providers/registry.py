"""Provider registries.

A registry plus a decorator keeps provider selection open for extension and
closed for modification: a new vendor is a new module that decorates its class,
and no existing file changes. Nothing here imports a vendor SDK, so registering
every provider stays cheap - the heavy import happens inside `_build()`, only
for the provider actually selected.
"""
from __future__ import annotations

from typing import TypeVar

from backend.config import Settings
from backend.llm.providers.base import ChatProvider, EmbeddingProvider, ProviderError

_CHAT: dict[str, type[ChatProvider]] = {}
_EMBEDDING: dict[str, type[EmbeddingProvider]] = {}

C = TypeVar("C", bound=ChatProvider)
E = TypeVar("E", bound=EmbeddingProvider)


def register_chat(cls: type[C]) -> type[C]:
    """Class decorator: make `cls` selectable via LLM_PROVIDER."""
    _register(_CHAT, getattr(cls, "name", ""), cls, "chat")
    return cls


def register_embedding(cls: type[E]) -> type[E]:
    """Class decorator: make `cls` selectable via EMBEDDING_PROVIDER."""
    _register(_EMBEDDING, getattr(cls, "name", ""), cls, "embedding")
    return cls


def _register(target: dict, name: str, cls: type, kind: str) -> None:
    # Normalised first, so validation sees what lookup will see. Registering
    # under the same normalisation means a provider cannot be listed as
    # available yet be unresolvable, and a whitespace-only name is rejected
    # rather than claiming the empty key.
    # (A bare `name: ClassVar[str]` annotation defines no attribute, hence the
    # getattr default at the call sites.)
    name = (name or "").strip().lower()
    if not name:
        raise ProviderError(f"{cls.__name__} must define a non-empty `name`.")
    existing = target.get(name)
    if existing is not None and existing is not cls:
        raise ProviderError(
            f"Two {kind} providers claim the name {name!r}: "
            f"{existing.__name__} and {cls.__name__}."
        )
    target[name] = cls


def available_chat() -> list[str]:
    return sorted(_CHAT)


def available_embedding() -> list[str]:
    return sorted(_EMBEDDING)


def create_chat(name: str, settings: Settings) -> ChatProvider:
    return _create(_CHAT, name, settings, "LLM_PROVIDER")


def create_embedding(name: str, settings: Settings) -> EmbeddingProvider:
    return _create(_EMBEDDING, name, settings, "EMBEDDING_PROVIDER")


def _create(target: dict, name: str, settings: Settings, env_name: str):
    key = (name or "").strip().lower()
    cls = target.get(key)
    if cls is None:
        raise ProviderError(
            f"{env_name}={name!r} is not supported. "
            f"Available: {', '.join(sorted(target))}."
        )
    return cls(settings)
