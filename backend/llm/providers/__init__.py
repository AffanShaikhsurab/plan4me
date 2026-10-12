"""Provider package: importing it registers every built-in provider.

The concrete modules are imported for their registration side effect. They only
import vendor SDKs lazily inside `_build()` / `embed()`, so importing this
package does not require any provider's package to be installed.

To add a provider: create a class in `chat.py` (or your own module) decorated
with `@register_chat`, implement `_build()`, and set `name` plus the two default
model names. Nothing else in the codebase changes.
"""
from __future__ import annotations

from backend.llm.providers import chat as _chat  # noqa: F401  (registration)
from backend.llm.providers import embedding as _embedding  # noqa: F401
from backend.llm.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ModelSpec,
    ProviderError,
    Role,
)
from backend.llm.providers.registry import (
    available_chat,
    available_embedding,
    create_chat,
    create_embedding,
    register_chat,
    register_embedding,
)

__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "ModelSpec",
    "ProviderError",
    "Role",
    "available_chat",
    "available_embedding",
    "create_chat",
    "create_embedding",
    "register_chat",
    "register_embedding",
]
