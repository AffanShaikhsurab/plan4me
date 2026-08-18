"""Provider abstractions for chat models and embeddings.

Design notes
------------
The pipeline depends on these abstract types, never on a vendor SDK (dependency
inversion). Each concrete provider owns exactly one concern - how to build its
own client (single responsibility) - and is substitutable for any other
(Liskov), so `extract_atoms` and `cluster_atoms` cannot tell which vendor is
answering.

`ChatProvider` is a template method: `extraction_llm()` / `synthesis_llm()`
resolve the model name and role parameters here and defer only vendor
construction to `_build()`. Adding a provider means implementing one method,
never editing dispatch code (open/closed).

Chat and embeddings are separate interfaces on purpose: most chat vendors serve
no embeddings endpoint, and a single merged interface would force
`NotImplementedError` stubs onto nearly every implementation (interface
segregation).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from backend.config import Settings

Role = Literal["extraction", "synthesis"]


class ProviderError(RuntimeError):
    """Configuration or dependency problem, raised with the fix in the message."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything a vendor client needs, expressed without vendor naming."""

    model: str
    temperature: float
    max_tokens: int


# Role parameters live here so every provider treats the two roles identically:
# extraction is high-volume and deterministic, synthesis is one call and may
# breathe a little.
_ROLE_PARAMS: dict[Role, tuple[float, int]] = {
    "extraction": (0.0, 4096),
    "synthesis": (0.2, 8192),
}


class ChatProvider(ABC):
    """Builds the chat models used for extraction and synthesis."""

    #: Registry key, i.e. the accepted value of LLM_PROVIDER.
    name: ClassVar[str]
    #: Vendor defaults, used when no override is configured.
    default_extraction_model: ClassVar[str]
    default_synthesis_model: ClassVar[str]
    #: Passed to `with_structured_output(method=...)`. None keeps the
    #: integration's own default, which is what Bedrock Converse wants.
    structured_output_method: ClassVar[str | None] = None
    #: Vendors disagree on the output-cap parameter name; Ollama has no
    #: `max_tokens` at all, it has `num_predict`.
    max_tokens_kwarg: ClassVar[str] = "max_tokens"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # -- template method ---------------------------------------------------
    def extraction_llm(self) -> Any:
        """Fast, cheap model for high-volume per-transcript atom extraction."""
        return self._build(self.spec("extraction"))

    def synthesis_llm(self) -> Any:
        """Stronger model for the final report."""
        return self._build(self.spec("synthesis"))

    def spec(self, role: Role) -> ModelSpec:
        self._check_role(role)
        temperature, max_tokens = _ROLE_PARAMS[role]
        return ModelSpec(self.model_for(role), temperature, max_tokens)

    def model_for(self, role: Role) -> str:
        """Configured override when present, else this provider's default."""
        self._check_role(role)
        return self._override_for(role) or self._default_for(role)

    # Split out so subclasses can insert extra sources (e.g. Bedrock's legacy
    # settings) without re-deriving "was an override supplied?".
    def _override_for(self, role: Role) -> str:
        return (
            self._settings.extraction_model
            if role == "extraction"
            else self._settings.synthesis_model
        ).strip()

    def _default_for(self, role: Role) -> str:
        return (
            self.default_extraction_model
            if role == "extraction"
            else self.default_synthesis_model
        )

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in _ROLE_PARAMS:
            raise ProviderError(
                f"Unknown role {role!r}; expected one of {tuple(_ROLE_PARAMS)}."
            )

    def bind_schema(self, llm: Any, schema: type) -> Any:
        """Constrain a model to `schema` using this vendor's best method."""
        if self.structured_output_method is None:
            return llm.with_structured_output(schema)
        return llm.with_structured_output(schema, method=self.structured_output_method)

    def describe(self) -> dict:
        """Resolved configuration, for /health and diagnostics."""
        return {
            "provider": self.name,
            "extraction_model": self.model_for("extraction"),
            "synthesis_model": self.model_for("synthesis"),
        }

    # -- the only required override ----------------------------------------
    @abstractmethod
    def _build(self, spec: ModelSpec) -> Any:
        """Construct the vendor client for `spec`."""

    # -- helpers for subclasses -------------------------------------------
    def _token_kwargs(self, spec: ModelSpec) -> dict:
        return {self.max_tokens_kwarg: spec.max_tokens}

    def _require(self, value: str, env_name: str, where: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ProviderError(
                f"LLM_PROVIDER={self.name} requires {env_name}. "
                f"Set it in .env (get a key at {where})."
            )
        return cleaned

    @staticmethod
    def _import(module: str, attr: str, package: str) -> Any:
        """Import a vendor integration, or explain how to install it."""
        try:
            mod = __import__(module, fromlist=[attr])
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ProviderError(
                f"{package} is not installed. Run: pip install {package}"
            ) from exc
        return getattr(mod, attr)


class EmbeddingProvider(ABC):
    """Turns claim text into vectors for similarity clustering."""

    name: ClassVar[str]
    #: Cosine cutoff suited to this provider's vector space. Semantic models
    #: score paraphrases far above lexical ones, so this cannot be shared.
    default_similarity_threshold: ClassVar[float]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, returning one vector per input, in input order."""

    def similarity_threshold(self) -> float:
        """Cutoff to use. Overridden where a setting should be able to tune it."""
        return self.default_similarity_threshold

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "similarity_threshold": self.similarity_threshold(),
        }
