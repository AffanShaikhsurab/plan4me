"""Concrete chat providers.

One class per vendor. Constructor keyword names were taken from each
integration's `model_fields` rather than from memory, because they disagree:
`ChatOllama` has no `max_tokens` (it uses `num_predict`), and
`ChatGoogleGenerativeAI` accepts `max_tokens` only as an alias of
`max_output_tokens`.

Structured output method per vendor. Every integration here accepts
"function_calling" and "json_schema", and all of them default to
"function_calling", so leaving it unset is a real choice rather than a gap:
  - Bedrock Converse  -> unset, i.e. the integration default. This is exactly
                         what the pre-refactor code did, so behaviour is
                         unchanged.
  - Anthropic         -> "json_schema" (native structured output)
  - OpenAI-compatible -> "function_calling" explicitly; a gateway that only
                         proxies chat completions still honours tool calls,
                         whereas `json_schema` often 400s
  - Gemini            -> unset (integration default)
  - Ollama            -> "json_schema", which maps to Ollama's own `format`
                         parameter; local models are frequently poor at tool
                         calling
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from backend.llm.providers.base import ChatProvider, ModelSpec
from backend.llm.providers.registry import register_chat


@register_chat
class BedrockChatProvider(ChatProvider):
    """Amazon Bedrock via the Converse API.

    Credentials come from the standard boto3 chain, so nothing is read here.
    """

    name = "bedrock"
    # Overridden by model_for(); Bedrock keeps its legacy setting names.
    default_extraction_model = "zai.glm-4.7-flash"
    default_synthesis_model = "deepseek.v3.2"

    # Providers on Bedrock that do not advertise streaming through Converse.
    _NON_STREAMING_PREFIXES = ("zai.", "deepseek.")

    def model_for(self, role):  # noqa: ANN001, ANN201
        """Resolve in precedence order: generic override, legacy id, default.

        Bedrock predates the generic EXTRACTION_MODEL / SYNTHESIS_MODEL names,
        so the older EXTRACTION_MODEL_ID / SYNTHESIS_MODEL_ID still work.

        This asks whether a setting was *supplied* rather than comparing the
        resolved value against the class default. Comparing against the default
        silently discarded a legitimate override whenever the user set the
        generic name to the same string as the default - which is exactly the
        string they would type, since the two share a value.
        """
        self._check_role(role)
        return (
            self._override_for(role)
            or self._legacy_for(role)
            or self._default_for(role)
        )

    def _legacy_for(self, role) -> str:  # noqa: ANN001
        """The pre-generic setting names, stripped like every other source."""
        return (
            self._settings.extraction_model_id
            if role == "extraction"
            else self._settings.synthesis_model_id
        ).strip()

    def describe(self) -> dict:
        return {**super().describe(), "region": self._settings.aws_region}

    def _build(self, spec: ModelSpec) -> Any:
        cls = self._import("langchain_aws", "ChatBedrockConverse", "langchain-aws")
        return cls(
            model=spec.model,
            region_name=self._settings.aws_region,
            temperature=spec.temperature,
            disable_streaming=spec.model.startswith(self._NON_STREAMING_PREFIXES),
            **self._token_kwargs(spec),
        )


class _OpenAICompatibleProvider(ChatProvider):
    """Shared base for anything speaking the OpenAI wire format."""

    structured_output_method = "function_calling"

    #: Subclasses point these at the right settings fields.
    _key_env: str = "OPENAI_API_KEY"
    _key_url: str = "https://platform.openai.com/api-keys"

    # Abstract, not NotImplementedError: a subclass that forgets one fails at
    # instantiation with a TypeError naming the method, instead of registering
    # successfully and then raising from inside /health or mid-pipeline.
    @abstractmethod
    def _api_key(self) -> str:
        """The configured credential for this vendor."""

    @abstractmethod
    def _base_url(self) -> str | None:
        """Endpoint override, or None for the vendor default."""

    def describe(self) -> dict:
        described = super().describe()
        base_url = self._base_url()
        if base_url:
            described["base_url"] = base_url
        return described

    def _build(self, spec: ModelSpec) -> Any:
        cls = self._import("langchain_openai", "ChatOpenAI", "langchain-openai")
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "api_key": self._require(self._api_key(), self._key_env, self._key_url),
            "temperature": spec.temperature,
            **self._token_kwargs(spec),
        }
        base_url = self._base_url()
        if base_url:
            kwargs["base_url"] = base_url
        return cls(**kwargs)


@register_chat
class OpenAIChatProvider(_OpenAICompatibleProvider):
    """OpenAI, or any gateway exposed through OPENAI_BASE_URL.

    Note that langchain-openai strips `temperature` for gpt-5.x models, whose
    API rejects the parameter. Extraction still behaves deterministically
    there; it just is not our doing.
    """

    name = "openai"
    # Verified against developers.openai.com/api/docs/models (Aug 2026);
    # the gpt-4o family is superseded by the 5.6 series.
    default_extraction_model = "gpt-5.6-luna"   # cost-optimised
    default_synthesis_model = "gpt-5.6-sol"     # flagship

    def _api_key(self) -> str:
        return self._settings.openai_api_key

    def _base_url(self) -> str | None:
        return self._settings.openai_base_url.strip() or None


@register_chat
class MoonshotChatProvider(_OpenAICompatibleProvider):
    """Moonshot (Kimi), which implements the OpenAI wire format."""

    name = "moonshot"
    default_extraction_model = "kimi-k2.5"
    default_synthesis_model = "kimi-k2.5"
    _key_env = "MOONSHOT_API_KEY"
    _key_url = "https://platform.moonshot.ai"

    def _api_key(self) -> str:
        return self._settings.moonshot_api_key

    def _base_url(self) -> str | None:
        return self._settings.moonshot_base_url.strip() or None


@register_chat
class AnthropicChatProvider(ChatProvider):
    """Anthropic's API through langchain-anthropic (wraps the official SDK)."""

    name = "anthropic"
    # Extraction runs once per transcript chunk, so it uses the cheap tier;
    # synthesis is a single call and gets the strongest model. Override with
    # EXTRACTION_MODEL / SYNTHESIS_MODEL to change either.
    default_extraction_model = "claude-haiku-4-5"
    default_synthesis_model = "claude-opus-5"
    # Native structured output; needs langchain-anthropic >= 1.1.0.
    structured_output_method = "json_schema"

    def _build(self, spec: ModelSpec) -> Any:
        cls = self._import(
            "langchain_anthropic", "ChatAnthropic", "langchain-anthropic"
        )
        return cls(
            model=spec.model,
            api_key=self._require(
                self._settings.anthropic_api_key,
                "ANTHROPIC_API_KEY",
                "https://console.anthropic.com/settings/keys",
            ),
            temperature=spec.temperature,
            **self._token_kwargs(spec),
        )


@register_chat
class GeminiChatProvider(ChatProvider):
    """Google Gemini via langchain-google-genai."""

    name = "gemini"
    # Verified against ai.google.dev/gemini-api/docs/models (Aug 2026).
    # The 2.5 family is scheduled for shutdown on 2026-10-16, so it is not
    # used as a default.
    default_extraction_model = "gemini-3.5-flash-lite"
    default_synthesis_model = "gemini-3.7-flash"

    def _build(self, spec: ModelSpec) -> Any:
        cls = self._import(
            "langchain_google_genai",
            "ChatGoogleGenerativeAI",
            "langchain-google-genai",
        )
        return cls(
            model=spec.model,
            api_key=self._require(
                self._settings.google_api_key,
                "GOOGLE_API_KEY",
                "https://aistudio.google.com/apikey",
            ),
            temperature=spec.temperature,
            # `max_tokens` is only an alias here; pass the real field name.
            max_output_tokens=spec.max_tokens,
        )


@register_chat
class OllamaChatProvider(ChatProvider):
    """Local models through an Ollama server. No API key involved."""

    name = "ollama"
    default_extraction_model = "llama3.1"
    default_synthesis_model = "llama3.1"
    #: Ollama exposes the output cap as `num_predict`.
    max_tokens_kwarg = "num_predict"
    #: Local models are often weak at tool calling; Ollama's native structured
    #: output (its `format` parameter) is the more reliable path.
    structured_output_method = "json_schema"

    def describe(self) -> dict:
        return {**super().describe(), "base_url": self._settings.ollama_base_url}

    def _build(self, spec: ModelSpec) -> Any:
        cls = self._import("langchain_ollama", "ChatOllama", "langchain-ollama")
        return cls(
            model=spec.model,
            base_url=self._settings.ollama_base_url,
            temperature=spec.temperature,
            **self._token_kwargs(spec),
        )
