"""Per-provider construction tests.

These build a real vendor client object for every provider - no network call is
made, since LangChain chat models validate and store configuration at
construction and only reach out on `.invoke()`. That is enough to catch the
failure this refactor is most exposed to: passing a keyword a vendor does not
accept (`max_tokens` to Ollama, say), which no amount of registry testing would
reveal.
"""
from __future__ import annotations

import pytest

from backend.llm.providers import ProviderError, create_chat

# Every provider, the credential it needs, and the attribute proving the model
# name reached the client. Attribute names differ per integration.
CASES = [
    pytest.param("openai", {"openai_api_key": "sk-test"}, "model_name", id="openai"),
    pytest.param(
        "moonshot", {"moonshot_api_key": "sk-test"}, "model_name", id="moonshot"
    ),
    pytest.param(
        "anthropic", {"anthropic_api_key": "sk-ant-test"}, "model", id="anthropic"
    ),
    pytest.param("gemini", {"google_api_key": "AIza-test"}, "model", id="gemini"),
    pytest.param("ollama", {}, "model", id="ollama"),
]


@pytest.mark.parametrize("name,creds,model_attr", CASES)
def test_provider_builds_both_clients(name, creds, model_attr, settings_factory):
    settings = settings_factory(llm_provider=name, **creds)
    provider = create_chat(name, settings)

    for role, build in (
        ("extraction", provider.extraction_llm),
        ("synthesis", provider.synthesis_llm),
    ):
        client = build()
        resolved = getattr(client, model_attr)
        # Gemini normalises to "models/<id>"; compare on the suffix.
        assert provider.model_for(role) in str(resolved)


@pytest.mark.parametrize("name,creds,model_attr", CASES)
def test_extraction_temperature_is_zero_where_supported(
    name, creds, model_attr, settings_factory
):
    """Extraction must be deterministic wherever temperature is honoured.

    Some models reject the parameter outright - OpenAI's gpt-5.x reasoning
    models are stripped of it by langchain-openai - and None means exactly
    that, so it is an acceptable outcome. A nonzero value is never acceptable.
    """
    settings = settings_factory(llm_provider=name, **creds)
    temperature = create_chat(name, settings).extraction_llm().temperature
    assert temperature in (0.0, None)


def test_openai_reasoning_models_drop_temperature(settings_factory):
    """Documents vendor behaviour we rely on rather than fight.

    langchain-openai removes `temperature` for gpt-5.x because the API rejects
    it. Determinism there comes from the model, not from our parameters.
    """
    reasoning = create_chat(
        "openai",
        settings_factory(openai_api_key="sk-test", extraction_model="gpt-5.6-luna"),
    ).extraction_llm()
    classic = create_chat(
        "openai",
        settings_factory(openai_api_key="sk-test", extraction_model="gpt-4o-mini"),
    ).extraction_llm()
    assert reasoning.temperature is None
    assert classic.temperature == 0.0


def test_bedrock_builds_without_explicit_credentials(settings_factory):
    """Bedrock uses the boto3 chain, so construction must not demand a key."""
    settings = settings_factory(llm_provider="bedrock", aws_region="us-west-2")
    client = create_chat("bedrock", settings).extraction_llm()
    assert client.region_name == "us-west-2"


def test_ollama_receives_num_predict_not_max_tokens(settings_factory):
    """Regression guard: ChatOllama has no `max_tokens` field."""
    provider = create_chat("ollama", settings_factory(llm_provider="ollama"))
    client = provider.extraction_llm()
    assert client.num_predict == provider.spec("extraction").max_tokens


def test_gemini_receives_max_output_tokens(settings_factory):
    provider = create_chat(
        "gemini", settings_factory(google_api_key="AIza-test")
    )
    client = provider.extraction_llm()
    assert client.max_output_tokens == provider.spec("extraction").max_tokens


def test_ollama_base_url_is_configurable(settings_factory):
    settings = settings_factory(ollama_base_url="http://gpu-box:11434")
    client = create_chat("ollama", settings).extraction_llm()
    assert "gpu-box" in str(client.base_url)


def test_openai_base_url_override_enables_gateways(settings_factory):
    """An OpenAI-compatible gateway is reachable without a new provider class."""
    settings = settings_factory(
        openai_api_key="sk-test", openai_base_url="https://gateway.internal/v1"
    )
    client = create_chat("openai", settings).extraction_llm()
    assert "gateway.internal" in str(client.openai_api_base)


def test_moonshot_defaults_to_the_moonshot_endpoint(settings_factory):
    settings = settings_factory(moonshot_api_key="sk-test")
    provider = create_chat("moonshot", settings)
    assert "moonshot" in provider.describe()["base_url"]


# --- credential errors -----------------------------------------------------
MISSING_KEY_CASES = [
    pytest.param("openai", "OPENAI_API_KEY", id="openai"),
    pytest.param("moonshot", "MOONSHOT_API_KEY", id="moonshot"),
    pytest.param("anthropic", "ANTHROPIC_API_KEY", id="anthropic"),
    pytest.param("gemini", "GOOGLE_API_KEY", id="gemini"),
]


@pytest.mark.parametrize("name,env_var", MISSING_KEY_CASES)
def test_missing_credential_names_the_variable_to_set(name, env_var, settings_factory):
    provider = create_chat(name, settings_factory(llm_provider=name))
    with pytest.raises(ProviderError) as exc:
        provider.extraction_llm()
    message = str(exc.value)
    assert env_var in message
    # The message must tell the user where to get one.
    assert "http" in message


@pytest.mark.parametrize("name,env_var", MISSING_KEY_CASES)
def test_whitespace_only_credential_is_treated_as_missing(
    name, env_var, settings_factory
):
    field = env_var.lower()
    provider = create_chat(name, settings_factory(**{field: "   "}))
    with pytest.raises(ProviderError, match=env_var):
        provider.extraction_llm()


def test_ollama_needs_no_credential(settings_factory):
    """Local inference must work with no key configured at all."""
    create_chat("ollama", settings_factory()).extraction_llm()


# --- model resolution ------------------------------------------------------
def test_generic_override_applies_to_any_provider(settings_factory):
    settings = settings_factory(
        openai_api_key="sk-test",
        extraction_model="my-fast-model",
        synthesis_model="my-strong-model",
    )
    provider = create_chat("openai", settings)
    assert provider.model_for("extraction") == "my-fast-model"
    assert provider.model_for("synthesis") == "my-strong-model"


def test_provider_defaults_apply_when_no_override(settings_factory):
    provider = create_chat("gemini", settings_factory(google_api_key="AIza-test"))
    assert provider.model_for("extraction") == provider.default_extraction_model


def test_bedrock_honours_legacy_model_id_settings(settings_factory):
    """Existing deployments set EXTRACTION_MODEL_ID; that must keep working."""
    settings = settings_factory(
        extraction_model_id="legacy.fast", synthesis_model_id="legacy.strong"
    )
    provider = create_chat("bedrock", settings)
    assert provider.model_for("extraction") == "legacy.fast"
    assert provider.model_for("synthesis") == "legacy.strong"


def test_generic_override_beats_legacy_bedrock_setting(settings_factory):
    settings = settings_factory(
        extraction_model_id="legacy.fast", extraction_model="generic.fast"
    )
    provider = create_chat("bedrock", settings)
    assert provider.model_for("extraction") == "generic.fast"


def test_bedrock_disables_streaming_only_for_providers_that_lack_it(
    settings_factory,
):
    unsupported = create_chat(
        "bedrock", settings_factory(extraction_model_id="zai.glm-4.7-flash")
    ).extraction_llm()
    supported = create_chat(
        "bedrock", settings_factory(extraction_model_id="anthropic.claude-3-haiku")
    ).extraction_llm()
    assert unsupported.disable_streaming is True
    assert supported.disable_streaming is False


# --- structured output -----------------------------------------------------
class _Recorder:
    """Captures how `with_structured_output` was called."""

    def __init__(self):
        self.kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.kwargs = kwargs
        return "bound"


@pytest.mark.parametrize(
    "name,expected_method",
    [
        ("bedrock", None),
        ("openai", "function_calling"),
        ("moonshot", "function_calling"),
        ("anthropic", "json_schema"),
        ("gemini", None),
        ("ollama", "json_schema"),
    ],
)
def test_schema_binding_uses_the_right_method(
    name, expected_method, settings_factory
):
    provider = create_chat(name, settings_factory())
    recorder = _Recorder()
    assert provider.bind_schema(recorder, dict) == "bound"
    if expected_method is None:
        # Unset means "let the integration decide" - pass no method at all.
        assert recorder.kwargs == {}
    else:
        assert recorder.kwargs == {"method": expected_method}


@pytest.mark.parametrize(
    "module,cls",
    [
        ("langchain_aws", "ChatBedrockConverse"),
        ("langchain_openai", "ChatOpenAI"),
        ("langchain_anthropic", "ChatAnthropic"),
        ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        ("langchain_ollama", "ChatOllama"),
    ],
)
def test_configured_structured_output_methods_are_actually_supported(module, cls):
    """Guard against a method name the installed integration would reject."""
    import inspect
    import re

    target = getattr(__import__(module, fromlist=[cls]), cls)
    source = inspect.getsource(target.with_structured_output)
    supported = set(re.findall(r'"(function_calling|json_mode|json_schema)"', source))
    assert {"function_calling", "json_schema"} <= supported


# --- regressions found by adversarial review -------------------------------
def test_generic_override_wins_even_when_equal_to_the_default(settings_factory):
    """Regression: the override was detected by comparing against the default.

    So setting EXTRACTION_MODEL to the same string as the provider default made
    the legacy EXTRACTION_MODEL_ID win instead - and that string is exactly what
    a user would type, since the two share a value.
    """
    from backend.llm.providers.chat import BedrockChatProvider

    default = BedrockChatProvider.default_extraction_model
    settings = settings_factory(
        extraction_model=default, extraction_model_id="legacy.fast"
    )
    provider = create_chat("bedrock", settings)
    assert provider.model_for("extraction") == default
    # The wrong value must not reach the client either.
    assert provider.extraction_llm().model_id == default


def test_synthesis_generic_override_wins_when_equal_to_default(settings_factory):
    from backend.llm.providers.chat import BedrockChatProvider

    default = BedrockChatProvider.default_synthesis_model
    settings = settings_factory(
        synthesis_model=default, synthesis_model_id="legacy.strong"
    )
    assert create_chat("bedrock", settings).model_for("synthesis") == default


def test_blank_legacy_model_id_falls_back_to_the_default(settings_factory):
    """Regression: a blank legacy id produced ChatBedrockConverse(model='')."""
    provider = create_chat("bedrock", settings_factory(extraction_model_id="   "))
    assert provider.model_for("extraction") == provider.default_extraction_model


def test_legacy_model_id_is_stripped(settings_factory):
    provider = create_chat(
        "bedrock", settings_factory(extraction_model_id="  legacy.fast  ")
    )
    assert provider.model_for("extraction") == "legacy.fast"


def test_unknown_role_is_rejected(settings_factory):
    """model_for used to silently return the synthesis model for any junk role."""
    provider = create_chat("openai", settings_factory())
    with pytest.raises(ProviderError, match="Unknown role"):
        provider.model_for("bogus")
    with pytest.raises(ProviderError, match="Unknown role"):
        provider.spec("bogus")


def test_openai_compatible_base_cannot_be_instantiated(settings_factory):
    """Its credential hooks are abstract, so a forgetful subclass fails early."""
    from backend.llm.providers.chat import _OpenAICompatibleProvider

    with pytest.raises(TypeError):
        _OpenAICompatibleProvider(settings_factory())


def test_subclass_missing_credential_hooks_fails_at_instantiation(settings_factory):
    from backend.llm.providers.chat import _OpenAICompatibleProvider

    class _Incomplete(_OpenAICompatibleProvider):
        name = "incomplete-probe"
        default_extraction_model = "x"
        default_synthesis_model = "x"

    with pytest.raises(TypeError, match="_api_key|_base_url"):
        _Incomplete(settings_factory())


def test_provider_without_a_name_is_rejected_clearly():
    """Regression: raised AttributeError instead of the friendly ProviderError."""
    from backend.llm.providers.base import ChatProvider
    from backend.llm.providers.registry import register_chat

    class _NoName(ChatProvider):
        default_extraction_model = "x"
        default_synthesis_model = "x"

        def _build(self, spec):  # pragma: no cover - never built
            raise AssertionError

    with pytest.raises(ProviderError, match="non-empty"):
        register_chat(_NoName)


def test_registered_names_are_always_resolvable():
    """Regression: a mixed-case name was listed as available but unresolvable."""
    from backend.llm.providers import available_chat
    from backend.llm.providers.base import ChatProvider
    from backend.llm.providers.registry import _CHAT, register_chat

    class _Upper(ChatProvider):
        name = "MixedCaseProbe"
        default_extraction_model = "x"
        default_synthesis_model = "x"

        def _build(self, spec):  # pragma: no cover - never built
            raise AssertionError

    register_chat(_Upper)
    try:
        for listed in available_chat():
            # Everything advertised must actually construct.
            assert create_chat(listed, Settings(_env_file=None)) is not None
    finally:
        _CHAT.pop("mixedcaseprobe", None)


from backend.config import Settings  # noqa: E402  (used above)
