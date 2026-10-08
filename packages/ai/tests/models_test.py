"""测试 ``Models`` 注册表、``create_provider`` 及模型辅助函数。"""

from __future__ import annotations

import pytest

from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import AuthCheck, AuthResult, Credential, ModelAuth, ProviderAuth
from pi_ai.models import (
    CreateProviderOptions,
    ModelsError,
    ModelsStoreEntry,
    calculate_cost,
    clamp_thinking_level,
    create_models,
    create_provider,
    get_supported_thinking_levels,
    has_api,
    merge_headers,
    models_are_equal,
)
from pi_ai.models_store import InMemoryModelsStore
from pi_ai.auth.credential_store import InMemoryCredentialStore
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    ModelCostTier,
    ModelThinkingLevel,
    StopReason,
    StreamOptions,
    Tool,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream


class _AuthContext:
    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._env = env or {}

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


class _StubStreams:
    """记录收到的调用的最小 ProviderStreams 实现。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Model, object]] = []

    def stream(self, model, context, options=None):
        self.calls.append(("stream", model, options))
        stream = AssistantMessageEventStream()
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, stop_reason=StopReason.STOP, timestamp=0
        )
        from pi_ai.utils.events import done_event

        stream.push(done_event(StopReason.STOP, message))
        return stream

    def stream_simple(self, model, context, options=None):
        self.calls.append(("stream_simple", model, options))
        stream = AssistantMessageEventStream()
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, stop_reason=StopReason.STOP, timestamp=0
        )
        from pi_ai.utils.events import done_event

        stream.push(done_event(StopReason.STOP, message))
        return stream


def _model(**overrides) -> Model:
    base = {
        "id": "test-model",
        "name": "Test",
        "api": "openai-responses",
        "provider": "test",
        "base_url": "https://api.test",
        "input": ["text"],
        "reasoning": True,
        "context_window": 100000,
        "max_tokens": 4096,
    }
    base.update(overrides)
    return Model(**base)


def _provider(*, streams: _StubStreams | None = None, models: list[Model] | None = None, **overrides):
    options = {
        "id": "test",
        "name": "Test",
        "auth": ProviderAuth(api_key=env_api_key_auth("Test key", ["TEST_API_KEY"])),
        "models": models if models is not None else [_model()],
        "api": streams if streams is not None else _StubStreams(),
    }
    options.update(overrides)
    return create_provider(CreateProviderOptions(**options))


# ---------------------------------------------------------------------------------
# 测试 create_provider
# ---------------------------------------------------------------------------------


def test_create_provider_requires_at_least_one_implementation():
    with pytest.raises(ValueError, match='"api" is required'):
        create_provider(CreateProviderOptions(id="x", auth=ProviderAuth(), models=[_model()]))


def test_provider_lists_only_chat_models_from_get_models():
    provider = _provider(models=[_model()])
    assert [model.id for model in provider.get_models()] == ["test-model"]
    assert {model.id for model in provider.get_all_models()} == {"test-model"}


def test_provider_dispatches_by_api_when_given_a_map():
    streams = _StubStreams()
    provider = _provider(api={"openai-responses": streams})
    provider.stream(_model(), Context(messages=[]))
    assert streams.calls[0][0] == "stream"


async def test_provider_without_an_api_yields_a_stream_error_message():
    provider = _provider(api={"anthropic-messages": _StubStreams()})
    stream = provider.stream(_model(), Context(messages=[]))
    message = await stream.result()
    assert message.stop_reason == "error"
    assert "no API implementation" in (message.error_message or "")


async def test_provider_dynamic_models_overlay_and_restore():
    provider = _provider()
    store = InMemoryModelsStore()

    async def fetch(context):
        return [_model(id="dynamic")]

    provider._fetch_models = fetch  # noqa: SLF001 - 直接验证 refresh 路径
    from pi_ai.models import ModelsPublication, RefreshModelsContext

    async def publish(publication: ModelsPublication) -> bool:
        if publication.persist is not None:
            await store.write("test", publication.persist)
        if publication.update is not None:
            publication.update()
        return True

    await provider.refresh_models(RefreshModelsContext(allow_network=True, publish=publish))
    assert [model.id for model in provider.get_models()] == ["test-model", "dynamic"]

    stored = await store.read("test")
    assert stored is not None and [model.id for model in stored.models] == ["dynamic"]


async def test_provider_refresh_skips_network_when_disallowed():
    provider = _provider()
    called: list[bool] = []

    async def fetch(context):
        called.append(True)
        return []

    provider._fetch_models = fetch  # noqa: SLF001
    from pi_ai.models import RefreshModelsContext

    await provider.refresh_models(RefreshModelsContext(allow_network=False))
    assert called == []


# ---------------------------------------------------------------------------------
# Models 注册表
# ---------------------------------------------------------------------------------


async def test_models_registry_lookup_and_listing():
    models = create_models()
    models.set_provider(_provider())
    assert [model.id for model in models.get_models()] == ["test-model"]
    assert models.get_model("test", "test-model") is not None
    assert models.get_model("test", "missing") is None
    assert models.get_provider("test") is not None
    assert models.get_provider("missing") is None
    models.delete_provider("test")
    assert models.get_providers() == []


async def test_models_get_models_is_best_effort_for_throwing_providers():
    class _Bad:
        id = "bad"
        name = "Bad"
        auth = ProviderAuth()
        base_url = None
        headers = None

        def get_models(self):
            raise RuntimeError("boom")

        def get_all_models(self):
            raise RuntimeError("boom")

    models = create_models()
    models.set_provider(_Bad())
    models.set_provider(_provider())
    assert [model.id for model in models.get_models()] == ["test-model"]


async def test_models_complete_returns_the_stream_result():
    models = create_models()
    models.set_provider(_provider())
    message = await models.complete(_model(), Context(messages=[]), StreamOptions(api_key="k"))
    assert message.stop_reason == "stop"


async def test_models_stream_reports_unknown_provider_as_stream_error():
    models = create_models()
    message = await models.complete(_model(), Context(messages=[]), StreamOptions(api_key="k")).__await__().__next__() if False else None
    stream = models.stream(_model(), Context(messages=[]))
    result = await stream.result()
    assert result.stop_reason == "error"
    assert "Unknown provider" in (result.error_message or "")


async def test_models_complete_reports_unconfigured_provider():
    models = create_models()
    models.set_provider(_provider())
    result = await models.complete(_model(), Context(messages=[]))
    assert result.stop_reason == "error"
    assert "not configured" in (result.error_message or "")


async def test_models_applies_env_api_key_and_headers():
    streams = _StubStreams()
    models = create_models()
    models.set_provider(_provider(streams=streams))
    stream = models.stream(_model(), Context(messages=[]), StreamOptions(api_key="secret"))
    await stream.result()
    _, request_model, request_options = streams.calls[0]
    assert request_options.api_key == "secret"


async def test_models_applies_model_base_url_override_from_auth_resolution():
    streams = _StubStreams()

    class _Auth:
        name = "Custom"

        async def resolve(self, *, ctx, credential=None, signal):
            return AuthResult(auth=ModelAuth(api_key="k", base_url="https://override.test"), source="custom")

    models = create_models()
    models.set_provider(_provider(streams=streams, auth=ProviderAuth(api_key=_Auth())))
    await models.stream(_model(), Context(messages=[])).result()
    _, request_model, _ = streams.calls[0]
    assert request_model.base_url == "https://override.test"


async def test_models_transform_headers_runs_last():
    streams = _StubStreams()
    models = create_models()
    models.set_provider(_provider(streams=streams))

    async def transform(headers):
        return {**headers, "x-added": "1"}

    await models.stream(_model(), Context(messages=[]), StreamOptions(api_key="k"), transform_headers=transform).result()
    _, _, request_options = streams.calls[0]
    assert request_options.headers["x-added"] == "1"


async def test_models_normalizes_context_before_dispatch():
    streams = _StubStreams()
    seen: dict[str, object] = {}

    class _Recording(_StubStreams):
        def stream(self, model, context, options=None):
            seen["messages"] = context.messages
            return super().stream(model, context, options)

    models = create_models()
    models.set_provider(_provider(streams=_Recording()))
    await models.stream(
        _model(),
        Context(system_prompt="sys", messages=[], tools=[Tool(name="t", description="", parameters={})]),
        StreamOptions(api_key="k"),
    ).result()
    messages = seen["messages"]
    assert messages[0].role == "system"
    assert messages[0].content == "sys"


async def test_check_auth_reports_env_source():
    models = create_models()
    models._auth_context = _AuthContext({"TEST_API_KEY": "abc"})  # noqa: SLF001
    models.set_provider(_provider())
    check = await models.check_auth("test")
    assert check is not None and check.type == "api_key" and check.source == "TEST_API_KEY"
    assert await models.check_auth("missing") is None


async def test_get_available_filters_unconfigured_providers():
    models = create_models()
    models.set_provider(_provider())
    assert await models.get_available() == []
    models._auth_context = _AuthContext({"TEST_API_KEY": "abc"})  # noqa: SLF001
    assert [model.id for model in await models.get_available()] == ["test-model"]


async def test_login_and_logout_round_trip():
    from pi_ai.auth.types import ApiKeyCredential, AuthPromptSecret

    class _Interaction:
        signal = None

        async def prompt(self, prompt):
            return "typed-key"

        def notify(self, event):
            pass

    models = create_models()
    models.set_provider(_provider())
    credential = await models.login("test", "api_key", _Interaction())
    assert isinstance(credential, ApiKeyCredential) and credential.key == "typed-key"
    assert await models.get_auth("test") is not None
    await models.logout("test")
    assert await models.get_auth("test") is None


async def test_login_rejects_unknown_provider_and_unsupported_type():
    models = create_models()
    with pytest.raises(ModelsError) as unknown:
        await models.login("nope", "api_key", _Interaction())
    assert unknown.value.code == "provider"

    models.set_provider(_provider())
    with pytest.raises(ModelsError) as unsupported:
        await models.login("test", "oauth", _Interaction())
    assert unsupported.value.code == "auth"


class _Interaction:
    signal = None

    async def prompt(self, prompt):
        return "typed-key"

    def notify(self, event):
        pass


# ---------------------------------------------------------------------------------
# 纯函数辅助
# ---------------------------------------------------------------------------------


def test_calculate_cost_uses_base_rates():
    model = _model(cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75))
    usage = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000)
    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(3.0)
    assert cost.output == pytest.approx(15.0)
    assert cost.cache_read == pytest.approx(0.3)
    assert cost.cache_write == pytest.approx(3.75)
    assert cost.total == pytest.approx(22.05)


def test_calculate_cost_applies_the_highest_matching_tier():
    model = _model(
        cost=ModelCost(
            input=1.0,
            output=2.0,
            cache_read=0.1,
            cache_write=1.0,
            tiers=[ModelCostTier(input=10.0, output=20.0, cache_read=1.0, cache_write=10.0, input_tokens_above=1000)],
        )
    )
    usage = Usage(input=2000, output=1000)
    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(0.02)
    assert cost.output == pytest.approx(0.02)


def test_calculate_cost_charges_double_input_for_one_hour_cache_writes():
    model = _model(cost=ModelCost(input=1.0, output=0.0, cache_read=0.0, cache_write=2.0))
    usage = Usage(cache_write=1_000_000, cache_write_1h=1_000_000)
    cost = calculate_cost(model, usage)
    assert cost.cache_write == pytest.approx(2.0 * 0 + 1.0 * 2 * 1_000_000 / 1_000_000)


def test_supported_thinking_levels_for_non_reasoning_model():
    assert get_supported_thinking_levels(_model(reasoning=False)) == ["off"]


def test_supported_thinking_levels_honours_the_map():
    model = _model(thinking_level_map={"off": None, "xhigh": "xhigh"})
    levels = get_supported_thinking_levels(model)
    assert "off" not in levels
    assert "minimal" in levels
    assert "xhigh" in levels
    assert "max" not in levels


def test_clamp_thinking_level_moves_to_the_nearest_supported_level():
    # 只有 `minimal` 被显式排除；`off` 因未出现在映射中而保留可用。
    model = _model(thinking_level_map={"minimal": None})
    assert get_supported_thinking_levels(model) == ["off", "low", "medium", "high"]
    assert clamp_thinking_level(model, "minimal") == "low"
    assert clamp_thinking_level(model, "high") == "high"

    # `high` 是支持的最高档，因此 `xhigh` 向下收敛到它。
    capped = _model(thinking_level_map={"high": "high", "xhigh": None, "max": None})
    assert get_supported_thinking_levels(capped) == ["off", "minimal", "low", "medium", "high"]
    assert clamp_thinking_level(capped, "xhigh") == "high"
    assert clamp_thinking_level(capped, "max") == "high"

    # 未知档位回退到第一个受支持的档位。
    assert clamp_thinking_level(capped, "bogus") == "off"


def test_has_api_narrows_only_chat_models():
    model = _model()
    assert has_api(model, "openai-responses")
    assert not has_api(model, "anthropic-messages")


def test_models_are_equal_compares_type_id_and_provider():
    left = _model()
    right = _model()
    assert models_are_equal(left, right)
    assert not models_are_equal(left, _model(provider="other"))
    assert not models_are_equal(left, None)
    assert not models_are_equal(None, None)


def test_merge_headers_is_case_insensitive_and_override_wins():
    merged = merge_headers({"X-A": "1", "B": "2"}, {"x-a": "9"})
    assert merged == {"x-a": "9", "B": "2"}
    assert merge_headers(None, None) is None
