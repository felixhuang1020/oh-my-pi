"""移植 ``packages/ai/test/providers.test.ts``。

``createProvider``/``fauxProvider`` 的桩沿用 ``models_test.py`` 的模式：本地
``_AuthContext``、记录型流和手写交互，确保没有测试触碰网络或 AWS/GCP SDK。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from pi_ai.api.lazy import lazy_api
from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import (
    ApiKeyCredential,
    AuthResult,
    ModelAuth,
    ProviderAuth,
)
from pi_ai.compat import get_model as get_compat_model
from pi_ai.compat import get_models as get_compat_models
from pi_ai.models import (
    CreateModelsOptions,
    CreateProviderOptions,
    ModelsRefreshOptions,
    create_models,
    create_provider,
    get_supported_thinking_levels,
)
from pi_ai.models_store import InMemoryModelsStore
from pi_ai.providers.all import (
    builtin_models,
    builtin_providers,
    get_builtin_model,
    get_builtin_models,
)
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.catalog import chat_model_catalog, provider_models
from pi_ai.providers.faux import (
    FauxDeferredOptions,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_provider,
)
from pi_ai.types import (
    Context,
    DeferredFetchOptions,
    DeferredHandle,
    Model,
    ModelCost,
    ProviderRequestOptions,
    SimpleStreamOptions,
    StopReason,
    UserMessage,
)
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.events import done_event, start_event
from pi_ai.utils.serde import to_json
from pi_ai.utils.transcript import normalize_context

#: 上游 ``Date.now()`` 的确定性替代。
NOW = 1_700_000_000_000

context = normalize_context(Context(messages=[UserMessage(content="hi", timestamp=NOW)]))


# ---------------------------------------------------------------------------------
# 测试桩（Fakes）
# ---------------------------------------------------------------------------------


class _AuthContext:
    """上游的 ``fakeAuthContext(env, files)``。"""

    def __init__(self, env: dict[str, str] | None = None, files: list[str] | None = None) -> None:
        self._env = env or {}
        self._files = files or []

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return path in self._files


class _Interaction:
    """脚本化的 ``ProviderAuthInteraction``，记录提示与事件。"""

    def __init__(self, answers: list[str] | None = None, signal: AbortSignal | None = None) -> None:
        self.answers = list(answers or [])
        self.signal = signal or AbortSignal()
        self.prompt_types: list[str] = []
        self.events: list[object] = []

    async def prompt(self, prompt):
        self.prompt_types.append(prompt.type)
        return self.answers.pop(0)

    def notify(self, event) -> None:
        self.events.append(event)


class _ResolveAuth:
    """始终已配置、且不解析出 model auth 的 api-key 认证。"""

    name = "Test"

    async def resolve(self, *, ctx=None, credential=None, signal=None) -> AuthResult:
        return AuthResult(auth=ModelAuth())


class _RecordingStreams:
    """上游的 ``recordingStreams(label, calls)``。"""

    def __init__(self, label: str, calls: list[str]) -> None:
        self.label = label
        self.calls = calls

    def _respond(self, model: Model) -> AssistantMessageEventStream:
        self.calls.append(f"{self.label}:{model.id}")
        stream = AssistantMessageEventStream()
        message = faux_assistant_message("ok")
        stream.push(start_event(message))
        stream.push(done_event(StopReason.STOP, message))
        return stream

    def stream(self, model, context, options=None):
        return self._respond(model)

    def stream_simple(self, model, context, options=None):
        return self._respond(model)


def _test_model(api: str, model_id: str) -> Model:
    """上游的 ``testModel(api, id)``。"""
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider="mixed",
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10000,
        max_tokens=1000,
    )


def _compat_flag(model: Model | None, name: str):
    if model is None or model.compat is None:
        return None
    return getattr(model.compat, name)


# ---------------------------------------------------------------------------------
# 内置 provider（builtin providers）
# ---------------------------------------------------------------------------------


def test_builtin_models_registers_every_builtin_provider_with_models():
    models = builtin_models()
    providers = models.get_providers()
    assert len(providers) == len(builtin_providers())
    assert "anthropic" in [provider.id for provider in providers]

    anthropic = models.get_model("anthropic", "claude-haiku-4-5")
    assert anthropic is not None
    assert anthropic.api == "anthropic-messages"

    all_models = models.get_models()
    assert len(all_models) > 100

    for provider in providers:
        listed = models.get_all_models(provider.id)
        assert len(listed) > 0
        assert all(model.provider == provider.id for model in listed)


def test_returns_empty_results_for_unknown_provider_ids():
    unknown_provider = "not-a-provider"
    unknown_model = "x"

    assert get_builtin_model(unknown_provider, unknown_model) is None
    assert get_builtin_models(unknown_provider) == []
    assert provider_models(unknown_provider) == []
    assert chat_model_catalog(unknown_provider) == {}
    assert get_compat_model(unknown_provider, unknown_model) is None
    assert get_compat_models(unknown_provider) == []


def test_stores_native_constrained_sampling_capabilities_in_model_metadata():
    gpt4o = get_builtin_model("openai", "gpt-4o")
    assert gpt4o is not None and gpt4o.compat is not None
    assert gpt4o.compat.supports_strict_mode is True
    assert gpt4o.compat.supports_openai_grammar_tools is None

    gpt54 = get_builtin_model("openai", "gpt-5.4")
    assert gpt54 is not None and gpt54.compat is not None
    assert gpt54.compat.supports_strict_mode is True
    assert gpt54.compat.supports_openai_grammar_tools is True

    haiku = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert haiku is not None and haiku.compat is not None
    assert haiku.compat.supports_strict_tools is True


def test_records_known_direct_provider_request_limits():
    haiku = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert haiku is not None and haiku.input_limits is not None
    assert haiku.input_limits.max_request_bytes == 32 * 1024 * 1024

    gpt4o = get_builtin_model("openai", "gpt-4o")
    assert gpt4o is not None and gpt4o.input_limits is not None
    assert gpt4o.input_limits.max_request_bytes == 512 * 1024 * 1024

    gemini = get_builtin_model("google", "gemini-2.5-flash")
    assert gemini is not None and gemini.input_limits is not None
    assert gemini.input_limits.max_request_bytes == 20 * 1024 * 1024


def test_uses_models_dev_effort_levels_for_google_thinking_models():
    # 针对 https://github.com/earendil-works/pi/issues/9455 的回归测试
    provider = "google"
    flash36 = get_builtin_model(provider, "gemini-3.6-flash")
    assert flash36 is not None
    assert "minimal" in get_supported_thinking_levels(flash36)

    flash38 = get_builtin_model(provider, "gemini-3.8-flash")
    assert flash38 is not None
    assert get_supported_thinking_levels(flash38) == ["low", "medium", "high"]

    pro = get_builtin_model(provider, "gemini-3.1-pro-preview")
    assert pro is not None
    assert get_supported_thinking_levels(pro) == ["low", "medium", "high"]

    gemma = get_builtin_model("google", "gemma-4-31b-it")
    assert gemma is not None
    assert get_supported_thinking_levels(gemma) == ["minimal", "high"]


def test_enables_mid_conversation_system_messages_only_for_verified_models():
    models = builtin_models()
    supported = [
        ("moonshotai", "kimi-k2.6"),
        ("moonshotai", "kimi-k2.7-code"),
        ("moonshotai", "kimi-k2.7-code-highspeed"),
        ("moonshotai", "kimi-k3"),
        ("moonshotai-cn", "kimi-k2.6"),
        ("moonshotai-cn", "kimi-k2.7-code"),
        ("moonshotai-cn", "kimi-k2.7-code-highspeed"),
        ("moonshotai-cn", "kimi-k3"),
        ("openai", "gpt-5.4"),
        ("openai", "gpt-5.5"),
        ("openai", "gpt-6-astra"),
        ("anthropic", "claude-opus-5"),
        ("deepseek", "deepseek-v4-pro"),
    ]
    unsupported = [
        ("openai", "gpt-4.1"),
        ("openai", "gpt-5.2"),
        ("anthropic", "claude-sonnet-4-5"),
        ("google", "gemini-2.5-pro"),
        ("google", "gemini-2.5-flash"),
        ("deepseek", "deepseek-flash"),
        ("minimax", "MiniMax-M3"),
    ]
    for provider, model_id in supported:
        model = models.get_model(provider, model_id)
        assert _compat_flag(model, "supports_mid_convo_system_messages") is True, (
            f"{provider}/{model_id}"
        )
    for provider, model_id in unsupported:
        model = models.get_model(provider, model_id)
        assert _compat_flag(model, "supports_mid_convo_system_messages") is None, (
            f"{provider}/{model_id}"
        )


def test_routes_proxied_tool_changes_through_verified_transports_only():
    models = builtin_models()

    # Anthropic 直接端点支持对话中途替换 tool。
    assert (
        _compat_flag(models.get_model("anthropic", "claude-opus-5"), "supports_mid_convo_tool_changes")
        is True
    )

    # 只支持中途插入 system 消息、不支持中途 tool 变更的模型。
    assert (
        _compat_flag(models.get_model("moonshotai", "kimi-k2.6"), "supports_mid_convo_tool_changes")
        is None
    )


def test_uses_official_kimi_k3_pricing_for_moonshot_providers():
    models = builtin_models()
    for provider in ("moonshotai", "moonshotai-cn"):
        model = models.get_model(provider, "kimi-k3")
        assert model is not None
        assert to_json(model.cost) == {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 0}


# ---------------------------------------------------------------------------------
# provider 认证流程（provider auth flows）
# ---------------------------------------------------------------------------------


async def test_resolves_anthropic_bearer_auth_from_env_with_auth_token_precedence():
    models = create_models(
        CreateModelsOptions(
            auth_context=_AuthContext(
                {
                    "ANTHROPIC_AUTH_TOKEN": "auth-token",
                    "ANTHROPIC_API_KEY": "api-key",
                }
            )
        )
    )
    models.set_provider(anthropic_provider())

    result = await models.get_auth("anthropic")
    assert to_json(result) == {
        "auth": {"headers": {"Authorization": "Bearer auth-token"}},
        "source": "ANTHROPIC_AUTH_TOKEN",
    }


async def test_prefers_the_stored_credential_key_and_falls_back_through_env_vars_in_order():
    auth = env_api_key_auth("Test key", ["FIRST_KEY", "SECOND_KEY"])

    stored = await auth.resolve(
        ctx=_AuthContext({"FIRST_KEY": "env"}),
        credential=ApiKeyCredential(type="api_key", key="stored"),
        signal=AbortSignal(),
    )
    assert stored is not None
    assert stored.auth.api_key == "stored"
    assert stored.source == "stored credential"

    second = await auth.resolve(ctx=_AuthContext({"SECOND_KEY": "second"}), signal=AbortSignal())
    assert second is not None
    assert second.auth.api_key == "second"
    assert second.source == "SECOND_KEY"

    assert await auth.resolve(ctx=_AuthContext({}), signal=AbortSignal()) is None


async def test_login_prompts_for_a_secret_and_returns_an_api_key_credential():
    auth = env_api_key_auth("Test key", ["TEST_KEY"])
    interaction = _Interaction(["entered-key"])
    credential = await auth.login(interaction)
    assert interaction.prompt_types == ["secret"]
    assert to_json(credential) == {"type": "api_key", "key": "entered-key"}


# ---------------------------------------------------------------------------------
# createProvider（创建 provider）
# ---------------------------------------------------------------------------------


async def test_lazily_exposes_only_declared_deferred_capabilities():
    loads = 0
    calls: list[str] = []
    streams = _RecordingStreams("deferred", calls)
    streams.fetch_deferred = lambda model, handle=None, options=None: streams.stream_simple(model, context)

    async def load():
        nonlocal loads
        loads += 1
        return streams

    api = lazy_api(load, fetch_deferred=True)
    model = _test_model("api-a", "model-a")
    handle = DeferredHandle(provider=model.provider, model_id=model.id, api=model.api, id="response-1")

    assert loads == 0
    assert api.cancel_deferred is None
    assert (await api.fetch_deferred(model, handle).result()).stop_reason == "stop"
    assert loads == 1


async def test_dispatches_on_model_api_for_mixed_api_providers():
    calls: list[str] = []
    provider = create_provider(
        CreateProviderOptions(
            id="mixed",
            auth=ProviderAuth(api_key=_ResolveAuth()),
            models=[_test_model("api-a", "model-a"), _test_model("api-b", "model-b")],
            api={"api-a": _RecordingStreams("a", calls), "api-b": _RecordingStreams("b", calls)},
        )
    )
    models = create_models()
    models.set_provider(provider)

    await models.complete_simple(_test_model("api-a", "model-a"), context)
    await models.complete_simple(_test_model("api-b", "model-b"), context)
    assert calls == ["a:model-a", "b:model-b"]


async def test_merges_provider_resolved_env_into_stream_options():
    captured: dict[str, object] = {}

    class _EnvAuth:
        name = "Test"

        async def resolve(self, *, ctx=None, credential=None, signal=None) -> AuthResult:
            return AuthResult(
                auth=ModelAuth(api_key="provider-key"),
                env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
            )

    class _CaptureStreams:
        def stream(self, model, request_context, options=None):
            captured["env"] = options.env
            captured["api_key"] = options.api_key
            return _RecordingStreams("a", []).stream(model, request_context, options)

        def stream_simple(self, model, request_context, options=None):
            captured["env"] = options.env
            captured["api_key"] = options.api_key
            return _RecordingStreams("a", []).stream_simple(model, request_context, options)

    env_model = replace(_test_model("api-a", "model-a"), provider="env-provider")
    provider = create_provider(
        CreateProviderOptions(
            id="env-provider",
            auth=ProviderAuth(api_key=_EnvAuth()),
            models=[env_model],
            api=_CaptureStreams(),
        )
    )
    models = create_models()
    models.set_provider(provider)

    await models.complete_simple(
        env_model,
        context,
        SimpleStreamOptions(
            api_key="request-key",
            env={"REQUEST_ONLY": "request", "SHARED": "request"},
        ),
    )

    assert captured["api_key"] == "request-key"
    assert captured["env"] == {
        "PROVIDER_ONLY": "provider",
        "REQUEST_ONLY": "request",
        "SHARED": "request",
    }


async def test_applies_resolved_request_options_to_deferred_fetch_and_cancellation():
    captured: dict[str, object] = {}

    class _ResolvedAuth:
        name = "Test"

        async def resolve(self, *, ctx=None, credential=None, signal=None) -> AuthResult:
            return AuthResult(
                auth=ModelAuth(
                    api_key="provider-key",
                    base_url="https://resolved.test/v1",
                    headers={"Authorization": "Bearer provider", "X-Shared": "provider"},
                ),
                env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
            )

    deferred_model = replace(_test_model("api-a", "model-a"), provider="deferred-provider")
    streams = _RecordingStreams("deferred", [])

    def fetch_deferred(model, handle=None, options=None):
        captured["model"] = model
        captured["fetch"] = options
        return streams.stream_simple(model, context)

    async def cancel_deferred(model, handle=None, options=None):
        captured["cancel"] = options

    streams.fetch_deferred = fetch_deferred
    streams.cancel_deferred = cancel_deferred

    provider = create_provider(
        CreateProviderOptions(
            id="deferred-provider",
            auth=ProviderAuth(api_key=_ResolvedAuth()),
            models=[deferred_model],
            api=streams,
        )
    )
    models = create_models()
    models.set_provider(provider)
    handle = DeferredHandle(
        provider=deferred_model.provider,
        model_id=deferred_model.id,
        api=deferred_model.api,
        id="response-1",
    )

    await models.fetch_deferred(
        deferred_model,
        handle,
        DeferredFetchOptions(
            wait=50,
            timeout_ms=100,
            api_key="request-key",
            headers={"X-Request": "request", "x-shared": "request"},
            env={"REQUEST_ONLY": "request", "SHARED": "request"},
        ),
        transform_headers=lambda headers: {**headers, "X-Transformed": "yes"},
    )
    await models.cancel_deferred(
        deferred_model,
        handle,
        ProviderRequestOptions(timeout_ms=200),
        transform_headers=lambda headers: {**headers, "X-Cancel": "yes"},
    )

    fetched_model = captured["model"]
    assert fetched_model.base_url == "https://resolved.test/v1"

    fetched_options = captured["fetch"]
    assert fetched_options.wait == 50
    assert fetched_options.timeout_ms == 100
    assert fetched_options.api_key == "request-key"
    assert fetched_options.headers == {
        "Authorization": "Bearer provider",
        "X-Request": "request",
        "x-shared": "request",
        "X-Transformed": "yes",
    }
    assert fetched_options.env == {
        "PROVIDER_ONLY": "provider",
        "REQUEST_ONLY": "request",
        "SHARED": "request",
    }

    cancelled_options = captured["cancel"]
    assert cancelled_options.timeout_ms == 200
    assert cancelled_options.api_key == "provider-key"
    assert cancelled_options.headers == {
        "Authorization": "Bearer provider",
        "X-Shared": "provider",
        "X-Cancel": "yes",
    }
    assert cancelled_options.env == {"PROVIDER_ONLY": "provider", "SHARED": "provider"}


async def test_produces_a_stream_error_for_a_model_whose_api_has_no_implementation():
    provider = create_provider(
        CreateProviderOptions(
            id="mixed",
            auth=ProviderAuth(api_key=_ResolveAuth()),
            models=[_test_model("api-a", "model-a")],
            api={"api-a": _RecordingStreams("a", [])},
        )
    )
    result = await provider.stream_simple(_test_model("api-ghost", "model-x"), context).result()
    assert result.stop_reason == "error"
    assert "no API implementation" in result.error_message


async def test_lets_a_newer_dynamic_refresh_bypass_and_supersede_older_network_work():
    fetches = 0
    first_started = asyncio.Event()
    finish_first = asyncio.Event()

    async def fetch_models(_context):
        nonlocal fetches
        fetches += 1
        current = fetches
        if current == 1:
            first_started.set()
            await finish_first.wait()
        return [_test_model("api-a", f"listed-{current}")]

    provider = create_provider(
        CreateProviderOptions(
            id="dynamic",
            auth=ProviderAuth(api_key=_ResolveAuth()),
            models=[],
            fetch_models=fetch_models,
            api=_RecordingStreams("a", []),
        )
    )

    store = InMemoryModelsStore()
    models = create_models(CreateModelsOptions(models_store=store))
    models.set_provider(provider)
    assert provider.get_models() == []

    first = asyncio.ensure_future(models.refresh(ModelsRefreshOptions(providers=["dynamic"])))
    await first_started.wait()
    second = asyncio.ensure_future(models.refresh(ModelsRefreshOptions(providers=["dynamic"])))
    second_result = await second
    assert second_result.aborted is False
    first_result = await first
    assert first_result.aborted is False
    assert fetches == 2
    assert [model.id for model in provider.get_models()] == ["listed-2"]
    entry = await store.read("dynamic")
    assert entry is not None
    assert [model.id for model in entry.models] == ["listed-2"]

    finish_first.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert [model.id for model in provider.get_models()] == ["listed-2"]
    entry = await store.read("dynamic")
    assert entry is not None
    assert [model.id for model in entry.models] == ["listed-2"]


# ---------------------------------------------------------------------------------
# fauxProvider（faux 假 provider）
# ---------------------------------------------------------------------------------


async def test_streams_queued_responses_through_a_models_collection():
    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)
    faux.set_responses([faux_assistant_message("hello from faux")])

    model = models.get_models(faux.provider.id)[0]
    result = await models.complete_simple(model, context)
    assert result.stop_reason == "stop"
    assert to_json(result.content) == [{"type": "text", "text": "hello from faux"}]
    assert faux.state.call_count == 1


async def test_submits_polls_and_redeems_deferred_responses():
    faux = faux_provider(
        RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=1, poll_after_ms=25))
    )
    models = create_models()
    models.set_provider(faux.provider)
    faux.set_responses([faux_assistant_message("ready")])
    model = faux.get_model()
    assert model is not None

    submission = models.stream_simple(model, context, SimpleStreamOptions(deferred={"window": "1h"}))
    event_types = [event.type async for event in submission]
    deferred = await submission.result()
    assert event_types == ["start", "done"]
    assert deferred.stop_reason == "deferred"
    assert deferred.content == []

    handle = deferred.deferred
    assert handle is not None
    assert handle.provider == model.provider
    assert handle.model_id == model.id
    assert handle.api == model.api
    assert isinstance(handle.id, str) and handle.id != ""
    assert handle.poll_after_ms == 25

    pending = await models.fetch_deferred(model, handle)
    assert pending.stop_reason == "deferred"
    assert pending.deferred == handle

    ready = await models.fetch_deferred(model, handle, DeferredFetchOptions(wait=0))
    assert ready.stop_reason == "stop"
    assert to_json(ready.content) == [{"type": "text", "text": "ready"}]
    assert ready.usage.total_tokens > 0
    assert faux.state.call_count == 1
    assert faux.state.deferred_fetch_count == 2


async def test_records_cancellation_and_returns_deferred_fetch_failures_in_band():
    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)

    async def failing_response(_context, _options, _state, _model):
        raise RuntimeError("deferred failed")

    faux.set_responses([failing_response, faux_assistant_message("cancelled")])
    model = faux.get_model()
    assert model is not None

    failed_submission = await models.complete_simple(model, context, SimpleStreamOptions(deferred=True))
    assert failed_submission.deferred is not None
    failed = await models.fetch_deferred(model, failed_submission.deferred)
    assert failed.stop_reason == "error"
    assert failed.error_message == "deferred failed"

    cancelled_submission = await models.complete_simple(model, context, SimpleStreamOptions(deferred=True))
    assert cancelled_submission.deferred is not None
    await models.cancel_deferred(model, cancelled_submission.deferred)
    assert to_json(faux.state.cancelled_deferred) == [to_json(cancelled_submission.deferred)]
    cancelled = await models.fetch_deferred(model, cancelled_submission.deferred)
    assert cancelled.stop_reason == "error"
    assert "was cancelled" in (cancelled.error_message or "")
