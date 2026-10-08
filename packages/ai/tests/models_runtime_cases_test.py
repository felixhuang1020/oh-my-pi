"""移植自 ``packages/ai/test/models-runtime.test.ts``（"Models runtime"）。

上游文件构建假 provider/store，全面测试运行时 ``Models`` 集合：凭据枚举、
provider 注册表、refresh/发布代次、auth 解析以及请求级 auth 应用。
全部离线运行；上游测试套件同样不会触碰任何真实 provider。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from pi_ai.auth.credential_store import InMemoryCredentialStore
from pi_ai.auth.resolve import AuthResolutionOverrides
from pi_ai.auth.types import (
    ApiKeyCredential,
    AuthCheck,
    AuthOperationOptions,
    AuthResult,
    CredentialInfo,
    ModelAuth,
    ProviderAuth,
)
from pi_ai.models import (
    CreateModelsOptions,
    CreateProviderOptions,
    ModelsError,
    ModelsPublication,
    ModelsRefreshOptions,
    calculate_cost,
    create_models,
    create_provider,
    has_api,
)
from pi_ai.models_store import InMemoryModelsStore, ModelsStoreEntry
from pi_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    Model,
    ModelCost,
    ModelCostTier,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    Usage,
    UserMessage,
)
from pi_ai.utils.abort import AbortController, AbortError, AbortSignal
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.events import done_event, start_event
from pi_ai.utils.serde import to_json


# ---------------------------------------------------------------------------------
# 上游 fixture
# ---------------------------------------------------------------------------------


def _test_model(provider: str, id: str, **overrides: Any) -> Model:
    """上游 ``testModel(provider, id)``。"""
    fields: dict[str, Any] = {
        "id": id,
        "name": id,
        "api": "test-api",
        "provider": provider,
        "base_url": "https://example.test/v1",
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 10000,
        "max_tokens": 1000,
    }
    fields.update(overrides)
    return Model(**fields)


def _done_message(model: Model, text: str) -> AssistantMessage:
    """上游 ``doneMessage(model, text)``。"""
    return AssistantMessage(
        content=[TextContent(text=text)],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cost=Cost()),
        stop_reason=StopReason.STOP,
        timestamp=0,
    )


def _context() -> Context:
    """上游 ``context``。"""
    return Context(messages=[UserMessage(content="hi", timestamp=0)])


class _AmbientAuth:
    """上游 ``ambientAuth``：无任何 auth 值也报告 "configured"。"""

    name = "Ambient"

    async def resolve(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthResult:
        return AuthResult(auth=ModelAuth())


class _EnvKeyAuth:
    """上游 ``envKeyAuth(key)``：已存 key 优先，否则使用固定的环境 key。"""

    def __init__(self, key: str | None) -> None:
        self.name = "Test API key"
        self._key = key

    async def resolve(
        self, *, ctx: Any, credential: ApiKeyCredential | None = None, signal: AbortSignal
    ) -> AuthResult | None:
        resolved = (
            credential.key if credential is not None and credential.key is not None else self._key
        )
        if not resolved:
            return None
        return AuthResult(
            auth=ModelAuth(api_key=resolved),
            source="stored" if credential is not None else "env",
        )


@dataclass
class _ProviderCall:
    """上游 ``ProviderCall``。"""

    model: Model
    options: Any


class _FakeProvider:
    """上游 ``testProvider(input)``。

    ``refresh_models`` 只在调用方提供时才挂上，对应上游对象字面量中
    静态 provider 的该属性为 ``undefined`` 的写法。
    """

    def __init__(
        self,
        *,
        id: str,
        models: list[Model] | None = None,
        auth: ProviderAuth | None = None,
        get_models: Any = None,
        get_all_models: Any = None,
        refresh_models: Any = None,
        calls: list[_ProviderCall] | None = None,
    ) -> None:
        self.id = id
        self.name = id
        self.auth = auth if auth is not None else ProviderAuth(api_key=_AmbientAuth())
        self._models = list(models) if models is not None else [_test_model(id, "model-a")]
        self._get_models = get_models
        self._get_all_models = get_all_models
        self.calls: list[_ProviderCall] = calls if calls is not None else []
        if refresh_models is not None:
            self.refresh_models = refresh_models

    def get_models(self) -> list[Model]:
        source = self._get_models
        return list(source()) if source is not None else list(self._models)

    def get_all_models(self) -> list[Model]:
        source = self._get_all_models
        return list(source()) if source is not None else list(self._models)

    def _respond(self, model: Model, options: Any) -> AssistantMessageEventStream:
        self.calls.append(_ProviderCall(model=model, options=options))
        stream = AssistantMessageEventStream()
        message = _done_message(model, "ok")
        stream.push(start_event(message))
        stream.push(done_event(StopReason.STOP, message))
        stream.end(message)
        return stream

    def stream(self, model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        return self._respond(model, options)

    def stream_simple(
        self, model: Model, context: Any, options: Any = None
    ) -> AssistantMessageEventStream:
        return self._respond(model, options)


class _StubStreams:
    """供 ``create_provider`` 相关测试使用的最小 ``api`` 实现。"""

    def stream(self, model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        return AssistantMessageEventStream()

    def stream_simple(
        self, model: Model, context: Any, options: Any = None
    ) -> AssistantMessageEventStream:
        return AssistantMessageEventStream()


class _Interaction:
    """最小化登录交互（对应上游内联的 ``{ signal, prompt, notify }``）。"""

    def __init__(self, signal: AbortSignal | None = None) -> None:
        self.signal = signal

    async def prompt(self, prompt: Any) -> str:
        return "unused"

    def notify(self, event: Any) -> None:
        pass


class _AuthContext:
    """无环境变量的确定性 auth 上下文。"""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = values or {}

    async def env(self, name: str) -> str | None:
        return self._values.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


def _constant(value: Any) -> Any:
    """对应上游的 ``async () => value`` modify 回调。"""

    async def fn(_current: Any) -> Any:
        return value

    return fn


# ---------------------------------------------------------------------------------
# 凭据元数据
# ---------------------------------------------------------------------------------


async def test_enumerates_credential_metadata_without_exposing_secrets():
    credentials = InMemoryCredentialStore()
    await credentials.modify(
        "api-provider", _constant(ApiKeyCredential(type="api_key", key="secret"))
    )

    assert [to_json(info) for info in await credentials.list()] == [
        {"providerId": "api-provider", "type": "api_key"},
    ]


# ---------------------------------------------------------------------------------
# 定价
# ---------------------------------------------------------------------------------


def test_applies_request_wide_pricing_tiers_above_the_configured_input_threshold():
    model = _test_model("openai", "gpt-5.6-sol")
    model.cost = ModelCost(
        input=5,
        output=30,
        cache_read=0.5,
        cache_write=6.25,
        tiers=[
            ModelCostTier(
                input_tokens_above=272000,
                input=10,
                output=45,
                cache_read=1,
                cache_write=12.5,
            )
        ],
    )

    def create_usage(cache_write: int) -> Usage:
        return Usage(
            input=200000,
            output=100000,
            cache_read=72000,
            cache_write=cache_write,
            total_tokens=372000 + cache_write,
            cost=Cost(),
        )

    short = calculate_cost(model, create_usage(0))
    assert short.input == pytest.approx(1)
    assert short.output == pytest.approx(3)
    assert short.cache_read == pytest.approx(0.036)
    assert short.cache_write == pytest.approx(0)

    long = calculate_cost(model, create_usage(1))
    assert long.input == pytest.approx(2)
    assert long.output == pytest.approx(4.5)
    assert long.cache_read == pytest.approx(0.072)
    assert long.cache_write == pytest.approx(0.0000125)


# ---------------------------------------------------------------------------------
# provider 注册表
# ---------------------------------------------------------------------------------


def test_registers_replaces_and_deletes_providers():
    models = create_models()
    models.set_provider(_FakeProvider(id="p1"))
    models.set_provider(_FakeProvider(id="p2"))
    assert [provider.id for provider in models.get_providers()] == ["p1", "p2"]

    replacement = _FakeProvider(id="p1")
    models.set_provider(replacement)
    assert models.get_provider("p1") is replacement
    assert len(models.get_providers()) == 2

    models.delete_provider("p1")
    assert models.get_provider("p1") is None

    models.clear_providers()
    assert len(models.get_providers()) == 0


async def test_lists_and_finds_models_per_provider():
    models = create_models()
    models.set_provider(
        _FakeProvider(
            id="p1", models=[_test_model("p1", "m1"), _test_model("p1", "m2")]
        )
    )
    models.set_provider(_FakeProvider(id="p2", models=[_test_model("p2", "m3")]))

    assert [model.id for model in models.get_models()] == ["m1", "m2", "m3"]
    assert [model.id for model in models.get_models("p1")] == ["m1", "m2"]
    assert len(models.get_models("nope")) == 0
    found_p2 = models.get_model("p2", "m3")
    assert found_p2 is not None and found_p2.id == "m3"
    assert models.get_model("p2", "missing") is None

    # has_api() 对动态查找到的模型做运行时检查以收窄类型
    found = models.get_model("p2", "m3")
    assert bool(found and has_api(found, "openai-completions")) is False
    assert bool(found and has_api(found, "test-api")) is True
    if found is not None and has_api(found, "test-api"):
        _typed: Model = found
        assert _typed.id == "m3"


async def test_keeps_chat_reads_independent_from_the_all_model_catalog():
    def raise_unavailable() -> list[Model]:
        raise RuntimeError("all models unavailable")

    provider = _FakeProvider(id="chat-only", get_all_models=raise_unavailable)
    models = create_models()
    models.set_provider(provider)

    assert [model.id for model in models.get_models("chat-only")] == ["model-a"]
    found = models.get_model("chat-only", "model-a")
    assert found is not None and found.id == "model-a"
    assert [model.id for model in await models.get_available("chat-only")] == ["model-a"]
    assert list(models.get_all_models("chat-only")) == []


def test_swallows_provider_source_failures_for_both_all_provider_and_single_provider_listing():
    def boom() -> list[Model]:
        raise RuntimeError("boom")

    models = create_models()
    models.set_provider(_FakeProvider(id="broken", get_models=boom))
    models.set_provider(_FakeProvider(id="ok", models=[_test_model("ok", "m1")]))

    assert [model.id for model in models.get_models()] == ["m1"]
    assert list(models.get_models("broken")) == []
    # 精确的错误信息需直接调用 provider 才能拿到
    with pytest.raises(RuntimeError, match="boom"):
        models.get_provider("broken").get_models()


# ---------------------------------------------------------------------------------
# 测试 refresh
# ---------------------------------------------------------------------------------


async def test_refresh_updates_every_configured_dynamic_provider_and_reports_failures():
    current: list[Model] = [_test_model("dyn", "before")]
    refreshes = 0

    async def refresh_dyn(context: Any) -> None:
        nonlocal refreshes
        if not context.allow_network:
            return
        refreshes += 1

        def update() -> None:
            current[:] = [_test_model("dyn", "after")]

        await context.publish(ModelsPublication(update=update))

    models = create_models()
    models.set_provider(
        _FakeProvider(id="dyn", get_models=lambda: list(current), refresh_models=refresh_dyn)
    )
    models.set_provider(_FakeProvider(id="static", models=[_test_model("static", "s1")]))

    assert models.get_model("dyn", "before") is not None
    first = await models.refresh()
    assert len(first.errors) == 0
    assert refreshes == 1
    assert models.get_model("dyn", "after") is not None
    assert models.get_model("dyn", "before") is None

    async def refresh_flaky(context: Any) -> None:
        if context.allow_network:
            raise RuntimeError("fetch failed")

    models.set_provider(_FakeProvider(id="flaky", refresh_models=refresh_flaky))
    second = await models.refresh()
    assert refreshes == 2
    assert str(second.errors["flaky"]) == "fetch failed"


async def test_restricts_refresh_work_to_selected_providers():
    calls: list[str] = []
    models = create_models()
    for provider_id in ("one", "two"):

        async def refresh(context: Any, provider_id: str = provider_id) -> None:
            calls.append(f"{provider_id}:{'network' if context.allow_network else 'cache'}")

        models.set_provider(_FakeProvider(id=provider_id, refresh_models=refresh))

    result = await models.refresh(ModelsRefreshOptions(providers=["two", "unknown"]))

    assert len(result.errors) == 0
    assert calls == ["two:cache", "two:network"]


async def test_restores_cached_models_before_waiting_for_network_auth():
    store = InMemoryModelsStore()
    await store.write("dynamic", ModelsStoreEntry(models=[_test_model("dynamic", "cached")]))

    auth_started = asyncio.Event()
    finish_auth = asyncio.Event()

    class _BlockedAuth:
        name = "Blocked auth"

        async def resolve(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthResult:
            auth_started.set()
            await finish_auth.wait()
            return AuthResult(auth=ModelAuth(api_key="key"))

    async def fetch_models(context: Any) -> list[Model]:
        raise RuntimeError("must not fetch")

    provider = create_provider(
        CreateProviderOptions(
            id="dynamic",
            auth=ProviderAuth(api_key=_BlockedAuth()),
            models=[],
            fetch_models=fetch_models,
            api=_StubStreams(),
        )
    )
    models = create_models(CreateModelsOptions(models_store=store))
    models.set_provider(provider)
    controller = AbortController()
    pending = asyncio.ensure_future(
        models.refresh(ModelsRefreshOptions(providers=["dynamic"], signal=controller.signal))
    )
    await auth_started.wait()

    assert models.get_model("dynamic", "cached") is not None
    controller.abort()
    result = await pending
    assert result.aborted is True
    finish_auth.set()
    await asyncio.sleep(0)


async def test_lets_providers_choose_persistent_deletion_and_ephemeral_publication_atomically():
    class _Store:
        def __init__(self, entry: ModelsStoreEntry | None) -> None:
            self.entry = entry

        async def read(self, provider_id: str, options: Any = None) -> ModelsStoreEntry | None:
            return self.entry

        async def write(
            self, provider_id: str, next_entry: ModelsStoreEntry, options: Any = None
        ) -> None:
            self.entry = next_entry

        async def delete(self, provider_id: str, options: Any = None) -> None:
            self.entry = None

    store = _Store(ModelsStoreEntry(models=[_test_model("dynamic", "stored")]))
    state = ["initial"]

    async def refresh(context: Any) -> None:
        assert context.stored is not None and context.stored.models[0].id == "stored"

        def delete_update() -> None:
            assert store.entry is None
            state[0] = "deleted"

        await context.publish(ModelsPublication(persist=None, update=delete_update))

        def ephemeral_update() -> None:
            state[0] = "ephemeral"

        await context.publish(ModelsPublication(update=ephemeral_update))

    models = create_models(CreateModelsOptions(models_store=store))
    models.set_provider(_FakeProvider(id="dynamic", refresh_models=refresh))

    result = await models.refresh(ModelsRefreshOptions(allow_network=False))

    assert len(result.errors) == 0
    assert store.entry is None
    assert state[0] == "ephemeral"


async def test_persists_dynamic_catalogs_and_restores_them_without_network_access():
    credentials = InMemoryCredentialStore()
    models_store = InMemoryModelsStore()
    await credentials.modify("dynamic", _constant(ApiKeyCredential(type="api_key", key="key")))

    def create_dynamic_provider(fetch_models: Any = None):
        return create_provider(
            CreateProviderOptions(
                id="dynamic",
                auth=ProviderAuth(api_key=_EnvKeyAuth(None)),
                models=[],
                fetch_models=fetch_models,
                api=_StubStreams(),
            )
        )

    async def fetch_online(context: Any) -> list[Model]:
        return [_test_model("dynamic", "fetched")]

    online = create_models(
        CreateModelsOptions(credentials=credentials, models_store=models_store)
    )
    online.set_provider(create_dynamic_provider(fetch_online))
    assert len((await online.refresh()).errors) == 0
    assert online.get_model("dynamic", "fetched") is not None

    async def fetch_forbidden(context: Any) -> list[Model]:
        raise RuntimeError("must not fetch")

    offline = create_models(
        CreateModelsOptions(credentials=credentials, models_store=models_store)
    )
    offline.set_provider(create_dynamic_provider(fetch_forbidden))
    assert len((await offline.refresh(ModelsRefreshOptions(allow_network=False))).errors) == 0
    assert offline.get_model("dynamic", "fetched") is not None


async def test_passes_effective_api_key_credentials_and_refresh_options_while_skipping_unconfigured_providers():
    effective_credential: Any = None
    force_refresh: bool | None = None
    unconfigured_refreshes = 0

    async def refresh_configured(context: Any) -> None:
        nonlocal effective_credential, force_refresh
        if not context.allow_network:
            return
        effective_credential = context.credential
        force_refresh = context.force

    async def refresh_unconfigured(context: Any) -> None:
        nonlocal unconfigured_refreshes
        if context.allow_network:
            unconfigured_refreshes += 1

    models = create_models()
    models.set_provider(
        _FakeProvider(
            id="configured",
            auth=ProviderAuth(api_key=_EnvKeyAuth("ambient-key")),
            refresh_models=refresh_configured,
        )
    )
    models.set_provider(
        _FakeProvider(
            id="unconfigured",
            auth=ProviderAuth(api_key=_EnvKeyAuth(None)),
            refresh_models=refresh_unconfigured,
        )
    )

    await models.refresh(ModelsRefreshOptions(force=True))
    assert effective_credential == ApiKeyCredential(type="api_key", key="ambient-key")
    assert effective_credential.env is None
    assert force_refresh is True
    assert unconfigured_refreshes == 0


async def test_always_gives_providers_a_concrete_signal():
    received: list[AbortSignal] = []

    async def refresh(context: Any) -> None:
        received.append(context.signal)

    models = create_models()
    models.set_provider(_FakeProvider(id="dynamic", refresh_models=refresh))

    result = await models.refresh()
    assert result.aborted is False
    assert isinstance(received[-1], AbortSignal)
    assert received[-1].aborted is False


async def test_binds_model_store_waits_to_the_provider_refresh_signal():
    storage_signals: list[AbortSignal | None] = []

    class _Store:
        async def read(self, provider_id: str, options: Any = None) -> ModelsStoreEntry | None:
            storage_signals.append(options.signal if options is not None else None)
            return None

        async def write(self, provider_id: str, entry: Any, options: Any = None) -> None:
            storage_signals.append(options.signal if options is not None else None)

        async def delete(self, provider_id: str, options: Any = None) -> None:
            storage_signals.append(options.signal if options is not None else None)

    provider_signals: list[AbortSignal] = []

    async def refresh(context: Any) -> None:
        provider_signals.append(context.signal)
        if not context.allow_network:
            return
        await context.publish(
            ModelsPublication(persist=ModelsStoreEntry(models=[_test_model("dynamic", "fresh")]))
        )

    models = create_models(CreateModelsOptions(models_store=_Store()))
    models.set_provider(
        _FakeProvider(
            id="dynamic",
            auth=ProviderAuth(api_key=_EnvKeyAuth("key")),
            refresh_models=refresh,
        )
    )

    result = await models.refresh(ModelsRefreshOptions(providers=["dynamic"]))

    assert len(result.errors) == 0
    assert len(storage_signals) == 3
    assert all(signal is provider_signals[-1] for signal in storage_signals)


async def test_returns_aborted_state_without_reporting_cancellation_as_a_provider_error():
    controller = AbortController()

    async def refresh(context: Any) -> None:
        controller.abort()
        if context.signal.aborted:
            return

    models = create_models()
    models.set_provider(_FakeProvider(id="dynamic", refresh_models=refresh))

    result = await models.refresh(ModelsRefreshOptions(signal=controller.signal))
    assert result.aborted is True
    assert len(result.errors) == 0


async def test_stops_waiting_on_abort_when_a_provider_ignores_its_signal():
    controller = AbortController()
    started = asyncio.Event()
    stalled: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    calls = 0

    async def refresh(context: Any) -> None:
        nonlocal calls
        calls += 1
        if calls != 1:
            return
        started.set()
        await stalled

    models = create_models()
    models.set_provider(_FakeProvider(id="dynamic", refresh_models=refresh))

    pending = asyncio.ensure_future(models.refresh(ModelsRefreshOptions(signal=controller.signal)))
    await started.wait()
    controller.abort()

    result = await pending
    assert result.aborted is True
    assert len(result.errors) == 0

    stalled.set_exception(RuntimeError("late provider failure"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(result.errors) == 0


async def test_rejects_late_publication_from_a_superseded_non_cooperative_provider():
    store = InMemoryModelsStore()
    state = ["initial"]
    calls = 0
    first_started = asyncio.Event()
    first_blocked = asyncio.Event()

    async def refresh(context: Any) -> None:
        nonlocal calls
        if not context.allow_network:
            return
        calls += 1
        current = calls
        if current == 1:
            first_started.set()
            await first_blocked.wait()
        value = f"generation-{current}"

        def update() -> None:
            state[0] = value

        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=[_test_model("dynamic", value)]),
                update=update,
            )
        )

    models = create_models(CreateModelsOptions(models_store=store))
    models.set_provider(_FakeProvider(id="dynamic", refresh_models=refresh))

    first = asyncio.ensure_future(models.refresh(ModelsRefreshOptions(providers=["dynamic"])))
    await first_started.wait()
    second = asyncio.ensure_future(models.refresh(ModelsRefreshOptions(providers=["dynamic"])))
    await second
    await first
    first_blocked.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert state[0] == "generation-2"
    stored = await store.read("dynamic")
    assert stored is not None and stored.models[0].id == "generation-2"


# ---------------------------------------------------------------------------------
# auth 回调与取消
# ---------------------------------------------------------------------------------


async def test_passes_caller_signals_to_provider_auth_callbacks():
    controller = AbortController()
    received: list[AbortSignal] = []

    class _SignalAuth:
        name = "Signal auth"

        async def login(self, interaction: Any) -> ApiKeyCredential:
            received.append(interaction.signal)
            return ApiKeyCredential(type="api_key", key="saved")

        async def check(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthCheck:
            received.append(signal)
            return AuthCheck(type="api_key")

        async def resolve(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthResult:
            received.append(signal)
            return AuthResult(auth=ModelAuth(api_key="resolved"))

    models = create_models()
    models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=_SignalAuth())))

    await models.check_auth("p1", AuthOperationOptions(signal=controller.signal))
    await models.get_auth("p1", AuthResolutionOverrides(signal=controller.signal))
    await models.login(
        "p1",
        "api_key",
        _Interaction(controller.signal),
    )

    assert received == [controller.signal, controller.signal, controller.signal]


async def test_stops_waiting_for_non_cooperative_auth_callbacks():
    check_started = asyncio.Event()
    blocked_check = asyncio.Event()
    resolve_started = asyncio.Event()
    blocked_resolve = asyncio.Event()

    class _BlockedAuth:
        name = "Blocked auth"

        async def check(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthCheck:
            check_started.set()
            await blocked_check.wait()
            return AuthCheck(type="api_key")

        async def resolve(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthResult:
            resolve_started.set()
            await blocked_resolve.wait()
            return AuthResult(auth=ModelAuth(api_key="key"))

    models = create_models()
    models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=_BlockedAuth())))

    available_controller = AbortController()
    available = asyncio.ensure_future(
        models.get_available(None, AuthOperationOptions(signal=available_controller.signal))
    )
    await check_started.wait()
    available_controller.abort()
    with pytest.raises(AbortError) as available_error:
        await available
    assert type(available_error.value).__name__ == "AbortError"

    auth_controller = AbortController()
    auth = asyncio.ensure_future(
        models.get_auth("p1", AuthResolutionOverrides(signal=auth_controller.signal))
    )
    await resolve_started.wait()
    auth_controller.abort()
    with pytest.raises(AbortError) as auth_error:
        await auth
    assert type(auth_error.value).__name__ == "AbortError"

    blocked_check.set()
    blocked_resolve.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def test_cancels_queued_credential_mutations_without_running_them_later():
    credentials = InMemoryCredentialStore()
    first_blocked = asyncio.Event()
    second_ran = False

    async def first_mutation(_current: Any) -> ApiKeyCredential:
        await first_blocked.wait()
        return ApiKeyCredential(type="api_key", key="first")

    first = asyncio.ensure_future(credentials.modify("p1", first_mutation))
    await asyncio.sleep(0)

    async def second_mutation(_current: Any) -> ApiKeyCredential:
        nonlocal second_ran
        second_ran = True
        return ApiKeyCredential(type="api_key", key="second")

    controller = AbortController()
    second = asyncio.ensure_future(
        credentials.modify("p1", second_mutation, AuthOperationOptions(signal=controller.signal))
    )
    await asyncio.sleep(0)

    controller.abort()
    with pytest.raises(AbortError):
        await second
    first_blocked.set()
    await first
    await asyncio.sleep(0)

    assert second_ran is False
    assert to_json(await credentials.read("p1")) == {"type": "api_key", "key": "first"}


# ---------------------------------------------------------------------------------
# auth 解析
# ---------------------------------------------------------------------------------


async def test_resolves_auth_stored_credential_owns_the_provider_ambient_only_when_nothing_stored():
    credentials = InMemoryCredentialStore()
    models = create_models(CreateModelsOptions(credentials=credentials))
    models.set_provider(
        _FakeProvider(
            id="p1",
            auth=ProviderAuth(api_key=_EnvKeyAuth("env-key")),
        )
    )
    model = _test_model("p1", "model-a")

    # 按 model 与按 provider-id 两种调用方式应解析到同一份 provider 作用域的 auth
    model_auth = await models.get_auth(model)
    assert model_auth is not None and model_auth.auth.api_key == "env-key"
    provider_auth = await models.get_auth(model.provider)
    assert provider_auth is not None and provider_auth.auth.api_key == "env-key"
    explicit = await models.get_auth(model, AuthResolutionOverrides(api_key="explicit-key"))
    assert explicit is not None and explicit.auth.api_key == "explicit-key"

    # 已存的 api-key 凭据走 apiKey auth 解析，优先于环境变量
    await credentials.modify(
        "p1", _constant(ApiKeyCredential(type="api_key", key="stored-key"))
    )
    api_key_resolution = await models.get_auth(model.provider)
    assert api_key_resolution is not None and api_key_resolution.auth.api_key == "stored-key"
    assert api_key_resolution.source == "stored"


@dataclass
class _UnknownCredential:
    """非 api-key 的已存凭据（旧配置遗留），本项目没有任何 handler 处理它。"""

    type: str = "oauth"


async def test_a_stored_credential_without_a_matching_handler_blocks_ambient_fallback():
    credentials = InMemoryCredentialStore()
    models = create_models(CreateModelsOptions(credentials=credentials))
    # provider 只配置了 apiKey auth，但存储里遗留了未知类型的凭据。
    models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=_EnvKeyAuth("env-key"))))
    await credentials.modify("p1", _constant(_UnknownCredential()))

    # 已存凭据独占 provider：不得静默回退到环境变量。
    assert await models.get_auth("p1") is None


async def test_checks_provider_auth_and_filters_available_models():
    credentials = InMemoryCredentialStore()

    models = create_models(CreateModelsOptions(credentials=credentials))
    models.set_provider(
        _FakeProvider(id="ambient", auth=ProviderAuth(api_key=_EnvKeyAuth("env-key")))
    )
    models.set_provider(_FakeProvider(id="explicit", auth=ProviderAuth(api_key=_EnvKeyAuth(None))))
    await credentials.modify(
        "explicit", _constant(ApiKeyCredential(type="api_key", key="explicit-key"))
    )

    assert to_json(await models.check_auth("ambient")) == {"source": "env", "type": "api_key"}
    assert to_json(await models.check_auth("explicit")) == {"source": "stored", "type": "api_key"}
    assert [model.provider for model in await models.get_available()] == ["ambient", "explicit"]
    assert [model.provider for model in await models.get_available("ambient")] == ["ambient"]


async def test_runs_provider_login_and_logout_through_the_credential_store():
    credentials = InMemoryCredentialStore()
    api_key = _EnvKeyAuth(None)

    async def login(interaction: Any) -> ApiKeyCredential:
        return ApiKeyCredential(type="api_key", key="logged-in")

    api_key.login = login

    models = create_models(CreateModelsOptions(credentials=credentials))
    models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=api_key)))

    credential = await models.login("p1", "api_key", _Interaction())
    assert to_json(credential) == {"type": "api_key", "key": "logged-in"}
    assert await credentials.read("p1") == credential

    await models.logout("p1")
    assert await credentials.read("p1") is None


async def test_wraps_credential_store_failures_in_models_error():
    # read 失败
    class _ReadFailingStore:
        async def read(self, provider_id: str, options: Any = None) -> Any:
            raise RuntimeError("disk on fire")

        async def list(self, options: Any = None) -> Any:
            return []

        async def modify(self, provider_id: str, fn: Any, options: Any = None) -> Any:
            return None

        async def delete(self, provider_id: str, options: Any = None) -> None:
            return None

    models = create_models(CreateModelsOptions(credentials=_ReadFailingStore()))
    models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=_EnvKeyAuth("env-key"))))
    with pytest.raises(ModelsError) as read_error:
        await models.get_auth("p1")
    assert read_error.value.code == "auth"

    # 测试 login 过程中 modify 失败
    class _ModifyFailingStore:
        async def read(self, provider_id: str, options: Any = None) -> Any:
            return None

        async def list(self, options: Any = None) -> Any:
            return [CredentialInfo(provider_id="p1", type="api_key")]

        async def modify(self, provider_id: str, fn: Any, options: Any = None) -> Any:
            raise RuntimeError("disk on fire")

        async def delete(self, provider_id: str, options: Any = None) -> None:
            return None

    api_key = _EnvKeyAuth(None)

    async def login(interaction: Any) -> ApiKeyCredential:
        return ApiKeyCredential(type="api_key", key="logged-in")

    api_key.login = login
    modify_models = create_models(CreateModelsOptions(credentials=_ModifyFailingStore()))
    modify_models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=api_key)))
    with pytest.raises(ModelsError) as modify_error:
        await modify_models.login("p1", "api_key", _Interaction())
    assert modify_error.value.code == "auth"


async def test_wraps_api_key_auth_failures_in_models_error():
    class _FailingAuth:
        name = "Failing"

        async def resolve(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthResult:
            raise RuntimeError("nope")

    models = create_models()
    models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=_FailingAuth())))
    with pytest.raises(ModelsError) as error:
        await models.get_auth("p1")
    assert error.value.code == "auth"


# ---------------------------------------------------------------------------------
# 请求级 auth 应用
# ---------------------------------------------------------------------------------


async def test_uses_explicit_request_api_key_and_env_during_provider_auth_resolution():
    calls: list[_ProviderCall] = []

    class _ScopedAuth:
        name = "Scoped"

        async def resolve(
            self, *, ctx: Any, credential: ApiKeyCredential | None = None, signal: AbortSignal
        ) -> AuthResult | None:
            account = (
                credential.env.get("ACCOUNT_ID")
                if credential is not None and credential.env is not None
                else None
            )
            if account is None:
                account = await ctx.env("ACCOUNT_ID")
            if credential is None or not credential.key or not account:
                return None
            return AuthResult(
                auth=ModelAuth(
                    api_key=credential.key, base_url=f"https://example.test/{account}"
                ),
                env={"ACCOUNT_ID": account},
            )

    models = create_models(CreateModelsOptions(auth_context=_AuthContext()))
    models.set_provider(
        _FakeProvider(id="p1", auth=ProviderAuth(api_key=_ScopedAuth()), calls=calls)
    )
    model = _test_model("p1", "model-a")

    await models.complete_simple(
        model,
        _context(),
        SimpleStreamOptions(api_key="explicit-key", env={"ACCOUNT_ID": "acct"}),
    )

    assert calls[0].model.base_url == "https://example.test/acct"
    assert calls[0].options.api_key == "explicit-key"
    assert calls[0].options.env == {"ACCOUNT_ID": "acct"}


async def test_merges_resolved_auth_into_stream_options_explicit_options_win_per_field():
    calls: list[_ProviderCall] = []

    class _MergeAuth:
        name = "Test"

        async def resolve(self, *, ctx: Any, credential: Any = None, signal: AbortSignal) -> AuthResult:
            return AuthResult(
                auth=ModelAuth(
                    api_key="resolved-key",
                    headers={
                        "Authorization": "Bearer resolved-key",
                        "x-a": "auth",
                        "x-b": "auth",
                    },
                    base_url="https://auth.test/v1",
                )
            )

    models = create_models()
    models.set_provider(_FakeProvider(id="p1", auth=ProviderAuth(api_key=_MergeAuth()), calls=calls))
    model = _test_model("p1", "model-a")

    result = await models.complete_simple(
        model,
        _context(),
        SimpleStreamOptions(
            api_key="explicit-key",
            headers={"authorization": "Explicit token", "x-b": "explicit"},
        ),
    )
    assert result.stop_reason == "stop"
    assert len(calls) == 1
    assert calls[0].options.api_key == "explicit-key"
    assert calls[0].options.headers == {
        "authorization": "Explicit token",
        "x-a": "auth",
        "x-b": "explicit",
    }
    assert calls[0].model.base_url == "https://auth.test/v1"

    # 未显式传 options 时，应用解析出的 auth
    result2 = await models.complete_simple(model, _context())
    assert result2.stop_reason == "stop"
    assert calls[1].options.api_key == "resolved-key"


async def test_adds_model_headers_only_for_model_auth_and_transforms_assembled_headers_once():
    calls: list[_ProviderCall] = []
    models = create_models()
    models.set_provider(
        _FakeProvider(id="p1", auth=ProviderAuth(api_key=_EnvKeyAuth("key")), calls=calls)
    )
    model = _test_model("p1", "model-a")
    model.headers = {"x-model": "model", "x-shared": "model"}

    provider_auth = await models.get_auth("p1")
    assert provider_auth is not None and provider_auth.auth.headers is None
    model_auth = await models.get_auth(model)
    assert model_auth is not None
    assert model_auth.auth.headers == {"x-model": "model", "x-shared": "model"}

    transforms = 0

    async def transform_headers(headers: Any) -> Any:
        nonlocal transforms
        transforms += 1
        assert headers == {
            "x-model": "model",
            "x-explicit": "explicit",
            "X-Shared": "explicit",
        }
        return {**headers, "x-transformed": "yes"}

    await models.complete_simple(
        model,
        _context(),
        SimpleStreamOptions(headers={"x-explicit": "explicit", "X-Shared": "explicit"}),
        transform_headers=transform_headers,
    )

    assert transforms == 1
    assert calls[0].options.headers == {
        "x-model": "model",
        "x-explicit": "explicit",
        "X-Shared": "explicit",
        "x-transformed": "yes",
    }
    assert not hasattr(calls[0].options, "transform_headers")


async def test_produces_an_error_stream_for_unknown_providers_instead_of_throwing():
    models = create_models()
    result = await models.complete_simple(_test_model("ghost", "model-a"), _context())
    assert result.stop_reason == "error"
    assert "Unknown provider: ghost" in (result.error_message or "")


async def test_streams_through_the_provider():
    models = create_models()
    models.set_provider(_FakeProvider(id="p1"))
    model = _test_model("p1", "model-a")

    events: list[str] = []
    stream = models.stream_simple(model, _context())
    async for event in stream:
        events.append(event.type)
    assert events == ["start", "done"]
    message = await stream.result()
    assert message.stop_reason == "stop"
