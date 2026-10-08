"""移植 ``packages/ai/test/telemetry-options.test.ts``。

上游从 ``@earendil-works/pi-telemetry`` 导入 ``NOOP_TELEMETRY_CONTEXT``。Python 移植将该
字段类型标为 ``Any``（``pi_ai.types.ProviderRequestOptions.telemetry_context``），因此用
本地的恒等哨兵代替 noop context：上游断言全部是同一性检查，这里能精确保持。
"""

from __future__ import annotations

from pi_ai.api.simple_options import build_base_options
from pi_ai.auth.types import AuthResult, ModelAuth, ProviderAuth
from pi_ai.models import CreateProviderOptions, create_models, create_provider
from pi_ai.types import (
    AssistantMessage,
    Context,
    DeferredFetchOptions,
    DeferredHandle,
    Model,
    ModelCost,
    ProviderRequestOptions,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.events import done_event
from pi_ai.utils.transcript import normalize_context

#: 上游 ``NOOP_TELEMETRY_CONTEXT`` 的替代物；仅断言对象同一性。
TELEMETRY_CONTEXT = object()

context = normalize_context(Context(messages=[]))

model = Model(
    id="model",
    name="Model",
    api="telemetry-test",
    provider="telemetry-provider",
    base_url="https://example.test",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=1000,
    max_tokens=100,
)


def _completed_stream(request_model: Model) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = AssistantMessage(
        role="assistant",
        content=[],
        api=request_model.api,
        provider=request_model.provider,
        model=request_model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=0,
    )
    stream.push(done_event("stop", message))
    return stream


def _no_auth() -> ProviderAuth:
    class _Auth:
        name = "Test"

        async def resolve(self, *, ctx=None, credential=None, signal=None) -> AuthResult:
            return AuthResult(auth=ModelAuth())

    return ProviderAuth(api_key=_Auth())


def test_is_inherited_by_every_request_option_surface_and_simple_stream_conversion():
    options = StreamOptions(telemetry_context=TELEMETRY_CONTEXT)
    assert options.telemetry_context is TELEMETRY_CONTEXT
    base = build_base_options(model, context, SimpleStreamOptions(telemetry_context=TELEMETRY_CONTEXT))
    assert base.telemetry_context is TELEMETRY_CONTEXT


async def test_survives_provider_and_models_stream_deferred_dispatch():
    observed: list[object] = []
    handle = DeferredHandle(
        provider=model.provider,
        model_id=model.id,
        api=model.api,
        id="response",
    )

    class _Api:
        def stream(self, request_model, _context, options=None):
            observed.append(options.telemetry_context if options else None)
            return _completed_stream(request_model)

        def stream_simple(self, request_model, _context, options=None):
            observed.append(options.telemetry_context if options else None)
            return _completed_stream(request_model)

        def fetch_deferred(self, request_model, _handle, options=None):
            observed.append(options.telemetry_context if options else None)
            return _completed_stream(request_model)

        async def cancel_deferred(self, _request_model, _handle, options=None):
            observed.append(options.telemetry_context if options else None)

    provider = create_provider(
        CreateProviderOptions(
            id=model.provider,
            auth=_no_auth(),
            models=[model],
            api=_Api(),
        )
    )

    await provider.stream(model, context, StreamOptions(telemetry_context=TELEMETRY_CONTEXT)).result()
    await provider.stream_simple(
        model, context, SimpleStreamOptions(telemetry_context=TELEMETRY_CONTEXT)
    ).result()
    await provider.fetch_deferred(
        model, handle, DeferredFetchOptions(telemetry_context=TELEMETRY_CONTEXT)
    ).result()
    await provider.cancel_deferred(
        model, handle, ProviderRequestOptions(telemetry_context=TELEMETRY_CONTEXT)
    )

    models = create_models()
    models.set_provider(provider)
    await models.stream(model, context, StreamOptions(telemetry_context=TELEMETRY_CONTEXT)).result()
    await models.stream_simple(
        model, context, SimpleStreamOptions(telemetry_context=TELEMETRY_CONTEXT)
    ).result()
    await models.fetch_deferred(model, handle, DeferredFetchOptions(telemetry_context=TELEMETRY_CONTEXT))
    await models.cancel_deferred(
        model, handle, ProviderRequestOptions(telemetry_context=TELEMETRY_CONTEXT)
    )

    assert len(observed) == 8
    assert all(value is TELEMETRY_CONTEXT for value in observed)
