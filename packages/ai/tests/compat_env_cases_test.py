"""移植 ``packages/ai/test/compat-env.test.ts``（"compat legacy API fallback" 用例）。"""

from __future__ import annotations

import pytest

from pi_ai.compat import (
    ApiProvider,
    complete,
    register_api_provider,
    reset_api_providers,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    StreamOptions,
    TextContent,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.events import done_event, start_event

#: 上游 ``Date.now()`` 的确定性替身。
NOW = 1_700_000_000_000

CONTEXT = Context(messages=[UserMessage(content="hi", timestamp=NOW)])

MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-responses",
    provider="custom-openai",
    base_url="https://example.test/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=128000,
    max_tokens=4096,
)


def _message() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="ok")],
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        ),
        stop_reason="stop",
        timestamp=NOW,
    )


@pytest.fixture(autouse=True)
def _reset_registered_api_providers():
    """对应上游 ``afterEach``：恢复内置 api 注册表。"""
    yield
    reset_api_providers()


async def test_dispatches_unknown_providers_through_the_legacy_api_registry():
    captured: dict[str, str | None] = {}

    def fake(model: Model, context: Context, options: StreamOptions | None = None):
        captured["api_key"] = options.api_key if options is not None else None
        stream = AssistantMessageEventStream()
        output = _message()
        stream.push(start_event(output))
        stream.push(done_event("stop", output))
        stream.end(output)
        return stream

    register_api_provider(
        ApiProvider(api="openai-responses", stream=fake, stream_simple=fake)
    )

    await complete(MODEL, CONTEXT, StreamOptions(api_key="request-key"))

    assert captured["api_key"] == "request-key"
