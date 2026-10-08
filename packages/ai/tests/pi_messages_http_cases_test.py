"""移植自 ``packages/ai/test/pi-messages.test.ts``。

上游启动本地 ``node:http`` 服务器；这里忠实的注入点是适配器的 ``fetch`` 选项，测试用它
记录发出的 :class:`httpx.Request` 并返回预设的 SSE 响应体（或后端错误响应）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from pi_ai.api.pi_messages import PiMessagesOptions, stream, stream_simple
from pi_ai.compat import get_api_provider
from pi_ai.types import (
    KNOWN_APIS,
    AssistantMessageEvent,
    Context,
    Cost,
    Model,
    ModelCost,
    StopReason,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

CONTEXT = Context(messages=[UserMessage(content="Hello", timestamp=1_700_000_000_000)])

USAGE = Usage(
    input=10,
    output=5,
    cache_read=0,
    cache_write=0,
    total_tokens=15,
    cost=Cost(input=0.1, output=0.2, cache_read=0, cache_write=0, total=0.3),
)


def create_model(base_url: str) -> Model:
    return Model(
        id="auto",
        name="Radius Auto",
        api="pi-messages",
        provider="radius",
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1, output=2, cache_read=0.1, cache_write=0.2),
        context_window=128000,
        max_tokens=16384,
    )


class _SSEByteStream(httpx.AsyncByteStream):
    """每个 chunk 交付一条 SSE 记录，记录之间让出执行，模拟真实服务器。"""

    def __init__(self, records: list[bytes]) -> None:
        self._records = records

    async def __aiter__(self):
        for record in self._records:
            await asyncio.sleep(0)
            yield record


class FakeBackend:
    """记录请求，并以预设的 SSE 事件或错误响应体作答。"""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        events: list[Any] | None = None,
        raw_body: str | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.events = events or []
        self.raw_body = raw_body
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(
                self.status,
                content=(self.raw_body or "{}").encode("utf-8"),
                headers={"content-type": "application/json"},
                request=request,
            )
        records = [f"data: {json.dumps(event)}\n\n".encode("utf-8") for event in self.events]
        return httpx.Response(
            200,
            stream=_SSEByteStream(records),
            headers={"content-type": "text/event-stream", **self.headers},
            request=request,
        )


def _recorded_body(request: httpx.Request) -> Any:
    return json.loads(request.content) if request.content else None


async def test_streams_text_and_tool_calls_and_resolves_the_terminal_message() -> None:
    backend = FakeBackend(
        events=[
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "Hel"},
            {"type": "text_delta", "contentIndex": 0, "delta": "lo"},
            {"type": "text_end", "contentIndex": 0, "content": "Hello"},
            {"type": "toolcall_start", "contentIndex": 1, "id": "call_1", "toolName": "read"},
            {"type": "toolcall_delta", "contentIndex": 1, "delta": '{"path":'},
            {"type": "toolcall_delta", "contentIndex": 1, "delta": '"a.txt"}'},
            {
                "type": "toolcall_end",
                "contentIndex": 1,
                "toolCall": {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "a.txt"}},
            },
            {
                "type": "done",
                "reason": "toolUse",
                "usage": {
                    "input": 10,
                    "output": 5,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 15,
                    "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0, "cacheWrite": 0, "total": 0.3},
                },
                "responseId": "resp_1",
                "providerThinkingLevel": "high",
            },
        ]
    )
    model = create_model("http://127.0.0.1:9/v1")

    events: list[AssistantMessageEvent] = []
    partial_stop_reasons: list[str] = []
    event_stream = stream(
        model,
        normalize_context(CONTEXT),
        PiMessagesOptions(
            api_key="test-key",
            session_id="session-1",
            tool_choice="auto",
            max_tokens=100,
            headers={"x-custom": "1"},
            fetch=backend,
        ),
    )
    async for event in event_stream:
        partial = getattr(event, "partial", None)
        if partial is not None:
            partial_stop_reasons.append(partial.stop_reason)
        events.append(event)
    message = await event_stream.result()

    assert partial_stop_reasons[0] == "pending"
    assert message.stop_reason == StopReason.TOOL_USE
    assert message.usage == USAGE
    assert message.response_id == "resp_1"
    assert message.provider_thinking_level == "high"
    assert message.model == "auto"
    assert message.provider == "radius"
    assert message.content == [
        TextContent(type="text", text="Hello", text_signature=None),
        ToolCall(type="toolCall", id="call_1", name="read", arguments={"path": "a.txt"}),
    ]
    assert any(event.type == "text_delta" for event in events)
    assert len([event for event in events if event.type == "toolcall_end"]) == 1

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.url.path == "/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-custom"] == "1"
    assert _recorded_body(request) == {
        "model": "auto",
        "context": json.loads(json.dumps({"messages": [{"role": "user", "content": "Hello", "timestamp": 1_700_000_000_000}]})),
        "options": {"maxTokens": 100, "sessionId": "session-1", "toolChoice": "auto"},
    }


async def test_forwards_parsed_wire_events_in_order_before_converting_them() -> None:
    wire_events = [
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_delta", "contentIndex": 0, "delta": "Hello", "gatewayField": "upstream-value"},
        {"type": "text_end", "contentIndex": 0, "content": "Hello"},
        {
            "type": "done",
            "reason": "stop",
            "usage": {"input": 10, "output": 5, "totalTokens": 15},
            "responseId": "resp_1",
        },
    ]
    backend = FakeBackend(events=wire_events)
    model = create_model("http://127.0.0.1:9/v1")
    received: list[Any] = []
    event_models: list[Model] = []

    async def on_provider_stream_event(event: Any, event_model: Model) -> None:
        received.append(event)
        event_models.append(event_model)

    message = await stream_simple(
        model,
        normalize_context(CONTEXT),
        PiMessagesOptions(
            api_key="test-key",
            fetch=backend,
            on_provider_stream_event=on_provider_stream_event,
        ),
    ).result()

    assert received == wire_events
    assert event_models == [model for _ in wire_events]
    assert message.stop_reason == "stop"
    assert message.response_id == "resp_1"
    assert message.content == [TextContent(type="text", text="Hello", text_signature=None)]


async def test_appends_debug_1_and_reports_response_headers_via_on_response() -> None:
    backend = FakeBackend(
        headers={"x-pi-gateway-upstream-provider": "anthropic"},
        events=[{"type": "done", "reason": "stop", "usage": {"input": 1, "output": 1}}],
    )
    model = create_model("http://127.0.0.1:9/v1")

    observed: list[dict[str, str]] = []
    message = await stream_simple(
        model,
        normalize_context(CONTEXT),
        PiMessagesOptions(
            api_key="test-key",
            debug=True,
            fetch=backend,
            on_response=lambda response, _model: observed.append(response.headers),
        ),
    ).result()

    assert message.stop_reason == "stop"
    assert backend.requests[0].url.path == "/v1/messages"
    assert backend.requests[0].url.query.decode() == "debug=1"
    assert observed
    headers = {key.lower(): value for key, value in observed[0].items()}
    assert headers["x-pi-gateway-upstream-provider"] == "anthropic"


async def test_surfaces_backend_error_responses_with_diagnostics() -> None:
    backend = FakeBackend(
        status=401,
        raw_body=json.dumps({"error": {"message": "Token expired", "code": "unauthorized"}}),
    )
    model = create_model("http://127.0.0.1:9/v1")

    message = await stream(model, normalize_context(CONTEXT), PiMessagesOptions(api_key="stale", fetch=backend)).result()

    assert message.stop_reason == StopReason.ERROR
    assert "401" in (message.error_message or "")
    assert "Token expired" in (message.error_message or "")
    assert "unauthorized" in (message.error_message or "")
    assert message.diagnostics is not None
    assert message.diagnostics[0].type == "pi_messages_response_failure"
    assert message.diagnostics[0].details["status"] == 401


async def test_propagates_server_sent_error_events() -> None:
    backend = FakeBackend(
        events=[
            {"type": "start"},
            {
                "type": "error",
                "reason": "error",
                "usage": {"input": 10, "output": 5, "totalTokens": 15},
                "errorMessage": "Upstream failed",
            },
        ]
    )
    model = create_model("http://127.0.0.1:9/v1")

    message = await stream(model, normalize_context(CONTEXT), PiMessagesOptions(api_key="test-key", fetch=backend)).result()

    assert message.stop_reason == StopReason.ERROR
    assert message.error_message == "Upstream failed"
    assert message.usage == Usage(input=10, output=5, total_tokens=15)


async def test_errors_when_no_api_key_is_provided() -> None:
    async def must_not_fetch(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the adapter must not issue a request without an API key")

    model = create_model("http://127.0.0.1:9/v1")

    message = await stream(model, normalize_context(CONTEXT), PiMessagesOptions(fetch=must_not_fetch)).result()

    assert message.stop_reason == StopReason.ERROR
    assert "No API key provided" in (message.error_message or "")


async def test_errors_when_the_stream_ends_without_a_terminal_event() -> None:
    backend = FakeBackend(
        events=[
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "partial"},
        ]
    )
    model = create_model("http://127.0.0.1:9/v1")

    message = await stream(model, normalize_context(CONTEXT), PiMessagesOptions(api_key="test-key", fetch=backend)).result()

    assert message.stop_reason == StopReason.ERROR
    assert "stream ended without a terminal event" in (message.error_message or "")


def test_is_registered_as_a_builtin_api_provider() -> None:
    assert get_api_provider("pi-messages") is not None


def test_is_a_known_api_usable_on_models() -> None:
    assert "pi-messages" in KNOWN_APIS
