"""Anthropic Messages 适配器的端到端测试。

通过 mock 的 HTTP 传输驱动真实适配器，让请求体、SSE 解码器与产出的事件序列被
一并覆盖。
"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions, anthropic_messages_api, stream, stream_simple
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StopReason,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

MODEL = Model(
    id="claude-test",
    name="Claude Test",
    api="anthropic-messages",
    provider="anthropic",
    base_url="https://api.anthropic.com",
    input=["text", "image"],
    reasoning=True,
    context_window=200_000,
    max_tokens=8192,
)


def _sse(*events: tuple[str, dict]) -> bytes:
    chunks = []
    for name, payload in events:
        chunks.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n")
    return "".join(chunks).encode()


MESSAGE_START = (
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": "claude-test-20260101",
            "usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 2},
        },
    },
)


def _text_stream() -> bytes:
    return _sse(
        MESSAGE_START,
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    )


def _tool_stream() -> bytes:
    return _sse(
        MESSAGE_START,
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city"'}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ': "Paris"}'}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}),
        ("message_stop", {"type": "message_stop"}),
    )


def _recording_fetch(body: bytes, *, status: int = 200, sink: list[httpx.Request] | None = None):
    async def fetch(request: httpx.Request) -> httpx.Response:
        if sink is not None:
            sink.append(request)
        return httpx.Response(status, headers={"content-type": "text/event-stream"}, content=body, request=request)

    return fetch


async def _collect(stream_object) -> list:
    return [event async for event in stream_object]


# ---------------------------------------------------------------------------------
# 流式
# ---------------------------------------------------------------------------------


async def test_text_response_produces_the_full_event_sequence():
    options = AnthropicOptions(api_key="k", fetch=_recording_fetch(_text_stream()))
    events = await _collect(stream(MODEL, normalize_context(Context(messages=[UserMessage(content="hi", timestamp=0)])), options))

    assert events[0].type == "start"
    assert [event.type for event in events[1:-1]] == [
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
    ]
    assert [event.delta for event in events if event.type == "text_delta"] == ["Hel", "lo"]

    final = events[-1]
    assert final.type == "done"
    message = final.message
    assert message.stop_reason == StopReason.STOP
    assert message.content == [TextContent(text="Hello")]
    assert message.model == "claude-test"
    assert message.response_model == "claude-test-20260101"
    assert message.response_id == "msg_1"
    assert message.usage.input == 10
    assert message.usage.cache_read == 2
    assert message.usage.output == 5


async def test_result_equals_the_done_event_message():
    options = AnthropicOptions(api_key="k", fetch=_recording_fetch(_text_stream()))
    stream_object = stream(MODEL, normalize_context(Context(messages=[])), options)
    async for _ in stream_object:
        pass
    assert (await stream_object.result()).content == [TextContent(text="Hello")]


async def test_tool_use_produces_a_tool_call_block():
    options = AnthropicOptions(api_key="k", fetch=_recording_fetch(_tool_stream()))
    events = await _collect(stream(MODEL, normalize_context(Context(messages=[])), options))
    assert [event.type for event in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    message = events[-1].message
    assert message.stop_reason == StopReason.TOOL_USE
    assert len(message.content) == 1
    tool_call = message.content[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.id == "tool_1" or tool_call.id == "toolu_1"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"city": "Paris"}


async def test_request_body_carries_the_transcript_and_tools():
    sink: list[httpx.Request] = []
    options = AnthropicOptions(api_key="k", fetch=_recording_fetch(_text_stream(), sink=sink), max_tokens=64, temperature=0.5)
    context = normalize_context(
        Context(
            system_prompt="be terse",
            messages=[UserMessage(content="hi", timestamp=0)],
            tools=[Tool(name="get_weather", description="d", parameters={"type": "object", "properties": {}})],
        )
    )
    await _collect(stream(MODEL, context, options))

    assert len(sink) == 1
    request = sink[0]
    payload = json.loads(request.content)
    assert request.method == "POST"
    assert str(request.url).startswith("https://api.anthropic.com/v1/messages")
    assert payload["model"] == "claude-test"
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.5
    assert payload["stream"] is True
    # system prompt 以 content block 列表发送，以便附加 cache_control，
    # 且 sanitize_surrogates 会处理其中的文本（与 TypeScript 版一致）。
    assert payload["system"][0]["type"] == "text"
    assert payload["system"][0]["text"] == "be terse"
    assert [tool["name"] for tool in payload["tools"]] == ["get_weather"]
    # 缓存保留默认为 "short"，因此最后一条 user 消息块带有缓存标记。
    assert [message["role"] for message in payload["messages"]] == ["user"]
    assert payload["messages"][0]["content"][0]["text"] == "hi"
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert request.headers["x-api-key"] == "k"


async def test_tool_results_are_encoded_as_tool_result_blocks():
    sink: list[httpx.Request] = []
    context = normalize_context(
        Context(
            messages=[
                UserMessage(content="weather?", timestamp=0),
                AssistantMessage(
                    content=[ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"})],
                    stop_reason=StopReason.TOOL_USE,
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="toolu_1",
                    tool_name="get_weather",
                    content=[TextContent(text="sunny")],
                    timestamp=2,
                ),
            ]
        )
    )
    options = AnthropicOptions(api_key="k", fetch=_recording_fetch(_text_stream(), sink=sink))
    await _collect(stream(MODEL, context, options))

    payload = json.loads(sink[0].content)
    tool_result = payload["messages"][-1]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "toolu_1"


# ---------------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------------


async def test_http_error_becomes_an_error_message_in_the_stream():
    body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "bad model"}}).encode()
    options = AnthropicOptions(api_key="k", fetch=_recording_fetch(body, status=400))
    events = await _collect(stream(MODEL, normalize_context(Context(messages=[])), options))

    assert events[-1].type == "error"
    message = events[-1].error
    assert message.stop_reason == StopReason.ERROR
    assert "bad model" in (message.error_message or "")


async def test_malformed_sse_payload_becomes_an_error_not_an_exception():
    options = AnthropicOptions(api_key="k", fetch=_recording_fetch(b"event: message_start\ndata: {oops\n\n"))
    events = await _collect(stream(MODEL, normalize_context(Context(messages=[])), options))
    assert events[-1].type == "error"


async def test_abort_before_the_request_yields_an_aborted_message():
    from pi_ai.utils.abort import AbortController

    controller = AbortController()
    controller.abort()
    options = AnthropicOptions(api_key="k", signal=controller.signal, fetch=_recording_fetch(_text_stream()))
    events = await _collect(stream(MODEL, normalize_context(Context(messages=[])), options))
    assert events[-1].type == "error"
    assert events[-1].error.stop_reason in ("aborted", "error")


async def test_missing_api_key_is_reported_on_the_stream():
    context = normalize_context(Context(messages=[]))
    events = await _collect(stream(MODEL, context, AnthropicOptions(fetch=_recording_fetch(_text_stream()))))
    assert events[-1].type == "error"
    assert events[-1].error.stop_reason == StopReason.ERROR


# ---------------------------------------------------------------------------------
# 测试 stream_simple
# ---------------------------------------------------------------------------------


async def test_stream_simple_applies_reasoning_and_thinking_budgets():
    sink: list[httpx.Request] = []
    options = SimpleStreamOptions(
        api_key="k",
        reasoning="high",
        fetch=_recording_fetch(_text_stream(), sink=sink),
    )
    await _collect(stream_simple(MODEL, normalize_context(Context(messages=[])), options))
    payload = json.loads(sink[0].content)
    assert payload["max_tokens"] >= 8192
    assert payload["thinking"]["type"] in ("enabled", "adaptive")
    assert payload["thinking"].get("budget_tokens", 0) > 0 or payload["thinking"]["type"] == "adaptive"


async def test_stream_simple_without_reasoning_explicitly_disables_thinking():
    # 具备推理能力的模型在 `off` 档位没有显式映射时，会得到明确的
    # `thinking: {type: "disabled"}`，防止 adaptive thinking 悄悄开启。
    sink: list[httpx.Request] = []
    options = SimpleStreamOptions(api_key="k", fetch=_recording_fetch(_text_stream(), sink=sink))
    await _collect(stream_simple(MODEL, normalize_context(Context(messages=[])), options))
    assert json.loads(sink[0].content)["thinking"] == {"type": "disabled"}


async def test_stream_simple_omits_thinking_when_off_is_mapped_to_null():
    sink: list[httpx.Request] = []
    model = MODEL
    model.thinking_level_map = {"off": None}
    options = SimpleStreamOptions(api_key="k", fetch=_recording_fetch(_text_stream(), sink=sink))
    await _collect(stream_simple(model, normalize_context(Context(messages=[])), options))
    assert "thinking" not in json.loads(sink[0].content)


# ---------------------------------------------------------------------------------
# 装配（wiring）
# ---------------------------------------------------------------------------------


def test_lazy_api_exposes_the_module_streams():
    api = anthropic_messages_api()
    assert callable(api.stream)
    assert callable(api.stream_simple)
    assert api.fetch_deferred is None
