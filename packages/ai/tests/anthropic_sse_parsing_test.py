"""移植自 ``anthropic-sse-parsing.test.ts``。

TypeScript 套件通过伪造的 ``@anthropic-ai/sdk`` 客户端驱动适配器，其
``beta.messages.create().asResponse()`` 返回固定的 SSE ``Response``。本移植保留同样的
固定 SSE 响应体，但改经 ``options.fetch`` 接缝驱动真实适配器，使解码器、请求编码器与
产出的事件序列被一并覆盖，且全程不触网。
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.api.transform_messages import transform_messages
from pi_ai.compat import get_model, normalize_context
from pi_ai.types import (
    AnthropicAllowedFallbackModel,
    Context,
    ModelCompat,
    ModelCost,
    StopReason,
    Tool,
    ToolCall,
    UserMessage,
)
from pi_ai.utils.serde import to_json


# ---------------------------------------------------------------------------------
# SSE 辅助函数
# ---------------------------------------------------------------------------------


def _sse_response_body(events: list[dict[str, str]]) -> bytes:
    """上游的 ``createSseResponse``：用空行分隔的 ``event:``/``data:`` 帧。"""
    body = "\n".join(f"event: {event['event']}\ndata: {event['data']}\n" for event in events)
    return body.encode()


def _sse_fetch(
    events: list[dict[str, str]],
    *,
    sink: list[httpx.Request] | None = None,
):
    """记录请求并返回固定 SSE 响应的 ``fetch`` 实现。"""
    body = _sse_response_body(events)

    async def fetch(request: httpx.Request) -> httpx.Response:
        if sink is not None:
            sink.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )

    return fetch


def _beta_features(request: httpx.Request) -> list[str]:
    """将 ``anthropic-beta`` 请求头解析为特性列表。"""
    header = request.headers.get("anthropic-beta")
    if not header:
        return []
    return [feature.strip() for feature in header.split(",") if feature.strip()]


def _data(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _minimal_anthropic_events() -> list[dict[str, str]]:
    return [
        {
            "event": "message_start",
            "data": _data(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        },
        {
            "event": "content_block_start",
            "data": _data(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            ),
        },
        {
            "event": "content_block_delta",
            "data": _data(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                }
            ),
        },
        {
            "event": "content_block_stop",
            "data": _data({"type": "content_block_stop", "index": 0}),
        },
        {
            "event": "message_delta",
            "data": _data(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        },
        {
            "event": "message_stop",
            "data": _data({"type": "message_stop"}),
        },
    ]


def _response_model_sse_events(model: str, content_block: dict[str, Any]) -> list[dict[str, str]]:
    """上游的 ``createResponseModelSseResponse``。"""
    return [
        {
            "event": "message_start",
            "data": _data(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_response_model",
                        "model": model,
                        "usage": {"input_tokens": 100, "output_tokens": 0},
                    },
                }
            ),
        },
        {
            "event": "content_block_start",
            "data": _data(
                {"type": "content_block_start", "index": 0, "content_block": content_block}
            ),
        },
        {
            "event": "content_block_stop",
            "data": _data({"type": "content_block_stop", "index": 0}),
        },
        {
            "event": "message_delta",
            "data": _data(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }
            ),
        },
        {
            "event": "message_stop",
            "data": _data({"type": "message_stop"}),
        },
    ]


def _user_context(text: str = "Hello") -> Any:
    return normalize_context(Context(messages=[UserMessage(content=text, timestamp=1)]))


# ---------------------------------------------------------------------------------
# Anthropic 原始 SSE 解析
# ---------------------------------------------------------------------------------


async def test_forwards_parsed_provider_stream_events_in_order():
    model = get_model("anthropic", "claude-haiku-4-5")
    provider_events: list[Any] = []
    event_models: list[Any] = []

    async def on_provider_stream_event(event, event_model):
        provider_events.append(event)
        event_models.append(event_model)

    result = await stream(
        model,
        _user_context(),
        AnthropicOptions(
            api_key="test-key",
            fetch=_sse_fetch(_minimal_anthropic_events()),
            on_provider_stream_event=on_provider_stream_event,
        ),
    ).result()

    assert result.stop_reason == StopReason.STOP
    assert [event["type"] for event in provider_events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert event_models == [model, model, model, model, model, model]


async def test_keeps_signed_thinking_replayable_when_a_proxy_relabels_the_model():
    # earendil-works/pi#9188 回归测试。
    model = get_model("anthropic", "claude-opus-5")
    response_model = "kimi-for-coding"
    initial_context = _user_context()

    first = await stream(
        model,
        initial_context,
        AnthropicOptions(
            api_key="test-key",
            fetch=_sse_fetch(
                _response_model_sse_events(
                    response_model,
                    {"type": "thinking", "thinking": "reasoning", "signature": "signature"},
                )
            ),
        ),
    ).result()

    assert first.model == model.id
    assert first.response_model == response_model

    transformed = transform_messages([*initial_context.messages, first], model)
    replayed_assistant = next(message for message in transformed if message.role == "assistant")
    assert to_json(replayed_assistant.content) == [
        {"type": "thinking", "thinking": "reasoning", "thinkingSignature": "signature"}
    ]


async def test_uses_a_returned_fallback_model_for_cost_attribution():
    fallback_model = "fallback-model"
    model = replace(
        get_model("anthropic", "claude-opus-5"),
        compat=ModelCompat(
            allowed_fallback_models=[
                AnthropicAllowedFallbackModel(
                    provider="anthropic",
                    model=fallback_model,
                    cost=ModelCost(input=3, output=5, cache_read=0, cache_write=0),
                )
            ]
        ),
    )

    result = await stream(
        model,
        _user_context(),
        AnthropicOptions(
            api_key="test-key",
            fetch=_sse_fetch(
                _response_model_sse_events(fallback_model, {"type": "text", "text": "done"})
            ),
        ),
    ).result()

    assert result.model == model.id
    assert result.response_model == fallback_model
    assert result.usage.cost.input == pytest.approx(0.0003, abs=0.5e-10)
    assert result.usage.cost.output == pytest.approx(0.0001, abs=0.5e-10)


async def test_fails_safely_when_anthropic_falls_back_after_output_begins():
    model = get_model("anthropic", "claude-opus-5")
    events = [
        {
            "event": "message_start",
            "data": _data(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_fallback",
                        "model": "claude-opus-5",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                }
            ),
        },
        {
            "event": "content_block_start",
            "data": _data(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "partial"},
                }
            ),
        },
        {
            "event": "content_block_stop",
            "data": _data({"type": "content_block_stop", "index": 0}),
        },
        {
            "event": "content_block_start",
            "data": _data(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "fallback",
                        "from": {"model": "claude-opus-5"},
                        "to": {"model": "claude-opus-4-8"},
                    },
                }
            ),
        },
    ]

    result = await stream(
        model,
        _user_context(),
        AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events)),
    ).result()

    assert result.stop_reason == StopReason.ERROR
    assert "unsupported mid-output model fallback" in (result.error_message or "")


async def test_forces_streaming_after_an_on_payload_replacement():
    sink: list[httpx.Request] = []

    def on_payload(payload, _model):
        return {**payload, "stream": False}

    await stream(
        get_model("anthropic", "claude-fable-5-1"),
        _user_context(),
        AnthropicOptions(
            api_key="test-key",
            fetch=_sse_fetch(_minimal_anthropic_events(), sink=sink),
            on_payload=on_payload,
        ),
    ).result()

    assert json.loads(sink[0].content)["stream"] is True


async def test_omits_the_interleaved_thinking_beta_when_thinking_is_disabled():
    sink: list[httpx.Request] = []

    await stream(
        get_model("anthropic", "claude-haiku-4-5"),
        _user_context(),
        AnthropicOptions(
            api_key="test-key",
            fetch=_sse_fetch(_minimal_anthropic_events(), sink=sink),
            thinking_enabled=False,
        ),
    ).result()

    assert "interleaved-thinking-2025-05-14" not in _beta_features(sink[0])


async def test_passes_managed_beta_features_to_injected_clients():
    sink: list[httpx.Request] = []

    result = await stream(
        get_model("anthropic", "claude-fable-5-1"),
        _user_context(),
        AnthropicOptions(
            api_key="test-key",
            fetch=_sse_fetch(_minimal_anthropic_events(), sink=sink),
        ),
    ).result()

    assert result.stop_reason == StopReason.STOP
    beta_features = _beta_features(sink[0])
    assert "mid-conversation-output-config-2026-07-01" in beta_features
    assert "thinking-binding-controls-2026-08-01" in beta_features


async def test_uses_the_serving_model_input_transformations_from_the_final_stream_event():
    events = [dict(event) for event in _minimal_anthropic_events()]
    events[0]["data"] = _data(
        {
            "type": "message_start",
            "message": {
                "id": "msg_transformations",
                "model": "claude-fable-5-1",
                "usage": {"input_tokens": 12, "output_tokens": 0},
                "input_transformations": [
                    {
                        "type": "thinking_dropped",
                        "path": "messages.1.content.0",
                        "reason": "prefix_binding_mismatch",
                    }
                ],
            },
        }
    )
    delta = json.loads(events[4]["data"])
    delta["input_transformations"] = [
        {
            "type": "thinking_dropped",
            "path": "messages.3.content.0",
            "reason": "model_binding_mismatch",
        }
    ]
    events[4]["data"] = _data(delta)

    result = await stream(
        get_model("anthropic", "claude-fable-5-1"),
        _user_context(),
        AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events)),
    ).result()

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.type == "anthropic_input_transformations"
    assert isinstance(diagnostic.timestamp, (int, float))
    assert diagnostic.details == {
        "transformations": [
            {
                "type": "thinking_dropped",
                "path": "messages.3.content.0",
                "reason": "model_binding_mismatch",
            }
        ]
    }


async def test_repairs_malformed_sse_json_and_malformed_streamed_tool_json():
    model = get_model("anthropic", "claude-haiku-4-5")
    context = normalize_context(
        Context(
            messages=[UserMessage(content="Use the edit tool.", timestamp=1)],
            tools=[
                Tool(
                    name="edit",
                    description="Edit a file.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                )
            ],
        )
    )

    # ``String.raw`` 模板：``\"`` 对外层 JSON 而言仍是转义引号，而 ``\H`` 与字面制表符
    # 会原样进入流式传输的工具 JSON，解析器必须将其修复。
    malformed_tool_json_delta = (
        '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"path\\":\\"A\\H\\",\\"text\\":\\"col1\tcol2\\"}"}}'
    )

    events = [
        {
            "event": "message_start",
            "data": _data(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        },
        {
            "event": "content_block_start",
            "data": _data(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_test",
                        "name": "edit",
                        "input": {},
                    },
                }
            ),
        },
        {"event": "content_block_delta", "data": malformed_tool_json_delta},
        {
            "event": "content_block_stop",
            "data": _data({"type": "content_block_stop", "index": 0}),
        },
        {
            "event": "message_delta",
            "data": _data(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        },
        {
            "event": "message_stop",
            "data": _data({"type": "message_stop"}),
        },
    ]

    stream_object = stream(
        model, context, AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events))
    )
    result = await stream_object.result()

    assert result.stop_reason == StopReason.TOOL_USE
    assert result.error_message is None

    tool_call = next(
        (block for block in result.content if getattr(block, "type", None) == "toolCall"), None
    )
    assert tool_call is not None
    assert isinstance(tool_call, ToolCall)
    assert tool_call.arguments == {"path": "A\\H", "text": "col1\tcol2"}


async def test_preserves_content_from_content_block_start_events():
    model = get_model("anthropic", "claude-haiku-4-5")
    context = normalize_context(
        Context(messages=[UserMessage(content="Say hello.", timestamp=1)])
    )
    events = [
        {
            "event": "message_start",
            "data": _data(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_initial_content",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        },
        {
            "event": "content_block_start",
            "data": _data(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "Initial text"},
                }
            ),
        },
        {
            "event": "content_block_delta",
            "data": _data(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " plus delta"},
                }
            ),
        },
        {
            "event": "content_block_stop",
            "data": _data({"type": "content_block_stop", "index": 0}),
        },
        {
            "event": "content_block_start",
            "data": _data(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "Initial thinking",
                        "signature": "initial signature",
                    },
                }
            ),
        },
        {
            "event": "content_block_delta",
            "data": _data(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "thinking_delta", "thinking": " plus delta"},
                }
            ),
        },
        {
            "event": "content_block_delta",
            "data": _data(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "signature_delta", "signature": " plus delta"},
                }
            ),
        },
        {
            "event": "content_block_stop",
            "data": _data({"type": "content_block_stop", "index": 1}),
        },
        {
            "event": "message_delta",
            "data": _data(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        },
        {
            "event": "message_stop",
            "data": _data({"type": "message_stop"}),
        },
    ]

    stream_object = stream(
        model, context, AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events))
    )
    result = await stream_object.result()

    assert to_json(result.content) == [
        {"type": "text", "text": "Initial text plus delta"},
        {
            "type": "thinking",
            "thinking": "Initial thinking plus delta",
            "thinkingSignature": "initial signature plus delta",
        },
    ]


async def test_preserves_refusal_stop_details_from_message_delta():
    model = get_model("anthropic", "claude-fable-5")
    context = normalize_context(
        Context(messages=[UserMessage(content="blocked request", timestamp=1)])
    )
    explanation = (
        "This request triggered restrictions on violative cyber content and was blocked under "
        "Anthropic's Usage Policy. To learn more, provide feedback, or request an exemption "
        "based on how you use Claude, visit our help center: "
        "https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude."
    )
    events = [
        {
            "event": "message_start",
            "data": _data(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_01XFUDYJgAACzvnptvVoYEL",
                        "usage": {
                            "input_tokens": 412,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        },
        {
            "event": "message_delta",
            "data": _data(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "refusal",
                        "stop_details": {
                            "type": "refusal",
                            "category": "cyber",
                            "explanation": explanation,
                        },
                    },
                    "usage": {
                        "input_tokens": 412,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        },
        {
            "event": "message_stop",
            "data": _data({"type": "message_stop"}),
        },
    ]

    stream_object = stream(
        model, context, AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events))
    )
    result = await stream_object.result()

    assert result.stop_reason == StopReason.ERROR
    assert result.raw_stop_reason == "refusal"
    assert result.error_message == explanation


async def test_preserves_sensitive_stop_reasons_with_a_descriptive_error_message():
    model = get_model("anthropic", "claude-haiku-4-5")
    context = normalize_context(
        Context(messages=[UserMessage(content="blocked request", timestamp=1)])
    )
    events = [
        {
            "event": "message_start",
            "data": _data(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_sensitive",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        },
        {
            "event": "message_delta",
            "data": _data(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "sensitive"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        },
        {
            "event": "message_stop",
            "data": _data({"type": "message_stop"}),
        },
    ]

    stream_object = stream(
        model, context, AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events))
    )
    result = await stream_object.result()

    assert result.stop_reason == StopReason.ERROR
    assert result.raw_stop_reason == "sensitive"
    assert result.error_message == "Provider stopped with: sensitive"


async def test_treats_message_delta_without_usage_as_a_no_op_for_usage_accumulation():
    model = get_model("anthropic", "claude-haiku-4-5")
    context = normalize_context(
        Context(messages=[UserMessage(content="Say hello.", timestamp=1)])
    )
    events = [
        (
            {
                "event": "message_delta",
                "data": _data(
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
                ),
            }
            if event["event"] == "message_delta"
            else event
        )
        for event in _minimal_anthropic_events()
    ]

    stream_object = stream(
        model, context, AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events))
    )
    result = await stream_object.result()

    assert result.stop_reason == StopReason.STOP
    assert result.error_message is None
    assert to_json(result.content) == [{"type": "text", "text": "Hello"}]
    assert result.usage.input == 12
    assert result.usage.total_tokens == 12


async def test_ignores_unknown_sse_events_after_message_stop():
    model = get_model("anthropic", "claude-haiku-4-5")
    context = normalize_context(
        Context(messages=[UserMessage(content="Say hello.", timestamp=1)])
    )
    events = [
        *_minimal_anthropic_events(),
        {"event": "done", "data": "[DONE]"},
        {"event": "proxy.stats", "data": "not json"},
    ]

    stream_object = stream(
        model, context, AnthropicOptions(api_key="test-key", fetch=_sse_fetch(events))
    )
    result = await stream_object.result()

    assert result.stop_reason == StopReason.STOP
    assert result.error_message is None
    assert to_json(result.content) == [{"type": "text", "text": "Hello"}]
