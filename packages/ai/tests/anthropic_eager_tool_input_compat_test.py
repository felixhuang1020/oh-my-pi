"""Anthropic eager tool-input 流式兼容性的测试。

移植自 ``packages/ai/test/anthropic-eager-tool-input-compat.test.ts``。TypeScript 版用
一次性 ``node:http`` 服务器捕获请求；本移植改用适配器的 ``fetch`` 传输接缝并记录
:class:`httpx.Request`。
"""

from __future__ import annotations

import json

import httpx

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.types import Context, Model, ModelCompat, Tool, UserMessage
from pi_ai.utils.transcript import normalize_context

TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
)


def create_model(compat: ModelCompat | None = None) -> Model:
    resolved = ModelCompat(force_adaptive_thinking=True)
    if compat is not None:
        for name, value in compat.__dict__.items():
            setattr(resolved, name, value)
    return Model(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        api="anthropic-messages",
        provider="test-anthropic",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        context_window=200000,
        max_tokens=32000,
        compat=resolved,
    )


def create_context(tools: list[Tool] | None = None) -> Context:
    context = Context(messages=[UserMessage(content="Use the tool", timestamp=0)])
    if tools is None:
        tools = [TOOL]
    if tools:
        context.tools = tools
    return normalize_context(context)


def get_first_tool(body: dict) -> dict:
    tools = body.get("tools")
    assert isinstance(tools, list) and tools and isinstance(tools[0], dict), "Expected first tool in request body"
    return tools[0]


async def capture_anthropic_request(compat: ModelCompat | None, context: Context) -> httpx.Request:
    captured: list[httpx.Request] = []

    async def fetch(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"", request=request)

    stream_object = stream(
        create_model(compat),
        context,
        AnthropicOptions(api_key="test-key", cache_retention="none", fetch=fetch),
    )
    async for event in stream_object:
        if event.type in ("done", "error"):
            break

    assert captured, "Anthropic request was not captured"
    return captured[0]


async def test_sends_per_tool_eager_input_streaming_by_default():
    request = await capture_anthropic_request(None, create_context())
    body = json.loads(request.content)

    assert get_first_tool(body)["eager_input_streaming"] is True
    assert request.headers.get("anthropic-beta") is None


async def test_uses_the_legacy_fine_grained_tool_streaming_beta_when_eager_tool_input_streaming_is_disabled():
    request = await capture_anthropic_request(
        ModelCompat(supports_eager_tool_input_streaming=False), create_context()
    )
    body = json.loads(request.content)

    assert "eager_input_streaming" not in get_first_tool(body)
    assert request.headers.get("anthropic-beta") == "fine-grained-tool-streaming-2025-05-14"


async def test_does_not_send_the_legacy_fine_grained_tool_streaming_beta_when_there_are_no_tools():
    request = await capture_anthropic_request(
        ModelCompat(supports_eager_tool_input_streaming=False), create_context([])
    )
    body = json.loads(request.content)

    assert body.get("tools") is None
    assert request.headers.get("anthropic-beta") is None
