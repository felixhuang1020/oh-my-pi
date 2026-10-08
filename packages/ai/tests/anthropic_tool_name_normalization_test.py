"""Anthropic 工具名归一化的测试。

移植自 ``packages/ai/test/anthropic-tool-name-normalization.test.ts``。上游套件完全依赖
本地保存的 Anthropic 凭据并请求真实 API；这四个用例以同样的门控移植（离线时
跳过）。由于归一化本身是确定性的，本文件另用 mock 传输在适配器上覆盖文档所述的
出站/入站往返。
"""

from __future__ import annotations


import json
import os

import httpx
import pytest

from auth_test_utils import AUTH_PATH

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.compat import get_model, normalize_context
from pi_ai.types import Context, Model, Tool, UserMessage


def _anthropic_api_key_present() -> bool:
    """同步门控：环境中或 ``auth.json`` 中存在 Anthropic api key。"""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    try:
        storage = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    entry = storage.get("anthropic") if isinstance(storage, dict) else None
    return isinstance(entry, dict) and entry.get("type") == "api_key" and bool(entry.get("key"))


HAS_ANTHROPIC_API_KEY = _anthropic_api_key_present()

MODEL = get_model("anthropic", "claude-sonnet-4-6")


def _tool_use_sse(tool_name: str, arguments_json: str) -> bytes:
    events = [
        (
            "message_start",
            {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6", "usage": {"input_tokens": 5, "output_tokens": 1}}},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": {}},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": arguments_json}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()


def _context(tool: Tool) -> Context:
    return Context(
        system_prompt=f"Use the {tool.name} tool when asked.",
        messages=[UserMessage(content=f"Use the {tool.name} tool.", timestamp=0)],
        tools=[tool],
    )


@pytest.mark.parametrize(
    ("user_name", "wire_name"),
    [
        ("todowrite", "TodoWrite"),
        ("read", "Read"),
        ("write", "Write"),
        ("edit", "Edit"),
        ("bash", "Bash"),
        ("find", "find"),
        ("my_custom_tool", "my_custom_tool"),
    ],
)
async def test_outbound_names_use_the_cc_casing_and_inbound_names_round_trip(user_name, wire_name):
    tool = Tool(
        name=user_name,
        description="A tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    )
    sink: list[httpx.Request] = []

    async def fetch(request: httpx.Request) -> httpx.Response:
        sink.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_tool_use_sse(wire_name, '{"value": "x"}'),
            request=request,
        )

    events = [
        event
        async for event in stream(
            MODEL, normalize_context(_context(tool)), AnthropicOptions(api_key="sk-ant-oat-test", fetch=fetch)
        )
    ]

    body = json.loads(sink[0].content)
    assert [entry["name"] for entry in body["tools"]] == [wire_name]

    final = events[-1]
    assert final.type == "done"
    tool_calls = [block for block in final.message.content if block.type == "toolCall"]
    assert [call.name for call in tool_calls] == [user_name]


async def test_inbound_names_without_a_matching_tool_pass_through():
    tool = Tool(name="find", description="Find files", parameters={"type": "object", "properties": {}})

    async def fetch(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_tool_use_sse("Glob", '{"pattern": "*.ts"}'),
            request=request,
        )

    events = [
        event
        async for event in stream(
            MODEL, normalize_context(_context(tool)), AnthropicOptions(api_key="sk-ant-oat-test", fetch=fetch)
        )
    ]
    final = events[-1]
    tool_calls = [block for block in final.message.content if block.type == "toolCall"]
    assert [call.name for call in tool_calls] == ["Glob"]


# ---------------------------------------------------------------------------------
# 在线 E2E（门控）
# ---------------------------------------------------------------------------------


async def _live_tool_call_name(tool: Tool) -> str | None:
    """用保存的 api key 请求真实 API，并截获工具调用名。"""
    from auth_test_utils import resolve_api_key

    token = await resolve_api_key("anthropic")
    stream_object = stream(MODEL, normalize_context(_context(tool)), AnthropicOptions(api_key=token))
    tool_call_name: str | None = None
    async for event in stream_object:
        if event.type == "toolcall_end":
            block = event.partial.content[event.content_index]
            if block.type == "toolCall":
                tool_call_name = block.name
    response = await stream_object.result()
    assert response.stop_reason == "toolUse", response.error_message
    return tool_call_name


@pytest.mark.skipif(not HAS_ANTHROPIC_API_KEY, reason="requires a stored Anthropic api key")
async def test_should_normalize_user_defined_tool_matching_cc_name():
    tool = Tool(
        name="todowrite",
        description="Write a todo item",
        parameters={"type": "object", "properties": {"task": {"type": "string", "description": "The task to add"}}},
    )
    assert await _live_tool_call_name(tool) == "todowrite"


@pytest.mark.skipif(not HAS_ANTHROPIC_API_KEY, reason="requires a stored Anthropic api key")
async def test_should_handle_pis_built_in_tools():
    tool = Tool(
        name="read",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}},
    )
    assert await _live_tool_call_name(tool) == "read"


@pytest.mark.skipif(not HAS_ANTHROPIC_API_KEY, reason="requires a stored Anthropic api key")
async def test_should_not_map_find_to_glob():
    tool = Tool(
        name="find",
        description="Find files by pattern",
        parameters={"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob pattern"}}},
    )
    assert await _live_tool_call_name(tool) == "find"


@pytest.mark.skipif(not HAS_ANTHROPIC_API_KEY, reason="requires a stored Anthropic api key")
async def test_should_handle_custom_tools_that_do_not_match_any_cc_tool_names():
    tool = Tool(
        name="my_custom_tool",
        description="A custom tool",
        parameters={"type": "object", "properties": {"input": {"type": "string", "description": "Input value"}}},
    )
    assert await _live_tool_call_name(tool) == "my_custom_tool"
