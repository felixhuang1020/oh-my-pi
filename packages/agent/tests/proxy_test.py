"""代理 stream 函数测试。

移植自 ``packages/agent/test/proxy.test.ts``。TypeScript 测试对全局 ``fetch`` 打桩；本移植
改为经 ``ProxyStreamOptions.fetch`` 注入假 fetch，测试保持确定性且从不触网。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pi_ai.types import Context, Model, ModelCost
from pi_ai.utils.transcript import normalize_context

from pi_agent import ProxyStreamOptions, stream_proxy

MODEL = Model(
    id="gpt-5.4",
    name="GPT-5.4",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=400000,
    max_tokens=128000,
)

USAGE: dict[str, Any] = {
    "input": 0,
    "output": 0,
    "cacheRead": 0,
    "cacheWrite": 0,
    "totalTokens": 0,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
}


def make_fetch(body: str, status_code: int = 200):
    """确定性的 ``FetchFunction``，始终以 ``body`` 作为响应内容。"""
    requests: list[httpx.Request] = []

    async def fetch(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, content=body.encode("utf-8"), request=request)

    fetch.requests = requests  # type: ignore[attr-defined]
    return fetch


def proxy_context():
    return normalize_context(Context(system_prompt="", messages=[]))


async def collect(stream: Any) -> tuple[list[Any], Any]:
    events: list[Any] = []
    async for event in stream:
        events.append(event)
    result = await stream.result()
    return events, result


async def test_preserves_tool_call_metadata_received_only_on_toolcall_end() -> None:
    proxy_events: list[dict[str, Any]] = [
        {"type": "start"},
        {"type": "toolcall_start", "contentIndex": 0, "id": "call_test|fc_test", "toolName": "lookup"},
        {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"value":"hello"}'},
        {
            "type": "toolcall_end",
            "contentIndex": 0,
            "toolCall": {
                "type": "toolCall",
                "id": "call_test|fc_test",
                "name": "lookup",
                "arguments": {"value": "hello"},
                "namespace": "dynamic_tools",
            },
        },
        {"type": "done", "reason": "toolUse", "usage": USAGE},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in proxy_events)
    fetch = make_fetch(body)

    stream = stream_proxy(
        MODEL,
        proxy_context(),
        ProxyStreamOptions(auth_token="test-token", proxy_url="https://proxy.example.com", fetch=fetch),
    )
    events, result = await collect(stream)
    end_event = next(event for event in events if event.type == "toolcall_end")

    assert end_event.tool_call.namespace == "dynamic_tools"
    assert result.content[0].type == "toolCall"
    assert result.content[0].arguments == {"value": "hello"}
    assert result.content[0].namespace == "dynamic_tools"

    request = fetch.requests[0]  # type: ignore[attr-defined]
    assert str(request.url) == "https://proxy.example.com/api/stream"
    assert request.headers["authorization"] == "Bearer test-token"
    payload = json.loads(request.content)
    assert payload["model"]["id"] == "gpt-5.4"
    assert payload["context"] == {"messages": []}
    assert payload["options"] == {}


async def test_processes_terminal_metadata_when_the_event_is_not_newline_terminated() -> None:
    start = f"data: {json.dumps({'type': 'start'})}\n\n"
    done = f"data: {json.dumps({'type': 'done', 'reason': 'stop', 'usage': USAGE, 'providerThinkingLevel': 'high'})}"
    fetch = make_fetch(start + done)

    stream = stream_proxy(
        MODEL,
        proxy_context(),
        ProxyStreamOptions(auth_token="test-token", proxy_url="https://proxy.example.com", fetch=fetch),
    )
    events, result = await collect(stream)

    assert [event.type for event in events] == ["start", "done"]
    assert result.stop_reason == "stop"
    assert result.provider_thinking_level == "high"


async def test_emits_an_error_instead_of_hanging_when_the_stream_ends_without_a_terminal_event() -> None:
    body = f"data: {json.dumps({'type': 'start'})}\n\n"
    fetch = make_fetch(body)

    stream = stream_proxy(
        MODEL,
        proxy_context(),
        ProxyStreamOptions(auth_token="test-token", proxy_url="https://proxy.example.com", fetch=fetch),
    )
    events, result = await collect(stream)

    assert [event.type for event in events] == ["start", "error"]
    assert result.stop_reason == "error"
    assert "Connection closed by proxy server" in result.error_message


async def test_surfaces_a_non_ok_response_as_an_error_event() -> None:
    fetch = make_fetch(json.dumps({"error": "quota exceeded"}), status_code=429)

    stream = stream_proxy(
        MODEL,
        proxy_context(),
        ProxyStreamOptions(auth_token="test-token", proxy_url="https://proxy.example.com", fetch=fetch),
    )
    events, result = await collect(stream)

    assert [event.type for event in events] == ["error"]
    assert result.stop_reason == "error"
    assert result.error_message == "Proxy error: quota exceeded"
