"""移植 ``packages/ai/test/google-raw-stop-reason.test.ts``。

上游用 ``vi.mock("@google/genai")`` 把 SDK 客户端替换为脚本化的
``generateContentStream``。移植版没有 Google SDK；adapter 通过 :mod:`httpx`
执行同样的 ``:streamGenerateContent?alt=sse`` 调用，因此被 mock 的 SDK 换成了
可注入的 ``options.fetch`` 接缝和预置 SSE 响应体。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from platform import machine, release, system
from typing import Any

import httpx
import pytest

from pi_ai.api.google_generative_ai import GoogleOptions, stream as stream_google_generative_ai
from pi_ai.compat import get_model
from pi_ai.types import Context, StopReason, TextContent, TranscriptContext, UserMessage
from pi_ai.utils.transcript import normalize_context

PI_USER_AGENT = f"pi ({system().lower()} {release()}; {machine()})"

CONTEXT: TranscriptContext = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=0)]))

GOOGLE_MODEL = get_model("google", "gemini-2.5-flash")

ADAPTERS = [
    pytest.param("google", id="Google Generative AI"),
]

USAGE_METADATA = {"promptTokenCount": 1, "candidatesTokenCount": 0, "totalTokenCount": 1}


def _sse(*chunks: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode()


def _finish_chunk(finish_reason: str, *, include_function_call: bool = False) -> dict[str, Any]:
    candidate: dict[str, Any] = {"finishReason": finish_reason}
    if include_function_call:
        candidate["content"] = {
            "parts": [
                {
                    "functionCall": {
                        "id": "call-1",
                        "name": "echo",
                        "args": {"value": "truncated"},
                    }
                }
            ]
        }
    return {
        "responseId": "google-response-id",
        "candidates": [candidate],
        "usageMetadata": USAGE_METADATA,
    }


class RecordingFetch:
    """充当 ``options.fetch`` 传输的脚本化 SDK 流。"""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=self.body,
            request=request,
        )


def model_for(adapter: str):
    return GOOGLE_MODEL


def open_stream(
    adapter: str,
    fetch: RecordingFetch,
    *,
    headers: dict[str, str] | None = None,
    on_provider_stream_event: Callable[..., Any] | None = None,
):
    model = model_for(adapter)
    return stream_google_generative_ai(
        model,
        CONTEXT,
        GoogleOptions(
            api_key="test-api-key",
            headers=headers,
            fetch=fetch,
            on_provider_stream_event=on_provider_stream_event,
        ),
    )


async def test_preserves_raw_gemini_finish_reasons_for_google_generative_ai_errors() -> None:
    fetch = RecordingFetch(_sse(_finish_chunk("MALFORMED_FUNCTION_CALL")))

    message = await open_stream("google", fetch).result()

    assert message.stop_reason == StopReason.ERROR
    assert message.raw_stop_reason == "MALFORMED_FUNCTION_CALL"
    assert message.error_message == "Provider stopped with: MALFORMED_FUNCTION_CALL"


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_preserves_max_tokens_with_a_tool_call_as_length(adapter: str) -> None:
    fetch = RecordingFetch(_sse(_finish_chunk("MAX_TOKENS", include_function_call=True)))

    message = await open_stream(adapter, fetch).result()

    assert message.stop_reason == StopReason.LENGTH
    assert message.raw_stop_reason == "MAX_TOKENS"
    assert any(block.type == "toolCall" for block in message.content) is True


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_maps_stop_with_a_tool_call_to_tool_use(adapter: str) -> None:
    fetch = RecordingFetch(_sse(_finish_chunk("STOP", include_function_call=True)))

    message = await open_stream(adapter, fetch).result()

    assert message.stop_reason == StopReason.TOOL_USE
    assert message.raw_stop_reason == "STOP"
    assert any(block.type == "toolCall" for block in message.content) is True


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_forwards_each_sdk_chunk_in_order_before_normalizing_it(adapter: str) -> None:
    stream_chunks = [
        {
            "responseId": "resp_google",
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
        },
        {
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
        },
    ]
    fetch = RecordingFetch(_sse(*stream_chunks))
    received: list[Any] = []
    event_models: list[Any] = []

    async def on_provider_stream_event(chunk: Any, event_model: Any) -> None:
        received.append(chunk)
        event_models.append(event_model)

    result = await open_stream(adapter, fetch, on_provider_stream_event=on_provider_stream_event).result()

    # 上游还断言对象同一性（``toBe``）；移植版从 JSON 重新解析每个 SSE 帧，
    # 因此 chunk 按值相等但都是新建的 dict。
    assert received == stream_chunks
    assert received[0] == stream_chunks[0]
    assert received[1] == stream_chunks[1]
    assert event_models == [model_for(adapter), model_for(adapter)]
    assert result.stop_reason == StopReason.STOP
    assert result.response_id == "resp_google"
    assert result.content == [TextContent(text="hello")]


async def capture_google_headers(headers: dict[str, str] | None = None) -> httpx.Headers:
    fetch = RecordingFetch(_sse(_finish_chunk("STOP")))

    await open_stream("google", fetch, headers=headers).result()

    assert len(fetch.requests) == 1
    return fetch.requests[0].headers


async def test_uses_pis_user_agent_by_default() -> None:
    assert (await capture_google_headers())["User-Agent"] == PI_USER_AGENT


async def test_lets_explicit_headers_override_the_default_user_agent() -> None:
    assert (await capture_google_headers({"User-Agent": "custom-agent"}))["User-Agent"] == "custom-agent"
