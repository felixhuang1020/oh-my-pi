"""Anthropic 1 小时缓存写入计价的测试。

移植自 ``packages/ai/test/anthropic-cache-write-1h-cost.test.ts``。TypeScript 版注入
伪造的 SDK 客户端；本移植通过 ``AnthropicOptions.client`` 注入鸭子类型的
``AnthropicClientProtocol``。
"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.compat import get_model, normalize_context
from pi_ai.types import Context, Model, UserMessage

CONTEXT = normalize_context(Context(messages=[UserMessage(content="hi", timestamp=0)]))


def create_sse_response(events: list[tuple[str, str]]) -> httpx.Response:
    body = "".join(f"event: {event}\ndata: {data}\n\n" for event, data in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


class FakeAnthropicClient:
    """返回固定响应的最小 ``AnthropicClientProtocol`` 实现。"""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def create_message(self, params, *, signal=None, timeout_ms=None) -> httpx.Response:
        return self._response


def events_with_cache_creation(cache_creation: dict[str, int] | None) -> list[tuple[str, str]]:
    start_usage: dict = {
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000_000,
    }
    if cache_creation:
        start_usage["cache_creation"] = cache_creation
    return [
        ("message_start", json.dumps({"type": "message_start", "message": {"id": "msg_test", "usage": start_usage}})),
        (
            "content_block_start",
            json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        ),
        ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
        (
            "message_delta",
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 1_000_000,
                    },
                }
            ),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]


async def _run(model: Model, response: httpx.Response):
    return await stream(model, CONTEXT, AnthropicOptions(client=FakeAnthropicClient(response))).result()


async def test_prices_the_1h_portion_at_2x_input_and_the_rest_at_the_5m_rate():
    model = get_model("anthropic", "claude-opus-4-8")
    response = create_sse_response(
        events_with_cache_creation({"ephemeral_5m_input_tokens": 600_000, "ephemeral_1h_input_tokens": 400_000})
    )
    result = await _run(model, response)

    assert result.usage.cache_write == 1_000_000
    assert result.usage.cache_write_1h == 400_000
    # 计费：600k × 6.25/Mtok + 400k × 10/Mtok = 3.75 + 4.0 = 7.75
    assert result.usage.cost.cache_write == pytest.approx(7.75, abs=1e-10)


async def test_prices_1h_cache_writes_reported_only_in_message_delta():
    model = get_model("anthropic", "claude-haiku-4-5")
    response = create_sse_response(
        [
            (
                "message_start",
                json.dumps(
                    {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 0, "output_tokens": 0}}}
                ),
            ),
            (
                "message_delta",
                json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 4,
                            "cache_creation_input_tokens": 6535,
                            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6535},
                        },
                    }
                ),
            ),
            ("message_stop", json.dumps({"type": "message_stop"})),
        ]
    )
    result = await _run(model, response)

    assert result.usage.cache_write == 6535
    assert result.usage.cache_write_1h == 6535
    assert result.usage.cost.cache_write == pytest.approx((6535 * model.cost.input * 2) / 1_000_000, abs=1e-10)


async def test_falls_back_to_the_5m_rate_when_no_breakdown_is_reported():
    model = get_model("anthropic", "claude-opus-4-8")
    response = create_sse_response(events_with_cache_creation(None))
    result = await _run(model, response)

    assert result.usage.cache_write == 1_000_000
    assert (result.usage.cache_write_1h or 0) == 0
    # 计费：1M × 6.25/Mtok = 6.25
    assert result.usage.cost.cache_write == pytest.approx(6.25, abs=1e-10)
