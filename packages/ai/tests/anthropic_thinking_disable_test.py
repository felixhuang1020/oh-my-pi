"""Anthropic thinking 禁用载荷的测试（附带一个门控的在线 E2E）。

移植自 ``packages/ai/test/anthropic-thinking-disable.test.ts``。最后一个 E2E 用例会请求
真实的 Anthropic API，未设置 ``ANTHROPIC_API_KEY`` 时跳过。
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

import pytest

from pi_ai.compat import get_model, stream_simple
from pi_ai.types import Context, Model, SimpleStreamOptions, UserMessage


class PayloadCaptured(Exception):
    """从 ``onPayload`` 抛出的哨兵异常，用于截获请求体。"""


def make_payload_capture_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=0)])


async def capture_payload(model: Model, **options) -> dict:
    captured: dict = {}

    def on_payload(payload, captured_model):
        captured["payload"] = payload
        raise PayloadCaptured()

    resolved_model = replace(model, base_url="http://127.0.0.1:9")
    stream_object = stream_simple(
        resolved_model,
        make_payload_capture_context(),
        SimpleStreamOptions(api_key="fake-key", on_payload=on_payload, **options),
    )
    await stream_object.result()

    assert "payload" in captured, "Expected payload to be captured before request failure"
    return captured["payload"]


async def test_sends_thinking_disabled_for_budget_based_reasoning_models_when_thinking_is_off():
    payload = await capture_payload(get_model("anthropic", "claude-sonnet-4-5"))
    assert payload["thinking"] == {"type": "disabled"}
    assert payload.get("output_config") is None


async def test_sends_thinking_disabled_for_adaptive_reasoning_models_when_thinking_is_off():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-6"))
    assert payload["thinking"] == {"type": "disabled"}
    assert payload.get("output_config") is None


async def test_sends_thinking_disabled_for_claude_opus_4_8_when_thinking_is_off():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-8"))
    assert payload["thinking"] == {"type": "disabled"}
    assert payload.get("output_config") is None


async def test_omits_thinking_disabled_for_claude_fable_5_when_thinking_is_off():
    payload = await capture_payload(get_model("anthropic", "claude-fable-5"))
    assert payload.get("thinking") is None
    assert payload.get("output_config") is None


async def test_uses_adaptive_thinking_for_claude_opus_4_8_when_reasoning_is_enabled():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-8"), reasoning="high")
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}


async def test_uses_adaptive_thinking_for_claude_sonnet_5_when_reasoning_is_enabled():
    payload = await capture_payload(get_model("anthropic", "claude-sonnet-5"), reasoning="high")
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}


async def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-8"), reasoning="xhigh")
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}


# ---------------------------------------------------------------------------------
# 在线 E2E（门控）
# ---------------------------------------------------------------------------------


def make_e2e_context() -> Context:
    return Context(
        system_prompt="You are a precise assistant. Follow the requested output format exactly.",
        messages=[
            UserMessage(
                content=(
                    "Before replying, carefully solve 36863 * 5279 internally. Then reply with the word pong "
                    "repeated exactly 40 times, separated by single spaces. Do not add any other text."
                ),
                timestamp=0,
            )
        ],
    )


def count_pongs(text: str) -> int:
    return len(re.findall(r"\bpong\b", text, flags=re.IGNORECASE))


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY")
async def test_disables_thinking_for_claude_reasoning_models():
    model = get_model("anthropic", "claude-sonnet-4-5")
    stream_object = stream_simple(model, make_e2e_context(), SimpleStreamOptions(temperature=0, max_tokens=160))

    thinking_event_count = 0
    thinking_char_count = 0
    async for event in stream_object:
        if event.type in ("thinking_start", "thinking_end"):
            thinking_event_count += 1
        if event.type == "thinking_delta":
            thinking_event_count += 1
            thinking_char_count += len(event.delta or "")

    response = await stream_object.result()
    assert response.stop_reason == "stop", response.error_message

    text = "".join(block.text for block in response.content if block.type == "text").strip()

    assert thinking_event_count == 0
    assert thinking_char_count == 0
    assert "thinking" not in [block.type for block in response.content]
    assert count_pongs(text) >= 35
