"""Anthropic Claude Opus 4.8 在线冒烟测试。

移植自 ``packages/ai/test/anthropic-opus-4-8-smoke.test.ts``，需要
``ANTHROPIC_API_KEY`` 才会执行。
"""

from __future__ import annotations

import os

import pytest

from pi_ai.compat import get_model, stream_simple
from pi_ai.types import Context, SimpleStreamOptions, UserMessage

ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY"))


def make_context() -> Context:
    return Context(
        system_prompt="You are a precise assistant. Follow the user's instructions exactly.",
        messages=[
            UserMessage(
                content=(
                    "Compute 48291 * 7317 and 90844 - 17729, add the results, and determine whether the sum is "
                    "divisible by 11. Reply with exactly this format and nothing else: "
                    "sum=<sum>; divisibleBy11=<yes|no>"
                ),
                timestamp=0,
            )
        ],
    )


@pytest.mark.skipif(not ENABLED, reason="requires ANTHROPIC_API_KEY")
async def test_streams_claude_opus_4_8_with_reasoning_enabled():
    model = get_model("anthropic", "claude-opus-4-8")
    captured: dict[str, dict] = {}

    def on_payload(payload, captured_model):
        captured["payload"] = payload
        return payload

    stream_object = stream_simple(
        model,
        make_context(),
        SimpleStreamOptions(reasoning="high", max_tokens=1024, on_payload=on_payload),
    )

    saw_thinking = False
    async for event in stream_object:
        if event.type in ("thinking_start", "thinking_delta", "thinking_end"):
            saw_thinking = True

    response = await stream_object.result()
    assert response.stop_reason == "stop", response.error_message
    assert not response.error_message
    assert captured["payload"]["thinking"] == {"type": "adaptive"}
    assert captured["payload"]["output_config"] == {"effort": "high"}
    assert saw_thinking is True

    thinking_block = next((block for block in response.content if block.type == "thinking"), None)
    assert thinking_block is not None and thinking_block.type == "thinking"
    assert isinstance(thinking_block.thinking_signature, str)
    assert thinking_block.thinking_signature
    assert len(thinking_block.thinking_signature) > 0

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    assert text == "sum=353418362; divisibleBy11=yes"
