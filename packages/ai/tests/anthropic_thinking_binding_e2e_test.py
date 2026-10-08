"""Anthropic 签名 thinking 块绑定（signed-thinking block binding）的在线 E2E 一致性测试。

移植自 ``packages/ai/test/anthropic-thinking-binding-e2e.test.ts``，需要
``ANTHROPIC_API_KEY`` 才会执行，无密钥时整套用例保持离线。
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.compat import get_model, normalize_context
from pi_ai.types import AssistantMessage, Context, UserMessage

ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY"))
MODEL = get_model("anthropic", "claude-fable-5-1")


def user(content: str, timestamp: int) -> UserMessage:
    return UserMessage(content=content, timestamp=timestamp)


def strict_binding(payload, model=None):
    thinking = payload.get("thinking") if isinstance(payload, dict) else None
    if isinstance(thinking, dict) and isinstance(thinking.get("block_binding"), dict):
        thinking["block_binding"]["prefix_mismatch_behavior"] = "error"
    return payload


async def request(context: Context, effort: str) -> AssistantMessage:
    stream_object = stream(
        MODEL,
        normalize_context(context),
        AnthropicOptions(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            cache_retention="none",
            max_tokens=1536,
            thinking_enabled=True,
            thinking_display="summarized",
            effort=effort,
            on_payload=strict_binding,
        ),
    )
    return await stream_object.result()


@pytest.mark.skipif(not ENABLED, reason="requires ANTHROPIC_API_KEY")
async def test_replays_managed_effort_markers_required_by_signed_fable_thinking():
    first_user = user("Compute 982451653 multiplied by 961748941. Return only the integer.", 1)
    first = await request(Context(messages=[first_user]), "low")
    assert first.stop_reason == "stop", first.error_message
    assert any(
        block.type == "thinking"
        and isinstance(block.thinking_signature, str)
        and len(block.thinking_signature) > 0
        for block in first.content
    )
    assert first.provider_thinking_level == "low"

    second_user = user("Reply with exactly: ok", 2)
    exact = await request(Context(messages=[first_user, first, second_user]), "high")
    assert exact.stop_reason == "stop", exact.error_message

    unmanaged_history = replace(first, provider_thinking_level=None)
    missing_marker = await request(Context(messages=[first_user, unmanaged_history, second_user]), "high")
    assert missing_marker.stop_reason == "error"
    assert "Invalid `signature`" in (missing_marker.error_message or "")
