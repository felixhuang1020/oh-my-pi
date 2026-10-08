"""Anthropic 对话中途 effort 标记的测试。

移植自 ``packages/ai/test/anthropic-mid-conversation-effort.test.ts``。
"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.compat import get_model, normalize_context
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCompat,
    TextContent,
    ThinkingContent,
    Usage,
    UserMessage,
)


def managed_model(provider: str = "anthropic") -> Model:
    return Model(
        id="claude-fable-5-1",
        name="Claude Fable 5.1",
        api="anthropic-messages",
        provider=provider,
        base_url="http://127.0.0.1:9",
        reasoning=True,
        thinking_level_map={"off": None, "minimal": "low", "low": "low", "medium": "medium", "high": "high", "max": "max"},
        input=["text"],
        context_window=200000,
        max_tokens=32000,
        compat=ModelCompat(force_adaptive_thinking=True, supports_mid_convo_effort=True),
    )


def assistant(model: Model, level: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ThinkingContent(thinking="reasoning", thinking_signature="signature"),
            TextContent(text="answer"),
        ],
        api="anthropic-messages",
        provider=model.provider,
        model=model.id,
        provider_thinking_level=level,
        usage=Usage(),
        stop_reason="stop",
        timestamp=1,
    )


async def capture(model: Model, context: Context, effort: str | None = None):
    captured: dict = {}

    def on_payload(payload, captured_model):
        captured["payload"] = payload
        raise RuntimeError("payload captured")

    stream_object = stream(
        model,
        normalize_context(context),
        AnthropicOptions(
            api_key="test-key",
            cache_retention="none",
            thinking_enabled=True,
            effort=effort,
            on_payload=on_payload,
        ),
    )
    message = await stream_object.result()
    assert "payload" in captured, "Expected payload capture"
    return captured["payload"], message


def user(text: str, timestamp: int) -> UserMessage:
    return UserMessage(content=text, timestamp=timestamp)


def effort_messages(payload: dict) -> list[dict]:
    return [message for message in payload["messages"] if message["role"] == "system"]


async def test_reconstructs_an_exact_historical_marker_prefix_and_appends_the_current_marker():
    model = managed_model()
    first_payload, first_message = await capture(model, Context(messages=[user("one", 1)]), "low")
    second_payload, _ = await capture(
        model,
        Context(messages=[user("one", 1), assistant(model, "low"), user("two", 2)]),
        "high",
    )

    assert first_payload["messages"] == [
        {"role": "user", "content": "one"},
        {"role": "system", "content": [], "output_config": {"effort": "low"}},
    ]
    assert second_payload["messages"][: len(first_payload["messages"])] == first_payload["messages"]
    assert second_payload["messages"][-1] == {
        "role": "system",
        "content": [],
        "output_config": {"effort": "high"},
    }
    assert first_payload["output_config"] == {"effort": "high"}
    assert second_payload["output_config"] == {"effort": "high"}
    assert second_payload["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert first_message.provider_thinking_level == "low"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_preserves_native_effort(effort):
    model = managed_model()
    payload, message = await capture(model, Context(messages=[user("one", 1)]), effort)
    assert effort_messages(payload) == [{"role": "system", "content": [], "output_config": {"effort": effort}}]
    assert message.provider_thinking_level == effort


async def test_defaults_omitted_effort_to_high_and_still_enables_drop_block():
    payload, message = await capture(managed_model(), Context(messages=[user("one", 1)]))
    assert payload["messages"][-1] == {
        "role": "system",
        "content": [],
        "output_config": {"effort": "high"},
    }
    assert payload["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "drop_block"
    assert message.provider_thinking_level == "high"


async def test_does_not_invent_markers_for_legacy_or_other_provider_assistants():
    model = managed_model()
    legacy = assistant(model)
    other_provider = assistant(model, "low")
    other_provider.provider = "other-provider"
    payload, _ = await capture(
        model,
        Context(
            messages=[
                user("one", 1),
                legacy,
                user("two", 2),
                other_provider,
                user("three", 3),
            ]
        ),
        "medium",
    )
    assert effort_messages(payload) == [{"role": "system", "content": [], "output_config": {"effort": "medium"}}]


async def test_leaves_unsupported_models_on_top_level_effort():
    model = managed_model()
    model.compat = ModelCompat(force_adaptive_thinking=True)
    payload, message = await capture(model, Context(messages=[user("one", 1)]), "low")
    assert payload["messages"] == [{"role": "user", "content": "one"}]
    assert payload["output_config"] == {"effort": "low"}
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert message.provider_thinking_level is None


async def test_sends_the_effort_and_binding_beta_headers():
    captured_beta: dict[str, str | None] = {}

    events = [
        {"type": "message_start", "message": {"id": "msg_test", "model": "claude-fable-5-1", "usage": {"input_tokens": 1, "output_tokens": 0}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 1, "output_tokens": 1}},
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)

    async def fetch(request: httpx.Request) -> httpx.Response:
        captured_beta["beta"] = request.headers.get("anthropic-beta")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body, request=request)

    result = await stream(
        managed_model(),
        normalize_context(Context(messages=[user("one", 1)])),
        AnthropicOptions(api_key="test-key", cache_retention="none", fetch=fetch),
    ).result()

    assert result.stop_reason == "stop"
    assert "mid-conversation-output-config-2026-07-01" in (captured_beta["beta"] or "")
    assert "thinking-binding-controls-2026-08-01" in (captured_beta["beta"] or "")


def test_generates_exact_model_and_transport_gates():
    direct = get_model("anthropic", "claude-fable-5-1")
    unsupported = get_model("anthropic", "claude-opus-4-8")

    assert direct.compat.supports_mid_convo_effort is True
    assert direct.thinking_level_map["off"] is None
    assert (unsupported.compat.supports_mid_convo_effort if unsupported.compat else None) is None
    # 原生 Anthropic 传输上的 Opus 5 不配置 ``configuration_update`` 回退模型。
    opus_5_anthropic = get_model("anthropic", "claude-opus-5")
    assert (opus_5_anthropic.compat.allowed_fallback_models if opus_5_anthropic.compat else None) is None
