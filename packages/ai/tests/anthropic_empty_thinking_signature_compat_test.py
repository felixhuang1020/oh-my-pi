"""Anthropic 空 thinking 签名兼容性的测试。

移植自 ``packages/ai/test/anthropic-empty-thinking-signature-compat.test.ts``。
"""

from __future__ import annotations

from pi_ai.compat import stream_simple
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCompat,
    SimpleStreamOptions,
    ThinkingContent,
    Usage,
    UserMessage,
)


class PayloadCaptured(Exception):
    """从 ``on_payload`` 抛出的哨兵异常，用于截获请求体。"""


def make_model(allow_empty_signature: bool | None = None) -> Model:
    compat = None if allow_empty_signature is None else ModelCompat(allow_empty_signature=allow_empty_signature)
    return Model(
        id="mimo-v2.5-pro",
        name="MiMo-V2.5-Pro",
        api="anthropic-messages",
        provider="xiaomi-token-plan-ams",
        base_url="http://127.0.0.1:9/anthropic",
        reasoning=True,
        input=["text"],
        context_window=1048576,
        max_tokens=1024,
        compat=compat,
    )


def make_context(
    thinking_signature: str,
    thinking: str = "internal reasoning",
    provider: str = "xiaomi-token-plan-ams",
    model: str = "mimo-v2.5-pro",
) -> Context:
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking=thinking, thinking_signature=thinking_signature)],
        provider=provider,
        api="anthropic-messages",
        model=model,
        timestamp=0,
        usage=Usage(),
        stop_reason="stop",
    )
    return Context(
        messages=[
            UserMessage(content="first", timestamp=0),
            assistant,
            UserMessage(content="second", timestamp=0),
        ]
    )


async def capture_payload(model: Model, context: Context) -> dict:
    captured: dict = {}

    def on_payload(payload, captured_model):
        captured["payload"] = payload
        raise PayloadCaptured()

    stream_object = stream_simple(model, context, SimpleStreamOptions(api_key="fake-key", on_payload=on_payload))
    await stream_object.result()

    assert "payload" in captured, "Expected payload capture before request"
    return captured["payload"]


def _assistant_content(payload: dict) -> list:
    assistant = next(message for message in payload["messages"] if message["role"] == "assistant")
    return assistant["content"]


async def test_converts_empty_signature_thinking_to_text_by_default():
    payload = await capture_payload(make_model(), make_context(""))
    assert _assistant_content(payload) == [{"type": "text", "text": "internal reasoning"}]


async def test_preserves_empty_thinking_text_when_the_signature_is_present():
    payload = await capture_payload(make_model(), make_context("signed-thinking", ""))
    assert _assistant_content(payload) == [{"type": "thinking", "thinking": "", "signature": "signed-thinking"}]


async def test_preserves_empty_signature_thinking_when_allow_empty_signature_is_enabled():
    payload = await capture_payload(make_model(True), make_context(" "))
    assert _assistant_content(payload) == [
        {"type": "thinking", "thinking": "internal reasoning", "signature": ""}
    ]
