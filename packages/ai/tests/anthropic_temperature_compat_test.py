"""Anthropic temperature 兼容性的测试。

移植自 ``packages/ai/test/anthropic-temperature-compat.test.ts``。``onPayload`` 在任何
HTTP 请求发出前抛出哨兵异常以截获请求体，与 TS 测试的做法完全一致。
"""

from __future__ import annotations

from dataclasses import replace

from pi_ai.compat import get_model, stream_simple
from pi_ai.types import Context, Model, ModelCompat, SimpleStreamOptions, UserMessage


class PayloadCaptured(Exception):
    """从 ``onPayload`` 抛出的哨兵异常，在请求发出前将其终止。"""


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=0)])


def make_custom_model(compat: ModelCompat | None = None) -> Model:
    return Model(
        id="vendor--claude-opus-4-7",
        name="Vendor Proxy Opus 4.7",
        api="anthropic-messages",
        provider="vendor-proxy",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        context_window=200000,
        max_tokens=32000,
        compat=compat,
    )


async def capture_payload(model: Model, **options) -> dict:
    captured: dict = {}

    def on_payload(payload, captured_model):
        captured["payload"] = payload
        raise PayloadCaptured()

    resolved_model = replace(model, base_url="http://127.0.0.1:9")
    stream_object = stream_simple(
        resolved_model,
        make_context(),
        SimpleStreamOptions(api_key="fake-key", on_payload=on_payload, **options),
    )
    await stream_object.result()

    assert "payload" in captured, "Expected payload to be captured before request failure"
    return captured["payload"]


async def test_omits_temperature_for_claude_opus_4_7():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-7"), temperature=0)
    assert payload.get("temperature") is None


async def test_omits_temperature_for_claude_opus_4_8():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-8"), temperature=0)
    assert payload.get("temperature") is None


async def test_omits_default_temperature_for_claude_opus_4_7():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-7"), temperature=1)
    assert payload.get("temperature") is None


async def test_keeps_temperature_for_claude_opus_4_6():
    payload = await capture_payload(get_model("anthropic", "claude-opus-4-6"), temperature=0)
    assert payload.get("temperature") == 0


async def test_keeps_temperature_for_claude_sonnet_4_6():
    payload = await capture_payload(get_model("anthropic", "claude-sonnet-4-6"), temperature=0)
    assert payload.get("temperature") == 0


async def test_omits_temperature_for_custom_models_with_supports_temperature_disabled():
    payload = await capture_payload(make_custom_model(ModelCompat(supports_temperature=False)), temperature=0)
    assert payload.get("temperature") is None
