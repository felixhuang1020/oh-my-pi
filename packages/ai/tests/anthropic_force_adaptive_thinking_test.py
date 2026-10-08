"""Anthropic ``forceAdaptiveThinking`` 兼容性覆盖项的测试。

移植自 ``packages/ai/test/anthropic-force-adaptive-thinking.test.ts``。
"""

from __future__ import annotations

from dataclasses import replace

from pi_ai.compat import get_model, stream_simple
from pi_ai.types import Context, Model, ModelCompat, SimpleStreamOptions, UserMessage


class PayloadCaptured(Exception):
    """从 ``onPayload`` 抛出的哨兵异常，用于截获请求体。"""


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=0)])


def make_custom_model(compat: ModelCompat | None = None) -> Model:
    return Model(
        # Id 故意不匹配任何内置的自适应模型子串，用来模拟
        # ``anthropic--claude-opus-latest`` 之类的企业代理方案。
        id="vendor--claude-opus-latest",
        name="Vendor Proxy Opus Latest",
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


async def test_sends_legacy_thinking_payload_for_custom_model_ids_by_default():
    payload = await capture_payload(make_custom_model(), reasoning="medium")
    assert payload["thinking"]["type"] == "enabled"
    assert payload.get("output_config") is None


async def test_sends_adaptive_thinking_payload_when_compat_force_adaptive_thinking_is_true():
    payload = await capture_payload(make_custom_model(ModelCompat(force_adaptive_thinking=True)), reasoning="medium")
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "medium"}


async def test_uses_adaptive_thinking_with_native_xhigh_effort_for_claude_fable_5():
    payload = await capture_payload(get_model("anthropic", "claude-fable-5"), reasoning="xhigh")
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}


async def test_allows_built_in_adaptive_models_to_opt_out_with_compat_force_adaptive_thinking_false():
    model = replace(get_model("anthropic", "claude-opus-4-8"), compat=ModelCompat(force_adaptive_thinking=False))
    payload = await capture_payload(model, reasoning="medium")
    assert payload["thinking"]["type"] == "enabled"
    assert payload.get("output_config") is None


async def test_preserves_thinking_disabled_when_reasoning_is_off_regardless_of_override():
    payload = await capture_payload(make_custom_model(ModelCompat(force_adaptive_thinking=True)))
    assert payload["thinking"] == {"type": "disabled"}
    assert payload.get("output_config") is None
