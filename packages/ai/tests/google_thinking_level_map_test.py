"""移植 ``packages/ai/test/google-thinking-level-map.test.ts``。

上游在任何 HTTP 调用之前就从 ``onPayload`` 抛异常来捕获请求 payload，
因此这些测试从不触网。
"""

from __future__ import annotations

from typing import Any

import pytest

from pi_ai.api.google_generative_ai import stream_simple as stream_simple_google
from pi_ai.api.google_shared import resolve_google_thinking_level
from pi_ai.types import (
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    ThinkingBudgets,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

CONTEXT = normalize_context(Context(messages=[UserMessage(content="Hello", timestamp=0)]))


class PayloadCaptured(Exception):
    """从 ``on_payload`` 抛出的哨兵异常，用于捕获请求体。"""


def google_model(id: str, thinking_level_map: dict[str, str | None]) -> Model:
    return Model(
        id=id,
        name=id,
        api="google-generative-ai",
        provider="test-google",
        base_url="https://example.invalid/v1beta",
        reasoning=True,
        thinking_level_map=thinking_level_map,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
    )


async def capture_payload(
    model: Model,
    reasoning: str | None = None,
    thinking_budgets: ThinkingBudgets | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def on_payload(payload: dict[str, Any], captured_model: Model) -> None:
        captured["payload"] = payload
        raise PayloadCaptured("payload captured")

    result = await stream_simple_google(
        model,
        CONTEXT,
        SimpleStreamOptions(
            api_key="test",
            reasoning=reasoning,
            thinking_budgets=thinking_budgets,
            on_payload=on_payload,
        ),
    ).result()

    assert result.error_message is not None
    assert "payload captured" in result.error_message
    assert "payload" in captured, f"{model.id} payload was not captured"
    return captured["payload"]


# ---------------------------------------------------------------------------------
# 测试 resolveGoogleThinkingLevel
# ---------------------------------------------------------------------------------


def test_resolves_default_logical_levels() -> None:
    default_expectations = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    for level, expected in default_expectations.items():
        assert resolve_google_thinking_level(google_model("gemini-3.7-flash", {}), level) == expected


def test_exhaustively_resolves_supported_logical_levels_and_mapping_values() -> None:
    mapped_expectations = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "MINIMAL": "minimal",
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
    }
    for mapped, expected in mapped_expectations.items():
        model = google_model("gemini-3.7-flash", {"high": mapped, "xhigh": mapped, "max": mapped})
        assert resolve_google_thinking_level(model, "high") == expected
        assert resolve_google_thinking_level(model, "xhigh") == expected
        assert resolve_google_thinking_level(model, "max") == expected


def test_raises_for_an_unsupported_mapping_value() -> None:
    invalid_model = google_model("gemini-3.7-flash", {"xhigh": "extreme"})
    with pytest.raises(ValueError) as raised:
        resolve_google_thinking_level(invalid_model, "xhigh")
    assert (
        "Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme"
        in str(raised.value)
    )


def test_raises_for_a_missing_mapping_value() -> None:
    with pytest.raises(ValueError) as raised:
        resolve_google_thinking_level(google_model("gemini-3.7-flash", {}), "max")
    assert (
        "Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: max -> undefined"
        in str(raised.value)
    )


# ---------------------------------------------------------------------------------
# payload 级 thinking 配置
# ---------------------------------------------------------------------------------

async def test_uses_the_lowest_supported_level_when_reasoning_is_omitted() -> None:
    model = google_model(
        "gemini-3.8-flash",
        {
            "off": None,
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": None,
            "max": None,
        },
    )
    payload = await capture_payload(model)

    assert payload["config"]["thinkingConfig"] == {"thinkingLevel": "LOW"}


async def test_preserves_native_medium_effort_for_gemini_3_1_pro() -> None:
    model = google_model(
        "gemini-3.1-pro-preview",
        {"off": None, "minimal": None, "low": "low", "medium": "medium", "high": "high", "xhigh": None, "max": None},
    )
    payload = await capture_payload(model, "medium")

    assert payload["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "MEDIUM"}


async def test_disables_gemini_2_5_thinking_when_reasoning_is_omitted() -> None:
    model = google_model("gemini-2.5-flash", {})
    payload = await capture_payload(model)

    assert payload["config"]["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.parametrize("reasoning", ["xhigh", "max"])
async def test_maps_google_generative_ai_extended_levels_to_a_supported_level(reasoning: str) -> None:
    payload = await capture_payload(
        google_model("gemini-3.7-flash", {"xhigh": "high", "max": "high"}),
        reasoning,
    )

    thinking_config = payload["config"]["thinkingConfig"]
    assert thinking_config["includeThoughts"] is True
    assert thinking_config["thinkingLevel"] == "HIGH"


async def test_honors_uppercase_provider_values_for_standard_google_generative_ai_levels() -> None:
    payload = await capture_payload(google_model("gemini-3.7-flash", {"high": "LOW"}), "high")

    assert payload["config"]["thinkingConfig"]["thinkingLevel"] == "LOW"


async def test_uses_mapped_google_generative_ai_levels_for_token_budgets() -> None:
    payload = await capture_payload(
        google_model("gemini-2.5-flash", {"xhigh": "high"}),
        "xhigh",
        ThinkingBudgets(high=1234),
    )

    assert payload["config"]["thinkingConfig"]["thinkingBudget"] == 1234
