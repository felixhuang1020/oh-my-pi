"""移植 ``packages/ai/test/max-thinking.test.ts``。

所有用例均离线运行：涉及真实 provider 目录的 Codex 用例已随裁剪删除，
保留下来的用例针对手工构造的 :class:`~pi_ai.types.Model`。
"""

from __future__ import annotations

from pi_ai.models import clamp_thinking_level, get_supported_thinking_levels
from pi_ai.types import Model, ModelCost


def test_max_thinking_level_is_opt_in_for_ordinary_reasoning_models() -> None:
    model = Model(
        id="ordinary-reasoning",
        name="Ordinary Reasoning",
        api="openai-completions",
        provider="test",
        base_url="https://example.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
    )

    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high"]
    assert clamp_thinking_level(model, "max") == "high"


def test_supports_a_hole_between_high_and_max() -> None:
    model = Model(
        id="high-and-max",
        name="High and Max",
        api="openai-completions",
        provider="test",
        base_url="https://example.com/v1",
        reasoning=True,
        thinking_level_map={"xhigh": None, "max": "max"},
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
    )

    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high", "max"]
    assert clamp_thinking_level(model, "xhigh") == "max"
