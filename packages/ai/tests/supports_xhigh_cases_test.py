"""移植 ``packages/ai/test/supports-xhigh.test.ts``。"""

from __future__ import annotations

import pytest

from pi_ai.models import get_supported_thinking_levels
from pi_ai.providers.catalog import chat_model_catalog
from pi_ai.utils.serde import to_json


def _model(provider: str, model_id: str):
    """在生成的目录上执行上游的 ``getModel(provider, modelId)``。"""
    return chat_model_catalog(provider).get(model_id)


def _levels(provider: str, model_id: str) -> list[str]:
    model = _model(provider, model_id)
    assert model is not None, f"{provider}/{model_id} is missing from the catalog"
    return get_supported_thinking_levels(model)


# ---------------------------------------------------------------------------------
# Anthropic 模型
# ---------------------------------------------------------------------------------


def test_includes_max_but_not_xhigh_for_anthropic_opus_4_6_on_anthropic_messages_api():
    model = _model("anthropic", "claude-opus-4-6")
    assert model is not None
    levels = get_supported_thinking_levels(model)
    assert "max" in levels
    assert "xhigh" not in levels


def test_includes_xhigh_and_max_for_anthropic_opus_4_8_on_anthropic_messages_api():
    model = _model("anthropic", "claude-opus-4-8")
    assert model is not None
    levels = get_supported_thinking_levels(model)
    assert "xhigh" in levels
    assert "max" in levels


def test_includes_xhigh_and_max_for_anthropic_opus_5_on_anthropic_messages_api():
    model = _model("anthropic", "claude-opus-5")
    assert model is not None
    levels = get_supported_thinking_levels(model)
    assert "xhigh" in levels
    assert "max" in levels


def test_includes_claude_opus_5_5_with_its_always_on_effort_levels_and_official_pricing():
    model = _model("anthropic", "claude-opus-5-5")
    assert model is not None
    assert to_json(model.cost) == {"input": 4, "output": 20, "cacheRead": 0.2, "cacheWrite": 5}
    assert model.context_window == 1_000_000
    assert model.max_tokens == 128_000
    assert model.compat is not None
    assert model.compat.force_adaptive_thinking is True
    assert model.compat.supports_mid_convo_effort is True
    assert model.compat.supports_mid_convo_system_messages is True
    assert model.compat.supports_mid_convo_tool_changes is True
    assert get_supported_thinking_levels(model) == ["low", "medium", "high", "xhigh", "max"]


def test_includes_claude_sonnet_5_5_with_managed_effort_levels_and_official_pricing():
    model = _model("anthropic", "claude-sonnet-5-5")
    assert model is not None
    assert to_json(model.cost) == {"input": 2, "output": 10, "cacheRead": 0.2, "cacheWrite": 2.5}
    assert model.context_window == 1_000_000
    assert model.max_tokens == 128_000
    assert model.compat is not None
    assert model.compat.force_adaptive_thinking is True
    assert model.compat.supports_mid_convo_effort is True
    assert model.compat.supports_mid_convo_system_messages is True
    assert model.compat.supports_mid_convo_tool_changes is True
    assert model.compat.supports_temperature is False
    assert get_supported_thinking_levels(model) == ["low", "medium", "high", "xhigh", "max"]


def test_includes_max_but_not_xhigh_for_anthropic_sonnet_4_6_on_anthropic_messages_api():
    model = _model("anthropic", "claude-sonnet-4-6")
    assert model is not None
    levels = get_supported_thinking_levels(model)
    assert "max" in levels
    assert "xhigh" not in levels


def test_includes_xhigh_and_max_for_anthropic_sonnet_5_on_anthropic_messages_api():
    model = _model("anthropic", "claude-sonnet-5")
    assert model is not None
    levels = get_supported_thinking_levels(model)
    assert "xhigh" in levels
    assert "max" in levels


def test_includes_xhigh_and_max_but_not_off_for_anthropic_claude_fable_5_on_anthropic_messages_api():
    model = _model("anthropic", "claude-fable-5")
    assert model is not None
    levels = get_supported_thinking_levels(model)
    assert "xhigh" in levels
    assert "max" in levels
    assert "off" not in levels


def test_does_not_include_xhigh_or_max_for_claude_sonnet_4_5():
    model = _model("anthropic", "claude-sonnet-4-5")
    assert model is not None
    levels = get_supported_thinking_levels(model)
    assert "xhigh" not in levels
    assert "max" not in levels


# ---------------------------------------------------------------------------------
# OpenAI 模型
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-sol", "gpt-6-luna"],
)
def test_includes_xhigh_and_max_for_openai_models(model_id: str):
    model = _model("openai", model_id)
    assert model is not None
    assert get_supported_thinking_levels(model) == ["off", "low", "medium", "high", "xhigh", "max"]


# OpenAI 对 GPT-6.1 Sol 拒绝 reasoning.effort 取值 "none"。
def test_does_not_support_off_for_gpt_6_1_sol():
    model = _model("openai", "gpt-6.1-sol")
    assert model is not None
    assert get_supported_thinking_levels(model) == ["low", "medium", "high", "xhigh", "max"]
    assert model.thinking_level_map is not None
    assert model.thinking_level_map.get("off") is None


@pytest.mark.parametrize(
    ("model_id", "cost"),
    [
        ("gpt-6-sol", {"input": 2, "output": 10, "cacheRead": 0.2, "cacheWrite": 2.5}),
        ("gpt-6-luna", {"input": 0.1, "output": 0.5, "cacheRead": 0.01, "cacheWrite": 0.125}),
        ("gpt-6.1-sol", {"input": 2, "output": 10, "cacheRead": 0.1, "cacheWrite": 2.5}),
    ],
)
def test_includes_official_metadata_for_openai_models(model_id: str, cost: dict[str, float]):
    model = _model("openai", model_id)
    assert model is not None
    assert model.input == ["text", "image"]
    assert to_json(model.cost) == {
        **cost,
        "tiers": [
            {
                "inputTokensAbove": 272000,
                "input": cost["input"] * 2,
                "output": cost["output"] * 1.5,
                "cacheRead": cost["cacheRead"] * 2,
                "cacheWrite": cost["cacheWrite"] * 2,
            }
        ],
    }
    assert model.context_window == 272000
    assert model.max_tokens == 128000
    assert model.compat is not None
    assert model.compat.supports_additional_tools is True
    assert model.compat.supports_mid_convo_system_messages is True
    assert model.compat.supports_openai_grammar_tools is True
    assert model.compat.supports_tool_search is True


def test_includes_only_medium_high_xhigh_for_openai_gpt_5_5_pro():
    model = _model("openai", "gpt-5.5-pro")
    assert model is not None
    assert get_supported_thinking_levels(model) == ["medium", "high", "xhigh"]


# ---------------------------------------------------------------------------------
# DeepSeek 模型
# ---------------------------------------------------------------------------------


def test_includes_low_high_max_plus_off_for_deepseek_v4_1_flash_on_the_deepseek_provider():
    model = _model("deepseek", "deepseek-flash")
    assert model is not None
    assert get_supported_thinking_levels(model) == ["off", "low", "high", "max"]


# ---------------------------------------------------------------------------------
# Moonshot / Kimi 模型
# ---------------------------------------------------------------------------------


def test_excludes_thinking_off_for_moonshot_kimi_k2_7_code_models():
    for provider in ("moonshotai", "moonshotai-cn"):
        model = _model(provider, "kimi-k2.7-code")
        assert model is not None
        assert get_supported_thinking_levels(model) == ["minimal", "low", "medium", "high"]


@pytest.mark.parametrize("provider", ["moonshotai", "moonshotai-cn"])
def test_uses_the_verified_effort_options_for_kimi_k3(provider: str):
    model = _model(provider, "kimi-k3")
    assert model is not None
    assert get_supported_thinking_levels(model) == ["low", "high", "max"]
