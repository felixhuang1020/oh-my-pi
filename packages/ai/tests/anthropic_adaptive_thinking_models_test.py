"""Anthropic 自适应思考（adaptive thinking）模型的目录元数据测试。

移植自 ``packages/ai/test/anthropic-adaptive-thinking-models.test.ts``。
"""

from __future__ import annotations

import re

from pi_ai.compat import get_models, get_providers

EXPECTED_CURRENT_ADAPTIVE_THINKING_MODELS = [
    "anthropic/claude-fable-5",
    "anthropic/claude-fable-5-1",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-5-5",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-5-5",
]

#: #9323 回归：依据目录的 effort 元数据与已验证的回退模型判定，
#: 而非依赖固定的自适应模型名单。
ADAPTIVE_NAME_PATTERN = re.compile(r"opus[-.](4[-.][678]|5)|sonnet[-.]4[-.]6|sonnet[-.]5|fable[-.]5")


def test_marks_builtin_anthropic_messages_models_that_use_adaptive_thinking():
    flagged_models = sorted(
        f"{model.provider}/{model.id}"
        for provider in get_providers()
        for model in get_models(provider)
        if getattr(model, "api", None) == "anthropic-messages"
        and (model.compat is not None and getattr(model.compat, "force_adaptive_thinking", None) is True)
    )

    assert set(EXPECTED_CURRENT_ADAPTIVE_THINKING_MODELS).issubset(set(flagged_models))
    assert all(ADAPTIVE_NAME_PATTERN.search(model_id) for model_id in flagged_models)
