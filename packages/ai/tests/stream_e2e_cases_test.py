"""移植自 ``packages/ai/test/stream.test.ts``（真实 provider E2E）。"""

from __future__ import annotations

import re

import pytest


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", value).strip("-").lower()


def _case(group: str, test_name: str, gate: str) -> pytest.ParameterSet:
    return pytest.param(
        group,
        test_name,
        gate,
        id=f"{_slug(group)}--{_slug(test_name)}",
        marks=pytest.mark.skip(
            reason=(
                "live provider E2E: upstream gates on "
                f"{gate}; the Python suite is offline"
            )
        ),
    )


CASES = [
    _case('Gemini Provider (gemini-2.5-flash)', 'should complete basic text generation', 'GEMINI_API_KEY'),
    _case('Gemini Provider (gemini-2.5-flash)', 'should handle tool calling', 'GEMINI_API_KEY'),
    _case('Gemini Provider (gemini-2.5-flash)', 'should handle streaming', 'GEMINI_API_KEY'),
    _case('Gemini Provider (gemini-2.5-flash)', 'should handle thinking', 'GEMINI_API_KEY'),
    _case('Gemini Provider (gemini-2.5-flash)', 'should handle multi-turn with thinking and tools', 'GEMINI_API_KEY'),
    _case('OpenAI Responses Provider (gpt-5.4)', 'should complete basic text generation', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider (gpt-5.4)', 'should handle tool calling', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider (gpt-5.4)', 'should handle streaming', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider (gpt-5.4)', 'should handle thinking', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider (gpt-5.4)', 'should handle multi-turn with thinking and tools', 'OPENAI_API_KEY'),
    _case('Anthropic Provider (claude-haiku-4-5)', 'should complete basic text generation', 'ANTHROPIC_API_KEY'),
    _case('Anthropic Provider (claude-haiku-4-5)', 'should handle tool calling', 'ANTHROPIC_API_KEY'),
    _case('Anthropic Provider (claude-haiku-4-5)', 'should handle streaming', 'ANTHROPIC_API_KEY'),
    _case('MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)', 'should complete basic text generation', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)', 'should handle tool calling', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)', 'should handle streaming', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)', 'should handle thinking mode', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)', 'should handle multi-turn with thinking and tools', 'MINIMAX_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via Anthropic Messages)', 'should complete basic text generation', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via Anthropic Messages)', 'should handle tool calling', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via Anthropic Messages)', 'should handle streaming', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via Anthropic Messages)', 'should handle thinking mode', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via Anthropic Messages)', 'should handle multi-turn with thinking and tools', 'XIAOMI_API_KEY'),
]


@pytest.mark.parametrize("group,test_name,gate", CASES)
def test_stream_live_case(group: str, test_name: str, gate: str) -> None:
    """上游每个用例都会发起一次真实 provider 请求。"""
    raise AssertionError("live provider E2E must not run")
