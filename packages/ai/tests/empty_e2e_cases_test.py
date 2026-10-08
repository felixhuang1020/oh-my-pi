"""移植 ``packages/ai/test/empty.test.ts``（live provider E2E）。"""

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
    _case('Google Provider Empty Messages', 'should handle empty content array', 'GEMINI_API_KEY'),
    _case('Google Provider Empty Messages', 'should handle empty string content', 'GEMINI_API_KEY'),
    _case('Google Provider Empty Messages', 'should handle whitespace-only content', 'GEMINI_API_KEY'),
    _case('Google Provider Empty Messages', 'should handle empty assistant message in conversation', 'GEMINI_API_KEY'),
    _case('OpenAI Responses Provider Empty Messages', 'should handle empty content array', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider Empty Messages', 'should handle empty string content', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider Empty Messages', 'should handle whitespace-only content', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider Empty Messages', 'should handle empty assistant message in conversation', 'OPENAI_API_KEY'),
    _case('Anthropic Provider Empty Messages', 'should handle empty content array', 'ANTHROPIC_API_KEY'),
    _case('Anthropic Provider Empty Messages', 'should handle empty string content', 'ANTHROPIC_API_KEY'),
    _case('Anthropic Provider Empty Messages', 'should handle whitespace-only content', 'ANTHROPIC_API_KEY'),
    _case('Anthropic Provider Empty Messages', 'should handle empty assistant message in conversation', 'ANTHROPIC_API_KEY'),
    _case('MiniMax Provider Empty Messages', 'should handle empty content array', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider Empty Messages', 'should handle empty string content', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider Empty Messages', 'should handle whitespace-only content', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider Empty Messages', 'should handle empty assistant message in conversation', 'MINIMAX_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider Empty Messages', 'should handle empty content array', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider Empty Messages', 'should handle empty string content', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider Empty Messages', 'should handle whitespace-only content', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider Empty Messages', 'should handle empty assistant message in conversation', 'XIAOMI_API_KEY'),
]


@pytest.mark.parametrize("group,test_name,gate", CASES)
def test_empty_live_case(group: str, test_name: str, gate: str) -> None:
    """上游对每个用例发起一次真实 provider 请求。"""
    raise AssertionError("live provider E2E must not run")
