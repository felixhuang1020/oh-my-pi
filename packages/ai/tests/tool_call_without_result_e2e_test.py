"""移植自 ``packages/ai/test/tool-call-without-result.test.ts``（真实 provider E2E）。"""

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
    _case('Google Provider', 'should filter out tool calls without corresponding tool results', 'GEMINI_API_KEY'),
    _case('OpenAI Responses Provider', 'should filter out tool calls without corresponding tool results', 'OPENAI_API_KEY'),
    _case('Anthropic Provider', 'should filter out tool calls without corresponding tool results', 'ANTHROPIC_API_KEY'),
    _case('MiniMax Provider', 'should filter out tool calls without corresponding tool results', 'MINIMAX_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider', 'should filter out tool calls without corresponding tool results', 'XIAOMI_API_KEY'),
]


@pytest.mark.parametrize("group,test_name,gate", CASES)
def test_tool_call_without_result_live_case(group: str, test_name: str, gate: str) -> None:
    """上游每个用例都会发起一次真实 provider 请求。"""
    raise AssertionError("live provider E2E must not run")
