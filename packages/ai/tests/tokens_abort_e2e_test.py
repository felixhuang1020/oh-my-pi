"""移植自 ``packages/ai/test/tokens.test.ts``（真实 provider E2E）。"""

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
    _case('Google Provider', 'should include token stats when aborted mid-stream', 'GEMINI_API_KEY'),
    _case('OpenAI Responses Provider', 'should include token stats when aborted mid-stream', 'OPENAI_API_KEY'),
    _case('Anthropic Provider', 'should include token stats when aborted mid-stream', 'ANTHROPIC_API_KEY'),
    _case('MiniMax Provider', 'should include token stats when aborted mid-stream', 'MINIMAX_API_KEY'),
]


@pytest.mark.parametrize("group,test_name,gate", CASES)
def test_tokens_live_case(group: str, test_name: str, gate: str) -> None:
    """上游每个用例都会发起一次真实 provider 请求。"""
    raise AssertionError("live provider E2E must not run")
