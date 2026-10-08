"""移植 ``packages/ai/test/context-overflow.test.ts``。

上游每个用例都会驱动真实 provider 直到其报告上下文窗口溢出，再对真实的 assistant
消息断言 ``isContextOverflow``。上游文件没有离线 fixture（错误文本来自实时响应），
因此这里把每个用例保留为带完整上游门控条件的跳过测试。

纯 ``isContextOverflow`` 逻辑本身由 ``packages/ai/tests/overflow_test.py``
离线覆盖（由另一工作流负责）。
"""

from __future__ import annotations

import re

import pytest

NOTE = (
    "live provider E2E: upstream sends an over-long prompt to {gate} and asserts "
    "isContextOverflow on the real error response; the Python suite is offline"
)


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", value).strip("-").lower()


def _case(group: str, test_name: str, gate: str) -> pytest.ParameterSet:
    return pytest.param(
        group,
        test_name,
        gate,
        id=f"{_slug(group)}--{_slug(test_name)}",
        marks=pytest.mark.skip(reason=NOTE.format(gate=gate)),
    )


CASES = [
    _case("Anthropic (API Key)", "claude-haiku-4-5 - should detect overflow via isContextOverflow", "ANTHROPIC_API_KEY"),
    _case("OpenAI Responses", "gpt-4o - should detect overflow via isContextOverflow", "OPENAI_API_KEY"),
    _case("Google", "gemini-2.5-flash - should detect overflow via isContextOverflow", "GEMINI_API_KEY"),
    _case("MiniMax", "MiniMax-M2.7 - should detect overflow via isContextOverflow", "MINIMAX_API_KEY"),
    _case("Xiaomi MiMo (API billing)", "mimo-v2.5-pro - should detect overflow via isContextOverflow", "XIAOMI_API_KEY"),
]


@pytest.mark.parametrize("group,test_name,gate", CASES)
def test_context_overflow_live_case(group: str, test_name: str, gate: str) -> None:
    """上游用例：向 ``group`` 发送超长 prompt 并断言溢出。

    实时实现会解析 provider 凭据，用超过 ``model.context_window`` 的 prompt 调用
    ``complete``，然后断言 ``stop_reason == "error"``、``error_message`` 匹配
    provider 的溢出文案，且
    ``is_context_overflow(message, model.context_window) is True``。
    """
    raise AssertionError("live provider E2E must not run")
