"""移植 ``packages/ai/test/cross-provider-handoff.test.ts``。

上游文件为每个 provider/model 组合构建一份录制 transcript（模块加载时解析真实
凭据），写入磁盘后对真实目标 provider 回放每一次 handoff。所有用例都以
``hasAnyApiKey()`` / 各组合的凭据为门控，因此这里把整个矩阵保留为带说明的跳过测试。
"""

from __future__ import annotations

import pytest

NOTE = (
    "live provider E2E: upstream resolves credentials for its provider/model fixtures "
    "(gated on hasAnyApiKey() and per-pair API keys) and replays every handoff against a "
    "live target; the Python suite is offline"
)


CASES = [
    pytest.param(
        "should have at least 2 fixtures to test handoffs",
        id="fixture-count",
        marks=pytest.mark.skip(reason=NOTE),
    ),
    pytest.param(
        "should handle cross-provider handoffs for each target",
        id="handoff-for-each-target",
        marks=pytest.mark.skip(reason=NOTE),
    ),
]


@pytest.mark.parametrize("test_name", CASES)
def test_cross_provider_handoff_live_case(test_name: str) -> None:
    """上游对真实 provider 回放录制的跨 provider transcript。"""
    raise AssertionError("live provider E2E must not run")
