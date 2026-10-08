"""移植 ``packages/ai/test/google-shared-retry.test.ts``。

上游测试驱动 vitest 假定时器并调用 ``advanceTimersByTimeAsync(500)``。
移植版把重试辅助函数的 sleep 接缝（``pi_ai.utils.provider_retry._abortable_sleep``）
替换为记录用的 no-op，因此没有测试真正等待；记录下来的延迟会对照上游定时器
推进所覆盖的指数退避窗口断言。
"""

from __future__ import annotations

from typing import Any

import pytest

from pi_ai.api.google_shared import retry_google_request
from pi_ai.types import StreamOptions


def google_api_error(status: int) -> Exception:
    """形似 ``@google/genai`` 的 ApiError：有 ``status``，但没有 ``headers``。"""
    error = RuntimeError(f"got status: {status}")
    error.status = status  # type: ignore[attr-defined]
    return error


class SequencedRequest:
    """``vi.fn()`` 式替身：每次调用消耗下一个结果，异常则抛出。"""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self) -> Any:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """记录重试辅助函数的退避延迟，代替真正睡眠。"""
    sleeps: list[float] = []

    async def fake_sleep(ms: float, signal: Any = None) -> None:
        sleeps.append(ms)

    monkeypatch.setattr("pi_ai.utils.provider_retry._abortable_sleep", fake_sleep)
    return sleeps


async def test_retries_a_headers_less_sdk_error_with_a_retryable_status(
    recorded_sleeps: list[float],
) -> None:
    request = SequencedRequest([google_api_error(429), "ok"])

    result = await retry_google_request(request, StreamOptions(max_retries=1))

    assert result == "ok"
    assert request.calls == 2
    # 上游假定时器推进的 500ms 窗口内发生一次退避 sleep。
    assert len(recorded_sleeps) == 1
    assert 375 <= recorded_sleeps[0] <= 500


async def test_does_not_retry_when_max_retries_is_unset(recorded_sleeps: list[float]) -> None:
    error = google_api_error(429)
    request = SequencedRequest([error])

    with pytest.raises(RuntimeError) as raised:
        await retry_google_request(request)

    assert raised.value is error
    assert request.calls == 1
    assert recorded_sleeps == []


async def test_does_not_retry_a_non_retryable_status(recorded_sleeps: list[float]) -> None:
    error = google_api_error(400)
    request = SequencedRequest([error])

    with pytest.raises(RuntimeError) as raised:
        await retry_google_request(request, StreamOptions(max_retries=2))

    assert raised.value is error
    assert request.calls == 1
    assert recorded_sleeps == []
