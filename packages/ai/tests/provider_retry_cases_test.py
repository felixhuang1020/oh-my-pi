"""移植自 ``packages/ai/test/provider-retry.test.ts``。

演练 :func:`pi_ai.utils.provider_retry.retry_provider_request`：重试次数、provider 通过
``x-should-retry`` 声明的不重试、服务端要求的重试延迟上限，以及中断进行中的重试等待。

Vitest 用 ``vi.useFakeTimers()`` 驱动；本移植的等价注入点是模块私有的可中断 sleep，
因此用确定性的假 sleep 记录计算出的延迟，并显式放行（或中断）。全程不运行真实定时器。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pi_ai.utils import provider_retry
from pi_ai.utils.abort import AbortController, AbortError
from pi_ai.utils.provider_retry import ProviderRetryOptions, retry_provider_request


class _ProviderError(Exception):
    """等价于 ``Object.assign(new Error(`Provider error: ${status}`), { status, headers })``。"""

    def __init__(self, status: int | None, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"Provider error: {status}")
        self.status = status
        self.headers = headers or {}


def provider_error(status: int | None, headers: dict[str, str] | None = None) -> _ProviderError:
    return _ProviderError(status, headers)


class _FakeRequest:
    """记录调用次数的请求可调用对象，按顺序 resolve/raise 预设结果。"""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self) -> Any:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _GatedSleep:
    """假的可中断 sleep：记录延迟并阻塞，直到测试放行。"""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, ms: float, signal: Any) -> None:
        self.delays.append(ms)
        self.entered.set()
        await self.release.wait()


class _AbortableSleep:
    """模拟真实实现的假可中断 sleep：中断时抛出 ``AbortError``。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, ms: float, signal: Any) -> None:
        self.delays.append(ms)
        if signal is None:
            return
        await signal.wait()
        raise AbortError("Request aborted")


class _NoSleep:
    """守护用例：任何情况下都不允许请求重试延迟。"""

    async def __call__(self, ms: float, signal: Any) -> None:
        raise AssertionError(f"unexpected retry sleep of {ms}ms")


async def _until(predicate: Any, attempts: int = 100) -> None:
    """不断让出事件循环直到 ``predicate`` 成立（不做真实等待）。"""
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was never reached")


async def test_retries_retryable_provider_errors(monkeypatch):
    sleep = _GatedSleep()
    monkeypatch.setattr(provider_retry, "_abortable_sleep", sleep)
    request = _FakeRequest([provider_error(429, {"retry-after-ms": "1000"}), "ok"])

    result = asyncio.ensure_future(
        retry_provider_request(request, ProviderRetryOptions(max_retries=1))
    )
    await _until(sleep.entered.is_set)
    assert request.calls == 1
    assert sleep.delays == [1000.0]

    sleep.release.set()
    assert await result == "ok"
    assert request.calls == 2


async def test_does_not_retry_errors_the_provider_marks_as_non_retryable(monkeypatch):
    monkeypatch.setattr(provider_retry, "_abortable_sleep", _NoSleep())
    error = provider_error(429, {"x-should-retry": "false"})
    request = _FakeRequest([error])

    with pytest.raises(_ProviderError) as exc_info:
        await retry_provider_request(request, ProviderRetryOptions(max_retries=2))

    assert exc_info.value is error
    assert request.calls == 1


async def test_rejects_a_provider_requested_retry_delay_above_the_limit(monkeypatch):
    monkeypatch.setattr(provider_retry, "_abortable_sleep", _NoSleep())
    request = _FakeRequest([provider_error(429, {"retry-after": "277403"})])

    with pytest.raises(RuntimeError) as exc_info:
        await retry_provider_request(
            request, ProviderRetryOptions(max_retries=1, max_retry_delay_ms=1000)
        )

    assert "Server requested 277403s retry delay (max: 1s)" in str(exc_info.value)
    assert request.calls == 1


async def test_allows_disabling_the_provider_requested_retry_delay_cap(monkeypatch):
    sleep = _GatedSleep()
    monkeypatch.setattr(provider_retry, "_abortable_sleep", sleep)
    request = _FakeRequest([provider_error(429, {"retry-after": "2"}), "ok"])

    result = asyncio.ensure_future(
        retry_provider_request(
            request, ProviderRetryOptions(max_retries=1, max_retry_delay_ms=0)
        )
    )
    await _until(sleep.entered.is_set)
    assert request.calls == 1
    assert sleep.delays == [2000.0]

    sleep.release.set()
    assert await result == "ok"
    assert request.calls == 2


async def test_aborts_a_provider_requested_retry_delay(monkeypatch):
    sleep = _AbortableSleep()
    monkeypatch.setattr(provider_retry, "_abortable_sleep", sleep)
    controller = AbortController()
    request = _FakeRequest([provider_error(429, {"retry-after": "277403"})])

    result = asyncio.ensure_future(
        retry_provider_request(
            request,
            ProviderRetryOptions(max_retries=2, max_retry_delay_ms=0, signal=controller.signal),
        )
    )
    await _until(lambda: bool(sleep.delays))
    assert request.calls == 1
    assert sleep.delays == [277403000.0]

    controller.abort()

    with pytest.raises(AbortError):
        await result
    assert request.calls == 1
