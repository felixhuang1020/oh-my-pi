"""移植版 AbortSignal / AbortController 原语的测试。"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from pi_ai.utils.abort import (
    AbortController,
    AbortError,
    AbortSignal,
    combine_abort_signals,
    operation_signal,
    race_with_abort_signal,
)


async def test_throw_if_aborted_raises_default_abort_error():
    controller = AbortController()
    controller.abort()
    with pytest.raises(AbortError):
        controller.signal.throw_if_aborted()


async def test_throw_if_aborted_reraises_explicit_reason():
    controller = AbortController()
    controller.abort(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        controller.signal.throw_if_aborted()


async def test_add_listener_after_abort_does_not_fire():
    controller = AbortController()
    controller.abort()
    calls: list[str] = []
    controller.signal.add_listener(lambda: calls.append("fired"))
    assert calls == []


async def test_listeners_fire_once_and_are_cleared():
    controller = AbortController()
    calls: list[int] = []
    controller.signal.add_listener(lambda: calls.append(1))
    controller.signal.add_listener(lambda: calls.append(2))
    controller.abort()
    controller.abort()
    assert calls == [1, 2]


async def test_a_faulty_listener_does_not_block_the_others():
    controller = AbortController()
    calls: list[str] = []

    def bad() -> None:
        raise RuntimeError("listener failed")

    controller.signal.add_listener(bad)
    controller.signal.add_listener(lambda: calls.append("ok"))
    controller.abort()
    assert calls == ["ok"]


async def test_wait_resolves_on_abort():
    controller = AbortController()
    waiter = asyncio.ensure_future(controller.signal.wait())
    await asyncio.sleep(0)
    controller.abort()
    await asyncio.wait_for(waiter, timeout=1)


async def test_wait_returns_immediately_when_already_aborted():
    controller = AbortController()
    controller.abort()
    await asyncio.wait_for(controller.signal.wait(), timeout=1)


async def test_race_returns_the_operation_result():
    signal = AbortSignal()

    async def work() -> str:
        return "done"

    assert await race_with_abort_signal(work(), signal) == "done"


async def test_race_does_not_start_work_when_already_aborted():
    # TypeScript 版此处已持有运行中的 promise；Python 可以做得更好——直接不启动任务，
    # 使被中止的请求无法在后台触发真实 I/O，被放弃的协程会被直接关闭。
    controller = AbortController()
    controller.abort()
    observed: list[str] = []

    async def work() -> str:
        observed.append("ran")
        return "done"

    coroutine = work()
    with pytest.raises(AbortError):
        await race_with_abort_signal(coroutine, controller.signal)
    await asyncio.sleep(0)
    assert observed == []
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


async def test_race_abandons_a_slow_operation_and_swallows_its_late_exception():
    controller = AbortController()
    release = asyncio.Event()

    async def failing() -> None:
        await release.wait()
        raise ValueError("late failure")

    task = asyncio.ensure_future(race_with_abort_signal(failing(), controller.signal))
    await asyncio.sleep(0)
    controller.abort()
    with pytest.raises(AbortError):
        await task

    # 被放弃的操作仍会跑完；其失败必须被显式取走，避免冒泡成
    # "exception was never retrieved" 警告。
    release.set()
    await asyncio.sleep(0.01)


async def test_operation_error_wins_when_it_settles_before_the_abort():
    controller = AbortController()

    async def failing() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await race_with_abort_signal(failing(), controller.signal)


async def test_race_propagates_operation_failure():
    signal = AbortSignal()

    async def failing() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await race_with_abort_signal(failing(), signal)


async def test_signal_any_aborts_with_the_first_reason():
    first, second = AbortController(), AbortController()
    combined = AbortSignal.any([first.signal, second.signal])
    second.abort("second")
    assert combined.aborted
    assert combined.reason == "second"


async def test_signal_any_short_circuits_when_one_is_already_aborted():
    first, second = AbortController(), AbortController()
    first.abort("early")
    combined = AbortSignal.any([first.signal, second.signal])
    assert combined.reason == "early"


async def test_signal_timeout_aborts_with_timeout_error():
    signal = AbortSignal.timeout(10)
    await asyncio.wait_for(signal.wait(), timeout=1)
    assert isinstance(signal.reason, TimeoutError)


async def test_combine_abort_signals_cleanup_detaches_listeners():
    first, second = AbortController(), AbortController()
    combined = combine_abort_signals([first.signal, second.signal])
    assert combined.signal is not None
    combined.cleanup()
    first.abort("later")
    assert not combined.signal.aborted


async def test_combine_abort_signals_returns_the_single_signal_unchanged():
    controller = AbortController()
    combined = combine_abort_signals([None, controller.signal])
    assert combined.signal is controller.signal
    combined.cleanup()


async def test_combine_abort_signals_with_no_signals_yields_no_signal():
    combined = combine_abort_signals([None, None])
    assert combined.signal is None
    combined.cleanup()


def test_operation_signal_returns_a_fresh_never_aborted_signal():
    assert operation_signal(None).aborted is False
    existing = AbortSignal()
    assert operation_signal(existing) is existing
