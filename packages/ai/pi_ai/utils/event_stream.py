"""通用的推送式异步事件流.

移植自 ``packages/ai/src/utils/event-stream.ts``。生产者可在任意位置
:meth:`~EventStream.push` 事件；消费者用 ``async for`` 迭代，或 await
:meth:`~EventStream.result` 取得终止值。
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Any, Generic, TypeVar

__all__ = [
    "AssistantMessageEventStream",
    "EventStream",
    "create_assistant_message_event_stream",
]

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    """可异步迭代的队列，其终止事件同时决定结果 future.

    ``is_complete`` 决定哪个事件终止流；``extract_result`` 从该事件导出终止值。
    :meth:`end` 可以不经由事件终止流，并可直接给出结果。

    Attributes:
        _is_complete: 判断事件是否为终止事件。
        _extract_result: 从终止事件导出结果值。
        _queue: 已推入但尚无消费者的事件。
        _waiters: 正在等待下一个事件的消费者 future。
        _result_waiters: 正在等待终止值的 future。
        _done: 流是否已终止。
        _result: 已确定的终止值。
        _result_ready: 终止值是否已就绪。
    """

    __slots__ = (
        "_done",
        "_extract_result",
        "_is_complete",
        "_queue",
        "_result",
        "_result_ready",
        "_result_waiters",
        "_waiters",
    )

    def __init__(self, is_complete: Callable[[T], bool], extract_result: Callable[[T], R]) -> None:
        """初始化事件流.

        Args:
            is_complete: 判断事件是否为终止事件的回调函数。
            extract_result: 从终止事件中提取最终结果的回调函数。

        Note:
            内部使用双端队列存储事件和等待者，确保生产者和消费者的线程安全。
        """
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._queue: deque[T] = deque()
        self._waiters: deque[asyncio.Future[tuple[bool, T | None]]] = deque()
        self._result_waiters: deque[asyncio.Future[R]] = deque()
        self._done = False
        self._result: R | None = None
        self._result_ready = False

    # -- 生产者一侧 ----------------------------------------------------------

    def push(self, event: T) -> None:
        """把 ``event`` 交给等待中的消费者，否则入队.

        Args:
            event: 待推送的事件；若它是终止事件，同时据此结算结果并关闭流。
        """
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            self._settle(self._extract_result(event))
        self._deliver(event)

    def end(self, result: R | None = None) -> None:
        """终止流，可同时确定最终结果.

        Args:
            result: 直接给出的最终结果；为 ``None`` 时不结算结果（保留 ``None``）。
        """
        self._done = True
        if result is not None:
            self._settle(result)
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result((False, None))

    # -- 消费者一侧 ----------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        """返回本事件流的异步迭代器."""
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        """依次产出队列中的事件，流终止后结束迭代。

        Yields:
            下一个可用事件。
        """
        while True:
            has_value, value = await self._next()
            if not has_value:
                return
            yield value  # type: ignore[misc]  # 该分支已保证 value 为 T

    def result(self) -> Any:
        """等待终止值.

        返回可等待对象而非协程，调用方可先立即取得（``stream.result()``）再稍后
        await，对齐 TypeScript 中 promise 的行为。

        Returns:
            结果已就绪时返回立即可 await 的协程，否则返回注册后的 future。
        """
        if self._result_ready:
            return _ready(self._result)
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[R] = loop.create_future()
        self._result_waiters.append(waiter)
        return waiter

    # -- 内部实现 -------------------------------------------------------------

    def _deliver(self, event: T) -> None:
        """把事件直接交给等待者，没有等待者时入队。

        Args:
            event: 待送达的事件。
        """
        if self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result((True, event))
                return
        self._queue.append(event)

    def _settle(self, value: R | None) -> None:
        """记录终止值并唤醒所有等待结果的 future。

        Args:
            value: 流的结果值。
        """
        self._result = value
        self._result_ready = True
        while self._result_waiters:
            waiter = self._result_waiters.popleft()
            if not waiter.done():
                waiter.set_result(value)  # type: ignore[arg-type]  # 与 future 的 R 参数一致

    async def _next(self) -> tuple[bool, T | None]:
        """获取下一个事件，原子性地检查队列和注册等待者.

        Returns:
            元组 (has_value, value)，has_value 为 True 表示有新事件，value 为事件对象。
            如果流已结束，返回 (False, None)。

        Note:
            在单线程 asyncio 中，由于没有真正的并发，这个操作是原子的。
            但在多线程场景下需要加锁保护。
        """
        if self._queue:
            return True, self._queue.popleft()
        if self._done:
            return False, None
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[tuple[bool, T | None]] = loop.create_future()
        self._waiters.append(waiter)
        return await waiter


async def _ready(value: Any) -> Any:
    """把已确定的结果包装成可 await 的对象。

    Args:
        value: 已就绪的结果值。

    Returns:
        立即完成的协程结果，即 ``value`` 本身。
    """
    return value


class AssistantMessageEventStream(EventStream[Any, Any]):
    """每个 provider ``stream`` 实现返回的流类型."""

    def __init__(self) -> None:
        """以 ``done``/``error`` 为终止事件、以其消息为结果初始化流."""
        super().__init__(_is_terminal_event, _extract_final_message)


def _is_terminal_event(event: Any) -> bool:
    """判断事件是否为 ``done`` 或 ``error`` 终止事件。

    Args:
        event: 待判断的事件。

    Returns:
        事件类型是否为 ``"done"`` 或 ``"error"``。
    """
    return getattr(event, "type", None) in ("done", "error")


def _extract_final_message(event: Any) -> Any:
    """从终止事件中取出最终消息或错误。

    Args:
        event: ``done`` 或 ``error`` 类型的事件。

    Returns:
        ``done`` 事件的 ``message``，或 ``error`` 事件的 ``error``。

    Raises:
        ValueError: 当事件类型不是终止事件时。
    """
    if event.type == "done":
        return event.message
    if event.type == "error":
        return event.error
    raise ValueError(f"Unexpected event type for final result: {event.type!r}")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """对齐 ``createAssistantMessageEventStream`` 的工厂函数，供扩展使用."""
    return AssistantMessageEventStream()
