"""对齐 Web ``AbortController``/``AbortSignal`` API 的取消原语.

TypeScript 原版把 ``AbortSignal`` 贯穿每个 provider 请求；移植版保留这一形态，
而不改写为围绕 ``asyncio.Task.cancel`` 的控制流，使每个移植模块仍可与源文件
逐行对照。

    controller = AbortController()
    signal = controller.signal
    signal.throw_if_aborted()
    await race_with_abort_signal(client.post(...), signal)
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

__all__ = [
    "AbortController",
    "AbortError",
    "AbortSignal",
    "CombinedAbortSignal",
    "combine_abort_signals",
    "operation_signal",
    "race_with_abort_signal",
    "sleep_cancellable",
]

T = TypeVar("T")


class AbortError(Exception):
    """操作被取消时抛出.

    对齐 DOM 的 ``AbortError``；调用方以显式值中止时，``Exception.args[0]``
    携带该中止原因。

    Attributes:
        reason: 中止原因；未提供时为 ``None``。
    """

    def __init__(self, reason: Any = None) -> None:
        """构造中止异常。

        Args:
            reason: 中止原因；为 ``None`` 或本身是 :class:`AbortError` 时使用默认文案。
        """
        if reason is None or isinstance(reason, AbortError):
            super().__init__("The operation was aborted")
        else:
            super().__init__(str(reason))
        self.reason = reason


class AbortSignal:
    """单向取消标志，支持监听器和异步等待.

    实例由 :class:`AbortController`、:meth:`AbortSignal.any` 或
    :meth:`AbortSignal.timeout` 创建；需要中止操作的调用方不应直接构造它。

    Attributes:
        aborted: 是否已中止。
        reason: 中止原因；未中止时为 ``None``。
        _aborted: 中止标志的内部存储。
        _reason: 中止原因的内部存储。
        _listeners: 已注册且尚未触发的无参回调。
        _waiters: 正在 :meth:`wait` 上等待的 future 列表。
    """

    __slots__ = ("_aborted", "_listeners", "_reason", "_waiters")

    def __init__(self) -> None:
        """初始化未触发的 AbortSignal."""
        self._aborted = False
        self._reason: Any = None
        self._listeners: list[Callable[[], None]] = []
        self._waiters: list[asyncio.Future[None]] = []

    # -- 状态 ----------------------------------------------------------------

    @property
    def aborted(self) -> bool:
        """操作是否已被取消."""
        return self._aborted

    @property
    def reason(self) -> Any:
        """中止原因；未中止时为 ``None``."""
        return self._reason

    def throw_if_aborted(self) -> None:
        """已中止时抛出中止原因."""
        if self._aborted:
            raise self.exception()

    def exception(self) -> BaseException:
        """``throw_if_aborted`` 对当前原因会抛出的异常."""
        return self._as_exception()

    def _as_exception(self) -> BaseException:
        """把中止原因归一化为可抛出的异常。

        Returns:
            原因本身（若已是 :class:`BaseException`），否则包装后的 :class:`AbortError`。
        """
        reason = self._reason
        if isinstance(reason, BaseException):
            return reason
        return AbortError(reason)

    # -- 监听器 ---------------------------------------------------------------

    def add_listener(self, callback: Callable[[], None]) -> None:
        """注册 ``callback``；与 DOM 一致，已中止的信号不会触发它.

        Args:
            callback: 中止时调用的无参回调。
        """
        if self._aborted:
            return
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        """若存在则注销 ``callback``.

        Args:
            callback: 待注销的回调；不存在时静默忽略。
        """
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    async def wait(self) -> None:
        """等待信号被中止."""
        if self._aborted:
            return
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)

    # -- 工厂方法 --------------------------------------------------------------

    @staticmethod
    def any(signals: Iterable[AbortSignal]) -> AbortSignal:
        """任一输入信号中止时立即中止的组合信号.

        Args:
            signals: 参与组合的信号集合。

        Returns:
            在首个中止信号触发时中止的新信号。
        """
        combined = AbortSignal()
        for signal in signals:
            if signal.aborted:
                combined._abort(signal._reason)
                return combined
        for signal in signals:
            signal.add_listener(lambda signal=signal: combined._abort(signal._reason))
        return combined

    @staticmethod
    def timeout(milliseconds: float) -> AbortSignal:
        """经过 ``milliseconds`` 毫秒后以 :class:`TimeoutError` 中止的信号.

        Args:
            milliseconds: 超时时间（毫秒）；负数按 0 处理。

        Returns:
            到期自动中止的新信号。
        """
        combined = AbortSignal()
        seconds = max(0.0, milliseconds / 1000.0)

        def fire() -> None:
            """计时器到期时中止组合信号."""
            combined._abort(TimeoutError("The operation timed out"))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            timer = threading.Timer(seconds, fire)
            timer.daemon = True
            timer.start()
        else:
            loop.call_later(seconds, fire)
        return combined

    # -- 内部实现 --------------------------------------------------------------

    def _abort(self, reason: Any) -> None:
        """执行一次中止：置位、摘除监听器并唤醒等待者。

        重复调用无副作用。

        Args:
            reason: 记录到 :attr:`reason` 的中止原因。
        """
        if self._aborted:
            return
        self._aborted = True
        self._reason = reason
        listeners, self._listeners = self._listeners, []
        waiters, self._waiters = self._waiters, []
        for callback in listeners:
            try:
                callback()
            except Exception:  # pragma: no cover - 监听器故障不得影响中止流程
                pass
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)


class AbortController:
    """持有 :class:`AbortSignal`，可对其执行一次中止.

    Attributes:
        signal: 本控制器创建并持有的信号。
    """

    __slots__ = ("signal",)

    def __init__(self) -> None:
        """创建一个未中止的信号并绑定到 :attr:`signal`."""
        self.signal = AbortSignal()

    @property
    def aborted(self) -> bool:
        """本控制器是否已中止其信号."""
        return self.signal.aborted

    def abort(self, reason: Any = None) -> None:
        """中止持有的信号；后续调用不再生效.

        Args:
            reason: 传给信号的中止原因。
        """
        self.signal._abort(reason)


def operation_signal(signal: AbortSignal | None) -> AbortSignal:
    """为可选的公开参数返回 ``signal``，缺省时给一个从未中止的新信号.

    Args:
        signal: 调用方传入的可选信号。

    Returns:
        原信号，或一个永不会中止的占位信号。
    """
    return signal if signal is not None else AbortSignal()


async def race_with_abort_signal(operation: Awaitable[T], signal: AbortSignal) -> T:
    """等待 ``operation``，但 ``signal`` 一中止便停止等待.

    被放弃的可等待对象会继续运行，其最终异常始终会被取走，
    因此不会漏出 "exception was never retrieved" 警告。

    Args:
        operation: 待等待的可等待对象。
        signal: 用于取消等待的信号。

    Returns:
        ``operation`` 的结果。

    Raises:
        BaseException: 当 ``signal`` 已中止或先于 ``operation`` 中止时，抛出其中止原因。
        asyncio.CancelledError: 当外层任务被取消时，原样向上传播。
    """
    if signal.aborted:
        _observe(operation)
        raise signal._as_exception()

    task = asyncio.ensure_future(operation)
    waiter = asyncio.ensure_future(signal.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        waiter.cancel()
        raise

    if task in done:
        waiter.cancel()
        return task.result()

    task.add_done_callback(_swallow_result)
    raise signal._as_exception()


async def sleep_cancellable(ms: float, signal: AbortSignal | None = None) -> None:
    """可被中止信号打断的毫秒睡眠。

    Args:
        ms: 睡眠时长（毫秒）；负数按 0 处理。
        signal: 中止信号；为 ``None`` 时退化为普通睡眠。

    Raises:
        AbortError: 信号在睡眠前或睡眠期间被中止时。
        asyncio.CancelledError: 当外层任务被取消时，原样向上传播。
    """
    try:
        await race_with_abort_signal(asyncio.sleep(max(0.0, ms) / 1000.0), operation_signal(signal))
    except asyncio.CancelledError:
        raise
    except BaseException as error:  # noqa: BLE001 - 中止原因统一为 AbortError，便于调用方按类型捕获
        raise AbortError("Request aborted") from error


def _observe(operation: Awaitable[Any]) -> None:
    """处理还没来得及被 await 就已中止的操作.

    未被 await 的协程直接关闭而不再调度：调用方已取消，启动工作只会在后台
    触发无人观察结果的真实 I/O。已在运行的（``Future``/``Task``）则保留并
    取走其异常，避免漏出 "exception was never retrieved" 警告。

    Args:
        operation: 因信号中止而不再等待的可等待对象。
    """
    if inspect.iscoroutine(operation):
        operation.close()
        return
    task = asyncio.ensure_future(operation)
    task.add_done_callback(_swallow_result)


def _swallow_result(task: asyncio.Future[Any]) -> None:
    """取走已完成任务的异常，避免未观察异常告警。

    Args:
        task: 完成的 future；已取消时直接返回。
    """
    if task.cancelled():
        return
    task.exception()


class CombinedAbortSignal:
    """合并后的信号，外加用于摘除其监听器的清理函数.

    Attributes:
        signal: 合并后的信号；没有输入信号时为 ``None``。
    """

    __slots__ = ("signal", "_cleanup")

    def __init__(self, signal: AbortSignal | None, cleanup: Callable[[], None]) -> None:
        """绑定合并信号与清理回调。

        Args:
            signal: 合并后的信号，可为 ``None``。
            cleanup: 摘除已注册监听器的无参回调。
        """
        self.signal = signal
        self._cleanup = cleanup

    def cleanup(self) -> None:
        """摘除 :func:`combine_abort_signals` 注册的所有监听器."""
        self._cleanup()


def combine_abort_signals(signals: Iterable[AbortSignal | None]) -> CombinedAbortSignal:
    """把若干可选信号合并为一个，并提供显式的监听器清理.

    Args:
        signals: 待合并的信号序列，允许包含 ``None``。

    Returns:
        合并结果；输入为空时 ``signal`` 为 ``None``，仅有一个有效信号时直接复用该信号。
    """
    active = [signal for signal in signals if signal is not None]
    if not active:
        return CombinedAbortSignal(None, lambda: None)
    if len(active) == 1:
        return CombinedAbortSignal(active[0], lambda: None)

    controller = AbortController()
    listeners: list[tuple[AbortSignal, Callable[[], None]]] = []

    for signal in active:
        if signal.aborted:
            controller.abort(signal.reason)
            break
        listener = lambda signal=signal: controller.abort(signal.reason)  # noqa: E731 - 就地捕获当前信号
        signal.add_listener(listener)
        listeners.append((signal, listener))

    def cleanup() -> None:
        """摘除为本次合并注册到各输入信号上的全部监听器."""
        for signal, listener in listeners:
            signal.remove_listener(listener)

    return CombinedAbortSignal(controller.signal, cleanup)
