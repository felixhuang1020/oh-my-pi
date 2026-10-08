"""默认的内存 credential 存储。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from ..utils.abort import operation_signal, race_with_abort_signal
from .types import AuthOperationOptions, Credential, CredentialInfo

__all__ = ["InMemoryCredentialStore"]

T = TypeVar("T")


class InMemoryCredentialStore:
    """以 provider id 为键的 :class:`~pi_ai.auth.types.CredentialStore` 实现。

    每个 provider 的写操作通过任务链串行化，并发的读-改-写周期不会交错执行。
    """

    def __init__(self) -> None:
        """初始化空的存储与按 provider 划分的任务链表。"""
        self._credentials: dict[str, Credential] = {}
        self._chains: dict[str, asyncio.Future[Any]] = {}

    async def _enqueue(
        self,
        provider_id: str,
        task: Callable[[], Awaitable[T]],
        options: AuthOperationOptions | None = None,
    ) -> T:
        """把写任务追加到指定 provider 的任务链尾并等待其完成。

        Args:
            provider_id: 目标 provider 标识。
            task: 待执行的异步任务。
            options: 可选的取消控制。

        Returns:
            任务自身的返回值。

        Raises:
            Exception: 取消信号触发时抛出取消异常。
        """
        signal = operation_signal(options.signal if options else None)
        previous = self._chains.get(provider_id)

        async def run() -> T:
            """等待前序任务结束后执行当前任务。"""
            if previous is not None:
                try:
                    await previous
                except Exception:  # noqa: BLE001 - 前序任务失败不应阻塞整条链
                    pass
            signal.throw_if_aborted()
            return await task()

        queued: asyncio.Task[T] = asyncio.ensure_future(run())
        tail: asyncio.Task[None] = asyncio.ensure_future(_settle(queued))
        self._chains[provider_id] = tail
        tail.add_done_callback(lambda finished, pid=provider_id: self._release(pid, finished))

        return await race_with_abort_signal(queued, signal)

    def _release(self, provider_id: str, finished: asyncio.Future[Any]) -> None:
        """任务链完成后清理对应条目。

        Args:
            provider_id: 目标 provider 标识。
            finished: 已完成的链尾任务。
        """
        if self._chains.get(provider_id) is finished:
            self._chains.pop(provider_id, None)

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        """读取已存储的 credential（可能已过期）。

        Args:
            provider_id: 目标 provider 标识。
            options: 可选的取消控制。

        Returns:
            已存储的 credential；不存在时为 ``None``。
        """
        if options is not None and options.signal is not None:
            options.signal.throw_if_aborted()
        return self._credentials.get(provider_id)

    async def list(self, options: AuthOperationOptions | None = None) -> Sequence[CredentialInfo]:
        """列出已存储的 credential 元数据，不暴露任何 secret。

        Args:
            options: 可选的取消控制。

        Returns:
            credential 元数据序列。
        """
        if options is not None and options.signal is not None:
            options.signal.throw_if_aborted()
        return [CredentialInfo(provider_id=provider_id, type=credential.type) for provider_id, credential in self._credentials.items()]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        """串行化的读-改-写；唯一的写入路径。

        Args:
            provider_id: 目标 provider 标识。
            fn: 接收当前 credential（可能为 ``None``）并返回新值的回调。
            options: 可选的取消控制。

        Returns:
            修改后的 credential；回调返回 ``None`` 时返回当前值。
        """

        async def task() -> Credential | None:
            """在任务链内执行一次读-改-写。"""
            current = self._credentials.get(provider_id)
            next_credential = await fn(current)
            if options is not None and options.signal is not None:
                options.signal.throw_if_aborted()
            if next_credential is not None:
                self._credentials[provider_id] = next_credential
            return next_credential if next_credential is not None else current

        return await self._enqueue(provider_id, task, options)

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        """删除 credential（登出），与 :meth:`modify` 串行执行。

        Args:
            provider_id: 目标 provider 标识。
            options: 可选的取消控制。
        """

        async def task() -> None:
            """在任务链内移除指定 provider 的 credential。"""
            self._credentials.pop(provider_id, None)

        await self._enqueue(provider_id, task, options)


async def _settle(future: asyncio.Future[Any]) -> None:
    """队列操作的完成标记；不会向外传播其结果。

    Args:
        future: 需要等待完成的链尾任务。
    """
    try:
        await asyncio.shield(future)
    except (Exception, asyncio.CancelledError):  # noqa: BLE001 - 链条标记仅表示完成
        pass
