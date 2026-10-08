"""懒加载 API 包装：同步返回 stream，初始化在后台进行。

移植自 ``packages/ai/src/api/lazy.ts``。auth 解析和实现加载都是异步的，但每个
provider 的 ``stream`` 都必须立即交回
:class:`~pi_ai.utils.event_stream.AssistantMessageEventStream`；因此初始化失败会以
``error`` 事件终止该 stream，而不是抛出异常。
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..types import (
    AssistantMessage,
    Model,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.events import error_event

__all__ = ["LazyApi", "lazy_api", "lazy_load", "lazy_stream"]


def create_setup_error_message(model: Model, error: Any) -> AssistantMessage:
    """把初始化失败编码为一条终止性的 assistant 消息。

    Args:
        model: 初始化失败所针对的模型。
        error: 捕获到的异常或错误值。

    Returns:
        带 ``error`` stop reason 与错误文本的 assistant 消息。
    """
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="error",
        error_message=str(error) if isinstance(error, BaseException) else str(error),
        timestamp=int(time.time() * 1000),
    )


async def forward_stream(target: AssistantMessageEventStream, source: Any) -> None:
    """把 ``source`` 的所有事件转发到 ``target``，并结算其 result。

    Args:
        target: 对调用方可见的外层 stream。
        source: 由真实实现返回的内层 stream。
    """
    async for event in source:
        target.push(event)
    result = None
    result_method = getattr(source, "result", None)
    if callable(result_method):
        result = await result_method()
    target.end(result)


def lazy_stream(model: Model, setup: Callable[[], Awaitable[Any]]) -> AssistantMessageEventStream:
    """同步返回一个 stream，``setup`` 在其背后执行。

    初始化失败时以 error 事件终止 stream。

    Args:
        model: 请求将要发往的模型，用于构造错误消息。
        setup: 异步初始化函数，返回真实的实现对象。

    Returns:
        立即返回、随后由后台任务填充事件的外层 stream。
    """
    outer = AssistantMessageEventStream()

    async def run() -> None:
        """执行 ``setup`` 并把内层 stream 的事件转发到外层。"""
        try:
            inner = await setup()
        except Exception as error:  # noqa: BLE001 - 会编码进 stream 协议
            message = create_setup_error_message(model, error)
            outer.push(error_event("error", message))
            outer.end(message)
            return
        await forward_stream(outer, inner)

    import asyncio

    task = asyncio.ensure_future(run())
    task.add_done_callback(_swallow)
    return outer


def _swallow(task: Any) -> None:
    """消费后台任务的异常，避免 "exception never retrieved" 警告。

    Args:
        task: 已完成的后台任务。
    """
    if task.cancelled():
        return
    task.exception()


class LazyApi:
    """把动态导入的 API 实现模块包装为 provider streams 对象。

    模块在首次调用时加载，import 缓存保证不会重复加载。加载失败会以 error 事件终止
    返回的 stream。

    Attributes:
        _load: 首次调用时执行的异步 loader。
        _supports_fetch_deferred: 被包装的 API 是否支持延迟响应。
        _supports_cancel_deferred: 被包装的 API 是否支持取消延迟响应。
    """

    def __init__(
        self,
        load: Callable[[], Awaitable[Any]],
        supports_fetch_deferred: bool = False,
        supports_cancel_deferred: bool = False,
    ) -> None:
        """初始化懒加载包装。

        Args:
            load: 返回实现模块的异步 loader。
            supports_fetch_deferred: 是否转发 ``fetch_deferred`` 能力。
            supports_cancel_deferred: 是否转发 ``cancel_deferred`` 能力。
        """
        self._load = load
        self._supports_fetch_deferred = supports_fetch_deferred
        self._supports_cancel_deferred = supports_cancel_deferred

    async def _implementation(self) -> Any:
        """加载并返回被包装的实现模块。"""
        return await self._load()

    def stream(
        self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """以完整的 provider options 进行 stream。

        Args:
            model: 请求将要发往的模型。
            context: 已构建的 transcript。
            options: provider 专有选项。

        Returns:
            立即返回、由后台任务填充事件的 stream。
        """

        async def setup() -> Any:
            """加载实现模块并调用其 ``stream``。"""
            return (await self._implementation()).stream(model, context, options)

        return lazy_stream(model, setup)

    def stream_simple(
        self,
        model: Model,
        context: TranscriptContext,
        options: Any = None,
    ) -> AssistantMessageEventStream:
        """以 provider 无关的 simple options 进行 stream。

        Args:
            model: 请求将要发往的模型。
            context: 已构建的 transcript。
            options: provider 无关的简单选项。

        Returns:
            立即返回、由后台任务填充事件的 stream。
        """

        async def setup() -> Any:
            """加载实现模块并调用其 ``stream_simple``。"""
            return (await self._implementation()).stream_simple(model, context, options)

        return lazy_stream(model, setup)

    @property
    def fetch_deferred(self) -> Any:
        """获取延迟响应；被包装的 API 支持时才有值。"""
        if not self._supports_fetch_deferred:
            return None
        return self._fetch_deferred

    def _fetch_deferred(self, model: Model, handle: Any, options: Any = None) -> AssistantMessageEventStream:
        """转发被包装 API 的 ``fetch_deferred`` 调用。

        Args:
            model: 延迟响应所属的模型。
            handle: 延迟响应的句柄。
            options: 可选的请求级选项。

        Returns:
            承载延迟响应事件、立即返回的 stream。
        """

        async def setup() -> Any:
            """加载实现模块并取回延迟响应。"""
            implementation = await self._implementation()
            if getattr(implementation, "fetch_deferred", None) is None:
                raise RuntimeError("API does not support deferred responses")
            return implementation.fetch_deferred(model, handle, options)

        return lazy_stream(model, setup)

    @property
    def cancel_deferred(self) -> Any:
        """取消延迟响应；被包装的 API 支持时才有值。"""
        if not self._supports_cancel_deferred:
            return None
        return self._cancel_deferred

    async def _cancel_deferred(self, model: Model, handle: Any, options: Any = None) -> None:
        """转发被包装 API 的 ``cancel_deferred`` 调用。

        Args:
            model: 延迟响应所属的模型。
            handle: 延迟响应的句柄。
            options: 可选的请求级选项。

        Raises:
            RuntimeError: 被包装的 API 不支持取消延迟响应时。
        """
        implementation = await self._implementation()
        cancel = getattr(implementation, "cancel_deferred", None)
        if cancel is None:
            raise RuntimeError("API cannot cancel deferred responses")
        await cancel(model, handle, options)


def lazy_load(module_name: str, attribute: str | None = None) -> Callable[[], Awaitable[Any]]:
    """返回一个 loader：首次调用时导入 ``module_name``（可选再取其属性）。

    Args:
        module_name: 要导入的模块全名。
        attribute: 可选属性名；给出时返回该属性而不是模块本身。

    Returns:
        无参的异步 loader。
    """

    async def load() -> Any:
        """导入模块并返回模块或指定属性。"""
        module = importlib.import_module(module_name)
        return module if attribute is None else getattr(module, attribute)

    return load


def lazy_api(load: Callable[[], Awaitable[Any]], **capabilities: bool) -> LazyApi:
    """用一个 loader 构建 :class:`LazyApi`。

    Args:
        load: 返回实现模块的异步 loader。
        **capabilities: 能力开关，支持 ``fetch_deferred`` 与 ``cancel_deferred``。

    Returns:
        包装后的 :class:`LazyApi`。
    """
    return LazyApi(
        load,
        supports_fetch_deferred=bool(capabilities.get("fetch_deferred")),
        supports_cancel_deferred=bool(capabilities.get("cancel_deferred")),
    )
