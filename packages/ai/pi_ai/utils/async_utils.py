"""针对可能同步或异步的回调的小型 async 辅助函数.

TypeScript 的选项类型接受 ``T | Promise<T>``（例如 ``onPayload``、
``onResponse``、``finishTurn``）。Python 调用方可能返回普通值或可等待对象，
因此每个调用点都经由 :func:`maybe_await` 处理。
"""

from __future__ import annotations

import inspect
from typing import Any, TypeVar

__all__ = ["maybe_await"]

T = TypeVar("T")


async def maybe_await(value: Any) -> Any:
    """如果值是可等待对象则 await 它，否则直接返回.

    Args:
        value: 任意值，可能是协程、Future 或普通值。

    Returns:
        如果 value 是可等待对象，返回 await 的结果；否则原样返回。

    Examples:
        >>> await maybe_await(42)
        42
        >>> await maybe_await(asyncio.sleep(0))
        None
    """
    if inspect.isawaitable(value):
        return await value
    return value
