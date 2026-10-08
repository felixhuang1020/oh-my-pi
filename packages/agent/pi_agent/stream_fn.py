"""可配置的默认 stream 函数。

移植自 ``packages/agent/src/stream-fn.ts``。
"""

from __future__ import annotations

from .types import StreamFn

__all__ = ["get_default_stream_fn", "set_default_stream_fn"]

_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    """配置 ``Agent`` 与底层循环在调用方未指定 ``stream_fn`` 时使用的回退实现。

    宿主程序若内置默认模型运行时，可在此注册其 stream 函数，
    而无需让 pi-agent-core 依赖 provider 目录或兼容层。

    Args:
        stream_fn: 要注册为默认值的 stream 函数；传入 ``None`` 表示清除已注册的默认值。
    """
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    """返回已配置的默认 stream 函数。

    Raises:
        RuntimeError: 尚未配置默认 stream 函数时。
    """
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
        )
    return _default_stream_fn
