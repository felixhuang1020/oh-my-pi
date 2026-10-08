"""进程级的会话资源清理注册表。

移植自 ``packages/ai/src/session-resources.ts``。provider（及 agent 运行时）只需
注册一次清理回调；:func:`cleanup_session_resources` 会针对某个 session id 运行
全部回调，并将所有失败汇总后重新抛出，确保单个清理失败不会掩盖其他结果。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

__all__ = [
    "SessionResourceCleanup",
    "cleanup_session_resources",
    "register_session_resource_cleanup",
]

#: 清理回调，接收正在拆除的 session id（可能为 None）。
SessionResourceCleanup: TypeAlias = Callable[[str | None], None]

_session_resource_cleanups: set[SessionResourceCleanup] = set()


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    """注册 ``cleanup``，并返回用于注销它的函数。

    Args:
        cleanup: 接收 session id 的清理回调。

    Returns:
        调用即可移除该回调的注销函数；对同一回调重复调用是安全的。
    """
    _session_resource_cleanups.add(cleanup)

    def unregister() -> None:
        """从注册表中移除本次注册的清理回调。"""
        _session_resource_cleanups.discard(cleanup)

    return unregister


def cleanup_session_resources(session_id: str | None = None) -> None:
    """运行所有已注册的清理回调；若有失败则抛出异常组。

    Args:
        session_id: 正在拆除的 session id；``None`` 表示进程级清理。

    Raises:
        ExceptionGroup: 至少一个清理回调抛出异常时，汇总全部失败后抛出。
    """
    errors: list[Exception] = []
    for cleanup in tuple(_session_resource_cleanups):
        try:
            cleanup(session_id)
        except Exception as error:  # noqa: BLE001 - 即使出错也要保证所有清理都执行
            errors.append(error)
    if errors:
        raise ExceptionGroup("Failed to cleanup session resources", errors)
