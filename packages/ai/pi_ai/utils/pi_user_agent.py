"""pi 的 user-agent 字符串."""

from __future__ import annotations

import platform

__all__ = ["get_pi_user_agent"]

_USER_AGENT: str | None = None


def get_pi_user_agent() -> str:
    """返回 ``pi (<platform> <release>; <arch>)`` 格式的 user-agent，进程内缓存.

    Returns:
        进程内缓存的 user-agent 字符串。
    """
    global _USER_AGENT
    if _USER_AGENT is None:
        _USER_AGENT = f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"
    return _USER_AGENT
