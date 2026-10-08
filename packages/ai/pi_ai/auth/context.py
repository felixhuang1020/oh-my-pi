"""基于进程环境变量与文件系统的默认认证上下文。"""

from __future__ import annotations

import os
from pathlib import Path

from .types import AuthContext

__all__ = ["DefaultProviderAuthContext", "default_provider_auth_context"]


class DefaultProviderAuthContext:
    """基于 ``os.environ`` 与 :mod:`pathlib` 的认证上下文。

    空白值解析为 ``None``，避免已导出但为空的环境变量被误当作配置；
    存在性检查前会先展开 ``~``。
    """

    async def env(self, name: str) -> str | None:
        """返回环境变量值；未设置或为空白时返回 ``None``。

        Args:
            name: 环境变量名。

        Returns:
            去掉首尾判断后的原始环境变量值；未设置或为空白时为 ``None``。
        """
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value
        return None

    async def file_exists(self, path: str) -> bool:
        """判断 ``path`` 是否存在（先展开开头的 ``~``）。

        Args:
            path: 待检查的文件路径。

        Returns:
            路径存在时为 ``True``；路径非法时为 ``False``。
        """
        try:
            resolved = Path(path).expanduser()
        except (RuntimeError, ValueError):
            return False
        return resolved.exists()


def default_provider_auth_context() -> AuthContext:
    """构建默认的 :class:`~pi_ai.auth.types.AuthContext`。"""
    return DefaultProviderAuthContext()
