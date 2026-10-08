"""按 provider 隔离的环境变量读取."""

from __future__ import annotations

import os

from ..types import ProviderEnv

__all__ = ["get_provider_env_value"]


def get_provider_env_value(name: str, env: ProviderEnv | None = None) -> str | None:
    """解析环境变量：先查 provider 作用域的覆盖值，再回落到 ``os.environ``.

    对应 ``packages/ai/src/utils/provider-env.ts``。原实现中 Bun 沙箱会回退读取
    ``/proc/self/environ``，此处无需对应逻辑：CPython 总能看到真实的进程环境。

    Args:
        name: 环境变量名。
        env: 可选的 provider 作用域环境表。

    Returns:
        变量值；不存在或为空时返回 ``None``。
    """
    if env is not None:
        scoped = env.get(name)
        if scoped:
            return scoped
    value = os.environ.get(name)
    return value if value else None
