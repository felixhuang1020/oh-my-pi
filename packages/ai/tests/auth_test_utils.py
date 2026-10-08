"""测试助手：从环境或 ``~/.pi/agent/auth.json`` 解析 API key。

本项目只支持 api-key 认证，因此这里只读取 ``api_key`` 类型的 credential。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pi_ai.env_api_keys import get_api_key_env_vars
from pi_ai.utils.provider_env import get_provider_env_value

__all__ = ["AUTH_PATH", "load_auth_storage", "resolve_api_key"]

AUTH_PATH = Path.home() / ".pi" / "agent" / "auth.json"


def load_auth_storage() -> dict[str, Any]:
    """读取 ``auth.json``；文件缺失或损坏时返回空映射。"""
    if not AUTH_PATH.exists():
        return {}
    try:
        raw = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


async def resolve_api_key(provider: str) -> str | None:
    """解析 ``provider`` 的 API key：先查已存储 credential，再查环境变量。"""
    entry = load_auth_storage().get(provider)
    if isinstance(entry, dict) and entry.get("type") == "api_key":
        key = entry.get("key")
        if isinstance(key, str) and key:
            return key
    for env_var in get_api_key_env_vars(provider) or ():
        value = get_provider_env_value(env_var, None)
        if value:
            return value
    return None
