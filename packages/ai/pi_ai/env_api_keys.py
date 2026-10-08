"""从环境变量发现 API key。

移植自 ``packages/ai/src/env-api-keys.ts``。本项目只保留九家内置 provider，
因此环境变量映射也相应裁剪。

TypeScript 原版通过动态 import 加载 ``node:fs``、``node:os`` 和 ``node:path``，
以免 bundler 把 Node 内置模块拉进浏览器构建；CPython 始终有标准库，因此移植版
直接 import :mod:`os`/:mod:`pathlib`。
"""

from __future__ import annotations

from .types import ProviderEnv
from .utils.provider_env import get_provider_env_value

__all__ = [
    "ANTHROPIC_API_KEY_ENV",
    "ANTHROPIC_AUTH_TOKEN_ENV",
    "ANTHROPIC_FEDERATION_RULE_ID_ENV",
    "ANTHROPIC_IDENTITY_TOKEN_FILE_ENV",
    "ANTHROPIC_ORGANIZATION_ID_ENV",
    "ANTHROPIC_SERVICE_ACCOUNT_ID_ENV",
    "ANTHROPIC_WORKSPACE_ID_ENV",
    "find_env_keys",
    "get_env_api_key",
]

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_FEDERATION_RULE_ID_ENV = "ANTHROPIC_FEDERATION_RULE_ID"
ANTHROPIC_ORGANIZATION_ID_ENV = "ANTHROPIC_ORGANIZATION_ID"
ANTHROPIC_SERVICE_ACCOUNT_ID_ENV = "ANTHROPIC_SERVICE_ACCOUNT_ID"
ANTHROPIC_IDENTITY_TOKEN_FILE_ENV = "ANTHROPIC_IDENTITY_TOKEN_FILE"
ANTHROPIC_WORKSPACE_ID_ENV = "ANTHROPIC_WORKSPACE_ID"

#: 内置 provider 到其 API key 环境变量的映射。
_API_KEY_ENV_VARS: dict[str, str] = {
    "anthropic": ANTHROPIC_API_KEY_ENV,
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "moonshotai-cn": "MOONSHOT_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
}


def get_api_key_env_vars(provider: str) -> tuple[str, ...] | None:
    """返回 :func:`find_env_keys` 为 ``provider`` 扫描的环境变量列表。

    Args:
        provider: provider id。

    Returns:
        按优先级排列的环境变量名；该 provider 不使用环境 API key 时为
        ``None``。
    """
    # ANTHROPIC_AUTH_TOKEN 参与环境发现/状态检查，但 get_env_api_key() 会跳过它，
    # 因为请求必须将其作为 `Authorization: Bearer` 传递。
    if provider == "anthropic":
        return (ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV)

    env_var = _API_KEY_ENV_VARS.get(provider)
    return (env_var,) if env_var else None


def find_env_keys(provider: str, env: ProviderEnv | None = None) -> list[str] | None:
    """查找已配置、可为 provider 提供 API key 的环境变量。

    Args:
        provider: provider id。
        env: provider 环境变量映射；``None`` 时读取进程环境。

    Returns:
        已配置的 API key 变量名列表；无匹配时为 ``None``。
    """
    env_vars = get_api_key_env_vars(provider)
    if not env_vars:
        return None

    found = [env_var for env_var in env_vars if get_provider_env_value(env_var, env)]
    return found if found else None


def get_env_api_key(provider: str, env: ProviderEnv | None = None) -> str | None:
    """从已知环境变量中获取 ``provider`` 的 API key。

    Args:
        provider: provider id。
        env: provider 环境变量映射；``None`` 时读取进程环境。

    Returns:
        环境中的 API key；未配置时为 ``None``。
    """
    env_keys = find_env_keys(provider, env)
    if env_keys:
        api_key_env = (
            next((key for key in env_keys if key != ANTHROPIC_AUTH_TOKEN_ENV), None)
            if provider == "anthropic"
            else env_keys[0]
        )
        if api_key_env:
            return get_provider_env_value(api_key_env, env)
    return None
