"""供 provider 工厂复用的认证实现。

本项目只支持 api-key 认证，因此这里只保留 :class:`EnvApiKeyAuth`。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..utils.abort import AbortSignal
from .types import (
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthResult,
    ModelAuth,
    ProviderAuthInteraction,
)

__all__ = ["EnvApiKeyAuth", "env_api_key_auth"]


class EnvApiKeyAuth:
    """标准 api-key 认证：优先使用已存储的 key，否则取第一个已设置的环境变量。

    解析方式非标准的 provider（provider 环境变量、环境文件、IAM 等）
    会自行实现同形状的 api-key auth 对象。

    Attributes:
        name: 认证方式的人类可读名称。
        env_vars: 按优先级排列的环境变量名。
    """

    def __init__(self, name: str, env_vars: Sequence[str]) -> None:
        """初始化 api-key 认证。

        Args:
            name: 认证方式的人类可读名称。
            env_vars: 按优先级排列的环境变量名。
        """
        self.name = name
        self.env_vars = tuple(env_vars)

    async def login(self, interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        """交互式提示输入 key。

        Args:
            interaction: 登录交互对象。

        Returns:
            新得到的 api-key credential。
        """
        interaction.signal.throw_if_aborted()
        key = await interaction.prompt(_secret_prompt(f"Enter {self.name}"))
        interaction.signal.throw_if_aborted()
        return ApiKeyCredential(type="api_key", key=key)

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None = None,
        signal: AbortSignal,
    ) -> AuthResult | None:
        """先取已存储的 credential，再依次尝试配置的环境变量。

        Args:
            ctx: 认证上下文，用于读取环境变量。
            credential: 已存储的 credential；未存储时为 ``None``。
            signal: 取消信号。

        Returns:
            解析结果；两处都未命中时返回 ``None``。
        """
        signal.throw_if_aborted()
        if credential is not None and credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=credential.key),
                env=credential.env,
                source="stored credential",
            )
        for env_var in self.env_vars:
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None

    async def check(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None = None,
        signal: AbortSignal,
    ) -> AuthCheck | None:
        """无副作用的可用性检查。

        Args:
            ctx: 认证上下文，用于读取环境变量。
            credential: 已存储的 credential；未存储时为 ``None``。
            signal: 取消信号。

        Returns:
            检查结果；未配置 key 时返回 ``None``。
        """
        signal.throw_if_aborted()
        if credential is not None and credential.key:
            return AuthCheck(type="api_key", source="stored credential")
        for env_var in self.env_vars:
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if value:
                return AuthCheck(type="api_key", source=env_var)
        return None


def env_api_key_auth(name: str, env_vars: Sequence[str]) -> EnvApiKeyAuth:
    """构建标准的 :class:`EnvApiKeyAuth`。

    Args:
        name: 认证方式的人类可读名称。
        env_vars: 按优先级排列的环境变量名。

    Returns:
        构造好的 api-key 认证对象。
    """
    return EnvApiKeyAuth(name, env_vars)


def _secret_prompt(message: str) -> Any:
    """构造不回显的机密值输入提示。

    Args:
        message: 展示给用户的提示文字。

    Returns:
        构造好的 :class:`~pi_ai.auth.types.AuthPromptSecret`。
    """
    from .types import AuthPromptSecret

    return AuthPromptSecret(type="secret", message=message)
