"""Anthropic provider 工厂。

移植自 ``packages/ai/src/providers/anthropic.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.types import (
    ApiKeyCredential,
    AuthContext,
    AuthResult,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from ..env_api_keys import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_FEDERATION_RULE_ID_ENV,
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV,
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV,
    ANTHROPIC_WORKSPACE_ID_ENV,
)
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from ..types import ProviderEnv
from ..utils.abort import AbortSignal
from .catalog import chat_model_catalog

__all__ = ["anthropic_provider"]


class _AnthropicApiKeyAuth:
    """Anthropic api-key auth，并带 SDK 的工作负载身份联合（workload identity federation）回退。"""

    name = "Anthropic API key"

    async def login(self, interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        """通过交互式提示收集 Anthropic API key。

        Args:
            interaction: provider 认证交互句柄，用于提示用户输入并携带
                取消信号。

        Returns:
            含用户输入 key 的 api-key 凭证。

        Raises:
            AbortError: 交互信号已被取消时（由 ``throw_if_aborted`` 抛出）。
        """
        interaction.signal.throw_if_aborted()
        key = await interaction.prompt(_secret("Enter Anthropic API key"))
        interaction.signal.throw_if_aborted()
        return ApiKeyCredential(type="api_key", key=key)

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None = None,
        signal: AbortSignal,
    ) -> AuthResult | None:
        """按优先级解析可用的 Anthropic 认证信息。

        依次尝试已存储凭证、``ANTHROPIC_AUTH_TOKEN``、``ANTHROPIC_API_KEY``，
        最后回退到 Anthropic SDK 的工作负载身份联合
        （workload identity federation）。

        Args:
            ctx: 认证上下文，用于读取环境变量。
            credential: 已存储的 api-key 凭证；为 ``None`` 时只走环境变量。
            signal: 取消信号，每步之间检查一次。

        Returns:
            解析出的认证结果；没有任何可用认证时返回 ``None``。

        Raises:
            AbortError: 解析过程中请求被取消时（由 ``throw_if_aborted`` 抛出）。
        """
        signal.throw_if_aborted()
        if credential is not None and credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=credential.key),
                env=credential.env,
                source="stored credential",
            )

        auth_token = await ctx.env(ANTHROPIC_AUTH_TOKEN_ENV)
        signal.throw_if_aborted()
        if auth_token:
            return AuthResult(
                auth=ModelAuth(headers={"Authorization": f"Bearer {auth_token}"}),
                source=ANTHROPIC_AUTH_TOKEN_ENV,
            )

        api_key = await ctx.env(ANTHROPIC_API_KEY_ENV)
        signal.throw_if_aborted()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=ANTHROPIC_API_KEY_ENV)

        # 工作负载身份联合：Anthropic SDK 用身份令牌换取短期访问令牌并自行刷新。
        # 放在最后，让 API key 与 ANTHROPIC_AUTH_TOKEN 保持优先，与 SDK 行为一致。
        # 这些 id 属于 provider 配置而非认证，因此随 `env` 一起传递。
        federation: ProviderEnv = {}
        for env_var in (
            ANTHROPIC_FEDERATION_RULE_ID_ENV,
            ANTHROPIC_ORGANIZATION_ID_ENV,
            ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
        ):
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if not value:
                return None
            federation[env_var] = value
        for env_var in (ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV):
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if value:
                federation[env_var] = value
        return AuthResult(auth=ModelAuth(), env=federation, source="workload identity federation")


def anthropic_api_key_auth() -> _AnthropicApiKeyAuth:
    """构建 Anthropic api-key auth 对象。"""
    return _AnthropicApiKeyAuth()


def anthropic_provider() -> ProviderImpl:
    """构建 Anthropic provider。"""
    return create_provider(
        CreateProviderOptions(
            id="anthropic",
            name="Anthropic",
            base_url="https://api.anthropic.com",
            auth=ProviderAuth(api_key=anthropic_api_key_auth()),
            models=list(chat_model_catalog("anthropic").values()),
            api=lazy_api(lazy_load("pi_ai.api.anthropic_messages")),
        )
    )


def _secret(message: str):
    """构造一个 secret 类型的认证提示。

    Args:
        message: 展示给用户的提示文案。

    Returns:
        携带 ``message`` 的 :class:`AuthPromptSecret`。
    """
    from ..auth.types import AuthPromptSecret

    return AuthPromptSecret(type="secret", message=message)
