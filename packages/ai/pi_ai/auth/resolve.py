""":class:`~pi_ai.models.Models` 集合中所有操作共用的认证解析。

移植自 ``packages/ai/src/auth/resolve.ts``，但只保留 api-key 认证：已存储的
credential 优先独占 provider，仅在没有任何存储时才查询环境来源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import ProviderEnv
from ..utils.abort import AbortSignal, operation_signal, race_with_abort_signal
from ..utils.models_error import ModelsError
from .types import (
    ApiKeyCredential,
    AuthContext,
    AuthResult,
    Credential,
    CredentialStore,
)

__all__ = [
    "AuthResolutionOverrides",
    "resolve_provider_auth",
]


@dataclass
class AuthResolutionOverrides:
    """在已存储与环境认证之上叠加的单次调用覆盖项。

    Attributes:
        api_key: 覆盖使用的 API key；``None`` 表示不覆盖。
        env: 覆盖的 provider 作用域环境/配置值。
        signal: 调用方传入的取消信号。
    """

    api_key: str | None = None
    env: ProviderEnv | None = None
    signal: AbortSignal | None = None


async def resolve_provider_auth(
    provider: Any,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None = None,
) -> AuthResult | None:
    """解析 ``provider`` 的请求认证，整体与取消信号竞速。

    Args:
        provider: 目标 provider 对象，需提供 ``id`` 与 ``auth`` 属性。
        credentials: 应用自持的 credential 存储。
        auth_context: 认证上下文，用于读取环境变量与文件。
        overrides: 单次调用的覆盖项。

    Returns:
        解析结果；未配置认证时返回 ``None``。

    Raises:
        ModelsError: 认证解析或 credential 读取失败时。
    """
    signal = operation_signal(overrides.signal if overrides else None)
    return await race_with_abort_signal(
        _resolve_with_signal(provider, credentials, auth_context, overrides, signal),
        signal,
    )


async def _resolve_with_signal(
    provider: Any,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None,
    signal: AbortSignal,
) -> AuthResult | None:
    """在已绑定的取消信号下解析请求认证。

    先尝试调用方覆盖项，其次读取已存储的 credential，最后才查询环境来源。

    Args:
        provider: 目标 provider 对象。
        credentials: 应用自持的 credential 存储。
        auth_context: 认证上下文。
        overrides: 单次调用的覆盖项。
        signal: 已合并的取消信号。

    Returns:
        解析结果；未配置认证时返回 ``None``。

    Raises:
        ModelsError: 底层认证实现失败时。
    """
    signal.throw_if_aborted()
    api_key_auth = provider.auth.api_key
    request_auth_context = (
        _overlay_env_auth_context(auth_context, overrides.env)
        if overrides is not None and overrides.env
        else auth_context
    )

    if overrides is not None and overrides.api_key is not None and api_key_auth is not None:
        return await _resolve_api_key(
            request_auth_context,
            api_key_auth,
            provider.id,
            ApiKeyCredential(type="api_key", key=overrides.api_key, env=overrides.env),
            signal,
        )

    stored = await _read_credential(credentials, provider.id, signal)
    if stored is not None:
        if stored.type == "api_key" and api_key_auth is not None:
            credential = stored
            if overrides is not None and overrides.env:
                credential = ApiKeyCredential(
                    type="api_key",
                    key=stored.key,
                    env={**(stored.env or {}), **overrides.env},
                )
            return await _resolve_api_key(request_auth_context, api_key_auth, provider.id, credential, signal)
        return None

    # 环境来源（环境变量、环境文件）。
    if api_key_auth is not None:
        return await _resolve_api_key(request_auth_context, api_key_auth, provider.id, None, signal)
    return None


def _overlay_env_auth_context(base: AuthContext, env: ProviderEnv) -> AuthContext:
    """在基础认证上下文之上叠加一层环境值覆盖。

    Args:
        base: 基础认证上下文。
        env: 优先返回的环境/配置值映射。

    Returns:
        覆盖后的认证上下文。
    """
    from .context import DefaultProviderAuthContext

    class _Overlaid(DefaultProviderAuthContext):
        """在基础上下文之上优先返回覆盖值的认证上下文。"""

        async def env(self, name: str) -> str | None:
            """先查覆盖值，未命中时回退到基础上下文。

            Args:
                name: 环境变量名。

            Returns:
                环境变量值；两处都未设置时为 ``None``。
            """
            overlay = env.get(name)
            if overlay:
                return overlay
            return await base.env(name)

        async def file_exists(self, path: str) -> bool:
            """委托基础上下文判断文件是否存在。

            Args:
                path: 待检查的文件路径。

            Returns:
                路径存在时为 ``True``。
            """
            return await base.file_exists(path)

    return _Overlaid()


async def _resolve_api_key(
    auth_context: AuthContext,
    api_key: Any,
    provider_id: str,
    credential: ApiKeyCredential | None,
    signal: AbortSignal,
) -> AuthResult | None:
    """调用 api-key 认证实现解析请求认证，并包装底层异常。

    Args:
        auth_context: 认证上下文。
        api_key: provider 的 api-key 认证实现。
        provider_id: 目标 provider 标识，仅用于错误信息。
        credential: 已存储的 api-key credential；无则为 ``None``。
        signal: 取消信号。

    Returns:
        解析结果；未配置时返回 ``None``。

    Raises:
        ModelsError: 底层认证实现抛出异常时。
    """
    try:
        return await api_key.resolve(ctx=auth_context, credential=credential, signal=signal)
    except Exception as error:  # noqa: BLE001 - 包装为类型化认证错误
        raise ModelsError("auth", f"API key auth failed for provider {provider_id}", cause=error) from error


async def _read_credential(
    credentials: CredentialStore,
    provider_id: str,
    signal: AbortSignal,
) -> Credential | None:
    """读取 provider 已存储的 credential，并包装底层异常。

    Args:
        credentials: 应用自持的 credential 存储。
        provider_id: 目标 provider 标识。
        signal: 取消信号。

    Returns:
        已存储的 credential；不存在时为 ``None``。

    Raises:
        ModelsError: store 读取失败时。
    """
    from .types import AuthOperationOptions

    try:
        return await credentials.read(provider_id, AuthOperationOptions(signal=signal))
    except Exception as error:  # noqa: BLE001 - 包装为类型化认证错误
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error) from error
