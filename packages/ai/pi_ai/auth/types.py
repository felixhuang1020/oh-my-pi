"""认证相关类型：credential、provider 认证方式与登录交互。

移植自 ``packages/ai/src/auth/types.ts``。结构契约使用 ``Protocol``，
应用提供的 store 与 provider 认证对象无需继承这些类型。

本项目只支持 api-key 认证：OAuth credential、OAuth 流程与相关交互事件
均已移除。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from ..types import ProviderEnv, ProviderHeaders
from ..utils.abort import AbortSignal

__all__ = [
    "ApiKeyAuth",
    "ApiKeyCredential",
    "AuthCheck",
    "AuthContext",
    "AuthInteraction",
    "AuthOperationOptions",
    "AuthPrompt",
    "AuthPromptSecret",
    "AuthPromptText",
    "AuthResult",
    "AuthType",
    "Credential",
    "CredentialInfo",
    "CredentialStore",
    "ModelAuth",
    "ProviderAuth",
    "ProviderAuthInteraction",
]


@dataclass
class ModelAuth:
    """单次模型请求所用的请求认证。

    Attributes:
        api_key: 请求使用的 API key；``None`` 表示不附加 key。
        headers: 附加到请求上的 provider 头。
        base_url: 覆盖 provider 默认的 base URL；``None`` 表示沿用默认值。
    """

    api_key: str | None = field(default=None, metadata={"alias": "apiKey"})
    headers: ProviderHeaders | None = None
    base_url: str | None = field(default=None, metadata={"alias": "baseUrl"})


@dataclass
class ApiKeyCredential:
    """已存储的 api-key credential。

    ``env`` 保存 provider 作用域的环境/配置值，如 Cloudflare 账户与 gateway id。

    Attributes:
        type: credential 类型标签，固定为 ``"api_key"``。
        key: 已存储的 API key；未设置时为 ``None``。
        env: provider 作用域的环境/配置值。
    """

    type: str = "api_key"
    key: str | None = None
    env: ProviderEnv | None = None


#: 每个 provider 一条带类型标签的 credential。
Credential: TypeAlias = ApiKeyCredential


@dataclass
class CredentialInfo:
    """用于账户/状态枚举的非机密 credential 元数据。

    Attributes:
        provider_id: credential 所属的 provider 标识。
        type: credential 类型标签。
    """

    provider_id: str = field(default="", metadata={"alias": "providerId"})
    type: str = "api_key"


@dataclass
class AuthOperationOptions:
    """公开认证与 credential 操作的可选取消控制。

    Attributes:
        signal: 取消信号；``None`` 表示操作不可取消。
    """

    signal: AbortSignal | None = None


@runtime_checkable
class CredentialStore(Protocol):
    """应用自持的 credential 存储，以 ``Provider.id`` 为键。

    ``modify`` 是唯一写入路径，所有变更都是串行化的读-改-写。
    """

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        """读取指定 provider 的 credential。

        Args:
            provider_id: 目标 provider 标识，对应 ``Provider.id``。
            options: 可选的取消控制。

        Returns:
            已存储的 credential；不存在时为 ``None``。
        """
        ...

    async def list(self, options: AuthOperationOptions | None = None) -> Sequence[CredentialInfo]:
        """列出所有已存储 credential 的非机密元数据。

        Args:
            options: 可选的取消控制。

        Returns:
            credential 元数据序列。
        """
        ...

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        """以串行化的读-改-写方式修改指定 provider 的 credential。

        Args:
            provider_id: 目标 provider 标识，对应 ``Provider.id``。
            fn: 接收当前 credential（可能为 ``None``）并返回新值的回调。
            options: 可选的取消控制。

        Returns:
            修改后的 credential；回调返回 ``None`` 时返回当前值。
        """
        ...

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        """删除指定 provider 的 credential。

        Args:
            provider_id: 目标 provider 标识，对应 ``Provider.id``。
            options: 可选的取消控制。
        """
        ...


@runtime_checkable
class AuthContext(Protocol):
    """认证解析所需的环境访问接口；可注入，便于测试与浏览器环境使用。"""

    async def env(self, name: str) -> str | None:
        """读取环境变量的值。

        Args:
            name: 环境变量名。

        Returns:
            环境变量值；未设置或为空时为 ``None``。
        """
        ...

    async def file_exists(self, path: str) -> bool:
        """判断给定路径是否存在。

        Args:
            path: 待检查的文件路径。

        Returns:
            路径存在时为 ``True``。
        """
        ...


@dataclass
class AuthResult:
    """解析模型认证的结果。

    Attributes:
        auth: 可直接用于模型请求的请求认证。
        env: 从 credential 与环境上下文解析出的 provider 作用域环境/配置值。
        source: 状态 UI 显示的人类可读标签：``"ANTHROPIC_API_KEY"``、``"OAuth"`` 等。
    """

    auth: ModelAuth = field(default_factory=ModelAuth)
    #: 从 credential 与环境上下文解析出的 provider 作用域环境/配置值。
    env: ProviderEnv | None = None
    #: 状态 UI 显示的人类可读标签：``"ANTHROPIC_API_KEY"``、``"OAuth"`` 等。
    source: str | None = None


@dataclass
class AuthCheck:
    """无副作用的可用性检查结果。

    Attributes:
        type: 命中的认证类型标签。
        source: 认证来源标签，如环境变量名；未命中时为 ``None``。
    """

    type: str = "api_key"
    source: str | None = None


#: provider 的认证方式。
AuthType: TypeAlias = str


@dataclass
class AuthPromptText:
    """自由文本输入提示。

    Attributes:
        message: 展示给用户的提示文字。
        type: 提示类型标签，固定为 ``"text"``。
        placeholder: 输入框占位文字。
        signal: 取消信号；``None`` 表示沿用交互对象的信号。
    """

    message: str = ""
    type: str = "text"
    placeholder: str | None = None
    signal: AbortSignal | None = None


@dataclass
class AuthPromptSecret:
    """不回显的机密值输入提示。

    Attributes:
        message: 展示给用户的提示文字。
        type: 提示类型标签，固定为 ``"secret"``。
        placeholder: 输入框占位文字。
        signal: 取消信号；``None`` 表示沿用交互对象的信号。
    """

    message: str = ""
    type: str = "secret"
    placeholder: str | None = None
    signal: AbortSignal | None = None


#: 登录期间展示给用户的提示。
AuthPrompt: TypeAlias = AuthPromptText | AuthPromptSecret


@runtime_checkable
class AuthInteraction(Protocol):
    """api-key 登录交互回调。

    ``prompt`` 返回用户输入的字符串，取消或中止时抛出异常。
    """

    signal: AbortSignal | None

    async def prompt(self, prompt: AuthPrompt) -> str:
        """展示提示并等待用户输入。

        Args:
            prompt: 要展示的提示。

        Returns:
            用户输入的字符串。
        """
        ...


@runtime_checkable
class ProviderAuthInteraction(AuthInteraction, Protocol):
    """传递给 provider 登录实现的规范化交互对象。"""

    signal: AbortSignal


class ProviderAuthInteractionImpl:
    """始终携带 signal 的具体 :class:`ProviderAuthInteraction` 实现。"""

    def __init__(self, interaction: AuthInteraction) -> None:
        """包装交互对象并补齐 ``signal``。

        Args:
            interaction: 被包装的交互对象。
        """
        self._interaction = interaction
        self.signal = interaction.signal or AbortSignal()

    async def prompt(self, prompt: AuthPrompt) -> str:
        """转发提示请求到底层交互对象。

        Args:
            prompt: 要展示的提示。

        Returns:
            用户输入的字符串。
        """
        return await self._interaction.prompt(prompt)



@runtime_checkable
class ApiKeyAuth(Protocol):
    """api-key 认证：已存储 key、provider 环境变量及环境来源。"""

    @property
    def name(self) -> str:
        """认证方式的人类可读名称。"""
        ...

    async def login(self, interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        """交互式获取 api key。

        Args:
            interaction: 登录交互对象。

        Returns:
            新得到的 api-key credential。
        """
        ...

    async def check(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None = None,
        signal: AbortSignal,
    ) -> AuthCheck | None:
        """无副作用地判断认证是否可用。

        Args:
            ctx: 认证上下文，用于读取环境变量与文件。
            credential: 已存储的 credential；未存储时为 ``None``。
            signal: 取消信号。

        Returns:
            检查结果；不可用时为 ``None``。
        """
        ...

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None = None,
        signal: AbortSignal,
    ) -> AuthResult | None:
        """解析出请求认证。

        Args:
            ctx: 认证上下文，用于读取环境变量与文件。
            credential: 已存储的 credential；未存储时为 ``None``。
            signal: 取消信号。

        Returns:
            解析结果；未配置时返回 ``None``。
        """
        ...


@dataclass
class ProviderAuth:
    """provider 的认证方式。

    即使是仅依赖环境 credential 的 provider 和无 key 的本地服务，也会提供
    ``api_key`` 认证，由其 ``resolve()`` 报告 provider 是否已配置。

    Attributes:
        api_key: api-key 认证实现。
    """

    api_key: Any = field(default=None, metadata={"alias": "apiKey"})
