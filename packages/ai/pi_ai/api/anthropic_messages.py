"""Anthropic Messages API 适配器。

移植自 ``packages/ai/src/api/anthropic-messages.ts``。

TypeScript 实现由 ``@anthropic-ai/sdk`` 驱动；移植版保持相同的请求/stream 形状，但自己通过
:mod:`httpx` 发起流式 ``POST /v1/messages?beta=true``（见 :class:`AnthropicMessagesClient`），
因此重试、取消与 SSE 解码都走其他适配器共用的辅助函数。适配器编码为带
``stop_reason="error"``/``"aborted"`` 的 ``AssistantMessage`` 的一切行为均与 TypeScript 一致：
返回的 stream 上不会抛出异常。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from typing import Any, Protocol, TypeAlias, runtime_checkable

import httpx

from ..models import calculate_cost
from ..types import (
    AnthropicAllowedFallbackModel,
    AssistantMessage,
    CacheRetention,
    Message,
    Model,
    ModelCompat,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UserContent,
)
from ..utils.abort import AbortSignal, race_with_abort_signal
from ..utils.async_utils import maybe_await
from ..utils.diagnostics import AssistantMessageDiagnostic, append_assistant_message_diagnostic
from ..utils.error_body import MAX_PROVIDER_ERROR_BODY_CHARS, safe_json_stringify, truncate_error_text
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.events import (
    done_event,
    error_event,
    start_event,
    text_delta_event,
    text_end_event,
    text_start_event,
    thinking_delta_event,
    thinking_end_event,
    thinking_start_event,
    tool_call_delta_event,
    tool_call_end_event,
    tool_call_start_event,
)
from ..utils.headers import headers_to_record
from ..utils.http import (
    build_headers,
    create_client as create_http_client,
    read_error_body,
    resolve_timeout,
)
from ..utils.json_parse import parse_json_with_repair, parse_streaming_json
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.provider_env import get_provider_env_value
from ..env_api_keys import (
    ANTHROPIC_FEDERATION_RULE_ID_ENV,
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV,
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV,
    ANTHROPIC_WORKSPACE_ID_ENV,
)
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.sse import SSEEvent, iter_sse_events
from ..utils.text import get_system_message_text, render_system_message_update
from ..utils.transcript import (
    get_current_tools,
    get_declared_tools,
    get_initial_system_message,
    has_tool_redefinitions,
    resolve_transcript,
)

from .constrained_sampling import (
    get_json_schema_tool_parameters,
    resolve_json_schema_strict_sampling,
)
from .simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
)
from .transform_messages import transform_messages

__all__ = [
    "ANTHROPIC_OPTIONS_DOC",
    "AnthropicAPIError",
    "AnthropicClientProtocol",
    "AnthropicEffort",
    "AnthropicMessagesClient",
    "AnthropicOptions",
    "AnthropicThinkingDisplay",
    "anthropic_messages_api",
    "stream",
    "stream_simple",
]

#: Anthropic 线上协议别名：``MessageParam`` 在线上就是一个普通 JSON 对象。
MessageParam: TypeAlias = dict[str, Any]
ContentBlockParam: TypeAlias = dict[str, Any]
CacheControlEphemeral: TypeAlias = dict[str, str]

#: Anthropic 专有选项的文档，按移植版其余部分消费 ``ANTHROPIC_OPTIONS_DOC`` 的方式，
#: 保留为模块常量。
ANTHROPIC_OPTIONS_DOC = """\
Anthropic Messages options beyond StreamOptions:

thinkingEnabled
    Enable extended thinking.  For adaptive thinking models the model decides when and how
    much to think; for older models budget-based thinking with ``thinkingBudgetTokens`` is
    used.  Default: unset (``streamSimple`` maps a reasoning level, or callers set it).
thinkingBudgetTokens
    Token budget for extended thinking (older models only, ignored by adaptive models).
    Default: 1024 when ``thinkingEnabled`` is true and no budget is provided.
effort
    Effort level for adaptive thinking models: ``low``, ``medium``, ``high``, ``xhigh`` or
    ``max``.  Ignored by older models.
thinkingDisplay
    ``summarized`` (default when thinking is enabled) or ``omitted``.  ``omitted`` still
    returns the encrypted signature for multi-turn continuity.
interleavedThinking
    Request the interleaved-thinking beta header for non-adaptive thinking models.
    Default: true; adaptive models always skip it.
toolChoice
    ``"auto"``, ``"any"``, ``"none"`` or ``{"type": "tool", "name": ...}``.  Default:
    omitted (Anthropic's own default).
client
    A pre-built messages client.  When provided, internal client construction is skipped.
"""


class AnthropicEffort(StrEnum):
    """Anthropic 支持的自适应 thinking effort 等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class AnthropicThinkingDisplay(StrEnum):
    """API 返回 thinking 内容的方式。"""

    SUMMARIZED = "summarized"
    OMITTED = "omitted"


# --------------------------------------------------------------------------------------
# 选项
# --------------------------------------------------------------------------------------


@dataclass
class AnthropicOptions(StreamOptions):
    """Anthropic 专有的 stream 选项（TS ``AnthropicOptions`` 的扩展字段）。"""

    #: 是否启用 extended thinking。
    thinking_enabled: bool | None = field(default=None, metadata={"alias": "thinkingEnabled"})
    #: extended thinking 的 token 预算（仅限旧模型）。
    thinking_budget_tokens: int | None = field(default=None, metadata={"alias": "thinkingBudgetTokens"})
    #: 自适应 thinking 模型的 effort 等级。
    effort: AnthropicEffort | str | None = None
    #: thinking 内容的返回方式：``summarized`` 或 ``omitted``。
    thinking_display: AnthropicThinkingDisplay | str | None = field(default=None, metadata={"alias": "thinkingDisplay"})
    #: 为非自适应 thinking 模型请求 interleaved thinking beta。
    interleaved_thinking: bool | None = field(default=None, metadata={"alias": "interleavedThinking"})
    #: Anthropic 的 tool choice：``"auto" | "any" | "none" | {"type": "tool", "name": str}``。
    tool_choice: str | dict[str, Any] | None = field(default=None, metadata={"alias": "toolChoice"})
    #: 预构建的 client（实现 :class:`AnthropicClientProtocol`），提供后跳过内部构建。
    client: Any = None


# --------------------------------------------------------------------------------------
# 缓存控制
# --------------------------------------------------------------------------------------


@dataclass
class _CacheControl:
    """解析后的缓存保留策略，以及要附加的 ephemeral 缓存标记。

    Attributes:
        retention: 解析后的缓存保留策略：``"none"``/``"short"``/``"long"``。
        cache_control: 要附加的 ephemeral 缓存标记；``retention`` 为 ``"none"`` 时为 ``None``。
    """

    retention: str
    cache_control: CacheControlEphemeral | None = None


def _resolve_cache_retention(
    cache_retention: CacheRetention | str | None = None,
    env: ProviderEnv | None = None,
) -> str:
    """解析缓存保留偏好，默认 ``short``。

    Args:
        cache_retention: 调用方显式指定的缓存保留策略；为 ``None`` 时回退到环境变量。
        env: provider 环境变量来源；为 ``None`` 时读取进程环境。

    Returns:
        解析后的保留策略字符串，默认 ``"short"``。
    """
    if cache_retention:
        return str(cache_retention)
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _resolve_cache_control(
    model: Model,
    cache_retention: CacheRetention | str | None = None,
    env: ProviderEnv | None = None,
) -> _CacheControl:
    """解析请求的缓存保留策略与 ``ephemeral`` 缓存标记。

    Args:
        model: 目标模型，用于读取 compat 中的长缓存保留能力。
        cache_retention: 调用方显式指定的缓存保留策略。
        env: provider 环境变量来源。

    Returns:
        解析后的 :class:`_CacheControl`；``retention`` 为 ``"none"`` 时不带缓存标记。
    """
    retention = _resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return _CacheControl(retention=retention)
    ttl = "1h" if retention == "long" and _get_anthropic_compat(model).supports_long_cache_retention else None
    cache_control: CacheControlEphemeral = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return _CacheControl(retention=retention, cache_control=cache_control)


# --------------------------------------------------------------------------------------
# Claude Code 隐身模式
# --------------------------------------------------------------------------------------

#: 严格仿照 Claude Code 的工具命名。
_CLAUDE_CODE_VERSION = "2.1.280"

#: Claude Code 2.x 的工具名（规范大小写）。
_CLAUDE_CODE_TOOLS: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
)

_CC_TOOL_LOOKUP: dict[str, str] = {tool.lower(): tool for tool in _CLAUDE_CODE_TOOLS}


def _to_claude_code_name(name: str) -> str:
    """若工具名匹配（不区分大小写），转换为 CC 的规范大小写。

    Args:
        name: 调用方使用的工具名。

    Returns:
        匹配到时返回 Claude Code 规范大小写，否则原样返回。
    """
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def _from_claude_code_name(name: str, tools: list[Tool] | None) -> str:
    """把返回的工具名还原回调用方自己的大小写。

    Args:
        name: Anthropic 返回的工具名。
        tools: 调用方声明的工具列表；为 ``None`` 时无法还原。

    Returns:
        还原后的工具名；找不到匹配时保持原样。
    """
    if tools:
        lower_name = name.lower()
        for tool in tools:
            if tool.name.lower() == lower_name:
                return tool.name
    return name


# --------------------------------------------------------------------------------------
# 内容转换
# --------------------------------------------------------------------------------------


def _convert_content_blocks(content: list[UserContent]) -> str:
    """把用户/工具结果的内容块转换为 Anthropic API 格式。

    Args:
        content: 用户或 tool result 的内容块列表。

    Returns:
        拼接后的文本。
    """
    return sanitize_surrogates("\n".join(getattr(block, "text", "") for block in content))


# --------------------------------------------------------------------------------------
# 请求头辅助函数
# --------------------------------------------------------------------------------------


def _merge_headers(*header_sources: ProviderHeaders | None) -> ProviderHeaders:
    """类似对象展开赋值：后面的来源覆盖前面的，``None`` 值原样保留。

    Args:
        *header_sources: 按优先级从低到高排列的请求头来源。

    Returns:
        合并后的请求头。
    """
    merged: ProviderHeaders = {}
    for headers in header_sources:
        if headers:
            merged.update(headers)
    return merged


def _merge_client_headers(*header_sources: ProviderHeaders | None) -> ProviderHeaders:
    """在 pi 默认 user agent 之上合并 client 请求头。

    Args:
        *header_sources: 按优先级从低到高排列的请求头来源。

    Returns:
        含 ``User-Agent`` 的合并请求头。
    """
    return _merge_headers({"User-Agent": get_pi_user_agent()}, *header_sources)


def _has_header(headers: ProviderHeaders | None, name: str) -> bool:
    """``headers`` 中是否存在 ``name`` 对应的非空值（不区分大小写）。

    Args:
        headers: 待检查的请求头；可为 ``None``。
        name: 请求头名称。

    Returns:
        存在非空值时返回 ``True``。
    """
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value is not None and value.strip():
            return True
    return False


def _has_request_auth(api_key: str | None, headers: ProviderHeaders | None) -> bool:
    """请求是否携带任何可用凭证。

    Args:
        api_key: 解析到的 API key。
        headers: 请求头。

    Returns:
        存在 API key 或任一认证头时返回 ``True``。
    """
    return bool(api_key) or _has_header(headers, "authorization") or _has_header(
        headers, "x-api-key"
    ) or _has_header(headers, "cf-aig-authorization")


def _assert_request_auth(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> None:
    """未解析到 key 或认证头时抛出异常。

    Args:
        provider: provider 名称，用于错误消息。
        api_key: 解析到的 API key。
        headers: 请求头。

    Raises:
        RuntimeError: 既无 API key 也无认证头时。
    """
    if not _has_request_auth(api_key, headers):
        raise RuntimeError(f"No API key for provider: {provider}")


# --------------------------------------------------------------------------------------
# 兼容性标志
# --------------------------------------------------------------------------------------


@dataclass
class _AnthropicCompat:
    """某个模型解析后的 Anthropic 兼容性标志。

    Attributes:
        supports_eager_tool_input_streaming: 是否支持 eager tool input streaming。
        supports_long_cache_retention: 是否支持 ``1h`` 长缓存保留。
        send_session_affinity_headers: 是否发送会话亲和请求头。
        session_affinity_format: 会话亲和请求头格式（如 ``"openrouter"``）。
        supports_cache_control_on_tools: 是否支持在工具声明上设置缓存标记。
        supports_temperature: 是否支持 ``temperature`` 参数。
        allow_empty_signature: 是否接受空 thinking 签名。
        supports_strict_tools: 是否支持 strict 工具模式。
        supports_mid_convo_system_messages: 是否支持会话中途插入 system 消息。
        supports_mid_convo_tool_changes: 是否支持会话中途变更工具。
    """

    supports_eager_tool_input_streaming: bool = True
    supports_long_cache_retention: bool = True
    send_session_affinity_headers: bool = False
    session_affinity_format: str | None = None
    supports_cache_control_on_tools: bool = True
    supports_temperature: bool = True
    allow_empty_signature: bool = False
    supports_strict_tools: bool = False
    supports_mid_convo_system_messages: bool = False
    supports_mid_convo_tool_changes: bool = False


def _model_compat(model: Model, name: str) -> Any:
    """``model.compat?.<name>``——标志值，未设置时为 ``None``。

    Args:
        model: 目标模型。
        name: compat 字段名。

    Returns:
        标志值；``model.compat`` 缺失或字段未设置时返回 ``None``。
    """
    compat: ModelCompat | None = model.compat
    return getattr(compat, name, None) if compat is not None else None


def _compat_flag(model: Model, name: str, default: Any) -> Any:
    """``model.compat?.<name> ?? default`` 的语义：未设置时回退到默认值。

    Args:
        model: 目标模型。
        name: compat 字段名。
        default: 字段未设置时使用的默认值。

    Returns:
        标志值或 ``default``。
    """
    value = _model_compat(model, name)
    return default if value is None else value


def _get_anthropic_compat(model: Model) -> _AnthropicCompat:
    """解析所有 Anthropic 兼容性标志，未设置时使用内置默认值。

    Args:
        model: 目标模型。

    Returns:
        解析后的 :class:`_AnthropicCompat`。
    """
    is_open_router = model.provider == "openrouter" or "openrouter.ai" in model.base_url
    return _AnthropicCompat(
        supports_eager_tool_input_streaming=_compat_flag(model, "supports_eager_tool_input_streaming", True),
        supports_long_cache_retention=_compat_flag(model, "supports_long_cache_retention", True),
        send_session_affinity_headers=_compat_flag(model, "send_session_affinity_headers", is_open_router),
        session_affinity_format=_compat_flag(
            model, "session_affinity_format", "openrouter" if is_open_router else None
        ),
        supports_cache_control_on_tools=_compat_flag(model, "supports_cache_control_on_tools", True),
        supports_temperature=_compat_flag(model, "supports_temperature", True),
        allow_empty_signature=_compat_flag(model, "allow_empty_signature", False),
        supports_strict_tools=_compat_flag(model, "supports_strict_tools", False),
        supports_mid_convo_system_messages=_compat_flag(model, "supports_mid_convo_system_messages", False),
        supports_mid_convo_tool_changes=_compat_flag(model, "supports_mid_convo_tool_changes", False),
    )


def _should_use_server_side_fallback_beta(model: Model) -> bool:
    """该模型是否需要服务端 fallback beta。

    Args:
        model: 目标模型。

    Returns:
        配置了 ``allowed_fallback_models`` 时返回 ``True``。
    """
    return len(_model_compat(model, "allowed_fallback_models") or []) > 0


# --------------------------------------------------------------------------------------
# 工作负载身份联邦
# --------------------------------------------------------------------------------------

#: ``NonNullable<ClientOptions["config"]>`` 的冻结表示。
AnthropicFederationConfig: TypeAlias = dict[str, Any]


def _get_anthropic_federation(
    model: Model,
    api_key: str | None,
    headers: ProviderHeaders | None,
    env: ProviderEnv | None,
) -> AnthropicFederationConfig | None:
    """在可用时解析 ANTHROPIC_* 工作负载身份联邦配置。

    仅适用于 ``anthropic`` provider，且仅在未解析到 key 或认证头时生效。
    TypeScript 版将其交给 SDK 执行 OIDC token 交换；Python 移植版忠实解析该配置，
    但无法执行 SDK 内部的交换流程（见模块说明）。

    Args:
        model: 目标模型，仅 ``anthropic`` provider 会启用。
        api_key: 已解析的 API key。
        headers: 请求头；存在认证头时跳过联邦。
        env: provider 环境变量来源。

    Returns:
        联邦配置字典；不适用或缺少必需环境变量时返回 ``None``。
    """
    if model.provider != "anthropic" or _has_request_auth(api_key, headers):
        return None
    federation_rule_id = get_provider_env_value(ANTHROPIC_FEDERATION_RULE_ID_ENV, env)
    organization_id = get_provider_env_value(ANTHROPIC_ORGANIZATION_ID_ENV, env)
    identity_token_file = get_provider_env_value(ANTHROPIC_IDENTITY_TOKEN_FILE_ENV, env)
    if not federation_rule_id or not organization_id or not identity_token_file:
        return None
    return {
        "organization_id": organization_id,
        "workspace_id": get_provider_env_value(ANTHROPIC_WORKSPACE_ID_ENV, env),
        "authentication": {
            "type": "oidc_federation",
            "federation_rule_id": federation_rule_id,
            "service_account_id": get_provider_env_value(ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, env),
            "identity_token": {"source": "file", "path": identity_token_file},
        },
    }


# --------------------------------------------------------------------------------------
# HTTP 客户端
# --------------------------------------------------------------------------------------


class AnthropicAPIError(Exception):
    """非 2xx 的 Anthropic 响应，结构仿照 SDK 的 ``APIError``。

    ``status`` 与 ``headers`` 是 :func:`pi_ai.utils.provider_retry.retry_provider_request`
    检查的字段，因此即使没有 SDK，重试行为也与 TypeScript 版一致。
    """

    def __init__(self, status: int, headers: Any, body_text: str = "") -> None:
        """构造一个非 2xx 的 Anthropic 响应异常。

        Args:
            status: HTTP 状态码。
            headers: 响应头，供重试逻辑读取。
            body_text: 响应体文本，用于拼装错误消息。
        """
        super().__init__(_api_error_message(status, body_text))
        self.status = status
        self.headers = headers
        self.body = body_text


def _api_error_message(status: int, body_text: str) -> str:
    """构造 SDK 风格的错误消息：``"<status> <body>"``。

    Args:
        status: HTTP 状态码。
        body_text: 响应体文本。

    Returns:
        截断并规范化后的错误消息。
    """
    text = truncate_error_text(body_text.strip(), MAX_PROVIDER_ERROR_BODY_CHARS)
    if not text:
        return f"{status} status code (no body)"
    try:
        parsed = json.loads(text)
    except ValueError:
        return f"{status} {text}"
    if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
        return f"{status} {parsed['message']}"
    return f"{status} {safe_json_stringify(parsed)}"


@runtime_checkable
class AnthropicClientProtocol(Protocol):
    """本适配器用到的 Anthropic SDK client 的最小子集。

    以 ``AnthropicOptions.client`` 传入的预构建 client 只需实现该接口：
    发起流式请求并交回原始 :class:`httpx.Response`。
    """

    async def create_message(
        self,
        params: MessageParam,
        *,
        signal: AbortSignal | None = None,
        timeout_ms: int | None = None,
    ) -> httpx.Response:
        """协议声明：发送流式 Messages 请求并返回原始响应。

        Args:
            params: Messages 请求体。
            signal: 中止信号；仅为接口对齐而接受。
            timeout_ms: 请求超时（毫秒）。

        Returns:
            原始 :class:`httpx.Response`。
        """
        ...


def _build_url(base_url: str, path: str) -> str:
    """按 SDK ``buildURL`` 的方式拼接 ``base_url`` 与 ``path``。

    Args:
        base_url: 基础 URL。
        path: 请求路径。

    Returns:
        拼接后的完整 URL，避免出现重复的 ``/``。
    """
    if base_url.endswith("/") and path.startswith("/"):
        return base_url + path[1:]
    return base_url + path


class AnthropicMessagesClient:
    """``@anthropic-ai/sdk`` beta Messages client 的最小替代实现。

    保持 SDK 的线上协议——``POST {baseURL}/v1/messages?beta=true``、把 body 中的 ``betas``
    移入 ``anthropic-beta`` 头、``x-api-key`` 或 ``Authorization: Bearer`` 认证——并返回原始
    响应供适配器的 SSE 解码器消费。HTTP client 懒创建，因此同一个实例可在 :meth:`aclose`
    之前服务于多次请求。
    """

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str | None = None,
        auth_token: str | None = None,
        default_headers: ProviderHeaders | None = None,
        fetch: Any = None,
        timeout_ms: int | None = None,
        config: AnthropicFederationConfig | None = None,
    ) -> None:
        """初始化最小替代版 Messages client。

        Args:
            base_url: Anthropic API 基础 URL。
            api_key: API key；与 ``auth_token`` 二选一用于认证。
            auth_token: Bearer token；优先于 ``api_key``。
            default_headers: 随每次请求发送的默认请求头。
            fetch: 自定义 fetch 实现；透传给底层 HTTP client。
            timeout_ms: 默认请求超时（毫秒）。
            config: 工作负载身份联邦配置；仅为 API 对齐而接受。
        """
        self.base_url = base_url
        self.api_key = api_key
        self.auth_token = auth_token
        self.default_headers: ProviderHeaders = dict(default_headers or {})
        self.fetch = fetch
        self.timeout_ms = timeout_ms
        #: 接受 federation 配置以保持 API 对齐；SDK 侧的 OIDC 交换未移植。
        self.config = config
        self._http: httpx.AsyncClient | None = None

    def _http_client(self) -> httpx.AsyncClient:
        """懒创建并返回底层 HTTP client，同时按认证方式补齐请求头。

        Returns:
            配置好认证头与 ``anthropic-version`` 的复用型 :class:`httpx.AsyncClient`。
        """
        if self._http is None:
            headers = build_headers(self.default_headers)
            if self.auth_token:
                headers["authorization"] = f"Bearer {self.auth_token}"
            elif self.api_key:
                headers["x-api-key"] = self.api_key
            headers.setdefault("anthropic-version", "2023-06-01")
            self._http = create_http_client(
                headers=headers,
                timeout_ms=self.timeout_ms,
                fetch=self.fetch,
                base_url=self.base_url,
            )
        return self._http

    async def create_message(
        self,
        params: MessageParam,
        *,
        signal: AbortSignal | None = None,
        timeout_ms: int | None = None,
    ) -> httpx.Response:
        """发送流式 Messages 请求并返回原始响应。

        ``timeout_ms`` 覆盖 client 自身的请求超时；``signal`` 仅为 API 对齐而接受，
        取消由调用方通过竞争（race）处理。

        Args:
            params: Messages 请求体。
            signal: 中止信号；仅用于 API 对齐。
            timeout_ms: 覆盖 client 默认请求超时（毫秒）。

        Returns:
            尚未读取 body 的流式 :class:`httpx.Response`。

        Raises:
            AnthropicAPIError: 响应状态码不小于 400 时。
        """
        http = self._http_client()
        body = dict(params)
        betas = body.pop("betas", None)
        headers: dict[str, str] = {}
        if betas is not None:
            headers["anthropic-beta"] = ",".join(str(feature) for feature in betas)
        request = http.build_request(
            "POST",
            _build_url(self.base_url, "/v1/messages?beta=true"),
            json=body,
            headers=headers,
        )
        if timeout_ms is not None:
            request.extensions = {**request.extensions, "timeout": resolve_timeout(timeout_ms).as_dict()}
        response = await http.send(request, stream=True)
        if response.status_code >= 400:
            body_text = await read_error_body(response)
            await response.aclose()
            raise AnthropicAPIError(response.status_code, response.headers, body_text)
        return response

    async def aclose(self) -> None:
        """关闭底层 HTTP client。"""
        if self._http is not None:
            http, self._http = self._http, None
            await http.aclose()


@dataclass
class _CreatedClient:
    """:func:`_create_client` 的结果：client 及其认证方式。

    Attributes:
        client: 构建好的 messages client。
        is_oauth_token: 是否使用 Claude Code OAuth token 认证。
    """

    client: AnthropicClientProtocol
    is_oauth_token: bool


def _is_oauth_token(api_key: str) -> bool:
    """Claude Code OAuth token 带有 ``sk-ant-oat`` 标记。

    Args:
        api_key: 待判断的 token。

    Returns:
        是 OAuth token 时返回 ``True``。
    """
    return "sk-ant-oat" in api_key


def _create_client(
    model: Model,
    api_key: str | None,
    options_headers: ProviderHeaders | None = None,
    fetch: Any = None,
    session_id: str | None = None,
    federation: AnthropicFederationConfig | None = None,
    timeout_ms: int | None = None,
) -> _CreatedClient:
    """为单次请求构建 messages client，对应 SDK 构造函数的选项。

    Args:
        model: 目标模型，决定 base URL 与请求头。
        api_key: 已解析的 API key；OAuth 时用作 Bearer token。
        options_headers: 调用方通过选项传入的请求头。
        fetch: 自定义 fetch 实现；透传给底层 HTTP client。
        session_id: 会话 id，用于会话亲和请求头。
        federation: 工作负载身份联邦配置。
        timeout_ms: 请求超时（毫秒）。

    Returns:
        构造好的 :class:`_CreatedClient`，其 ``is_oauth_token`` 标识认证方式。
    """
    # OAuth：Bearer 认证，并携带 Claude Code 身份请求头。
    if api_key and _is_oauth_token(api_key):
        return _CreatedClient(
            AnthropicMessagesClient(
                base_url=model.base_url,
                api_key=None,
                auth_token=api_key,
                default_headers=_merge_client_headers(
                    {
                        "accept": "application/json",
                        "anthropic-dangerous-direct-browser-access": "true",
                        "user-agent": f"claude-cli/{_CLAUDE_CODE_VERSION}",
                        "x-app": "cli",
                    },
                    model.headers,
                    options_headers,
                ),
                fetch=fetch,
                timeout_ms=timeout_ms,
            ),
            True,
        )

    # API key、由请求头自带的认证，或工作负载身份联邦。
    compat = _get_anthropic_compat(model)
    session_affinity_headers: ProviderHeaders = {}
    if session_id and compat.send_session_affinity_headers:
        header = "x-session-id" if compat.session_affinity_format == "openrouter" else "x-session-affinity"
        session_affinity_headers[header] = session_id
    default_headers = _merge_client_headers(
        {
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        },
        session_affinity_headers,
        model.headers,
        options_headers,
    )
    if federation:
        return _CreatedClient(
            AnthropicMessagesClient(
                base_url=model.base_url,
                api_key=None,
                auth_token=None,
                default_headers=default_headers,
                fetch=fetch,
                config=federation,
                timeout_ms=timeout_ms,
            ),
            False,
        )

    return _CreatedClient(
        AnthropicMessagesClient(
            base_url=model.base_url,
            api_key=api_key,
            auth_token=None,
            default_headers=default_headers,
            fetch=fetch,
            timeout_ms=timeout_ms,
        ),
        False,
    )


# --------------------------------------------------------------------------------------
# Beta 特性与请求参数
# --------------------------------------------------------------------------------------

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MID_CONVERSATION_OUTPUT_CONFIG_BETA = "mid-conversation-output-config-2026-07-01"
THINKING_BINDING_CONTROLS_BETA = "thinking-binding-controls-2026-08-01"
MID_CONVERSATION_TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01"

#: 启用原生工具变更时始终声明的稳定 deferred 工具。只要任一工具带 ``defer_loading``，
#: Anthropic 就会添加隐藏的 prompt 脚手架；从首个请求起就声明该占位符，能让脚手架留在
#: 缓存前缀中，使第一个真正的延迟工具不会导致缓存失效。它永远不会被激活，模型也看不到它。
DEFERRED_TOOL_PLACEHOLDER: ContentBlockParam = {
    "name": "__pi_deferred_placeholder__",
    "description": "Reserved placeholder. Never available. Never call this.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
    "defer_loading": True,
}

#: 哨兵值，表示「完全未配置 ``anthropic-beta`` 请求头」。
_UNSET: Any = object()


def _should_use_fine_grained_tool_streaming_beta(model: Model, context: TranscriptContext) -> bool:
    """是否需要旧版 fine-grained tool streaming beta。

    Args:
        model: 目标模型。
        context: 会话上下文，用于读取当前工具。

    Returns:
        存在工具且模型不支持 eager tool input streaming 时返回 ``True``。
    """
    return len(get_current_tools(context.messages)) > 0 and not _get_anthropic_compat(
        model
    ).supports_eager_tool_input_streaming


def _get_beta_features(
    model: Model,
    context: TranscriptContext,
    is_oauth_token: bool,
    native_tool_changes: bool,
    options: AnthropicOptions | None = None,
) -> list[str]:
    """解析单次请求的 ``anthropic-beta`` 特性列表。

    Args:
        model: 目标模型。
        context: 会话上下文。
        is_oauth_token: 是否为 Claude Code OAuth token。
        native_tool_changes: 是否启用原生工具变更。
        options: Anthropic 专属选项。

    Returns:
        去重后的 beta 特性列表；显式配置了 ``anthropic-beta`` 头时以该配置为准。
    """
    configured_features: Any = _UNSET
    for headers in (model.headers, options.headers if options else None):
        for name, value in (headers or {}).items():
            if name.lower() == "anthropic-beta":
                configured_features = value
    if configured_features is None:
        return []
    if configured_features is not _UNSET:
        return list(
            dict.fromkeys(
                feature.strip() for feature in str(configured_features).split(",") if feature.strip()
            )
        )

    features: list[str] = []
    if is_oauth_token:
        features.extend(["claude-code-20250219", "oauth-2025-04-20"])
    if _should_use_fine_grained_tool_streaming_beta(model, context):
        features.append(FINE_GRAINED_TOOL_STREAMING_BETA)
    thinking_enabled = options.thinking_enabled if options else None
    interleaved = (options.interleaved_thinking if options and options.interleaved_thinking is not None else True)
    if (
        model.reasoning
        and thinking_enabled is True
        and interleaved
        and _model_compat(model, "force_adaptive_thinking") is not True
    ):
        features.append(INTERLEAVED_THINKING_BETA)
    if _should_use_server_side_fallback_beta(model):
        features.append(SERVER_SIDE_FALLBACK_BETA)
    if _model_compat(model, "supports_mid_convo_effort") is True:
        features.extend([MID_CONVERSATION_OUTPUT_CONFIG_BETA, THINKING_BINDING_CONTROLS_BETA])
    if native_tool_changes:
        features.append(MID_CONVERSATION_TOOL_CHANGES_BETA)
    return list(dict.fromkeys(features))


def _normalize_tool_call_id(value: str, *_: Any) -> str:
    """把 tool call id 规范为 Anthropic 要求的字符集和长度。

    Args:
        value: 原始 tool call id。
        *_: 为兼容调用约定而接受的多余参数，忽略。

    Returns:
        仅含 ``[a-zA-Z0-9_-]`` 且长度不超过 64 的 id。
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:64]


def _convert_tool_result(message: ToolResultMessage) -> ContentBlockParam:
    """把 tool result 消息转换为 Anthropic 的 ``tool_result`` 块。

    Args:
        message: tool result 消息。

    Returns:
        Anthropic ``tool_result`` 内容块。
    """
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": _convert_content_blocks(message.content),
        "is_error": message.is_error,
    }


@dataclass
class _ConvertedAnthropicMessages:
    """线上消息列表，以及每条携带 effort 的 assistant 轮次对应的线上索引。

    Attributes:
        messages: 转换后的 Anthropic 线上消息列表。
        assistant_levels: 线上索引到该轮 assistant 消息 effort 的映射。
    """

    messages: list[MessageParam] = field(default_factory=list)
    assistant_levels: dict[int, str] = field(default_factory=dict)


def _is_anthropic_effort(value: Any) -> bool:
    """``value`` 是否为自适应 thinking 的 effort 等级之一。

    Args:
        value: 待判断的值。

    Returns:
        是已知 effort 等级时返回 ``True``。
    """
    return value in ("low", "medium", "high", "xhigh", "max")


def _convert_messages(
    transformed_messages: list[Message],
    is_oauth_token: bool,
    cache_control: CacheControlEphemeral | None = None,
    allow_empty_signature: bool = False,
    managed_provider: str | None = None,
    native_tool_changes: bool = False,
) -> _ConvertedAnthropicMessages:
    """把会话记录转换为 Anthropic 线上消息。

    Args:
        transformed_messages: 已由 :func:`transform_messages` 规范化的消息。
        is_oauth_token: 是否为 Claude Code OAuth token，决定工具名大小写。
        cache_control: 要附加到末尾的 ``ephemeral`` 缓存标记。
        allow_empty_signature: 是否允许空 thinking 签名（部分兼容 provider 支持）。
        managed_provider: 托管 effort 的 provider 名；非 ``None`` 时记录每轮 effort。
        native_tool_changes: 是否把 system 消息中的工具增删编码为原生块。

    Returns:
        转换结果，含线上消息列表与 assistant 轮次对应的 effort 索引。
    """
    params: list[MessageParam] = []
    assistant_levels: dict[int, str] = {}
    # 后续 system 消息先被暂存，直接发到下一条 assistant 消息之前（或会话末尾）。
    # Anthropic 要求 ``tool_result`` 块必须紧跟其 ``tool_use``，夹在中间的 system 消息会被
    # 拒绝；这也与托管 effort 的 system 消息插入位置一致。
    pending_system_messages: list[MessageParam] = []

    def flush_pending_system_messages() -> None:
        """把暂存的 system 消息追加到线上消息列表并清空暂存区。"""
        params.extend(pending_system_messages)
        pending_system_messages.clear()

    index = 0
    while index < len(transformed_messages):
        message = transformed_messages[index]
        role = getattr(message, "role", None)

        if role == "system":
            # 只有模型原生支持时，后续 system 消息才会走到这里；否则会话记录已被折叠进
            # 首条消息。
            text = render_system_message_update(message)
            blocks: list[ContentBlockParam] = []
            if len(text) > 0:
                blocks.append({"type": "text", "text": sanitize_surrogates(text)})
            if native_tool_changes:
                for tool in message.tools_removed or []:  # type: ignore[union-attr] - 已按 role 分派
                    blocks.append(
                        {
                            "type": "tool_removal",
                            "tool": {
                                "type": "tool_reference",
                                "name": _to_claude_code_name(tool.name) if is_oauth_token else tool.name,
                            },
                        }
                    )
                for tool in message.tools_added or []:  # type: ignore[union-attr] - 已按 role 分派
                    blocks.append(
                        {
                            "type": "tool_addition",
                            "tool": {
                                "type": "tool_reference",
                                "name": _to_claude_code_name(tool.name) if is_oauth_token else tool.name,
                            },
                        }
                    )
            if blocks:
                pending_system_messages.append({"role": "system", "content": blocks})
            index += 1
            continue

        if role == "user":
            content = message.content  # type: ignore[union-attr] - 已按 role 分派
            if isinstance(content, str):
                if content.strip():
                    params.append({"role": "user", "content": sanitize_surrogates(content)})
            else:
                blocks = [
                    {"type": "text", "text": sanitize_surrogates(item.text)}
                    for item in content
                    if getattr(item, "type", None) == "text"
                ]
                filtered_blocks = [block for block in blocks if block["text"].strip()]
                if filtered_blocks:
                    params.append({"role": "user", "content": filtered_blocks})
            index += 1
            continue

        if role == "assistant":
            flush_pending_system_messages()
            blocks = []
            for block in message.content:  # type: ignore[union-attr] - 已按 role 分派
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    if not block.text.strip():  # type: ignore[attr-defined] - 已按 block.type 分派
                        continue
                    blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})  # type: ignore[attr-defined] - 已按 block.type 分派
                elif block_type == "thinking":
                    # 已加密的 thinking：把不透明载荷原样作为 redacted_thinking 传回。
                    if block.redacted:  # type: ignore[attr-defined] - 已按 block.type 分派
                        blocks.append({"type": "redacted_thinking", "data": block.thinking_signature})  # type: ignore[attr-defined] - 已按 block.type 分派
                        continue
                    thinking_signature = block.thinking_signature  # type: ignore[attr-defined] - 已按 block.type 分派
                    has_thinking_signature = bool(thinking_signature) and bool(thinking_signature.strip())
                    if not block.thinking.strip() and not has_thinking_signature:  # type: ignore[attr-defined] - 已按 block.type 分派
                        continue
                    # 签名缺失/为空（例如 stream 被中断）时，为 Anthropic 转成纯文本。
                    # 某些兼容 provider 会发出并接受空签名，因此对已标记的模型保留该块。
                    if not has_thinking_signature:
                        if allow_empty_signature:
                            blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(block.thinking),  # type: ignore[attr-defined] - 已按 block.type 分派
                                    "signature": "",
                                }
                            )
                        else:
                            blocks.append(
                                {"type": "text", "text": sanitize_surrogates(block.thinking)}  # type: ignore[attr-defined] - 已按 block.type 分派
                            )
                    else:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(block.thinking),  # type: ignore[attr-defined] - 已按 block.type 分派
                                "signature": thinking_signature,
                            }
                        )
                elif block_type == "toolCall":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,  # type: ignore[attr-defined] - 已按 block.type 分派
                            "name": _to_claude_code_name(block.name)  # type: ignore[attr-defined] - 已按 block.type 分派
                            if is_oauth_token
                            else block.name,  # type: ignore[attr-defined] - 已按 block.type 分派
                            "input": block.arguments or {},  # type: ignore[attr-defined] - 已按 block.type 分派
                        }
                    )
            if not blocks:
                index += 1
                continue
            message_index = len(params)
            params.append({"role": "assistant", "content": blocks})
            if (
                managed_provider is not None
                and getattr(message, "api", None) == "anthropic-messages"
                and getattr(message, "provider", None) == managed_provider
                and _is_anthropic_effort(getattr(message, "provider_thinking_level", None))
            ):
                assistant_levels[message_index] = message.provider_thinking_level  # type: ignore[union-attr] - 已按 role 分派
            index += 1
            continue

        if role == "toolResult":
            # 收集所有连续的 toolResult 消息，z.ai 的端点需要这种形式。
            tool_results: list[ContentBlockParam] = []
            next_index = index
            while (
                next_index < len(transformed_messages)
                and getattr(transformed_messages[next_index], "role", None) == "toolResult"
            ):
                tool_results.append(_convert_tool_result(transformed_messages[next_index]))  # type: ignore[arg-type] - 该分支已确认是 toolResult
                next_index += 1
            index = next_index
            params.append({"role": "user", "content": tool_results})
            continue

        index += 1

    flush_pending_system_messages()

    # 在最后一条 user 或 system 消息上加 cache_control，以缓存会话历史。
    if cache_control and params:
        last_message = params[-1]
        if last_message["role"] in ("user", "system"):
            content = last_message["content"]
            if isinstance(content, list):
                if content:
                    last_block = content[-1]
                    if isinstance(last_block, dict) and last_block.get("type") in (
                        "text",
                        "tool_result",
                        "tool_addition",
                        "tool_removal",
                    ):
                        last_block["cache_control"] = cache_control
            elif isinstance(content, str):
                last_message["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_control}
                ]

    return _ConvertedAnthropicMessages(messages=params, assistant_levels=assistant_levels)


def _insert_thinking_level_messages(
    converted: _ConvertedAnthropicMessages,
    active_effort: str,
) -> list[MessageParam]:
    """为托管会话中 effort 的模型插入逐条的 effort 标记。

    Args:
        converted: :func:`_convert_messages` 的转换结果。
        active_effort: 当前轮次使用的 effort。

    Returns:
        在每条历史 assistant 轮次前插入其 effort 标记、并在末尾追加当前 effort 的消息列表。
    """
    messages: list[MessageParam] = []
    for index, message in enumerate(converted.messages):
        historical_effort = converted.assistant_levels.get(index)
        if historical_effort is not None:
            messages.append(
                {"role": "system", "content": [], "output_config": {"effort": historical_effort}}
            )
        messages.append(message)
    messages.append({"role": "system", "content": [], "output_config": {"effort": active_effort}})
    return messages


# Anthropic strict tool use 会因这些关键字让整个请求返回 400。
# 参见官方文档：https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
_ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "maxItems",
        "uniqueItems",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
    }
)
_ANTHROPIC_STRICT_STRING_FORMATS = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)


def _is_anthropic_strict_unsupported_keyword(key: str, value: Any) -> bool:
    """Anthropic strict 模式是否会拒绝这个本可使用的关键字。

    Args:
        key: JSON Schema 关键字。
        value: 关键字取值。

    Returns:
        会被拒绝时返回 ``True``。
    """
    if key in _ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS:
        return True
    if key == "minItems":
        if isinstance(value, bool):
            return True
        return not (value == 0 or value == 1)
    if key == "format":
        return not isinstance(value, str) or value not in _ANTHROPIC_STRICT_STRING_FORMATS
    return False


def _convert_tools(
    tools: list[Tool] | None,
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: CacheControlEphemeral | None = None,
) -> list[ContentBlockParam]:
    """把工具声明转换为 Anthropic ``BetaTool`` 线上对象。

    Args:
        tools: 工具声明列表；为空时返回空列表。
        is_oauth_token: 是否为 Claude Code OAuth token，决定工具名大小写。
        supports_eager_tool_input_streaming: 模型是否支持 eager tool input streaming。
        supports_strict_tools: 模型是否支持 strict 工具模式。
        cache_control: 附加到最后一个工具上的缓存标记。

    Returns:
        转换后的 Anthropic 工具对象列表。
    """
    if not tools:
        return []

    converted: list[ContentBlockParam] = []
    for index, tool in enumerate(tools):
        strict = resolve_json_schema_strict_sampling(
            tool, supports_strict_tools, _is_anthropic_strict_unsupported_keyword
        )
        parameters = get_json_schema_tool_parameters(tool, strict)
        schema = parameters if isinstance(parameters, dict) else {}
        legacy_input_schema: ContentBlockParam = {
            "type": "object",
            "properties": schema.get("properties") or {},
            "required": schema.get("required") or [],
        }
        input_schema = {**parameters, **legacy_input_schema} if strict is True else legacy_input_schema

        converted_tool: ContentBlockParam = {
            "name": _to_claude_code_name(tool.name) if is_oauth_token else tool.name,
            "description": tool.description,
        }
        if supports_eager_tool_input_streaming:
            converted_tool["eager_input_streaming"] = True
        if strict is True:
            converted_tool["strict"] = True
        converted_tool["input_schema"] = input_schema
        if cache_control and index == len(tools) - 1:
            converted_tool["cache_control"] = cache_control
        converted.append(converted_tool)
    return converted


def _build_params(
    model: Model,
    context: TranscriptContext,
    is_oauth_token: bool,
    options: AnthropicOptions | None = None,
) -> MessageParam:
    """构造单次请求的 ``MessageCreateParamsStreaming`` 请求体。

    Args:
        model: 目标模型。
        context: 会话上下文。
        is_oauth_token: 是否为 Claude Code OAuth token。
        options: Anthropic 专属选项。

    Returns:
        可直接发送的请求体字典。
    """
    cache_control = _resolve_cache_control(
        model,
        options.cache_retention if options else None,
        options.env if options else None,
    ).cache_control
    compat = _get_anthropic_compat(model)
    initial_system_message = get_initial_system_message(context.messages)
    initial_system_text = get_system_message_text(initial_system_message) if initial_system_message else ""
    transformed_messages = transform_messages(context.messages, model, _normalize_tool_call_id)
    conversation_messages = transformed_messages[1:] if initial_system_message else transformed_messages
    # 原生工具变更按名称引用工具，无法表达同名重定义；而 Anthropic 会拒绝工具列表全部
    # 为 deferred 的请求，因此必须有一个初始的活跃工具来锚定后续的 deferred 工具。
    initial_tools = (initial_system_message.tools_added if initial_system_message else None) or []
    native_tool_changes = (
        compat.supports_mid_convo_system_messages
        and compat.supports_mid_convo_tool_changes
        and len(initial_tools) > 0
        and not has_tool_redefinitions(context.messages)
    )
    managed = _model_compat(model, "supports_mid_convo_effort") is True
    converted = _convert_messages(
        conversation_messages,
        is_oauth_token,
        cache_control,
        compat.allow_empty_signature,
        model.provider if managed else None,
        native_tool_changes,
    )
    active_effort = (options.effort if options and options.effort is not None else None) or "high"
    beta_features = _get_beta_features(model, context, is_oauth_token, native_tool_changes, options)
    params: MessageParam = {
        "model": model.id,
        "messages": _insert_thinking_level_messages(converted, active_effort) if managed else converted.messages,
        "max_tokens": (
            options.max_tokens
            if options and options.max_tokens is not None
            else model.max_tokens
        ),
        "stream": True,
    }
    if beta_features:
        params["betas"] = beta_features

    # OAuth token 必须带上 Claude Code 身份。
    if is_oauth_token:
        identity_block: ContentBlockParam = {
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
        }
        if cache_control:
            identity_block["cache_control"] = cache_control
        system_blocks: list[ContentBlockParam] = [identity_block]
        if initial_system_text:
            system_text_block: ContentBlockParam = {
                "type": "text",
                "text": sanitize_surrogates(initial_system_text),
            }
            if cache_control:
                system_text_block["cache_control"] = cache_control
            system_blocks.append(system_text_block)
        params["system"] = system_blocks
    elif initial_system_text:
        # 非 OAuth token 时给 system prompt 加缓存控制。
        system_block: ContentBlockParam = {
            "type": "text",
            "text": sanitize_surrogates(initial_system_text),
        }
        if cache_control:
            system_block["cache_control"] = cache_control
        params["system"] = [system_block]

    # temperature 与 extended thinking 不兼容，且 Claude Opus 4.7+ 不支持。
    if (
        options is not None
        and options.temperature is not None
        and not options.thinking_enabled
        and not managed
        and compat.supports_temperature
    ):
        params["temperature"] = options.temperature

    tool_cache_control = cache_control if compat.supports_cache_control_on_tools else None
    if native_tool_changes:
        # 初始工具保持激活，缓存断点落在最后一个上。之后的声明都是 deferred，只通过
        # ``tool_addition`` 块显现；被移除的工具仍留在声明列表中，由 ``tool_removal`` 撤回。
        # 因此请求级工具列表只会增长，工具变更时缓存前缀保持完整。
        initial_names = {tool.name for tool in initial_tools}
        later_tools = [tool for tool in get_declared_tools(context.messages) if tool.name not in initial_names]
        params["tools"] = [
            *_convert_tools(
                initial_tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                tool_cache_control,
            ),
            DEFERRED_TOOL_PLACEHOLDER,
            *[
                {**tool, "defer_loading": True}
                for tool in _convert_tools(
                    later_tools,
                    is_oauth_token,
                    compat.supports_eager_tool_input_streaming,
                    compat.supports_strict_tools,
                )
            ],
        ]
    else:
        tools = get_current_tools(context.messages)
        if tools:
            params["tools"] = _convert_tools(
                tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                tool_cache_control,
            )

    # 托管 effort 的模型始终使用 adaptive thinking，使前缀不匹配可被丢弃，而不是反复
    # 表现为 400 响应。
    if managed:
        params["thinking"] = {
            "type": "adaptive",
            "display": (options.thinking_display if options and options.thinking_display else None) or "summarized",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        }
        params["output_config"] = {"effort": "high"}
    elif model.reasoning:
        if options is not None and options.thinking_enabled:
            # 默认 "summarized"，使 Opus 4.7 与 Mythos Preview 和旧款 Claude 4 模型行为
            # 一致（后者的 API 默认值也是 "summarized"）。
            display = (options.thinking_display or "summarized")
            if _model_compat(model, "force_adaptive_thinking") is True:
                # adaptive thinking：由 Claude 决定何时思考、思考多少。
                params["thinking"] = {"type": "adaptive", "display": display}
                if options.effort:
                    params["output_config"] = {"effort": options.effort}
            else:
                # 旧模型使用基于预算的 thinking。
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens or 1024,
                    "display": display,
                }
        elif (
            options is not None
            and options.thinking_enabled is False
            and (model.thinking_level_map or {}).get("off", _UNSET) is not None
        ):
            params["thinking"] = {"type": "disabled"}

    if options is not None and options.metadata:
        user_id = options.metadata.get("user_id") if isinstance(options.metadata, dict) else None
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    if options is not None and options.tool_choice:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    allowed_fallback_models: list[AnthropicAllowedFallbackModel] | None = _model_compat(
        model, "allowed_fallback_models"
    )
    if allowed_fallback_models:
        params["fallbacks"] = [{"model": fallback.model} for fallback in allowed_fallback_models]

    return params


def _map_stop_reason(
    reason: str,
    stop_details: dict[str, Any] | None = None,
) -> tuple[StopReason, str | None]:
    """把 Anthropic 的 stop reason 映射到 pi 的 :class:`~pi_ai.types.StopReason`。

    Args:
        reason: Anthropic 返回的 stop reason。
        stop_details: ``refusal`` 等场景附带的细节。

    Returns:
        二元组 ``(stop_reason, error_message)``；``error_message`` 为 ``None`` 表示正常结束。

    Raises:
        RuntimeError: 遇到无法识别的 stop reason 时。
    """
    if reason == "end_turn":
        return StopReason.STOP, None
    if reason == "max_tokens":
        return StopReason.LENGTH, None
    if reason == "tool_use":
        return StopReason.TOOL_USE, None
    if reason == "refusal":
        explanation = None
        if isinstance(stop_details, dict):
            explanation = stop_details.get("explanation")
        return StopReason.ERROR, explanation or "The model refused to complete the request"
    if reason == "pause_turn":  # 视为 Stop 即可，之后重新提交。
        return StopReason.STOP, None
    if reason == "stop_sequence":  # 我们不提供 stop sequences，因此不会出现。
        return StopReason.STOP, None
    if reason == "sensitive":  # 内容被安全过滤器标记（SDK 类型尚未收录）。
        return StopReason.ERROR, "Provider stopped with: sensitive"
    # 优雅处理未知的 stop reason（API 可能新增取值）。
    raise RuntimeError(f"Unhandled stop reason: {reason}")


# --------------------------------------------------------------------------------------
# SSE 解码
# --------------------------------------------------------------------------------------

#: 适配器转发给 stream 消费者的 Anthropic 消息事件。
ANTHROPIC_MESSAGE_EVENTS: frozenset[str] = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)


async def _iterate_sse(response: httpx.Response, signal: AbortSignal | None = None) -> AsyncIterator[SSEEvent]:
    """迭代解码后的 SSE 事件，每次读取之间响应 abort signal。

    Args:
        response: 流式 HTTP 响应。
        signal: 中止信号；为 ``None`` 时不参与竞争。

    Yields:
        解码后的单个 :class:`SSEEvent`。

    Raises:
        AbortError: 收到中止信号时。
    """
    iterator = iter_sse_events(response.aiter_lines()).__aiter__()
    while True:
        if signal is not None:
            signal.throw_if_aborted()
        try:
            if signal is None:
                sse = await anext(iterator)
            else:
                sse = await race_with_abort_signal(anext(iterator), signal)
        except StopAsyncIteration:
            return
        yield sse


async def _iterate_anthropic_events(
    response: httpx.Response,
    signal: AbortSignal | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """从流式响应中产出解析好的 Anthropic 消息事件。

    Args:
        response: 流式 HTTP 响应。
        signal: 中止信号。

    Yields:
        解析后的 Anthropic 消息事件字典。

    Raises:
        RuntimeError: 收到 ``error`` 事件、JSON 解析失败，或流在 ``message_stop`` 之前结束时。
    """
    saw_message_start = False
    saw_message_end = False

    async for sse in _iterate_sse(response, signal):
        if sse.event == "error":
            raise RuntimeError(sse.data)

        if (sse.event or "") not in ANTHROPIC_MESSAGE_EVENTS:
            continue

        try:
            event = parse_json_with_repair(sse.data)
        except Exception as error:  # noqa: BLE001 - 会补充 provider 上下文后重新抛出
            raise RuntimeError(
                f"Could not parse Anthropic SSE event {sse.event}: {error}; "
                f"data={sse.data}; raw={'\\n'.join(sse.raw)}"
            ) from error

        event_type = event.get("type") if isinstance(event, dict) else None
        if event_type == "message_start":
            saw_message_start = True
        elif event_type == "message_stop":
            saw_message_end = True
        yield event

    if saw_message_start and not saw_message_end:
        raise RuntimeError("Anthropic stream ended before message_stop")


# --------------------------------------------------------------------------------------
# 流式传输（stream）
# --------------------------------------------------------------------------------------


def _swallow_task(task: asyncio.Task[None]) -> None:
    """消费已结束任务的异常，避免被报告为未取回（unretrieved）的异常。

    Args:
        task: 已完成（或已取消）的任务。
    """
    if task.cancelled():
        return
    task.exception()


def stream(
    model: Model,
    context: TranscriptContext,
    options: AnthropicOptions | None = None,
) -> AssistantMessageEventStream:
    """流式返回 Anthropic Messages 响应，所有失败都编码进 stream。

    Args:
        model: 目标模型（含 base URL、compat 等元数据）。
        context: 会话上下文（transcript 与工具声明）。
        options: Anthropic 专属选项；省略时使用 :class:`AnthropicOptions` 默认值。

    Returns:
        产出 assistant 消息事件的流；错误以
        ``stop_reason="error"``/``"aborted"`` 的消息编码，不向外抛异常。
    """
    event_stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(
        context, _get_anthropic_compat(model).supports_mid_convo_system_messages
    )
    current_tools = get_current_tools(normalized_context.messages)

    async def run() -> None:
        """后台执行整条 stream 流程，并把结果或错误编码进事件流。"""
        effort = options.effort if options else None
        provider_thinking_level = (
            (effort if effort is not None else "high")
            if _model_compat(model, "supports_mid_convo_effort")
            else None
        )
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason=StopReason.PENDING,
            timestamp=int(time.time() * 1000),
        )
        if provider_thinking_level is not None:
            output.provider_thinking_level = provider_thinking_level

        signal = options.signal if options else None
        timeout_ms = options.timeout_ms if options else None
        client: AnthropicClientProtocol | None = None
        owns_client = False
        response: httpx.Response | None = None

        try:
            is_oauth = False
            usage_model = model
            input_transformations: list[Any] | None = None

            if options is not None and options.client is not None:
                client = options.client
                is_oauth = False
            else:
                api_key = options.api_key if options else None
                options_headers = options.headers if options else None
                federation = _get_anthropic_federation(
                    model, api_key, options_headers, options.env if options else None
                )
                if not federation:
                    _assert_request_auth(model.provider, api_key, options_headers)

                cache_retention = _resolve_cache_retention(
                    options.cache_retention if options else None, options.env if options else None
                )
                cache_session_id = None if cache_retention == "none" else (options.session_id if options else None)

                created = _create_client(
                    model,
                    api_key,
                    options_headers,
                    options.fetch if options else None,
                    cache_session_id,
                    federation,
                    timeout_ms,
                )
                client = created.client
                is_oauth = created.is_oauth_token
                owns_client = True

            params = _build_params(model, normalized_context, is_oauth, options)
            if options is not None and options.on_payload is not None:
                next_params = await maybe_await(options.on_payload(params, model))
                if next_params is not None:
                    params = {**next_params, "stream": True}

            async def request() -> httpx.Response:
                """发起一次可重试的 Messages 请求，必要时与中止信号竞争。

                Returns:
                    原始流式 :class:`httpx.Response`。
                """
                assert client is not None
                if signal is None:
                    return await client.create_message(params, timeout_ms=timeout_ms)
                return await race_with_abort_signal(
                    client.create_message(params, timeout_ms=timeout_ms), signal
                )

            response = await retry_provider_request(
                request,
                ProviderRetryOptions(
                    max_retries=options.max_retries if options else None,
                    max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                    signal=signal,
                ),
            )
            if options is not None and options.on_response is not None:
                await maybe_await(
                    options.on_response(
                        ProviderResponse(
                            status=response.status_code,
                            headers=headers_to_record(response.headers),
                        ),
                        model,
                    )
                )
            event_stream.push(start_event(output))

            blocks = output.content
            # 沿用 TypeScript 块上的 ``index``/``partialJson`` 临时字段：每个内容位置的
            # 线上索引，以及 tool call 的原始流式 JSON。
            block_indices: list[int] = []
            partial_json: dict[int, str] = {}

            async for event in _iterate_anthropic_events(response, signal):
                if options is not None and options.on_provider_stream_event is not None:
                    await maybe_await(options.on_provider_stream_event(event, model))
                event_type = event.get("type") if isinstance(event, dict) else None

                if event_type == "message_start":
                    message = event.get("message") or {}
                    output.response_id = message.get("id")
                    transformations = message.get("input_transformations")
                    if isinstance(transformations, list):
                        input_transformations = transformations
                    response_model = message.get("model")
                    if response_model is not None and response_model != model.id:
                        output.response_model = response_model
                    fallback_cost = None
                    if response_model != model.id:
                        for fallback in _model_compat(model, "allowed_fallback_models") or []:
                            if fallback.provider == model.provider and fallback.model == response_model:
                                fallback_cost = fallback.cost
                                break
                    usage_model = (
                        replace(model, id=response_model, cost=fallback_cost) if fallback_cost else model
                    )
                    # 从 message_start 捕获初始 token 用量：即使提前中止，输入计数也必须
                    # 被保留下来。
                    usage = message.get("usage") or {}
                    output.usage.input = usage.get("input_tokens") or 0
                    output.usage.output = usage.get("output_tokens") or 0
                    output.usage.cache_read = usage.get("cache_read_input_tokens") or 0
                    output.usage.cache_write = usage.get("cache_creation_input_tokens") or 0
                    cache_creation = usage.get("cache_creation")
                    output.usage.cache_write_1h = (
                        (cache_creation or {}).get("ephemeral_1h_input_tokens") or 0
                        if isinstance(cache_creation, dict)
                        else 0
                    )
                    # Anthropic 不提供 total_tokens，由各分量累加得出。
                    output.usage.total_tokens = (
                        output.usage.input
                        + output.usage.output
                        + output.usage.cache_read
                        + output.usage.cache_write
                    )
                    calculate_cost(usage_model, output.usage)

                elif event_type == "content_block_start":
                    content_block = event.get("content_block") or {}
                    content_block_type = content_block.get("type")
                    if content_block_type == "fallback":
                        if output.content:
                            raise RuntimeError("Anthropic performed an unsupported mid-output model fallback")
                        continue
                    if content_block_type == "text":
                        output.content.append(
                            TextContent(type="text", text=content_block.get("text") or "")
                        )
                        block_indices.append(event.get("index"))
                        event_stream.push(text_start_event(len(output.content) - 1, output))
                    elif content_block_type == "thinking":
                        output.content.append(
                            ThinkingContent(
                                type="thinking",
                                thinking=content_block.get("thinking") or "",
                                thinking_signature=content_block.get("signature") or "",
                            )
                        )
                        block_indices.append(event.get("index"))
                        event_stream.push(thinking_start_event(len(output.content) - 1, output))
                    elif content_block_type == "redacted_thinking":
                        output.content.append(
                            ThinkingContent(
                                type="thinking",
                                thinking="[Reasoning redacted]",
                                thinking_signature=content_block.get("data"),
                                redacted=True,
                            )
                        )
                        block_indices.append(event.get("index"))
                        event_stream.push(thinking_start_event(len(output.content) - 1, output))
                    elif content_block_type == "tool_use":
                        name = content_block.get("name") or ""
                        output.content.append(
                            ToolCall(
                                type="toolCall",
                                id=content_block.get("id") or "",
                                name=_from_claude_code_name(name, current_tools) if is_oauth else name,
                                arguments=content_block.get("input") or {},
                            )
                        )
                        partial_json[len(output.content) - 1] = ""
                        block_indices.append(event.get("index"))
                        event_stream.push(tool_call_start_event(len(output.content) - 1, output))

                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        index = _find_block_index(block_indices, event.get("index"))
                        block = output.content[index] if index >= 0 else None
                        if isinstance(block, TextContent):
                            text = delta.get("text") or ""
                            block.text = block.text + text
                            event_stream.push(text_delta_event(index, text, output))
                    elif delta_type == "thinking_delta":
                        index = _find_block_index(block_indices, event.get("index"))
                        block = output.content[index] if index >= 0 else None
                        if isinstance(block, ThinkingContent):
                            thinking = delta.get("thinking") or ""
                            block.thinking = block.thinking + thinking
                            event_stream.push(thinking_delta_event(index, thinking, output))
                    elif delta_type == "input_json_delta":
                        index = _find_block_index(block_indices, event.get("index"))
                        block = output.content[index] if index >= 0 else None
                        if isinstance(block, ToolCall):
                            partial = delta.get("partial_json") or ""
                            accumulated = partial_json.get(index, "") + partial
                            partial_json[index] = accumulated
                            block.arguments = parse_streaming_json(accumulated)
                            event_stream.push(tool_call_delta_event(index, partial, output))
                    elif delta_type == "signature_delta":
                        index = _find_block_index(block_indices, event.get("index"))
                        block = output.content[index] if index >= 0 else None
                        if isinstance(block, ThinkingContent):
                            block.thinking_signature = (block.thinking_signature or "") + (
                                delta.get("signature") or ""
                            )

                elif event_type == "content_block_stop":
                    index = _find_block_index(block_indices, event.get("index"))
                    if index >= 0:
                        # 对应 TS 的 ``delete block.index``：该块不再匹配。
                        block_indices[index] = -1
                        block = output.content[index]
                        if isinstance(block, TextContent):
                            event_stream.push(text_end_event(index, block.text, output))
                        elif isinstance(block, ThinkingContent):
                            event_stream.push(thinking_end_event(index, block.thinking, output))
                        elif isinstance(block, ToolCall):
                            block.arguments = parse_streaming_json(partial_json.pop(index, ""))
                            event_stream.push(tool_call_end_event(index, block, output))

                elif event_type == "message_delta":
                    transformations = event.get("input_transformations")
                    if isinstance(transformations, list):
                        input_transformations = transformations
                    delta = event.get("delta") or {}
                    stop_reason = delta.get("stop_reason")
                    if stop_reason:
                        output.raw_stop_reason = stop_reason
                        mapped_reason, error_message = _map_stop_reason(
                            stop_reason, delta.get("stop_details")
                        )
                        output.stop_reason = mapped_reason
                        if error_message:
                            output.error_message = error_message
                    # 仅在字段存在（非 null）时更新用量：代理在 message_delta 中省略
                    # input_tokens 时，保留来自 message_start 的值。
                    usage = event.get("usage")
                    if usage:
                        if usage.get("input_tokens") is not None:
                            output.usage.input = usage["input_tokens"]
                        if usage.get("output_tokens") is not None:
                            output.usage.output = usage["output_tokens"]
                        if usage.get("cache_read_input_tokens") is not None:
                            output.usage.cache_read = usage["cache_read_input_tokens"]
                        if usage.get("cache_creation_input_tokens") is not None:
                            output.usage.cache_write = usage["cache_creation_input_tokens"]
                        # Vercel AI Gateway 会在 delta 中附带 TTL 明细，尽管 SDK 只在
                        # message_start 上为其定义了类型。
                        cache_creation = usage.get("cache_creation")
                        if (
                            isinstance(cache_creation, dict)
                            and cache_creation.get("ephemeral_1h_input_tokens") is not None
                        ):
                            output.usage.cache_write_1h = cache_creation["ephemeral_1h_input_tokens"]
                        # Anthropic 将 reasoning token 报告为 output token 的子集。
                        output_details = usage.get("output_tokens_details")
                        thinking_tokens = (
                            output_details.get("thinking_tokens")
                            if isinstance(output_details, dict)
                            else None
                        )
                        if thinking_tokens is not None:
                            output.usage.reasoning = thinking_tokens
                    # Anthropic 不提供 total_tokens，由各分量累加得出。
                    output.usage.total_tokens = (
                        output.usage.input
                        + output.usage.output
                        + output.usage.cache_read
                        + output.usage.cache_write
                    )
                    calculate_cost(usage_model, output.usage)

            if signal is not None and signal.aborted:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Anthropic stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")
            if input_transformations:
                append_assistant_message_diagnostic(
                    output,
                    AssistantMessageDiagnostic(
                        type="anthropic_input_transformations",
                        timestamp=int(time.time() * 1000),
                        details={
                            "transformations": [
                                _summarize_transformation(transformation)
                                for transformation in input_transformations
                            ]
                        },
                    ),
                )

            event_stream.push(done_event(output.stop_reason, output))
            event_stream.end()
        except Exception as error:  # noqa: BLE001 - 会编码进 stream 协议
            output.stop_reason = (
                StopReason.ABORTED if signal is not None and signal.aborted else StopReason.ERROR
            )
            output.error_message = str(error)
            event_stream.push(error_event(output.stop_reason, output))
            event_stream.end()
        finally:
            await _close_stream_resources(response, client if owns_client else None)

    task = asyncio.ensure_future(run())
    task.add_done_callback(_swallow_task)

    return event_stream


def _find_block_index(block_indices: list[int], provider_index: Any) -> int:
    """按事件的线上索引查找本地块下标。

    等价于 ``blocks.findIndex(block => block.index === event.index)``。

    Args:
        block_indices: 每个内容位置对应的线上索引。
        provider_index: 事件携带的线上索引。

    Returns:
        匹配的本地下标；找不到时返回 ``-1``。
    """
    try:
        return block_indices.index(provider_index)
    except ValueError:
        return -1


def _summarize_transformation(transformation: Any) -> dict[str, Any]:
    """提取一条 input transformation，按 ``?? undefined`` 的语义丢弃未设置的字段。

    Args:
        transformation: 原始 transformation 对象。

    Returns:
        仅保留 ``type``/``path``/``reason`` 中已设置字段的摘要字典。
    """
    summary: dict[str, Any] = {}
    if isinstance(transformation, dict):
        for key in ("type", "path", "reason"):
            value = transformation.get(key)
            if value is not None:
                summary[key] = value
    return summary


async def _close_stream_resources(
    response: httpx.Response | None,
    client: AnthropicClientProtocol | None,
) -> None:
    """释放 response，以及（若由本方创建）HTTP client；绝不抛出异常。

    Args:
        response: 流式响应；为 ``None`` 时跳过。
        client: 本适配器创建的 client；为 ``None`` 时跳过。
    """
    if response is not None:
        try:
            await response.aclose()
        except Exception:  # noqa: BLE001 - 清理不得破坏 stream 协议
            pass
    if client is not None:
        try:
            await client.aclose()  # type: ignore[attr-defined] - client 由本适配器创建，具备 aclose
        except Exception:  # noqa: BLE001 - 清理不得破坏 stream 协议
            pass


# --------------------------------------------------------------------------------------
# 简单 stream
# --------------------------------------------------------------------------------------


def _anthropic_options(base: StreamOptions, **overrides: Any) -> AnthropicOptions:
    """把 stream 选项对象展开进 :class:`AnthropicOptions`（对应 TS 的对象展开）。

    Args:
        base: 基础 stream 选项。
        **overrides: 需要覆盖的字段。

    Returns:
        合并后的 :class:`AnthropicOptions`。
    """
    values = {dataclass_field.name: getattr(base, dataclass_field.name) for dataclass_field in fields(StreamOptions)}
    for dataclass_field in fields(AnthropicOptions):
        if dataclass_field.name not in values:
            values[dataclass_field.name] = getattr(base, dataclass_field.name, None)
    values.update(overrides)
    return AnthropicOptions(**values)


def _map_thinking_level_to_effort(model: Model, level: str | None) -> str:
    """把 pi 的 thinking level 映射为 adaptive thinking 使用的 Anthropic effort。

    Args:
        model: 目标模型，可提供 ``thinking_level_map`` 覆盖。
        level: pi 的 thinking level；为 ``None`` 时回退到 ``high``。

    Returns:
        Anthropic effort 字符串。
    """
    mapped = (model.thinking_level_map or {}).get(level) if level else None
    if isinstance(mapped, str):
        return mapped
    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    if level == "high":
        return "high"
    return "high"


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """``anthropic-messages`` adapter 的简版选项入口。

    Args:
        model: 目标模型。
        context: 会话上下文。
        options: 简版流式选项。

    Returns:
        产出 assistant 消息事件的流。
    """
    api_key = options.api_key if options else None
    headers = options.headers if options else None
    if not _get_anthropic_federation(model, api_key, headers, options.env if options else None):
        _assert_request_auth(model.provider, api_key, headers)

    base = _anthropic_options(
        build_base_options(model, context, options, api_key),
        tool_choice=options.tool_choice if options else None,
    )
    if not (options.reasoning if options else None):
        return stream(model, context, _anthropic_options(base, thinking_enabled=False))

    # 自适应 thinking 模型：使用 effort 等级。
    # 旧模型：使用基于预算的 thinking。
    if _model_compat(model, "force_adaptive_thinking") is True:
        effort = _map_thinking_level_to_effort(model, options.reasoning if options else None)
        return stream(
            model, context, _anthropic_options(base, thinking_enabled=True, effort=effort)
        )

    # undefined 表示调用方未指定输出上限，让辅助函数使用模型上限。这里不要强制转成 0，
    # 否则 thinking 预算会变成整个 max_tokens 值。
    adjusted_max_tokens, adjusted_thinking_budget = adjust_max_tokens_for_thinking(
        base.max_tokens,
        model.max_tokens,
        options.reasoning if options else None,  # type: ignore[arg-type] - reasoning 为字符串字面量
        options.thinking_budgets if options else None,
    )

    max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens)

    return stream(
        model,
        context,
        _anthropic_options(
            base,
            max_tokens=max_tokens,
            thinking_enabled=True,
            thinking_budget_tokens=min(adjusted_thinking_budget, max(0, max_tokens - 1024)),
        ),
    )


# --------------------------------------------------------------------------------------
# 惰性 provider 入口
# --------------------------------------------------------------------------------------


def anthropic_messages_api() -> Any:
    """Anthropic Messages 惰性加载的 :class:`~pi_ai.types.ProviderStreams`。"""
    from .lazy import lazy_api, lazy_load

    return lazy_api(lazy_load("pi_ai.api.anthropic_messages"))
