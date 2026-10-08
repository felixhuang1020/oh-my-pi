"""OpenAI Responses API 流式响应。

移植自 ``packages/ai/src/api/openai-responses.ts``。该适配器服务 OpenAI 及
OpenAI 兼容 Responses 端点的 ``openai-responses`` API；与 provider 无关的转换与
stream 机制位于 :mod:`pi_ai.api.openai_responses_shared`。

TypeScript 原版通过 ``openai`` SDK 访问端点；本移植用 :mod:`httpx` 直接发 HTTP，
因此解析后的线上事件是普通 ``dict``。
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..models import clamp_thinking_level
from ..types import (
    AssistantMessage,
    CacheRetention,
    Model,
    ModelCompat,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from ..utils.async_utils import maybe_await
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.events import done_event, error_event, start_event
from ..utils.headers import headers_to_record
from ..utils.http import build_headers, create_client
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    ProviderHTTPError,
    convert_responses_messages,
    convert_responses_tools,
    iter_stream_payloads,
    join_url,
    process_responses_stream,
    send_streaming_request,
)
from .simple_options import build_base_options

__all__ = [
    "CHATGPT_USAGE_URL",
    "OPENAI_RESPONSES_MIN_OUTPUT_TOKENS",
    "OPENAI_TOOL_CALL_PROVIDERS",
    "OpenAIResponsesOptions",
    "ProviderHTTPError",
    "openai_responses_api",
    "stream",
    "stream_simple",
]

OPENAI_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})
#: OpenAI Responses 拒绝低于 16 的 ``max_output_tokens``。
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16
CHATGPT_USAGE_URL = "https://chatgpt.com/settings/usage"

#: 模块私有哨兵，替代 JavaScript 的 ``undefined``。
_MISSING = object()


def _now_ms() -> int:
    """返回当前 Unix 时间戳（毫秒），用于消息时间标记。"""
    return int(time.time() * 1000)


def _compat_or(model: Model, attribute: str, fallback: Any) -> Any:
    """读取模型的兼容性开关，模型未声明时回退到默认值。

    Args:
        model: 目标模型。
        attribute: ``ModelCompat`` 上的属性名。
        fallback: 模型未声明该属性时使用的默认值。

    Returns:
        模型声明的值；属性缺失或为 ``None`` 时返回 ``fallback``。
    """
    compat = model.compat
    value = getattr(compat, attribute, None) if compat is not None else None
    return fallback if value is None else value


def _level_map_get(model: Model, key: str) -> Any:
    """按 pi thinking 等级名查模型的 ``thinking_level_map``。

    Args:
        model: 目标模型。
        key: pi 侧等级名。

    Returns:
        映射值；映射不存在或键不存在时返回模块私有哨兵 ``_MISSING``。
    """
    level_map = model.thinking_level_map
    if not level_map or key not in level_map:
        return _MISSING
    return level_map[key]


def _mapped_effort(model: Model, key: str, fallback: Any) -> Any:
    """解析映射后的 reasoning effort，映射缺失时回退。

    Args:
        model: 目标模型。
        key: pi 侧等级名。
        fallback: 映射缺失或映射值为 ``None`` 时的回退值。

    Returns:
        映射后的 effort 或 ``fallback``。
    """
    value = _level_map_get(model, key)
    return fallback if value is _MISSING or value is None else value


def _off_is_not_null(model: Model) -> bool:
    """判断 ``off`` 等级是否需要显式发送（映射值不是 JSON null）。

    Args:
        model: 目标模型。

    Returns:
        当 ``off`` 未被显式映射为 ``None`` 时返回 ``True``。
    """
    level_map = model.thinking_level_map or {}
    return not ("off" in level_map and level_map["off"] is None)


def _extend_options(base: StreamOptions, option_type: type, **extras: Any) -> Any:
    """在 ``StreamOptions`` 字段基础上扩展字段，构造 provider 选项实例。

    Args:
        base: 提供基础字段的 ``StreamOptions`` 实例。
        option_type: 目标选项 dataclass 类型。
        **extras: 需要覆盖或新增的字段。

    Returns:
        新构造的 ``option_type`` 实例。
    """
    values = {
        dataclass_field.name: getattr(base, dataclass_field.name)
        for dataclass_field in dataclasses.fields(base)
    }
    values.update(extras)
    return option_type(**values)


# =============================================================================
# 选项
# =============================================================================


@dataclass
class OpenAIResponsesOptions(StreamOptions):
    """``openai-responses`` 适配器接受的选项。

    Attributes:
        reasoning_effort: reasoning effort 覆盖值。
        reasoning_summary: reasoning 摘要模式。
        service_tier: OpenAI 服务层级。
        tool_choice: 原样透传给端点的 tool choice。
    """

    reasoning_effort: str | None = field(default=None, metadata={"alias": "reasoningEffort"})
    reasoning_summary: str | None = field(default=None, metadata={"alias": "reasoningSummary"})
    service_tier: str | None = field(default=None, metadata={"alias": "serviceTier"})
    tool_choice: Any = None


# =============================================================================
# 小型辅助函数
# =============================================================================


def _is_chatgpt_sign_in(model: Model, api_key: str | None) -> bool:
    """直接发往 OpenAI 的非 ``sk-`` 凭据是 ChatGPT 访问令牌。

    Args:
        model: 目标模型。
        api_key: 待判断的凭据。

    Returns:
        符合 ChatGPT 登录特征时返回 ``True``。
    """
    return (
        model.provider == "openai"
        and model.base_url == "https://api.openai.com/v1"
        and api_key is not None
        and not api_key.startswith("sk-")
    )


def _has_header(headers: ProviderHeaders | None, name: str) -> bool:
    """不区分大小写地判断 headers 中是否存在非空值。

    Args:
        headers: 请求头映射。
        name: 头名称。

    Returns:
        存在同名且值非空的头时返回 ``True``。
    """
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value is not None and value.strip():
            return True
    return False


def _get_client_api_key(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> str:
    """取出或推导客户端 API key。

    Args:
        provider: provider 名称，用于错误消息。
        api_key: 显式传入的 API key。
        headers: 可能已携带鉴权头的请求头。

    Returns:
        实际使用的 API key；仅有鉴权头时返回占位值 ``"unused"``。

    Raises:
        RuntimeError: 既无 API key 也无鉴权头时。
    """
    if api_key:
        return api_key
    if _has_header(headers, "authorization") or _has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise RuntimeError(f"No API key for provider: {provider}")


def _detect_session_affinity_format(model: Model) -> str:
    """按 provider 与 base URL 推断会话粘性请求头格式。

    Args:
        model: 目标模型。

    Returns:
        ``"openrouter"`` 或 ``"openai"``。
    """
    return "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.base_url else "openai"


def _resolve_cache_retention(
    cache_retention: CacheRetention | str | None = None, env: ProviderEnv | None = None
) -> CacheRetention | str:
    """解析缓存保留策略，默认 ``"short"``（``PI_CACHE_RETENTION`` 优先）。

    Args:
        cache_retention: 显式指定的保留策略。
        env: provider 环境变量映射。

    Returns:
        解析后的保留策略。
    """
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _get_compat(model: Model) -> ModelCompat:
    """按模型声明补齐 ``ModelCompat`` 兼容性开关。

    Args:
        model: 目标模型。

    Returns:
        填充默认值后的 ``ModelCompat``。
    """
    return ModelCompat(
        supports_developer_role=_compat_or(model, "supports_developer_role", True),
        supports_mid_convo_system_messages=_compat_or(model, "supports_mid_convo_system_messages", False),
        session_affinity_format=_compat_or(
            model, "session_affinity_format", _detect_session_affinity_format(model)
        ),
        supports_long_cache_retention=_compat_or(model, "supports_long_cache_retention", True),
        supports_strict_mode=_compat_or(model, "supports_strict_mode", False),
        supports_openai_grammar_tools=_compat_or(model, "supports_openai_grammar_tools", False),
        supports_additional_tools=_compat_or(model, "supports_additional_tools", False),
        supports_tool_search=_compat_or(model, "supports_tool_search", False),
        supports_explicit_prompt_cache_mode=_compat_or(
            model, "supports_explicit_prompt_cache_mode", False
        ),
        supports_max_output_tokens=_compat_or(model, "supports_max_output_tokens", True),
    )


def _get_prompt_cache_retention(compat: ModelCompat, cache_retention: CacheRetention | str) -> str | None:
    """计算 ``prompt_cache_retention`` 字段值。

    Args:
        compat: 模型的兼容性开关。
        cache_retention: 解析后的缓存保留策略。

    Returns:
        ``"24h"``；不满足长期缓存条件时返回 ``None``。
    """
    return (
        "24h"
        if cache_retention == "long"
        and compat.supports_long_cache_retention
        and not compat.supports_explicit_prompt_cache_mode
        else None
    )


def _get_prompt_cache_options(
    compat: ModelCompat, cache_retention: CacheRetention | str
) -> dict[str, Any] | None:
    """计算显式 prompt cache 模式下的 ``prompt_cache_options``。

    Args:
        compat: 模型的兼容性开关。
        cache_retention: 解析后的缓存保留策略。

    Returns:
        选项字典；模型不支持显式模式或策略不适用时返回 ``None``。
    """
    if not compat.supports_explicit_prompt_cache_mode:
        return None
    if cache_retention == "none":
        return {"mode": "explicit"}
    if cache_retention == "long" and compat.supports_long_cache_retention:
        return {"ttl": "30m"}
    return None


# =============================================================================
# 客户端与请求参数
# =============================================================================


def _create_client(
    model: Model,
    api_key: str,
    options_headers: ProviderHeaders | None = None,
    fetch: Any = None,
    session_id: str | None = None,
    compat: ModelCompat | None = None,
    timeout_ms: int | None = None,
) -> httpx.AsyncClient:
    """创建带默认头与会话粘性头的 ``httpx.AsyncClient``。

    Args:
        model: 目标模型。
        api_key: 已解析的 API key。
        options_headers: 调用方 options 中的请求头。
        fetch: 自定义 fetch 实现。
        session_id: 会话 ID，用于会话粘性头。
        compat: 已解析的兼容性开关；为 ``None`` 时按模型重新解析。
        timeout_ms: 请求超时（毫秒）。

    Returns:
        配置完成的 ``httpx.AsyncClient``。
    """
    if compat is None:
        compat = _get_compat(model)

    default_headers: dict[str, str | None] = {"User-Agent": get_pi_user_agent()}
    if model.headers:
        default_headers.update(model.headers)

    if session_id:
        if compat.session_affinity_format == "openrouter":
            default_headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                default_headers["session_id"] = session_id
            default_headers["x-client-request-id"] = session_id

    # 最后合并 options headers，使其能够覆盖默认值。
    if options_headers:
        default_headers.update(options_headers)

    headers = build_headers({"Authorization": f"Bearer {api_key}"}, default_headers)
    return create_client(headers=headers, timeout_ms=timeout_ms, fetch=fetch)


def _build_params(
    model: Model,
    context: TranscriptContext,
    options: OpenAIResponsesOptions | None,
    compat: ModelCompat | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """构建 Responses 请求体。

    Args:
        model: 目标模型。
        context: 已规范化的会话上下文。
        options: provider 专属选项。
        compat: 已解析的兼容性开关；为 ``None`` 时按模型重新解析。
        grammar_tool_input_properties: 预计算的 grammar 工具输入属性。

    Returns:
        待发送的请求体字典。
    """
    if compat is None:
        compat = _get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), compat.supports_openai_grammar_tools
        )

    transcript_tools = resolve_transcript_tools(
        context.messages, bool(compat.supports_additional_tools) or bool(compat.supports_tool_search)
    )
    tool_options = ConvertResponsesToolsOptions(
        supports_strict_mode=compat.supports_strict_mode,
        supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
    )
    messages = convert_responses_messages(
        model,
        context,
        OPENAI_TOOL_CALL_PROVIDERS,
        ConvertResponsesMessagesOptions(
            grammar_tool_input_properties=grammar_tool_input_properties,
            supports_mid_convo_system_messages=compat.supports_mid_convo_system_messages,
            supports_additional_tools=compat.supports_additional_tools,
            supports_tool_search=compat.supports_tool_search,
            tool_options=tool_options,
        ),
    )

    cache_retention = _resolve_cache_retention(
        options.cache_retention if options else None, options.env if options else None
    )
    # 使用 ChatGPT 登录时会拒绝这些请求字段。
    omit_unsupported_fields = _is_chatgpt_sign_in(model, options.api_key if options else None)
    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }
    if cache_retention != "none":
        prompt_cache_key = clamp_openai_prompt_cache_key(options.session_id if options else None)
        if prompt_cache_key is not None:
            params["prompt_cache_key"] = prompt_cache_key
    if not omit_unsupported_fields:
        prompt_cache_retention = _get_prompt_cache_retention(compat, cache_retention)
        if prompt_cache_retention is not None:
            params["prompt_cache_retention"] = prompt_cache_retention
        prompt_cache_options = _get_prompt_cache_options(compat, cache_retention)
        if prompt_cache_options is not None:
            params["prompt_cache_options"] = prompt_cache_options

    if options and options.max_tokens and compat.supports_max_output_tokens and not omit_unsupported_fields:
        params["max_output_tokens"] = max(options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options and options.temperature is not None and not omit_unsupported_fields:
        params["temperature"] = options.temperature

    if options and options.service_tier is not None:
        params["service_tier"] = options.service_tier

    if len(transcript_tools.request_tools) > 0:
        params["tools"] = convert_responses_tools(transcript_tools.request_tools, tool_options)

    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    if model.reasoning:
        if options and (options.reasoning_effort or options.reasoning_summary):
            if options.reasoning_effort:
                effort = _mapped_effort(model, options.reasoning_effort, options.reasoning_effort)
            else:
                effort = "medium"
            params["reasoning"] = {"effort": effort, "summary": options.reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif _off_is_not_null(model):
            params["reasoning"] = {"effort": _mapped_effort(model, "off", "none")}

    # 放在最后，让自定义键覆盖具名请求字段；请求级键覆盖模型默认值。
    if model.sampling_params:
        params.update(model.sampling_params)
    if options and options.sampling_params:
        params.update(options.sampling_params)

    return params


def _get_service_tier_cost_multiplier(model: Model, service_tier: str | None) -> float:
    """返回服务层级对应的成本倍率。

    Args:
        model: 目标模型。
        service_tier: 服务层级。

    Returns:
        成本倍率；``flex`` 为 0.5，``priority``/``fast`` 为 2（``gpt-5.5`` 为 2.5）。
    """
    if service_tier == "flex":
        return 0.5
    if service_tier in ("priority", "fast"):
        return 2.5 if model.id == "gpt-5.5" else 2
    return 1


def _apply_service_tier_pricing(usage: Usage, service_tier: str | None, model: Model) -> None:
    """按服务层级倍率缩放 usage 成本。

    Args:
        usage: 待调整的用量记录。
        service_tier: 服务层级。
        model: 目标模型。
    """
    multiplier = _get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )


# =============================================================================
# 流式响应
# =============================================================================


def stream(
    model: Model,
    context: TranscriptContext,
    options: OpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """对 OpenAI Responses API 发起流式生成。

    Args:
        model: 目标模型。
        context: 会话上下文。
        options: provider 专属选项。

    Returns:
        已推入事件、可供消费的 :class:`AssistantMessageEventStream`。
    """
    output_stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(
        context, _get_compat(model).supports_mid_convo_system_messages
    )

    async def run() -> None:
        """执行请求、消费 stream 事件并把结果推入输出流。"""
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=_now_ms(),
        )
        signal = options.signal if options else None
        client: httpx.AsyncClient | None = None
        response: httpx.Response | None = None
        try:
            api_key = _get_client_api_key(
                model.provider,
                options.api_key if options else None,
                options.headers if options else None,
            )
            cache_retention = _resolve_cache_retention(
                options.cache_retention if options else None, options.env if options else None
            )
            cache_session_id = None if cache_retention == "none" else (options.session_id if options else None)
            compat = _get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages), compat.supports_openai_grammar_tools
            )
            client = _create_client(
                model,
                api_key,
                options.headers if options else None,
                options.fetch if options else None,
                cache_session_id,
                compat,
                options.timeout_ms if options else None,
            )
            params = _build_params(model, normalized_context, options, compat, grammar_tool_input_properties)
            if options and options.on_payload:
                next_params = await maybe_await(options.on_payload(params, model))
                if next_params is not None:
                    params = next_params

            url = join_url(model.base_url, "/responses")
            response = await retry_provider_request(
                lambda: send_streaming_request(client, url, params, signal),
                ProviderRetryOptions(
                    max_retries=options.max_retries if options else None,
                    max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                    signal=signal,
                ),
            )
            if options and options.on_response:
                await maybe_await(
                    options.on_response(
                        ProviderResponse(
                            status=response.status_code, headers=headers_to_record(response.headers)
                        ),
                        model,
                    )
                )
            output_stream.push(start_event(output))

            await process_responses_stream(
                iter_stream_payloads(response, signal),
                output,
                output_stream,
                model,
                OpenAIResponsesStreamOptions(
                    on_provider_stream_event=options.on_provider_stream_event if options else None,
                    service_tier=options.service_tier if options else None,
                    grammar_tool_input_properties=grammar_tool_input_properties,
                    apply_service_tier_pricing=lambda usage, service_tier: _apply_service_tier_pricing(
                        usage, service_tier, model
                    ),
                ),
            )

            if signal is not None and signal.aborted:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("OpenAI Responses stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            output_stream.push(done_event(output.stop_reason, output))
            output_stream.end()
        except Exception as error:  # noqa: BLE001 - 编码进 stream 协议
            for block in output.content:
                # 流式临时缓冲仅在解析期间使用，绝不持久化。
                for attribute in ("index", "partial_json", "custom_input"):
                    if hasattr(block, attribute):
                        delattr(block, attribute)
            output.stop_reason = "aborted" if signal is not None and signal.aborted else "error"
            error_message = format_provider_error(
                normalize_provider_error(error),
                f"{'OpenAI' if model.provider == 'openai' else model.provider} API error",
            )
            # 使用 ChatGPT 登录时，配额与订阅的用量上限和其他应用共享。
            output.error_message = (
                f"{error_message}\nCheck your ChatGPT usage: {CHATGPT_USAGE_URL}"
                if "subscription_sharing_usage_limit_exceeded" in error_message
                else error_message
            )
            output_stream.push(error_event(output.stop_reason, output))
            output_stream.end()
        finally:
            if response is not None:
                try:
                    await response.aclose()
                except Exception:  # noqa: BLE001 - 关闭失败不得掩盖真实结果
                    pass
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001 - 关闭失败不得掩盖真实结果
                    pass

    task = asyncio.ensure_future(run())
    task.add_done_callback(_swallow)
    return output_stream


def _swallow(task: asyncio.Future[Any]) -> None:
    """取走已结束任务的异常，避免被报告为未处理异常。

    Args:
        task: 已结束的异步任务。
    """
    if task.cancelled():
        return
    task.exception()


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """``openai-responses`` 适配器的简单选项入口。

    Args:
        model: 目标模型。
        context: 会话上下文。
        options: provider 中立的简单选项。

    Returns:
        已推入事件、可供消费的 :class:`AssistantMessageEventStream`。
    """
    _get_client_api_key(
        model.provider,
        options.api_key if options else None,
        options.headers if options else None,
    )

    base = build_base_options(model, context, options, options.api_key if options else None)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    provider_options = _extend_options(
        base,
        OpenAIResponsesOptions,
        tool_choice=options.tool_choice if options else None,
        reasoning_effort=reasoning_effort,
    )
    return stream(model, context, provider_options)


def openai_responses_api() -> Any:
    """``openai-responses`` 实现的延迟加载 provider-streams 包装。"""
    from .lazy import lazy_api, lazy_load

    return lazy_api(lazy_load("pi_ai.api.openai_responses"))
