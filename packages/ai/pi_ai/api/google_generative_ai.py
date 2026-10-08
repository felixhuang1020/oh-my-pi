"""Google Generative AI（Gemini API）聊天适配器。

移植自 ``packages/ai/src/api/google-generative-ai.ts``。

TypeScript 原版通过 ``@google/genai`` SDK 驱动该 API（``client.models.generateContentStream``）。
本移植用 :mod:`httpx` 直接发起同样的 ``:streamGenerateContent?alt=sse`` REST 调用，
因此 ``options.fetch``、``options.headers``、``options.env`` 和 ``options.timeout_ms``
均被原样遵循，``onPayload`` 收到的请求形态仍是 ``{model, contents, config}``。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, fields
from typing import Any

from ..models import calculate_cost, clamp_thinking_level
from ..types import (
    AssistantMessage,
    Model,
    ProviderHeaders,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ToolCall,
    TranscriptContext,
    Usage,
)
from ..utils.async_utils import maybe_await
from ..utils.error_body import format_provider_error, normalize_provider_error
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
from ..utils.headers import provider_headers_to_record
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.text import get_system_message_text
from ..utils.transcript import collapse_system_messages, get_current_tools, get_initial_system_message
from .google_shared import (
    GOOGLE_GENERATIVE_AI_API,
    GoogleApiThinkingLevel,
    GoogleThinkingOptions,
    convert_messages,
    convert_tools,
    get_disabled_google_thinking_config,
    is_thinking_part,
    map_stop_reason,
    open_google_generate_content_stream,
    resolve_google_function_calling_mode,
    resolve_google_thinking_level,
    retain_thought_signature,
    retry_google_request,
    supports_google_strict_tool_sampling,
    to_google_sdk_thinking_level,
    to_google_thinking_level,
    uses_google_thinking_level,
)
from .simple_options import build_base_options

__all__ = [
    "GoogleOptions",
    "google_generative_ai_api",
    "stream",
    "stream_simple",
]

#: Gemini API 默认端点（``apiVersion`` 已包含在目录 base URL 中）。
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class GoogleOptions(StreamOptions):
    """Google Generative AI API 的 provider 专属选项。

    Attributes:
        tool_choice: 调用方指定的 tool choice。
        thinking: 按请求的 thinking 控制。
    """

    tool_choice: str | None = None
    thinking: GoogleThinkingOptions | None = None


# 用于生成唯一 tool call ID 的计数器
_tool_call_counter = 0


def _swallow(task: asyncio.Task[None]) -> None:
    """取走已结束任务的异常，避免被报告为未处理异常。

    Args:
        task: 已结束的异步任务。
    """
    if task.cancelled():
        return
    task.exception()


def _now_ms() -> int:
    """返回当前 Unix 时间戳（毫秒），用于 tool call ID 与消息时间标记。"""
    return int(time.time() * 1000)


def _json_stringify(value: Any) -> str:
    """toolcall delta 载荷的 ``JSON.stringify`` 等价实现。

    Args:
        value: 待序列化的值。

    Returns:
        紧凑且不转义非 ASCII 字符的 JSON 字符串。
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _create_output(model: Model) -> AssistantMessage:
    """创建待填充的 assistant 消息。

    Args:
        model: 目标模型。

    Returns:
        ``stop_reason`` 为 ``PENDING`` 的空消息。
    """
    return AssistantMessage(
        role="assistant",
        content=[],
        api=GOOGLE_GENERATIVE_AI_API,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason=StopReason.PENDING,
        timestamp=_now_ms(),
    )


def stream(
    model: Model,
    context: TranscriptContext,
    options: GoogleOptions | None = None,
) -> AssistantMessageEventStream:
    """流式获取 Gemini API 响应。

    Args:
        model: 目标模型。
        context: 会话上下文。
        options: provider 专属选项。

    Returns:
        已推入事件、可供消费的 :class:`AssistantMessageEventStream`。
    """
    stream = AssistantMessageEventStream()
    normalized_context = collapse_system_messages(context)

    async def run() -> None:
        """执行请求、消费 SSE 块并把事件推入输出流。"""
        output = _create_output(model)
        google_stream = None
        try:
            api_key = options.api_key if options is not None else None
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")
            params = _build_params(model, normalized_context, options)
            if options is not None and options.on_payload is not None:
                next_params = await maybe_await(options.on_payload(params, model))
                if next_params is not None:
                    params = next_params
            url = _request_url(model)
            headers = _request_headers(model, api_key, options)
            google_stream = await retry_google_request(
                lambda: open_google_generate_content_stream(
                    url=url, headers=headers, params=params, options=options
                ),
                options,
            )

            stream.push(start_event(output))
            current_block: TextContent | ThinkingContent | None = None
            blocks = output.content

            def block_index() -> int:
                """返回当前输出内容块的最后一个下标。"""
                return len(blocks) - 1

            async for chunk in google_stream:
                if options is not None and options.on_provider_stream_event is not None:
                    await maybe_await(options.on_provider_stream_event(chunk, model))
                # @google/genai 将 GenerateContentResponse.responseId 文档化为仅输出字段，
                # 用于标识每次响应。保留首个非空值。
                if not output.response_id:
                    output.response_id = chunk.get("responseId")
                candidates = chunk.get("candidates") or []
                candidate = candidates[0] if candidates else None
                content = candidate.get("content") if candidate is not None else None
                parts = content.get("parts") if content is not None else None
                if parts:
                    for part in parts:
                        if part.get("text") is not None:
                            is_thinking = is_thinking_part(part)
                            if (
                                current_block is None
                                or (is_thinking and current_block.type != "thinking")
                                or (not is_thinking and current_block.type != "text")
                            ):
                                if current_block is not None:
                                    if current_block.type == "text":
                                        stream.push(
                                            text_end_event(len(blocks) - 1, current_block.text, output)
                                        )
                                    else:
                                        stream.push(
                                            thinking_end_event(block_index(), current_block.thinking, output)
                                        )
                                if is_thinking:
                                    current_block = ThinkingContent(
                                        type="thinking", thinking="", thinking_signature=None
                                    )
                                    output.content.append(current_block)
                                    stream.push(thinking_start_event(block_index(), output))
                                else:
                                    current_block = TextContent(type="text", text="")
                                    output.content.append(current_block)
                                    stream.push(text_start_event(block_index(), output))
                            if current_block.type == "thinking":
                                current_block.thinking += part["text"]
                                current_block.thinking_signature = retain_thought_signature(
                                    current_block.thinking_signature, part.get("thoughtSignature")
                                )
                                stream.push(thinking_delta_event(block_index(), part["text"], output))
                            else:
                                current_block.text += part["text"]
                                current_block.text_signature = retain_thought_signature(
                                    current_block.text_signature, part.get("thoughtSignature")
                                )
                                stream.push(text_delta_event(block_index(), part["text"], output))

                        function_call = part.get("functionCall")
                        if function_call:
                            if current_block is not None:
                                if current_block.type == "text":
                                    stream.push(
                                        text_end_event(block_index(), current_block.text, output)
                                    )
                                else:
                                    stream.push(
                                        thinking_end_event(block_index(), current_block.thinking, output)
                                    )
                                current_block = None

                            # 未提供 ID 或与已有 ID 重复时，生成唯一 ID。
                            provided_id = function_call.get("id")
                            needs_new_id = not provided_id or any(
                                block.type == "toolCall" and block.id == provided_id for block in output.content
                            )
                            name = function_call.get("name") or ""
                            if needs_new_id:
                                global _tool_call_counter
                                _tool_call_counter += 1
                                tool_call_id = f"{name}_{_now_ms()}_{_tool_call_counter}"
                            else:
                                tool_call_id = provided_id

                            tool_call = ToolCall(
                                type="toolCall",
                                id=tool_call_id,
                                name=name,
                                arguments=function_call.get("args")
                                if function_call.get("args") is not None
                                else {},
                            )
                            if part.get("thoughtSignature"):
                                tool_call.thought_signature = part["thoughtSignature"]

                            output.content.append(tool_call)
                            stream.push(tool_call_start_event(block_index(), output))
                            stream.push(
                                tool_call_delta_event(block_index(), _json_stringify(tool_call.arguments), output)
                            )
                            stream.push(tool_call_end_event(block_index(), tool_call, output))

                finish_reason = candidate.get("finishReason") if candidate is not None else None
                if finish_reason:
                    output.raw_stop_reason = finish_reason
                    output.stop_reason = map_stop_reason(finish_reason)
                    if any(block.type == "toolCall" for block in output.content) and output.stop_reason == (
                        StopReason.STOP
                    ):
                        output.stop_reason = StopReason.TOOL_USE

                usage_metadata = chunk.get("usageMetadata")
                if usage_metadata:
                    output.usage = Usage(
                        input=(usage_metadata.get("promptTokenCount") or 0)
                        - (usage_metadata.get("cachedContentTokenCount") or 0),
                        output=(usage_metadata.get("candidatesTokenCount") or 0)
                        + (usage_metadata.get("thoughtsTokenCount") or 0),
                        cache_read=usage_metadata.get("cachedContentTokenCount") or 0,
                        cache_write=0,
                        reasoning=usage_metadata.get("thoughtsTokenCount") or 0,
                        total_tokens=usage_metadata.get("totalTokenCount") or 0,
                    )
                    calculate_cost(model, output.usage)

            if current_block is not None:
                if current_block.type == "text":
                    stream.push(text_end_event(block_index(), current_block.text, output))
                else:
                    stream.push(thinking_end_event(block_index(), current_block.thinking, output))

            if options is not None and options.signal is not None and options.signal.aborted:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == StopReason.PENDING:
                raise RuntimeError("Google stream ended without a finish reason")
            if output.stop_reason == StopReason.ABORTED or output.stop_reason == StopReason.ERROR:
                error_message = (
                    f"Provider stopped with: {output.raw_stop_reason}"
                    if output.raw_stop_reason
                    else "An unknown error occurred"
                )
                raise RuntimeError(error_message)

            stream.push(done_event(output.stop_reason, output))
            stream.end()
        except Exception as error:  # noqa: BLE001 - 编码进 stream 协议
            # 移除 SDK 在流式期间附加的内部 index 属性。
            for block in output.content:
                if hasattr(block, "index"):
                    delattr(block, "index")
            aborted = options is not None and options.signal is not None and options.signal.aborted
            output.stop_reason = StopReason.ABORTED if aborted else StopReason.ERROR
            output.error_message = format_provider_error(normalize_provider_error(error))
            stream.push(error_event(output.stop_reason, output))
            stream.end()
        finally:
            if google_stream is not None:
                await google_stream.close()

    task = asyncio.ensure_future(run())
    task.add_done_callback(_swallow)
    return stream


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """使用 provider 中立的简单选项发起流式请求。

    Args:
        model: 目标模型。
        context: 会话上下文。
        options: provider 中立的简单选项。

    Returns:
        已推入事件、可供消费的 :class:`AssistantMessageEventStream`。
    """
    api_key = options.api_key if options is not None else None
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    base_fields = _stream_option_fields(base)
    tool_choice = options.tool_choice if options is not None else None
    if options is None or not options.reasoning:
        return stream(
            model,
            context,
            _make_google_options(base_fields, tool_choice, GoogleThinkingOptions(enabled=False)),
        )

    clamped_reasoning = clamp_thinking_level(model, options.reasoning)
    if clamped_reasoning == "off":
        return stream(
            model,
            context,
            _make_google_options(base_fields, tool_choice, GoogleThinkingOptions(enabled=False)),
        )
    resolved_level = resolve_google_thinking_level(model, clamped_reasoning)

    if uses_google_thinking_level(model):
        return stream(
            model,
            context,
            _make_google_options(
                base_fields,
                tool_choice,
                GoogleThinkingOptions(enabled=True, level=to_google_thinking_level(resolved_level)),
            ),
        )

    return stream(
        model,
        context,
        _make_google_options(
            base_fields,
            tool_choice,
            GoogleThinkingOptions(
                enabled=True,
                budget_tokens=_get_google_budget(model, resolved_level, options.thinking_budgets),
            ),
        ),
    )


def _stream_option_fields(options: StreamOptions) -> dict[str, Any]:
    """提取 ``options`` 的 ``StreamOptions`` 字段，用于构建 provider 选项 dataclass。

    Args:
        options: 任意 ``StreamOptions`` 实例。

    Returns:
        字段名到字段值的映射。
    """
    return {dataclass_field.name: getattr(options, dataclass_field.name) for dataclass_field in fields(StreamOptions)}


def _make_google_options(
    base_fields: dict[str, Any],
    tool_choice: str | None,
    thinking: GoogleThinkingOptions,
) -> GoogleOptions:
    """用基础字段与 Google 专属字段构造选项实例。

    Args:
        base_fields: 来自 ``StreamOptions`` 的字段映射。
        tool_choice: 调用方指定的 tool choice。
        thinking: 按请求的 thinking 控制。

    Returns:
        构造完成的 :class:`GoogleOptions`。
    """
    return GoogleOptions(**base_fields, tool_choice=tool_choice, thinking=thinking)


def _request_url(model: Model) -> str:
    """构建 ``:streamGenerateContent`` 请求 URL。

    Args:
        model: 目标模型。

    Returns:
        带 ``alt=sse`` 的完整 URL。
    """
    base_url = model.base_url.rstrip("/") if model.base_url else _DEFAULT_BASE_URL
    return f"{base_url}/{_model_path(model.id)}:streamGenerateContent?alt=sse"


def _model_path(model_id: str) -> str:
    """SDK 构建请求路径前应用的 ``tModel`` 变换。

    Args:
        model_id: 模型 ID。

    Returns:
        带 ``models/`` 或 ``tunedModels/`` 前缀的模型路径。

    Raises:
        ValueError: 模型 ID 含 ``..``、``?`` 或 ``&`` 等非法字符时。
    """
    if ".." in model_id or "?" in model_id or "&" in model_id:
        raise ValueError("invalid model parameter")
    if model_id.startswith("models/") or model_id.startswith("tunedModels/"):
        return model_id
    return f"models/{model_id}"


def _request_headers(model: Model, api_key: str, options: GoogleOptions | None) -> dict[str, str]:
    """合并 pi user agent、model headers 与请求 headers，再附上 API key。

    Args:
        model: 目标模型。
        api_key: Gemini API key。
        options: provider 专属选项。

    Returns:
        最终发送的请求头映射。
    """
    options_headers: ProviderHeaders | None = options.headers if options is not None else None
    headers = provider_headers_to_record({"User-Agent": get_pi_user_agent()}, model.headers, options_headers) or {}
    headers.setdefault("x-goog-api-key", api_key)
    return headers


def _build_params(
    model: Model,
    context: TranscriptContext,
    options: GoogleOptions | None = None,
) -> dict[str, Any]:
    """构建 ``onPayload`` 所观察到的 ``GenerateContentParameters`` 请求。

    Args:
        model: 目标模型。
        context: 会话上下文。
        options: provider 专属选项。

    Returns:
        ``{model, contents, config}`` 形态的请求参数。

    Raises:
        RuntimeError: 传入的 ``options.signal`` 已被中止时。
    """
    options = options if options is not None else GoogleOptions()
    contents = convert_messages(model, context)
    initial_system_message = get_initial_system_message(context.messages)
    current_tools = get_current_tools(context.messages)

    generation_config: dict[str, Any] = {}
    if options.temperature is not None:
        generation_config["temperature"] = options.temperature
    if options.max_tokens is not None:
        generation_config["maxOutputTokens"] = options.max_tokens

    supports_strict_mode = supports_google_strict_tool_sampling(model.id)
    function_calling_mode = (
        resolve_google_function_calling_mode(current_tools, options.tool_choice, supports_strict_mode)
        if len(current_tools) > 0
        else None
    )
    system_instruction = get_system_message_text(initial_system_message) if initial_system_message else ""
    config: dict[str, Any] = dict(generation_config)
    if system_instruction:
        config["systemInstruction"] = sanitize_surrogates(system_instruction)
    if len(current_tools) > 0:
        config["tools"] = convert_tools(current_tools, False, supports_strict_mode)
    if function_calling_mode is not None:
        config["toolConfig"] = {"functionCallingConfig": {"mode": function_calling_mode}}

    thinking = options.thinking
    if thinking is not None and thinking.enabled and model.reasoning:
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        if thinking.level is not None:
            thinking_config["thinkingLevel"] = to_google_sdk_thinking_level(GoogleApiThinkingLevel(thinking.level))
        elif thinking.budget_tokens is not None:
            thinking_config["thinkingBudget"] = thinking.budget_tokens
        config["thinkingConfig"] = thinking_config
    elif model.reasoning and thinking is not None and not thinking.enabled:
        config["thinkingConfig"] = get_disabled_google_thinking_config(model)

    if options.signal is not None:
        if options.signal.aborted:
            raise RuntimeError("Request aborted")
        config["abortSignal"] = options.signal

    return {"model": model.id, "contents": contents, "config": config}


def _get_google_budget(
    model: Model,
    level: Any,
    custom_budgets: ThinkingBudgets | None = None,
) -> int:
    """``level`` 对应的 thinking token 预算。

    Args:
        model: 目标模型。
        level: 已解析的 thinking 等级。
        custom_budgets: 调用方自定义的预算覆盖。

    Returns:
        对应的 token 预算；未知模型返回 ``-1``（动态预算）。
    """
    name = level.value if hasattr(level, "value") else str(level)

    if custom_budgets is not None:
        custom = getattr(custom_budgets, name, None)
        if custom is not None:
            return custom

    if "2.5-pro" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}
        return budgets[name]

    if "2.5-flash-lite" in model.id:
        budgets = {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}
        return budgets[name]

    if "2.5-flash" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}
        return budgets[name]

    return -1


def google_generative_ai_api():
    """延迟加载的 :class:`~pi_ai.types.ProviderStreams` 实现。"""
    from .lazy import lazy_api, lazy_load

    return lazy_api(lazy_load("pi_ai.api.google_generative_ai"))
