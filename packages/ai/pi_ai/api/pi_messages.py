"""pi-messages API 实现。

移植自 ``packages/ai/src/api/pi-messages.ts``。

将 pi 自有消息协议直接对接后端：请求是对 ``<baseUrl>/messages`` 的一次 POST
（``{ model, context, options }``），响应是序列化的 assistant 消息事件的 SSE stream，
最后以 ``done``/``error`` 事件收尾。这是 Radius 网关使用的线上协议，任何实现该协议的
后端均可接入，例如通过 models.json 自定义 provider 并设置 ``"api": "pi-messages"``。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, TypeAlias

from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    CacheRetention,
    JsonObject,
    Model,
    ProviderEnv,
    ProviderResponse,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingLevel,
    ToolCall,
    TranscriptContext,
    Usage,
)
from ..utils.abort import race_with_abort_signal
from ..utils.async_utils import maybe_await
from ..utils.diagnostics import (
    AssistantMessageDiagnostic,
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
)
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
from ..utils.headers import headers_to_record, provider_headers_to_record
from ..utils.http import create_client
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.serde import from_json, to_json

__all__ = [
    "PiMessagesDoneEvent",
    "PiMessagesErrorEvent",
    "PiMessagesEvent",
    "PiMessagesOptions",
    "PiMessagesResponseError",
    "PiMessagesRewriteImpact",
    "PiMessagesStartEvent",
    "PiMessagesTextDeltaEvent",
    "PiMessagesTextEndEvent",
    "PiMessagesTextStartEvent",
    "PiMessagesThinkingDeltaEvent",
    "PiMessagesThinkingEndEvent",
    "PiMessagesThinkingStartEvent",
    "PiMessagesToolCallDeltaEvent",
    "PiMessagesToolCallEndEvent",
    "PiMessagesToolCallStartEvent",
    "pi_messages_api",
    "stream",
    "stream_simple",
]


@dataclass
class PiMessagesOptions(StreamOptions):
    """pi-messages adapter 接受的选项。

    在 :class:`~pi_ai.types.StreamOptions` 通用字段之上追加本协议特有的控制项。

    Attributes:
        reasoning: thinking level；也可直接传入 provider 约定的字符串。
        tool_choice: 工具选择策略，序列化时别名为 ``toolChoice``。
        debug: 请求后端返回调试元数据（例如路由相关的响应头）。
    """

    reasoning: ThinkingLevel | str | None = None
    tool_choice: Any = field(default=None, metadata={"alias": "toolChoice"})
    #: 请求后端返回调试元数据（例如路由相关的响应头）。
    debug: bool | None = None


# --------------------------------------------------------------------------------------
# 线上事件（wire events）
# --------------------------------------------------------------------------------------


@dataclass
class PiMessagesRewriteImpact:
    """服务端消息重写（如网关策略）的影响摘要。

    Attributes:
        policy_id: 触发重写的策略 id。
        policy_version: 策略版本号。
        changed: 消息是否确实被改写。
        token_count_change: token 数量的变化量。
        message_count_change: 消息条数的变化量。
        system_prompt_changed: system prompt 是否被改写。
    """

    policy_id: str = field(default="", metadata={"alias": "policyId"})
    policy_version: int = field(default=0, metadata={"alias": "policyVersion"})
    changed: bool = False
    token_count_change: int = field(default=0, metadata={"alias": "tokenCountChange"})
    message_count_change: int = field(default=0, metadata={"alias": "messageCountChange"})
    system_prompt_changed: bool = field(default=False, metadata={"alias": "systemPromptChanged"})


@dataclass
class PiMessagesStartEvent:
    """后端开始生成。

    Attributes:
        type: 事件类型标识，固定为 ``start``。
    """

    type: str = "start"


@dataclass
class PiMessagesTextStartEvent:
    """一个文本块开始。

    Attributes:
        content_index: 该块在 assistant 消息 content 中的下标。
        type: 事件类型标识，固定为 ``text_start``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    type: str = "text_start"


@dataclass
class PiMessagesTextDeltaEvent:
    """一个文本块新增增量。

    Attributes:
        content_index: 该块在 assistant 消息 content 中的下标。
        delta: 本帧新增的文本增量。
        type: 事件类型标识，固定为 ``text_delta``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "text_delta"


@dataclass
class PiMessagesTextEndEvent:
    """一个文本块结束。

    Attributes:
        content_index: 该块在 assistant 消息 content 中的下标。
        content: 文本块的最终内容。
        content_signature: 文本块的签名，供需要回传签名的 provider 使用。
        type: 事件类型标识，固定为 ``text_end``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: str = ""
    content_signature: str | None = field(default=None, metadata={"alias": "contentSignature"})
    type: str = "text_end"


@dataclass
class PiMessagesThinkingStartEvent:
    """一个 thinking 块开始。

    Attributes:
        content_index: 该块在 assistant 消息 content 中的下标。
        type: 事件类型标识，固定为 ``thinking_start``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    type: str = "thinking_start"


@dataclass
class PiMessagesThinkingDeltaEvent:
    """一个 thinking 块新增增量。

    Attributes:
        content_index: 该块在 assistant 消息 content 中的下标。
        delta: 本帧新增的思考文本增量。
        type: 事件类型标识，固定为 ``thinking_delta``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "thinking_delta"


@dataclass
class PiMessagesThinkingEndEvent:
    """一个 thinking 块结束。

    Attributes:
        content_index: 该块在 assistant 消息 content 中的下标。
        content: thinking 块的最终内容。
        content_signature: thinking 块的签名。
        redacted: 该 thinking 块是否已被 provider 脱敏。
        type: 事件类型标识，固定为 ``thinking_end``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: str = ""
    content_signature: str | None = field(default=None, metadata={"alias": "contentSignature"})
    redacted: bool | None = None
    type: str = "thinking_end"


@dataclass
class PiMessagesToolCallStartEvent:
    """一次工具调用开始。

    Attributes:
        content_index: 该工具调用块在 assistant 消息 content 中的下标。
        id: 工具调用 id。
        tool_name: 被调用的工具名。
        type: 事件类型标识，固定为 ``toolcall_start``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    id: str = ""
    tool_name: str = field(default="", metadata={"alias": "toolName"})
    type: str = "toolcall_start"


@dataclass
class PiMessagesToolCallDeltaEvent:
    """一次工具调用的 JSON 参数新增增量。

    Attributes:
        content_index: 该工具调用块在 assistant 消息 content 中的下标。
        delta: 本帧新增的参数 JSON 片段。
        type: 事件类型标识，固定为 ``toolcall_delta``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "toolcall_delta"


@dataclass
class PiMessagesToolCallEndEvent:
    """一次工具调用结束。

    Attributes:
        content_index: 该工具调用块在 assistant 消息 content 中的下标。
        tool_call: 定型后的完整工具调用。
        type: 事件类型标识，固定为 ``toolcall_end``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    tool_call: ToolCall = field(default_factory=ToolCall, metadata={"alias": "toolCall"})
    type: str = "toolcall_end"


@dataclass
class PiMessagesDoneEvent:
    """后端成功完成生成。

    Attributes:
        reason: 归一化的停止原因。
        usage: 本次请求的 token 用量。
        response_id: 后端返回的响应标识。
        provider_thinking_level: provider 实际使用的 thinking level。
        rewrite: 服务端消息重写的影响摘要。
        type: 事件类型标识，固定为 ``done``。
    """

    reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    response_id: str | None = field(default=None, metadata={"alias": "responseId"})
    provider_thinking_level: str | None = field(default=None, metadata={"alias": "providerThinkingLevel"})
    rewrite: PiMessagesRewriteImpact | None = None
    type: str = "done"


@dataclass
class PiMessagesErrorEvent:
    """后端以错误或中止（abort）终止。

    Attributes:
        reason: 归一化的停止原因，为 ``error`` 或 ``aborted``。
        usage: 出错前已产生的 token 用量。
        error_message: 错误说明。
        response_id: 后端返回的响应标识。
        provider_thinking_level: provider 实际使用的 thinking level。
        rewrite: 服务端消息重写的影响摘要。
        type: 事件类型标识，固定为 ``error``。
    """

    reason: str = "error"
    usage: Usage = field(default_factory=Usage)
    error_message: str | None = field(default=None, metadata={"alias": "errorMessage"})
    response_id: str | None = field(default=None, metadata={"alias": "responseId"})
    provider_thinking_level: str | None = field(default=None, metadata={"alias": "providerThinkingLevel"})
    rewrite: PiMessagesRewriteImpact | None = None
    type: str = "error"


#: pi-messages 后端下发的序列化 assistant 消息事件。
PiMessagesEvent: TypeAlias = (
    PiMessagesStartEvent
    | PiMessagesTextStartEvent
    | PiMessagesTextDeltaEvent
    | PiMessagesTextEndEvent
    | PiMessagesThinkingStartEvent
    | PiMessagesThinkingDeltaEvent
    | PiMessagesThinkingEndEvent
    | PiMessagesToolCallStartEvent
    | PiMessagesToolCallDeltaEvent
    | PiMessagesToolCallEndEvent
    | PiMessagesDoneEvent
    | PiMessagesErrorEvent
)

#: 请求失败时后端可能返回的错误信封（error envelope）。
PiMessagesErrorBody: TypeAlias = dict[str, Any]


class PiMessagesResponseError(Exception):
    """非 2xx 的 pi-messages 响应，携带 provider 错误信封。"""

    def __init__(
        self,
        message: str,
        code: str | None,
        diagnostic_details: JsonObject,
    ) -> None:
        """记录错误消息、错误码与诊断详情。

        Args:
            message: 面向用户的错误消息。
            code: provider 错误信封中的错误码。
            diagnostic_details: 附加到诊断记录的结构化详情。
        """
        super().__init__(message)
        self.name = "PiMessagesResponseError"
        self.code = code
        self.diagnostic_details = diagnostic_details


# --------------------------------------------------------------------------------------
# 响应错误辅助函数
# --------------------------------------------------------------------------------------


def _parse_pi_messages_error_body(body: str) -> PiMessagesErrorBody | None:
    """解析 ``{error: {...}}`` 信封；不是该结构时返回 ``None``。

    Args:
        body: 原始响应体文本。

    Returns:
        含 ``error`` 对象的信封字典；结构不符时返回 ``None``。
    """
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    if error is None:
        return None
    return parsed if isinstance(error, dict) else None


def _truncate_diagnostic_string(value: str) -> str:
    """将诊断字符串截断至 8192 个字符。

    Args:
        value: 待截断的字符串。

    Returns:
        超长时以省略号结尾的截断结果，否则原样返回。
    """
    max_length = 8192
    return f"{value[:max_length]}…" if len(value) > max_length else value


def _format_pi_messages_response_error(
    response: Any,
    body: str,
    error_body: PiMessagesErrorBody | None,
) -> str:
    """生成 ``<status> <statusText>: <message>`` 格式的失败文本。

    Args:
        response: ``httpx`` 响应对象，提供状态码与状态短语。
        body: 原始响应体文本，作为消息的兜底来源。
        error_body: 已解析的错误信封；为 ``None`` 时退回 ``body``。

    Returns:
        面向用户的失败文本。
    """
    error = (error_body or {}).get("error") or {}
    message = error.get("message") if isinstance(error.get("message"), str) else None
    code = error.get("code") if isinstance(error.get("code"), str) else None
    suffix = message if message is not None else body
    code_suffix = f" ({code})" if code else ""
    return f"{response.status_code} {response.reason_phrase}: {suffix}{code_suffix}"


def _create_pi_messages_response_error(
    model: Model,
    url: str,
    response: Any,
    body: str,
) -> PiMessagesResponseError:
    """构建带诊断详情的类型化响应失败异常。

    Args:
        model: 目标模型元数据。
        url: 实际请求的端点地址。
        response: ``httpx`` 响应对象。
        body: 原始响应体文本。

    Returns:
        填充好诊断详情的 :class:`PiMessagesResponseError`。
    """
    error_body = _parse_pi_messages_error_body(body)
    error = (error_body or {}).get("error") or {}
    code = error.get("code") if isinstance(error.get("code"), str) else None
    details: JsonObject = {
        "version": 1,
        "provider": model.provider,
        "model": model.id,
        "url": url,
        "status": response.status_code,
        "statusText": response.reason_phrase,
    }
    if error_body is None:
        details["body"] = _truncate_diagnostic_string(body)
    else:
        details["error"] = to_json(error_body.get("error"))
    details["timestampMs"] = _now_ms()
    return PiMessagesResponseError(
        _format_pi_messages_response_error(response, body, error_body),
        code,
        details,
    )


def _now_ms() -> int:
    """返回当前 Unix 时间戳（毫秒）。"""
    return int(time.time() * 1000)


def _create_empty_usage() -> Usage:
    """全零 usage，与 TypeScript 字面量的构造结果一致。

    Returns:
        各计数字段均为 ``0`` 的 usage。
    """
    return Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0)


def _append_rewrite_diagnostic(
    message: AssistantMessage,
    rewrite: PiMessagesRewriteImpact | None,
) -> None:
    """在 assistant 消息上记录一次服务端重写。

    Args:
        message: 被写入诊断信息的 assistant 消息。
        rewrite: 重写影响摘要；为 ``None`` 时不记录。
    """
    if not rewrite:
        return
    append_assistant_message_diagnostic(
        message,
        AssistantMessageDiagnostic(
            type="pi_messages_rewrite",
            timestamp=_now_ms(),
            details=to_json(rewrite),
        ),
    )


# --------------------------------------------------------------------------------------
# 事件转换
# --------------------------------------------------------------------------------------


def _set_content(partial: AssistantMessage, index: int, block: Any) -> None:
    """赋值 ``partial.content[index]``，稀疏索引像 JS 数组一样用空块填充。

    Args:
        partial: 被就地修改的 assistant 消息。
        index: 目标下标。
        block: 写入该位置的内容块。
    """
    while len(partial.content) <= index:
        from ..types import TextContent

        partial.content.append(TextContent(type="text", text=""))
    partial.content[index] = block


def _create_event_converter(model: Model):
    """构建有状态的 ``PiMessagesEvent -> AssistantMessageEvent`` 转换器。

    Args:
        model: 目标模型元数据，用于初始化 assistant 消息骨架。

    Returns:
        接收线上事件、返回对应 assistant 事件的转换函数。
    """
    partial = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_create_empty_usage(),
        stop_reason="pending",
        timestamp=_now_ms(),
    )
    tool_json: dict[int, str] = {}

    def convert(event: PiMessagesEvent) -> AssistantMessageEvent:
        """把一条线上事件转换为 assistant 事件，并同步内部消息状态。

        Args:
            event: 已解析的 pi-messages 线上事件。

        Returns:
            与 ``event`` 对应的 assistant 消息事件。
        """
        if isinstance(event, PiMessagesDoneEvent):
            partial.stop_reason = event.reason
            partial.usage = event.usage
            partial.response_id = event.response_id
            if event.provider_thinking_level is not None:
                partial.provider_thinking_level = event.provider_thinking_level
            _append_rewrite_diagnostic(partial, event.rewrite)
            return done_event(event.reason, partial)
        if isinstance(event, PiMessagesErrorEvent):
            partial.stop_reason = event.reason
            partial.usage = event.usage
            partial.error_message = event.error_message
            partial.response_id = event.response_id
            if event.provider_thinking_level is not None:
                partial.provider_thinking_level = event.provider_thinking_level
            _append_rewrite_diagnostic(partial, event.rewrite)
            return error_event(event.reason, partial)
        if isinstance(event, PiMessagesStartEvent):
            return start_event(partial)
        if isinstance(event, PiMessagesTextStartEvent):
            from ..types import TextContent

            _set_content(partial, event.content_index, TextContent(type="text", text=""))
            return text_start_event(event.content_index, partial)
        if isinstance(event, PiMessagesTextDeltaEvent):
            block = partial.content[event.content_index]
            block.text += event.delta  # type: ignore[union-attr]  # 已按事件类型收窄
            return text_delta_event(event.content_index, event.delta, partial)
        if isinstance(event, PiMessagesTextEndEvent):
            block = partial.content[event.content_index]
            block.text = event.content  # type: ignore[union-attr]  # 已按事件类型收窄
            block.text_signature = event.content_signature  # type: ignore[union-attr]  # 已按事件类型收窄
            return text_end_event(event.content_index, event.content, partial)
        if isinstance(event, PiMessagesThinkingStartEvent):
            from ..types import ThinkingContent

            _set_content(partial, event.content_index, ThinkingContent(type="thinking", thinking=""))
            return thinking_start_event(event.content_index, partial)
        if isinstance(event, PiMessagesThinkingDeltaEvent):
            block = partial.content[event.content_index]
            block.thinking += event.delta  # type: ignore[union-attr]  # 已按事件类型收窄
            return thinking_delta_event(event.content_index, event.delta, partial)
        if isinstance(event, PiMessagesThinkingEndEvent):
            block = partial.content[event.content_index]
            block.thinking = event.content  # type: ignore[union-attr]  # 已按事件类型收窄
            block.thinking_signature = event.content_signature  # type: ignore[union-attr]  # 已按事件类型收窄
            block.redacted = event.redacted  # type: ignore[union-attr]  # 已按事件类型收窄
            return thinking_end_event(event.content_index, event.content, partial)
        if isinstance(event, PiMessagesToolCallStartEvent):
            _set_content(
                partial,
                event.content_index,
                ToolCall(type="toolCall", id=event.id, name=event.tool_name, arguments={}),
            )
            tool_json[event.content_index] = ""
            return tool_call_start_event(event.content_index, partial)
        if isinstance(event, PiMessagesToolCallDeltaEvent):
            accumulated = f"{tool_json.get(event.content_index) or ''}{event.delta}"
            tool_json[event.content_index] = accumulated
            block = partial.content[event.content_index]
            block.arguments = parse_streaming_json(accumulated)  # type: ignore[union-attr]  # 已按事件类型收窄
            return tool_call_delta_event(event.content_index, event.delta, partial)
        # 以下为 PiMessagesToolCallEndEvent
        block = partial.content[event.content_index]
        block.id = event.tool_call.id  # type: ignore[union-attr]  # 已按事件类型收窄
        block.name = event.tool_call.name  # type: ignore[union-attr]  # 已按事件类型收窄
        block.arguments = event.tool_call.arguments  # type: ignore[union-attr]  # 已按事件类型收窄
        tool_json.pop(event.content_index, None)
        return tool_call_end_event(event.content_index, block, partial)  # type: ignore[arg-type]  # 已按事件类型收窄

    return convert


# --------------------------------------------------------------------------------------
# SSE 解析
# --------------------------------------------------------------------------------------


def _parse_pi_messages_event(raw: str) -> tuple[dict[str, Any], PiMessagesEvent] | None:
    """将一条 SSE 记录解析为 ``(原始负载, 类型化事件)``。

    心跳和 ``[DONE]`` 返回 ``None``。保留原始负载是因为 ``on_provider_stream_event``
    会转发*解析后的线上事件*，包括本 adapter 未建模的字段。

    Args:
        raw: 单条 SSE 记录的原始文本。

    Returns:
        ``(原始负载, 类型化事件)`` 二元组；心跳、``[DONE]`` 或无法识别时返回 ``None``。
    """
    data = None
    for line in raw.split("\n"):
        if line.startswith("data:"):
            data = line[5:].strip()
            break

    if not data or data == "[DONE]":
        return None
    payload = json.loads(data)
    event = _pi_messages_event_from_json(payload)
    if event is None:
        return None
    return payload, event


def _pi_messages_event_from_json(data: Any) -> PiMessagesEvent | None:
    """根据解析后的 JSON 负载构建可辨识的线上事件。

    Args:
        data: 解析后的 JSON 负载。

    Returns:
        对应的线上事件；``type`` 缺失或未知时返回 ``None``。
    """
    if not isinstance(data, dict):
        return None
    event_type = data.get("type")
    if event_type == "start":
        return PiMessagesStartEvent()
    if event_type == "text_start":
        return PiMessagesTextStartEvent(content_index=int(data.get("contentIndex") or 0))
    if event_type == "text_delta":
        return PiMessagesTextDeltaEvent(
            content_index=int(data.get("contentIndex") or 0),
            delta=data.get("delta") or "",
        )
    if event_type == "text_end":
        return PiMessagesTextEndEvent(
            content_index=int(data.get("contentIndex") or 0),
            content=data.get("content") or "",
            content_signature=data.get("contentSignature"),
        )
    if event_type == "thinking_start":
        return PiMessagesThinkingStartEvent(content_index=int(data.get("contentIndex") or 0))
    if event_type == "thinking_delta":
        return PiMessagesThinkingDeltaEvent(
            content_index=int(data.get("contentIndex") or 0),
            delta=data.get("delta") or "",
        )
    if event_type == "thinking_end":
        return PiMessagesThinkingEndEvent(
            content_index=int(data.get("contentIndex") or 0),
            content=data.get("content") or "",
            content_signature=data.get("contentSignature"),
            redacted=data.get("redacted"),
        )
    if event_type == "toolcall_start":
        return PiMessagesToolCallStartEvent(
            content_index=int(data.get("contentIndex") or 0),
            id=data.get("id") or "",
            tool_name=data.get("toolName") or "",
        )
    if event_type == "toolcall_delta":
        return PiMessagesToolCallDeltaEvent(
            content_index=int(data.get("contentIndex") or 0),
            delta=data.get("delta") or "",
        )
    if event_type == "toolcall_end":
        return PiMessagesToolCallEndEvent(
            content_index=int(data.get("contentIndex") or 0),
            tool_call=from_json(ToolCall, data.get("toolCall") or {}),
        )
    if event_type == "done":
        return PiMessagesDoneEvent(
            reason=data.get("reason") or "stop",
            usage=from_json(Usage, data.get("usage") or {}),
            response_id=data.get("responseId"),
            provider_thinking_level=data.get("providerThinkingLevel"),
            rewrite=_rewrite_from_json(data.get("rewrite")),
        )
    if event_type == "error":
        return PiMessagesErrorEvent(
            reason=data.get("reason") or "error",
            usage=from_json(Usage, data.get("usage") or {}),
            error_message=data.get("errorMessage"),
            response_id=data.get("responseId"),
            provider_thinking_level=data.get("providerThinkingLevel"),
            rewrite=_rewrite_from_json(data.get("rewrite")),
        )
    return None


def _rewrite_from_json(value: Any) -> PiMessagesRewriteImpact | None:
    """解析可选的重写影响对象。

    Args:
        value: 负载中的 ``rewrite`` 字段。

    Returns:
        反序列化后的重写摘要；不是对象时返回 ``None``。
    """
    if not isinstance(value, dict):
        return None
    return from_json(PiMessagesRewriteImpact, value)


async def _read_pi_messages_events(response: Any):
    """从 SSE 响应中逐个产出线上事件，缓冲未完整的记录。

    Args:
        response: ``httpx`` 流式响应对象。

    Yields:
        ``(原始负载, 类型化事件)`` 二元组。
    """
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        buffer = buffer.replace("\r\n", "\n")

        split = buffer.find("\n\n")
        while split != -1:
            parsed = _parse_pi_messages_event(buffer[:split])
            if parsed is not None:
                yield parsed
            buffer = buffer[split + 2 :]
            split = buffer.find("\n\n")

    if buffer.strip():
        parsed = _parse_pi_messages_event(buffer)
        if parsed is not None:
            yield parsed


# --------------------------------------------------------------------------------------
# stream 实现
# --------------------------------------------------------------------------------------


def _create_error_event(model: Model, error: Any, aborted: bool) -> AssistantMessageEvent:
    """把 stream 失败编码为终止性的 error 事件。

    Args:
        model: 目标模型元数据。
        error: 捕获到的异常。
        aborted: 失败是否由调用方中止引起。

    Returns:
        ``error`` 类型的 assistant 事件。
    """
    reason = "aborted" if aborted else "error"
    assistant_message = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_create_empty_usage(),
        stop_reason=reason,
        error_message=str(error),
        timestamp=_now_ms(),
    )

    if not aborted and isinstance(error, PiMessagesResponseError):
        append_assistant_message_diagnostic(
            assistant_message,
            create_assistant_message_diagnostic(
                "pi_messages_response_failure", error, error.diagnostic_details
            ),
        )

    return error_event(reason, assistant_message)


def _resolve_cache_retention(
    cache_retention: CacheRetention | str | None,
    env: ProviderEnv | None,
) -> CacheRetention | str | None:
    """未显式设置时应用遗留的 ``PI_CACHE_RETENTION`` 开关。

    Args:
        cache_retention: 调用方显式设置的缓存策略。
        env: provider 环境变量表。

    Returns:
        最终生效的缓存策略；未设置且环境变量不是 ``long`` 时返回 ``None``。
    """
    if cache_retention:
        return cache_retention
    # 未设置时由后端默认值生效；这里只映射遗留的环境变量开关。
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else None


def _pi_messages_options_from(base: StreamOptions, **overrides: Any) -> PiMessagesOptions:
    """把 :class:`StreamOptions` 投影为 :class:`PiMessagesOptions`。

    Args:
        base: 提供通用字段的基类选项。
        **overrides: 需要覆盖的字段。

    Returns:
        合并后的 pi-messages 选项。
    """
    values = {item.name: getattr(base, item.name) for item in dataclass_fields(StreamOptions)}
    values.update(overrides)
    return PiMessagesOptions(**values)


def stream(
    model: Model,
    context: TranscriptContext,
    options: PiMessagesOptions | None = None,
) -> AssistantMessageEventStream:
    """使用完整 provider 选项发起 pi-messages 响应 stream。

    Args:
        model: 目标模型元数据。
        context: 待发送的对话上下文。
        options: 完整 provider 选项；为 ``None`` 时使用默认值。

    Returns:
        推送 assistant 消息事件的事件 stream。

    Raises:
        RuntimeError: 当 ``options`` 未提供 API key 时。
    """
    options = options or PiMessagesOptions()
    event_stream = AssistantMessageEventStream()
    convert_event = _create_event_converter(model)

    async def run() -> None:
        """在后台 task 中执行完整的请求、SSE 解析与事件推送流程。"""
        try:
            api_key = options.api_key
            if not api_key:
                raise RuntimeError(f'No API key provided for provider "{model.provider}"')

            url = f"{model.base_url.rstrip('/')}/messages"
            if options.debug:
                url = f"{url}?debug=1"

            payload: Any = {
                "model": model.id,
                "context": to_json(context),
                "options": {
                    key: value
                    for key, value in {
                        "temperature": options.temperature,
                        "maxTokens": options.max_tokens,
                        "reasoning": options.reasoning,
                        "cacheRetention": _resolve_cache_retention(options.cache_retention, options.env),
                        "sessionId": options.session_id,
                        "toolChoice": options.tool_choice,
                    }.items()
                    if value is not None
                },
            }
            if options.on_payload is not None:
                next_payload = await maybe_await(options.on_payload(payload, model))
                if next_payload is not None:
                    payload = next_payload

            headers = provider_headers_to_record(
                {
                    "authorization": f"Bearer {api_key}",
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                },
                options.headers,
            ) or {}

            client = create_client(
                headers=headers,
                timeout_ms=options.timeout_ms,
                fetch=options.fetch,
            )
            request = client.build_request("POST", url, json=payload)
            try:
                if options.signal is not None:
                    response = await race_with_abort_signal(client.send(request, stream=True), options.signal)
                else:
                    response = await client.send(request, stream=True)
            except BaseException:
                await client.aclose()
                raise

            try:
                if options.on_response is not None:
                    await maybe_await(
                        options.on_response(
                            ProviderResponse(
                                status=response.status_code,
                                headers=headers_to_record(dict(response.headers)),
                            ),
                            model,
                        )
                    )

                if not response.is_success:
                    body = (await response.aread()).decode(response.encoding or "utf-8", errors="replace")
                    raise _create_pi_messages_response_error(model, url, response, body)
                if response.stream is None:
                    raise RuntimeError(f"{model.provider} response has no body")

                async for raw_event, pi_event in _read_pi_messages_events(response):
                    if options.signal is not None:
                        options.signal.throw_if_aborted()
                    if options.on_provider_stream_event is not None:
                        # 上游把解析后的线上负载交给观察者，因此本 adapter 未建模的
                        # 字段对调用方依然可见。
                        await maybe_await(options.on_provider_stream_event(raw_event, model))
                    event = convert_event(pi_event)
                    event_stream.push(event)
                    if event.type in ("done", "error"):
                        return

                raise RuntimeError(f"{model.provider} stream ended without a terminal event")
            finally:
                await response.aclose()
                await client.aclose()
        except Exception as error:  # noqa: BLE001 - 异常编码进 stream 协议
            aborted = options.signal is not None and options.signal.aborted
            event_stream.push(_create_error_event(model, error, aborted))
            event_stream.end()

    task = asyncio.ensure_future(run())
    task.add_done_callback(_swallow_task)
    return event_stream


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """使用 provider 中立的简单选项发起 stream。

    Args:
        model: 目标模型元数据。
        context: 待发送的对话上下文。
        options: provider 中立的简单选项；为 ``None`` 时使用默认值。

    Returns:
        推送 assistant 消息事件的事件 stream。
    """
    base = options or SimpleStreamOptions()
    extra: Any = options
    return stream(
        model,
        context,
        _pi_messages_options_from(
            base,
            reasoning=options.reasoning if options else None,
            tool_choice=options.tool_choice if options else None,
            debug=getattr(extra, "debug", None),
        ),
    )


def pi_messages_api():
    """本模块的懒加载 provider-streams 包装。

    Returns:
        解析到本模块 ``stream``/``stream_simple`` 的懒加载 provider 描述。
    """
    from .lazy import lazy_api, lazy_load

    return lazy_api(lazy_load("pi_ai.api.pi_messages"))


def _swallow_task(task: asyncio.Task[Any]) -> None:
    """取出 task 的异常，避免其成为 "never retrieved"。

    Args:
        task: 已完成的 asyncio task。
    """
    if task.cancelled():
        return
    task.exception()
