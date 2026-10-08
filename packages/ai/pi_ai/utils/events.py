""":data:`pi_ai.types.AssistantMessageEvent` 协议的事件构造函数.

provider 适配器会发出长而高度重复的事件序列。这些辅助函数让发送点保持
可读，并保证每个事件都携带协议要求的 ``partial`` assistant message。
"""

from __future__ import annotations

from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StopReason,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

__all__ = [
    "done_event",
    "error_event",
    "start_event",
    "text_delta_event",
    "text_end_event",
    "text_start_event",
    "thinking_delta_event",
    "thinking_end_event",
    "thinking_start_event",
    "tool_call_delta_event",
    "tool_call_end_event",
    "tool_call_start_event",
]


def start_event(partial: AssistantMessage) -> StartEvent:
    """构造生成开始事件.

    Args:
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``StartEvent``。
    """
    return StartEvent(partial=partial)


def text_start_event(content_index: int, partial: AssistantMessage) -> TextStartEvent:
    """构造一个 text 块开始事件.

    Args:
        content_index: 该 text 块在消息内容中的下标。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``TextStartEvent``。
    """
    return TextStartEvent(content_index=content_index, partial=partial)


def text_delta_event(content_index: int, delta: str, partial: AssistantMessage) -> TextDeltaEvent:
    """构造 text 块增量事件.

    Args:
        content_index: 该 text 块在消息内容中的下标。
        delta: 本次新增的增量文本。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``TextDeltaEvent``。
    """
    return TextDeltaEvent(content_index=content_index, delta=delta, partial=partial)


def text_end_event(content_index: int, content: str, partial: AssistantMessage) -> TextEndEvent:
    """构造 text 块结束事件，携带权威的最终 ``content``.

    Args:
        content_index: 该 text 块在消息内容中的下标。
        content: 该 text 块的权威最终文本。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``TextEndEvent``。
    """
    return TextEndEvent(content_index=content_index, content=content, partial=partial)


def thinking_start_event(content_index: int, partial: AssistantMessage) -> ThinkingStartEvent:
    """构造一个 thinking 块开始事件.

    Args:
        content_index: 该 thinking 块在消息内容中的下标。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``ThinkingStartEvent``。
    """
    return ThinkingStartEvent(content_index=content_index, partial=partial)


def thinking_delta_event(content_index: int, delta: str, partial: AssistantMessage) -> ThinkingDeltaEvent:
    """构造 thinking 块增量事件.

    Args:
        content_index: 该 thinking 块在消息内容中的下标。
        delta: 本次新增的增量文本。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``ThinkingDeltaEvent``。
    """
    return ThinkingDeltaEvent(content_index=content_index, delta=delta, partial=partial)


def thinking_end_event(content_index: int, content: str, partial: AssistantMessage) -> ThinkingEndEvent:
    """构造 thinking 块结束事件，携带权威的最终 ``content``.

    Args:
        content_index: 该 thinking 块在消息内容中的下标。
        content: 该 thinking 块的权威最终文本。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``ThinkingEndEvent``。
    """
    return ThinkingEndEvent(content_index=content_index, content=content, partial=partial)


def tool_call_start_event(content_index: int, partial: AssistantMessage) -> ToolCallStartEvent:
    """构造一个 tool call 开始事件.

    Args:
        content_index: 该 tool call 在消息内容中的下标。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``ToolCallStartEvent``。
    """
    return ToolCallStartEvent(content_index=content_index, partial=partial)


def tool_call_delta_event(content_index: int, delta: str, partial: AssistantMessage) -> ToolCallDeltaEvent:
    """构造 tool call 的 JSON 参数增量事件.

    Args:
        content_index: 该 tool call 在消息内容中的下标。
        delta: 本次新增的 JSON 参数字符串片段。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``ToolCallDeltaEvent``。
    """
    return ToolCallDeltaEvent(content_index=content_index, delta=delta, partial=partial)


def tool_call_end_event(content_index: int, tool_call: ToolCall, partial: AssistantMessage) -> ToolCallEndEvent:
    """构造 tool call 结束事件.

    Args:
        content_index: 该 tool call 在消息内容中的下标。
        tool_call: 已完成的 tool call。
        partial: 事件携带的部分 assistant message。

    Returns:
        一个新的 ``ToolCallEndEvent``。
    """
    return ToolCallEndEvent(content_index=content_index, tool_call=tool_call, partial=partial)


def done_event(reason: StopReason | str, message: AssistantMessage) -> DoneEvent:
    """构造生成成功结束事件.

    Args:
        reason: 结束原因。
        message: 最终 assistant message。

    Returns:
        一个新的 ``DoneEvent``。
    """
    return DoneEvent(reason=reason, message=message)


def error_event(reason: StopReason | str, error: AssistantMessage) -> ErrorEvent:
    """构造以错误或中止结束的事件.

    Args:
        reason: 结束原因。
        error: 记录错误状态的 assistant message。

    Returns:
        一个新的 ``ErrorEvent``。
    """
    return ErrorEvent(reason=reason, error=error)
