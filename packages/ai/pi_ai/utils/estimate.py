"""上下文大小估算.

移植自 ``packages/ai/src/utils/estimate.ts``。以最近一个适用的 assistant
usage 块作为前缀的权威大小，仅估算其后的消息；没有这样的块时，整份
transcript 按四个字符一个 token 估算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from ..types import TranscriptContext, Usage
from .text import get_system_message_text

__all__ = [
    "CHARS_PER_TOKEN",
    "ContextUsageEstimate",
    "calculate_context_tokens",
    "estimate_content_tokens",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "estimate_text_tokens",
]

CHARS_PER_TOKEN = 4


@dataclass
class ContextUsageEstimate:
    """估算出的上下文大小，附带作为锚点的 usage 块."""

    #: 估算的上下文 token 总数。
    tokens: int = 0
    #: 最近一个适用的 assistant usage 块报告的 token 数。
    usage_tokens: int = 0
    #: 最近一个适用的 assistant usage 块之后的估算 token 数。
    trailing_tokens: int = 0
    #: 提供 usage 的适用消息下标；不存在时为 ``None``。
    last_usage_index: int | None = None


def calculate_context_tokens(usage: Usage) -> int:
    """usage 块报告的上下文 token 总数.

    Args:
        usage: assistant 消息携带的 usage 块。

    Returns:
        ``total_tokens``，为 0 时改用各分项之和。
    """
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def _safe_json(value: Any) -> str:
    """序列化为 JSON 文本；失败时退回占位字符串。

    Args:
        value: 待序列化的值。

    Returns:
        JSON 文本；无法序列化时返回 ``"[unserializable]"``。
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "[unserializable]"


def _content_chars(content: Any) -> int:
    """统计消息内容的字符数.

    Args:
        content: 字符串，或文本内容块列表。

    Returns:
        用于 token 估算的字符数。
    """
    if isinstance(content, str):
        return len(content)
    return sum(len(block.text) for block in content)


def estimate_text_tokens(text: str) -> int:
    """估算纯文本的 token 数.

    Args:
        text: 待估算的文本。

    Returns:
        按 :data:`CHARS_PER_TOKEN` 向上取整得到的 token 数。
    """
    return -(-len(text) // CHARS_PER_TOKEN)


def estimate_content_tokens(content: Any) -> int:
    """估算字符串或文本内容列表的 token 数.

    Args:
        content: 字符串，或文本内容块列表。

    Returns:
        按 :data:`CHARS_PER_TOKEN` 向上取整得到的 token 数。
    """
    return -(-_content_chars(content) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: Any) -> int:
    """估算单条消息对上下文贡献的 token 数.

    Args:
        message: 任意角色的 transcript 消息。

    Returns:
        该消息的估算 token 数；system message 会连同其工具声明一并计入。
    """
    role = getattr(message, "role", None)
    if role == "system":
        return (
            estimate_text_tokens(get_system_message_text(message))
            + _estimate_tools_tokens(message.tools_added)
            + _estimate_tools_tokens(message.tools_removed)
        )
    if role in ("user", "toolResult"):
        return estimate_content_tokens(message.content)

    chars = 0
    for block in message.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            chars += len(block.text)
        elif block_type == "thinking":
            chars += len(block.thinking)
        else:
            chars += len(block.name) + len(_safe_json(block.arguments))
    return -(-chars // CHARS_PER_TOKEN)


def _estimate_tools_tokens(tools: Sequence[Any] | None) -> int:
    """估算一组工具声明的 token 数.

    Args:
        tools: 工具声明列表；为空或 ``None`` 时计为 0。

    Returns:
        工具声明序列化后的估算 token 数。
    """
    if not tools:
        return 0
    from .serde import to_json

    return estimate_text_tokens(_safe_json([to_json(tool) for tool in tools]))


def _last_assistant_usage(messages: Sequence[Any]) -> tuple[Usage, int] | None:
    """找出最近一个仍能描述当前前缀的 assistant usage 块.

    Args:
        messages: transcript 的消息列表。

    Returns:
        ``(usage, index)``；不存在适用块时返回 ``None``。
    """
    latest_prefix_timestamp = float("-inf")
    usage_info: tuple[Usage, int] | None = None

    for index, message in enumerate(messages):
        if getattr(message, "role", None) == "assistant":
            # 这条响应之后又插入了更新的前缀消息（例如压缩摘要），
            # 因此它的 usage 已无法描述当前前缀。
            applies_to_prefix = message.timestamp >= latest_prefix_timestamp
            if (
                applies_to_prefix
                and message.stop_reason not in ("aborted", "error")
                and calculate_context_tokens(message.usage) > 0
            ):
                usage_info = (message.usage, index)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp)

    return usage_info


def estimate_context_tokens(context: TranscriptContext | Sequence[Any]) -> ContextUsageEstimate:
    """估算上下文大小，以最近一个适用的 usage 块为锚点.

    Args:
        context: ``TranscriptContext``，或直接给出的消息列表。

    Returns:
        含总 token 数、锚点 usage 与尾部估算的 :class:`ContextUsageEstimate`。
    """
    messages = context.messages if isinstance(context, TranscriptContext) else context
    usage_info = _last_assistant_usage(messages)
    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing = sum(estimate_message_tokens(message) for message in messages[index + 1 :])
        return ContextUsageEstimate(
            tokens=usage_tokens + trailing,
            usage_tokens=usage_tokens,
            trailing_tokens=trailing,
            last_usage_index=index,
        )

    tokens = sum(estimate_message_tokens(message) for message in messages)
    return ContextUsageEstimate(tokens=tokens, usage_tokens=0, trailing_tokens=tokens, last_usage_index=None)
