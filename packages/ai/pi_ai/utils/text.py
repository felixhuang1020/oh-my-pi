"""文本提取与 system message 渲染."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["content_text", "get_system_message_text", "render_system_message_update"]


def content_text(content: str | Sequence[Any], separator: str = "\n") -> str:
    """从消息 content 中提取并拼接文本.

    ``content`` 要么是字符串，要么是内容块序列，其中只有 text 块参与拼接。

    Args:
        content: 消息内容，字符串或内容块序列。
        separator: 拼接各文本块时使用的分隔符。

    Returns:
        拼接后的文本。
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type != "text":
            continue
        text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
        parts.append(text)
    return separator.join(parts)


def get_system_message_text(message: Any) -> str:
    """把 system message 渲染成完整 prompt：正文加其后所有 section.

    Args:
        message: 带 ``content`` 与 ``sections`` 属性的 system message。

    Returns:
        正文与各非空 section 以空行拼接后的完整 prompt 文本。
    """
    parts = [content_text(message.content)]
    for text in (message.sections or {}).values():
        if text is not None:
            parts.append(text)
    return "\n\n".join(part for part in parts if len(part) > 0)


def render_system_message_update(message: Any) -> str:
    """为支持对话中途 system message 的 API 渲染更新后的 system message.

    section 的变更会带上名称，便于模型将其与开头的 prompt 对应起来。
    这种包装只存在于请求时刻，可能随版本变化。

    Args:
        message: 带 ``content`` 与 ``sections`` 属性的 system message；
            section 值为 ``None`` 表示该 section 已被移除。

    Returns:
        描述正文与各 section 新增、更新或移除情况的更新文本。
    """
    parts: list[str] = []
    text = content_text(message.content)
    if len(text) > 0:
        parts.append(text)
    for name, value in (message.sections or {}).items():
        if value is None:
            parts.append(f'Removed system prompt section "{name}".')
        else:
            parts.append(f'Updated system prompt section "{name}":\n\n{value}')
    return "\n\n".join(parts)
