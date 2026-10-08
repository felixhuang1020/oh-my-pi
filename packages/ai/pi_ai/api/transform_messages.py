"""provider 无关的 transcript 变换。

移植自 ``packages/ai/src/api/transform-messages.ts``。每个 adapter 在构建请求前都会
先对 transcript 执行 :func:`transform_messages`：为非视觉模型降级图片、丢弃或转换
无法跨模型重放的 thinking 块、规范化 tool-call id，并为孤立的 tool call 插入合成
结果，使每个 provider 看到的都是结构合法的对话。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..types import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ..utils.serde import clone

__all__ = [
    "transform_messages",
]

#: tool-call id 改写器签名：``(toolCallId, model, assistantMessage) -> normalizedId``。
ToolCallIdNormalizer = Callable[[str, Any, AssistantMessage], str]


def _transform_thinking_block(block: Any, is_same_model: bool) -> list[Any]:
    """针对目标模型保留、丢弃或降级一个 thinking 块。

    Args:
        block: 待处理的 thinking 块。
        is_same_model: 该消息是否来自当前目标模型。

    Returns:
        要保留的块列表：可能为空（丢弃）、原块（保留）或降级后的文本块。
    """
    # 加密的 thinking 是不透明内容，只对同一个模型有效。
    if block.redacted:
        return [block] if is_same_model else []
    # 同模型时保留带签名的 thinking 块（重放需要），即使可见文本为空——
    # OpenAI 的加密 reasoning 本身不含明文。
    if is_same_model and block.thinking_signature:
        return [block]
    if not block.thinking or not block.thinking.strip():
        return []
    if is_same_model:
        return [block]
    return [TextContent(type="text", text=block.thinking)]


def _transform_assistant_content(
    message: AssistantMessage,
    model: Model,
    is_same_model: bool,
    normalize_tool_call_id: ToolCallIdNormalizer | None,
    tool_call_id_map: dict[str, str],
) -> list[Any]:
    """重写一条 assistant 消息的 content，使其适配目标模型。

    Args:
        message: 待处理的 assistant 消息。
        model: 请求将要发往的模型。
        is_same_model: 该消息是否来自当前目标模型。
        normalize_tool_call_id: 可选的 tool-call id 改写器。
        tool_call_id_map: 记录 id 改写结果的映射，供后续 toolResult 复用。

    Returns:
        变换后的 content 块列表。
    """
    transformed: list[Any] = []
    for block in message.content:
        block_type = getattr(block, "type", None)
        if block_type == "thinking":
            transformed.extend(_transform_thinking_block(block, is_same_model))
            continue
        if block_type == "text":
            transformed.append(clone(block) if not is_same_model else block)
            continue
        if block_type == "toolCall":
            normalized: ToolCall = block
            if not is_same_model and block.thought_signature:
                normalized = clone(block)
                normalized.thought_signature = None
            if not is_same_model and normalize_tool_call_id is not None:
                normalized_id = normalize_tool_call_id(block.id, model, message)
                if normalized_id != block.id:
                    tool_call_id_map[block.id] = normalized_id
                    normalized = clone(normalized)
                    normalized.id = normalized_id
            transformed.append(normalized)
            continue
        transformed.append(block)
    return transformed


def _synthetic_tool_result(tool_call: ToolCall) -> ToolResultMessage:
    """为没有被应答的 tool call 合成一条空的错误结果。

    Args:
        tool_call: 尚无对应结果的 tool call。

    Returns:
        标记为 ``is_error`` 的合成 tool result 消息。
    """
    import time

    return ToolResultMessage(
        role="toolResult",
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=[TextContent(type="text", text="No result provided")],
        is_error=True,
        timestamp=int(time.time() * 1000),
    )


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: ToolCallIdNormalizer | None = None,
) -> list[Message]:
    """针对目标 ``model`` 规范化 transcript。

    Args:
        messages: 待变换的对话。
        model: 请求将要发往的模型。
        normalize_tool_call_id: 可选的 tool-call id 改写器，用于 provider 无法原样表达
            的 id（OpenAI Responses 的 id 远长于 Anthropic 要求的
            ``^[a-zA-Z0-9_-]{1,64}$`` 形式）。

    Returns:
        结构合法、可发送给目标模型的消息列表。
    """
    tool_call_id_map: dict[str, str] = {}
    # 规范化来自无类型调用方（自定义工具、手工拼的历史、旧 session 文件）的 null
    # content，使下游可以依赖类型契约。
    normalized_messages: list[Message] = [
        message if message.content is not None else replace(message, content=[]) for message in messages
    ]

    # 第一遍：转换 thinking 块、规范化 tool-call id。
    transformed: list[Message] = []
    for message in normalized_messages:
        if message.role in ("system", "user"):
            transformed.append(message)
            continue
        if message.role == "toolResult":
            normalized_id = tool_call_id_map.get(message.tool_call_id)
            if normalized_id is not None and normalized_id != message.tool_call_id:
                transformed.append(replace(message, tool_call_id=normalized_id))
            else:
                transformed.append(message)
            continue
        if message.role == "assistant":
            is_same_model = message.provider == model.provider and message.api == model.api and message.model == model.id
            transformed.append(
                replace(
                    message,
                    content=_transform_assistant_content(
                        message, model, is_same_model, normalize_tool_call_id, tool_call_id_map
                    ),
                )
            )
            continue
        transformed.append(message)

    # 第二遍：为孤立的 tool call 插入合成的空 tool result。这既保留了 thinking
    # 签名，也满足 API 的结构要求。
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()
    # system 消息对 tool-call 记账是透明的：夹在 tool call 与其结果之间的 system 消息
    # 会被暂存，等结果（含合成结果）发出后再发出，避免为之后才被应答的调用造成重复
    # 结果。
    held_system_messages: list[Message] = []

    def close_pending_tool_calls() -> None:
        """关闭当前挂起的 tool call：补发合成结果，并释放暂存的 system 消息。"""
        nonlocal pending_tool_calls, existing_tool_result_ids
        if pending_tool_calls:
            for tool_call in pending_tool_calls:
                if tool_call.id not in existing_tool_result_ids:
                    result.append(_synthetic_tool_result(tool_call))
            pending_tool_calls = []
            existing_tool_result_ids = set()
        result.extend(held_system_messages)
        held_system_messages.clear()

    for message in transformed:
        if message.role == "assistant":
            # 先关闭上一条 assistant 留下的孤立 tool call。
            close_pending_tool_calls()
            # 出错/被中止的 assistant 消息是不完整的轮次，不能重放：
            # 它们可能只有 reasoning 没有消息，或只有半个 tool call。
            if message.stop_reason in ("error", "aborted"):
                continue
            tool_calls = [block for block in message.content if getattr(block, "type", None) == "toolCall"]
            if tool_calls:
                pending_tool_calls = list(tool_calls)
                existing_tool_result_ids = set()
            result.append(message)
        elif message.role == "toolResult":
            existing_tool_result_ids.add(message.tool_call_id)
            result.append(message)
        elif message.role == "system":
            if pending_tool_calls:
                held_system_messages.append(message)
            else:
                result.append(message)
        elif message.role == "user":
            # 新的 user 轮次打断 tool 流程：先关闭孤立的调用。
            close_pending_tool_calls()
            result.append(message)
        else:
            result.append(message)

    # 对话以未解决的 tool call 结尾时，在此合成对应结果。
    close_pending_tool_calls()
    return result


# 重新导出，方便 adapter 只从本模块引用具体的消息类型。
__all__ += ["AssistantMessage", "ToolResultMessage", "UserMessage"]
