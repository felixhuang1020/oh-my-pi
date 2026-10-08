"""紧凑、可重放的 assistant message 进度日志.

移植自 ``packages/ai/src/utils/assistant-message-frame.ts``。帧序列是一次 assistant
响应的无损内容日志：记录块的开始、消费者尚未见过的 delta，以及权威的块结束。
终态结算被刻意排除在外、必须单独持久化，因此
:class:`AssistantMessageFrameEncoder` 对 ``done``/``error`` 事件返回 ``None``。

编码器是带状态的。事件被消费期间 ``partial`` 始终是共享的活动累积器，
编码器记录每个块的偏移（``covered_chars``/``delta_chars``），当较旧的排队事件
到达时，绝不会重复发出 partial 中已可见的 delta。

每个帧 dataclass 都携带 ``type`` 判别默认值，使
:func:`pi_ai.utils.serde.from_json` 能重建 :data:`AssistantMessageFrame`
union 中正确的成员。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from ..types import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
    JsonObject,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .json_parse import parse_streaming_json
from .serde import clone

__all__ = [
    "AssistantMessageFrame",
    "AssistantMessageFrameEncoder",
    "StartFrame",
    "TextDeltaFrame",
    "TextEndFrame",
    "TextStartFrame",
    "ThinkingDeltaFrame",
    "ThinkingEndFrame",
    "ThinkingStartFrame",
    "ToolCallCheckpointFrame",
    "ToolCallDeltaFrame",
    "ToolCallEndFrame",
    "ToolCallStartFrame",
    "reduce_assistant_message_frames",
]


# --------------------------------------------------------------------------------------
# 帧（Frames）
# --------------------------------------------------------------------------------------


@dataclass
class StartFrame:
    """流开始；携带不含内容的消息快照.

    Attributes:
        partial: 不含内容的消息快照，作为重放时的初始状态。
        type: 帧判别值，固定为 ``"start"``。
    """

    partial: AssistantMessage
    type: str = "start"


@dataclass
class TextStartFrame:
    """text 块开始.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        content: 该 text 块的初始内容。
        type: 帧判别值，固定为 ``"text_start"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: TextContent = field(default_factory=TextContent)
    type: str = "text_start"


@dataclass
class TextDeltaFrame:
    """块中尚未可见的文本.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        delta: 消费者尚未见过的文本增量。
        type: 帧判别值，固定为 ``"text_delta"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "text_delta"


@dataclass
class TextEndFrame:
    """text 块结束，携带权威内容与签名.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        content: 权威的完整文本，重放时覆盖此前的增量。
        text_signature: provider 返回的文本签名；无签名时为 ``None``。
        type: 帧判别值，固定为 ``"text_end"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: str = ""
    text_signature: str | None = field(default=None, metadata={"alias": "textSignature"})
    type: str = "text_end"


@dataclass
class ThinkingStartFrame:
    """thinking 块开始.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        content: 该 thinking 块的初始内容。
        type: 帧判别值，固定为 ``"thinking_start"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: ThinkingContent = field(default_factory=ThinkingContent)
    type: str = "thinking_start"


@dataclass
class ThinkingDeltaFrame:
    """块中尚未可见的推理文本.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        delta: 消费者尚未见过的推理文本增量。
        type: 帧判别值，固定为 ``"thinking_delta"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "thinking_delta"


@dataclass
class ThinkingEndFrame:
    """thinking 块结束，携带权威内容与元数据.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        content: 权威的完整推理文本，重放时覆盖此前的增量。
        thinking_signature: provider 返回的推理签名；无签名时为 ``None``。
        redacted: 推理内容是否被 provider 脱敏；未知时为 ``None``。
        type: 帧判别值，固定为 ``"thinking_end"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: str = ""
    thinking_signature: str | None = field(default=None, metadata={"alias": "thinkingSignature"})
    redacted: bool | None = None
    type: str = "thinking_end"


@dataclass
class ToolCallStartFrame:
    """tool call 开始.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        tool_call: 该 tool call 的初始快照。
        type: 帧判别值，固定为 ``"toolcall_start"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    tool_call: ToolCall = field(default_factory=ToolCall, metadata={"alias": "toolCall"})
    type: str = "toolcall_start"


@dataclass
class ToolCallCheckpointFrame:
    """用紧凑形式代替此刻可见的全部 tool-call JSON.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        json: 追平开始快照时一次性给出的完整参数 JSON。
        type: 帧判别值，固定为 ``"toolcall_checkpoint"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    json: str = ""
    type: str = "toolcall_checkpoint"


@dataclass
class ToolCallDeltaFrame:
    """块中尚未可见的 tool-call JSON.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        delta: 消费者尚未见过的参数 JSON 增量。
        type: 帧判别值，固定为 ``"toolcall_delta"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "toolcall_delta"


@dataclass
class ToolCallEndFrame:
    """tool call 结束，携带权威标识与参数.

    Attributes:
        content_index: 块在 ``partial.content`` 中的下标。
        id: provider 给出的权威 tool call 标识。
        name: 工具名。
        arguments: 权威的调用参数对象。
        thought_signature: provider 返回的思考签名；无签名时为 ``None``。
        namespace: 工具命名空间；无命名空间时为 ``None``。
        type: 帧判别值，固定为 ``"toolcall_end"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    id: str = ""
    name: str = ""
    arguments: JsonObject = field(default_factory=dict)
    thought_signature: str | None = field(default=None, metadata={"alias": "thoughtSignature"})
    namespace: str | None = None
    type: str = "toolcall_end"


#: 可重放的 assistant message 进度日志中的全部帧类型。
AssistantMessageFrame: TypeAlias = (
    StartFrame
    | TextStartFrame
    | TextDeltaFrame
    | TextEndFrame
    | ThinkingStartFrame
    | ThinkingDeltaFrame
    | ThinkingEndFrame
    | ToolCallStartFrame
    | ToolCallCheckpointFrame
    | ToolCallDeltaFrame
    | ToolCallEndFrame
)

#: 携带 ``contentIndex`` 指向 ``partial.content`` 的流事件。
_BlockEvent: TypeAlias = (
    TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
)


# --------------------------------------------------------------------------------------
# 共用辅助
# --------------------------------------------------------------------------------------

#: JavaScript 视为安全数组下标的最大整数（``Number.isSafeInteger``）。
_MAX_SAFE_INTEGER = 2**53 - 1


def _clone_text_content(content: TextContent) -> TextContent:
    """复制 text 内容，避免帧与共享 ``partial`` 共享可变状态.

    Args:
        content: 待复制的 text 内容块。

    Returns:
        仅含 ``text`` 与 ``text_signature`` 的新内容块。
    """
    return TextContent(text=content.text, text_signature=content.text_signature)


def _clone_thinking_content(content: ThinkingContent) -> ThinkingContent:
    """复制 thinking 内容，避免帧与共享 ``partial`` 共享可变状态.

    Args:
        content: 待复制的 thinking 内容块。

    Returns:
        字段一一对应的新 thinking 内容块。
    """
    return ThinkingContent(
        thinking=content.thinking,
        thinking_signature=content.thinking_signature,
        redacted=content.redacted,
    )


def _clone_tool_call(tool_call: ToolCall) -> ToolCall:
    """深拷贝 tool call 的参数对象，避免后续 delta 修改已发出的帧.

    Args:
        tool_call: 待复制的 tool call。

    Returns:
        参数已深拷贝的新 tool call。
    """
    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=clone(tool_call.arguments),
        thought_signature=tool_call.thought_signature,
        namespace=tool_call.namespace,
    )


def _clone_start_message(message: AssistantMessage) -> AssistantMessage:
    """构造清空内容、``stop_reason`` 为 ``"pending"`` 的开始帧消息快照.

    Args:
        message: 触发 ``start`` 事件时携带的消息。

    Returns:
        保留元数据与 usage、但内容为空且未结算的新消息。
    """
    return AssistantMessage(
        api=message.api,
        provider=message.provider,
        model=message.model,
        content=[],
        usage=clone(message.usage),
        stop_reason="pending",
        timestamp=message.timestamp,
        response_model=message.response_model,
        response_id=message.response_id,
        provider_thinking_level=message.provider_thinking_level,
        diagnostics=None if message.diagnostics is None else clone(message.diagnostics),
    )


def _assert_content_index(content_index: int) -> None:
    """校验块下标是合法的 JavaScript 安全整数。

    Args:
        content_index: 待校验的块下标。

    Raises:
        ValueError: 当 ``content_index`` 不是整数、为布尔值、为负数或超出安全整数范围时。
    """
    if (
        isinstance(content_index, bool)
        or not isinstance(content_index, int)
        or content_index < 0
        or content_index > _MAX_SAFE_INTEGER
    ):
        raise ValueError(f"Invalid assistant message frame contentIndex: {content_index}")


def _event_block(event: _BlockEvent) -> AssistantContent:
    """取出流事件 ``contentIndex`` 指向的内容块。

    Args:
        event: 携带 ``contentIndex`` 的流事件。

    Returns:
        ``event.partial.content`` 中对应下标的内容块。

    Raises:
        ValueError: 当 ``contentIndex`` 非法，或该下标处没有内容块时。
    """
    content_index = event.content_index
    _assert_content_index(content_index)
    partial = event.partial
    block = partial.content[content_index] if content_index < len(partial.content) else None
    if block is None:
        raise ValueError(f"{event.type} event has no content block at index {content_index}")
    return block


def _serialized_arguments(arguments_value: Any) -> str:
    """工具调用参数的 ``JSON.stringify``（紧凑分隔符、保留非 ASCII 字符）.

    Args:
        arguments_value: 待序列化的参数对象。

    Returns:
        紧凑 JSON 文本。

    Raises:
        ValueError: 当参数对象无法 JSON 序列化时。
    """
    try:
        return json.dumps(arguments_value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Tool-call arguments are not JSON-serializable") from error


_EMPTY_PARSED_TOOL_ARGUMENTS = _serialized_arguments(parse_streaming_json(""))


def _json_identical(snapshot: Any, current: Any) -> bool:
    """对 tool-call 快照可能包含的 JSON 标量做 ``Object.is`` 式比较.

    Args:
        snapshot: 快照中的 JSON 值。
        current: 当前解析出的 JSON 值。

    Returns:
        两个标量是否按 ``Object.is`` 语义相等。
    """
    if isinstance(snapshot, bool) or isinstance(current, bool):
        return snapshot is current
    if isinstance(snapshot, (int, float)) and isinstance(current, (int, float)):
        return snapshot == current
    return type(snapshot) is type(current) and snapshot == current


def _is_json_prefix(snapshot: Any, current: Any) -> bool:
    """``snapshot`` 是否为 ``current`` 的结构前缀.

    Args:
        snapshot: 开始帧时的 JSON 快照。
        current: 当前累积解析出的 JSON 值。

    Returns:
        ``snapshot`` 是否递归地构成 ``current`` 的前缀。
    """
    if isinstance(snapshot, str):
        return isinstance(current, str) and current.startswith(snapshot)
    if isinstance(snapshot, list):
        return (
            isinstance(current, list)
            and len(snapshot) <= len(current)
            and all(_is_json_prefix(value, current[index]) for index, value in enumerate(snapshot))
        )
    if not isinstance(snapshot, dict):
        return _json_identical(snapshot, current)
    if not isinstance(current, dict):
        return False
    return all(key in current and _is_json_prefix(value, current[key]) for key, value in snapshot.items())


# --------------------------------------------------------------------------------------
# 编码器（Encoder）
# --------------------------------------------------------------------------------------


@dataclass
class _TextEncoderBlockState:
    """text 或 thinking 块的偏移记账.

    Attributes:
        kind: 块类别，取 ``"text"`` 或 ``"thinking"``。
        covered_chars: 已包含在共享 ``partial`` 中、消费者可见的字符数。
        delta_chars: 已发出的增量累计字符数，用于定位下一个未覆盖位置。
    """

    kind: str
    covered_chars: int = 0
    delta_chars: int = 0


@dataclass
class _ToolCallEncoderBlockState:
    """JSON delta 流是否已追平开始时的快照.

    Attributes:
        kind: 块类别，固定为 ``"toolCall"``。
        caught_up: delta 流是否已追平 ``toolcall_start`` 时的参数快照。
        catchup_json: 追平过程中累积的 JSON 文本。
        snapshot_arguments: 开始帧的参数快照（已序列化）；无需追平时为空串。
    """

    kind: str = "toolCall"
    caught_up: bool = False
    catchup_json: str = ""
    snapshot_arguments: str = ""


_EncoderBlockState: TypeAlias = _TextEncoderBlockState | _ToolCallEncoderBlockState


class AssistantMessageFrameEncoder:
    """编码一条 assistant 流.

    ``partial`` 始终是共享的活动累积器；编码器用每块偏移避免在消费较旧的
    排队事件时重放已可见的 delta。

    Attributes:
        _started: 是否已见过 ``start`` 事件。
        _terminal: 是否已见过 ``done``/``error`` 等终态事件。
        _blocks: 按块下标记录的活动块编码状态。
    """

    def __init__(self) -> None:
        """初始化尚未开始、无活动块的编码器."""
        self._started = False
        self._terminal = False
        self._blocks: dict[int, _EncoderBlockState] = {}

    def encode(self, event: AssistantMessageEvent) -> AssistantMessageFrame | None:
        """把 ``event`` 转为帧；不产生输出时返回 ``None``.

        Args:
            event: 待编码的 assistant message 流事件。

        Returns:
            对应的帧；终态事件与无需输出的 delta 返回 ``None``。

        Raises:
            ValueError: 当事件顺序非法（重复 ``start``、终态后仍有事件、块下标冲突
                或事件类型与块类型不符）时。
        """
        if self._terminal:
            raise ValueError(f"Assistant message event {event.type} follows a terminal event")

        if isinstance(event, StartEvent):
            if self._started:
                raise ValueError("Assistant message stream contains more than one start event")
            self._started = True
            return StartFrame(partial=_clone_start_message(event.partial))
        if isinstance(event, DoneEvent):
            if not self._started:
                raise ValueError("Assistant message done event appears before start")
            self._terminal = True
            return None
        if isinstance(event, ErrorEvent):
            self._terminal = True
            return None

        if not self._started:
            raise ValueError(f"Assistant message {event.type} event appears before start")

        if isinstance(event, TextStartEvent):
            content = _event_block(event)
            if not isinstance(content, TextContent):
                raise ValueError(f"text_start event points to {content.type} block at index {event.content_index}")
            self._start_block(
                event.content_index,
                _TextEncoderBlockState(kind="text", covered_chars=len(content.text), delta_chars=0),
            )
            return TextStartFrame(content_index=event.content_index, content=_clone_text_content(content))

        if isinstance(event, TextDeltaEvent):
            return self._encode_text_delta(event.content_index, event.delta, "text")

        if isinstance(event, TextEndEvent):
            content = _event_block(event)
            if not isinstance(content, TextContent):
                raise ValueError(f"text_end event points to {content.type} block at index {event.content_index}")
            self._end_block(event.content_index, "text")
            return TextEndFrame(
                content_index=event.content_index,
                content=event.content,
                text_signature=content.text_signature,
            )

        if isinstance(event, ThinkingStartEvent):
            content = _event_block(event)
            if not isinstance(content, ThinkingContent):
                raise ValueError(
                    f"thinking_start event points to {content.type} block at index {event.content_index}"
                )
            self._start_block(
                event.content_index,
                _TextEncoderBlockState(kind="thinking", covered_chars=len(content.thinking), delta_chars=0),
            )
            return ThinkingStartFrame(content_index=event.content_index, content=_clone_thinking_content(content))

        if isinstance(event, ThinkingDeltaEvent):
            return self._encode_text_delta(event.content_index, event.delta, "thinking")

        if isinstance(event, ThinkingEndEvent):
            content = _event_block(event)
            if not isinstance(content, ThinkingContent):
                raise ValueError(f"thinking_end event points to {content.type} block at index {event.content_index}")
            self._end_block(event.content_index, "thinking")
            return ThinkingEndFrame(
                content_index=event.content_index,
                content=event.content,
                thinking_signature=content.thinking_signature,
                redacted=content.redacted,
            )

        if isinstance(event, ToolCallStartEvent):
            content = _event_block(event)
            if not isinstance(content, ToolCall):
                raise ValueError(f"toolcall_start event points to {content.type} block at index {event.content_index}")
            snapshot_arguments = _serialized_arguments(content.arguments)
            caught_up = snapshot_arguments == _EMPTY_PARSED_TOOL_ARGUMENTS
            self._start_block(
                event.content_index,
                _ToolCallEncoderBlockState(
                    caught_up=caught_up,
                    catchup_json="",
                    snapshot_arguments="" if caught_up else snapshot_arguments,
                ),
            )
            return ToolCallStartFrame(content_index=event.content_index, tool_call=_clone_tool_call(content))

        if isinstance(event, ToolCallDeltaEvent):
            state = self._block(event.content_index, "toolCall")
            if not isinstance(state, _ToolCallEncoderBlockState):
                raise ValueError("Unreachable tool-call encoder state")
            if state.caught_up:
                if len(event.delta) == 0:
                    return None
                return ToolCallDeltaFrame(content_index=event.content_index, delta=event.delta)
            state.catchup_json += event.delta
            arguments_value = parse_streaming_json(state.catchup_json)
            if _serialized_arguments(arguments_value) != state.snapshot_arguments:
                # 旧版 grammar 调用会把初始输入放进 toolcall_start，但其 JSON delta 流
                # 仍从空输入开始。因此解析出的参数可能是对开始快照的扩展，而非精确复现。
                snapshot_arguments = parse_streaming_json(state.snapshot_arguments)
                if not _is_json_prefix(snapshot_arguments, arguments_value):
                    return None
            state.caught_up = True
            state.snapshot_arguments = ""
            frame_json = state.catchup_json
            state.catchup_json = ""
            if len(frame_json) == 0:
                return None
            return ToolCallCheckpointFrame(content_index=event.content_index, json=frame_json)

        if isinstance(event, ToolCallEndEvent):
            content = _event_block(event)
            if not isinstance(content, ToolCall):
                raise ValueError(f"toolcall_end event points to {content.type} block at index {event.content_index}")
            tool_call = event.tool_call
            if tool_call.type != "toolCall":
                raise ValueError(f"toolcall_end event has invalid tool call at index {event.content_index}")
            self._end_block(event.content_index, "toolCall")
            return ToolCallEndFrame(
                content_index=event.content_index,
                id=tool_call.id,
                name=tool_call.name,
                arguments=clone(tool_call.arguments),
                thought_signature=tool_call.thought_signature,
                namespace=tool_call.namespace,
            )

        # TypeScript 中针对其余事件类型的 switch 没有 default 分支，
        # 因此无法识别的事件不产生帧。
        return None

    # -- 块记账 -----------------------------------------------------------------------

    def _start_block(self, content_index: int, state: _EncoderBlockState) -> None:
        """登记一个刚开始的块。

        Args:
            content_index: 块下标。
            state: 该块的编码状态。

        Raises:
            ValueError: 当块下标非法或同一下标重复开始时。
        """
        _assert_content_index(content_index)
        if content_index in self._blocks:
            raise ValueError(f"Assistant message block {content_index} starts more than once")
        self._blocks[content_index] = state

    def _block(self, content_index: int, kind: str) -> _EncoderBlockState:
        """取出已开始且类别匹配的块状态。

        Args:
            content_index: 块下标。
            kind: 期望的块类别。

        Returns:
            该块对应的编码状态。

        Raises:
            ValueError: 当块下标非法、尚未开始或类别不匹配时。
        """
        _assert_content_index(content_index)
        state = self._blocks.get(content_index)
        if state is None:
            raise ValueError(f"Assistant message {kind} block {content_index} has not started")
        if state.kind != kind:
            raise ValueError(f"Assistant message block {content_index} is {state.kind}, not {kind}")
        return state

    def _end_block(self, content_index: int, kind: str) -> None:
        """结束并移除一个块。

        Args:
            content_index: 块下标。
            kind: 期望的块类别。

        Raises:
            ValueError: 当块尚未开始、类别不匹配或下标非法时。
        """
        self._block(content_index, kind)
        del self._blocks[content_index]

    def _encode_text_delta(
        self,
        content_index: int,
        delta: str,
        kind: str,
    ) -> AssistantMessageFrame | None:
        """按已覆盖字符数裁剪 delta，只发出消费者尚未见过的部分。

        Args:
            content_index: 块下标。
            delta: 本次事件携带的完整增量文本。
            kind: 块类别，取 ``"text"`` 或 ``"thinking"``。

        Returns:
            仅含未见部分的 delta 帧；全部已可见时返回 ``None``。

        Raises:
            ValueError: 当块尚未开始、类别不匹配或下标非法时。
        """
        state = self._block(content_index, kind)
        if not isinstance(state, _TextEncoderBlockState):
            raise ValueError("Unreachable text encoder state")
        delta_start = state.delta_chars
        state.delta_chars += len(delta)
        covered = max(0, state.covered_chars - delta_start)
        if covered >= len(delta):
            return None
        uncovered = delta if covered == 0 else delta[covered:]
        if kind == "text":
            return TextDeltaFrame(content_index=content_index, delta=uncovered)
        return ThinkingDeltaFrame(content_index=content_index, delta=uncovered)


# --------------------------------------------------------------------------------------
# 归约器（Reducer）
# --------------------------------------------------------------------------------------


@dataclass
class _TextReducerBlockState:
    """正在重放的 text 或 thinking 块.

    Attributes:
        kind: 块类别，取 ``"text"`` 或 ``"thinking"``。
        ended: 是否已收到对应的结束帧。
    """

    kind: str
    ended: bool = False


@dataclass
class _ToolCallReducerBlockState:
    """正在重放的 tool-call 块，附带尚未解析的 JSON 尾部.

    Attributes:
        kind: 块类别，固定为 ``"toolCall"``。
        ended: 是否已收到 ``toolcall_end`` 帧。
        json: 自上一次 checkpoint 起累积、尚未重新解析的参数 JSON。
    """

    kind: str = "toolCall"
    ended: bool = False
    json: str = ""


_ReducerBlockState: TypeAlias = _TextReducerBlockState | _ToolCallReducerBlockState


def _append_block(
    message: AssistantMessage,
    states: dict[int, _ReducerBlockState],
    content_index: int,
    block: AssistantContent,
    state: _ReducerBlockState,
) -> None:
    """把新块追加到重放中的消息，并登记其状态。

    Args:
        message: 正在重放的消息。
        states: 已登记块状态的映射。
        content_index: 新块下标。
        block: 待追加并克隆的内容块。
        state: 新块的归约状态。

    Raises:
        ValueError: 当下标非法、块已存在或会造成下标空洞时。
    """
    _assert_content_index(content_index)
    if content_index != len(message.content):
        reason = "already exists" if content_index < len(message.content) else "would leave a gap"
        raise ValueError(f"Cannot start assistant message block at index {content_index}: {reason}")
    message.content.append(clone(block))
    states[content_index] = state


def _active_block(
    message: AssistantMessage,
    states: dict[int, _ReducerBlockState],
    content_index: int,
    expected_kind: str,
    frame_type: str,
) -> tuple[AssistantContent, _ReducerBlockState]:
    """取出仍在进行、类型匹配的活动块及其状态。

    Args:
        message: 正在重放的消息。
        states: 已登记块状态的映射。
        content_index: 目标块下标。
        expected_kind: 期望的块类别。
        frame_type: 触发本次查找的帧类型，仅用于错误信息。

    Returns:
        ``(内容块, 归约状态)`` 二元组。

    Raises:
        ValueError: 当块未开始、类别不匹配或已结束后又收到该帧时。
    """
    _assert_content_index(content_index)
    state = states.get(content_index)
    block = message.content[content_index] if content_index < len(message.content) else None
    if state is None or block is None:
        raise ValueError(f"{frame_type} frame has no started block at index {content_index}")
    if state.kind != expected_kind or block.type != expected_kind:
        raise ValueError(
            f"{frame_type} frame expected {expected_kind} block at index {content_index}, found {block.type}"
        )
    if state.ended:
        raise ValueError(f"{frame_type} frame follows the end of block at index {content_index}")
    return block, state


def reduce_assistant_message_frames(frames: Iterable[AssistantMessageFrame]) -> AssistantMessage | None:
    """重放紧凑帧序列，不修改入参.

    可迭代对象中没有 start 帧时返回 ``None``。

    Args:
        frames: 已按发出顺序排列的帧序列。

    Returns:
        重放得到的消息；没有 ``start`` 帧时为 ``None``。

    Raises:
        ValueError: 当帧顺序非法（重复 start、start 前出现其他帧、块下标冲突、
            块类型不符或结束后仍有 delta）时。
    """
    message: AssistantMessage | None = None
    frame_before_start: str | None = None
    states: dict[int, _ReducerBlockState] = {}

    for frame in frames:
        if isinstance(frame, StartFrame):
            if message is not None:
                raise ValueError("Assistant message frame sequence contains more than one start frame")
            if frame_before_start is not None:
                raise ValueError(f"{frame_before_start} frame appears before the start frame")
            message = clone(frame.partial)
            continue
        if message is None:
            if frame_before_start is None:
                frame_before_start = frame.type
            continue

        if isinstance(frame, TextStartFrame):
            if not isinstance(frame.content, TextContent):
                raise ValueError(f"text_start frame contains {frame.content.type} content")
            _append_block(
                message,
                states,
                frame.content_index,
                frame.content,
                _TextReducerBlockState(kind="text", ended=False),
            )
            continue

        if isinstance(frame, TextDeltaFrame):
            block, _state = _active_block(message, states, frame.content_index, "text", frame.type)
            if not isinstance(block, TextContent):
                raise ValueError("Unreachable text frame state")
            block.text += frame.delta
            continue

        if isinstance(frame, TextEndFrame):
            block, state = _active_block(message, states, frame.content_index, "text", frame.type)
            if not isinstance(block, TextContent):
                raise ValueError("Unreachable text frame state")
            block.text = frame.content
            block.text_signature = frame.text_signature
            state.ended = True
            continue

        if isinstance(frame, ThinkingStartFrame):
            if not isinstance(frame.content, ThinkingContent):
                raise ValueError(f"thinking_start frame contains {frame.content.type} content")
            _append_block(
                message,
                states,
                frame.content_index,
                frame.content,
                _TextReducerBlockState(kind="thinking", ended=False),
            )
            continue

        if isinstance(frame, ThinkingDeltaFrame):
            block, _state = _active_block(message, states, frame.content_index, "thinking", frame.type)
            if not isinstance(block, ThinkingContent):
                raise ValueError("Unreachable thinking frame state")
            block.thinking += frame.delta
            continue

        if isinstance(frame, ThinkingEndFrame):
            block, state = _active_block(message, states, frame.content_index, "thinking", frame.type)
            if not isinstance(block, ThinkingContent):
                raise ValueError("Unreachable thinking frame state")
            block.thinking = frame.content
            block.thinking_signature = frame.thinking_signature
            block.redacted = frame.redacted
            state.ended = True
            continue

        if isinstance(frame, ToolCallStartFrame):
            if not isinstance(frame.tool_call, ToolCall):
                raise ValueError(f"toolcall_start frame contains {frame.tool_call.type} content")
            _append_block(
                message,
                states,
                frame.content_index,
                frame.tool_call,
                _ToolCallReducerBlockState(kind="toolCall", ended=False, json=""),
            )
            continue

        if isinstance(frame, ToolCallCheckpointFrame):
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame.type)
            if not isinstance(block, ToolCall) or not isinstance(state, _ToolCallReducerBlockState):
                raise ValueError("Unreachable tool-call checkpoint state")
            state.json = frame.json
            block.arguments = parse_streaming_json(frame.json)
            continue

        if isinstance(frame, ToolCallDeltaFrame):
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame.type)
            if not isinstance(block, ToolCall) or not isinstance(state, _ToolCallReducerBlockState):
                raise ValueError("Unreachable tool-call frame state")
            state.json += frame.delta
            continue

        if isinstance(frame, ToolCallEndFrame):
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame.type)
            if not isinstance(block, ToolCall):
                raise ValueError("Unreachable tool-call frame state")
            block.id = frame.id
            block.name = frame.name
            block.arguments = clone(frame.arguments)
            block.thought_signature = frame.thought_signature
            block.namespace = frame.namespace
            state.ended = True
            continue

    if message is None:
        return None

    for content_index, state in states.items():
        if not isinstance(state, _ToolCallReducerBlockState) or state.ended or len(state.json) == 0:
            continue
        block = message.content[content_index]
        if not isinstance(block, ToolCall):
            raise ValueError("Unreachable tool-call frame state")
        block.arguments = parse_streaming_json(state.json)

    return message
