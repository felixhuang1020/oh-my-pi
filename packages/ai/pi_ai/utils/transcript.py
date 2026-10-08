"""transcript 归一化与 system message 重放.

移植自 ``packages/ai/src/utils/transcript.ts``。transcript 是 prompt 与工具集的
唯一事实来源：首条 system message *就是* system prompt，其后的 system message
描述对它的变更，按序重放即可得到当前的 prompt 与工具集。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

from ..types import Context, Message, SystemMessage, Tool, ToolReference, TranscriptContext
from .serde import to_json
from .text import content_text, get_system_message_text

__all__ = [
    "ToolStateChanges",
    "TranscriptMessages",
    "TranscriptTools",
    "collapse_system_messages",
    "create_initial_system_message",
    "declarations_equal",
    "get_current_system_message",
    "get_current_system_prompt",
    "get_current_tools",
    "get_declared_tools",
    "get_initial_system_message",
    "get_tool_state_changes",
    "has_non_additive_tool_changes",
    "has_tool_redefinitions",
    "normalize_context",
    "resolve_transcript",
    "resolve_transcript_tools",
    "to_tool_declaration",
    "without_initial_system_message",
]


@runtime_checkable
class _HasRole(Protocol):
    """重放辅助函数可能检视的消息的结构化视图."""

    role: str


#: 任意消息列表；重放辅助函数只读取 role 为 ``"system"`` 的条目。
TranscriptMessages: TypeAlias = Sequence[_HasRole]


def create_initial_system_message(system_prompt: str | None, tools: list[Tool] | None) -> SystemMessage | None:
    """为 prompt 与工具集构建首条 system message.

    两者都为空时返回 ``None``，使空 transcript 保持为空。

    Args:
        system_prompt: 初始 system prompt 文本；为 ``None`` 或空串时不写入 prompt。
        tools: 初始工具声明列表；为 ``None`` 或空列表时不写入工具。

    Returns:
        构建出的首条 system message；prompt 与工具都为空时返回 ``None``。
    """
    has_system_prompt = system_prompt is not None and len(system_prompt) > 0
    has_tools = tools is not None and len(tools) > 0
    if not has_system_prompt and not has_tools:
        return None
    return SystemMessage(
        role="system",
        content=system_prompt if system_prompt is not None else "",
        tools_added=list(tools) if has_tools else None,
        timestamp=0,
    )


def normalize_context(context: Context) -> TranscriptContext:
    """将 Context.system_prompt 和 Context.tools 折叠到前导系统消息中.

    这是生成 TranscriptContext 的唯一入口；所有面向 provider 的函数都
    期望得到结果。

    **幂等性**：传入已经规范化过的 TranscriptContext 会原样返回。
    TypeScript 原版因为读取纯 ``{messages}`` 对象的 ``systemPrompt``/``tools``
    会得到 ``undefined``，所以天然具备幂等性。Python 需要显式探测可选字段。

    Args:
        context: 输入的 Context 对象。

    Returns:
        规范化后的 TranscriptContext。
    """
    system_prompt = getattr(context, "system_prompt", None)
    tools = getattr(context, "tools", None)
    initial_message = create_initial_system_message(system_prompt, tools)
    messages = [initial_message, *context.messages] if initial_message else list(context.messages)
    return TranscriptContext(messages=messages)


def _is_system_message(message: Any) -> bool:
    """判断 ``message`` 是否为 system message。

    Args:
        message: 待检查的消息对象。

    Returns:
        其 ``role`` 属性等于 ``"system"`` 时返回 ``True``。
    """
    return getattr(message, "role", None) == "system"


def get_initial_system_message(messages: Any) -> SystemMessage | None:
    """返回首条 system message（若 transcript 以其开头）。

    Args:
        messages: transcript 的消息列表。

    Returns:
        首条 system message；列表为空或首条不是 system message 时返回 ``None``。
    """
    if len(messages) == 0:
        return None
    first = messages[0]
    return first if _is_system_message(first) else None


def without_initial_system_message(messages: list[Message]) -> list[Message]:
    """为把 prompt 放在列表之外的 API 去掉首条 system message。

    Args:
        messages: transcript 的消息列表。

    Returns:
        去掉首条 system message 后的新列表；首条不是 system message 时原样返回。
    """
    return messages[1:] if get_initial_system_message(messages) else messages


def get_current_tools(messages: Any) -> list[Tool]:
    """按序应用 transcript 中的每次增量，解析出当前可用的工具。

    Args:
        messages: transcript 的消息列表。

    Returns:
        当前可用的工具列表，顺序与各工具最近一次被新增的顺序一致。
    """
    tools: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_removed or []:
            tools.pop(tool.name, None)
        for tool in message.tools_added or []:
            tools[tool.name] = tool
    return list(tools.values())


def get_current_system_message(messages: Any) -> SystemMessage | None:
    """把所有 system message 重放为一条携带当前 prompt 与工具的消息.

    后续 ``content`` 追加到基础 prompt，``sections`` 按名打补丁，
    工具则经 :func:`get_current_tools` 解析。

    Args:
        messages: transcript 的消息列表。

    Returns:
        重放得到的 system message；没有任何 system message 且无工具时
        返回 ``None``。
    """
    content: list[str] = []
    sections: dict[str, str] = {}
    timestamp: int | None = None
    for message in messages:
        if not _is_system_message(message):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        text = content_text(message.content)
        if len(text) > 0:
            content.append(text)
        for name, value in (message.sections or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
    tools = get_current_tools(messages)
    if timestamp is None and len(tools) == 0:
        return None
    return SystemMessage(
        role="system",
        content="\n\n".join(content),
        sections=dict(sections) if sections else None,
        tools_added=tools if tools else None,
        timestamp=timestamp if timestamp is not None else 0,
    )


def get_current_system_prompt(messages: Any) -> str:
    """重放全部 system message 后渲染当前 system prompt 文本.

    Args:
        messages: transcript 的消息列表。

    Returns:
        渲染出的 system prompt 文本；不存在 system message 时返回空串。
    """
    message = get_current_system_message(messages)
    return get_system_message_text(message) if message else ""


def collapse_system_messages(context: TranscriptContext) -> TranscriptContext:
    """为不支持对话中途 system message 的 API 重建 transcript.

    重放出的 system message 置于开头，其后的 system message 全部丢弃。

    Args:
        context: 待折叠的 TranscriptContext。

    Returns:
        只保留开头一条重放 system message 的新 TranscriptContext。
    """
    head = get_current_system_message(context.messages)
    messages = [message for message in context.messages if getattr(message, "role", None) != "system"]
    return TranscriptContext(messages=[head, *messages] if head else messages)


def resolve_transcript(
    context: TranscriptContext,
    supports_mid_convo_system_messages: bool | None,
) -> TranscriptContext:
    """模型接受时保留后续 system message，否则将它们折叠起来.

    Args:
        context: 已规范化的 TranscriptContext。
        supports_mid_convo_system_messages: 目标模型是否接受对话中途的
            system message；为假值（含 ``None``）时执行折叠。

    Returns:
        可直接发给 provider 的 TranscriptContext。
    """
    return context if supports_mid_convo_system_messages else collapse_system_messages(context)


def to_tool_declaration(tool: Tool) -> Tool:
    """在比较或持久化前，去掉工具中仅供执行与展示的字段.

    Args:
        tool: 待精简的工具声明。

    Returns:
        只保留向模型声明所需的 ``name``/``description``/``parameters``/
        ``constrained_sampling`` 的新 Tool。
    """
    parameters = to_json(tool.parameters)
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=parameters,
        constrained_sampling=tool.constrained_sampling,
    )


def declarations_equal(left: Tool, right: Tool) -> bool:
    """判断两个工具向模型声明的接口是否相同.

    两侧先经 :func:`to_tool_declaration`，再比较其序列化结果。
    :func:`to_json` 以确定顺序输出字段，因此这是精确的结构比较，
    无需引入深度相等依赖。

    Args:
        left: 左侧工具声明。
        right: 右侧工具声明。

    Returns:
        两者向模型声明的接口完全相同时返回 ``True``。
    """
    import json

    return json.dumps(to_json(to_tool_declaration(left)), sort_keys=True) == json.dumps(
        to_json(to_tool_declaration(right)), sort_keys=True
    )


class ToolStateChanges:
    """两份完整工具状态之间的增量.

    Attributes:
        tools_added: 新增或定义发生变化的工具。
        tools_removed: 被移除或定义发生变化的工具引用。
    """

    __slots__ = ("tools_added", "tools_removed")

    def __init__(self, tools_added: list[Tool], tools_removed: list[ToolReference]) -> None:
        """初始化工具状态增量。

        Args:
            tools_added: 新增或定义发生变化的工具。
            tools_removed: 被移除或定义发生变化的工具引用。
        """
        self.tools_added = tools_added
        self.tools_removed = tools_removed


def get_tool_state_changes(previous: list[Tool], current: list[Tool]) -> ToolStateChanges:
    """比较两份完整的工具状态.

    定义发生变化计为一次移除加一次新增。

    Args:
        previous: 变更前的完整工具列表。
        current: 变更后的完整工具列表。

    Returns:
        描述新增与移除的 :class:`ToolStateChanges`。
    """
    previous_tools = {tool.name: tool for tool in previous}
    current_tools = {tool.name: tool for tool in current}
    added = [
        to_tool_declaration(tool)
        for tool in current
        if (previous_tool := previous_tools.get(tool.name)) is None or not declarations_equal(previous_tool, tool)
    ]
    removed = [
        ToolReference(name=tool.name)
        for tool in previous
        if (current_tool := current_tools.get(tool.name)) is None or not declarations_equal(tool, current_tool)
    ]
    return ToolStateChanges(tools_added=added, tools_removed=removed)


def get_declared_tools(messages: Any) -> list[Tool]:
    """transcript 工具状态引用的全部定义，按首次声明的顺序.

    Args:
        messages: transcript 的消息列表。

    Returns:
        各工具名的最后一次声明，顺序为首次声明时的顺序。
    """
    definitions: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            definitions[tool.name] = tool
    return list(definitions.values())


def has_tool_redefinitions(messages: Any) -> bool:
    """是否存在同名工具被以不同定义声明两次.

    按名称引用工具的传输方式（Anthropic 的 ``tool_addition``/
    ``tool_removal``）无法表达这种情况。

    Args:
        messages: transcript 的消息列表。

    Returns:
        存在同名但定义不同的重复声明时返回 ``True``。
    """
    declared: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            previous = declared.get(tool.name)
            if previous is not None and not declarations_equal(previous, tool):
                return True
            declared[tool.name] = tool
    return False


def has_non_additive_tool_changes(messages: Any) -> bool:
    """工具历史中是否存在仅支持新增的传输方式无法重放的残余变更.

    Args:
        messages: transcript 的消息列表。

    Returns:
        出现工具移除或同名重复声明时返回 ``True``。
    """
    declared: set[str] = set()
    for message in messages:
        if not _is_system_message(message):
            continue
        if len(message.tools_removed or []) > 0:
            return True
        for tool in message.tools_added or []:
            if tool.name in declared:
                return True
            declared.add(tool.name)
    return False


class TranscriptTools:
    """拆分到顶层请求字段与原地新增两处的工具声明.

    Attributes:
        request_tools: 应放入顶层请求字段的工具列表。
        anchors_additions: 新增是否锚定在 system message 上（即是否可延迟加载）。
    """

    __slots__ = ("request_tools", "anchors_additions")

    def __init__(self, request_tools: list[Tool], anchors_additions: bool) -> None:
        """初始化拆分结果。

        Args:
            request_tools: 应放入顶层请求字段的工具列表。
            anchors_additions: 新增是否锚定在 system message 上。
        """
        self.request_tools = request_tools
        self.anchors_additions = anchors_additions


def resolve_transcript_tools(messages: Any, supports_tool_additions: bool) -> TranscriptTools:
    """把工具声明拆分到请求字段与原地新增两处.

    能把新增锚定在 system message 上的传输方式会在顶层保留初始工具、
    后续工具在其出现处加载；这仅在没有工具被移除或重复声明时可行，
    其余情况一律发送当前工具列表。

    Args:
        messages: transcript 的消息列表。
        supports_tool_additions: 目标传输方式是否支持在对话中途新增工具。

    Returns:
        拆分后的 :class:`TranscriptTools`。
    """
    anchors_additions = supports_tool_additions and not has_non_additive_tool_changes(messages)
    if anchors_additions:
        initial = get_initial_system_message(messages)
        request_tools = list(initial.tools_added) if initial and initial.tools_added else []
    else:
        request_tools = get_current_tools(messages)
    return TranscriptTools(request_tools=request_tools, anchors_additions=anchors_additions)
