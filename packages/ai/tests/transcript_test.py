"""转录归一化与 system 消息回放测试。"""

from __future__ import annotations

from pi_ai.types import (
    AssistantMessage,
    Context,
    SystemMessage,
    TextContent,
    Tool,
    ToolReference,
    TranscriptContext,
    UserMessage,
)
from pi_ai.utils.transcript import (
    collapse_system_messages,
    create_initial_system_message,
    declarations_equal,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_declared_tools,
    get_initial_system_message,
    get_tool_state_changes,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
    resolve_transcript,
    resolve_transcript_tools,
    to_tool_declaration,
    without_initial_system_message,
)


def _tool(name: str, description: str = "d") -> Tool:
    return Tool(name=name, description=description, parameters={"type": "object", "properties": {}})


def test_create_initial_system_message_is_none_when_empty():
    assert create_initial_system_message(None, None) is None
    assert create_initial_system_message("", []) is None


def test_create_initial_system_message_carries_prompt_and_tools():
    message = create_initial_system_message("be nice", [_tool("a")])
    assert message is not None
    assert message.content == "be nice"
    assert [tool.name for tool in message.tools_added or []] == ["a"]
    assert message.timestamp == 0


def test_create_initial_system_message_with_tools_only():
    message = create_initial_system_message(undefined := None, [_tool("a")])
    assert message is not None and message.content == ""


def test_normalize_context_prepends_the_system_message():
    context = Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=1)], tools=[_tool("a")])
    transcript = normalize_context(context)
    assert isinstance(transcript, TranscriptContext)
    assert transcript.messages[0].role == "system"
    assert transcript.messages[1].role == "user"


def test_normalize_context_is_idempotent():
    # `compat.stream()` 归一化一次后把转录交给 `Models`，后者会再归一化一次；
    # TypeScript 能扛住是因为缺失字段读出来是 `undefined`。
    transcript = normalize_context(
        Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=1)], tools=[_tool("a")])
    )
    again = normalize_context(transcript)
    assert again.messages == transcript.messages
    assert [message.role for message in again.messages] == ["system", "user"]


def test_normalize_context_without_prompt_or_tools_keeps_messages():
    context = Context(messages=[UserMessage(content="hi", timestamp=1)])
    assert [message.role for message in normalize_context(context).messages] == ["user"]


def test_get_initial_and_without_initial_system_message():
    transcript = normalize_context(Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=1)]))
    head = get_initial_system_message(transcript.messages)
    assert head is not None and head.content == "sys"
    assert [message.role for message in without_initial_system_message(transcript.messages)] == ["user"]


def test_get_current_tools_applies_additions_and_removals():
    messages = [
        SystemMessage(content="", timestamp=0, tools_added=[_tool("a"), _tool("b")]),
        SystemMessage(content="", timestamp=1, tools_added=[_tool("c")], tools_removed=[ToolReference("a")]),
    ]
    assert [tool.name for tool in get_current_tools(messages)] == ["b", "c"]


def test_sections_are_patched_by_name_and_removed_with_null():
    messages = [
        SystemMessage(content="base", timestamp=0, sections={"a": "A", "b": "B"}),
        SystemMessage(content="", timestamp=1, sections={"b": None, "c": "C"}),
    ]
    current = get_current_system_message(messages)
    assert current is not None
    assert current.sections == {"a": "A", "c": "C"}
    assert get_current_system_prompt(messages) == "base\n\nA\n\nC"


def test_later_system_content_is_appended_to_the_base_prompt():
    messages = [
        SystemMessage(content="base", timestamp=0),
        SystemMessage(content="extra", timestamp=1),
    ]
    current = get_current_system_message(messages)
    assert current is not None and current.content == "base\n\nextra"


def test_collapse_system_messages_drops_later_system_messages():
    transcript = TranscriptContext(
        messages=[
            SystemMessage(content="a", timestamp=0),
            UserMessage(content="hi", timestamp=1),
            SystemMessage(content="b", timestamp=2),
        ]
    )
    collapsed = collapse_system_messages(transcript)
    assert [message.role for message in collapsed.messages] == ["system", "user"]
    assert collapsed.messages[0].content == "a\n\nb"


def test_resolve_transcript_keeps_mid_conversation_messages_when_supported():
    transcript = TranscriptContext(messages=[SystemMessage(content="a", timestamp=0), SystemMessage(content="b", timestamp=1)])
    assert len(resolve_transcript(transcript, True).messages) == 2
    assert len(resolve_transcript(transcript, False).messages) == 1
    assert len(resolve_transcript(transcript, None).messages) == 1


def test_declarations_equal_ignores_key_order():
    left = Tool(name="a", description="d", parameters={"type": "object", "properties": {"x": {"type": "string"}}})
    right = Tool(name="a", description="d", parameters={"properties": {"x": {"type": "string"}}, "type": "object"})
    assert declarations_equal(left, right)


def test_declarations_equal_detects_a_real_change():
    assert not declarations_equal(_tool("a", "one"), _tool("a", "two"))


def test_to_tool_declaration_drops_execution_fields():
    tool = _tool("a")
    tool.constrained_sampling = True
    declaration = to_tool_declaration(tool)
    assert declaration.constrained_sampling is True
    assert declaration.parameters == tool.parameters


def test_get_tool_state_changes_reports_changed_definitions_both_ways():
    changes = get_tool_state_changes([_tool("a", "old"), _tool("b")], [_tool("a", "new"), _tool("c")])
    assert [tool.name for tool in changes.tools_added] == ["a", "c"]
    assert [tool.name for tool in changes.tools_removed] == ["a", "b"]


def test_has_tool_redefinitions_and_non_additive_changes():
    redefined = [SystemMessage(content="", timestamp=0, tools_added=[_tool("a", "1")]), SystemMessage(content="", timestamp=1, tools_added=[_tool("a", "2")])]
    assert has_tool_redefinitions(redefined)
    assert has_non_additive_tool_changes(redefined)

    additive = [SystemMessage(content="", timestamp=0, tools_added=[_tool("a")]), SystemMessage(content="", timestamp=1, tools_added=[_tool("b")])]
    assert not has_tool_redefinitions(additive)
    assert not has_non_additive_tool_changes(additive)

    removed = [SystemMessage(content="", timestamp=0, tools_added=[_tool("a")]), SystemMessage(content="", timestamp=1, tools_removed=[ToolReference("a")])]
    assert has_non_additive_tool_changes(removed)


def test_get_declared_tools_keeps_first_declaration_order():
    messages = [
        SystemMessage(content="", timestamp=0, tools_added=[_tool("a"), _tool("b")]),
        SystemMessage(content="", timestamp=1, tools_added=[_tool("b"), _tool("c")]),
    ]
    assert [tool.name for tool in get_declared_tools(messages)] == ["a", "b", "c"]


def test_resolve_transcript_tools_anchors_additions_when_possible():
    messages = [
        SystemMessage(content="", timestamp=0, tools_added=[_tool("a")]),
        SystemMessage(content="", timestamp=1, tools_added=[_tool("b")]),
    ]
    anchored = resolve_transcript_tools(messages, True)
    assert anchored.anchors_additions is True
    assert [tool.name for tool in anchored.request_tools] == ["a"]

    collapsed = resolve_transcript_tools(messages, False)
    assert collapsed.anchors_additions is False
    assert [tool.name for tool in collapsed.request_tools] == ["a", "b"]


def test_resolve_transcript_tools_falls_back_when_removals_exist():
    messages = [
        SystemMessage(content="", timestamp=0, tools_added=[_tool("a")]),
        SystemMessage(content="", timestamp=1, tools_removed=[ToolReference("a")]),
    ]
    resolved = resolve_transcript_tools(messages, True)
    assert resolved.anchors_additions is False
    assert resolved.request_tools == []


def test_replay_reads_system_messages_and_ignores_custom_roles():
    transcript = TranscriptContext(
        messages=[SystemMessage(content="a", timestamp=0), UserMessage(content="u", timestamp=1)]
    )
    assert get_current_system_prompt(transcript.messages) == "a"
