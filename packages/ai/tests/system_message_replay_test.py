"""移植自 ``packages/ai/test/system-message-replay.test.ts``。

将转录中的 system 消息回放为开头的单条 prompt、为不支持会话中途 system 消息的 API 折叠
转录，以及比较 tool 声明。
"""

from __future__ import annotations

from dataclasses import replace

from pi_ai.types import (
    AssistantMessage,
    Context,
    Message,
    SystemMessage,
    TextContent,
    Tool,
    ToolReference,
    TranscriptContext,
    UserMessage,
)
from pi_ai.utils.text import get_system_message_text, render_system_message_update
from pi_ai.utils.transcript import (
    collapse_system_messages,
    declarations_equal,
    get_current_system_message,
    get_current_system_prompt,
    get_tool_state_changes,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
)


def tool(name: str, description: str | None = None) -> Tool:
    # TypeBox 的 ``Type.Object({})`` 对应普通 JSON Schema。
    return Tool(
        name=name,
        description=description if description is not None else f"{name} tool",
        parameters={"type": "object", "properties": {}},
    )


def _build_transcript() -> TranscriptContext:
    return normalize_context(
        Context(
            messages=[
                SystemMessage(
                    content="base",
                    sections={"a": "<a>1</a>", "b": "<b>1</b>"},
                    tools_added=[tool("first")],
                    timestamp=10,
                ),
                UserMessage(content="hello", timestamp=11),
                SystemMessage(content="also do this", timestamp=12),
                AssistantMessage(content=[TextContent(text="ok")], timestamp=13),
                SystemMessage(
                    content="",
                    sections={"a": "<a>2</a>", "b": None, "c": "<c>1</c>"},
                    tools_removed=[ToolReference(name="first")],
                    tools_added=[tool("second")],
                    timestamp=14,
                ),
            ]
        )
    )


TRANSCRIPT = _build_transcript()


# --------------------------------------------------------------------------------------
# describe("system message replay")：system 消息回放
# --------------------------------------------------------------------------------------


def test_replays_content_sections_and_tools_into_one_leading_message() -> None:
    current = get_current_system_message(TRANSCRIPT.messages)
    assert current == SystemMessage(
        role="system",
        content="base\n\nalso do this",
        sections={"a": "<a>2</a>", "c": "<c>1</c>"},
        tools_added=[tool("second")],
        timestamp=10,
    )
    assert get_current_system_prompt(TRANSCRIPT.messages) == "base\n\nalso do this\n\n<a>2</a>\n\n<c>1</c>"


def test_collapse_keeps_only_non_system_messages_after_the_replayed_head() -> None:
    collapsed = collapse_system_messages(TRANSCRIPT)
    assert [message.role for message in collapsed.messages] == ["system", "user", "assistant"]
    assert collapse_system_messages(collapsed) == collapsed


def test_replay_of_a_transcript_without_system_messages_is_empty() -> None:
    context = normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)]))
    assert get_current_system_message(context.messages) is None
    assert get_current_system_prompt(context.messages) == ""
    assert collapse_system_messages(context).messages == context.messages


def test_a_late_full_patch_on_a_transcript_without_a_leading_message_replays_as_the_prompt() -> None:
    context = normalize_context(
        Context(
            messages=[
                UserMessage(content="old session", timestamp=1),
                SystemMessage(
                    content="",
                    sections={"preamble": "You are pi."},
                    tools_added=[tool("x")],
                    timestamp=2,
                ),
            ]
        )
    )
    assert get_current_system_prompt(context.messages) == "You are pi."
    head = collapse_system_messages(context).messages[0]
    assert head.role == "system"
    assert head.tools_added == [tool("x")]


def test_renders_complete_prompts_and_framed_updates() -> None:
    leading = TRANSCRIPT.messages[0]
    update = TRANSCRIPT.messages[4]
    assert leading.role == "system" and update.role == "system"
    assert get_system_message_text(leading) == "base\n\n<a>1</a>\n\n<b>1</b>"
    assert render_system_message_update(update) == "\n\n".join(
        [
            'Updated system prompt section "a":\n\n<a>2</a>',
            'Removed system prompt section "b".',
            'Updated system prompt section "c":\n\n<c>1</c>',
        ]
    )


def test_normalizes_the_legacy_prompt_and_tool_fields_into_a_leading_system_message() -> None:
    messages: list[Message] = [UserMessage(content="hi", timestamp=1)]
    assert normalize_context(Context(messages=messages)).messages == messages
    assert normalize_context(Context(system_prompt="", tools=[], messages=messages)).messages == messages
    assert normalize_context(
        Context(system_prompt="be brief", tools=[tool("a")], messages=messages)
    ).messages == [
        SystemMessage(role="system", content="be brief", tools_added=[tool("a")], timestamp=0),
        *messages,
    ]


def test_compares_tool_declarations_without_executable_or_undefined_fields() -> None:
    # 上游夹具展开了一个可执行 tool：`execute` 会被 `toToolDeclaration` 剥离，
    # `constrainedSampling: undefined` 会被 `to_json` 丢弃。
    # Python 的 `Tool` 上不存在这两个字段，因此普通的 `tool("a")` 即等价写法。
    executable = tool("a")
    assert declarations_equal(executable, tool("a")) is True
    assert declarations_equal(tool("a"), tool("a", "changed")) is False
    assert declarations_equal(tool("a"), replace(tool("a"), constrained_sampling=False)) is False


def test_tool_state_changes_treat_changed_definitions_as_removal_plus_addition() -> None:
    changes = get_tool_state_changes([tool("a"), tool("b")], [tool("b", "changed"), tool("c")])
    assert changes.tools_added == [tool("b", "changed"), tool("c")]
    assert changes.tools_removed == [ToolReference(name="a"), ToolReference(name="b")]

    unchanged = get_tool_state_changes([tool("a")], [tool("a")])
    assert unchanged.tools_added == []
    assert unchanged.tools_removed == []


def test_detects_non_additive_tool_history_and_redefinitions() -> None:
    assert has_non_additive_tool_changes(TRANSCRIPT.messages) is True
    assert has_tool_redefinitions(TRANSCRIPT.messages) is False

    additive = normalize_context(
        Context(
            messages=[
                SystemMessage(content="", tools_added=[tool("a")], timestamp=1),
                SystemMessage(content="", tools_added=[tool("b")], timestamp=2),
            ]
        )
    )
    assert has_non_additive_tool_changes(additive.messages) is False

    redeclared = normalize_context(
        Context(
            messages=[
                SystemMessage(content="", tools_added=[tool("a")], timestamp=1),
                SystemMessage(content="", tools_added=[tool("a", "changed")], timestamp=2),
            ]
        )
    )
    assert has_non_additive_tool_changes(redeclared.messages) is True
    assert has_tool_redefinitions(redeclared.messages) is True
