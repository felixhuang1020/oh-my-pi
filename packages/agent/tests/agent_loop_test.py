"""底层 agent loop 测试。

移植自 ``packages/agent/test/agent-loop.test.ts``。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest
from pi_ai.types import (
    AssistantMessage,
    Cost,
    Model,
    ModelCost,
    SystemMessage,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import create_assistant_message_event_stream
from pi_ai.utils.events import done_event, error_event

from pi_agent import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    RunToolCallOptions,
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_tool_call,
    set_default_stream_fn,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def create_usage() -> Usage:
    return Usage(cost=Cost())


def create_model() -> Model:
    return Model(
        id="mock",
        name="mock",
        api="openai-responses",
        provider="openai",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=8192,
        max_tokens=2048,
    )


def create_assistant_message(content: list[Any], stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason=stop_reason,
        timestamp=now_ms(),
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(role="user", content=text, timestamp=now_ms())


def identity_converter(messages: list[Any]) -> list[Any]:
    return [
        message
        for message in messages
        if getattr(message, "role", None) in ("system", "user", "assistant", "toolResult")
    ]


def text_message(text: str) -> AssistantMessage:
    return create_assistant_message([TextContent(text=text)])


def scripted_stream_fn(messages: list[Any]):
    """按队列顺序返回预设助手消息的 stream 函数。"""
    queue = list(messages)

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        entry = queue.pop(0) if queue else text_message("done")
        message = entry(_context) if callable(entry) else entry
        stream.push(done_event(message.stop_reason, message))
        return stream

    return stream_fn


async def collect(stream: Any) -> tuple[list[Any], list[Any]]:
    events: list[Any] = []
    async for event in stream:
        events.append(event)
    messages = await stream.result()
    return events, messages


VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}


def make_echo_tool(execute: Any, **overrides: Any) -> AgentTool:
    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters=VALUE_SCHEMA,
        execute=execute,
    )
    for key, value in overrides.items():
        setattr(tool, key, value)
    return tool


# ---------------------------------------------------------------------------------------
# 默认 stream 函数兼容性（default stream function compatibility）
# ---------------------------------------------------------------------------------------


async def test_uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() -> None:
    calls = 0

    def default_stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        nonlocal calls
        calls += 1
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", text_message("fallback")))
        return stream

    set_default_stream_fn(default_stream_fn)
    try:
        context = AgentContext(messages=[], tools=[])
        config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
        stream = agent_loop([create_user_message("Hello")], context, config)
        await stream.result()
        assert calls == 1
    finally:
        set_default_stream_fn(None)


# ---------------------------------------------------------------------------------------
# agentLoop 配合 AgentMessage
# ---------------------------------------------------------------------------------------


async def test_emits_events_with_agent_message_types() -> None:
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    stream = agent_loop(
        [create_user_message("Hello")],
        context,
        config,
        None,
        scripted_stream_fn([text_message("Hi there!")]),
    )
    events, messages = await collect(stream)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    event_types = [event.type for event in events]
    for expected in ("agent_start", "turn_start", "message_start", "message_end", "turn_end", "agent_end"):
        assert expected in event_types


async def test_builds_provider_context_exclusively_from_transcript_messages() -> None:
    initial_system = SystemMessage(role="system", content="Transcript prompt", tools_added=[], timestamp=1)
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    seen: dict[str, Any] = {}

    def stream_fn(_model: Any, provider_context: Any, _options: Any = None) -> Any:
        seen["keys"] = list(vars(provider_context))
        seen["first"] = provider_context.messages[0]
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", text_message("done")))
        return stream

    stream = agent_loop([initial_system, create_user_message("Hello")], context, config, None, stream_fn)
    await stream.result()

    assert seen["keys"] == ["messages"]
    assert seen["first"] is initial_system


@dataclass
class _Notification:
    role: str = "notification"
    text: str = ""
    timestamp: int = 0


async def test_handles_custom_message_types_via_convert_to_llm() -> None:
    notification = _Notification(text="This is a notification", timestamp=now_ms())
    context = AgentContext(messages=[notification], tools=[])

    converted: list[Any] = []

    def convert_to_llm(messages: list[Any]) -> list[Any]:
        nonlocal converted
        converted = [
            message
            for message in messages
            if getattr(message, "role", None) in ("user", "assistant", "toolResult")
        ]
        return converted

    config = AgentLoopConfig(model=create_model(), convert_to_llm=convert_to_llm)
    stream = agent_loop(
        [create_user_message("Hello")],
        context,
        config,
        None,
        scripted_stream_fn([text_message("Response")]),
    )
    await collect(stream)

    assert len(converted) == 1
    assert converted[0].role == "user"


async def test_applies_transform_context_before_convert_to_llm() -> None:
    context = AgentContext(
        messages=[
            create_user_message("old message 1"),
            text_message("old response 1"),
            create_user_message("old message 2"),
            text_message("old response 2"),
        ],
        tools=[],
    )

    transformed: list[Any] = []
    converted: list[Any] = []

    def transform_context(messages: list[Any], _signal: Any = None) -> list[Any]:
        nonlocal transformed
        transformed = messages[-2:]
        return transformed

    def convert_to_llm(messages: list[Any]) -> list[Any]:
        nonlocal converted
        converted = [m for m in messages if getattr(m, "role", None) != "system"]
        return converted

    config = AgentLoopConfig(
        model=create_model(),
        transform_context=transform_context,
        convert_to_llm=convert_to_llm,
    )
    stream = agent_loop(
        [create_user_message("new message")],
        context,
        config,
        None,
        scripted_stream_fn([text_message("Response")]),
    )
    await collect(stream)

    assert len(transformed) == 2
    assert len(converted) == 2


async def test_handles_tool_calls_and_results() -> None:
    executed: list[str] = []
    tool_usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1.0),
    )
    patched_tool_usage = Usage(
        input=5,
        output=6,
        cache_read=7,
        cache_write=8,
        total_tokens=26,
        cost=Cost(input=0.5, output=0.6, cache_read=0.7, cache_write=0.8, total=2.6),
    )
    observed: dict[str, Any] = {}

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            usage=tool_usage,
        )

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])

    async def after_tool_call(ctx: Any, _signal: Any = None) -> AfterToolCallResult:
        observed["usage"] = ctx.result.usage
        return AfterToolCallResult(usage=patched_tool_usage)

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        after_tool_call=after_tool_call,
    )

    stream = agent_loop(
        [create_user_message("echo something")],
        context,
        config,
        None,
        scripted_stream_fn(
            [
                create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                ),
                text_message("done"),
            ]
        ),
    )
    events, messages = await collect(stream)

    assert executed == ["hello"]
    assert any(event.type == "tool_execution_start" for event in events)
    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_end.is_error is False
    assert observed["usage"] == tool_usage
    tool_result = next(message for message in messages if message.role == "toolResult")
    assert tool_result.usage == patched_tool_usage


async def test_does_not_execute_tool_calls_from_a_length_truncated_assistant_message() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    calls = {"count": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        if calls["count"] == 0:
            message = create_assistant_message(
                [ToolCall(id="tool-1", name="echo", arguments={"value": "hel"})], "length"
            )
            stream.push(done_event("length", message))
        else:
            stream.push(done_event("stop", text_message("done")))
        calls["count"] += 1
        return stream

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    events, messages = await collect(stream)

    assert executed == []
    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_end.is_error is True
    text = next(block for block in tool_end.result.content if block.type == "text")
    assert "output token limit" in text.text
    assert calls["count"] == 2
    assert messages[-1].role == "assistant"


async def test_executes_mutated_before_tool_call_args_without_revalidation() -> None:
    executed: list[Any] = []

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])

    async def before_tool_call(ctx: Any, _signal: Any = None) -> None:
        ctx.args["value"] = 123
        return None

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        before_tool_call=before_tool_call,
    )
    stream = agent_loop(
        [create_user_message("echo something")],
        context,
        config,
        None,
        scripted_stream_fn(
            [
                create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                ),
                text_message("done"),
            ]
        ),
    )
    await collect(stream)

    assert executed == [123]


async def test_prepares_tool_arguments_for_validation() -> None:
    edit_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                    "required": ["oldText", "newText"],
                },
            }
        },
        "required": ["edits"],
    }
    executed: list[Any] = []

    def prepare_arguments(args: Any) -> Any:
        if not isinstance(args, dict):
            return args
        if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
            return args
        return {"edits": [*args.get("edits", []), {"oldText": args["oldText"], "newText": args["newText"]}]}

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        executed.append(params["edits"])
        return AgentToolResult(content=[TextContent(text=f"edited {len(params['edits'])}")], details={"count": len(params["edits"])})

    tool = AgentTool(
        name="edit",
        label="Edit",
        description="Edit tool",
        parameters=edit_schema,
        prepare_arguments=prepare_arguments,
        execute=execute,
    )
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    stream = agent_loop(
        [create_user_message("edit something")],
        context,
        config,
        None,
        scripted_stream_fn(
            [
                create_assistant_message(
                    [ToolCall(id="tool-1", name="edit", arguments={"oldText": "before", "newText": "after"})],
                    "toolUse",
                ),
                text_message("done"),
            ]
        ),
    )
    await collect(stream)

    assert executed == [[{"oldText": "before", "newText": "after"}]]


async def test_emits_tool_execution_end_in_completion_order_but_persists_results_in_source_order() -> None:
    loop = asyncio.get_running_loop()
    first_done = loop.create_future()
    state = {"first_resolved": False, "parallel_observed": False}

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        if params["value"] == "first":
            await first_done
            state["first_resolved"] = True
        if params["value"] == "second" and not state["first_resolved"]:
            state["parallel_observed"] = True
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, tool_execution="parallel")
    calls = {"count": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        if calls["count"] == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream.push(done_event("toolUse", message))
            loop.call_later(0.02, lambda: first_done.done() or first_done.set_result(None))
        else:
            stream.push(done_event("stop", text_message("done")))
        calls["count"] += 1
        return stream

    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    events, _messages = await collect(stream)

    tool_execution_end_ids = [
        event.tool_call_id for event in events if event.type == "tool_execution_end"
    ]
    tool_result_ids = [
        event.message.tool_call_id
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    turn_tool_result_ids = [
        result.tool_call_id for event in events if event.type == "turn_end" for result in event.tool_results
    ]

    assert state["parallel_observed"] is True
    assert tool_execution_end_ids == ["tool-2", "tool-1"]
    assert tool_result_ids == ["tool-1", "tool-2"]
    assert turn_tool_result_ids == ["tool-1", "tool-2"]


async def test_injects_queued_messages_after_all_tool_calls_complete() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"ok:{params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])
    queued_user_message = create_user_message("interrupt")
    delivered = {"value": False}
    calls = {"count": 0}
    seen = {"interrupt_in_context": False}

    def get_steering_messages() -> list[Any]:
        if len(executed) >= 1 and not delivered["value"]:
            delivered["value"] = True
            return [queued_user_message]
        return []

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        tool_execution="sequential",
        get_steering_messages=get_steering_messages,
    )

    def stream_fn(_model: Any, ctx: Any, _options: Any = None) -> Any:
        if calls["count"] == 1:
            seen["interrupt_in_context"] = any(
                getattr(message, "role", None) == "user"
                and isinstance(message.content, str)
                and message.content == "interrupt"
                for message in ctx.messages
            )
        stream = create_assistant_message_event_stream()
        if calls["count"] == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream.push(done_event("toolUse", message))
        else:
            stream.push(done_event("stop", text_message("done")))
        calls["count"] += 1
        return stream

    stream = agent_loop([create_user_message("start")], context, config, None, stream_fn)
    events, _messages = await collect(stream)

    assert executed == ["first", "second"]
    tool_ends = [event for event in events if event.type == "tool_execution_end"]
    assert len(tool_ends) == 2
    assert tool_ends[0].is_error is False
    assert tool_ends[1].is_error is False

    event_sequence = []
    for event in events:
        if event.type != "message_start":
            continue
        if getattr(event.message, "role", None) == "toolResult":
            event_sequence.append(f"tool:{event.message.tool_call_id}")
        elif getattr(event.message, "role", None) == "user" and isinstance(event.message.content, str):
            event_sequence.append(event.message.content)
    assert "interrupt" in event_sequence
    assert event_sequence.index("tool:tool-1") < event_sequence.index("interrupt")
    assert event_sequence.index("tool:tool-2") < event_sequence.index("interrupt")
    assert seen["interrupt_in_context"] is True


async def test_forces_sequential_execution_when_a_tool_has_execution_mode_sequential() -> None:
    loop = asyncio.get_running_loop()
    first_done = loop.create_future()
    state = {"first_resolved": False, "parallel_observed": False}

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        if params["value"] == "first":
            await first_done
            state["first_resolved"] = True
        if params["value"] == "second" and not state["first_resolved"]:
            state["parallel_observed"] = True
        return AgentToolResult(content=[TextContent(text=f"slow: {params['value']}")], details={"value": params["value"]})

    tool = AgentTool(
        name="slow",
        label="Slow",
        description="Slow tool",
        parameters=VALUE_SCHEMA,
        execution_mode="sequential",
        execute=execute,
    )
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    calls = {"count": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        if calls["count"] == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="slow", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="slow", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream.push(done_event("toolUse", message))
            loop.call_later(0.02, lambda: first_done.done() or first_done.set_result(None))
        else:
            stream.push(done_event("stop", text_message("done")))
        calls["count"] += 1
        return stream

    stream = agent_loop([create_user_message("run both")], context, config, None, stream_fn)
    events, _messages = await collect(stream)

    assert state["parallel_observed"] is False
    tool_result_ids = [
        event.message.tool_call_id
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_result_ids == ["tool-1", "tool-2"]


async def test_forces_sequential_execution_when_one_of_multiple_tools_is_sequential() -> None:
    loop = asyncio.get_running_loop()
    slow_done = loop.create_future()
    execution_order: list[str] = []

    async def slow_execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        execution_order.append(f"slow:{params['value']}")
        if params["value"] == "a":
            await slow_done
        return AgentToolResult(content=[TextContent(text=f"slow: {params['value']}")], details={"value": params["value"]})

    async def fast_execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        execution_order.append(f"fast:{params['value']}")
        return AgentToolResult(content=[TextContent(text=f"fast: {params['value']}")], details={"value": params["value"]})

    slow_tool = AgentTool(
        name="slow",
        label="Slow",
        description="Slow tool",
        parameters=VALUE_SCHEMA,
        execution_mode="sequential",
        execute=slow_execute,
    )
    fast_tool = AgentTool(
        name="fast",
        label="Fast",
        description="Fast tool",
        parameters=VALUE_SCHEMA,
        execute=fast_execute,
    )
    context = AgentContext(messages=[], tools=[slow_tool, fast_tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    calls = {"count": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        if calls["count"] == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="slow", arguments={"value": "a"}),
                    ToolCall(id="tool-2", name="fast", arguments={"value": "b"}),
                ],
                "toolUse",
            )
            stream.push(done_event("toolUse", message))
            loop.call_later(0.02, lambda: slow_done.done() or slow_done.set_result(None))
        else:
            stream.push(done_event("stop", text_message("done")))
        calls["count"] += 1
        return stream

    stream = agent_loop([create_user_message("run both")], context, config, None, stream_fn)
    await collect(stream)

    assert execution_order[0] == "slow:a"
    assert "fast:b" in execution_order


async def test_allows_parallel_execution_when_all_tools_are_parallel() -> None:
    loop = asyncio.get_running_loop()
    first_done = loop.create_future()
    state = {"first_resolved": False, "parallel_observed": False}

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        if params["value"] == "first":
            await first_done
            state["first_resolved"] = True
        if params["value"] == "second" and not state["first_resolved"]:
            state["parallel_observed"] = True
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute, execution_mode="parallel")
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    calls = {"count": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        if calls["count"] == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream.push(done_event("toolUse", message))
            loop.call_later(0.02, lambda: first_done.done() or first_done.set_result(None))
        else:
            stream.push(done_event("stop", text_message("done")))
        calls["count"] += 1
        return stream

    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    await collect(stream)

    assert state["parallel_observed"] is True


async def test_runs_finish_turn_after_tool_result_messages_and_before_turn_end() -> None:
    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text=params["value"])], details={"value": params["value"]}, terminate=True
        )

    tool = make_echo_tool(execute)
    ordering: list[str] = []

    def finish_turn(turn: Any, _signal: Any = None) -> None:
        ordering.append("finishTurn")
        assert len(turn.tool_results) == 1
        assert getattr(turn.context.messages[-1], "role", None) == "toolResult"
        return None

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, finish_turn=finish_turn)

    def emit(event: Any) -> None:
        if event.type == "message_end":
            ordering.append(f"message_end:{getattr(event.message, 'role', None)}")
        if event.type == "turn_end":
            ordering.append("turn_end")

    await run_agent_loop(
        [create_user_message("echo")],
        AgentContext(messages=[], tools=[tool]),
        config,
        emit,
        None,
        scripted_stream_fn(
            [
                create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                )
            ]
        ),
    )

    assert ordering[-3:] == ["message_end:toolResult", "finishTurn", "turn_end"]


@pytest.mark.parametrize("reason", ["error", "aborted"])
async def test_runs_finish_turn_for_a_failed_assistant_before_turn_end(reason: str) -> None:
    ordering: list[str] = []
    counts = {"provider": 0, "steering": 0, "follow_up": 0}

    def finish_turn(turn: Any, _signal: Any = None) -> Any:
        assert turn.message.stop_reason == reason
        ordering.append("finishTurn")
        return {"action": "continue"}

    def get_steering_messages() -> list[Any]:
        counts["steering"] += 1
        return []

    def get_follow_up_messages() -> list[Any]:
        counts["follow_up"] += 1
        return [create_user_message("queued")]

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=finish_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )

    def emit(event: Any) -> None:
        if event.type == "turn_end":
            ordering.append("turn_end")

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["provider"] += 1
        stream = create_assistant_message_event_stream()
        message = create_assistant_message([], reason)
        message.error_message = reason
        stream.push(error_event(reason, message))
        return stream

    await run_agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[]),
        config,
        emit,
        None,
        stream_fn,
    )

    assert ordering == ["finishTurn", "turn_end"]
    assert counts["provider"] == 1
    assert counts["steering"] == 1
    assert counts["follow_up"] == 0


async def test_action_end_skips_queue_polling_and_next_turn_preparation() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="done")], details=None)

    tool = AgentTool(
        name="noop", label="Noop", description="Noop tool", parameters={"type": "object"}, execute=execute
    )
    counts = {"provider": 0, "steering": 0, "follow_up": 0, "prepare_next_turn": 0}

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=lambda *_args: {"action": "end"},
        prepare_next_turn=lambda *_args: counts.__setitem__("prepare_next_turn", counts["prepare_next_turn"] + 1),
        get_steering_messages=lambda: counts.__setitem__("steering", counts["steering"] + 1) or [],
        get_follow_up_messages=lambda: counts.__setitem__("follow_up", counts["follow_up"] + 1)
        or [create_user_message("queued")],
    )

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["provider"] += 1
        stream = create_assistant_message_event_stream()
        stream.push(
            done_event(
                "toolUse",
                create_assistant_message(
                    [ToolCall(id="tool-1", name="noop", arguments={})], "toolUse"
                ),
            )
        )
        return stream

    stream = agent_loop([create_user_message("run")], AgentContext(messages=[], tools=[tool]), config, None, stream_fn)
    await stream.result()

    assert counts["provider"] == 1
    assert counts["steering"] == 1
    assert counts["follow_up"] == 0
    assert counts["prepare_next_turn"] == 0


async def test_makes_exactly_one_context_only_request_when_no_natural_request_satisfies_continuation() -> None:
    counts = {"provider": 0, "finish": 0}

    def finish_turn(*_args: Any) -> Any:
        counts["finish"] += 1
        return {"action": "continue"} if counts["finish"] == 1 else None

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, finish_turn=finish_turn)

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["provider"] += 1
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", text_message(f"response {counts['provider']}")))
        return stream

    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[]), config, None, stream_fn
    )
    await stream.result()

    assert counts["provider"] == 2
    assert counts["finish"] == 2


async def test_lets_a_natural_tool_result_request_satisfy_continuation() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="done")], details=None)

    tool = AgentTool(
        name="noop", label="Noop", description="Noop tool", parameters={"type": "object"}, execute=execute
    )
    counts = {"provider": 0, "finish": 0}

    def finish_turn(*_args: Any) -> Any:
        counts["finish"] += 1
        return {"action": "continue"} if counts["finish"] == 1 else None

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, finish_turn=finish_turn)

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["provider"] += 1
        stream = create_assistant_message_event_stream()
        if counts["provider"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_message([ToolCall(id="tool-1", name="noop", arguments={})], "toolUse"),
                )
            )
        else:
            stream.push(done_event("stop", text_message("done")))
        return stream

    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[tool]), config, None, stream_fn
    )
    await stream.result()

    assert counts["provider"] == 2
    assert counts["finish"] == 2


@pytest.mark.parametrize("queue_kind", ["steering", "follow-up"])
async def test_lets_a_natural_queue_request_satisfy_continuation(queue_kind: str) -> None:
    queued_message = create_user_message(queue_kind)
    counts = {"provider": 0, "finish": 0, "steering": 0}
    delivered = {"follow_up": False}
    second_request_users: list[str] = []

    def finish_turn(*_args: Any) -> Any:
        counts["finish"] += 1
        return {"action": "continue"} if counts["finish"] == 1 else None

    def get_steering_messages() -> list[Any]:
        counts["steering"] += 1
        return [queued_message] if queue_kind == "steering" and counts["steering"] == 2 else []

    def get_follow_up_messages() -> list[Any]:
        if queue_kind != "follow-up" or delivered["follow_up"]:
            return []
        delivered["follow_up"] = True
        return [queued_message]

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=finish_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        counts["provider"] += 1
        if counts["provider"] == 2:
            second_request_users.extend(
                message.content
                for message in context.messages
                if getattr(message, "role", None) == "user" and isinstance(message.content, str)
            )
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", text_message("done")))
        return stream

    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[]), config, None, stream_fn
    )
    await stream.result()

    assert counts["provider"] == 2
    assert counts["finish"] == 2
    assert queue_kind in second_request_users


async def test_prepares_the_initial_request_after_pending_messages_and_can_replace_request_state() -> None:
    replacement_model = Model(
        id="replacement",
        name="replacement",
        api="openai-responses",
        provider="openai",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=8192,
        max_tokens=2048,
    )
    canonical_message = create_user_message("canonical projection")
    steering_message = create_user_message("steering")
    completed_messages: list[Any] = []
    state = {"delivered": False, "prepare_calls": 0}
    seen: dict[str, Any] = {}

    def get_steering_messages() -> list[Any]:
        if state["delivered"]:
            return []
        state["delivered"] = True
        return [steering_message]

    def prepare_request(request: Any, _signal: Any = None) -> Any:
        state["prepare_calls"] += 1
        assert steering_message in completed_messages
        assert steering_message in request.context.messages
        return {
            "context": AgentContext(messages=[canonical_message]),
            "model": replacement_model,
            "thinkingLevel": "high",
        }

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        get_steering_messages=get_steering_messages,
        prepare_request=prepare_request,
    )

    def emit(event: Any) -> None:
        if event.type == "message_end":
            completed_messages.append(event.message)

    def stream_fn(model: Any, context: Any, options: Any = None) -> Any:
        seen["model"] = model
        seen["messages"] = context.messages
        seen["reasoning"] = options.reasoning
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", text_message("done")))
        return stream

    await run_agent_loop(
        [create_user_message("prompt")],
        AgentContext(messages=[], tools=[]),
        config,
        emit,
        None,
        stream_fn,
    )

    assert seen["model"] is replacement_model
    assert seen["messages"] == [canonical_message]
    assert seen["reasoning"] == "high"
    assert state["prepare_calls"] == 1


async def test_does_not_poll_steering_after_prepare_request() -> None:
    queued: list[Any] = []
    late_steering = create_user_message("late steering")
    request_included_steering: list[bool] = []
    state = {"preparations": 0, "steering_polls": 0}

    def get_steering_messages() -> list[Any]:
        state["steering_polls"] += 1
        drained = list(queued)
        queued.clear()
        return drained

    def prepare_request(*_args: Any) -> None:
        state["preparations"] += 1
        if state["preparations"] == 1:
            queued.append(late_steering)
        return None

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        get_steering_messages=get_steering_messages,
        prepare_request=prepare_request,
    )

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        request_included_steering.append(late_steering in context.messages)
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", text_message("done")))
        return stream

    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[]), config, None, stream_fn
    )
    await stream.result()

    assert request_included_steering == [False, True]
    assert state["preparations"] == 2
    assert state["steering_polls"] == 3


async def test_uses_prepare_next_turn_snapshot_before_continuing() -> None:
    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])
    state = {"prepared": False, "prepare_calls": 0, "has_update": False, "llm_calls": 0}

    def prepare_next_turn(current: Any) -> Any:
        state["prepare_calls"] += 1
        if state["prepared"]:
            return None
        state["prepared"] = True
        return {
            "context": AgentContext(messages=list(current.context.messages), tools=current.context.tools),
            "messages": [SystemMessage(role="system", content="updated guidance", timestamp=1)],
        }

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        prepare_next_turn=prepare_next_turn,
    )

    def stream_fn(_model: Any, ctx: Any, _options: Any = None) -> Any:
        state["llm_calls"] += 1
        if state["llm_calls"] == 2:
            state["has_update"] = any(
                getattr(message, "role", None) == "system" and message.content == "updated guidance"
                for message in ctx.messages
            )
        stream = create_assistant_message_event_stream()
        if state["llm_calls"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_message(
                        [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                    ),
                )
            )
        else:
            stream.push(done_event("stop", text_message("done")))
        return stream

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    await collect(stream)

    assert state["llm_calls"] == 2
    assert state["prepare_calls"] == 1
    assert state["has_update"] is True


async def test_picks_up_steering_queued_during_prepare_next_turn_before_the_next_request() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="done")], details=None)

    tool = AgentTool(
        name="noop", label="Noop", description="Noop tool", parameters={"type": "object"}, execute=execute
    )
    queued: list[Any] = []
    late_steering = create_user_message("late steering")
    counts = {"provider": 0}
    state = {"included": False}

    def prepare_next_turn(*_args: Any) -> None:
        queued.append(late_steering)
        return None

    def get_steering_messages() -> list[Any]:
        drained = list(queued)
        queued.clear()
        return drained

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        prepare_next_turn=prepare_next_turn,
        get_steering_messages=get_steering_messages,
    )

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        counts["provider"] += 1
        if counts["provider"] == 2:
            state["included"] = late_steering in context.messages
        stream = create_assistant_message_event_stream()
        if counts["provider"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_message([ToolCall(id="tool-1", name="noop", arguments={})], "toolUse"),
                )
            )
        else:
            stream.push(done_event("stop", text_message("done")))
        return stream

    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[tool]), config, None, stream_fn
    )
    await stream.result()

    assert counts["provider"] == 2
    assert state["included"] is True


async def test_action_end_receives_finalized_turn_context_and_stops_before_queue_polling() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])
    counts = {"steering": 0, "follow_up": 0, "llm": 0}
    captured: dict[str, Any] = {}

    def finish_turn(turn: Any, _signal: Any = None) -> Any:
        captured["message_role"] = turn.message.role
        captured["tool_result_ids"] = [result.tool_call_id for result in turn.tool_results]
        captured["context_roles"] = [getattr(message, "role", None) for message in turn.context.messages]
        return {"action": "end"}

    def get_steering_messages() -> list[Any]:
        counts["steering"] += 1
        return []

    def get_follow_up_messages() -> list[Any]:
        counts["follow_up"] += 1
        return [create_user_message("follow up should stay queued")]

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=finish_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["llm"] += 1
        stream = create_assistant_message_event_stream()
        if counts["llm"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_message(
                        [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                    ),
                )
            )
        else:
            stream.push(done_event("stop", text_message("should not run")))
        return stream

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    events, messages = await collect(stream)

    assert counts["llm"] == 1
    assert executed == ["hello"]
    assert counts["steering"] == 1
    assert counts["follow_up"] == 0
    assert captured["tool_result_ids"] == ["tool-1"]
    assert captured["context_roles"] == ["system", "user", "assistant", "toolResult"]
    assert [message.role for message in messages] == ["system", "user", "assistant", "toolResult"]
    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


async def test_stops_after_a_tool_batch_when_every_tool_result_terminates() -> None:
    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            terminate=True,
        )

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    counts = {"llm": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["llm"] += 1
        stream = create_assistant_message_event_stream()
        stream.push(
            done_event(
                "toolUse",
                create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                ),
            )
        )
        return stream

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    events, messages = await collect(stream)

    assert counts["llm"] == 1
    assert [message.role for message in messages] == ["system", "user", "assistant", "toolResult"]
    assert len([event for event in events if event.type == "turn_end"]) == 1


async def test_stops_after_a_blocked_tool_call_when_before_tool_call_sets_terminate() -> None:
    executed = {"value": False}

    async def execute() -> AgentToolResult:
        executed["value"] = True
        return AgentToolResult(content=[TextContent(text="should not execute")], details={"value": "unexpected"})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])

    async def before_tool_call(*_args: Any) -> Any:
        return {"block": True, "reason": "Blocked by policy", "terminate": True}

    config = AgentLoopConfig(
        model=create_model(), convert_to_llm=identity_converter, before_tool_call=before_tool_call
    )
    counts = {"llm": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["llm"] += 1
        stream = create_assistant_message_event_stream()
        if counts["llm"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_message(
                        [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                    ),
                )
            )
        else:
            stream.push(done_event("stop", text_message("should not run")))
        return stream

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    _events, messages = await collect(stream)

    tool_result = next(message for message in messages if message.role == "toolResult")
    assert executed["value"] is False
    assert counts["llm"] == 1
    assert tool_result.is_error is True
    assert TextContent(text="Blocked by policy") in tool_result.content


async def test_continues_after_a_mixed_batch_with_one_terminating_blocked_call() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])

    async def before_tool_call(ctx: Any, _signal: Any = None) -> Any:
        if ctx.args["value"] == "first":
            return {"block": True, "reason": "Blocked first", "terminate": True}
        return None

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        tool_execution="parallel",
        before_tool_call=before_tool_call,
    )
    counts = {"llm": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["llm"] += 1
        stream = create_assistant_message_event_stream()
        if counts["llm"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_message(
                        [
                            ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                            ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                        ],
                        "toolUse",
                    ),
                )
            )
        else:
            stream.push(done_event("stop", text_message("done")))
        return stream

    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    await collect(stream)

    assert executed == ["second"]
    assert counts["llm"] == 2


async def test_continues_after_parallel_tool_calls_when_not_all_results_terminate() -> None:
    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            terminate=params["value"] == "first",
        )

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, tool_execution="parallel")
    calls = {"count": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        if calls["count"] == 0:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_message(
                        [
                            ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                            ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                        ],
                        "toolUse",
                    ),
                )
            )
        else:
            stream.push(done_event("stop", text_message("done")))
        calls["count"] += 1
        return stream

    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    _events, messages = await collect(stream)

    assert calls["count"] == 2
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "toolResult",
        "toolResult",
        "assistant",
    ]


async def test_allows_after_tool_call_to_mark_a_tool_batch_as_terminating() -> None:
    async def execute(_tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]})

    tool = make_echo_tool(execute)
    context = AgentContext(messages=[], tools=[tool])

    async def after_tool_call(*_args: Any) -> AfterToolCallResult:
        return AfterToolCallResult(terminate=True)

    config = AgentLoopConfig(
        model=create_model(), convert_to_llm=identity_converter, after_tool_call=after_tool_call
    )
    counts = {"llm": 0}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        counts["llm"] += 1
        stream = create_assistant_message_event_stream()
        stream.push(
            done_event(
                "toolUse",
                create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
                ),
            )
        )
        return stream

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    await collect(stream)

    assert counts["llm"] == 1


# ---------------------------------------------------------------------------------------
# agentLoopContinue 配合 AgentMessage
# ---------------------------------------------------------------------------------------


async def test_agent_loop_continue_throws_when_context_has_no_messages() -> None:
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    def stream_fn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Unexpected stream call")

    with pytest.raises(ValueError, match="Cannot continue: no messages in context"):
        agent_loop_continue(context, config, None, stream_fn)


async def test_agent_loop_continue_continues_without_emitting_user_message_events() -> None:
    user_message = create_user_message("Hello")
    context = AgentContext(messages=[user_message], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    stream = agent_loop_continue(context, config, None, scripted_stream_fn([text_message("Response")]))
    events, messages = await collect(stream)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    message_end_events = [event for event in events if event.type == "message_end"]
    assert len(message_end_events) == 1
    assert message_end_events[0].message.role == "assistant"


@dataclass
class _CustomMessage:
    role: str = "custom"
    text: str = ""
    timestamp: int = 0


async def test_agent_loop_continue_allows_custom_message_types_as_last_message() -> None:
    custom_message = _CustomMessage(text="Hook content", timestamp=now_ms())
    context = AgentContext(messages=[custom_message], tools=[])

    def convert_to_llm(messages: list[Any]) -> list[Any]:
        converted = []
        for message in messages:
            if getattr(message, "role", None) == "custom":
                converted.append(UserMessage(role="user", content=message.text, timestamp=message.timestamp))
            elif getattr(message, "role", None) in ("user", "assistant", "toolResult"):
                converted.append(message)
        return converted

    config = AgentLoopConfig(model=create_model(), convert_to_llm=convert_to_llm)
    stream = agent_loop_continue(
        context, config, None, scripted_stream_fn([text_message("Response to custom message")])
    )
    _events, messages = await collect(stream)

    assert len(messages) == 1
    assert messages[0].role == "assistant"


# ---------------------------------------------------------------------------------------
# runToolCall（执行 tool 调用）
# ---------------------------------------------------------------------------------------

ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}


async def _echo_execute(_tool_call_id: str, params: dict[str, Any], _signal: Any = None, on_update: Any = None) -> AgentToolResult:
    if on_update is not None:
        on_update(AgentToolResult(content=[TextContent(text="partial")], details={}))
    return AgentToolResult(
        content=[TextContent(text=params["value"])],
        details={},
        structured_content={"value": params["value"]},
    )


async def _failing_execute(*_args: Any, **_kwargs: Any) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text="bad")], details={"partial": True}, is_error=True
    )


def _call(tool_call_id: str, name: str, arguments: dict[str, Any]) -> AgentToolCall:
    return ToolCall(id=tool_call_id, name=name, arguments=arguments)


async def test_run_tool_call_validates_runs_hooks_and_reports_failures() -> None:
    echo = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters=ECHO_SCHEMA,
        output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        execute=_echo_execute,
    )
    failing = AgentTool(
        name="failing",
        label="Failing",
        description="Returns an error result",
        parameters={"type": "object"},
        execute=_failing_execute,
    )
    assistant_message = create_assistant_message([])
    hook_calls: list[str] = []
    updates: list[Any] = []

    async def before_tool_call(ctx: Any, _signal: Any = None) -> Any:
        hook_calls.append(f"before {ctx.tool_call.id}")
        if isinstance(ctx.args, dict) and ctx.args.get("value") == "blocked":
            return {"block": True, "reason": "nope"}
        return None

    async def after_tool_call(ctx: Any, _signal: Any = None) -> None:
        hook_calls.append(f"after {ctx.tool_call.id}")
        return None

    options = RunToolCallOptions(
        tools=[echo, failing],
        assistant_message=assistant_message,
        context=AgentContext(messages=[]),
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        on_update=updates.append,
    )

    outcome_a = await run_tool_call(_call("a", "echo", {"value": "a"}), options)
    assert outcome_a.tool_call.id == "a"
    assert outcome_a.result.structured_content == {"value": "a"}
    assert outcome_a.is_error is False

    outcome_b = await run_tool_call(_call("b", "echo", {"value": {"nested": True}}), options)  # type: ignore[arg-type]
    assert outcome_b.is_error is True

    outcome_c = await run_tool_call(_call("c", "echo", {"value": "blocked"}), options)
    assert outcome_c.result.content == [TextContent(text="nope")]
    assert outcome_c.is_error is True

    outcome_d = await run_tool_call(_call("d", "missing", {}), options)
    assert outcome_d.result.content == [TextContent(text="Tool missing not found")]
    assert outcome_d.is_error is True

    outcome_e = await run_tool_call(_call("e", "failing", {}), options)
    assert outcome_e.result.details == {"partial": True}
    assert outcome_e.is_error is True

    assert updates == [AgentToolResult(content=[TextContent(text="partial")], details={})]
    assert hook_calls == ["before a", "after a", "before c", "before e", "after e"]


async def test_run_tool_call_lets_after_tool_call_replace_structured_content() -> None:
    echo = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters=ECHO_SCHEMA,
        output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        execute=_echo_execute,
    )
    assistant_message = create_assistant_message([])
    redacted = [TextContent(text="redacted")]
    results = [
        AfterToolCallResult(content=redacted),
        AfterToolCallResult(structured_content={"value": "replaced"}),
        AfterToolCallResult(content=redacted, structured_content={"value": "both"}),
        AfterToolCallResult(details={"note": "kept"}),
    ]
    seen: list[Any] = []
    for after_result in results:

        async def after_tool_call(_ctx: Any, _signal: Any = None, result: Any = after_result) -> Any:
            return result

        outcome = await run_tool_call(
            _call("x", "echo", {"value": "original"}),
            RunToolCallOptions(
                tools=[echo],
                assistant_message=assistant_message,
                context=AgentContext(messages=[]),
                after_tool_call=after_tool_call,
            ),
        )
        seen.append(outcome.result.structured_content)

    assert seen == [None, {"value": "replaced"}, {"value": "both"}, {"value": "original"}]
