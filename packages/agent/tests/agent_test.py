"""有状态 Agent 测试。

移植自 ``packages/agent/test/agent.test.ts``。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pi_ai.types import (
    AssistantMessage,
    Cost,
    Model,
    ModelCost,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.event_stream import create_assistant_message_event_stream
from pi_ai.utils.events import done_event, error_event, start_event
from pi_ai.utils.transcript import get_current_system_message, to_tool_declaration

from pi_agent import (
    Agent,
    AgentInitialState,
    AgentTool,
    AgentToolResult,
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


def create_user_message(text: str) -> UserMessage:
    return UserMessage(role="user", content=text, timestamp=now_ms())


def create_assistant_message(text: str, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(text=text)],
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason=stop_reason,
        timestamp=now_ms(),
    )


def create_tool(name: str) -> AgentTool:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=name)], details={})

    return AgentTool(
        name=name,
        label=name,
        description=f"{name} tool",
        parameters={"type": "object"},
        execute=execute,
    )


def create_assistant_tool_use_message(content: list[Any]) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason="toolUse",
        timestamp=now_ms(),
    )


def unused_stream_function(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Unexpected stream call")


def scripted_stream_fn(messages: list[Any]):
    queue = list(messages)

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        entry = queue.pop(0) if queue else create_assistant_message("done")
        message = entry(_context) if callable(entry) else entry
        stream.push(done_event(message.stop_reason, message))
        return stream

    return stream_fn


def create_deferred() -> asyncio.Future[None]:
    return asyncio.get_running_loop().create_future()


async def test_uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() -> None:
    calls = {"count": 0}

    def default_stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        calls["count"] += 1
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("fallback")))
        return stream

    set_default_stream_fn(default_stream_fn)
    try:
        agent = Agent()
        await agent.prompt("Hello")
        assert calls["count"] == 1
    finally:
        set_default_stream_fn(None)


def test_creates_an_agent_instance_with_default_state() -> None:
    agent = Agent(stream_fn=unused_stream_function)
    state = agent.state

    assert state.model is not None
    assert state.thinking_level == "off"
    assert state.tools == []
    assert state.messages == []
    assert state.is_streaming is False
    assert state.streaming_message is None
    assert state.pending_tool_calls == set()
    assert state.error_message is None


def test_creates_an_agent_instance_with_custom_initial_state() -> None:
    custom_model = create_model()
    agent = Agent(
        stream_fn=unused_stream_function,
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant.",
            model=custom_model,
            thinking_level="low",
        ),
    )

    assert agent.state.messages == [SystemMessage(content="You are a helpful assistant.", timestamp=0)]
    assert agent.state.model is custom_model
    assert agent.state.thinking_level == "low"


def test_converts_initial_prompt_and_tools_into_transcript_state() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="echo")], details={})

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo input",
        parameters={"type": "object"},
        execute=execute,
    )
    agent = Agent(
        initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[tool]),
        stream_fn=unused_stream_function,
    )

    initial = agent.state.messages[0]
    assert initial.role == "system"
    assert initial.content == "You are helpful."
    assert [declared.name for declared in initial.tools_added] == ["echo"]


async def test_declares_tool_loadout_changes_to_the_model_before_the_next_request() -> None:
    first = create_tool("first")
    second = create_tool("second")
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        requests.append(
            [
                entry
                for message in context.messages
                if getattr(message, "role", None) == "system"
                for entry in (
                    f"+{','.join(tool.name for tool in (message.tools_added or []))}",
                    f"-{','.join(tool.name for tool in (message.tools_removed or []))}",
                )
            ]
        )
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("done")))
        return stream

    agent = Agent(
        stream_fn=stream_fn,
        initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[first]),
    )

    await agent.prompt("one")
    agent.state.tools = [second]
    await agent.prompt("two")
    await agent.prompt("three")

    assert requests == [
        ["+first", "-"],
        ["+first", "-", "+second", "-first"],
        ["+first", "-", "+second", "-first"],
    ]

    update = next(
        message
        for message in agent.state.messages
        if getattr(message, "role", None) == "system" and message.tools_removed
    )
    assert update.content == ""
    assert [tool.name for tool in update.tools_added] == ["second"]
    assert update.tools_added[0].description == "second tool"
    assert update.tools_added[0].parameters == {"type": "object"}
    assert update.tools_removed == [ToolReference(name="first")]
    assert isinstance(update.timestamp, int)

    initial = agent.state.messages[0]
    assert initial.role == "system"
    assert initial.tools_added[0] == Tool(name="first", description="first tool", parameters={"type": "object"})


async def test_merges_tool_changes_into_a_pending_system_message() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="echo")], details={})

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo input",
        parameters={"type": "object"},
        execute=execute,
    )
    seen: dict[str, int] = {}

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        seen["systems"] = len(
            [message for message in context.messages if getattr(message, "role", None) == "system"]
        )
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("done")))
        return stream

    agent = Agent(initial_state=AgentInitialState(system_prompt="You are helpful."), stream_fn=stream_fn)

    agent.state.tools = [tool]
    await agent.prompt(
        [
            SystemMessage(role="system", content="", sections={"skills": "<skills>x</skills>"}, timestamp=1),
            UserMessage(role="user", content="hi", timestamp=2),
        ]
    )

    assert seen["systems"] == 2
    assert agent.state.messages[1] == SystemMessage(
        role="system",
        content="",
        sections={"skills": "<skills>x</skills>"},
        tools_added=[Tool(name="echo", description="Echo input", parameters={"type": "object"})],
        timestamp=1,
    )


async def test_rewrites_pending_tool_declarations_to_match_the_executable_set() -> None:
    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("done")))
        return stream

    agent = Agent(
        initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[create_tool("first")]),
        stream_fn=stream_fn,
    )

    await agent.prompt(
        [
            SystemMessage(
                role="system",
                content="",
                sections={"note": "<note>x</note>"},
                tools_added=[to_tool_declaration(create_tool("second"))],
                tools_removed=[ToolReference(name="first")],
                timestamp=1,
            ),
            UserMessage(role="user", content="hi", timestamp=2),
        ]
    )

    assert agent.state.messages[1] == SystemMessage(
        role="system", content="", sections={"note": "<note>x</note>"}, timestamp=1
    )
    current = get_current_system_message(agent.state.messages)
    assert [tool.name for tool in (current.tools_added or [])] == ["first"]


def test_restores_the_transcript_baseline_when_reset() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="echo")], details={})

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo input",
        parameters={"type": "object"},
        execute=execute,
    )
    agent = Agent(
        initial_state=AgentInitialState(
            system_prompt="You are helpful.",
            tools=[tool],
            messages=[UserMessage(role="user", content="old", timestamp=1)],
        ),
        stream_fn=unused_stream_function,
    )

    agent.reset()

    assert len(agent.state.messages) == 1
    initial = agent.state.messages[0]
    assert initial.role == "system"
    assert initial.content == "You are helpful."
    assert [declared.name for declared in initial.tools_added] == ["echo"]


def test_subscribes_to_events() -> None:
    agent = Agent(stream_fn=unused_stream_function)
    count = {"events": 0}
    unsubscribe = agent.subscribe(lambda _event, _signal: count.__setitem__("events", count["events"] + 1))

    assert count["events"] == 0
    agent.state.thinking_level = "low"
    assert count["events"] == 0
    assert agent.state.thinking_level == "low"

    unsubscribe()
    agent.state.thinking_level = "high"
    assert count["events"] == 0


async def test_emits_full_lifecycle_events_for_thrown_run_failures() -> None:
    def stream_fn(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider exploded")

    agent = Agent(stream_fn=stream_fn)
    events: list[str] = []
    agent.subscribe(lambda event, _signal: events.append(event.type))

    await agent.prompt("hello")

    assert events == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    last = agent.state.messages[-1]
    assert last.role == "assistant"
    assert last.stop_reason == "error"
    assert last.error_message == "provider exploded"
    assert agent.state.error_message == "provider exploded"


async def test_awaits_async_subscribers_before_prompt_resolves() -> None:
    barrier = create_deferred()
    agent = Agent(stream_fn=scripted_stream_fn([create_assistant_message("ok")]))
    state = {"listener_finished": False}

    async def listener(event: Any, _signal: Any) -> None:
        if event.type == "agent_end":
            await barrier
            state["listener_finished"] = True

    agent.subscribe(listener)

    resolved = {"value": False}
    task = asyncio.ensure_future(agent.prompt("hello"))

    await asyncio.sleep(0.01)
    assert resolved["value"] is False
    assert state["listener_finished"] is False
    assert agent.state.is_streaming is True

    barrier.set_result(None)
    await task
    resolved["value"] = True

    assert state["listener_finished"] is True
    assert resolved["value"] is True
    assert agent.state.is_streaming is False


async def test_wait_for_idle_waits_for_async_subscribers() -> None:
    barrier = create_deferred()
    agent = Agent(stream_fn=scripted_stream_fn([create_assistant_message("ok")]))

    async def listener(event: Any, _signal: Any) -> None:
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
            await barrier

    agent.subscribe(listener)

    task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.sleep(0)
    idle = {"resolved": False}

    async def wait_idle() -> None:
        await agent.wait_for_idle()
        idle["resolved"] = True

    idle_task = asyncio.ensure_future(wait_idle())

    await asyncio.sleep(0.01)
    assert idle["resolved"] is False
    assert agent.state.is_streaming is True

    barrier.set_result(None)
    await asyncio.gather(task, idle_task)

    assert idle["resolved"] is True
    assert agent.state.is_streaming is False


async def test_passes_the_active_abort_signal_to_subscribers() -> None:
    received: dict[str, Any] = {}

    def stream_fn(_model: Any, _context: Any, options: Any = None) -> Any:
        received["signal"] = options.signal
        stream = create_assistant_message_event_stream()
        stream.push(start_event(create_assistant_message("")))

        async def watch() -> None:
            while not options.signal.aborted:
                await asyncio.sleep(0.005)
            stream.push(error_event("aborted", create_assistant_message("Aborted", "aborted")))

        asyncio.ensure_future(watch())
        return stream

    agent = Agent(stream_fn=stream_fn)
    agent.subscribe(
        lambda event, signal: received.__setitem__("listener_signal", signal)
        if event.type == "agent_start"
        else None
    )

    task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.sleep(0.01)

    assert received["listener_signal"] is not None
    assert received["listener_signal"].aborted is False

    agent.abort()
    await task

    assert received["listener_signal"].aborted is True


async def test_ignores_tool_updates_after_the_tool_execution_settles() -> None:
    captured: dict[str, Any] = {}
    events: list[Any] = []

    async def execute(_tool_call_id: str, _params: Any, _signal: Any, on_update: Any) -> AgentToolResult:
        captured["on_update"] = on_update
        on_update(AgentToolResult(content=[TextContent(text="running")], details={"status": "running"}))
        return AgentToolResult(
            content=[TextContent(text="ok")], details={"status": "done"}, terminate=True
        )

    tool = AgentTool(
        name="delayed_tool",
        label="Delayed Tool",
        description="Captures progress callbacks",
        parameters={"type": "object"},
        execute=execute,
    )
    agent = Agent(
        initial_state=AgentInitialState(tools=[tool]),
        stream_fn=scripted_stream_fn(
            [
                create_assistant_tool_use_message(
                    [ToolCall(id="call-1", name="delayed_tool", arguments={})]
                )
            ]
        ),
    )
    agent.subscribe(lambda event, _signal: events.append(event))

    await agent.prompt("run tool")
    event_count_after_prompt = len(events)

    captured["on_update"](AgentToolResult(content=[TextContent(text="late")], details={"status": "late"}))
    await asyncio.sleep(0)

    assert len([event for event in events if event.type == "tool_execution_update"]) == 1
    assert len(events) == event_count_after_prompt


async def test_ignores_a_settled_parallel_tool_update_while_another_tool_is_running() -> None:
    loop = asyncio.get_running_loop()
    slow_started = loop.create_future()
    settled_tool_ended = loop.create_future()
    release_slow = loop.create_future()
    captured: dict[str, Any] = {}
    events: list[Any] = []

    async def settled_execute(_tool_call_id: str, _params: Any, _signal: Any, on_update: Any) -> AgentToolResult:
        captured["on_update"] = on_update
        return AgentToolResult(
            content=[TextContent(text="done")], details={"status": "done"}, terminate=True
        )

    async def slow_execute() -> AgentToolResult:
        slow_started.set_result(None)
        await release_slow
        return AgentToolResult(
            content=[TextContent(text="done")], details={"status": "done"}, terminate=True
        )

    settled_tool = AgentTool(
        name="settled_tool",
        label="Settled Tool",
        description="Captures progress callbacks",
        parameters={"type": "object"},
        execute=settled_execute,
    )
    slow_tool = AgentTool(
        name="slow_tool",
        label="Slow Tool",
        description="Keeps the agent run active",
        parameters={"type": "object"},
        execute=slow_execute,
    )
    agent = Agent(
        initial_state=AgentInitialState(tools=[settled_tool, slow_tool]),
        stream_fn=scripted_stream_fn(
            [
                create_assistant_tool_use_message(
                    [
                        ToolCall(id="call-1", name="settled_tool", arguments={}),
                        ToolCall(id="call-2", name="slow_tool", arguments={}),
                    ]
                )
            ]
        ),
    )

    def listener(event: Any, _signal: Any) -> None:
        events.append(event)
        if event.type == "tool_execution_end" and event.tool_call_id == "call-1" and not settled_tool_ended.done():
            settled_tool_ended.set_result(None)

    agent.subscribe(listener)

    task = asyncio.ensure_future(agent.prompt("run tools"))
    await asyncio.gather(slow_started, settled_tool_ended)
    event_count_before_late_update = len(events)

    captured["on_update"](AgentToolResult(content=[TextContent(text="late")], details={"status": "late"}))
    await asyncio.sleep(0)
    assert len(events) == event_count_before_late_update

    release_slow.set_result(None)
    await task
    assert len([event for event in events if event.type == "tool_execution_update"]) == 0


def test_updates_state_with_mutators() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    new_model = create_model()
    agent.state.model = new_model
    assert agent.state.model is new_model

    agent.state.thinking_level = "high"
    assert agent.state.thinking_level == "high"

    tools = [AgentTool(name="test", description="test tool")]
    agent.state.tools = tools
    assert agent.state.tools == tools
    assert agent.state.tools is not tools

    messages = [UserMessage(role="user", content="Hello", timestamp=now_ms())]
    agent.state.messages = messages
    assert agent.state.messages == messages
    assert agent.state.messages is not messages

    new_message = AssistantMessage(role="assistant", content=[TextContent(text="Hi")])
    agent.state.messages.append(new_message)
    assert len(agent.state.messages) == 2
    assert agent.state.messages[1] is new_message

    agent.state.messages = []
    assert agent.state.messages == []


async def test_supports_steering_and_follow_up_message_queues() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    steering = UserMessage(role="user", content="Steering message", timestamp=now_ms())
    agent.steer(steering)
    assert steering not in agent.state.messages

    follow_up = UserMessage(role="user", content="Follow-up message", timestamp=now_ms())
    agent.follow_up(follow_up)
    assert follow_up not in agent.state.messages


def test_handles_abort_controller() -> None:
    agent = Agent(stream_fn=unused_stream_function)
    agent.abort()  # 即使当前没有任务在运行也不应抛异常


async def test_rejects_reset_while_processing_without_corrupting_the_transcript() -> None:
    loop = asyncio.get_running_loop()
    stream_started = loop.create_future()
    release_response = loop.create_future()

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()

        async def produce() -> None:
            stream.push(start_event(create_assistant_message("")))
            stream_started.set_result(None)
            await release_response
            stream.push(done_event("stop", create_assistant_message("Done")))

        asyncio.ensure_future(produce())
        return stream

    agent = Agent(stream_fn=stream_fn)
    task = asyncio.ensure_future(agent.prompt("Hello"))
    await stream_started

    try:
        assert agent.state.is_streaming is True
        assert [message.role for message in agent.state.messages] == ["user"]
        with pytest.raises(RuntimeError) as error:
            agent.reset()
        assert "Agent is already processing. Wait for completion before resetting." in str(error.value)
        assert agent.state.is_streaming is True
        assert [message.role for message in agent.state.messages] == ["user"]
    finally:
        release_response.set_result(None)
        await task

    assert agent.state.is_streaming is False
    assert [message.role for message in agent.state.messages] == ["user", "assistant"]


async def test_throws_when_prompt_is_called_while_streaming() -> None:
    received: dict[str, Any] = {}

    def stream_fn(_model: Any, _context: Any, options: Any = None) -> Any:
        received["signal"] = options.signal
        stream = create_assistant_message_event_stream()
        stream.push(start_event(create_assistant_message("")))

        async def watch() -> None:
            while not received["signal"].aborted:
                await asyncio.sleep(0.005)
            stream.push(error_event("aborted", create_assistant_message("Aborted", "aborted")))

        asyncio.ensure_future(watch())
        return stream

    agent = Agent(stream_fn=stream_fn)
    first_prompt = asyncio.ensure_future(agent.prompt("First message"))

    await asyncio.sleep(0.01)
    assert agent.state.is_streaming is True

    with pytest.raises(RuntimeError) as error:
        await agent.prompt("Second message")
    assert (
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, "
        "or wait for completion." in str(error.value)
    )

    agent.abort()
    await first_prompt


async def test_throws_when_continue_is_called_while_streaming() -> None:
    received: dict[str, Any] = {}

    def stream_fn(_model: Any, _context: Any, options: Any = None) -> Any:
        received["signal"] = options.signal
        stream = create_assistant_message_event_stream()
        stream.push(start_event(create_assistant_message("")))

        async def watch() -> None:
            while not received["signal"].aborted:
                await asyncio.sleep(0.005)
            stream.push(error_event("aborted", create_assistant_message("Aborted", "aborted")))

        asyncio.ensure_future(watch())
        return stream

    agent = Agent(stream_fn=stream_fn)
    first_prompt = asyncio.ensure_future(agent.prompt("First message"))

    await asyncio.sleep(0.01)
    assert agent.state.is_streaming is True

    with pytest.raises(RuntimeError) as error:
        await agent.continue_()
    assert "Agent is already processing. Wait for completion before continuing." in str(error.value)

    agent.abort()
    await first_prompt


async def test_continue_processes_queued_follow_up_messages_after_an_assistant_turn() -> None:
    agent = Agent(stream_fn=scripted_stream_fn([create_assistant_message("Processed")]))

    agent.state.messages = [
        UserMessage(role="user", content=[TextContent(text="Initial")], timestamp=now_ms() - 10),
        create_assistant_message("Initial response"),
    ]
    agent.follow_up(
        UserMessage(role="user", content=[TextContent(text="Queued follow-up")], timestamp=now_ms())
    )

    await agent.continue_()

    def is_queued_follow_up(message: Any) -> bool:
        if getattr(message, "role", None) != "user":
            return False
        content = message.content
        if isinstance(content, str):
            return content == "Queued follow-up"
        return any(block.type == "text" and block.text == "Queued follow-up" for block in content)

    assert any(is_queued_follow_up(message) for message in agent.state.messages)
    assert agent.state.messages[-1].role == "assistant"


@pytest.mark.parametrize(
    ("mode", "expected_requests"),
    [("one-at-a-time", 2), ("all", 1)],
)
async def test_continue_keeps_steering_semantics_for_assistant_tail_fallback(
    mode: str, expected_requests: int
) -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        requests.append(
            [
                message.content
                for message in context.messages
                if getattr(message, "role", None) == "user" and isinstance(message.content, str)
            ]
        )
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("Processed")))
        return stream

    agent = Agent(steering_mode=mode, stream_fn=stream_fn)
    agent.state.messages = [create_user_message("Initial"), create_assistant_message("Initial response")]
    agent.steer(create_user_message("Steering 1"))
    agent.steer(create_user_message("Steering 2"))

    assert await agent.continue_() is None

    assert len(requests) == expected_requests
    assert "Steering 1" in requests[0]
    if mode == "one-at-a-time":
        assert "Steering 2" not in requests[0]
        assert "Steering 2" in requests[1]
    else:
        assert "Steering 2" in requests[0]


async def test_keeps_legacy_prepare_next_turn_signal_callback_behavior() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    tool = AgentTool(
        name="noop", label="Noop", description="Noop tool", parameters={"type": "object"}, execute=execute
    )
    state = {"request_count": 0, "saw_abort_signal": False}

    async def prepare_next_turn(signal: Any) -> None:
        state["saw_abort_signal"] = isinstance(signal, AbortSignal)
        return None

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        state["request_count"] += 1
        stream = create_assistant_message_event_stream()
        if state["request_count"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_tool_use_message(
                        [ToolCall(id="tool-1", name="noop", arguments={})]
                    ),
                )
            )
        else:
            stream.push(done_event("stop", create_assistant_message("done")))
        return stream

    agent = Agent(
        initial_state=AgentInitialState(tools=[tool]),
        prepare_next_turn=prepare_next_turn,
        stream_fn=stream_fn,
    )

    await agent.prompt("start")

    assert state["request_count"] == 2
    assert state["saw_abort_signal"] is True


async def test_forwards_finish_turn_through_agent_options_with_the_active_abort_signal() -> None:
    async def execute() -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="tool complete")], details={})

    tool = AgentTool(
        name="noop", label="Noop", description="Noop tool", parameters={"type": "object"}, execute=execute
    )
    state: dict[str, Any] = {"request_count": 0, "saw_abort_signal": False, "roles": []}

    def finish_turn(turn: Any, signal: Any) -> Any:
        state["saw_abort_signal"] = isinstance(signal, AbortSignal)
        state["roles"] = [getattr(message, "role", None) for message in turn.context.messages]
        return {"action": "end"}

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        state["request_count"] += 1
        stream = create_assistant_message_event_stream()
        if state["request_count"] == 1:
            stream.push(
                done_event(
                    "toolUse",
                    create_assistant_tool_use_message(
                        [ToolCall(id="tool-1", name="noop", arguments={})]
                    ),
                )
            )
        else:
            stream.push(done_event("stop", create_assistant_message("should not run")))
        return stream

    agent = Agent(
        initial_state=AgentInitialState(tools=[tool]),
        finish_turn=finish_turn,
        stream_fn=stream_fn,
    )

    await agent.prompt("start")

    assert state["request_count"] == 1
    assert state["saw_abort_signal"] is True
    assert state["roles"] == ["system", "user", "assistant", "toolResult"]


@pytest.mark.parametrize(
    "messages",
    [[], [SystemMessage(role="system", content="system only", timestamp=1)]],
    ids=["empty", "system-only"],
)
async def test_rejects_a_queued_continuation_from_empty_context_without_draining_queues(
    messages: list[Any],
) -> None:
    agent = Agent(
        initial_state=AgentInitialState(messages=messages), stream_fn=unused_stream_function
    )
    steering = create_user_message("steering")
    follow_up = create_user_message("follow-up")
    agent.steer(steering)
    agent.follow_up(follow_up)

    with pytest.raises(RuntimeError) as error:
        await agent.continue_()
    assert "No messages to continue from" in str(error.value)

    assert agent.peek_queued_messages() == [steering]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


def _tool_result_tail() -> list[Any]:
    return [
        create_user_message("existing user"),
        create_assistant_tool_use_message([ToolCall(id="call-1", name="noop", arguments={})]),
        ToolResultMessage(
            role="toolResult",
            tool_call_id="call-1",
            tool_name="noop",
            content=[TextContent(text="done")],
            is_error=False,
            timestamp=1,
        ),
    ]


@pytest.mark.parametrize(
    "messages",
    [[create_user_message("existing user")], _tool_result_tail()],
    ids=["user", "toolResult"],
)
async def test_defers_follow_up_input_on_the_first_continuation_request(messages: list[Any]) -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        requests.append(
            [
                message.content
                for message in context.messages
                if getattr(message, "role", None) == "user" and isinstance(message.content, str)
            ]
        )
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("done")))
        return stream

    agent = Agent(initial_state=AgentInitialState(messages=messages), stream_fn=stream_fn)
    agent.follow_up(create_user_message("follow-up"))

    await agent.continue_()

    assert len(requests) == 2
    assert "follow-up" not in requests[0]
    assert "follow-up" in requests[1]


@pytest.mark.parametrize(
    ("mode", "expected_requests"),
    [("one-at-a-time", 2), ("all", 1)],
)
async def test_polls_steering_at_continuation_startup(mode: str, expected_requests: int) -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        requests.append(
            [
                message.content
                for message in context.messages
                if getattr(message, "role", None) == "user" and isinstance(message.content, str)
            ]
        )
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("done")))
        return stream

    agent = Agent(
        initial_state=AgentInitialState(messages=[create_user_message("existing")]),
        steering_mode=mode,
        stream_fn=stream_fn,
    )
    agent.steer(create_user_message("first"))
    agent.steer(create_user_message("second"))

    await agent.continue_()

    assert len(requests) == expected_requests
    assert "first" in requests[0]
    if mode == "one-at-a-time":
        assert "second" not in requests[0]
        assert "second" in requests[1]
    else:
        assert "second" in requests[0]


async def test_keeps_steering_ahead_of_follow_up_from_a_non_assistant_continuation_tail() -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any = None) -> Any:
        requests.append(
            [
                message.content
                for message in context.messages
                if getattr(message, "role", None) == "user" and isinstance(message.content, str)
            ]
        )
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("done")))
        return stream

    agent = Agent(
        initial_state=AgentInitialState(messages=[create_user_message("existing")]),
        stream_fn=stream_fn,
    )
    agent.steer(create_user_message("steering"))
    agent.follow_up(create_user_message("follow-up"))

    await agent.continue_()

    assert len(requests) == 2
    assert "steering" in requests[0]
    assert "follow-up" not in requests[0]
    assert "follow-up" in requests[1]


@pytest.mark.parametrize("stop_reason", ["error", "aborted"])
async def test_keeps_queues_on_a_failed_response_even_when_finish_turn_requests_continuation(
    stop_reason: str,
) -> None:
    queued_during_response = create_user_message("steering")
    follow_up = create_user_message("follow-up")

    def stream_fn(_model: Any, _context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        message = create_assistant_message(stop_reason, stop_reason)
        message.error_message = stop_reason
        stream.push(error_event(stop_reason, message))
        return stream

    agent = Agent(finish_turn=lambda *_args: {"action": "continue"}, stream_fn=stream_fn)
    agent.follow_up(follow_up)
    agent.subscribe(
        lambda event, _signal: agent.steer(queued_during_response)
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
        else None
    )

    await agent.prompt("start")

    assert agent.peek_queued_messages() == [queued_during_response]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


async def test_keeps_queues_when_finish_turn_ends_the_run() -> None:
    queued_during_response = create_user_message("steering")
    follow_up = create_user_message("follow-up")

    agent = Agent(
        finish_turn=lambda *_args: {"action": "end"},
        stream_fn=scripted_stream_fn([create_assistant_message("done")]),
    )
    agent.follow_up(follow_up)
    agent.subscribe(
        lambda event, _signal: agent.steer(queued_during_response)
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
        else None
    )

    await agent.prompt("start")

    assert agent.peek_queued_messages() == [queued_during_response]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


def test_previews_the_next_selected_queued_messages_without_consuming_them() -> None:
    agent = Agent(
        steering_mode="one-at-a-time",
        follow_up_mode="all",
        stream_fn=unused_stream_function,
    )
    first = create_user_message("first steering")
    second = create_user_message("second steering")
    follow_up = create_user_message("follow-up")
    agent.steer(first)
    agent.steer(second)
    agent.follow_up(follow_up)

    assert agent.peek_queued_messages() == [first]
    assert agent.peek_queued_messages() == [first]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


async def test_forwards_provider_stream_event_observers_through_agent_options() -> None:
    provider_events: list[Any] = []

    def stream_fn(_model: Any, _context: Any, options: Any = None) -> Any:
        options.on_provider_stream_event({"request_cost": 0.01}, _model)
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("ok")))
        return stream

    agent = Agent(
        on_provider_stream_event=lambda data, _model: provider_events.append(data),
        stream_fn=stream_fn,
    )

    await agent.prompt("hello")

    assert provider_events == [{"request_cost": 0.01}]


async def test_forwards_session_id_to_stream_function_options() -> None:
    received: dict[str, Any] = {}

    def stream_fn(_model: Any, _context: Any, options: Any = None) -> Any:
        received["session_id"] = options.session_id
        stream = create_assistant_message_event_stream()
        stream.push(done_event("stop", create_assistant_message("ok")))
        return stream

    agent = Agent(session_id="session-abc", stream_fn=stream_fn)

    await agent.prompt("hello")
    assert received["session_id"] == "session-abc"

    agent.session_id = "session-def"
    assert agent.session_id == "session-def"

    await agent.prompt("hello again")
    assert received["session_id"] == "session-def"
