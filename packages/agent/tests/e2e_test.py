"""基于脚本化 stream 函数的端到端 agent 测试。

移植自 ``packages/agent/test/e2e.test.ts``。TypeScript 套件用 ``pi-ai/compat`` 的 faux
provider 驱动真实 ``streamSimple``；``pi_ai`` 的 Python 移植尚未提供 faux provider，因此
这些测试安装了一个行为等价的小型确定性脚本 provider（无网络、除 ``asyncio`` 外无定时器）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from pi_ai.types import (
    AssistantMessage,
    Cost,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import create_assistant_message_event_stream
from pi_ai.utils.events import (
    done_event,
    error_event,
    start_event,
    text_delta_event,
    text_end_event,
    text_start_event,
    thinking_delta_event,
    thinking_end_event,
    thinking_start_event,
)

from pi_agent import Agent, AgentInitialState

from utils.calculate import calculate_tool


def create_usage() -> Usage:
    return Usage(cost=Cost())


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(name: str, arguments: dict[str, Any], tool_call_id: str) -> ToolCall:
    return ToolCall(id=tool_call_id, name=name, arguments=arguments)


def faux_assistant_message(content: Any, stop_reason: str = "stop") -> AssistantMessage:
    blocks = content if isinstance(content, list) else [faux_text(content)]
    return AssistantMessage(
        role="assistant",
        content=blocks,
        api="faux",
        provider="faux",
        model="faux-model",
        usage=create_usage(),
        stop_reason=stop_reason,
        timestamp=0,
    )


def create_faux_model(*, reasoning: bool = False) -> Model:
    return Model(
        id="faux-model",
        name="Faux Model",
        api="faux",
        provider="faux",
        base_url="https://example.invalid",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=4096,
    )


@dataclass
class ScriptedProvider:
    """faux provider ``streamSimple`` 的确定性替身。"""

    responses: list[Any]
    index: int = 0

    def __call__(self, _model: Model, context: Any, _options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        entry = self.responses[self.index]
        self.index += 1
        message = entry(context) if callable(entry) and not isinstance(entry, AssistantMessage) else entry
        stream.push(start_event(message))
        for content_index, block in enumerate(message.content):
            if isinstance(block, TextContent):
                stream.push(text_start_event(content_index, message))
                for token in block.text.split(" "):
                    stream.push(text_delta_event(content_index, token + " ", message))
                stream.push(text_end_event(content_index, block.text, message))
            elif isinstance(block, ThinkingContent):
                stream.push(thinking_start_event(content_index, message))
                for token in block.thinking.split(" "):
                    stream.push(thinking_delta_event(content_index, token + " ", message))
                stream.push(thinking_end_event(content_index, block.thinking, message))
        stream.push(done_event(message.stop_reason, message))
        return stream


def get_text_content(message: AssistantMessage | ToolResultMessage) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


async def basic_prompt(model: Model) -> None:
    agent = Agent(
        stream_fn=ScriptedProvider([faux_assistant_message("4")]),
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant. Keep your responses concise.",
            model=model,
            thinking_level="off",
            tools=[],
        ),
    )

    await agent.prompt("What is 2+2? Answer with just the number.")

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 3
    assert agent.state.messages[0].role == "system"
    assert agent.state.messages[1].role == "user"
    assert agent.state.messages[2].role == "assistant"

    assistant_message = agent.state.messages[2]
    assert get_text_content(assistant_message).strip() == "4"


async def tool_execution(model: Model) -> None:
    agent = Agent(
        stream_fn=ScriptedProvider(
            [
                faux_assistant_message(
                    [faux_text("Let me calculate that."), faux_tool_call("calculate", {"expression": "123 * 456"}, "calc-1")],
                    stop_reason="toolUse",
                ),
                faux_assistant_message("The result is 56088."),
            ]
        ),
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant. Always use the calculator tool for math.",
            model=model,
            thinking_level="off",
            tools=[calculate_tool],
        ),
    )

    pending_tool_calls_during_events: list[dict[str, Any]] = []
    agent.subscribe(
        lambda event, _signal: pending_tool_calls_during_events.append(
            {"type": event.type, "ids": list(agent.state.pending_tool_calls)}
        )
        if event.type in ("tool_execution_start", "tool_execution_end")
        else None
    )

    await agent.prompt("Calculate 123 * 456 using the calculator tool.")

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 4
    tool_result_msg = next(message for message in agent.state.messages if message.role == "toolResult")
    assert "123 * 456 = 56088" in get_text_content(tool_result_msg)

    final_message = agent.state.messages[-1]
    assert final_message.role == "assistant"
    assert "56088" in get_text_content(final_message)
    assert agent.state.pending_tool_calls == set()
    assert pending_tool_calls_during_events == [
        {"type": "tool_execution_start", "ids": ["calc-1"]},
        {"type": "tool_execution_end", "ids": []},
    ]


async def abort_execution(model: Model) -> None:
    def stream_fn(_model: Model, _context: Any, options: Any = None) -> Any:
        stream = create_assistant_message_event_stream()
        message = faux_assistant_message("one two three four five six seven eight nine ten eleven twelve")
        stream.push(start_event(message))

        async def produce() -> None:
            for token in message.content[0].text.split(" "):
                if options.signal.aborted:
                    aborted = faux_assistant_message("", stop_reason="aborted")
                    aborted.error_message = "Request aborted by user"
                    stream.push(error_event("aborted", aborted))
                    return
                stream.push(text_delta_event(0, token + " ", message))
                await asyncio.sleep(0.02)
            stream.push(done_event("stop", message))

        asyncio.ensure_future(produce())
        return stream

    agent = Agent(
        stream_fn=stream_fn,
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant.",
            model=model,
            thinking_level="off",
            tools=[],
        ),
    )

    asyncio.get_running_loop().call_later(0.03, agent.abort)
    await agent.prompt("Count slowly from 1 to 20.")

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 2
    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert last_message.stop_reason == "aborted"
    assert last_message.error_message is not None
    assert agent.state.error_message == last_message.error_message


async def state_updates(model: Model) -> None:
    agent = Agent(
        stream_fn=ScriptedProvider([faux_assistant_message("1 2 3 4 5")]),
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant.",
            model=model,
            thinking_level="off",
            tools=[],
        ),
    )

    events: list[str] = []
    agent.subscribe(lambda event, _signal: events.append(event.type))

    await agent.prompt("Count from 1 to 5.")

    for expected in (
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ):
        assert expected in events
    assert events.index("agent_start") < events.index("message_start")
    assert events.index("message_start") < events.index("message_end")
    assert events.index("message_end") < len(events) - 1 - events[::-1].index("agent_end")

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 3


async def multi_turn_conversation(model: Model) -> None:
    def second_response(context: Any) -> AssistantMessage:
        has_alice = any(
            getattr(message, "role", None) == "user"
            and (
                (isinstance(message.content, str) and "Alice" in message.content)
                or (
                    isinstance(message.content, list)
                    and any(block.type == "text" and "Alice" in block.text for block in message.content)
                )
            )
            for message in context.messages
        )
        return faux_assistant_message("Your name is Alice." if has_alice else "I do not know your name.")

    agent = Agent(
        stream_fn=ScriptedProvider([faux_assistant_message("Nice to meet you, Alice."), second_response]),
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant.",
            model=model,
            thinking_level="off",
            tools=[],
        ),
    )

    await agent.prompt("My name is Alice.")
    assert len(agent.state.messages) == 3

    await agent.prompt("What is my name?")
    assert len(agent.state.messages) == 5

    last_message = agent.state.messages[4]
    assert last_message.role == "assistant"
    assert "alice" in get_text_content(last_message).lower()


async def test_handles_a_basic_text_prompt() -> None:
    await basic_prompt(create_faux_model())


async def test_executes_tools_and_tracks_pending_tool_calls() -> None:
    await tool_execution(create_faux_model())


async def test_handles_abort_during_streaming() -> None:
    await abort_execution(create_faux_model())


async def test_emits_lifecycle_updates_while_streaming() -> None:
    await state_updates(create_faux_model())


async def test_maintains_context_across_multiple_turns() -> None:
    await multi_turn_conversation(create_faux_model())


async def test_preserves_thinking_content_blocks() -> None:
    faux = ScriptedProvider([faux_assistant_message([faux_thinking("step by step"), faux_text("4")])])
    model = create_faux_model(reasoning=True)
    agent = Agent(
        stream_fn=faux,
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant.",
            model=model,
            thinking_level="low",
            tools=[],
        ),
    )

    await agent.prompt("What is 2+2?")

    assistant_message = agent.state.messages[2]
    assert assistant_message.role == "assistant"
    assert assistant_message.content == [faux_thinking("step by step"), faux_text("4")]


async def test_continue_throws_when_no_messages_in_context() -> None:
    agent = Agent(
        stream_fn=ScriptedProvider([]),
        initial_state=AgentInitialState(system_prompt="Test", model=create_faux_model()),
    )

    with pytest.raises(RuntimeError) as error:
        await agent.continue_()
    assert "No messages to continue from" in str(error.value)


async def test_continue_throws_when_last_message_is_assistant() -> None:
    model = create_faux_model()
    agent = Agent(
        stream_fn=ScriptedProvider([]),
        initial_state=AgentInitialState(system_prompt="Test", model=model),
    )
    agent.state.messages = [faux_assistant_message("Hello")]

    with pytest.raises(RuntimeError) as error:
        await agent.continue_()
    assert "Cannot continue from message role: assistant" in str(error.value)


async def test_continue_gets_a_response_when_last_message_is_user() -> None:
    provider = ScriptedProvider([faux_assistant_message("HELLO WORLD")])
    agent = Agent(
        stream_fn=provider,
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant. Follow instructions exactly.",
            model=create_faux_model(),
            thinking_level="off",
            tools=[],
        ),
    )
    agent.state.messages = [
        UserMessage(role="user", content=[TextContent(text="Say exactly: HELLO WORLD")], timestamp=0)
    ]

    await agent.continue_()

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 2
    assert agent.state.messages[0].role == "user"
    assert agent.state.messages[1].role == "assistant"
    assert "HELLO WORLD" in get_text_content(agent.state.messages[1]).upper()


async def test_continue_processes_tool_results() -> None:
    model = create_faux_model()
    agent = Agent(
        stream_fn=ScriptedProvider([faux_assistant_message("The answer is 8.")]),
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant. After getting a calculation result, state the answer clearly.",
            model=model,
            thinking_level="off",
            tools=[calculate_tool],
        ),
    )

    user_message = UserMessage(role="user", content=[TextContent(text="What is 5 + 3?")], timestamp=0)
    assistant_message = faux_assistant_message(
        [
            TextContent(text="Let me calculate that."),
            ToolCall(id="calc-1", name="calculate", arguments={"expression": "5 + 3"}),
        ],
        stop_reason="toolUse",
    )
    tool_result = ToolResultMessage(
        role="toolResult",
        tool_call_id="calc-1",
        tool_name="calculate",
        content=[TextContent(text="5 + 3 = 8")],
        is_error=False,
        timestamp=0,
    )
    agent.state.messages = [user_message, assistant_message, tool_result]

    await agent.continue_()

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 4
    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert "8" in get_text_content(last_message)
