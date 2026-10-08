"""移植 ``packages/ai/test/faux-provider.test.ts``（"faux provider" 用例）。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pi_ai.compat import complete, register_faux_provider, stream
from pi_ai.providers.faux import (
    FauxModelDefinition,
    FauxProviderRegistration,
    FauxTokenSize,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from pi_ai.types import (
    Context,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.abort import AbortController
from pi_ai.utils.serde import to_json

#: 上游 ``Date.now()`` 的确定性替身。
NOW = 1_700_000_000_000

#: 上游 ``Type.Object({ text: Type.String() })`` 序列化后恰好是这个结果。
ECHO_PARAMETERS = {
    "type": "object",
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}

_registrations: list[FauxProviderRegistration] = []


@pytest.fixture(autouse=True)
def _unregister_faux_providers():
    """对应上游 ``afterEach``：丢弃测试创建的所有注册。"""
    yield
    while _registrations:
        _registrations.pop().unregister()


def _register(options: RegisterFauxProviderOptions | None = None) -> FauxProviderRegistration:
    registration = register_faux_provider(options)
    _registrations.append(registration)
    return registration


async def _collect_events(stream_result) -> list:
    """把流消费为列表，与上游 ``collectEvents`` 辅助函数一致。"""
    return [event async for event in stream_result]


def _content(message) -> list:
    return to_json(message.content)


# ---------------------------------------------------------------------------------
# 注册、脚本与用量
# ---------------------------------------------------------------------------------


async def test_registers_a_custom_provider_and_estimates_usage():
    registration = _register()
    registration.set_responses([faux_assistant_message("hello world")])

    context = Context(
        system_prompt="Be concise.",
        messages=[UserMessage(content="hi there", timestamp=NOW)],
    )

    response = await complete(registration.get_model(), context)
    assert _content(response) == [{"type": "text", "text": "hello world"}]
    assert response.usage.input > 0
    assert response.usage.output > 0
    assert response.usage.total_tokens == response.usage.input + response.usage.output
    assert registration.state.call_count == 1


async def test_supports_helper_blocks_for_text_thinking_and_tool_calls():
    registration = _register()
    registration.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("think"), faux_tool_call("echo", {"text": "hi"}), faux_text("done")],
                stop_reason="toolUse",
            )
        ]
    )

    response = await complete(
        registration.get_model(),
        Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
    )

    blocks = _content(response)
    assert blocks[0] == {"type": "thinking", "thinking": "think"}
    assert isinstance(blocks[1]["id"], str) and blocks[1]["id"] != ""
    assert {key: value for key, value in blocks[1].items() if key != "id"} == {
        "type": "toolCall",
        "name": "echo",
        "arguments": {"text": "hi"},
    }
    assert blocks[2] == {"type": "text", "text": "done"}
    assert response.stop_reason == "toolUse"


async def test_supports_multiple_models_with_per_model_reasoning_and_model_aware_factories():
    registration = _register(
        RegisterFauxProviderOptions(
            models=[
                FauxModelDefinition(id="faux-fast", name="Faux Fast", reasoning=False),
                FauxModelDefinition(id="faux-thinker", name="Faux Thinker", reasoning=True),
            ]
        )
    )
    registration.set_responses(
        [
            lambda _context, _options, _state, model: faux_assistant_message(
                f"{model.id}:{model.reasoning}"
            ),
            lambda _context, _options, _state, model: faux_assistant_message(
                f"{model.id}:{model.reasoning}"
            ),
        ]
    )

    assert [model.id for model in registration.models] == ["faux-fast", "faux-thinker"]
    assert registration.get_model() is registration.models[0]
    fast_model = registration.get_model("faux-fast")
    thinker_model = registration.get_model("faux-thinker")
    assert fast_model is not None and fast_model.reasoning is False
    assert thinker_model is not None and thinker_model.reasoning is True

    fast = await complete(
        registration.get_model("faux-fast"), Context(messages=[UserMessage(content="hi", timestamp=NOW)])
    )
    thinker = await complete(
        registration.get_model("faux-thinker"), Context(messages=[UserMessage(content="hi", timestamp=NOW)])
    )

    assert _content(fast) == [{"type": "text", "text": "faux-fast:False"}]
    assert _content(thinker) == [{"type": "text", "text": "faux-thinker:True"}]


async def test_rewrites_api_provider_and_model_on_returned_messages():
    registration = _register(
        RegisterFauxProviderOptions(
            api="faux:test",
            provider="faux-provider",
            models=[FauxModelDefinition(id="faux-model")],
        )
    )
    registration.set_responses([faux_assistant_message("hello")])

    response = await complete(
        registration.get_model(),
        Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
    )

    assert response.api == "faux:test"
    assert response.provider == "faux-provider"
    assert response.model == "faux-model"


async def test_consumes_queued_responses_in_order_and_errors_when_exhausted():
    registration = _register()
    registration.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])

    context = Context(messages=[UserMessage(content="hi", timestamp=NOW)])

    first = await complete(registration.get_model(), context)
    second = await complete(registration.get_model(), context)
    exhausted = await complete(registration.get_model(), context)

    assert _content(first) == [{"type": "text", "text": "first"}]
    assert _content(second) == [{"type": "text", "text": "second"}]
    assert exhausted.stop_reason == "error"
    assert exhausted.error_message == "No more faux responses queued"
    assert registration.get_pending_response_count() == 0
    assert registration.state.call_count == 3


async def test_can_replace_and_append_queued_responses():
    registration = _register()
    registration.set_responses([faux_assistant_message("first")])

    context = Context(messages=[UserMessage(content="hi", timestamp=NOW)])

    assert _content(await complete(registration.get_model(), context)) == [
        {"type": "text", "text": "first"}
    ]
    assert registration.get_pending_response_count() == 0

    registration.set_responses([faux_assistant_message("second")])
    assert registration.get_pending_response_count() == 1
    assert _content(await complete(registration.get_model(), context)) == [
        {"type": "text", "text": "second"}
    ]

    registration.append_responses([faux_assistant_message("third"), faux_assistant_message("fourth")])
    assert registration.get_pending_response_count() == 2
    assert _content(await complete(registration.get_model(), context)) == [
        {"type": "text", "text": "third"}
    ]
    assert _content(await complete(registration.get_model(), context)) == [
        {"type": "text", "text": "fourth"}
    ]
    assert registration.get_pending_response_count() == 0


async def test_supports_async_response_factories():
    registration = _register()

    # 移植版的 FauxResponseFactory 是 4 参；上游的 3 参箭头函数依赖
    # JavaScript 会忽略多余的 `model` 参数。
    async def factory(context, _options, state, _model):
        return faux_assistant_message(f"{len(context.messages)}:{state.call_count}")

    registration.set_responses([factory])

    response = await complete(
        registration.get_model(),
        Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
    )

    assert _content(response) == [{"type": "text", "text": "1:1"}]


async def test_emits_an_error_when_a_response_factory_throws():
    registration = _register()

    def factory(_context, _options, _state, _model):
        raise RuntimeError("boom")

    registration.set_responses([factory])

    events = await _collect_events(
        stream(registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=NOW)]))
    )

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].error.stop_reason == "error"
    assert events[0].error.error_message == "boom"


async def test_rejects_a_queued_response_without_a_terminal_stop_reason():
    registration = _register()
    registration.set_responses([faux_assistant_message("partial", stop_reason="pending")])

    events = await _collect_events(
        stream(registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=NOW)]))
    )

    assert not any(event.type == "done" for event in events)
    terminal = events[-1]
    assert terminal.type == "error"
    assert terminal.error.stop_reason == "error"
    assert terminal.error.error_message == "Faux response ended without a stop reason"


async def test_estimates_prompt_and_output_tokens_from_serialized_context():
    registration = _register()
    registration.set_responses([faux_assistant_message("done")])

    tool = Tool(name="echo", description="Echo back text", parameters=ECHO_PARAMETERS)
    context = Context(
        system_prompt="sys",
        messages=[
            UserMessage(
                content=[TextContent(type="text", text="hello")],
                timestamp=1,
            ),
            faux_assistant_message("prior"),
            ToolResultMessage(
                tool_call_id="tool-1",
                tool_name="echo",
                content=[TextContent(type="text", text="tool out")],
                is_error=False,
                timestamp=2,
            ),
        ],
        tools=[tool],
    )

    response = await complete(registration.get_model(), context)

    # 上游计算 ceil(promptText.length / 4) 和 ceil("done".length / 4)；
    # 上述序列化后的 context 恰好得到 53 和 1。
    assert response.usage.input == 53
    assert response.usage.output == 1
    assert response.usage.cache_read == 0
    assert response.usage.cache_write == 0
    assert response.usage.total_tokens == 53 + 1


async def test_does_not_share_cache_across_sessions_or_requests_without_session_id():
    registration = _register()
    registration.set_responses(
        [
            faux_assistant_message("first"),
            faux_assistant_message("second"),
            faux_assistant_message("third"),
        ]
    )

    context = Context(messages=[UserMessage(content="hello", timestamp=NOW)])

    first = await complete(
        registration.get_model(),
        context,
        SimpleStreamOptions(session_id="session-1", cache_retention="short"),
    )
    assert first.usage.cache_write > 0
    context.messages.append(first)
    context.messages.append(UserMessage(content="follow up", timestamp=NOW + 1))

    second = await complete(
        registration.get_model(),
        context,
        SimpleStreamOptions(session_id="session-2", cache_retention="short"),
    )
    assert second.usage.cache_read == 0
    assert second.usage.cache_write > 0

    third = await complete(registration.get_model(), context)
    assert third.usage.cache_read == 0
    assert third.usage.cache_write == 0


async def test_simulates_prompt_caching_per_session_id():
    registration = _register()
    registration.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])

    context = Context(
        system_prompt="Be concise.",
        messages=[UserMessage(content="hello", timestamp=NOW)],
    )

    first = await complete(
        registration.get_model(),
        context,
        SimpleStreamOptions(session_id="session-1", cache_retention="short"),
    )
    assert first.usage.cache_read == 0
    assert first.usage.cache_write > 0

    context.messages.append(first)
    context.messages.append(UserMessage(content="follow up", timestamp=NOW + 1))

    second = await complete(
        registration.get_model(),
        context,
        SimpleStreamOptions(session_id="session-1", cache_retention="short"),
    )
    assert second.usage.cache_read > 0
    assert second.usage.input + second.usage.cache_read > second.usage.input


async def test_does_not_simulate_caching_when_cache_retention_is_none():
    registration = _register()
    registration.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])

    context = Context(messages=[UserMessage(content="hello", timestamp=NOW)])

    await complete(
        registration.get_model(),
        context,
        SimpleStreamOptions(session_id="session-1", cache_retention="none"),
    )
    context.messages.append(faux_assistant_message("first"))
    context.messages.append(UserMessage(content="follow up", timestamp=NOW + 1))
    second = await complete(
        registration.get_model(),
        context,
        SimpleStreamOptions(session_id="session-1", cache_retention="none"),
    )
    assert second.usage.cache_read == 0
    assert second.usage.cache_write == 0


# ---------------------------------------------------------------------------------
# 流式响应
# ---------------------------------------------------------------------------------


async def test_streams_thinking_text_and_partial_tool_call_deltas():
    registration = _register()
    registration.set_responses(
        [
            faux_assistant_message(
                [
                    faux_thinking("thinking text"),
                    faux_text("answer text"),
                    faux_tool_call("echo", {"text": "hi", "count": 12}, {"id": "tool-1"}),
                ],
                stop_reason="toolUse",
            )
        ]
    )

    events: list[str] = []
    tool_call_deltas: list[str] = []
    stream_result = stream(
        registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=NOW)])
    )
    async for event in stream_result:
        events.append(event.type)
        if event.type == "toolcall_delta":
            tool_call_deltas.append(event.delta)

    assert "thinking_start" in events
    assert "thinking_delta" in events
    assert "text_start" in events
    assert "text_delta" in events
    assert "toolcall_start" in events
    assert "toolcall_delta" in events
    assert "toolcall_end" in events
    assert len(tool_call_deltas) > 1
    assert json.loads("".join(tool_call_deltas)) == {"text": "hi", "count": 12}


async def test_streams_an_exact_event_order_for_fixed_size_chunks():
    registration = _register(RegisterFauxProviderOptions(token_size=FauxTokenSize(min=1, max=1)))
    registration.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("go"), faux_text("ok"), faux_tool_call("echo", {}, {"id": "tool-1"})],
                stop_reason="toolUse",
            )
        ]
    )

    events = await _collect_events(
        stream(registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=NOW)]))
    )

    assert events[0].type == "start"
    assert events[0].partial.stop_reason == "pending"
    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]


async def test_streams_multiple_tool_calls_in_one_message():
    registration = _register()
    registration.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("echo", {"text": "one"}, {"id": "tool-1"}),
                    faux_tool_call("echo", {"text": "two"}, {"id": "tool-2"}),
                ],
                stop_reason="toolUse",
            )
        ]
    )

    events = await _collect_events(
        stream(registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=NOW)]))
    )

    assert len([event for event in events if event.type == "toolcall_start"]) == 2
    assert len([event for event in events if event.type == "toolcall_end"]) == 2


async def test_streams_an_explicit_assistant_error_message_as_a_terminal_error():
    registration = _register(RegisterFauxProviderOptions(token_size=FauxTokenSize(min=2, max=2)))
    registration.set_responses(
        [
            replace(
                faux_assistant_message("partial"),
                stop_reason="error",
                error_message="upstream failed",
            )
        ]
    )

    events = await _collect_events(
        stream(registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=NOW)]))
    )

    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    terminal = events[-1]
    assert terminal.type == "error"
    assert terminal.reason == "error"
    assert terminal.error.stop_reason == "error"
    assert terminal.error.error_message == "upstream failed"


async def test_streams_an_explicit_assistant_aborted_message_as_a_terminal_error():
    registration = _register(RegisterFauxProviderOptions(token_size=FauxTokenSize(min=2, max=2)))
    registration.set_responses(
        [
            replace(
                faux_assistant_message("partial"),
                stop_reason="aborted",
                error_message="Request was aborted",
            )
        ]
    )

    events = await _collect_events(
        stream(registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=NOW)]))
    )

    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    terminal = events[-1]
    assert terminal.type == "error"
    assert terminal.reason == "aborted"
    assert terminal.error.stop_reason == "aborted"
    assert terminal.error.error_message == "Request was aborted"


# ---------------------------------------------------------------------------------
# 节奏与取消
#
# 上游用真实定时器驱动这些用例（`tokensPerSecond`）；移植版保留相同接缝，
# 但提高速率，使每个脚本化 chunk 约 3ms 而非 30ms。
# ---------------------------------------------------------------------------------


async def test_supports_aborting_before_the_first_chunk():
    registration = _register(
        RegisterFauxProviderOptions(tokens_per_second=1000, token_size=FauxTokenSize(min=3, max=3))
    )
    registration.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

    controller = AbortController()
    controller.abort()
    events = await _collect_events(
        stream(
            registration.get_model(),
            Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
            SimpleStreamOptions(signal=controller.signal),
        )
    )

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].reason == "aborted"
    assert events[0].error.stop_reason == "aborted"


async def test_supports_aborting_mid_text_stream_when_paced():
    registration = _register(
        RegisterFauxProviderOptions(tokens_per_second=1000, token_size=FauxTokenSize(min=3, max=3))
    )
    registration.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

    controller = AbortController()
    events: list[str] = []
    text_delta_count = 0
    stream_result = stream(
        registration.get_model(),
        Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
        SimpleStreamOptions(signal=controller.signal),
    )
    async for event in stream_result:
        events.append(event.type)
        if event.type == "text_delta":
            text_delta_count += 1
            controller.abort()

    assert text_delta_count == 1
    assert "text_start" in events
    assert "text_delta" in events
    assert "error" in events
    assert "text_end" not in events


async def test_supports_aborting_mid_thinking_stream_when_paced():
    registration = _register(
        RegisterFauxProviderOptions(tokens_per_second=1000, token_size=FauxTokenSize(min=3, max=3))
    )
    registration.set_responses(
        [
            replace(
                faux_assistant_message("ignored"),
                content=[faux_thinking("abcdefghijklmnopqrstuvwxyz")],
            )
        ]
    )

    controller = AbortController()
    events: list[str] = []
    thinking_delta_count = 0
    stream_result = stream(
        registration.get_model(),
        Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
        SimpleStreamOptions(signal=controller.signal),
    )
    async for event in stream_result:
        events.append(event.type)
        if event.type == "thinking_delta":
            thinking_delta_count += 1
            controller.abort()

    assert thinking_delta_count == 1
    assert "thinking_start" in events
    assert "thinking_delta" in events
    assert "error" in events
    assert "thinking_end" not in events


async def test_supports_aborting_mid_toolcall_stream_when_paced():
    registration = _register(
        RegisterFauxProviderOptions(tokens_per_second=1000, token_size=FauxTokenSize(min=3, max=3))
    )
    registration.set_responses(
        [
            replace(
                faux_assistant_message("done"),
                content=[
                    ToolCall(
                        type="toolCall",
                        id="tool-1",
                        name="echo",
                        arguments={"text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789},
                    )
                ],
                stop_reason="toolUse",
            )
        ]
    )

    controller = AbortController()
    events: list[str] = []
    tool_call_delta_count = 0
    stream_result = stream(
        registration.get_model(),
        Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
        SimpleStreamOptions(signal=controller.signal),
    )
    async for event in stream_result:
        events.append(event.type)
        if event.type == "toolcall_delta":
            tool_call_delta_count += 1
            controller.abort()

    assert tool_call_delta_count == 1
    assert "toolcall_start" in events
    assert "toolcall_delta" in events
    assert "error" in events
    assert "toolcall_end" not in events


async def test_unregisters_the_provider():
    registration = _register()
    registration.set_responses([faux_assistant_message("hello")])
    registration.unregister()

    with pytest.raises(RuntimeError, match=f"No API provider registered for api: {registration.api}"):
        await complete(
            registration.get_model(),
            Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
        )


# ---------------------------------------------------------------------------------
# 移植差异防护
# ---------------------------------------------------------------------------------


async def test_accepts_plain_stream_options_like_upstream():
    from pi_ai.types import StreamOptions

    registration = _register()
    registration.set_responses([faux_assistant_message("hello")])

    response = await complete(
        registration.get_model(),
        Context(messages=[UserMessage(content="hi", timestamp=NOW)]),
        StreamOptions(session_id="session-1", cache_retention="short"),
    )
    assert response.stop_reason == StopReason.STOP
    assert response.error_message is None
