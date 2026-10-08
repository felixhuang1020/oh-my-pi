"""移植自 ``openai-responses-terminal-event.test.ts``。

上游的 ``vi.mock("openai")`` 包装流，在移植版中变成经 ``options.fetch``
提供的预置 SSE 响应体。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pi_ai.api.openai_responses import OpenAIResponsesOptions, stream
from pi_ai.api.openai_responses_shared import process_responses_stream
from pi_ai.types import AssistantMessage, Model, ModelCost, TextContent, Usage, UserMessage
from pi_ai.utils.event_stream import AssistantMessageEventStream

from openai_port_harness import (
    FakeFetch,
    collect_events,
    responses_context,
    sse_response,
    sse_stream_response,
    stream_result,
)


def _create_model() -> Model:
    return Model(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=400_000,
        max_tokens=128_000,
    )


def _create_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=0,
    )


class _RecordingStream(AssistantMessageEventStream):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []
        #: 在 push 时采样的 ``partial.stop_reason``，与 TypeScript 的 push 间谍一致。
        self.observed_stop_reasons: list[object] = []

    def push(self, event) -> None:
        self.events.append(event)
        if hasattr(event, "partial"):
            self.observed_stop_reasons.append(event.partial.stop_reason)
        super().push(event)


def _wrapper_early_eof_events() -> list[dict]:
    return [
        {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_wrapper_early_eof"}},
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_wrapper_early_eof", "summary": []},
        },
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": 2,
            "output_index": 0,
            "content_index": 0,
            "item_id": "rs_wrapper_early_eof",
            "delta": "partial reasoning before the wrapper stream ends",
        },
    ]


async def _create_early_eof_events() -> AsyncIterator[dict]:
    yield {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_early_eof"}}
    yield {
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_early_eof", "summary": []},
    }
    yield {
        "type": "response.reasoning_text.delta",
        "sequence_number": 2,
        "output_index": 0,
        "content_index": 0,
        "item_id": "rs_early_eof",
        "delta": "partial reasoning before the stream ends",
    }


async def _create_completed_events() -> AsyncIterator[dict]:
    yield {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "id": "resp_completed",
            "status": "completed",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 7,
                "total_tokens": 27,
                "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
            },
        },
    }


async def _create_incomplete_events(reason: str = "max_output_tokens") -> AsyncIterator[dict]:
    yield {
        "type": "response.incomplete",
        "sequence_number": 0,
        "response": {
            "id": "resp_incomplete",
            "status": "incomplete",
            "incomplete_details": {"reason": reason},
            "usage": {
                "input_tokens": 30,
                "output_tokens": 12,
                "total_tokens": 42,
                "input_tokens_details": {"cached_tokens": 5},
            },
        },
    }


async def _create_failed_events() -> AsyncIterator[dict]:
    yield {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "id": "resp_failed",
            "status": "failed",
            "error": {"code": "server_error", "message": "boom"},
        },
    }


async def _create_phased_message_events(phases, terminal_status: str = "completed") -> AsyncIterator[dict]:
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_phase",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
            "phase": phases[0],
        },
    }
    yield {
        "type": "response.output_item.done",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_phase",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
            "phase": phases[1],
        },
    }
    if terminal_status == "incomplete":
        yield {
            "type": "response.incomplete",
            "sequence_number": 2,
            "response": {
                "id": "resp_phase",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        }
        return
    yield {"type": "response.completed", "sequence_number": 2, "response": {"id": "resp_phase", "status": "completed"}}


async def _create_unfinished_tool_call_events() -> AsyncIterator[dict]:
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "bash", "arguments": ""},
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "sequence_number": 1,
        "output_index": 0,
        "item_id": "fc_1",
        "delta": '{"command":"rm -rf /tmp/build',
    }
    yield {"type": "response.completed", "sequence_number": 2, "response": {"id": "resp_unfinished", "status": "completed"}}


async def _create_tool_calls_without_output_index_events() -> AsyncIterator[dict]:
    def call(name: str) -> dict:
        return {"type": "function_call", "id": f"fc_{name}", "call_id": f"call_{name}", "name": "bash"}

    events = [
        {"type": "response.output_item.added", "item": {**call("a"), "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_a", "delta": '{"command":"echo a"}'},
        {"type": "response.output_item.added", "item": {**call("b"), "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_b", "delta": '{"command":"echo b"}'},
        {"type": "response.output_item.done", "item": {**call("a"), "arguments": '{"command":"echo a"}'}},
        {"type": "response.output_item.done", "item": {**call("b"), "arguments": '{"command":"echo b"}'}},
        {"type": "response.completed", "response": {"id": "resp_no_output_index", "status": "completed"}},
    ]
    for event in events:
        yield event


async def test_rejects_streams_that_end_before_a_terminal_response_event():
    model = _create_model()
    with pytest.raises(RuntimeError, match="OpenAI Responses stream ended before a terminal response event"):
        await process_responses_stream(
            _create_early_eof_events(), _create_output(model), AssistantMessageEventStream(), model
        )


async def test_rejects_completed_streams_whose_tool_call_never_received_output_item_done():
    model = _create_model()
    with pytest.raises(
        RuntimeError,
        match=r"OpenAI Responses stream completed with an unfinished tool call: bash \(call_1\|fc_1\)",
    ):
        await process_responses_stream(
            _create_unfinished_tool_call_events(),
            _create_output(model),
            AssistantMessageEventStream(),
            model,
        )


async def test_rejects_parallel_tool_calls_without_output_index_instead_of_running_mixed_up_calls():
    model = _create_model()
    with pytest.raises(
        RuntimeError,
        match=r"OpenAI Responses stream completed with an unfinished tool call: bash \(call_a\|fc_a\)",
    ):
        await process_responses_stream(
            _create_tool_calls_without_output_index_events(),
            _create_output(model),
            AssistantMessageEventStream(),
            model,
        )


async def test_forwards_parsed_provider_stream_events_in_order():
    model = _create_model()
    context = responses_context(
        [UserMessage(content=[TextContent(type="text", text="hi")], timestamp=0)]
    )
    provider_events: list[object] = []
    event_models: list[object] = []

    async def on_provider_stream_event(event, event_model):
        provider_events.append(event)
        event_models.append(event_model)

    fetch = FakeFetch(sse_response(_wrapper_early_eof_events()))
    await stream_result(
        stream(
            model,
            context,
            OpenAIResponsesOptions(
                api_key="test", fetch=fetch, on_provider_stream_event=on_provider_stream_event
            ),
        )
    )

    assert len(provider_events) == 3
    assert [event["type"] for event in provider_events] == [
        "response.created",
        "response.output_item.added",
        "response.reasoning_text.delta",
    ]
    assert event_models == [model, model, model]


async def test_emits_an_error_final_result_when_the_wrapper_stream_ends_before_a_terminal_response_event():
    model = _create_model()
    context = responses_context(
        [UserMessage(content=[TextContent(type="text", text="hi")], timestamp=0)]
    )
    fetch = FakeFetch(sse_stream_response(_wrapper_early_eof_events()))
    stream_obj = stream(model, context, OpenAIResponsesOptions(api_key="test", fetch=fetch))
    events: list[object] = []
    initial_stop_reason = None
    async for event in stream_obj:
        if event.type == "start":
            # 在迭代过程中采样：start 事件的 partial 就是活跃的输出消息。
            initial_stop_reason = event.partial.stop_reason
        events.append(event)
    result = await stream_obj.result()

    assert initial_stop_reason == "pending"
    assert events[-1].type == "error"
    assert result.stop_reason == "error"
    assert result.error_message == "OpenAI Responses stream ended before a terminal response event"


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        (["commentary", "commentary"], ["pending", "pending"]),
        (["final_answer", "final_answer"], ["stop", "stop"]),
        (["commentary", "final_answer"], ["pending", "stop"]),
    ],
)
async def test_tracks_message_phases(phases, expected):
    model = _create_model()
    output = _create_output(model)
    stream_obj = _RecordingStream()

    await process_responses_stream(_create_phased_message_events(phases), output, stream_obj, model)

    assert stream_obj.observed_stop_reasons == expected
    assert output.stop_reason == "stop"


async def test_replaces_a_provisional_final_answer_stop_with_an_incomplete_terminal_reason():
    model = _create_model()
    output = _create_output(model)
    stream_obj = _RecordingStream()

    await process_responses_stream(
        _create_phased_message_events(["final_answer", "final_answer"], "incomplete"),
        output,
        stream_obj,
        model,
    )

    assert stream_obj.observed_stop_reasons == ["stop", "stop"]
    assert output.stop_reason == "length"


async def test_finalizes_completed_terminal_events_as_stop():
    model = _create_model()
    output = _create_output(model)

    await process_responses_stream(
        _create_completed_events(), output, AssistantMessageEventStream(), model
    )

    assert output.response_id == "resp_completed"
    assert output.stop_reason == "stop"
    assert output.raw_stop_reason == "completed"
    assert output.usage.input == 15
    assert output.usage.output == 7
    assert output.usage.cache_read == 2
    assert output.usage.cache_write == 3
    assert output.usage.total_tokens == 27


async def test_finalizes_incomplete_terminal_events_as_length_stops():
    model = _create_model()
    output = _create_output(model)

    await process_responses_stream(
        _create_incomplete_events(), output, AssistantMessageEventStream(), model
    )

    assert output.response_id == "resp_incomplete"
    assert output.stop_reason == "length"
    assert output.raw_stop_reason == "incomplete.max_output_tokens"
    assert output.usage.input == 25
    assert output.usage.output == 12
    assert output.usage.cache_read == 5
    assert output.usage.cache_write == 0
    assert output.usage.total_tokens == 42


async def test_finalizes_content_filtered_incomplete_responses_as_non_retryable_errors():
    model = _create_model()
    output = _create_output(model)

    await process_responses_stream(
        _create_incomplete_events("content_filter"), output, AssistantMessageEventStream(), model
    )

    assert output.stop_reason == "error"
    assert output.raw_stop_reason == "incomplete.content_filter"
    assert output.error_message == "Response incomplete: content_filter"


async def test_preserves_unknown_provider_incomplete_reasons_as_non_retryable_errors():
    model = _create_model()
    output = _create_output(model)

    await process_responses_stream(
        _create_incomplete_events("max_time_limit"), output, AssistantMessageEventStream(), model
    )

    assert output.stop_reason == "error"
    assert output.raw_stop_reason == "incomplete.max_time_limit"
    assert output.error_message == "Response incomplete: max_time_limit"


async def test_rejects_failed_terminal_events_with_the_provider_error():
    model = _create_model()
    output = _create_output(model)

    with pytest.raises(RuntimeError, match="server_error: boom"):
        await process_responses_stream(
            _create_failed_events(), output, AssistantMessageEventStream(), model
        )
    assert output.raw_stop_reason == "failed"
