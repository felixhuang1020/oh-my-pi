"""移植自 ``openai-responses-partial-json-cleanup.test.ts``。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pi_ai.api.openai_responses_shared import process_responses_stream
from pi_ai.types import AssistantMessage, Model, ModelCost, Usage
from pi_ai.utils.event_stream import AssistantMessageEventStream


class _RecordingStream(AssistantMessageEventStream):
    """记住所有推入事件的事件流（移植版的 ``push`` 间谍）。"""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def push(self, event) -> None:
        self.events.append(event)
        super().push(event)


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


async def _create_function_call_events(arguments_json: str) -> AsyncIterator[dict]:
    yield {
        "type": "response.output_item.added",
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "edit",
            "arguments": "",
        },
    }
    yield {"type": "response.function_call_arguments.delta", "delta": '{"path":"README.md"'}
    yield {"type": "response.function_call_arguments.delta", "delta": ',"content":"updated"}'}
    yield {"type": "response.function_call_arguments.done", "arguments": arguments_json}
    yield {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "edit",
            "arguments": arguments_json,
        },
    }
    yield {"type": "response.completed", "sequence_number": 5, "response": {"id": "resp_test", "status": "completed"}}


async def test_removes_partial_json_from_persisted_tool_call_blocks_at_output_item_done():
    model = Model(
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
    output = _create_output(model)
    stream = _RecordingStream()
    arguments_json = '{"path":"README.md","content":"updated"}'

    await process_responses_stream(_create_function_call_events(arguments_json), output, stream, model)

    assert len(output.content) == 1
    persisted_tool_call = output.content[0]
    assert persisted_tool_call.type == "toolCall"
    assert persisted_tool_call.arguments == {"path": "README.md", "content": "updated"}
    assert not hasattr(persisted_tool_call, "partial_json")

    tool_call_end = next(event for event in stream.events if event.type == "toolcall_end")
    assert tool_call_end.tool_call is persisted_tool_call
    assert not hasattr(tool_call_end.tool_call, "partial_json")
