"""移植自 ``openai-responses-namespace.test.ts``。"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator

from pi_ai.api.openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    process_responses_stream,
)
from pi_ai.types import AssistantMessage, Model, ModelCost, ToolCall, Usage
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.serde import to_json

from openai_port_harness import responses_context

_MODEL = Model(
    id="gpt-5.4",
    name="GPT-5.4",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=400_000,
    max_tokens=128_000,
)


def _create_output() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=_MODEL.api,
        provider=_MODEL.provider,
        model=_MODEL.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=0,
    )


async def _create_function_call_events() -> AsyncIterator[dict]:
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "lookup",
            "arguments": "",
        },
    }
    yield {
        "type": "response.output_item.done",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "lookup",
            "arguments": '{"value":"hello"}',
            "namespace": "dynamic_tools",
        },
    }
    yield {"type": "response.completed", "sequence_number": 2, "response": {"id": "resp_test", "status": "completed"}}


async def _create_custom_tool_call_events() -> AsyncIterator[dict]:
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "type": "custom_tool_call",
            "id": "ctc_test",
            "call_id": "call_test",
            "name": "query",
            "input": "",
        },
    }
    yield {
        "type": "response.output_item.done",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "custom_tool_call",
            "id": "ctc_test",
            "call_id": "call_test",
            "name": "query",
            "input": "hello",
            "namespace": "dynamic_tools",
        },
    }
    yield {"type": "response.completed", "sequence_number": 2, "response": {"id": "resp_test", "status": "completed"}}


def _get_tool_call(output: AssistantMessage) -> ToolCall:
    block = output.content[0]
    assert block.type == "toolCall", "Expected toolCall block"
    return block


async def test_omits_an_absent_error_message():
    output = _create_output()
    await process_responses_stream(
        _create_function_call_events(), output, AssistantMessageEventStream(), _MODEL
    )

    assert output.error_message is None


async def test_round_trips_a_function_namespace_received_only_on_output_item_done():
    output = _create_output()
    await process_responses_stream(
        _create_function_call_events(), output, AssistantMessageEventStream(), _MODEL
    )

    tool_call = _get_tool_call(output)
    serialized = to_json(tool_call)
    assert serialized["id"] == "call_test|fc_test"
    assert serialized["name"] == "lookup"
    assert serialized["arguments"] == {"value": "hello"}
    assert serialized["namespace"] == "dynamic_tools"

    replayed = next(
        item
        for item in convert_responses_messages(_MODEL, responses_context([output]), {"openai"})
        if item.get("type") == "function_call"
    )
    assert replayed == {
        "type": "function_call",
        "id": "fc_test",
        "call_id": "call_test",
        "name": "lookup",
        "arguments": '{"value":"hello"}',
        "namespace": "dynamic_tools",
    }


async def test_round_trips_a_custom_tool_namespace_received_only_on_output_item_done():
    output = _create_output()
    options = ConvertResponsesMessagesOptions(grammar_tool_input_properties={"query": "input"})
    await process_responses_stream(
        _create_custom_tool_call_events(),
        output,
        AssistantMessageEventStream(),
        _MODEL,
        # 流处理器接收 OpenAIResponsesStreamOptions；grammar 属性存放于此。
        OpenAIResponsesStreamOptions(grammar_tool_input_properties={"query": "input"}),
    )

    tool_call = _get_tool_call(output)
    serialized = to_json(tool_call)
    assert serialized["id"] == "call_test|ctc_test"
    assert serialized["name"] == "query"
    assert serialized["arguments"] == {"input": "hello"}
    assert serialized["namespace"] == "dynamic_tools"

    replayed = next(
        item
        for item in convert_responses_messages(
            _MODEL, responses_context([output]), {"openai"}, options
        )
        if item.get("type") == "custom_tool_call"
    )
    assert replayed == {
        "type": "custom_tool_call",
        "id": "ctc_test",
        "call_id": "call_test",
        "name": "query",
        "input": "hello",
        "namespace": "dynamic_tools",
    }


def test_drops_namespaces_when_the_target_cannot_replay_their_load_items():
    output = _create_output()
    output.content.extend(
        [
            ToolCall(
                type="toolCall",
                id="call_function|fc_test",
                name="lookup",
                arguments={"value": "hello"},
                namespace="dynamic_tools",
            ),
            ToolCall(
                type="toolCall",
                id="call_custom|ctc_test",
                name="query",
                arguments={"input": "hello"},
                namespace="dynamic_tools",
            ),
        ]
    )
    target_models = [
        dataclasses.replace(_MODEL, id="gpt-5.2", name="GPT-5.2"),
        dataclasses.replace(_MODEL, provider="azure-openai-responses"),
        dataclasses.replace(
            _MODEL,
            api="openai-codex-responses",
            provider="openai-codex",
            id="gpt-5.3-codex-spark",
            name="GPT-5.3 Codex Spark",
        ),
    ]
    options = ConvertResponsesMessagesOptions(grammar_tool_input_properties={"query": "input"})

    for target_model in target_models:
        replayed = convert_responses_messages(
            target_model, responses_context([output]), {"openai"}, options
        )
        function_call = next(item for item in replayed if item.get("type") == "function_call")
        custom_tool_call = next(item for item in replayed if item.get("type") == "custom_tool_call")
        assert function_call is not None
        assert "namespace" not in function_call
        assert custom_tool_call is not None
        assert "namespace" not in custom_tool_call


def test_does_not_add_a_namespace_to_ordinary_function_calls():
    output = _create_output()
    output.content.append(
        ToolCall(
            type="toolCall",
            id="call_test|fc_test",
            name="lookup",
            arguments={"value": "hello"},
        )
    )

    replayed = next(
        item
        for item in convert_responses_messages(_MODEL, responses_context([output]), {"openai"})
        if item.get("type") == "function_call"
    )
    assert replayed is not None
    assert "namespace" not in replayed
