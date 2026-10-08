"""移植自 ``constrained-sampling.test.ts``。

覆盖 provider 侧的约束采样辅助函数（严格 JSON-Schema 改写与 grammar 工具）、
Responses 工具转换器对它们的使用、Responses 消息转换器中的 grammar 回放路径，
以及流式 custom-tool-call 路径。

上游 fixture 使用 TypeBox 构建 schema；移植版以普通 JSON Schema 字典表示 schema，
因此 fixture 按 TypeBox 序列化后的 JSON 形式写出。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    UnsupportedStrictJsonSchemaError,
    append_grammar_tool_input_json_delta,
    make_strict_json_schema,
    resolve_json_schema_strict_sampling,
)
from pi_ai.api.openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    GrammarConstrainedSampling,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.serde import to_json
from pi_ai.utils.transcript import normalize_context


def _make_model() -> Model:
    return Model(
        id="gpt-test",
        name="GPT Test",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128_000,
        max_tokens=4096,
    )


def _make_output() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api="openai-responses",
        provider="openai",
        model="gpt-test",
        usage=Usage(),
        stop_reason="pending",
        timestamp=0,
    )


def _make_tool(**overrides: Any) -> Tool:
    base: dict[str, Any] = {
        "name": "sample_tool",
        "description": "Sample tool",
        "parameters": {
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "required": ["payload"],
            "additionalProperties": False,
        },
    }
    base.update(overrides)
    return Tool(**base)


def _capture_tool_call_events(stream: AssistantMessageEventStream) -> tuple[list[dict[str, Any]], list[str]]:
    """对应上游 ``captureToolCallEvents``：替换 ``push`` 并记录 start/delta 事件。"""
    starts: list[dict[str, Any]] = []
    deltas: list[str] = []
    original_push = stream.push

    def push(event: Any) -> None:
        if event.type == "toolcall_start":
            block = event.partial.content[event.content_index]
            if getattr(block, "type", None) == "toolCall":
                starts.append(dict(block.arguments))
        elif event.type == "toolcall_delta":
            deltas.append(event.delta)
        original_push(event)

    stream.push = push  # type: ignore[method-assign]
    return starts, deltas


async def _iterate_events(events: Sequence[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for event in events:
        yield event


def test_converts_supported_constraints_and_falls_back_when_unsupported() -> None:
    converted = convert_responses_tools(
        [_make_tool(constrained_sampling=JsonSchemaConstrainedSampling(type="json_schema", strict="prefer"))]
    )
    assert converted[0]["type"] == "function"
    assert converted[0]["name"] == "sample_tool"
    assert converted[0]["strict"] is True

    with pytest.raises(ValueError) as excinfo:
        convert_responses_tools(
            [_make_tool(constrained_sampling=JsonSchemaConstrainedSampling(type="json_schema", strict="require"))],
            ConvertResponsesToolsOptions(supports_strict_mode=False),
        )
    assert 'Tool "sample_tool" requires JSON-schema constrained sampling' in str(excinfo.value)

    grammar_tool = _make_tool(
        constrained_sampling=GrammarConstrainedSampling(
            type="grammar", variants={"openai_lark": "start: /[a-z]+/"}
        )
    )
    grammar_converted = convert_responses_tools(
        [grammar_tool], ConvertResponsesToolsOptions(supports_openai_grammar_tools=True)
    )[0]
    assert grammar_converted["type"] == "custom"
    assert grammar_converted["name"] == "sample_tool"
    assert grammar_converted["format"] == {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: /[a-z]+/",
    }

    with pytest.raises(ValueError) as excinfo:
        convert_responses_tools(
            [_make_tool(constrained_sampling=GrammarConstrainedSampling(type="grammar", variants={}))],
            ConvertResponsesToolsOptions(supports_openai_grammar_tools=True),
        )
    assert (
        'Tool "sample_tool" cannot use grammar constrained sampling: no supported grammar variant was provided'
        in str(excinfo.value)
    )

    fallback = convert_responses_tools(
        [grammar_tool],
        ConvertResponsesToolsOptions(supports_openai_grammar_tools=False, supports_strict_mode=False),
    )[0]
    assert fallback["type"] == "function"
    assert fallback["name"] == "sample_tool"
    assert "strict" not in fallback

    assert convert_responses_tools([_make_tool(constrained_sampling=False)]) == convert_responses_tools(
        [_make_tool()]
    )


def test_derives_strict_provider_schemas_without_changing_tool_definitions() -> None:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "number"},
            "metadata": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": [],
            },
            "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["path", "metadata"],
    }

    strict = make_strict_json_schema(parameters)

    assert "additionalProperties" not in parameters
    assert parameters["required"] == ["path", "metadata"]
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["path", "offset", "metadata", "nullable"]
    assert strict["properties"]["offset"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert strict["properties"]["metadata"]["additionalProperties"] is False
    assert strict["properties"]["metadata"]["required"] == ["enabled"]
    assert strict["properties"]["metadata"]["properties"]["enabled"] == {
        "anyOf": [{"type": "boolean"}, {"type": "null"}]
    }
    assert strict["properties"]["nullable"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "metadata": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["metadata"],
            },
            "additionalProperties is unsupported",
        ),
        (
            {
                "allOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                    {"type": "object", "properties": {"b": {"type": "number"}}, "required": ["b"]},
                ]
            },
            "allOf schemas are unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"nested": {"type": "string"}},
                                "required": ["nested"],
                            },
                            {"type": "null"},
                        ]
                    }
                },
                "required": ["value"],
            },
            "object and array unions are unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {"child": {"$ref": "https://example.com/child.json"}},
                "required": ["child"],
            },
            "$ref schemas are unsupported",
        ),
    ],
)
def test_falls_back_or_rejects_schemas_that_cannot_be_safely_converted(
    parameters: dict[str, Any], error: str
) -> None:
    with pytest.raises(UnsupportedStrictJsonSchemaError) as excinfo:
        make_strict_json_schema(parameters)
    assert error in str(excinfo.value)

    tool = _make_tool(
        parameters=parameters,
        constrained_sampling=JsonSchemaConstrainedSampling(type="json_schema", strict="prefer"),
    )
    assert resolve_json_schema_strict_sampling(tool, True) is None

    converted = convert_responses_tools([tool], ConvertResponsesToolsOptions(supports_strict_mode=True))[0]
    assert converted["strict"] is False
    assert converted["parameters"] == parameters

    tool.constrained_sampling = JsonSchemaConstrainedSampling(type="json_schema", strict="require")
    with pytest.raises(ValueError) as excinfo:
        resolve_json_schema_strict_sampling(tool, True)
    assert error in str(excinfo.value)


def test_replays_grammar_calls_as_custom_responses_items() -> None:
    replayed_tool_call = ToolCall(
        type="toolCall", id="call_1|ctc_1", name="sample_tool", arguments={"payload": "abc"}
    )
    context = normalize_context(
        Context(
            messages=[
                AssistantMessage(
                    api="openai-responses",
                    provider="openai",
                    model="gpt-test",
                    content=[replayed_tool_call],
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call_1|ctc_1",
                    tool_name="sample_tool",
                    content=[TextContent(type="text", text="done")],
                    is_error=False,
                    timestamp=0,
                ),
            ]
        )
    )
    options = ConvertResponsesMessagesOptions(grammar_tool_input_properties={"sample_tool": "payload"})

    for invalid_arguments in ({}, {"payload": 42}):
        replayed_tool_call.arguments = invalid_arguments
        with pytest.raises(ValueError) as excinfo:
            convert_responses_messages(_make_model(), context, {"openai"}, options)
        assert 'Grammar tool call "sample_tool" requires argument "payload" to be a string' in str(
            excinfo.value
        )

    replayed_tool_call.arguments = {"payload": "abc"}
    messages = convert_responses_messages(_make_model(), context, {"openai"}, options)

    assert {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "sample_tool",
        "input": "abc",
    } in messages
    assert {"type": "custom_tool_call_output", "call_id": "call_1", "output": "done"} in messages


def test_drops_foreign_item_ids_when_replaying_grammar_calls_as_custom_responses_items() -> None:
    context = normalize_context(
        Context(
            messages=[
                AssistantMessage(
                    api="pi-messages",
                    provider="radius",
                    model="gpt-other",
                    content=[
                        ToolCall(
                            type="toolCall",
                            id="call_1|ctc_1",
                            name="sample_tool",
                            arguments={"payload": "abc"},
                        )
                    ],
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call_1|ctc_1",
                    tool_name="sample_tool",
                    content=[TextContent(type="text", text="done")],
                    is_error=False,
                    timestamp=0,
                ),
            ]
        )
    )

    messages = convert_responses_messages(
        _make_model(),
        context,
        {"openai"},
        ConvertResponsesMessagesOptions(grammar_tool_input_properties={"sample_tool": "payload"}),
    )

    call = next(item for item in messages if item.get("type") == "custom_tool_call")
    assert call["type"] == "custom_tool_call"
    assert call["call_id"] == "call_1"
    assert call["input"] == "abc"
    assert call.get("id") is None


def test_keeps_grammar_input_json_deltas_append_only() -> None:
    buffer = GrammarToolInputJsonBuffer()
    first = append_grammar_tool_input_json_delta(buffer, "payload", 'a"', False)
    second = append_grammar_tool_input_json_delta(buffer, "payload", 'a"\nb', True)

    assert json.loads(f"{first}{second}") == {"payload": 'a"\nb'}
    assert append_grammar_tool_input_json_delta(buffer, "payload", 'a"\nb', True) is None

    with pytest.raises(ValueError) as excinfo:
        append_grammar_tool_input_json_delta(buffer, "payload", "changed", True)
    assert 'grammar tool input for property "payload" changed after it was closed' in str(excinfo.value)


async def test_starts_custom_responses_tool_calls_with_their_initial_input() -> None:
    output = _make_output()
    stream = AssistantMessageEventStream()
    starts, deltas = _capture_tool_call_events(stream)
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "id": "ctc_1",
                "name": "sample_tool",
                "input": "a",
            },
        },
        {
            "type": "response.custom_tool_call_input.delta",
            "output_index": 0,
            "item_id": "ctc_1",
            "delta": "b",
        },
        {
            "type": "response.custom_tool_call_input.done",
            "output_index": 0,
            "item_id": "ctc_1",
            "input": "abc",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "id": "ctc_1",
                "name": "sample_tool",
                "input": "abc",
            },
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        },
    ]

    await process_responses_stream(
        _iterate_events(events),
        output,
        stream,
        _make_model(),
        OpenAIResponsesStreamOptions(grammar_tool_input_properties={"sample_tool": "payload"}),
    )

    assert output.stop_reason == "toolUse"
    assert starts == [{"payload": "a"}]
    assert to_json(output.content) == [
        {"type": "toolCall", "id": "call_1|ctc_1", "name": "sample_tool", "arguments": {"payload": "abc"}}
    ]
    assert json.loads("".join(deltas)) == {"payload": "abc"}
