"""Anthropic strict 工具 schema 的测试。

移植自 ``packages/ai/test/anthropic-strict-tool-schema.test.ts``。
"""

from __future__ import annotations

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.types import Context, JsonSchemaConstrainedSampling, Model, ModelCompat, Tool, UserMessage
from pi_ai.utils.transcript import normalize_context


class PayloadCaptured(Exception):
    """从 ``onPayload`` 抛出的哨兵异常，用于截获请求体。"""


def create_model() -> Model:
    return Model(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        api="anthropic-messages",
        provider="test-anthropic",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        context_window=200000,
        max_tokens=32000,
        compat=ModelCompat(force_adaptive_thinking=True, supports_strict_tools=True),
    )


def create_tool(parameters: dict, constrained_sampling: JsonSchemaConstrainedSampling | None = None) -> Tool:
    tool = Tool(name="lookup", description="Look up a value", parameters=parameters)
    if constrained_sampling is not None:
        tool.constrained_sampling = constrained_sampling
    return tool


def create_strict_tool(parameters: dict) -> Tool:
    return create_tool(parameters, JsonSchemaConstrainedSampling(type="json_schema", strict="prefer"))


async def capture_first_tool(tool: Tool) -> dict:
    captured: dict = {}

    def on_payload(payload, captured_model):
        captured["payload"] = payload
        raise PayloadCaptured()

    stream_object = stream(
        create_model(),
        normalize_context(Context(messages=[UserMessage(content="Use the tool", timestamp=0)], tools=[tool])),
        AnthropicOptions(api_key="test-key", cache_retention="none", on_payload=on_payload),
    )
    await stream_object.result()

    payload = captured.get("payload")
    tools = payload.get("tools") if payload else None
    assert tools, "Expected a tool in the captured Anthropic payload"
    return tools[0]


async def test_only_sends_the_full_input_schema_for_strict_json_schema_tools():
    legacy_parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
        "title": "LookupInput",
    }
    legacy_tool = await capture_first_tool(create_tool(legacy_parameters))
    assert "strict" not in legacy_tool
    assert legacy_tool["input_schema"] == {
        "type": "object",
        "properties": legacy_parameters["properties"],
        "required": legacy_parameters["required"],
    }

    strict_parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "optional": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "required": ["value", "optional"],
        "title": "StrictLookupInput",
    }
    strict_tool = await capture_first_tool(create_strict_tool(strict_parameters))
    assert strict_tool["strict"] is True
    assert strict_tool["input_schema"]["additionalProperties"] is False
    assert strict_tool["input_schema"]["required"] == ["value", "optional"]
    assert strict_tool["input_schema"]["properties"]["optional"] == {
        "anyOf": [{"type": "number"}, {"type": "null"}]
    }
    assert strict_tool["input_schema"]["title"] == "StrictLookupInput"


async def test_sends_prefer_tools_non_strict_when_they_use_keywords_anthropic_strict_mode_rejects():
    unsupported_parameters = [
        {"type": "object", "properties": {"timeoutMs": {"type": "integer", "minimum": 1, "maximum": 300000}}},
        {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {"tags": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
                }
            },
        },
        {"type": "object", "properties": {"expression": {"type": "string", "format": "regex"}}},
    ]
    for parameters in unsupported_parameters:
        tool = await capture_first_tool(create_strict_tool(parameters))
        assert "strict" not in tool

    supported_tool = await capture_first_tool(
        create_strict_tool(
            {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "minLength": 1, "maxLength": 1000, "pattern": "^[a-z]+$"},
                    "url": {"type": "string", "format": "uri"},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
            }
        )
    )
    assert supported_tool["strict"] is True
