"""转录变换与受限采样（constrained sampling）辅助函数测试。"""

from __future__ import annotations

import pytest

from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    UnsupportedStrictJsonSchemaError,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    make_strict_json_schema,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from pi_ai.api.transform_messages import (
    transform_messages,
)
from pi_ai.types import (
    AssistantMessage,
    GrammarConstrainedSampling as GrammarConfig,
    JsonSchemaConstrainedSampling,
    Model,
    StopReason,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _model(**overrides) -> Model:
    base = {
        "id": "m",
        "provider": "p",
        "api": "openai-completions",
        "input": ["text", "image"],
    }
    base.update(overrides)
    return Model(**base)


def _assistant(content, *, provider="p", api="openai-completions", model="m", stop_reason=StopReason.TOOL_USE):
    return AssistantMessage(
        provider=provider, api=api, model=model, content=list(content), stop_reason=stop_reason, timestamp=0
    )


# ---------------------------------------------------------------------------------
# transform_messages（转录变换）
# ---------------------------------------------------------------------------------


def test_cross_model_thinking_becomes_text_and_is_dropped_when_empty():
    from pi_ai.types import ThinkingContent

    content = [
        ThinkingContent(thinking="reasoning", thinking_signature="sig"),
        ThinkingContent(thinking="   "),
    ]
    assistant = _assistant(content, provider="other")
    transformed = transform_messages([assistant], _model())
    assert len(transformed) == 1
    assert transformed[0].content[0].type == "text"
    assert transformed[0].content[0].text == "reasoning"


def test_same_model_signed_thinking_is_preserved_even_when_empty():
    from pi_ai.types import ThinkingContent

    assistant = _assistant([ThinkingContent(thinking="", thinking_signature="sig")])
    transformed = transform_messages([assistant], _model())
    assert transformed[0].content[0].type == "thinking"
    assert transformed[0].content[0].thinking_signature == "sig"


def test_redacted_thinking_is_dropped_cross_model_but_kept_for_same_model():
    from pi_ai.types import ThinkingContent

    redacted = ThinkingContent(thinking="", thinking_signature="enc", redacted=True)
    same = transform_messages([_assistant([redacted])], _model())
    assert same[0].content and same[0].content[0].redacted is True

    cross = transform_messages([_assistant([redacted], provider="other")], _model())
    assert cross[0].content == []


def test_cross_model_tool_call_thought_signature_is_dropped():
    call = ToolCall(id="t", name="f", thought_signature="sig")
    transformed = transform_messages([_assistant([call], provider="other")], _model())
    assert transformed[0].content[0].thought_signature is None


def test_tool_call_id_normalizer_rewrites_calls_and_results():
    call = ToolCall(id="very|long|id", name="f")
    transcript = [
        _assistant([call]),
        ToolResultMessage(tool_call_id="very|long|id", tool_name="f", content=[], timestamp=0),
    ]
    transformed = transform_messages(transcript, _model(provider="other"), lambda tool_id, model, message: "short")
    assert transformed[0].content[0].id == "short"
    assert transformed[1].tool_call_id == "short"


def test_normalizer_is_not_applied_for_the_same_model():
    call = ToolCall(id="keep", name="f")
    transcript = [_assistant([call]), ToolResultMessage(tool_call_id="keep", tool_name="f", content=[], timestamp=0)]
    transformed = transform_messages(transcript, _model(), lambda tool_id, model, message: "changed")
    assert transformed[0].content[0].id == "keep"


def test_orphaned_tool_calls_get_synthetic_error_results():
    transcript = [_assistant([ToolCall(id="t", name="f")]), UserMessage(content="next", timestamp=1)]
    transformed = transform_messages(transcript, _model())
    synthetic = [message for message in transformed if message.role == "toolResult"]
    assert len(synthetic) == 1
    assert synthetic[0].is_error is True
    assert synthetic[0].content[0].text == "No result provided"
    assert [message.role for message in transformed] == ["assistant", "toolResult", "user"]


def test_answered_tool_calls_do_not_get_synthetic_results():
    transcript = [
        _assistant([ToolCall(id="t", name="f")]),
        ToolResultMessage(tool_call_id="t", tool_name="f", content=[TextContent(text="ok")], timestamp=1),
    ]
    transformed = transform_messages(transcript, _model())
    assert sum(1 for message in transformed if message.role == "toolResult") == 1


def test_trailing_orphaned_tool_calls_are_closed_at_the_end():
    transformed = transform_messages([_assistant([ToolCall(id="t", name="f")])], _model())
    assert [message.role for message in transformed] == ["assistant", "toolResult"]


def test_errored_and_aborted_assistant_messages_are_dropped():
    errored = _assistant([TextContent(text="partial")], stop_reason=StopReason.ERROR)
    aborted = _assistant([TextContent(text="partial")], stop_reason=StopReason.ABORTED)
    transformed = transform_messages([errored, aborted, UserMessage(content="hi", timestamp=0)], _model())
    assert [message.role for message in transformed] == ["user"]


def test_system_message_between_tool_call_and_result_is_held_back():
    transcript = [
        _assistant([ToolCall(id="t", name="f")]),
        SystemMessage(content="update", timestamp=2),
        ToolResultMessage(tool_call_id="t", tool_name="f", content=[], timestamp=3),
    ]
    transformed = transform_messages(transcript, _model())
    assert [message.role for message in transformed] == ["assistant", "toolResult", "system"]


def test_none_content_is_normalized_to_an_empty_list():
    assistant = _assistant([])
    assistant.content = None
    transformed = transform_messages([assistant], _model())
    assert transformed[0].content == []


# ---------------------------------------------------------------------------------
# constrained sampling（受限采样）
# ---------------------------------------------------------------------------------


def test_strict_json_schema_requires_every_property_and_forbids_extras():
    strict = make_strict_json_schema({"type": "object", "properties": {"a": {"type": "string"}}, "required": []})
    assert strict["required"] == ["a"]
    assert strict["additionalProperties"] is False
    assert strict["properties"]["a"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


def test_strict_json_schema_keeps_required_properties_plain():
    strict = make_strict_json_schema(
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    )
    assert strict["properties"]["a"] == {"type": "string"}


def test_strict_json_schema_rejects_a_boolean_root_schema():
    with pytest.raises(UnsupportedStrictJsonSchemaError, match="root schema must have type object"):
        make_strict_json_schema(True)


def test_strict_json_schema_rejects_nested_boolean_schemas():
    with pytest.raises(UnsupportedStrictJsonSchemaError, match="boolean schemas"):
        make_strict_json_schema({"type": "object", "properties": {"a": True}})


def test_strict_json_schema_rejects_a_non_object_root_schema():
    with pytest.raises(UnsupportedStrictJsonSchemaError, match="root schema must have type object"):
        make_strict_json_schema({"type": "string"})


def test_strict_json_schema_rejects_unsupported_keywords():
    with pytest.raises(UnsupportedStrictJsonSchemaError, match=r"\$ref"):
        make_strict_json_schema({"type": "object", "$ref": "#/x"})


def test_strict_json_schema_rejects_structured_unions():
    with pytest.raises(UnsupportedStrictJsonSchemaError, match="object and array unions"):
        make_strict_json_schema(
            {
                "type": "object",
                "properties": {"a": {"anyOf": [{"type": "object", "properties": {}}, {"type": "string"}]}},
            }
        )


def test_strict_json_schema_rejects_schema_valued_additional_properties():
    with pytest.raises(UnsupportedStrictJsonSchemaError, match="additionalProperties"):
        make_strict_json_schema({"type": "object", "properties": {}, "additionalProperties": {"type": "string"}})


def test_strict_json_schema_rejects_unknown_required_property():
    with pytest.raises(UnsupportedStrictJsonSchemaError, match="unknown property"):
        make_strict_json_schema({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]})


def test_strict_json_schema_does_not_mutate_the_input():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    make_strict_json_schema(schema)
    assert schema == {"type": "object", "properties": {"a": {"type": "string"}}}


def test_resolve_json_schema_strict_sampling_prefers_and_requires():
    tool = Tool(name="t", parameters={"type": "object", "properties": {}})
    tool.constrained_sampling = JsonSchemaConstrainedSampling(strict="prefer")
    assert resolve_json_schema_strict_sampling(tool, True) is True
    assert resolve_json_schema_strict_sampling(tool, False) is None

    tool.constrained_sampling = JsonSchemaConstrainedSampling(strict="require")
    assert resolve_json_schema_strict_sampling(tool, True) is True
    with pytest.raises(ValueError, match="requires JSON-schema constrained sampling"):
        resolve_json_schema_strict_sampling(tool, False)


def test_resolve_json_schema_strict_sampling_ignores_unconstrained_tools():
    assert resolve_json_schema_strict_sampling(Tool(name="t", parameters={}), True) is None


def test_resolve_json_schema_strict_sampling_falls_back_for_unsupported_schemas():
    tool = Tool(name="t", parameters={"type": "object", "$ref": "#/x"})
    tool.constrained_sampling = JsonSchemaConstrainedSampling(strict="prefer")
    assert resolve_json_schema_strict_sampling(tool, True) is None


def _grammar_tool(strict_required: bool = True) -> Tool:
    tool = Tool(
        name="grammar",
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
    )
    tool.constrained_sampling = GrammarConfig(variants={"openai_lark": "start: /.+/"})
    return tool


def test_resolve_grammar_constrained_sampling_infers_the_input_property():
    resolved = resolve_grammar_constrained_sampling(_grammar_tool(), True)
    assert resolved is not None
    assert resolved.format == "lark"
    assert resolved.definition == "start: /.+/"
    assert resolved.input_property == "input"


def test_resolve_grammar_constrained_sampling_returns_none_when_unsupported():
    assert resolve_grammar_constrained_sampling(_grammar_tool(), False) is None


def test_resolve_grammar_constrained_sampling_requires_a_variant():
    tool = _grammar_tool()
    tool.constrained_sampling = GrammarConfig(variants={})
    with pytest.raises(ValueError, match="no supported grammar variant"):
        resolve_grammar_constrained_sampling(tool, True)


def test_resolve_grammar_constrained_sampling_rejects_a_multi_property_schema():
    tool = _grammar_tool()
    tool.parameters["required"] = ["input", "other"]
    with pytest.raises(ValueError, match="exactly one required string property"):
        resolve_grammar_constrained_sampling(tool, True)


def test_create_grammar_tool_input_properties_maps_names():
    assert create_grammar_tool_input_properties([_grammar_tool(), Tool(name="plain", parameters={})], True) == {
        "grammar": "input"
    }
    assert create_grammar_tool_input_properties(None, True) == {}


def test_get_grammar_tool_input_requires_a_string():
    assert get_grammar_tool_input("t", {"input": "abc"}, "input") == "abc"
    with pytest.raises(ValueError, match="requires argument"):
        get_grammar_tool_input("t", {"input": 1}, "input")


def test_append_grammar_tool_input_json_delta_builds_incremental_json():
    buffer = GrammarToolInputJsonBuffer()
    # JSON 包装以 `{"input":"` 开头，只在最后一次调用时闭合。
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", False) == '{"input":"ab'
    assert append_grammar_tool_input_json_delta(buffer, "input", "abcd", False) == "cd"
    assert append_grammar_tool_input_json_delta(buffer, "input", "abcd", True) == '"}'
    assert buffer.closed is True


def test_append_grammar_tool_input_json_delta_is_idempotent_after_close():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "input", "ab", True)
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", True) is None
    with pytest.raises(ValueError, match="changed after it was closed"):
        append_grammar_tool_input_json_delta(buffer, "input", "abc", True)


def test_append_grammar_tool_input_json_delta_rejects_non_monotonic_input():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "input", "abc", False)
    with pytest.raises(ValueError, match="non-monotonically"):
        append_grammar_tool_input_json_delta(buffer, "input", "ab", False)


def test_append_grammar_tool_input_json_delta_returns_none_when_nothing_changed():
    buffer = GrammarToolInputJsonBuffer()
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", False) == '{"input":"ab'
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", False) is None
