"""tool 调用参数校验与类型强转测试。"""

from __future__ import annotations

import json

import pytest

from pi_ai.types import Tool, ToolCall
from pi_ai.utils.validation import (
    coerce_with_json_schema,
    normalize_optional_nulls,
    validate_tool_arguments,
    validate_tool_call,
)


def _tool(parameters: dict, name: str = "f") -> Tool:
    return Tool(name=name, description="d", parameters=parameters)


def _call(arguments: dict, name: str = "f") -> ToolCall:
    return ToolCall(id="1", name=name, arguments=arguments)


STRING_SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}
NUMBER_SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
BOOL_SCHEMA = {"type": "object", "properties": {"flag": {"type": "boolean"}}, "required": ["flag"]}


# ---------------------------------------------------------------------------------
# 校验（validation）
# ---------------------------------------------------------------------------------


def test_valid_arguments_pass_through():
    assert validate_tool_arguments(_tool(STRING_SCHEMA), _call({"value": "ok"})) == {"value": "ok"}


def test_unknown_tool_name_raises():
    with pytest.raises(ValueError, match='Tool "missing" not found'):
        validate_tool_call([_tool(STRING_SCHEMA)], _call({}, name="missing"))


def test_missing_required_property_reports_the_path():
    with pytest.raises(ValueError) as error:
        validate_tool_arguments(_tool(STRING_SCHEMA), _call({}))
    message = str(error.value)
    assert 'Validation failed for tool "f"' in message
    assert "value" in message
    assert "Received arguments" in message


def test_wrong_type_is_reported_after_coercion_attempts():
    with pytest.raises(ValueError, match="Validation failed"):
        validate_tool_arguments(
            _tool({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}),
            _call({"n": "not a number"}),
        )


def test_validated_arguments_are_echoed_in_the_error_message():
    with pytest.raises(ValueError) as error:
        validate_tool_arguments(_tool(STRING_SCHEMA), _call({"value": {"nested": 1}}))
    assert json.dumps({"value": {"nested": 1}}, indent=2) in str(error.value)


# ---------------------------------------------------------------------------------
# 类型强转（coercion）
# ---------------------------------------------------------------------------------


def test_string_number_is_coerced_to_integer():
    assert validate_tool_arguments(_tool(NUMBER_SCHEMA), _call({"count": "42"})) == {"count": 42}


def test_string_boolean_is_coerced():
    assert validate_tool_arguments(_tool(BOOL_SCHEMA), _call({"flag": "true"})) == {"flag": True}
    assert validate_tool_arguments(_tool(BOOL_SCHEMA), _call({"flag": "false"})) == {"flag": False}


def test_numeric_boolean_is_coerced():
    assert validate_tool_arguments(_tool(BOOL_SCHEMA), _call({"flag": 1})) == {"flag": True}


def test_number_is_coerced_to_string():
    assert validate_tool_arguments(_tool(STRING_SCHEMA), _call({"value": 7})) == {"value": "7"}


def test_null_number_becomes_zero():
    assert validate_tool_arguments(_tool(NUMBER_SCHEMA), _call({"count": None})) == {"count": 0}


def test_nested_object_properties_are_coerced():
    schema = {
        "type": "object",
        "properties": {"nested": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        "required": ["nested"],
    }
    assert validate_tool_arguments(_tool(schema), _call({"nested": {"n": "3"}})) == {"nested": {"n": 3}}


def test_array_items_are_coerced():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
        "required": ["items"],
    }
    assert validate_tool_arguments(_tool(schema), _call({"items": ["1", "2"]})) == {"items": [1, 2]}


def test_coerce_with_json_schema_handles_union_types():
    schema = {"type": ["integer", "string"]}
    assert coerce_with_json_schema("7", schema) == "7"
    assert coerce_with_json_schema(7, schema) == 7


def test_coerce_with_json_schema_ignores_non_dict_schemas():
    assert coerce_with_json_schema("x", True) == "x"


# ---------------------------------------------------------------------------------
# 可选字段 null 归一化（optional null normalisation）
# ---------------------------------------------------------------------------------


def test_optional_null_properties_are_dropped():
    schema = {
        "type": "object",
        "properties": {"required": {"type": "string"}, "optional": {"type": "string"}},
        "required": ["required"],
    }
    arguments = {"required": "a", "optional": None}
    normalize_optional_nulls(arguments, schema)
    assert arguments == {"required": "a"}


def test_required_null_properties_are_kept_for_the_validator():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}
    arguments = {"value": None}
    normalize_optional_nulls(arguments, schema)
    assert arguments == {"value": None}


def test_nullable_optional_properties_are_kept():
    schema = {"type": "object", "properties": {"maybe": {"type": ["string", "null"]}}, "required": []}
    arguments = {"maybe": None}
    normalize_optional_nulls(arguments, schema)
    assert arguments == {"maybe": None}


def test_nested_optional_nulls_are_dropped():
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"kept": {"type": "string"}, "dropped": {"type": "string"}},
                "required": ["kept"],
            }
        },
        "required": ["nested"],
    }
    arguments = {"nested": {"kept": "a", "dropped": None}}
    normalize_optional_nulls(arguments, schema)
    assert arguments == {"nested": {"kept": "a"}}


def test_array_optional_nulls_are_dropped():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"a": {"type": "string"}}, "required": []},
            }
        },
        "required": ["items"],
    }
    arguments = {"items": [{"a": None}]}
    normalize_optional_nulls(arguments, schema)
    assert arguments == {"items": [{}]}


def test_schema_without_properties_is_left_alone():
    arguments = {"anything": None}
    normalize_optional_nulls(arguments, {"type": "object"})
    assert arguments == {"anything": None}


def test_ref_properties_are_not_stripped():
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/a"}}, "required": []}
    arguments = {"a": None}
    normalize_optional_nulls(arguments, schema)
    assert arguments == {"a": None}
