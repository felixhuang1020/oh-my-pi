"""移植 ``packages/ai/test/google-shared-convert-tools.test.ts``。"""

from __future__ import annotations

from typing import Any

import pytest

from pi_ai.api.google_shared import (
    convert_tools,
    resolve_google_function_calling_mode,
    supports_google_strict_tool_sampling,
)
from pi_ai.types import JsonSchemaConstrainedSampling, Tool


def make_tool(parameters: dict[str, Any]) -> Tool:
    return Tool(name="test_tool", description="A test tool", parameters=parameters)


def test_strips_json_schema_meta_keys_from_parameters_when_use_parameters_true() -> None:
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$id": "urn:bash-tool",
                "$comment": "A bash tool for demonstration",
                "$defs": {
                    "commandDef": {"type": "string"},
                },
                "definitions": {
                    "legacyDef": {"type": "number"},
                },
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            }
        )
    ]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl is not None
    assert decl["parameters"] == {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    }
    for meta_key in ("$schema", "$id", "$comment", "$defs", "definitions"):
        assert meta_key not in decl["parameters"]


def test_recursively_strips_nested_json_schema_meta_keys() -> None:
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "deep": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "$id": "urn:nested",
                        "type": "string",
                    },
                },
            }
        )
    ]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl is not None
    assert decl["parameters"] == {
        "type": "object",
        "properties": {
            "deep": {
                "type": "string",
            },
        },
    }


def test_preserves_ref_while_stripping_meta_keys() -> None:
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "refProp": {
                        "$ref": "#/$defs/someDef",
                        "type": "string",
                    },
                },
            }
        )
    ]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl is not None
    assert decl["parameters"] == {
        "type": "object",
        "properties": {
            "refProp": {
                "$ref": "#/$defs/someDef",
                "type": "string",
            },
        },
    }


def test_does_not_mutate_the_original_tool_parameters_object() -> None:
    original_parameters = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    }
    tools = [make_tool(original_parameters)]

    convert_tools(tools, True)

    assert original_parameters == {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    }


def test_preserves_schema_in_parameters_json_schema_when_use_parameters_false() -> None:
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            }
        )
    ]

    result = convert_tools(tools, False)
    decl = result[0]["functionDeclarations"][0]

    assert decl is not None
    assert decl["parametersJsonSchema"] == {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    }


def test_handles_tools_without_schema_gracefully() -> None:
    tools = [
        make_tool(
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            }
        )
    ]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl is not None
    assert decl["parameters"] == {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
    }


def test_uses_validated_function_calling_for_strict_tools_on_gemini_3() -> None:
    tool = make_tool({"type": "object", "properties": {}})
    tool.constrained_sampling = JsonSchemaConstrainedSampling(type="json_schema", strict="require")

    assert supports_google_strict_tool_sampling("gemini-3.1-pro-preview") is True
    assert supports_google_strict_tool_sampling("gemini-2.5-pro") is False
    assert resolve_google_function_calling_mode([tool], None, True) == "VALIDATED"

    with pytest.raises(ValueError, match='Tool "test_tool" requires JSON-schema constrained sampling'):
        resolve_google_function_calling_mode([tool], None, False)


def test_returns_none_for_empty_tool_list() -> None:
    assert convert_tools([]) is None
    assert convert_tools([], True) is None
