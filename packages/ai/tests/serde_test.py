"""带别名的 JSON 序列化器测试。

复现 TypeScript 数据模型依赖的行为：线上格式用 camelCase，Python 侧用 snake_case，
``None`` 的 dataclass 字段省略，判别字段（discriminator）优先分发。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pi_ai.types import (
    AssistantMessage,
    Model,
    ModelCompat,
    StopReason,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.serde import clone, content_block, from_json, to_json, type_discriminator


@dataclass
class _Aliased:
    plain: str = "a"
    camel_case: int = field(default=0, metadata={"alias": "camelCase"})


@dataclass
class _CatchAll:
    known: str = "k"
    extra: dict = field(default_factory=dict, metadata={"catch_all": True})


def test_to_json_emits_alias_names():
    assert to_json(_Aliased(camel_case=3)) == {"plain": "a", "camelCase": 3}


def test_to_json_omits_none_fields_but_keeps_dict_nulls():
    message = SystemMessage(content="hi", timestamp=1, sections={"a": None, "b": "x"})
    payload = to_json(message)
    assert payload["sections"] == {"a": None, "b": "x"}
    assert "toolsAdded" not in payload


def test_to_json_emits_discriminator_first():
    assert list(to_json(TextContent(text="x")).keys())[:2] == ["type", "text"]


def test_from_json_maps_aliases_and_defaults():
    restored = from_json(_Aliased, {"camelCase": 9})
    assert restored.plain == "a"
    assert restored.camel_case == 9


def test_from_json_ignores_unknown_keys_without_catch_all():
    assert from_json(_Aliased, {"nope": 1}).plain == "a"


def test_catch_all_preserves_unknown_keys_round_trip():
    original = _CatchAll(known="v", extra={"providerSpecific": 7})
    payload = to_json(original)
    assert payload == {"known": "v", "providerSpecific": 7}
    assert from_json(_CatchAll, payload).extra == {"providerSpecific": 7}


def test_enum_values_serialize_as_strings():
    assert to_json(Usage())["cost"] == {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}
    message = AssistantMessage(stop_reason=StopReason.TOOL_USE)
    assert to_json(message)["stopReason"] == "toolUse"


def test_str_enum_compares_equal_to_plain_string():
    assert StopReason.TOOL_USE == "toolUse"
    assert AssistantMessage(stop_reason="toolUse").stop_reason == StopReason.TOOL_USE


def test_from_json_dispatches_message_union_on_role():
    payload = {"role": "user", "content": "hi", "timestamp": 1}
    assert isinstance(from_json(UserMessage, payload), UserMessage)
    system = from_json(SystemMessage, {"role": "system", "content": "s", "timestamp": 0})
    assert system.role == "system"


def test_from_json_dispatches_content_union_on_type():
    thinking = from_json(ThinkingContent, {"type": "thinking", "thinking": "why"})
    assert isinstance(thinking, ThinkingContent)
    tool_call = from_json(ToolCall, {"type": "toolCall", "id": "1", "name": "f", "arguments": {"a": 1}})
    assert tool_call.arguments == {"a": 1}


def test_from_json_string_union_does_not_stringify_lists():
    from pi_ai.types import UserContent

    restored = from_json(list[UserContent], [{"type": "text", "text": "hi"}])
    assert isinstance(restored[0], TextContent)
    assert from_json(str | list[UserContent], "plain") == "plain"


def test_model_round_trip_through_json_is_lossless():
    raw = {
        "type": "chat",
        "id": "m",
        "name": "M",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://example.test",
        "input": ["text", "image"],
        "cost": {"input": 1.0, "output": 2.0, "cacheRead": 0.5, "cacheWrite": 3.0},
        "reasoning": True,
        "contextWindow": 200000,
        "maxTokens": 8192,
        "thinkingLevelMap": {"off": None, "high": "high"},
        "promptCache": {"short": 300},
        "compat": {"supportsStrictMode": True, "supportsMidConvoSystemMessages": True},
    }
    model = from_json(Model, raw)
    assert model.base_url == "https://example.test"
    assert model.cost.cache_read == 0.5
    assert model.thinking_level_map == {"off": None, "high": "high"}
    assert model.compat is not None and model.compat.supports_strict_mode is True
    assert to_json(model) == raw


def test_tool_parameters_pass_through_as_json_schema():
    tool = from_json(Tool, {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}})
    assert tool.parameters == {"type": "object", "properties": {}}
    assert to_json(tool)["parameters"] == {"type": "object", "properties": {}}


def test_tool_result_details_keep_arbitrary_json():
    result = from_json(
        ToolResultMessage,
        {
            "role": "toolResult",
            "toolCallId": "t",
            "toolName": "f",
            "content": [],
            "isError": False,
            "timestamp": 0,
            "details": {"nested": [1, 2, {"deep": True}]},
        },
    )
    assert result.details == {"nested": [1, 2, {"deep": True}]}


def test_clone_is_deep():
    model = Model(id="m", compat=ModelCompat(supports_strict_mode=True))
    copied = clone(model)
    copied.compat.supports_strict_mode = False
    assert model.compat.supports_strict_mode is True


def test_content_block_decorator_records_wire_type():
    @content_block("custom")
    @dataclass
    class _Custom:
        value: str = ""
        type: str = "custom"

    assert type_discriminator(_Custom) == "custom"


def test_unknown_dataclass_is_returned_unchanged_for_any_typed_field():
    assert from_json(object, {"a": 1}) == {"a": 1}


def test_thinking_signature_alias_round_trips():
    block = ThinkingContent(thinking="", thinking_signature="sig", redacted=True)
    payload = to_json(block)
    assert payload == {"type": "thinking", "thinking": "", "thinkingSignature": "sig", "redacted": True}
    assert from_json(ThinkingContent, payload).thinking_signature == "sig"


@pytest.mark.parametrize("value", [None, True, 0, 1.5, "text", [1, 2], {"a": 1}])
def test_primitives_round_trip(value):
    assert to_json(value) == value
    assert from_json(type(value), value) == value
