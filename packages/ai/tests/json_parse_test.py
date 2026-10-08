"""容错 JSON 解析测试。"""

from __future__ import annotations

import pytest

from pi_ai.utils.json_parse import parse_json_with_repair, parse_streaming_json, repair_json


def test_valid_json_passes_through():
    assert parse_json_with_repair('{"a": 1}') == {"a": 1}


def test_repair_json_escapes_raw_control_characters_inside_strings():
    assert repair_json('{"a": "line\nbreak"}') == '{"a": "line\\nbreak"}'
    assert repair_json('{"a": "tab\there"}') == '{"a": "tab\\there"}'


def test_repair_json_doubles_backslashes_before_invalid_escapes():
    assert repair_json(r'{"a": "C:\path"}') == r'{"a": "C:\\path"}'


def test_repair_json_keeps_valid_escapes_and_unicode_escapes():
    assert repair_json(r'{"a": "quote\" and \u00e9"}') == r'{"a": "quote\" and \u00e9"}'


def test_repair_json_leaves_characters_outside_strings_alone():
    assert repair_json('{"a": 1}') == '{"a": 1}'
    assert repair_json("plain text") == "plain text"


def test_repair_json_handles_a_trailing_backslash():
    assert repair_json('{"a": "x\\') == '{"a": "x\\\\'


def test_parse_json_with_repair_recovers_control_characters():
    assert parse_json_with_repair('{"a": "x\ny"}') == {"a": "x\ny"}


def test_parse_json_with_repair_reraises_when_nothing_changed():
    with pytest.raises(ValueError):
        parse_json_with_repair("{not json")


def test_parse_streaming_json_returns_empty_for_blank_input():
    assert parse_streaming_json(None) == {}
    assert parse_streaming_json("") == {}
    assert parse_streaming_json("   ") == {}


def test_parse_streaming_json_reads_a_complete_document():
    assert parse_streaming_json('{"a": [1, 2]}') == {"a": [1, 2]}


def test_parse_streaming_json_closes_an_incomplete_object():
    assert parse_streaming_json('{"a": 1') == {"a": 1}
    assert parse_streaming_json('{"a": {"b": 2') == {"a": {"b": 2}}


def test_parse_streaming_json_closes_an_incomplete_string():
    assert parse_streaming_json('{"a": "he') == {"a": "he"}


def test_parse_streaming_json_drops_a_partial_literal():
    assert parse_streaming_json('{"a": tru') == {}
    assert parse_streaming_json('{"a": true, "b": fal') == {"a": True}


def test_parse_streaming_json_closes_an_incomplete_array():
    assert parse_streaming_json('{"a": [1, 2') == {"a": [1, 2]}
    assert parse_streaming_json("[1, 2") == [1, 2]


def test_parse_streaming_json_drops_a_trailing_comma_and_key():
    assert parse_streaming_json('{"a": 1,') == {"a": 1}
    assert parse_streaming_json('{"a": 1, "b"') == {"a": 1}
    assert parse_streaming_json('{"a": 1, "b":') == {"a": 1}


def test_parse_streaming_json_returns_empty_when_unrecoverable():
    assert parse_streaming_json("}{") == {}


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ('{"name": "wea', {"name": "wea"}),
        ('{"name": "weather", "city": "Par', {"name": "weather", "city": "Par"}),
        ('{"name": "weather", "args": {"city": "Paris"', {"name": "weather", "args": {"city": "Paris"}}),
        ('{"count": 4', {"count": 4}),
        ('{"flag": true', {"flag": True}),
        ('{"items": [1, 2, 3', {"items": [1, 2, 3]}),
    ],
)
def test_parse_streaming_json_handles_progressive_prefixes(prefix, expected):
    assert parse_streaming_json(prefix) == expected


def test_repair_json_is_idempotent_on_repaired_output():
    once = repair_json('{"a": "x\ny"}')
    assert repair_json(once) == once
