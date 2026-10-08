"""小型共享工具测试：哈希、SSE、错误响应体、用量估算、文本处理。"""

from __future__ import annotations

import pytest

from pi_ai.types import (
    AssistantMessage,
    StopReason,
    SystemMessage,
    TextContent,
    Tool,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UserMessage,
)
from pi_ai.utils.error_body import (
    MAX_PROVIDER_ERROR_BODY_CHARS,
    format_provider_error,
    normalize_provider_error,
    safe_json_stringify,
    truncate_error_text,
)
from pi_ai.utils.estimate import (
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_content_tokens,
    estimate_text_tokens,
)
from pi_ai.utils.hash import short_hash
from pi_ai.utils.headers import headers_to_record, provider_headers_to_record
from pi_ai.utils.sanitize_unicode import sanitize_surrogates
from pi_ai.utils.sse import SSEEvent, SSEDecoder, iter_sse_events
from pi_ai.utils.text import content_text, get_system_message_text, render_system_message_update
from pi_ai.utils.typebox_helpers import string_enum
from pi_ai.utils.uuid import uuidv7


# ---------------------------------------------------------------------------------
# 文本（text）
# ---------------------------------------------------------------------------------


def test_content_text_joins_text_blocks_only():
    blocks = [TextContent(text="a"), TextContent(text="b")]
    assert content_text(blocks) == "a\nb"
    assert content_text(blocks, separator=", ") == "a, b"
    assert content_text("already a string") == "already a string"


def test_get_system_message_text_appends_sections():
    message = SystemMessage(content="base", timestamp=0, sections={"a": "A", "b": None})
    assert get_system_message_text(message) == "base\n\nA"


def test_render_system_message_update_frames_section_changes():
    message = SystemMessage(content="patch", timestamp=0, sections={"a": "A", "b": None})
    rendered = render_system_message_update(message)
    assert rendered.startswith("patch")
    assert 'Updated system prompt section "a":\n\nA' in rendered
    assert 'Removed system prompt section "b".' in rendered


def test_string_enum_builds_the_provider_friendly_schema():
    assert string_enum(["a", "b"]) == {"type": "string", "enum": ["a", "b"]}
    assert string_enum(["a"], description="d", default="a") == {
        "type": "string",
        "enum": ["a"],
        "description": "d",
        "default": "a",
    }


# ---------------------------------------------------------------------------------
# 哈希 / HTTP 头 / Unicode
# ---------------------------------------------------------------------------------


def test_short_hash_is_deterministic_and_compact():
    first = short_hash("the quick brown fox")
    assert first == short_hash("the quick brown fox")
    assert first != short_hash("the quick brown fo")
    assert len(first) < 20


def test_short_hash_matches_the_typescript_cyrb53_derivation():
    # 黄金值由在 node 下运行 packages/ai/src/utils/hash.ts 产生。
    assert short_hash("") == "k4n83c7h0j2b"
    assert short_hash("hello") == "1h6qa0qrowduu"
    assert short_hash("the quick brown fox") == "116k32etn0sg2"
    assert short_hash("pi-ai") == "1qz2dom1t3pi79"


def test_short_hash_hashes_astral_characters_as_utf16_code_units():
    # ``charCodeAt`` 按 UTF-16 code unit 迭代，因此一个 emoji 是两个 unit。
    assert short_hash("\U0001f648") == "kphsz0153ms3q"
    assert short_hash("a\U0001f648b") == "q0megbstm1j3"


def test_headers_to_record_copies():
    assert headers_to_record({"A": "1"}) == {"A": "1"}


def test_provider_headers_to_record_merges_and_suppresses():
    assert provider_headers_to_record({"X-A": "1", "B": "2"}, {"x-a": "9"}) == {"x-a": "9", "B": "2"}
    assert provider_headers_to_record({"X-A": "1"}, {"x-a": None}) is None
    assert provider_headers_to_record(None, {}) is None


def test_sanitize_surrogates_removes_unpaired_units_only():
    assert sanitize_surrogates(f"Text {chr(0xD83D)} here") == "Text  here"
    emoji = chr(0x1F648)
    assert sanitize_surrogates(f"Hello {emoji}") == f"Hello {emoji}"


def test_uuidv7_is_time_ordered_and_well_formed():
    first = uuidv7(1_700_000_000_000)
    second = uuidv7(1_700_000_000_001)
    assert first < second
    parts = first.split("-")
    assert [len(part) for part in parts] == [8, 4, 4, 4, 12]
    assert first[14] == "7"
    assert first[19] in "89ab"


def test_uuidv7_rejects_out_of_range_timestamps():
    with pytest.raises(ValueError):
        uuidv7(-1)
    with pytest.raises(ValueError):
        uuidv7(0x1_0000_0000_0000)


def test_uuidv7_without_a_timestamp_is_monotonic():
    assert uuidv7() < uuidv7()


# ---------------------------------------------------------------------------------
# SSE（Server-Sent Events，服务端事件流）
# ---------------------------------------------------------------------------------


def test_sse_decoder_dispatches_on_blank_line():
    decoder = SSEDecoder()
    assert decoder.feed("event: message") == []
    assert decoder.feed('data: {"a":1}') == []
    events = decoder.feed("")
    assert len(events) == 1
    assert events[0].event == "message"
    assert events[0].data == '{"a":1}'
    assert events[0].json() == {"a": 1}


def test_sse_decoder_joins_multiple_data_lines():
    decoder = SSEDecoder()
    decoder.feed("data: line1")
    decoder.feed("data: line2")
    (event,) = decoder.feed("")
    assert event.data == "line1\nline2"


def test_sse_decoder_ignores_comments_and_tracks_id_and_retry():
    decoder = SSEDecoder()
    assert decoder.feed(": keep-alive") == []
    decoder.feed("id: 42")
    decoder.feed("retry: 1500")
    decoder.feed("data: x")
    (event,) = decoder.feed("")
    assert event.raw == [": keep-alive", "id: 42", "retry: 1500", "data: x"]
    assert event.id == "42"
    assert event.retry == 1500


def test_sse_decoder_requires_a_colon_for_field_parsing():
    decoder = SSEDecoder()
    decoder.feed("bare")
    (event,) = decoder.feed("")
    assert event.event is None
    assert event.data == ""


def test_sse_decoder_flush_dispatches_a_trailing_event():
    decoder = SSEDecoder()
    decoder.feed("data: tail")
    (event,) = decoder.flush()
    assert event.data == "tail"
    assert decoder.flush() == []


async def test_iter_sse_events_over_an_async_line_source():
    async def lines():
        for line in ["event: a", "data: 1", "", "data: 2", ""]:
            yield line

    events = [event async for event in iter_sse_events(lines())]
    assert [(event.event, event.data) for event in events] == [("a", "1"), (None, "2")]


async def test_iter_sse_events_accepts_a_sync_iterable_and_strips_carriage_returns():
    events = [event async for event in iter_sse_events(["data: 1\r\n", "\r\n"])]
    assert events[0].data == "1"


# ---------------------------------------------------------------------------------
# 错误响应体（error bodies）
# ---------------------------------------------------------------------------------


class _SdkError(Exception):
    def __init__(self, message, *, status=None, status_code=None, body=None, error=None, response=None):
        super().__init__(message)
        self.status = status
        self.status_code = status_code
        self.body = body
        self.error = error
        self.response = response


def test_normalize_provider_error_prefers_message_when_it_carries_the_body():
    error = _SdkError("bad request: invalid model", status=400, body="invalid model")
    normalized = normalize_provider_error(error)
    assert normalized.status == 400
    assert normalized.body == "invalid model"
    assert normalized.message_carries_body is True
    assert format_provider_error(normalized, "OpenAI") == "OpenAI (400): bad request: invalid model"


def test_normalize_provider_error_surfaces_a_body_the_message_omits():
    error = _SdkError("400 status code (no body)", status=400, body='{"error":"quota"}')
    normalized = normalize_provider_error(error)
    assert normalized.message_carries_body is False
    assert format_provider_error(normalized, "OpenAI") == 'OpenAI (400): {"error":"quota"}'
    assert format_provider_error(normalized) == '400: {"error":"quota"}'


def test_normalize_provider_error_reads_status_code_and_nested_response():
    assert normalize_provider_error(_SdkError("x", status_code=429)).status == 429
    assert normalize_provider_error(_SdkError("x", response={"statusCode": 503})).status == 503


def test_normalize_provider_error_reads_a_parsed_error_object():
    error = _SdkError("failure", error={"code": "bad"})
    normalized = normalize_provider_error(error)
    assert normalized.body == '{"code":"bad"}'


def test_normalize_provider_error_ignores_empty_bodies_and_streams():
    class _Stream:
        def pipe(self):
            return None

    assert normalize_provider_error(_SdkError("x", body="   ")).body is None
    assert normalize_provider_error(_SdkError("x", response={"body": _Stream()})).body is None


def test_normalize_provider_error_handles_non_exceptions():
    normalized = normalize_provider_error("plain string")
    assert normalized.message == '"plain string"'
    assert normalized.message_carries_body is False


def test_truncate_error_text_notes_how_much_was_dropped():
    assert truncate_error_text("abc", 10) == "abc"
    assert truncate_error_text("abcdef", 3) == "abc... [truncated 3 chars]"
    assert MAX_PROVIDER_ERROR_BODY_CHARS == 4000


def test_safe_json_stringify_uses_compact_separators():
    # JSON.stringify 在 `:` 和 `,` 后不输出空格。
    assert safe_json_stringify({"a": 1}) == '{"a":1}'
    assert safe_json_stringify({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'


def test_safe_json_stringify_falls_back_for_unserializable_values():
    assert "object" in safe_json_stringify(object())


# ---------------------------------------------------------------------------------
# 用量估算（estimation）
# ---------------------------------------------------------------------------------


def test_calculate_context_tokens_prefers_total_tokens():
    assert calculate_context_tokens(Usage(total_tokens=100, input=1)) == 100
    assert calculate_context_tokens(Usage(input=1, output=2, cache_read=3, cache_write=4)) == 10


def test_estimate_text_tokens_rounds_up_at_four_characters_per_token():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_estimate_content_tokens_counts_text_blocks():
    assert estimate_content_tokens("abcd") == 1
    assert estimate_content_tokens([TextContent(text="abcd")]) == 1


def test_estimate_message_tokens_by_role():
    from pi_ai.types import ThinkingContent

    assert estimate_message_tokens(UserMessage(content="abcd", timestamp=0)) == 1
    assert estimate_message_tokens(SystemMessage(content="abcd", timestamp=0)) >= 1
    assistant = AssistantMessage(
        content=[TextContent(text="abcd"), ThinkingContent(thinking="efgh")],
        stop_reason=StopReason.STOP,
        timestamp=0,
    )
    assert estimate_message_tokens(assistant) == 2


def test_estimate_message_tokens_includes_tool_declarations():
    without_tools = estimate_message_tokens(SystemMessage(content="x", timestamp=0))
    with_tools = estimate_message_tokens(
        SystemMessage(content="x", timestamp=0, tools_added=[Tool(name="t", description="d", parameters={})])
    )
    assert with_tools > without_tools


def test_estimate_context_tokens_anchors_on_the_last_usable_usage_block():
    messages = [
        UserMessage(content="abcd" * 100, timestamp=0),
        AssistantMessage(stop_reason=StopReason.STOP, usage=Usage(total_tokens=500), timestamp=1),
        UserMessage(content="abcd", timestamp=2),
    ]
    estimate = estimate_context_tokens(TranscriptContext(messages=messages))
    assert estimate.usage_tokens == 500
    assert estimate.last_usage_index == 1
    assert estimate.trailing_tokens == 1
    assert estimate.tokens == 501


def test_estimate_context_tokens_skips_failed_and_superseded_usage():
    messages = [
        AssistantMessage(stop_reason=StopReason.ERROR, usage=Usage(total_tokens=900), timestamp=0),
        AssistantMessage(stop_reason=StopReason.ABORTED, usage=Usage(total_tokens=900), timestamp=1),
        UserMessage(content="abcd", timestamp=2),
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.usage_tokens == 0
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_ignores_usage_superseded_by_a_newer_prefix_message():
    # 压缩摘要以*更晚*的时间戳插在响应之前，意味着响应的 usage
    # 已不再描述当前前缀。
    messages = [
        UserMessage(content="compaction summary", timestamp=10),
        AssistantMessage(stop_reason=StopReason.STOP, usage=Usage(total_tokens=500), timestamp=0),
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.last_usage_index is None
    assert estimate.usage_tokens == 0


def test_estimate_context_tokens_keeps_usage_appended_after_a_newer_prefix_message():
    messages = [
        AssistantMessage(stop_reason=StopReason.STOP, usage=Usage(total_tokens=500), timestamp=0),
        UserMessage(content="later turn", timestamp=10),
    ]
    assert estimate_context_tokens(messages).last_usage_index == 0


def test_estimate_context_tokens_accepts_a_plain_message_list():
    assert estimate_context_tokens([UserMessage(content="abcd", timestamp=0)]).tokens == 1


def test_estimate_message_tokens_for_tool_results():
    result = ToolResultMessage(
        tool_call_id="t", tool_name="f", content=[TextContent(text="abcd")], is_error=False, timestamp=0
    )
    assert estimate_message_tokens(result) == 1
