"""助手消息帧编码/归约器与小型支撑工具测试。"""

from __future__ import annotations

import pytest

from pi_ai.session_resources import cleanup_session_resources, register_session_resource_cleanup
from pi_ai.types import (
    AssistantMessage,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)
from pi_ai.utils.assistant_message_frame import (
    AssistantMessageFrame,
    AssistantMessageFrameEncoder,
    reduce_assistant_message_frames,
)
from pi_ai.utils.events import (
    done_event,
    start_event,
    text_delta_event,
    text_end_event,
    text_start_event,
    thinking_delta_event,
    thinking_end_event,
    thinking_start_event,
    tool_call_delta_event,
    tool_call_end_event,
    tool_call_start_event,
)
from pi_ai.utils.node_http_proxy import UNSUPPORTED_PROXY_PROTOCOL_MESSAGE, resolve_http_proxy_url_for_target
from pi_ai.utils.serde import from_json, to_json


def _empty() -> AssistantMessage:
    return AssistantMessage(
        api="anthropic-messages", provider="anthropic", model="m", usage=Usage(), stop_reason=StopReason.PENDING, timestamp=0
    )


def _encode_stream(build) -> tuple[list, list]:
    """按适配器的方式驱动 ``build(emit)``：先改 partial，再编码。

    编码器读取实时 partial，因此事件必须在产生的同一步编码；先攒批会让
    ``text_start`` 观察到已完成的文本并丢掉所有增量。
    """
    encoder = AssistantMessageFrameEncoder()
    frames: list = []

    def emit(event) -> None:
        frame = encoder.encode(event)
        if frame is not None:
            frames.append(frame)

    build(emit)
    return frames, encoder  # type: ignore[return-value]


def _text_frames() -> list:
    partial = _empty()

    def build(emit) -> None:
        emit(start_event(partial))
        partial.content.append(TextContent(text=""))
        emit(text_start_event(0, partial))
        partial.content[0].text = "Hel"
        emit(text_delta_event(0, "Hel", partial))
        partial.content[0].text = "Hello"
        emit(text_delta_event(0, "lo", partial))
        emit(text_end_event(0, "Hello", partial))

    frames, _ = _encode_stream(build)
    return frames


def _tool_frames() -> list:
    partial = _empty()
    call = ToolCall(id="t1", name="get", arguments={})

    def build(emit) -> None:
        emit(start_event(partial))
        partial.content.append(call)
        emit(tool_call_start_event(0, partial))
        call.arguments = {"city": "Par"}
        emit(tool_call_delta_event(0, '{"city": "Par', partial))
        call.arguments = {"city": "Paris"}
        emit(tool_call_delta_event(0, 'is"}', partial))
        emit(tool_call_end_event(0, call, partial))

    frames, _ = _encode_stream(build)
    return frames


def _thinking_frames() -> list:
    partial = _empty()

    def build(emit) -> None:
        emit(start_event(partial))
        partial.content.append(ThinkingContent(thinking=""))
        emit(thinking_start_event(0, partial))
        partial.content[0].thinking = "deep"
        emit(thinking_delta_event(0, "deep", partial))
        emit(thinking_end_event(0, "deep", partial))

    frames, _ = _encode_stream(build)
    return frames


# ---------------------------------------------------------------------------------
# 帧（frames）
# ---------------------------------------------------------------------------------


def test_encoding_a_text_stream_produces_frames_and_reduces_back():
    frames = _text_frames()

    assert [frame.type for frame in frames] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
    ]
    assert [frame.delta for frame in frames if frame.type == "text_delta"] == ["Hel", "lo"]

    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content == [TextContent(text="Hello")]
    assert reduced.stop_reason == StopReason.PENDING


def test_encoding_a_tool_stream_preserves_arguments():
    frames = _tool_frames()
    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert len(reduced.content) == 1
    assert reduced.content[0].name == "get"
    assert reduced.content[0].arguments == {"city": "Paris"}


def test_encoding_a_thinking_stream_reduces_to_thinking_content():
    frames = _thinking_frames()
    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content == [ThinkingContent(thinking="deep")]


def test_start_frame_snapshots_the_partial_at_start_time():
    frames = _text_frames()
    start = frames[0]
    assert start.type == "start"
    assert start.partial.content == []


def test_encoder_ignores_terminal_events():
    partial = _empty()
    encoder = AssistantMessageFrameEncoder()
    encoder.encode(start_event(partial))
    assert encoder.encode(done_event(StopReason.STOP, partial)) is None


def test_encoder_rejects_an_event_before_start():
    encoder = AssistantMessageFrameEncoder()
    with pytest.raises(Exception):
        encoder.encode(text_delta_event(0, "x", _empty()))


def test_reduce_of_no_frames_is_none():
    assert reduce_assistant_message_frames([]) is None


def test_frames_round_trip_through_json():
    for frame in _text_frames() + _tool_frames() + _thinking_frames():
        payload = to_json(frame)
        assert isinstance(payload, dict) and "type" in payload
        restored = from_json(AssistantMessageFrame, payload)
        assert type(restored) is type(frame)
        assert to_json(restored) == payload


def test_reduce_without_the_text_end_frame_uses_the_deltas_it_saw():
    frames = [frame for frame in _text_frames() if frame.type != "text_end"]
    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content[0].text == "Hello"


# ---------------------------------------------------------------------------------
# 代理解析（proxy resolution）
# ---------------------------------------------------------------------------------


def test_proxy_is_none_when_nothing_is_configured():
    assert resolve_http_proxy_url_for_target("https://api.example.test", {}) is None


def test_proxy_uses_scoped_env_over_the_process_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://process:8080")
    assert (
        resolve_http_proxy_url_for_target("https://api.example.test", {"HTTPS_PROXY": "http://scoped:8080"})
        == "http://scoped:8080"
    )


def test_scheme_less_proxy_values_take_the_target_protocol():
    # node-http-proxy.ts 用*目标* URL 的协议作前缀拼出 `${protocol}://`。
    assert (
        resolve_http_proxy_url_for_target("https://api.example.test", {"HTTPS_PROXY": "proxy.internal:3128"})
        == "https://proxy.internal:3128"
    )
    assert (
        resolve_http_proxy_url_for_target("http://api.example.test", {"HTTP_PROXY": "proxy.internal:3128"})
        == "http://proxy.internal:3128"
    )


def test_no_proxy_matches_exact_host_suffix_and_wildcard():
    env = {"HTTPS_PROXY": "http://proxy:8080", "NO_PROXY": "example.test"}
    assert resolve_http_proxy_url_for_target("https://api.example.test", env) is None
    assert resolve_http_proxy_url_for_target("https://other.test", env) == "http://proxy:8080"

    star = {"HTTPS_PROXY": "http://proxy:8080", "NO_PROXY": "*"}
    assert resolve_http_proxy_url_for_target("https://anything.test", star) is None


def test_socks_proxies_are_rejected_with_a_clear_message():
    with pytest.raises(Exception, match=UNSUPPORTED_PROXY_PROTOCOL_MESSAGE.split(".")[0][:20]):
        resolve_http_proxy_url_for_target("https://api.example.test", {"HTTPS_PROXY": "socks5://proxy:1080"})


# ---------------------------------------------------------------------------------
# 会话资源（session resources）
# ---------------------------------------------------------------------------------


def test_session_cleanup_runs_registered_cleanups_with_the_session_id():
    seen: list[str | None] = []

    def cleanup(session_id: str | None) -> None:
        seen.append(session_id)

    # 注册表是进程级的，且与 TypeScript 一样，cleanup 不会清空它，
    # 因此每个测试都要注销自己注册的内容。
    unregister = register_session_resource_cleanup(cleanup)
    try:
        cleanup_session_resources("session-1")
    finally:
        unregister()
    assert seen == ["session-1"]


def test_unregistering_stops_future_cleanups():
    calls: list[str] = []
    unregister = register_session_resource_cleanup(lambda session_id: calls.append("ran"))
    unregister()
    cleanup_session_resources()
    assert calls == []


def test_session_cleanup_continues_after_a_failure():
    calls: list[str] = []

    def broken(session_id: str | None) -> None:
        raise RuntimeError("cleanup failed")

    unregister_broken = register_session_resource_cleanup(broken)
    unregister_ok = register_session_resource_cleanup(lambda session_id: calls.append("second"))
    try:
        with pytest.raises(ExceptionGroup) as group:
            cleanup_session_resources()
        assert len(group.value.exceptions) == 1
        assert isinstance(group.value.exceptions[0], RuntimeError)
    finally:
        unregister_broken()
        unregister_ok()
    # 即便其中一个失败，所有 cleanup 仍都执行了。
    assert calls == ["second"]


def test_cleanup_session_resources_is_a_no_op_with_no_registrations():
    # 在模块生命周期的此刻还没有任何注册。
    cleanup_session_resources()
