"""基于推送的异步事件流测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_ai.types import AssistantMessage, DoneEvent, ErrorEvent, StopReason, TextDeltaEvent, Usage
from pi_ai.utils.event_stream import AssistantMessageEventStream, EventStream


def _message(stop_reason: StopReason = StopReason.STOP) -> AssistantMessage:
    return AssistantMessage(stop_reason=stop_reason, usage=Usage(), timestamp=0)


async def test_events_are_delivered_in_order():
    stream: EventStream[int, int] = EventStream(lambda event: event == -1, lambda event: event)
    stream.push(1)
    stream.push(2)
    stream.end()
    assert [event async for event in stream] == [1, 2]


async def test_consumer_waiting_before_producer_pushes():
    stream: EventStream[int, int] = EventStream(lambda event: event == -1, lambda event: event)
    received: list[int] = []

    async def consume() -> None:
        async for event in stream:
            received.append(event)

    consumer = asyncio.ensure_future(consume())
    await asyncio.sleep(0)
    stream.push(7)
    await asyncio.sleep(0)
    stream.end()
    await consumer
    assert received == [7]


async def test_push_after_end_is_ignored():
    stream: EventStream[int, int] = EventStream(lambda event: event == -1, lambda event: event)
    stream.end()
    stream.push(1)
    assert [event async for event in stream] == []


async def test_end_without_result_leaves_result_pending():
    stream: EventStream[int, int] = EventStream(lambda event: event == -1, lambda event: event)
    stream.end()
    finished = asyncio.ensure_future(asyncio.wait_for(stream.result(), timeout=0.05))
    with pytest.raises(TimeoutError):
        await finished


async def test_terminal_event_resolves_result_without_end():
    stream: EventStream[int, int] = EventStream(lambda event: event == -1, lambda event: event * 10)
    stream.push(-1)
    assert await stream.result() == -10
    assert [event async for event in stream] == [-1]


async def test_result_can_be_awaited_after_completion():
    stream: EventStream[int, int] = EventStream(lambda event: event == 0, lambda event: 42)
    stream.push(0)
    assert await stream.result() == 42
    assert await stream.result() == 42


async def test_assistant_stream_uses_done_event_as_result():
    stream = AssistantMessageEventStream()
    message = _message()
    stream.push(DoneEvent(reason=StopReason.STOP, message=message))
    assert await stream.result() is message


async def test_assistant_stream_uses_error_event_as_result():
    stream = AssistantMessageEventStream()
    message = _message(StopReason.ERROR)
    stream.push(ErrorEvent(reason=StopReason.ERROR, error=message))
    assert await stream.result() is message
    assert stream._result_ready  # noqa: SLF001 - white-box assertion of terminal settlement


async def test_partial_events_reach_the_consumer_before_termination():
    stream = AssistantMessageEventStream()
    partial = _message(StopReason.PENDING)

    async def produce() -> None:
        await asyncio.sleep(0)
        stream.push(TextDeltaEvent(content_index=0, delta="a", partial=partial))
        stream.push(TextDeltaEvent(content_index=0, delta="b", partial=partial))
        stream.push(DoneEvent(reason=StopReason.STOP, message=partial))

    producer = asyncio.ensure_future(produce())
    deltas = []
    async for event in stream:
        if event.type == "text_delta":
            deltas.append(event.delta)
    await producer
    assert deltas == ["a", "b"]


async def test_end_with_result_settles_pending_result_awaiters():
    stream: EventStream[int, str] = EventStream(lambda event: event == -1, lambda event: "from-event")
    waiter = asyncio.ensure_future(stream.result())
    await asyncio.sleep(0)
    stream.end("explicit")
    assert await waiter == "explicit"
