"""Shared harness for the ported ``openai-*`` provider tests.

The upstream TypeScript tests mock the ``openai`` npm SDK and hand it parsed chunk objects.
The Python adapters talk HTTP directly through :mod:`httpx`, so the faithful seam is
``options.fetch``: a coroutine ``(httpx.Request) -> httpx.Response``.  These helpers build
canned SSE responses and record the requests the adapter actually emitted, which keeps the
ported assertions on the wire payload rather than on an SDK mock.

Nothing here touches the network or the clock.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import httpx

from pi_ai.types import (
    AssistantContent,
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolResultMessage,
    TranscriptContext,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

__all__ = [
    "FakeFetch",
    "collect_events",
    "dumps",
    "json_response",
    "message_text",
    "plain_context",
    "responses_context",
    "responses_model",
    "sse_body",
    "sse_response",
    "sse_stream_response",
    "stream_result",
    "text_of",
    "user_message",
]

#: Compact separators matching JavaScript's ``JSON.stringify`` output.
_JSON_SEPARATORS = (",", ":")


def dumps(value: Any) -> str:
    """Serialize like ``JSON.stringify``: compact and without ASCII escaping."""
    return json.dumps(value, ensure_ascii=False, separators=_JSON_SEPARATORS, default=str)


# --------------------------------------------------------------------------------------
# SSE bodies and responses
# --------------------------------------------------------------------------------------


def sse_body(chunks: Iterable[Any], *, done: bool = True) -> bytes:
    """Encode ``chunks`` as a provider SSE body.

    Each chunk is either a payload mapping (written as a bare ``data:`` line) or an
    ``(event_name, payload)`` pair, which also emits the matching ``event:`` line.  Unless
    ``done`` is false, a trailing ``data: [DONE]`` frame is appended, as both OpenAI APIs
    send.
    """
    parts: list[str] = []
    for chunk in chunks:
        event: str | None = None
        data = chunk
        if isinstance(chunk, tuple):
            event, data = chunk
        if event is not None:
            parts.append(f"event: {event}\n")
        payload = data if isinstance(data, str) else dumps(data)
        parts.append(f"data: {payload}\n\n")
    if done:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


def sse_response(
    chunks: Iterable[Any],
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
    done: bool = True,
    request: httpx.Request | None = None,
) -> httpx.Response:
    """Build a streaming :class:`httpx.Response` carrying an SSE body."""
    return httpx.Response(
        status_code=status,
        headers=headers,
        content=sse_body(chunks, done=done),
        request=request,
    )


class _ChunkedAsyncStream(httpx.AsyncByteStream):
    """An SSE body that yields control between frames.

    A buffered :class:`httpx.Response` lets the adapter's producer coroutine run to
    completion before a consumer resumes, so events that alias the live assistant message
    already show the final state.  Real HTTP streams suspend between frames; this body
    reproduces that interleaving under test.
    """

    def __init__(self, frames: Sequence[bytes]) -> None:
        self._frames = list(frames)

    async def __aiter__(self):
        for frame in self._frames:
            await asyncio.sleep(0)
            yield frame


def sse_stream_response(
    chunks: Iterable[Any],
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
    done: bool = True,
) -> httpx.Response:
    """An SSE response whose frames are delivered one event-loop turn at a time."""
    body = sse_body(chunks, done=done)
    frames = [frame + b"\n\n" for frame in body.split(b"\n\n") if frame]
    return httpx.Response(
        status_code=status,
        headers=headers,
        stream=_ChunkedAsyncStream(frames),
    )


def json_response(
    payload: Any,
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
    request: httpx.Request | None = None,
) -> httpx.Response:
    """Build a non-streaming JSON response, used for provider error bodies."""
    return httpx.Response(
        status_code=status,
        headers={"content-type": "application/json", **dict(headers or {})},
        content=dumps(payload).encode("utf-8"),
        request=request,
    )


# --------------------------------------------------------------------------------------
# Fake fetch
# --------------------------------------------------------------------------------------


class FakeFetch:
    """A recording ``options.fetch`` implementation.

    ``responses`` may be :class:`httpx.Response` objects, exceptions to raise, or callables
    ``(request) -> response``.  When ``handler`` is given it is called as
    ``handler(request, call_index)`` for every request instead of consuming the queue; it
    may also raise.  Every request is appended to :attr:`requests`, and :attr:`payloads`
    exposes the decoded JSON bodies for assertions.
    """

    def __init__(
        self,
        *responses: Any,
        handler: Callable[[httpx.Request, int], Any] | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._queue: list[Any] = list(responses)
        self._handler = handler

    def queue(self, *responses: Any) -> FakeFetch:
        """Append more scripted responses."""
        self._queue.extend(responses)
        return self

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def payloads(self) -> list[Any]:
        """Decoded JSON request bodies, in call order."""
        return [json.loads(request.content) for request in self.requests]

    @property
    def last_payload(self) -> Any:
        return self.payloads[-1]

    @property
    def last_request(self) -> httpx.Request:
        return self.requests[-1]

    def __len__(self) -> int:
        return len(self.requests)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        if self._handler is not None:
            result = self._handler(request, index)
        elif self._queue:
            result = self._queue.pop(0)
        else:
            raise AssertionError(f"unexpected fetch call #{index + 1} to {request.url}")
        if callable(result):
            result = result(request)
        if isinstance(result, BaseException):
            raise result
        # The transport owns the response; bind the request so httpx helpers
        # (``response.request``, ``raise_for_status``, ``url``) work downstream.
        result.request = request
        return result


# --------------------------------------------------------------------------------------
# Models and contexts
# --------------------------------------------------------------------------------------


def _model(api: str, **overrides: Any) -> Model:
    base: dict[str, Any] = {
        "id": "test-model",
        "name": "Test Model",
        "api": api,
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        "context_window": 128_000,
        "max_tokens": 4096,
    }
    base.update(overrides)
    return Model(**base)


def responses_model(**overrides: Any) -> Model:
    """An ``openai-responses`` model; ``overrides`` win over the defaults."""
    return _model("openai-responses", **overrides)


def _context(
    messages: Sequence[Any] | None = None,
    *,
    system_prompt: str | None = None,
    tools: Sequence[Tool] | None = None,
) -> TranscriptContext:
    return normalize_context(plain_context(messages, system_prompt=system_prompt, tools=tools))


def plain_context(
    messages: Sequence[Any] | None = None,
    *,
    system_prompt: str | None = None,
    tools: Sequence[Tool] | None = None,
) -> Context:
    """An un-normalized :class:`Context`, for the ``pi_ai.compat`` entry points.

    ``pi_ai.compat.complete``/``complete_simple`` call ``normalize_context`` themselves, so
    passing an already-normalized :class:`TranscriptContext` there would fail.
    """
    return Context(
        messages=list(messages or []),
        system_prompt=system_prompt,
        tools=list(tools) if tools else None,
    )


def responses_context(
    messages: Sequence[Any] | None = None,
    *,
    system_prompt: str | None = None,
    tools: Sequence[Tool] | None = None,
) -> TranscriptContext:
    """A normalized transcript context for ``openai-responses``."""
    return _context(messages, system_prompt=system_prompt, tools=tools)


def user_message(text: str = "hi", *, timestamp: int = 0) -> UserMessage:
    """A plain user turn carrying ``text``."""
    return UserMessage(content=text, timestamp=timestamp)


def tool_result_message(
    tool_call_id: str,
    content: str | list[Any],
    *,
    tool_name: str = "tool",
    is_error: bool = False,
    timestamp: int = 0,
) -> ToolResultMessage:
    """A tool-result turn; ``content`` may be text, blocks or a mix."""
    blocks = content if isinstance(content, list) else [TextContent(text=content)]
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=blocks,
        is_error=is_error,
        timestamp=timestamp,
    )


# --------------------------------------------------------------------------------------
# Stream consumption
# --------------------------------------------------------------------------------------


async def collect_events(stream: Any) -> list[Any]:
    """Drain a stream's events (which also settles its result)."""
    return [event async for event in stream]


async def stream_result(stream: Any) -> AssistantMessage:
    """Await a stream's terminal :class:`AssistantMessage` without draining events."""
    return await stream.result()


def message_text(message: AssistantMessage) -> str:
    """Concatenate every text block of an assistant message."""
    return "".join(
        getattr(block, "text", "") for block in message.content if getattr(block, "type", None) == "text"
    )


def text_of(blocks: Sequence[AssistantContent] | None) -> str:
    """Concatenate the text of any content-block sequence."""
    return "".join(getattr(block, "text", "") for block in blocks or [] if getattr(block, "type", None) == "text")


# Re-exported for convenience in tests that build their own blocks.
__all__ += ["TextContent", "Tool"]
