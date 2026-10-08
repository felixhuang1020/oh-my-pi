"""上下文溢出检测测试。"""

from __future__ import annotations

import pytest

from pi_ai.types import AssistantMessage, StopReason, Usage
from pi_ai.utils.overflow import (
    get_overflow_patterns,
    is_context_overflow,
    is_recoverable_length,
)


def _error(message: str) -> AssistantMessage:
    return AssistantMessage(stop_reason=StopReason.ERROR, error_message=message, timestamp=0)


@pytest.mark.parametrize(
    "message",
    [
        "prompt is too long: 213462 tokens > 200000 maximum",
        "413 request_too_large: Request exceeds the maximum size",
        "Your input exceeds the context window of this model",
        "Requested token count exceeds the model's maximum context length of 131072 tokens",
        "Input length (265330) exceeds model's maximum context length (262144).",
        "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
        "This model's maximum prompt length is 131072 but the request contains 537812 tokens",
        "Please reduce the length of the messages or completion",
        "This endpoint's maximum context length is 200000 tokens",
        "Input length 500000 exceeds the maximum allowed input length of 200000 tokens.",
        "The input (300000 tokens) is longer than the model's context length (200000 tokens).",
        "prompt token count of 40000 exceeds the limit of 32000",
        "the request exceeds the available context size, try increasing it",
        "tokens to keep from the initial prompt is greater than the context length",
        "invalid params, context window exceeds limit",
        "Your request exceeded model token limit: 200000",
        "Prompt contains 40000 tokens and is too large for model with 32000 maximum context length",
        "Prompt has 40000 tokens, but the configured context size is 32000 tokens",
        "model_context_window_exceeded",
        "prompt too long; exceeded max context length by 100 tokens",
        "Range of input length should be [1, 128000]",
        "context_length_exceeded",
        "too many tokens",
        "token limit exceeded",
        "Prompt too long",
        "Prompt exceeds max length",
        "input is too long for requested model",
    ],
)
def test_known_overflow_messages_are_detected(message):
    assert is_context_overflow(_error(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Throttling error: Too many tokens, please wait before trying again.",
        "Service unavailable: too many tokens",
        "Rate limit exceeded, try again later",
        "429 too many requests",
    ],
)
def test_non_overflow_errors_are_excluded(message):
    assert is_context_overflow(_error(message)) is False


def test_silent_overflow_when_input_usage_exceeds_the_window():
    message = AssistantMessage(
        stop_reason=StopReason.STOP,
        usage=Usage(input=150_000, cache_read=10_000),
        timestamp=0,
    )
    assert is_context_overflow(message, context_window=150_000) is True
    assert is_context_overflow(message, context_window=200_000) is False


def test_silent_overflow_requires_a_context_window_argument():
    message = AssistantMessage(stop_reason=StopReason.STOP, usage=Usage(input=10_000_000), timestamp=0)
    assert is_context_overflow(message) is False


def test_length_stop_overflow_when_there_is_no_room_left_for_output():
    message = AssistantMessage(
        stop_reason=StopReason.LENGTH,
        usage=Usage(input=99_500, output=0, cache_read=0),
        timestamp=0,
    )
    assert is_context_overflow(message, context_window=100_000) is True


def test_length_stop_with_output_is_not_overflow():
    message = AssistantMessage(
        stop_reason=StopReason.LENGTH,
        usage=Usage(input=99_500, output=100),
        timestamp=0,
    )
    assert is_context_overflow(message, context_window=100_000) is False


def test_successful_response_within_the_window_is_not_overflow():
    message = AssistantMessage(stop_reason=StopReason.STOP, usage=Usage(input=10, output=10), timestamp=0)
    assert is_context_overflow(message, context_window=100_000) is False


def test_recoverable_length_requires_room_to_retry():
    short = AssistantMessage(stop_reason=StopReason.LENGTH, usage=Usage(output=10), timestamp=0)
    assert is_recoverable_length(short, 1000) is True
    assert is_recoverable_length(short, 0) is False

    full = AssistantMessage(stop_reason=StopReason.LENGTH, usage=Usage(output=1000), timestamp=0)
    assert is_recoverable_length(full, 1000) is False

    stopped = AssistantMessage(stop_reason=StopReason.STOP, usage=Usage(output=10), timestamp=0)
    assert is_recoverable_length(stopped, 1000) is False


def test_get_overflow_patterns_returns_a_copy():
    patterns = get_overflow_patterns()
    patterns.clear()
    assert len(get_overflow_patterns()) > 0
