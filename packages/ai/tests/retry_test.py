"""重试策略与已移植的 provider 请求重试辅助函数测试。"""

from __future__ import annotations

import pytest

from pi_ai.types import AssistantMessage, StopReason
from pi_ai.utils.abort import AbortController
from pi_ai.utils.provider_retry import (
    DEFAULT_MAX_RETRY_DELAY_MS,
    ProviderRetryOptions,
    retry_provider_request,
)
from pi_ai.utils.retry import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms,
)


def _error(message: str) -> AssistantMessage:
    return AssistantMessage(stop_reason=StopReason.ERROR, error_message=message, timestamp=0)


def _ok() -> AssistantMessage:
    return AssistantMessage(stop_reason=StopReason.STOP, timestamp=0)


# ---------------------------------------------------------------------------------
# 错误分类（classification）
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Overloaded",
        "rate limit exceeded",
        "429 Too Many Requests",
        "500 Internal Server Error",
        "upstream connect error",
        "connection refused",
        "fetch failed",
        "socket hang up",
        "request timed out",
        "stream ended without message_stop",
        "you can retry your request",
    ],
)
def test_transient_errors_are_retryable(message):
    assert is_retryable_assistant_error(_error(message)) is True


@pytest.mark.parametrize(
    "message",
    ["insufficient_quota", "out of budget", "quota exceeded", "billing hard limit reached", "GoUsageLimitError"],
)
def test_quota_and_billing_errors_are_not_retryable(message):
    assert is_retryable_assistant_error(_error(message)) is False


def test_non_error_messages_are_not_retryable():
    assert is_retryable_assistant_error(_ok()) is False
    assert is_retryable_assistant_error(AssistantMessage(stop_reason=StopReason.ERROR, timestamp=0)) is False


def test_rate_limit_text_wins_over_the_generic_too_many_tokens_pattern():
    assert is_retryable_assistant_error(_error("Rate limit reached")) is True


# ---------------------------------------------------------------------------------
# 延迟计算（delay computation）
# ---------------------------------------------------------------------------------


def test_retry_delay_is_exponential_and_capped():
    policy = RetryPolicy(base_delay_ms=1000)
    assert retry_delay_ms(policy, 1) == 1000
    assert retry_delay_ms(policy, 2) == 2000
    assert retry_delay_ms(policy, 3) == 4000


def test_retry_delay_respects_a_custom_cap():
    assert retry_delay_ms(RetryPolicy(base_delay_ms=1000, max_agent_delay_ms=1500), 3) == 1500


def test_retry_delay_default_cap_is_sixty_seconds():
    assert DEFAULT_MAX_AGENT_RETRY_DELAY_MS == 60_000
    assert retry_delay_ms(RetryPolicy(base_delay_ms=1000), 20) == DEFAULT_MAX_AGENT_RETRY_DELAY_MS


# ---------------------------------------------------------------------------------
# 重试循环（retry loop）
# ---------------------------------------------------------------------------------


async def test_disabled_policy_returns_the_first_response():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return _error("rate limit")

    result = await retry_assistant_call(produce, RetryPolicy(enabled=False, max_retries=5))
    assert calls == 1
    assert result.stop_reason == "error"


async def test_success_is_returned_without_retrying():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return _ok()

    policy = RetryPolicy(max_retries=3, base_delay_ms=0)
    assert (await retry_assistant_call(produce, policy)).stop_reason == "stop"
    assert calls == 1


async def test_transient_failure_is_retried_until_it_succeeds():
    responses = [_error("overloaded"), _error("overloaded"), _ok()]
    scheduled: list[tuple[int, int, int, str]] = []
    finished: list[tuple] = []

    async def produce():
        return responses.pop(0)

    def on_scheduled(attempt, max_attempts, delay_ms, error_message):
        scheduled.append((attempt, max_attempts, delay_ms, error_message))

    def on_finished(success, attempt, final_error=None):
        finished.append((success, attempt, final_error))

    policy = RetryPolicy(max_retries=5, base_delay_ms=0)
    result = await retry_assistant_call(
        produce, policy, None, RetryCallbacks(on_retry_scheduled=on_scheduled, on_retry_finished=on_finished)
    )
    assert result.stop_reason == "stop"
    assert [entry[0] for entry in scheduled] == [1, 2]
    assert finished == [(True, 2, None)]


async def test_non_retryable_error_fails_fast():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return _error("insufficient_quota")

    result = await retry_assistant_call(produce, RetryPolicy(max_retries=5, base_delay_ms=0))
    assert calls == 1
    assert result.error_message == "insufficient_quota"


async def test_retry_budget_is_exhausted():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return _error("overloaded")

    result = await retry_assistant_call(produce, RetryPolicy(max_retries=2, base_delay_ms=0))
    assert calls == 3
    assert result.error_message == "overloaded"


async def test_aborted_response_is_never_retried():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return AssistantMessage(stop_reason=StopReason.ABORTED, timestamp=0)

    result = await retry_assistant_call(produce, RetryPolicy(max_retries=3, base_delay_ms=0))
    assert calls == 1
    assert result.stop_reason == "aborted"


async def test_abort_during_backoff_normalizes_to_an_aborted_message():
    controller = AbortController()
    calls = 0
    finished: list[tuple] = []

    async def produce():
        nonlocal calls
        calls += 1
        return _error("overloaded")

    def on_scheduled(attempt, max_attempts, delay_ms, error_message):
        controller.abort()

    policy = RetryPolicy(max_retries=3, base_delay_ms=50)
    result = await retry_assistant_call(
        produce,
        policy,
        controller.signal,
        RetryCallbacks(on_retry_scheduled=on_scheduled, on_retry_finished=lambda *args: finished.append(args)),
    )
    assert calls == 1
    assert result.stop_reason == "aborted"
    assert result.error_message is None


async def test_retry_attempt_start_fires_after_the_backoff():
    events: list[str] = []

    async def produce():
        if not events:
            events.append("first")
            return _error("overloaded")
        events.append("retry")
        return _ok()

    policy = RetryPolicy(max_retries=1, base_delay_ms=0)
    await retry_assistant_call(produce, policy, None, RetryCallbacks(on_retry_attempt_start=lambda: events.append("start")))
    assert events == ["first", "start", "retry"]


# ---------------------------------------------------------------------------------
# provider 请求重试（provider-request retry）
# ---------------------------------------------------------------------------------


class _ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers or {}


async def test_provider_request_retries_retryable_statuses():
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _ProviderError("boom", status=503)
        return "ok"

    result = await retry_provider_request(request, ProviderRetryOptions(max_retries=3))
    assert result == "ok"
    assert attempts == 3


async def test_provider_request_does_not_retry_a_client_error():
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        raise _ProviderError("bad request", status=400)

    with pytest.raises(_ProviderError):
        await retry_provider_request(request, ProviderRetryOptions(max_retries=3))
    assert attempts == 1


async def test_provider_request_honours_the_should_retry_header():
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _ProviderError("nope", status=400, headers={"x-should-retry": "true"})
        return "ok"

    assert await retry_provider_request(request, ProviderRetryOptions(max_retries=1)) == "ok"
    assert attempts == 2


async def test_provider_request_stops_when_the_should_retry_header_forbids_it():
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        raise _ProviderError("nope", status=503, headers={"x-should-retry": "false"})

    with pytest.raises(_ProviderError):
        await retry_provider_request(request, ProviderRetryOptions(max_retries=3))
    assert attempts == 1


async def test_provider_request_fails_immediately_when_the_server_delay_exceeds_the_cap():
    async def request():
        raise _ProviderError("slow down", status=429, headers={"retry-after-ms": "120000"})

    with pytest.raises(RuntimeError, match="retry delay"):
        await retry_provider_request(request, ProviderRetryOptions(max_retries=1))
    assert DEFAULT_MAX_RETRY_DELAY_MS == 60_000


async def test_provider_request_aborts_immediately_when_the_signal_is_set():
    from pi_ai.utils.abort import AbortError

    controller = AbortController()
    controller.abort()

    async def request():
        raise _ProviderError("overloaded", status=503)

    with pytest.raises(AbortError):
        await retry_provider_request(request, ProviderRetryOptions(max_retries=3, signal=controller.signal))


async def test_provider_request_returns_immediately_without_retries():
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        raise _ProviderError("overloaded", status=503)

    with pytest.raises(_ProviderError):
        await retry_provider_request(request)
    assert attempts == 1
