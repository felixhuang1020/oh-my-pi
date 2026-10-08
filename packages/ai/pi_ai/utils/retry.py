"""assistant 生成调用的重试策略."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from ..types import AssistantMessage
from .abort import AbortError, AbortSignal, operation_signal, sleep_cancellable
from .async_utils import maybe_await

__all__ = [
    "DEFAULT_MAX_AGENT_RETRY_DELAY_MS",
    "NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN",
    "RETRYABLE_PROVIDER_ERROR_PATTERN",
    "RetryCallbacks",
    "RetryPolicy",
    "is_retryable_assistant_error",
    "retry_assistant_call",
    "retry_delay_ms",
]

#: 以 429 形式返回的订阅/账户限额错误，属于额度耗尽而非瞬时限流。
NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN = re.compile(
    "|".join(
        (
            "GoUsageLimitError",
            "FreeUsageLimitError",
            "Monthly usage limit reached",
            "available balance",
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "billing",
            "subscription_sharing_usage_limit_exceeded",
        )
    ),
    re.IGNORECASE,
)

#: 值得重试的 provider、传输层与流式瞬时故障。
RETRYABLE_PROVIDER_ERROR_PATTERN = re.compile(
    "|".join(
        (
            "overloaded",
            "currently experiencing high demand",
            "rate.?limit",
            "too many requests",
            "429",
            "500",
            "502",
            "503",
            "504",
            "520",
            "524",
            "service.?unavailable",
            "server.?error",
            "internal.?error",
            "provider.?returned.?error",
            "exceeded request buffer limit while retrying upstream",
            "network.?error",
            "connection.?error",
            "connection.?refused",
            "connection.?lost",
            "other side closed",
            "fetch failed",
            "getaddrinfo",
            "ENOTFOUND",
            "EAI_AGAIN",
            "upstream.?connect",
            "reset before headers",
            "socket hang up",
            "socket connection was closed",
            "timed? out",
            "timeout",
            "terminated",
            "websocket.?closed",
            "websocket.?error",
            "ended without",
            "stream ended before message_stop",
            "stream ended before a terminal response event",
            "http2 request did not get a response",
            "retry delay",
            "you can retry your request",
            "try your request again",
            "please retry your request",
            "ResourceExhausted",
            "subscription_sharing_usage_unavailable",
            "subscription_sharing_user_unavailable",
        )
    ),
    re.IGNORECASE,
)

#: 单次 agent 级重试延迟的默认上限。
DEFAULT_MAX_AGENT_RETRY_DELAY_MS = 60_000


@dataclass
class RetryPolicy:
    """带指数退避的有限次重试.

    Attributes:
        enabled: 是否启用重试。
        max_retries: 最大重试次数（0 表示不重试）；首次调用不计为一次重试。
        base_delay_ms: 基础延迟（毫秒）。每次尝试的延迟为
            ``base_delay_ms * 2 ** (attempt - 1)``。
        max_agent_delay_ms: agent 级重试延迟的可选上限（毫秒），默认 60 秒。
    """

    enabled: bool = True
    #: 最大重试次数（0 表示不重试）；首次调用不计为一次重试。
    max_retries: int = 0
    #: 基础延迟（毫秒）。每次尝试的延迟为 ``base_delay_ms * 2 ** (attempt - 1)``。
    base_delay_ms: int = 0
    #: agent 级重试延迟的可选上限（毫秒），默认 60 秒。
    max_agent_delay_ms: int | None = None


def retry_delay_ms(policy: RetryPolicy, attempt: int) -> int:
    """计算第 N 次重试的退避延迟时间.

    Args:
        policy: 重试策略配置。
        attempt: 1-based 的重试次数。

    Returns:
        延迟时间（毫秒）。

    Note:
        使用指数退避策略：base_delay_ms * 2^(attempt - 1)。
        延迟时间会被限制在 max_agent_delay_ms 以内（默认为 60 秒）。
    """
    delay = policy.base_delay_ms * 2 ** max(0, attempt - 1)
    safe_delay = delay if delay < 2**53 else 2**53 - 1
    return min(safe_delay, policy.max_agent_delay_ms if policy.max_agent_delay_ms is not None else DEFAULT_MAX_AGENT_RETRY_DELAY_MS)


@dataclass
class RetryCallbacks:
    """围绕每次重试发出的回调，可以是同步或异步的.

    Attributes:
        on_retry_scheduled: 每次重试退避睡眠之前触发（attempt 从 1 开始计数）。
        on_retry_attempt_start: 退避睡眠结束、重试调用即将开始之前触发。
        on_retry_finished: 循环结束时触发一次：后续调用正常完成则为成功。
    """

    #: 每次重试退避睡眠之前触发（attempt 从 1 开始计数）。
    on_retry_scheduled: Callable[[int, int, int, str], Any] | None = None
    #: 退避睡眠结束、重试调用即将开始之前触发。
    on_retry_attempt_start: Callable[[], Any] | None = None
    #: 循环结束时触发一次：后续调用正常完成则为成功。
    on_retry_finished: Callable[..., Any] | None = None


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    signal: AbortSignal | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """执行一次 assistant 生成调用，遇到瞬时错误时进行有界重试.

    * 成功响应立即返回。中止（abort）是终态、永不重试；但若发生在已安排重试
      之后，会按“未成功”上报。退避睡眠期间的中止会被归一化为 ``aborted`` 的
      ``AssistantMessage``，调用方无需关心取消发生的时机。
    * 不可重试的错误（含配额/计费耗尽）立即返回，让确定性错误快速失败。
    * 其余情况按指数退避最多重试 ``max_retries`` 次。

    ``policy`` 为 ``None`` 或已禁用时，原样返回首次响应。

    Args:
        produce: 每次尝试调用的生成函数，需返回 :class:`AssistantMessage`。
        policy: 重试策略；为 ``None`` 或未启用时不重试。
        signal: 取消信号；退避期间中止会被归一化为 ``aborted`` 响应。
        callbacks: 重试生命周期回调；为 ``None`` 时不触发任何回调。

    Returns:
        最终的 assistant message，可能是成功响应、错误响应或中止响应。
    """
    max_attempts = policy.max_retries if policy is not None and policy.enabled else 0
    effective_signal = operation_signal(signal)

    attempt = 0
    last_retry: tuple[int, str] | None = None
    while True:
        response = await produce()

        if response.stop_reason == "aborted":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished is not None:
                await maybe_await(callbacks.on_retry_finished(False, last_retry[0]))
            return response

        if response.stop_reason != "error":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished is not None:
                await maybe_await(callbacks.on_retry_finished(True, last_retry[0]))
            return response

        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished is not None:
                await maybe_await(callbacks.on_retry_finished(False, last_retry[0], response.error_message))
            return response

        attempt += 1
        last_retry = (attempt, response.error_message or "Unknown error")
        assert policy is not None
        delay_ms = retry_delay_ms(policy, attempt)
        if callbacks is not None and callbacks.on_retry_scheduled is not None:
            await maybe_await(callbacks.on_retry_scheduled(attempt, max_attempts, delay_ms, last_retry[1]))

        # 把重试退避期间的中止归一化为与 provider 流中止相同的 AssistantMessage 形态，
        # 调用方无需关心取消发生的时机。
        try:
            await sleep_cancellable(delay_ms, effective_signal)
        except AbortError:
            if callbacks is not None and callbacks.on_retry_finished is not None:
                await maybe_await(callbacks.on_retry_finished(False, attempt, last_retry[1]))
            return replace(response, stop_reason="aborted", error_message=None)
        if callbacks is not None and callbacks.on_retry_attempt_start is not None:
            await maybe_await(callbacks.on_retry_attempt_start())


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """判断失败的 assistant message 是否像 provider 或传输层的瞬时错误.

    这里不实现重试策略：调用方应先单独处理上下文溢出，再自行施加重试预算、
    退避与上报逻辑。

    Args:
        message: 待判断的 assistant message。

    Returns:
        ``stop_reason`` 为 ``"error"`` 且错误信息命中可重试模式时为 ``True``。
    """
    if message.stop_reason != "error" or not message.error_message:
        return False
    error_message = message.error_message
    if NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.search(error_message):
        return False
    return RETRYABLE_PROVIDER_ERROR_PATTERN.search(error_message) is not None
