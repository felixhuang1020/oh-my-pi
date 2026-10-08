"""遵循取消信号的 provider 请求重试.

移植自 ``packages/ai/src/utils/provider-retry.ts``。上游 OpenAI 与 Anthropic SDK
的重试计时器会忽略请求的 ``AbortSignal``，因此调用方禁用 SDK 自带重试，
改用本辅助函数包裹请求；服务端要求的延迟超过 ``max_retry_delay_ms`` 时
立即失败（默认 60 秒）。
"""

from __future__ import annotations

import asyncio
import email.utils
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .abort import AbortError, AbortSignal, sleep_cancellable

__all__ = ["DEFAULT_MAX_RETRY_DELAY_MS", "ProviderRetryOptions", "retry_provider_request"]

T = TypeVar("T")

#: 服务端要求延迟的默认上限（毫秒），超过即立即失败。
DEFAULT_MAX_RETRY_DELAY_MS = 60_000


@dataclass
class ProviderRetryOptions:
    """单次 provider 请求的重试预算与取消配置.

    Attributes:
        max_retries: 最大重试次数；为 ``None`` 时视为 0。
        max_retry_delay_ms: 允许的服务端延迟上限（毫秒）；
            为 ``None`` 时使用 :data:`DEFAULT_MAX_RETRY_DELAY_MS`。
        signal: 取消信号；中止时立即停止重试。
    """

    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    signal: AbortSignal | None = None


def _header(error: BaseException, name: str) -> str | None:
    """从异常关联的响应头中读取指定字段。

    Args:
        error: 携带 ``headers`` 属性的 provider 异常。
        name: 头部名称。

    Returns:
        头部值；没有该字段或取值不是字符串时返回 ``None``。
    """
    headers = getattr(error, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        return value if isinstance(value, str) else None
    return None


def _status(error: BaseException) -> int | None:
    """从异常中提取 HTTP 状态码。

    Args:
        error: 可能携带 ``status`` 或 ``status_code`` 的异常。

    Returns:
        状态码；缺失或不是整数时返回 ``None``。
    """
    for attribute in ("status", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_provider_error(error: Any) -> bool:
    """判断对象是否像携带状态码的 provider 错误。

    Args:
        error: 待判断的对象。

    Returns:
        是否为带 ``status``/``status_code`` 属性的异常。
    """
    return isinstance(error, BaseException) and (hasattr(error, "status") or hasattr(error, "status_code"))


def _is_retryable(error: BaseException) -> bool:
    """按 ``x-should-retry`` 头部或状态码判断错误是否值得重试。

    Args:
        error: 上一次失败的 provider 异常。

    Returns:
        是否应重试；无状态码时默认重试。
    """
    should_retry = _header(error, "x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False
    status = _status(error)
    if status is None:
        return True
    return status in (408, 409, 429) or status >= 500


def _validate_server_retry_delay(delay_ms: float, max_retry_delay_ms: int | None, message: str) -> float:
    """校验服务端要求的延迟是否在上限之内。

    Args:
        delay_ms: 服务端要求的延迟（毫秒）。
        max_retry_delay_ms: 允许的上限（毫秒）；为 ``None`` 时用默认上限。
        message: 追加到错误信息中的原始错误描述。

    Returns:
        原样返回 ``delay_ms``；超出上限时不会返回。

    Raises:
        RuntimeError: 当 ``delay_ms`` 超过正的延迟上限时。
    """
    cap = max_retry_delay_ms if max_retry_delay_ms is not None else DEFAULT_MAX_RETRY_DELAY_MS
    if cap > 0 and delay_ms > cap:
        # 对应 TypeScript 中的 ``Math.ceil(delayMs / 1000)``，这里渲染为整数。
        requested_seconds = math.ceil(delay_ms / 1000)
        cap_seconds = math.ceil(cap / 1000)
        raise RuntimeError(
            f"Server requested {requested_seconds}s retry delay (max: {cap_seconds}s). {message}"
        )
    return delay_ms


def _retry_delay_ms(error: BaseException, retry_index: int, max_retry_delay_ms: int | None) -> float:
    """根据错误信息和重试索引计算重试延迟（毫秒）.

    Args:
        error: 上一次失败的异常。
        retry_index: 当前重试次数（从 0 开始）。
        max_retry_delay_ms: 最大允许的重试延迟（毫秒）。

    Returns:
        计算得到的延迟时间（毫秒）。

    Note:
        优先使用服务端指定的 retry-after 头部，否则使用指数退避策略。
    """
    retry_after_ms = _header(error, "retry-after-ms")
    if retry_after_ms:
        try:
            return _validate_server_retry_delay(float(retry_after_ms), max_retry_delay_ms, str(error))
        except ValueError:
            pass

    retry_after = _header(error, "retry-after")
    if retry_after:
        try:
            delay_ms = float(retry_after) * 1000
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(retry_after)
                delay_ms = parsed.timestamp() * 1000 - time.time() * 1000
            except (TypeError, ValueError):
                delay_ms = float("nan")
        if delay_ms == delay_ms:  # 利用 NaN 不等于自身的特性排除非法解析结果
            return _validate_server_retry_delay(delay_ms, max_retry_delay_ms, str(error))

    exponential_ms = min(0.5 * 2**retry_index, 8) * 1000
    return exponential_ms * (1 - random.random() * 0.25)


async def _abortable_sleep(ms: float, signal: AbortSignal | None) -> None:
    """可被 ``signal`` 打断的睡眠。

    Args:
        ms: 睡眠时长（毫秒）；负数按 0 处理。
        signal: 取消信号；为 ``None`` 时退化为普通 ``asyncio.sleep``。

    Raises:
        AbortError: 当信号在睡眠前已中止，或在睡眠期间被中止时。
    """
    await sleep_cancellable(ms, signal)


async def retry_provider_request(
    request: Callable[[], Awaitable[T]],
    options: ProviderRetryOptions | None = None,
) -> T:
    """执行 provider 请求，并附带可中断的重试逻辑.

    Args:
        request: 一个返回 Awaitable 的可调用对象，每次重试都会创建一个新的请求。
        options: 重试选项，包含最大重试次数、最大延迟和取消信号。

    Returns:
        请求成功时的结果。

    Raises:
        AbortError: 当信号被中止时。
        Exception: 当重试次数耗尽或错误不可重试时，抛出原始异常。
    """
    options = options or ProviderRetryOptions()
    max_retries = options.max_retries if options.max_retries is not None else 0
    retries_remaining = max_retries

    while True:
        try:
            # 每次重试都是全新请求，因此重试计数头部始终为零。
            return await request()
        except Exception as error:  # noqa: BLE001 - 不可重试时原样重新抛出
            if options.signal is not None and options.signal.aborted:
                raise AbortError("Request aborted") from error
            if retries_remaining <= 0 or not _is_provider_error(error) or not _is_retryable(error):
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await _abortable_sleep(_retry_delay_ms(error, retry_index, options.max_retry_delay_ms), options.signal)
