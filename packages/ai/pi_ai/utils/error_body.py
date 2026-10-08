"""provider HTTP 错误对象的共用归一化.

移植自 ``packages/ai/src/utils/error-body.ts``。代理或网关后面的端点可能返回
非 2xx 响应，其响应体 provider SDK 无法折入 ``error.message``；此时 SDK 错误
仍会以 SDK 特定的字段名携带状态码和原始/已解析的响应体。
:func:`normalize_provider_error` 会探测这些形态，使适配器能展示真实原因，
而不是 ``"403 status code (no body)"``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MAX_PROVIDER_ERROR_BODY_CHARS",
    "NormalizedProviderError",
    "format_provider_error",
    "normalize_provider_error",
    "safe_json_stringify",
    "truncate_error_text",
]

#: provider 错误响应体对外展示的最大长度。
MAX_PROVIDER_ERROR_BODY_CHARS = 4000


@dataclass
class NormalizedProviderError:
    """归约后的 provider 错误，只保留值得展示的字段."""

    #: ``error.message``；抛出非 ``Exception`` 值时为其序列化形式。
    message: str = ""
    #: 能提取到时的 HTTP 状态码。
    status: int | None = None
    #: 原始 HTTP 响应体原因，已修剪并截断到上限。
    body: str | None = None
    #: ``message`` 已包含响应体（无需再附加独立 body）时为 ``True``。
    message_carries_body: bool = False


def safe_json_stringify(value: Any) -> str:
    """把 ``value`` 序列化为 JSON，无法序列化时回退到 ``str``.

    采用 ``JSON.stringify`` 的紧凑分隔符，使 provider 错误响应体呈现为
    ``{"a":1}`` 而非 Python 默认的 ``{"a": 1}``。

    Args:
        value: 待序列化的值。

    Returns:
        紧凑的 JSON 文本；无法序列化时回退为 ``str(value)``。
    """
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)
    return serialized


def truncate_error_text(text: str, max_chars: int) -> str:
    """把 ``text`` 截断到 ``max_chars``，并注明丢弃了多少字符.

    Args:
        text: 待截断的文本。
        max_chars: 保留的最大字符数。

    Returns:
        原文本，或截断并附带 ``[truncated N chars]`` 标注的文本。
    """
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _member(value: Any, name: str) -> Any:
    """按对象属性或映射键读取 ``name``.

    Args:
        value: 待读取的对象或映射。
        name: 属性名或键名。

    Returns:
        对应取值；不存在时返回 ``None``。
    """
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_status(error: BaseException) -> int | None:
    """依次探测 HTTP 状态码.

    探测顺序为 ``status_code`` → ``status`` → ``$metadata`` → ``$response``。

    Args:
        error: 抛出的 provider 错误。

    Returns:
        探测到的状态码；都未命中时返回 ``None``。
    """
    for attribute in ("status_code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for container in ("metadata", "$metadata"):
        value = _member(_member(error, container), "httpStatusCode")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for container in ("response", "$response"):
        response = _member(error, container)
        if response is not None:
            for attribute in ("status_code", "statusCode"):
                value = _member(response, attribute)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
    return None


def _is_plain_non_empty_mapping(value: Any) -> bool:
    """判断值是否为非空的普通字典.

    Args:
        value: 待判断的值。

    Returns:
        是非空 ``dict`` 时返回 ``True``。
    """
    return isinstance(value, dict) and len(value) > 0


def _is_stream_like(value: Any) -> bool:
    """判断值是否像可读流（带可调用的 ``pipe``）.

    Args:
        value: 待判断的值。

    Returns:
        具有可调用 ``pipe`` 属性时返回 ``True``。
    """
    return hasattr(value, "pipe") and callable(getattr(value, "pipe", None))


def _pick_body_text(error: BaseException) -> str | None:
    """从错误的各个可能位置提取原始响应体文本.

    Args:
        error: 抛出的 provider 错误。

    Returns:
        提取到的响应体文本；没有可用响应体时返回 ``None``。
    """
    body = getattr(error, "body", None)
    if isinstance(body, str):
        return body
    inner = getattr(error, "error", None)
    if _is_plain_non_empty_mapping(inner):
        return safe_json_stringify(inner)
    response = getattr(error, "response", None) or getattr(error, "$response", None)
    response_body = getattr(response, "body", None) if response is not None else None
    if isinstance(response_body, str):
        return response_body
    if _is_stream_like(response_body):
        return None
    if _is_plain_non_empty_mapping(response_body):
        return safe_json_stringify(response_body)
    return None


def _extract_body(error: BaseException) -> str | None:
    """提取、修剪并截断错误的响应体.

    Args:
        error: 抛出的 provider 错误。

    Returns:
        处理后的响应体文本；没有可用响应体时返回 ``None``。
    """
    body_text = _pick_body_text(error)
    if body_text is None:
        return None
    trimmed = body_text.strip()
    if not trimmed:
        return None
    return truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)


def normalize_provider_error(error: Any) -> NormalizedProviderError:
    """把抛出的 provider 错误归约为与展示相关的字段.

    Args:
        error: 捕获到的任意值；非 ``BaseException`` 时按其序列化形式处理。

    Returns:
        只含消息、状态码与响应体的 :class:`NormalizedProviderError`。
    """
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error), message_carries_body=False)

    status = _extract_status(error)
    body = _extract_body(error)
    message = str(error)
    message_carries_body = body is None or body in message
    return NormalizedProviderError(
        status=status,
        body=body,
        message=message,
        message_carries_body=message_carries_body,
    )


def format_provider_error(norm: NormalizedProviderError, prefix: str | None = None) -> str:
    """拼装展示字符串：找到响应体时为 ``"<prefix> (<status>): <body>"``.

    Args:
        norm: 归约后的 provider 错误。
        prefix: 可选的前缀，例如适配器名或请求目标。

    Returns:
        面向用户展示的错误字符串。
    """
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"
