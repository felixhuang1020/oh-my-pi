"""附加在失败或已恢复的 assistant message 上的诊断信息."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AssistantMessageDiagnostic",
    "DiagnosticErrorInfo",
    "append_assistant_message_diagnostic",
    "create_assistant_message_diagnostic",
    "extract_diagnostic_error",
    "format_thrown_value",
]


@dataclass
class DiagnosticErrorInfo:
    """抛出值的脱敏描述.

    Attributes:
        message: 抛出值的文本表示。
        name: 抛出值的名称，例如异常类名。
        stack: 可选的堆栈文本。
        code: 可选的字符串或整数错误码。
    """

    message: str
    name: str | None = None
    stack: str | None = None
    code: str | int | None = None


@dataclass
class AssistantMessageDiagnostic:
    """一条附加到 assistant message 的诊断记录.

    Attributes:
        type: 诊断类别标识。
        timestamp: 记录时刻，Unix 毫秒时间戳。
        error: 可选的抛出值描述。
        details: 可选的附加上下文数据。
    """

    type: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    error: DiagnosticErrorInfo | None = None
    details: dict[str, Any] | None = None


def format_thrown_value(value: Any) -> str:
    """把任意抛出值渲染为单行文本.

    Args:
        value: 任意抛出值。

    Returns:
        单行文本；异常取其 ``str()``，为空时退化为类型名。
    """
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: Any) -> DiagnosticErrorInfo:
    """提取错误的标识字段，避免泄露其负载内容.

    Args:
        error: 任意抛出值。

    Returns:
        仅含名称、消息、堆栈与错误码等标识字段的描述。
    """
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(name="ThrownValue", message=format_thrown_value(error))
    code = getattr(error, "code", None)
    return DiagnosticErrorInfo(
        name=type(error).__name__ or None,
        message=str(error) or type(error).__name__,
        stack=_format_stack(error),
        code=code if isinstance(code, (str, int)) and not isinstance(code, bool) else None,
    )


def _format_stack(error: BaseException) -> str | None:
    """格式化异常的堆栈文本.

    Args:
        error: 任意异常。

    Returns:
        完整的堆栈文本；异常没有 ``__traceback__`` 时返回 ``None``。
    """
    import traceback

    if error.__traceback__ is None:
        return None
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def create_assistant_message_diagnostic(
    type: str,
    error: Any,
    details: dict[str, Any] | None = None,
) -> AssistantMessageDiagnostic:
    """为 ``error`` 构建一条诊断记录，时间戳取当前时刻.

    Args:
        type: 诊断类别标识。
        error: 任意抛出值。
        details: 可选的附加上下文数据。

    Returns:
        新建的诊断记录。
    """
    return AssistantMessageDiagnostic(
        type=type,
        timestamp=int(time.time() * 1000),
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(message: Any, diagnostic: AssistantMessageDiagnostic) -> None:
    """把 ``diagnostic`` 追加到 ``message.diagnostics``.

    Args:
        message: 任意带 ``diagnostics`` 属性的 assistant message。
        diagnostic: 待追加的诊断记录。
    """
    message.diagnostics = [*(message.diagnostics or []), diagnostic]
