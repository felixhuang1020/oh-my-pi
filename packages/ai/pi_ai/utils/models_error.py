"""pi-ai 运行时抛出的错误.

错误不会越过 :class:`~pi_ai.utils.event_stream.AssistantMessageEventStream`：
provider 与传输层故障会编码为 ``stop_reason`` 为 ``"error"`` 或 ``"aborted"``
的 ``AssistantMessage``。``ModelsError`` 仅保留给通过 ``Models`` 门面暴露的
编程错误与配置错误。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from .diagnostics import format_thrown_value

__all__ = ["ModelsError", "ModelsErrorCode"]


class ModelsErrorCode(StrEnum):
    """稳定、可供程序匹配的失败类别."""

    MODEL_SOURCE = "model_source"
    MODEL_VALIDATION = "model_validation"
    PROVIDER = "provider"
    STREAM = "stream"
    AUTH = "auth"


class ModelsError(Exception):
    """带有稳定 :attr:`code` 的失败.

    调用方只展示 ``str(error)``，因此底层原因会被折入消息文本，与
    TypeScript 原版的处理方式一致。

    Attributes:
        code: 稳定的失败类别，取值为 :class:`ModelsErrorCode` 之一。
    """

    def __init__(self, code: ModelsErrorCode | str, message: str, *, cause: Any = None) -> None:
        """构造失败对象，并把底层原因折入消息文本.

        Args:
            code: 失败类别，可以是 :class:`ModelsErrorCode`，或与其取值
                等价的字符串。
            message: 面向调用方的错误消息。
            cause: 可选的底层原因；仅在它是 :class:`BaseException` 时设置为
                ``__cause__``，其余情况只作为文本细节并入消息。
        """
        self.code = ModelsErrorCode(code) if not isinstance(code, ModelsErrorCode) else code
        super().__init__(_with_cause_detail(message, cause))
        self.__cause__ = cause if isinstance(cause, BaseException) else None


def _with_cause_detail(message: str, cause: Any) -> str:
    """把 ``cause`` 的细节追加到 ``message`` 之后.

    当 ``cause`` 为空、细节为空、或细节已出现在消息中时，原样返回
    ``message``，避免重复拼接同一段底层原因。

    Args:
        message: 原始错误消息。
        cause: 任意抛出值。

    Returns:
        补充了底层原因细节的消息文本。
    """
    if cause is None:
        return message
    detail = format_thrown_value(cause).strip()
    if not detail or detail in message:
        return message
    return f"{message}: {detail}"
