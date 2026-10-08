"""工具 schema 辅助函数.

TypeScript 原版使用 TypeBox 构建 schema；移植版改用普通 JSON Schema 字典，
因此 :func:`string_enum` 生成的是 Google API 等 provider 要求的
``{"type": "string", "enum": [...]}`` 形式，而非 ``anyOf``/``const``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["string_enum"]


def string_enum(values: Sequence[str], description: str | None = None, default: str | None = None) -> dict[str, Any]:
    """构建字符串枚举的 JSON Schema.

    Args:
        values: 允许的字符串取值。
        description: 可选的字段说明。
        default: 可选的默认值。

    Returns:
        ``{"type": "string", "enum": [...]}`` 形式的 schema 字典。

    Examples:
        >>> string_enum(["add", "subtract"], description="The operation to perform")
        {'type': 'string', 'enum': ['add', 'subtract'], 'description': 'The operation to perform'}
    """
    schema: dict[str, Any] = {"type": "string", "enum": list(values)}
    if description:
        schema["description"] = description
    if default:
        schema["default"] = default
    return schema
