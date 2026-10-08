"""快速、确定性的字符串哈希."""

from __future__ import annotations

__all__ = ["short_hash"]

_MASK32 = 0xFFFFFFFF
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _imul(a: int, b: int) -> int:
    """带溢出截断的 32 位乘法，对齐 JavaScript 的 ``Math.imul``.

    Args:
        a: 第一个乘数。
        b: 第二个乘数。

    Returns:
        按 32 位有符号整数回绕后的乘积。
    """
    result = (a & _MASK32) * (b & _MASK32) & _MASK32
    return result - 0x100000000 if result >= 0x80000000 else result


def _to_int32(value: int) -> int:
    """把整数截断为 32 位有符号整数.

    Args:
        value: 任意整数。

    Returns:
        回绕到 ``[-2**31, 2**31)`` 后的结果。
    """
    value &= _MASK32
    return value - 0x100000000 if value >= 0x80000000 else value


def _to_uint32(value: int) -> int:
    """把整数截断为 32 位无符号整数.

    Args:
        value: 任意整数。

    Returns:
        保留低 32 位后的非负整数。
    """
    return value & _MASK32


def _base36(value: int) -> str:
    """把非负整数编码为 base36 字符串.

    Args:
        value: 非负整数。

    Returns:
        由 ``0-9a-z`` 组成的 base36 表示。
    """
    if value == 0:
        return "0"
    digits = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(_BASE36[remainder])
    return "".join(reversed(digits))


def short_hash(text: str) -> str:
    """把长字符串压缩为快速、确定性的哈希.

    与 ``packages/ai/src/utils/hash.ts`` 中基于 cyrb53 的实现逐字节兼容。
    JavaScript 的 ``charCodeAt`` 按 UTF-16 码元迭代，因此这里增补平面字符
    同样贡献两个码元。

    Args:
        text: 待哈希的字符串。

    Returns:
        由两段 base36 拼接而成的哈希字符串。
    """
    h1 = _to_int32(0xDEADBEEF)
    h2 = _to_int32(0x41C6CE57)
    for code in _utf16_code_units(text):
        h1 = _imul(h1 ^ code, 2654435761)
        h2 = _imul(h2 ^ code, 1597334677)
    h1 = _imul(h1 ^ (_to_uint32(h1) >> 16), 2246822507) ^ _imul(h2 ^ (_to_uint32(h2) >> 13), 3266489909)
    h2 = _imul(h2 ^ (_to_uint32(h2) >> 16), 2246822507) ^ _imul(h1 ^ (_to_uint32(h1) >> 13), 3266489909)
    return _base36(_to_uint32(h2)) + _base36(_to_uint32(h1))


def _utf16_code_units(text: str) -> list[int]:
    """把字符串展开为 UTF-16 码元序列.

    Args:
        text: 任意字符串。

    Returns:
        按 JavaScript ``charCodeAt`` 顺序排列的 UTF-16 码元数值列表。
    """
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    return [encoded[index] | (encoded[index + 1] << 8) for index in range(0, len(encoded), 2)]
