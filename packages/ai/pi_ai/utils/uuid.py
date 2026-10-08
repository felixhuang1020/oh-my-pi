"""按时间有序的 UUIDv7 生成."""

from __future__ import annotations

import secrets
import time

__all__ = ["MAX_UUID_V7_TIMESTAMP", "uuidv7"]

MAX_UUID_V7_TIMESTAMP = 0xFFFFFFFFFFFF
_MAX_SEQUENCE = (1 << 41) - 1

_last_ordinary_timestamp = -1
_sequence: int | None = None


def uuidv7(timestamp_ms: int | None = None) -> str:
    """生成按时间有序的 UUIDv7.

    显式传入的时间戳会被原样使用（供 follower id 场景）；普通调用返回的
    时间戳永不早于上一次，因此进程内 id 严格单调递增。

    Args:
        timestamp_ms: 可选的毫秒时间戳，省略时取当前时间。

    Returns:
        UUIDv7 字符串。

    Raises:
        ValueError: 时间戳越界，或生成器序列号耗尽。
    """
    global _last_ordinary_timestamp, _sequence

    requested = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    if not isinstance(requested, int) or requested < 0 or requested > MAX_UUID_V7_TIMESTAMP:
        raise ValueError(f"UUIDv7 timestamp must be an integer between 0 and {MAX_UUID_V7_TIMESTAMP}")

    if timestamp_ms is None:
        effective = max(requested, _last_ordinary_timestamp)
        _last_ordinary_timestamp = effective
    else:
        effective = requested

    raw = bytearray(secrets.token_bytes(16))
    if _sequence is None:
        _sequence = (raw[1] << 32) | (raw[2] << 24) | (raw[3] << 16) | (raw[4] << 8) | raw[5]
    else:
        if _sequence == _MAX_SEQUENCE:
            raise ValueError("UUIDv7 generator sequence exhausted")
        _sequence += 1

    for index in range(5, -1, -1):
        raw[index] = (effective >> ((5 - index) * 8)) & 0xFF
    raw[6] = 0x70 | ((_sequence >> 37) & 0x0F)
    raw[7] = (_sequence >> 29) & 0xFF
    raw[8] = 0x80 | ((_sequence >> 23) & 0x3F)
    raw[9] = (_sequence >> 15) & 0xFF
    raw[10] = (_sequence >> 7) & 0xFF
    raw[11] = ((_sequence & 0x7F) << 1) | (raw[11] & 0x01)

    hex_digits = [f"{byte:02x}" for byte in raw]
    return (
        f"{''.join(hex_digits[0:4])}-{''.join(hex_digits[4:6])}-{''.join(hex_digits[6:8])}"
        f"-{''.join(hex_digits[8:10])}-{''.join(hex_digits[10:])}"
    )
