"""移植自 ``packages/ai/test/uuid.test.ts``。

覆盖 :func:`pi_ai.utils.uuid.uuidv7`：借助记忆时间戳的单调有序、每个尾部的新鲜随机性、
时间戳边界以及非法时间戳。

Vitest 用 ``vi.useFakeTimers()`` 伪造系统时钟；本移植从 ``pi_ai.utils.uuid`` 模块经
``time.time()`` 打时间戳，因此在该模块上显式 monkeypatch 时钟。随机源
（``secrets.token_bytes``）被 monkeypatch 为按计数填充字节的桩，与上游
``crypto.getRandomValues`` 桩的做法完全一致。
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from pi_ai.utils import uuid as uuid_module
from pi_ai.utils.uuid import MAX_UUID_V7_TIMESTAMP, uuidv7

UUID_V7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
TIMESTAMP = 0x0123456789AB


def parse_timestamp(uuid: str) -> int:
    """等价于 ``Number.parseInt(uuid.replaceAll("-", "").slice(0, 12), 16)``。"""
    return int(uuid.replace("-", "")[:12], 16)


@pytest.fixture(autouse=True)
def _reset_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个测试都从全新的生成器状态开始。"""
    monkeypatch.setattr(uuid_module, "_last_ordinary_timestamp", -1)
    monkeypatch.setattr(uuid_module, "_sequence", None)


def _install_clock(monkeypatch: pytest.MonkeyPatch, start_ms: int) -> dict[str, int]:
    """冻结 ``pi_ai.utils.uuid`` 的时钟；返回的字典用于设置当前时间。"""
    clock = {"ms": start_ms}
    monkeypatch.setattr(uuid_module.time, "time", lambda: clock["ms"] / 1000.0)
    return clock


def test_generates_ordered_uuid_v7s_while_preserving_follower_timestamps(monkeypatch: pytest.MonkeyPatch):
    clock = _install_clock(monkeypatch, TIMESTAMP)

    first = uuidv7()
    second = uuidv7()
    clock["ms"] = TIMESTAMP - 1
    after_rollback = uuidv7()
    clock["ms"] = TIMESTAMP + 1
    after_advance = uuidv7()
    ordinary_ids = [first, second, after_rollback, after_advance]
    follower_timestamp = TIMESTAMP - 1_000
    followers = [uuidv7(follower_timestamp), uuidv7(follower_timestamp)]

    for identifier in [*ordinary_ids, *followers]:
        assert UUID_V7_RE.match(identifier)
    assert ordinary_ids == sorted(ordinary_ids)
    assert len(set(ordinary_ids)) == len(ordinary_ids)
    assert [parse_timestamp(identifier) for identifier in ordinary_ids] == [
        TIMESTAMP,
        TIMESTAMP,
        TIMESTAMP,
        TIMESTAMP + 1,
    ]
    assert [parse_timestamp(identifier) for identifier in followers] == [
        follower_timestamp,
        follower_timestamp,
    ]
    assert len(set(followers)) == len(followers)


def test_uses_fresh_randomness_for_every_uuid_tail(monkeypatch: pytest.MonkeyPatch):
    counter = {"value": 0}

    def fake_token_bytes(size: int) -> bytes:
        counter["value"] += 1
        return bytes([counter["value"]]) * size

    monkeypatch.setattr(uuid_module.secrets, "token_bytes", fake_token_bytes)

    assert [uuidv7(TIMESTAMP)[-8:], uuidv7(TIMESTAMP)[-8:]] == ["01010101", "02020202"]


@pytest.mark.parametrize("timestamp", [0, 2**48 - 1])
def test_accepts_timestamp_boundary(timestamp: int):
    assert parse_timestamp(uuidv7(timestamp)) == timestamp


@pytest.mark.parametrize("timestamp", [-1, 2**48, 1.5, float("nan"), float("inf")])
def test_rejects_invalid_timestamp(timestamp: Any):
    with pytest.raises(ValueError):
        uuidv7(timestamp)


def test_timestamp_boundary_constant_matches_the_upstream_max():
    assert MAX_UUID_V7_TIMESTAMP == 2**48 - 1
