"""移植 ``packages/ai/test/message-types.test.ts``。

上游文件的运行时半部分：JSON 兼容的 ``ToolResultMessage.details`` 值能经受
``to_json``/``from_json``，且 ``Context`` 的消息列表会保留它们。
上游的 ``expectTypeOf`` 断言只在编译期生效，Python 中没有对应物；
见 :func:`test_rejects_non_json_detail_types`。
"""

from __future__ import annotations

from typing import Any, get_type_hints

import pytest

from pi_ai.types import Context, JsonValue, ToolResultMessage
from pi_ai.utils.serde import from_json, to_json

BASE: dict[str, Any] = {
    "role": "toolResult",
    "tool_call_id": "call-1",
    "tool_name": "read",
    "content": [],
    "is_error": False,
    "timestamp": 1,
}


# --------------------------------------------------------------------------------------
# message JSON 类型（上游 describe("message JSON types") 分组）
# --------------------------------------------------------------------------------------


def test_preserves_json_compatible_detail_types_without_requiring_index_signatures() -> None:
    object_message = ToolResultMessage(
        **BASE,
        details={"path": "a.ts", "items": ["first"], "summary": {"lines": 3}},
    )
    array_message = ToolResultMessage(**BASE, details=["a", "b"])
    readonly_array_message = ToolResultMessage(**BASE, details=["a", {"count": 1}])
    primitive_message = ToolResultMessage(**BASE, details="diagnostic")
    null_message = ToolResultMessage(**BASE, details=None)

    transcript_message = object_message
    readonly_transcript_message = readonly_array_message
    context = Context(messages=[object_message, readonly_array_message])

    assert object_message.details["summary"]["lines"] == 3
    assert array_message.details == ["a", "b"]
    assert readonly_array_message.details == ["a", {"count": 1}]
    assert primitive_message.details == "diagnostic"
    assert null_message.details is None
    assert transcript_message is object_message
    assert readonly_transcript_message is readonly_array_message
    assert context.messages == [object_message, readonly_array_message]

    # TypeScript 的 `readonly [string, { readonly count?: number }]` 在 Python 中
    # 没有对应物；tuple 序列化为相同的 JSON 数组。
    tuple_message = ToolResultMessage(**BASE, details=("a", {"count": 1}))
    assert to_json(tuple_message)["details"] == ["a", {"count": 1}]

    # 所有 JSON 兼容形态的 details 都能经由线上格式往返。
    round_trips = [
        (object_message, {"path": "a.ts", "items": ["first"], "summary": {"lines": 3}}),
        (array_message, ["a", "b"]),
        (readonly_array_message, ["a", {"count": 1}]),
        (primitive_message, "diagnostic"),
        (null_message, None),
    ]
    for message, expected in round_trips:
        restored = from_json(ToolResultMessage, to_json(message))
        assert restored.details == expected


@pytest.mark.skip(
    reason=(
        "compile-time only: upstream asserts with expectTypeOf that the generic accepts "
        "JSON-compatible details (OptionalAny, undefined[], callbacks, unknown, Date) and "
        "rejects the rest; Python has no compile-time generic constraints to exercise"
    )
)
def test_rejects_non_json_detail_types() -> None:
    # 上游测试体（类型层面）：
    #   type OptionalAny = { value?: any };
    #   expectTypeOf<ToolResultMessage<OptionalAny>>().toEqualTypeOf<never>();
    #   expectTypeOf<ToolResultMessage<undefined[]>>().toEqualTypeOf<never>();
    #   expectTypeOf<ToolResultMessage<{ callback: () => void }>>().toEqualTypeOf<never>();
    #   expectTypeOf<ToolResultMessage<{ value: unknown }>>().toEqualTypeOf<never>();
    #   expectTypeOf<ToolResultMessage<{ value: Date }>>().toEqualTypeOf<never>();
    raise AssertionError("never runs")


def test_details_annotation_is_json_value() -> None:
    # 移植版声明 `details: JsonValue` 并带 `None` 默认值（上游为 `details?: JsonValue`）。
    # `JsonValue` 运行时即 `Any`，因此 `JsonValue | None` 塌缩为 `Any`。
    assert ToolResultMessage.__annotations__["details"] == "JsonValue"
    assert get_type_hints(ToolResultMessage)["details"] is Any
    assert JsonValue is Any
