"""面向 provider 流的容错 JSON 解析.

移植自 ``packages/ai/src/utils/json-parse.ts``。模型输出经常不合规范：字符串
里的裸控制字符、本应写成 ``\\\\x`` 却写成 ``\\x`` 的转义，以及仍在流式传输中
的 JSON。
:func:`repair_json` 修复前两种；:func:`parse_streaming_json` 通过丢弃不完整的
尾巴来闭合未终止的结构。
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

__all__ = ["parse_json_with_repair", "parse_streaming_json", "repair_json"]

T = TypeVar("T")

_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
_CONTROL_ESCAPES = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_control_character(char: str) -> bool:
    """判断字符是否为 JSON 字符串中不允许出现的裸控制字符.

    Args:
        char: 单个字符。

    Returns:
        码点位于 ``0x00``–``0x1F`` 时返回 ``True``。
    """
    return 0x00 <= ord(char) <= 0x1F


def _escape_control_character(char: str) -> str:
    """把裸控制字符转为其 JSON 转义写法.

    Args:
        char: 单个控制字符。

    Returns:
        具名转义（如 ``\\n``）或 ``\\uXXXX`` 形式。
    """
    return _CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}")


def repair_json(text: str) -> str:
    """修复不规范的 JSON 字符串字面量.

    把字符串内的裸控制字符转义，并在非法转义字符前补上反斜杠。

    Args:
        text: 待修复的 JSON 文本。

    Returns:
        修复后的文本；无需修复时与原文本相同。
    """
    repaired: list[str] = []
    in_string = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\":
            next_char = text[index + 1] if index + 1 < length else None
            if next_char is None:
                repaired.append("\\\\")
                index += 1
                continue
            if next_char == "u":
                digits = text[index + 2 : index + 6]
                if len(digits) == 4 and all(digit in _HEX_DIGITS for digit in digits):
                    repaired.append(f"\\u{digits}")
                    index += 6
                    continue
            if next_char in _VALID_JSON_ESCAPES:
                repaired.append(f"\\{next_char}")
                index += 2
                continue
            repaired.append("\\\\")
            index += 1
            continue

        repaired.append(_escape_control_character(char) if _is_control_character(char) else char)
        index += 1

    return "".join(repaired)


def parse_json_with_repair(text: str) -> Any:
    """解析 ``text``；格式不合法时经 :func:`repair_json` 修复后重试一次.

    Args:
        text: 待解析的 JSON 文本。

    Returns:
        解析出的 Python 对象。

    Raises:
        ValueError: 当原文与修复后的文本都无法解析时。
    """
    try:
        return json.loads(text)
    except ValueError as original:
        repaired = repair_json(text)
        if repaired != text:
            return json.loads(repaired)
        raise original


def _is_escaped(text: str, index: int) -> bool:
    """判断 ``index`` 处的字符是否被奇数个反斜杠转义.

    Args:
        text: 待检查的文本。
        index: 字符下标。

    Returns:
        该字符处于转义状态时返回 ``True``。
    """
    backslashes = 0
    position = index - 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _close_fragment(text: str) -> str | None:
    """闭合未终止的字符串与所有未闭合括号；括号不配对时返回 ``None``.

    Args:
        text: 可能不完整的 JSON 文本。

    Returns:
        补全引号与括号后的文本；括号不配对或结果为空时返回 ``None``。
    """
    stack: list[str] = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char == "]":
            if not stack or stack[-1] != "[":
                return None
            stack.pop()
        elif char == "}":
            if not stack or stack[-1] != "{":
                return None
            stack.pop()

    closed = text + ('"' if in_string else "")
    closed = closed.rstrip().rstrip(",:").rstrip()
    if not closed:
        return None
    for opener in reversed(stack):
        closed += "]" if opener == "[" else "}"
    return closed


def _drop_last_token(text: str) -> str | None:
    """去掉末尾（可能不完整的）token，以及没有对应值的键.

    Args:
        text: 可能不完整的 JSON 文本。

    Returns:
        截短后的文本；已无内容可删时返回 ``None``。
    """
    trimmed = text.rstrip()
    if not trimmed:
        return None

    index = len(trimmed)
    while index > 0 and (trimmed[index - 1].isalnum() or trimmed[index - 1] in "-+."):
        index -= 1
    trimmed = trimmed[:index].rstrip()

    if trimmed.endswith('"'):
        closing = len(trimmed) - 1
        opening = closing - 1
        while opening >= 0:
            if trimmed[opening] == '"' and not _is_escaped(trimmed, opening):
                break
            opening -= 1
        if opening < 0:
            return None
        trimmed = trimmed[:opening].rstrip()

    trimmed = trimmed.rstrip(",:").rstrip()
    return trimmed or None


def _partial_parse(text: str) -> Any:
    """通过不断丢弃末尾 token，尽力解析不完整的 JSON.

    Args:
        text: 可能不完整的 JSON 文本。

    Returns:
        解析出的 Python 对象。

    Raises:
        ValueError: 丢弃所有候选片段后仍无法解析时。
    """
    candidate: str | None = text.rstrip()
    for _ in range(256):
        if candidate is None:
            break
        closed = _close_fragment(candidate)
        if closed is not None:
            try:
                return json.loads(closed)
            except ValueError:
                pass
        candidate = _drop_last_token(candidate)
    raise ValueError("unparseable partial JSON")


def parse_streaming_json(partial_json: str | None) -> Any:
    """解析流中可能不完整的 JSON，始终返回一个合法值.

    无法挽救任何内容时回退为空对象。

    Args:
        partial_json: 流式传输到目前为止的 JSON 片段。

    Returns:
        解析出的 Python 对象；空输入或完全无法解析时返回空字典 ``{}``。
    """
    if not partial_json or partial_json.strip() == "":
        return {}
    try:
        return parse_json_with_repair(partial_json)
    except ValueError:
        try:
            return _partial_parse(partial_json)
        except ValueError:
            try:
                return _partial_parse(repair_json(partial_json))
            except ValueError:
                return {}
