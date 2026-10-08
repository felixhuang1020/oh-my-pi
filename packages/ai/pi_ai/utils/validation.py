"""工具调用参数的校验与强制转换（coercion）.

移植自 ``packages/ai/src/utils/validation.ts``。模型经常给出“差不多对”的
参数 —— 缺省可选项写成 ``null``、数字写成 ``"3"``、布尔写成 ``1`` —— 因此
校验先做归一化与类型强制转换，仍不匹配时再报告精确的 JSON-Schema 错误。

TypeBox 的 ``Value.Convert`` 由 :func:`coerce_with_json_schema` 替代，
对普通 JSON Schema 施加同样的定向强制转换。
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ..types import Tool, ToolCall

__all__ = [
    "coerce_with_json_schema",
    "normalize_optional_nulls",
    "validate_tool_arguments",
    "validate_tool_call",
]

_VALIDATOR_CACHE: dict[int, tuple[Draft202012Validator, int | str]] = {}


def _get_validator(schema: dict[str, Any]) -> Draft202012Validator:
    """获取或创建 JSON Schema 验证器，使用稳定的内容哈希作为缓存键.

    Args:
        schema: JSON Schema 字典。

    Returns:
        对应的 Draft202012Validator 实例。

    Note:
        使用 schema 的内容哈希作为缓存键，避免使用 id() 导致的内存地址复用
        问题。Python 的垃圾回收器可能回收旧对象并复用其内存地址，导致不同
        schema 共享缓存。
    """
    key = hash(json.dumps(schema, sort_keys=True, default=str))
    cached_entry = _VALIDATOR_CACHE.get(key)
    if cached_entry is not None:
        cached_validator, _ = cached_entry
        return cached_validator
    validator = Draft202012Validator(schema)
    _VALIDATOR_CACHE[key] = (validator, key)
    return validator


def _sub_schema_check(schema: Any, value: Any) -> bool:
    """判断 ``value`` 是否满足 ``schema``；无法构造的 schema 接受一切值.

    Args:
        schema: 待校验的 JSON Schema。
        value: 待校验的值。

    Returns:
        ``value`` 通过校验时返回 ``True``。
    """
    try:
        return _get_validator(schema).is_valid(value)
    except SchemaError:
        return True


def _schema_types(schema: dict[str, Any]) -> list[str]:
    """读取 schema 声明的 ``type``，统一为字符串列表.

    Args:
        schema: JSON Schema 片段。

    Returns:
        声明的类型名列表；未声明或形式非法时返回空列表。
    """
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [item for item in declared if isinstance(item, str)]
    return []


def _matches_json_type(value: Any, type_name: str) -> bool:
    """判断 Python 值是否已经是某个 JSON Schema 类型.

    Args:
        value: 待判断的值。
        type_name: JSON Schema 类型名，例如 ``"integer"``、``"boolean"``。

    Returns:
        类型匹配时返回 ``True``；未知类型名返回 ``False``。
    """
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "null":
        return value is None
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _coerce_primitive(value: Any, type_name: str) -> Any:
    """把基本类型的值强制转换为 ``type_name`` 指定的 JSON 类型.

    转换不可行时原样返回，交由后续的 schema 校验报错。

    Args:
        value: 待转换的值。
        type_name: 目标 JSON Schema 类型名。

    Returns:
        转换后的值；无法转换时返回原始 ``value``。
    """
    if type_name == "number":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                return value
        if isinstance(value, bool):
            return 1 if value else 0
        return value
    if type_name == "integer":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip():
            try:
                parsed = float(value)
            except ValueError:
                return value
            if parsed.is_integer():
                return int(parsed)
            return value
        if isinstance(value, bool):
            return 1 if value else 0
        return value
    if type_name == "boolean":
        if value is None:
            return False
        if value == "true":
            return True
        if value == "false":
            return False
        if isinstance(value, int) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        return value
    if type_name == "string":
        if value is None:
            return ""
        if isinstance(value, (int, float, bool)):
            return str(value)
        return value
    if type_name == "null":
        if value == "" or value == 0 or value is False:
            return None
        return value
    return value


def _apply_object_coercion(value: dict[str, Any], schema: dict[str, Any]) -> None:
    """就地为 JSON 对象的各属性施加 schema 的强制转换.

    Args:
        value: 待转换的 JSON 对象（原地修改）。
        schema: 描述该对象的 JSON Schema。
    """
    properties = schema.get("properties")
    defined = set(properties.keys()) if isinstance(properties, dict) else set()
    if isinstance(properties, dict):
        for key, property_schema in properties.items():
            if key in value:
                value[key] = coerce_with_json_schema(value[key], property_schema)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key in list(value.keys()):
            if key not in defined:
                value[key] = coerce_with_json_schema(value[key], additional)


def _apply_array_coercion(value: list[Any], schema: dict[str, Any]) -> None:
    """就地为 JSON 数组的各元素施加 ``items`` 声明的强制转换.

    Args:
        value: 待转换的 JSON 数组（原地修改）。
        schema: 描述该数组的 JSON Schema。
    """
    items = schema.get("items")
    if isinstance(items, list):
        for index, item_schema in enumerate(items):
            if index < len(value) and item_schema is not None:
                value[index] = coerce_with_json_schema(value[index], item_schema)
        return
    if isinstance(items, dict):
        for index in range(len(value)):
            value[index] = coerce_with_json_schema(value[index], items)


def _coerce_with_union(value: Any, schemas: list[dict[str, Any]]) -> Any:
    """从联合 schema 中选出第一个可行的分支并对其施加转换.

    Args:
        value: 待转换的值。
        schemas: ``anyOf``/``oneOf`` 的分支 schema 列表。

    Returns:
        转换后的值；没有分支可行时返回原始 ``value``。
    """
    for schema in schemas:
        if _sub_schema_check(schema, value):
            return value
    for schema in schemas:
        import copy

        candidate = copy.deepcopy(value)
        coerced = coerce_with_json_schema(candidate, schema)
        if _sub_schema_check(schema, coerced):
            return coerced
    return value


def coerce_with_json_schema(value: Any, schema: dict[str, Any]) -> Any:
    """按 ``schema`` 施加 TypeBox ``Value.Convert`` 语义的类型强制转换.

    Args:
        value: 待转换的值。
        schema: 目标 JSON Schema；不是字典时原样返回 ``value``。

    Returns:
        转换后的值。
    """
    if not isinstance(schema, dict):
        return value
    next_value = value

    for nested in schema.get("allOf") or []:
        next_value = coerce_with_json_schema(next_value, nested)
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            next_value = _coerce_with_union(next_value, variants)

    schema_types = _schema_types(schema)
    matches_union_member = len(schema_types) > 1 and any(
        _matches_json_type(next_value, type_name) for type_name in schema_types
    )
    if schema_types and not matches_union_member:
        for type_name in schema_types:
            candidate = _coerce_primitive(next_value, type_name)
            if candidate is not next_value:
                next_value = candidate
                break

    if "object" in schema_types and isinstance(next_value, dict):
        _apply_object_coercion(next_value, schema)
    if "array" in schema_types and isinstance(next_value, list):
        _apply_array_coercion(next_value, schema)
    return next_value


def normalize_optional_nulls(value: Any, schema: dict[str, Any]) -> None:
    """原地删除可空缺、且不可为 null 的可选属性上的 ``null`` 值.

    Args:
        value: 待清理的 JSON 值（对象或数组，原地修改）。
        schema: 描述该值的 JSON Schema。
    """
    if not isinstance(schema, dict):
        return
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index, item_schema in enumerate(items):
                if index < len(value) and item_schema is not None:
                    normalize_optional_nulls(value[index], item_schema)
        elif isinstance(items, dict):
            for item in value:
                normalize_optional_nulls(item, items)
        return
    if not isinstance(value, dict):
        return

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    required = set(schema.get("required") or [])
    for key, property_schema in properties.items():
        if key not in value:
            continue
        if (
            value[key] is None
            and key not in required
            and not (isinstance(property_schema, dict) and isinstance(property_schema.get("$ref"), str))
            and not _sub_schema_check(property_schema, None)
        ):
            del value[key]
        else:
            normalize_optional_nulls(value[key], property_schema)


def _format_validation_path(error: Any) -> str:
    """把 jsonschema 校验错误格式化为可读的字段路径.

    Args:
        error: ``jsonschema`` 的 ``ValidationError``。

    Returns:
        以 ``/`` 连接的路径；``required`` 错误会补上缺失的字段名。
    """
    path = "/".join(str(part) for part in error.absolute_path)
    if error.validator == "required" and isinstance(error.validator_value, list):
        missing = [name for name in error.validator_value if name not in (error.instance or {})]
        if missing:
            return f"{path}.{missing[0]}" if path else missing[0]
    return path or "root"


def validate_tool_call(tools: list[Tool], tool_call: ToolCall) -> Any:
    """按名称查找工具，并用其 schema 校验本次调用的参数.

    Args:
        tools: 当前可用的工具列表。
        tool_call: 待校验的工具调用。

    Returns:
        校验（并强制转换）后的参数对象。

    Raises:
        ValueError: 当 ``tools`` 中不存在同名工具时。
    """
    tool = next((candidate for candidate in tools if candidate.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    """按工具的 JSON Schema 校验（并强制转换）工具调用参数.

    Args:
        tool: 提供 ``parameters`` schema 的工具。
        tool_call: 待校验的工具调用。

    Returns:
        校验（并强制转换）后的参数对象。

    Raises:
        ValueError: 消息中逐条列出所有违反 schema 的位置。
    """
    import copy

    schema = tool.parameters if isinstance(tool.parameters, dict) else {}
    args = copy.deepcopy(tool_call.arguments)
    normalize_optional_nulls(args, schema)
    validator = _get_validator(schema)

    coerced = coerce_with_json_schema(args, schema)
    if coerced is not args:
        if isinstance(args, dict) and isinstance(coerced, dict):
            args.clear()
            args.update(coerced)
        else:
            return coerced if validator.is_valid(coerced) else args

    if validator.is_valid(args):
        return args

    lines = [f"  - {_format_validation_path(error)}: {error.message}" for error in validator.iter_errors(args)]
    errors = "\n".join(lines) or "Unknown validation error"
    raise ValueError(
        f'Validation failed for tool "{tool_call.name}":\n{errors}\n\n'
        f"Received arguments:\n{json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)}"
    )
