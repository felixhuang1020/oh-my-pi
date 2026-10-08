"""provider 侧的约束采样：严格 JSON Schema 与 grammar 工具。

移植自 ``packages/ai/src/api/constrained-sampling.ts``。各 provider 在一个严格 JSON
Schema 子集上实现约束采样；本模块负责校验工具的 schema 是否属于该子集、将其重写为
provider 接受的形态，并推导 grammar 工具的输入属性。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..types import Tool
from ..utils.serde import clone

__all__ = [
    "GrammarConstrainedSampling",
    "GrammarToolInputJsonBuffer",
    "UNSUPPORTED_STRICT_SCHEMA_KEYS",
    "UnsupportedStrictJsonSchemaError",
    "append_grammar_tool_input_json_delta",
    "create_grammar_tool_input_properties",
    "get_grammar_tool_input",
    "get_json_schema_tool_parameters",
    "make_strict_json_schema",
    "resolve_grammar_constrained_sampling",
    "resolve_json_schema_strict_sampling",
]

#: provider 严格模式会直接拒绝的关键字。
UNSUPPORTED_STRICT_SCHEMA_KEYS: tuple[str, ...] = (
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
)


class UnsupportedStrictJsonSchemaError(Exception):
    """schema 使用了 provider 严格模式无法表达的结构。"""


#: ``(key, value) -> bool``：provider 对本属允许范围的关键字的额外拒绝判定。
UnsupportedStrictSchemaKeywordCheck = Callable[[str, Any], bool]


def _is_json_schema_object(value: Any) -> bool:
    """``value`` 是否为 JSON 对象（即 Python ``dict``）。

    Args:
        value: 待判定的值。

    Returns:
        是 ``dict`` 时为 ``True``。
    """
    return isinstance(value, dict)


def _schema_types(schema: Mapping[str, Any]) -> list[str]:
    """取出 schema 中声明的 ``type``，统一为字符串列表。

    Args:
        schema: JSON Schema 片段。

    Returns:
        声明过的类型名列表；未声明或格式不符时为空列表。
    """
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [item for item in declared if isinstance(item, str)]
    return []


def _is_structured_schema(schema: Any) -> bool:
    """schema 是否描述对象或数组这类结构化类型。

    Args:
        schema: 待判定的 JSON Schema 片段。

    Returns:
        声明了 ``object``/``array``，或带 ``properties``/``items`` 时为 ``True``。
    """
    if not _is_json_schema_object(schema):
        return False
    types = _schema_types(schema)
    return (
        "object" in types
        or "array" in types
        or schema.get("properties") is not None
        or schema.get("items") is not None
    )


def _schema_allows_null(schema: Any) -> bool:
    """schema 是否显式允许 ``null``。

    Args:
        schema: JSON Schema 片段。

    Returns:
        ``type``/``const``/``enum``/``anyOf`` 任一允许 ``null`` 时为 ``True``。
    """
    if not _is_json_schema_object(schema):
        return False
    if schema.get("type") == "null":
        return True
    if isinstance(schema.get("type"), list) and "null" in schema["type"]:
        return True
    if schema.get("const", object()) is None:
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    any_of = schema.get("anyOf")
    return isinstance(any_of, list) and any(_schema_allows_null(variant) for variant in any_of)


def _make_json_schema_node_strict(
    schema: Any,
    is_unsupported_keyword: UnsupportedStrictSchemaKeywordCheck | None = None,
) -> None:
    """就地校验并重写一个 schema 节点，使其满足 provider 严格模式。

    Args:
        schema: 待处理的 JSON Schema 节点；对象属性会被就地改写。
        is_unsupported_keyword: 可选的 provider 额外拒绝判定。

    Raises:
        UnsupportedStrictJsonSchemaError: 节点使用了严格模式无法表达的结构时。
    """
    if not _is_json_schema_object(schema):
        raise UnsupportedStrictJsonSchemaError("boolean schemas are unsupported")

    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if schema.get(key) is not None:
            raise UnsupportedStrictJsonSchemaError(f"{key} schemas are unsupported")

    if is_unsupported_keyword is not None:
        for key, value in schema.items():
            if is_unsupported_keyword(key, value):
                raise UnsupportedStrictJsonSchemaError(f"{key}: {json.dumps(value)} is unsupported")

    any_of = schema.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or len(any_of) == 0:
            raise UnsupportedStrictJsonSchemaError("anyOf must contain at least one schema")
        for variant in any_of:
            if _is_structured_schema(variant):
                raise UnsupportedStrictJsonSchemaError("object and array unions are unsupported")
            _make_json_schema_node_strict(variant, is_unsupported_keyword)

    items = schema.get("items")
    if items is not None:
        if isinstance(items, list):
            raise UnsupportedStrictJsonSchemaError("tuple schemas are unsupported")
        _make_json_schema_node_strict(items, is_unsupported_keyword)

    is_object_schema = schema.get("type") == "object"
    if schema.get("properties") is not None and not is_object_schema:
        raise UnsupportedStrictJsonSchemaError("properties require type object")
    if not is_object_schema:
        return

    if schema.get("additionalProperties") is not None and schema.get("additionalProperties") is not False:
        raise UnsupportedStrictJsonSchemaError("schema-valued or true additionalProperties is unsupported")
    if schema.get("properties") is not None and not _is_json_schema_object(schema["properties"]):
        raise UnsupportedStrictJsonSchemaError("object properties must be a schema map")
    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list) or any(not isinstance(key, str) for key in required)
    ):
        raise UnsupportedStrictJsonSchemaError("object required must be a string array")

    properties: dict[str, Any] = schema.get("properties") or {}
    property_names = list(properties.keys())
    required_set = set(required if isinstance(required, list) else [])
    if any(key not in property_names for key in required_set):
        raise UnsupportedStrictJsonSchemaError("required contains an unknown property")

    for key, prop_schema in properties.items():
        _make_json_schema_node_strict(prop_schema, is_unsupported_keyword)
        # 严格模式要求每个声明的属性都必须是 required；可选属性改写为
        # ``anyOf [<schema>, null]`` 表示。
        if key not in required_set and not _schema_allows_null(prop_schema):
            properties[key] = {"anyOf": [prop_schema, {"type": "null"}]}

    schema["required"] = property_names
    schema["additionalProperties"] = False


def make_strict_json_schema(
    schema: Mapping[str, Any],
    is_unsupported_keyword: UnsupportedStrictSchemaKeywordCheck | None = None,
) -> dict[str, Any]:
    """把工具 schema 转换为 provider 约束采样要求的严格子集。

    Args:
        schema: 工具的参数 schema。
        is_unsupported_keyword: 可选的 provider 额外拒绝判定。

    Returns:
        重写后的严格 schema 副本。

    Raises:
        UnsupportedStrictJsonSchemaError: schema 无法表达为严格子集时。
    """
    if not _is_json_schema_object(schema):
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    cloned = clone(schema)
    if not _is_json_schema_object(cloned):
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    _make_json_schema_node_strict(cloned, is_unsupported_keyword)
    if cloned.get("type") != "object":
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    return cloned


def get_json_schema_tool_parameters(tool: Tool, strict: bool | None) -> dict[str, Any]:
    """``tool`` 实际要发送的 schema：``strict`` 为 true 时经过严格重写。

    Args:
        tool: 待发送的工具定义。
        strict: 是否启用严格模式。

    Returns:
        原始 schema；``strict`` 为 ``True`` 时为严格重写后的副本。
    """
    return make_strict_json_schema(tool.parameters) if strict is True else tool.parameters


@dataclass
class GrammarConstrainedSampling:
    """单个工具解析出的 grammar 采样配置。

    Attributes:
        format: grammar 格式，取 ``lark`` 或 ``regex``。
        definition: grammar 定义文本。
        input_property: 承载 grammar 输入的参数属性名。
    """

    format: str = "lark"
    definition: str = ""
    input_property: str = ""


@dataclass
class GrammarToolInputJsonBuffer:
    """跟踪 grammar 工具输入属性的增量 JSON 生成进度。

    Attributes:
        input: 目前已生成的输入字符串。
        started: 是否已写出外层 JSON 包装的开头。
        closed: 输入是否已封闭，不再接受追加。
    """

    input: str = ""
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: Mapping[str, Any], input_property: str) -> str:
    """从已完整解析的工具参数中提取 grammar 输入字符串。

    Args:
        tool_name: 工具名，用于构造错误消息。
        arguments: 已解析的工具参数。
        input_property: 承载 grammar 输入的参数属性名。

    Returns:
        该属性对应的字符串值。

    Raises:
        ValueError: 参数缺失或不是字符串时。
    """
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> str | None:
    """向 grammar 工具输入外层的 JSON 包装追加一段增量更新。

    Args:
        buffer: 当前生成进度的缓冲状态。
        input_property: 承载 grammar 输入的参数属性名。
        next_input: 本次到达的累积输入字符串。
        close: 是否已生成完毕、需要封闭 JSON。

    Returns:
        需要发出的原始 JSON delta；没有新内容需要发送时为 ``None``。

    Raises:
        ValueError: 输入在封闭后被修改、或出现非单调增长时。
    """
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not next_input.startswith(buffer.input):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')

    input_delta = next_input[len(buffer.input) :]
    if not close and not input_delta:
        return None

    delta = ""
    if not buffer.started:
        delta += "{" + json.dumps(input_property) + ':"'
        buffer.started = True
    delta += json.dumps(input_delta)[1:-1]
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    """从工具 schema 推断唯一必需的字符串参数，作为 grammar 输入属性。

    Args:
        tool: 待推断的工具定义。

    Returns:
        该工具唯一的 required 字符串属性名。

    Raises:
        ValueError: schema 不是对象、required 不是恰好一个字符串属性，或该属性不是
            ``string`` 类型时。
    """
    schema = tool.parameters
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")

    input_property = required[0]
    properties = schema.get("properties")
    if not isinstance(properties, dict) or input_property not in properties:
        raise ValueError(f"grammar constrained sampling requires a properties entry for {input_property}")
    if properties[input_property].get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {input_property} must have type string")
    return input_property


def resolve_json_schema_strict_sampling(
    tool: Tool,
    supports_strict_mode: bool,
    is_unsupported_keyword: UnsupportedStrictSchemaKeywordCheck | None = None,
) -> bool | None:
    """判断一个 JSON-schema 工具是否以严格模式发送。

    ``is_unsupported_keyword`` 允许 provider 拒绝其严格模式不接受的额外关键字，使
    "prefer" 工具回退到非严格模式而不是直接失败。

    Args:
        tool: 待判定的工具定义。
        supports_strict_mode: 目标 provider 是否支持严格模式。
        is_unsupported_keyword: 可选的 provider 额外拒绝判定。

    Returns:
        ``True`` 表示以严格模式发送；``None`` 表示回退到非严格模式。

    Raises:
        ValueError: 工具标记为 ``require`` 但严格模式不可用或无法表达其 schema 时。
    """
    config = tool.constrained_sampling
    if config is None or getattr(config, "type", None) != "json_schema":
        return None

    if supports_strict_mode:
        try:
            make_strict_json_schema(tool.parameters, is_unsupported_keyword)
            return True
        except UnsupportedStrictJsonSchemaError as error:
            if config.strict != "require":
                return None
            raise ValueError(
                f'Tool "{tool.name}" requires JSON-schema constrained sampling, but {error}.'
            ) from error
    if config.strict == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool,
    supports_openai_grammar_tools: bool,
) -> GrammarConstrainedSampling | None:
    """为 ``tool`` 解析 grammar 约束采样配置；不可用时返回 ``None``。

    Args:
        tool: 待解析的工具定义。
        supports_openai_grammar_tools: 目标 provider 是否支持 OpenAI grammar 工具。

    Returns:
        解析出的 grammar 采样配置；工具未使用 grammar 或 provider 不支持时为 ``None``。

    Raises:
        ValueError: 工具声明了 grammar 约束但未提供受支持的变体或输入属性时。
    """
    config = tool.constrained_sampling
    if config is None or getattr(config, "type", None) != "grammar":
        return None
    if not supports_openai_grammar_tools:
        return None

    variants = config.variants or {}
    lark_definition = variants.get("openai_lark")
    regex_definition = variants.get("openai_regex")
    has_lark = isinstance(lark_definition, str) and lark_definition.strip() != ""
    has_regex = isinstance(regex_definition, str) and regex_definition.strip() != ""
    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: '
            "no supported grammar variant was provided."
        )

    try:
        return GrammarConstrainedSampling(
            format="lark" if has_lark else "regex",
            definition=lark_definition if has_lark else (regex_definition or ""),
            input_property=_infer_grammar_input_property(tool),
        )
    except ValueError as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.') from error


def create_grammar_tool_input_properties(
    tools: Sequence[Tool] | None,
    supports_openai_grammar_tools: bool,
) -> dict[str, str]:
    """把每个 grammar 约束工具的名字映射到其输入属性名。

    Args:
        tools: 请求声明的工具列表；``None`` 视为空列表。
        supports_openai_grammar_tools: 目标 provider 是否支持 OpenAI grammar 工具。

    Returns:
        工具名到 grammar 输入属性名的映射。
    """
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            properties[tool.name] = grammar.input_property
    return properties
