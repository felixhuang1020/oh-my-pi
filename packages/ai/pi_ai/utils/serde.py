"""pi 数据模型的带别名 JSON 序列化/反序列化.

TypeScript 原版把每条消息、模型和凭据都当作普通 JSON 对象。Python 移植版
改用 snake_case 字段名的 dataclass，但在字段元数据中保留原始 camelCase
JSON 名，因此读取生成的 model 目录和序列化/反序列化线上载荷都是无损的。

    >>> from dataclasses import dataclass, field
    >>> @dataclass
    ... class Usage:
    ...     input: int = 0
    ...     cache_read: int = field(default=0, metadata={"alias": "cacheRead"})
    >>> to_json(Usage(cache_read=7))
    {'input': 0, 'cacheRead': 7}

值为 ``None`` 的 dataclass 字段会被省略，对齐 ``JSON.stringify`` 丢弃
``undefined`` 的行为；而普通 ``dict`` 或 ``list`` 中的 ``None`` 会被保留，
因为那里它表示 JSON ``null``。
"""

from __future__ import annotations

import copy
import enum
import types
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

__all__ = ["content_block", "from_json", "json_default", "to_json", "type_discriminator"]

T = TypeVar("T")

_NONE_TYPE = type(None)
_UNION_ORIGINS = (Union, types.UnionType)
#: 按序检查的 dataclass 字段，用于从 union 中挑选正确的成员。
_DISCRIMINATOR_KEYS = ("type", "role")


def content_block(type_name: str):
    """把 dataclass 标记为 ``type`` 判别目标 ``type_name``.

    这个注册本质是声明性文档：:func:`from_json` 会从类自身的 ``type`` 字段默认值
    推导出同样的信息，因此装饰器是可选的。之所以仍然使用它，是为了让读者
    一眼看出哪个 Python 类对应哪个线上 ``type``。

    Args:
        type_name: 该类在线上 JSON 中对应的 ``type`` 值。

    Returns:
        用于给 dataclass 打上 ``__json_type__`` 标记的类装饰器。
    """

    def decorate(cls: type) -> type:
        """给 ``cls`` 写入 ``__json_type__`` 标记后原样返回.

        ``cls`` 为被装饰的 dataclass。

        Returns:
            同一个类对象。
        """
        cls.__json_type__ = type_name  # type: ignore[attr-defined]
        return cls

    return decorate


def type_discriminator(cls: type) -> str | None:
    """返回 ``cls`` 声明的线上 ``type`` 值（若有）.

    ``cls`` 为待查询的 dataclass 类型。

    Returns:
        ``__json_type__`` 标记值，或 ``type``/``role`` 字段的字符串默认值；
        都不存在时返回 ``None``。
    """
    explicit = getattr(cls, "__json_type__", None)
    if isinstance(explicit, str):
        return explicit
    for key in _DISCRIMINATOR_KEYS:
        default = _field_default(cls, key)
        if isinstance(default, str):
            return default
    return None


def _field_default(cls: type, name: str) -> Any:
    """读取 dataclass 字段的默认值，供判别键探测使用。

    ``cls`` 为待查询的 dataclass 类型。

    Args:
        name: 字段名。

    Returns:
        静态默认值或 ``default_factory`` 的产物；字段不存在时返回 ``None``。
    """
    for dataclass_field in fields(cls):
        if dataclass_field.name != name:
            continue
        if dataclass_field.default is not MISSING:
            return dataclass_field.default
        if dataclass_field.default_factory is not MISSING:  # type: ignore[misc]
            return dataclass_field.default_factory()  # type: ignore[misc]
        return None
    return None


def _alias(dataclass_field: Any) -> str:
    """返回字段的 JSON 别名，未声明 ``alias`` 元数据时回退为字段名。

    Args:
        dataclass_field: dataclass 的 ``Field`` 对象。

    Returns:
        线上 JSON 使用的键名。
    """
    alias = dataclass_field.metadata.get("alias")
    return alias if isinstance(alias, str) else dataclass_field.name


def _catch_all_field(cls: type) -> Any | None:
    """收集 schema 未声明的 JSON 键的字段（若存在）.

    ``cls`` 为待查询的 dataclass 类型。

    Returns:
        声明了 ``catch_all`` 元数据的字段；不存在时返回 ``None``。
    """
    for dataclass_field in fields(cls):
        if dataclass_field.metadata.get("catch_all"):
            return dataclass_field
    return None


def _unwrap_alias(tp: Any) -> Any:
    """把 PEP 695 的 ``type X = ...`` 别名解包为其底层类型.

    Args:
        tp: 任意类型对象。

    Returns:
        类型别名的底层类型；不是别名时原样返回 ``tp``。
    """
    value = getattr(tp, "__value__", None)
    return value if value is not None else tp


def to_json(value: Any) -> Any:
    """将数据模型值转换为 JSON 兼容的 Python 对象.

    Args:
        value: 任意数据模型值。

    Returns:
        转换后的 JSON 兼容对象。

    Note:
        - None 值会被省略（模仿 JSON.stringify 对 undefined 的处理）
        - dataclass 字段会按照 metadata 中的 alias 转换为驼峰命名
        - None 在 dict 或 list 中会被保留（因为这表示 JSON null）
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        catch_all = _catch_all_field(type(value))
        payload: dict[str, Any] = {}
        ordered = sorted(fields(value), key=lambda item: item.name not in _DISCRIMINATOR_KEYS)
        for dataclass_field in ordered:
            if catch_all is not None and dataclass_field.name == catch_all.name:
                continue
            field_value = getattr(value, dataclass_field.name)
            if field_value is None:
                continue
            payload[_alias(dataclass_field)] = to_json(field_value)
        if catch_all is not None:
            extras = getattr(value, catch_all.name) or {}
            for key, item in extras.items():
                payload.setdefault(key, to_json(item))
        return payload
    if isinstance(value, Mapping):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json(item) for item in value]
    to_json_method = getattr(value, "to_json", None)
    if callable(to_json_method):
        return to_json_method()
    raise TypeError(f"Cannot serialise {type(value).__name__} to JSON")


def json_default(value: Any) -> Any:
    """委托给 :func:`to_json` 的 ``json.dumps(default=...)`` 钩子.

    Args:
        value: ``json.dumps`` 无法直接序列化的对象。

    Returns:
        :func:`to_json` 转换出的 JSON 兼容对象。

    Raises:
        TypeError: 当 ``value`` 无法被 :func:`to_json` 转换时。
    """
    converted = to_json(value)
    if converted is value and not isinstance(value, (Mapping, Sequence)):
        raise TypeError(f"Cannot serialise {type(value).__name__} to JSON")
    return converted


def _hints(cls: type) -> dict[str, Any]:
    """解析类的类型注解；解析失败时退回空表.

    ``cls`` 为待解析的 dataclass 类型。

    Returns:
        字段名到类型注解的映射；无法解析时返回空字典。
    """
    try:
        return get_type_hints(cls, include_extras=False)
    except Exception:  # pragma: no cover - 对未构建完成的类做防御性处理
        return {}


def _dataclass_candidates(types_: tuple[Any, ...]) -> list[type]:
    """挑出 union 成员中可作为 dataclass 反序列化目标的类型。

    Args:
        types_: union 的成员类型元组。

    Returns:
        其中的 dataclass 类型列表，顺序保持不变。
    """
    return [tp for tp in types_ if is_dataclass(tp) and isinstance(tp, type)]


def _pick_member(types_: tuple[Any, ...], data: Mapping[str, Any]) -> Any | None:
    """按 ``type``/``role`` 判别键从 union 成员中挑选唯一匹配的 dataclass。

    Args:
        types_: union 的成员类型元组。
        data: 待反序列化的 JSON 对象。

    Returns:
        唯一匹配的 dataclass 类型；无匹配或有歧义时返回 ``None``。
    """
    candidates = _dataclass_candidates(types_)
    if not candidates:
        return None
    for key in _DISCRIMINATOR_KEYS:
        wanted = data.get(key)
        if not isinstance(wanted, str):
            continue
        matches = [cls for cls in candidates if type_discriminator(cls) == wanted]
        if len(matches) == 1:
            return matches[0]
    return None


def from_json(tp: Any, data: Any) -> Any:
    """把 JSON 兼容的 Python 对象还原为数据模型 ``tp``.

    ``tp`` 可以是 dataclass、``list[...]``/``dict[str, ...]`` 泛型、
    ``Optional[...]``/union、``StrEnum`` 或基本类型。不支持的类型
    （``Any`` 之类）原样返回。

    Args:
        tp: 目标类型。
        data: JSON 兼容的 Python 对象。

    Returns:
        还原出的数据模型值；流中缺省的 ``None`` 保持为 ``None``。
    """
    if tp is Any or tp is None:
        return data
    tp = _unwrap_alias(tp)
    if tp is Any or tp is None:
        return data

    origin = get_origin(tp)
    if origin in _UNION_ORIGINS:
        return _from_union(tp, data)
    if origin in (list, Sequence):
        args = get_args(tp)
        item_type = args[0] if args else Any
        if data is None:
            return None
        return [from_json(item_type, item) for item in data]
    if origin in (dict, Mapping):
        args = get_args(tp)
        value_type = args[1] if len(args) > 1 else Any
        if data is None:
            return None
        return {key: from_json(value_type, item) for key, item in data.items()}

    if isinstance(tp, type) and is_dataclass(tp):
        return _build(tp, data)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(data)
    if tp is bool:
        return bool(data)
    if tp is int:
        return int(data)
    if tp is float:
        return float(data)
    if tp is str:
        return data if isinstance(data, str) else str(data)
    return data


def _accepts(member: Any, data: Any) -> bool:
    """判断 union 的某个成员是否可能描述 ``data``.

    没有这层防护时，``str`` 这样的基本类型成员会把本应属于相邻
    ``list[ContentBlock]`` 成员的列表强行转成字符串。

    Args:
        member: union 的某个成员类型。
        data: 待反序列化的 JSON 值。

    Returns:
        该成员在结构上可能描述 ``data`` 时返回 ``True``。
    """
    origin = get_origin(member)
    if origin in (list, Sequence):
        return isinstance(data, (list, tuple))
    if origin in (dict, Mapping):
        return isinstance(data, Mapping)
    if member is Any:
        return True
    if member is bool:
        return isinstance(data, bool)
    if member is int:
        return isinstance(data, int) and not isinstance(data, bool)
    if member is float:
        return isinstance(data, (int, float)) and not isinstance(data, bool)
    if member is str:
        return isinstance(data, str)
    if isinstance(member, type) and is_dataclass(member):
        return isinstance(data, Mapping)
    if isinstance(member, type) and issubclass(member, enum.Enum):
        return isinstance(data, str)
    return True


def _from_union(tp: Any, data: Any) -> Any:
    """按判别键或结构兼容性从 union 中选出成员并反序列化。

    Args:
        tp: union 类型。
        data: 待反序列化的 JSON 值。

    Returns:
        还原出的成员值；全部成员都不匹配时原样返回 ``data``。
    """
    types_ = get_args(tp)
    if data is None:
        return None
    non_none = tuple(item for item in types_ if item is not _NONE_TYPE)
    if not non_none:
        return None
    if isinstance(data, Mapping):
        member = _pick_member(non_none, data)
        if member is not None:
            return _build(member, data)
    for member in non_none:
        if not _accepts(member, data):
            continue
        try:
            return from_json(member, data)
        except (TypeError, ValueError, KeyError):
            continue
    return data


def _build(cls: type, data: Mapping[str, Any]) -> Any:
    """按字段别名与类型注解，从 JSON 对象构造 dataclass 实例。

    未在字段中声明的键会收进 ``catch_all`` 字段（若声明了该字段）。

    ``cls`` 为目标 dataclass 类型。

    Args:
        data: JSON 对象。

    Returns:
        ``cls`` 的实例。

    Raises:
        TypeError: 当 ``data`` 不是映射（JSON 对象）时。
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"{cls.__name__} expects a JSON object, got {type(data).__name__}")
    hints = _hints(cls)
    catch_all = _catch_all_field(cls)
    kwargs: dict[str, Any] = {}
    consumed: set[str] = set()
    for dataclass_field in fields(cls):
        name = dataclass_field.name
        if catch_all is not None and name == catch_all.name:
            continue
        alias = _alias(dataclass_field)
        if alias not in data:
            continue
        consumed.add(alias)
        raw = data[alias]
        field_type = hints.get(name, Any)
        kwargs[name] = None if raw is None else from_json(field_type, raw)
    if catch_all is not None:
        extras = {key: value for key, value in data.items() if key not in consumed}
        if extras or catch_all.name not in kwargs:
            kwargs[catch_all.name] = extras
    return cls(**kwargs)


def clone(value: T) -> T:
    """结构化克隆，对应 JavaScript 的 ``structuredClone``.

    Args:
        value: 待克隆的任意值。

    Returns:
        深拷贝出的新对象。
    """
    return copy.deepcopy(value)
