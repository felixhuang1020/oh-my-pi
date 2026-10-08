"""生成的 provider 模型目录的展平工具。

移植自 ``packages/ai/src/model-catalog.ts``。生成的 ``data/*.json`` 文件按 API
分组、再按 model key 分组；这些辅助函数把每组展平为 chat 模型的
``{model_id: model}`` 扁平目录。

本项目只保留 chat 模型：image 与 classifier 目录已随对应 API 一并删除。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from .types import Model
from .utils.serde import from_json

__all__ = [
    "ModelGroups",
    "flatten_chat_model_catalog",
    "flatten_model_catalog",
]

#: ``{api: {model_key: model_json}}``，由生成脚本产出。
ModelGroups: TypeAlias = Mapping[str, Mapping[str, Any]]


def flatten_model_catalog(groups: ModelGroups) -> dict[str, Model]:
    """展平所有分组为 ``{model_id: model}``，仅保留 chat 条目。

    Args:
        groups: ``{api: {model_key: model_json}}`` 形式的分组目录。

    Returns:
        以 model id 为键的扁平 chat 目录。
    """
    catalog: dict[str, Model] = {}
    for models in groups.values():
        for model in models.values():
            if not isinstance(model, Mapping):
                continue
            if model.get("type", "chat") != "chat":
                continue
            model_id = model.get("id")
            if not isinstance(model_id, str):
                continue
            catalog[model_id] = from_json(Model, model)
    return catalog


def flatten_chat_model_catalog(provider: str, groups: ModelGroups) -> dict[str, Model]:
    """展平单个 provider 的 chat 模型，以 model id 为键。

    Args:
        provider: provider id，仅用于调用方语义。
        groups: 该 provider 的 ``{api: {model_key: model_json}}`` 分组目录。

    Returns:
        以 model id 为键的 chat 模型目录。
    """
    return flatten_model_catalog(groups)
