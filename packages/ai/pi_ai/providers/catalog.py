"""生成的 provider 目录加载器。

TypeScript 原版为每个 provider 输出一个 ``<provider>.models.ts`` 垫片，
各自展平自己的 ``data/<provider>.json``。本项目裁剪后只保留九家 provider，
这些几乎一模一样的生成模块在此收敛为单一的带缓存加载器：JSON 保持原样，
展平按需进行。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..model_catalog import ModelGroups, flatten_chat_model_catalog
from ..types import Model

__all__ = [
    "PROVIDER_CATALOG_DATA_DIR",
    "available_provider_catalogs",
    "chat_model_catalog",
    "load_provider_data",
    "provider_models",
]

#: 存放 ``src/providers/data`` 原样副本的目录。
PROVIDER_CATALOG_DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def available_provider_catalogs() -> tuple[str, ...]:
    """附带生成目录的全部 provider id，排序后返回。

    跳过隐藏文件（``data/.manifest.json``）：与 shell 不同，``Path.glob``
    会匹配以点开头的文件，而 manifest 是簿记文件，并非 provider 目录。
    """
    return tuple(
        sorted(
            path.stem
            for path in PROVIDER_CATALOG_DATA_DIR.glob("*.json")
            if not path.name.startswith(".")
        )
    )


@lru_cache(maxsize=None)
def load_provider_data(provider_id: str) -> ModelGroups:
    """读取并缓存 ``data/<provider_id>.json``。

    若该 provider 没有生成目录则返回空映射；对动态与仅测试用的 provider 来说
    这是常态。

    Args:
        provider_id: provider 标识符，对应 ``data/<provider_id>.json`` 的文件名。

    Returns:
        该 provider 按模型类型分组的原始目录；文件不存在时返回空映射。
    """
    path = PROVIDER_CATALOG_DATA_DIR / f"{provider_id}.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


@lru_cache(maxsize=None)
def chat_model_catalog(provider_id: str) -> dict[str, Model]:
    """``provider_id`` 的全部 chat 模型，以模型 id 为键。

    Args:
        provider_id: provider 标识符。

    Returns:
        模型 id 到 :class:`~pi_ai.types.Model` 的映射。
    """
    return flatten_chat_model_catalog(provider_id, load_provider_data(provider_id))


def provider_models(provider_id: str) -> list[Model]:
    """``provider_id`` 的 chat 目录条目。

    返回全新的副本，调用方可以随意裁剪或修改列表。

    Args:
        provider_id: provider 标识符。

    Returns:
        该 provider 的全部 chat 模型。
    """
    return list(chat_model_catalog(provider_id).values())
