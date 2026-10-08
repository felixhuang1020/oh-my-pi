"""内置 provider 注册表。

移植自 ``packages/ai/src/providers/all.ts``：包括对生成目录的类型化读取，
以及新构建的全部内置 provider 列表。

本项目裁剪后只保留 :data:`pi_ai.types.KnownProvider` 中列出的九家 provider。

``all_providers()`` 是本移植为 ``builtinProviders()`` 提供的 Python 名称；
两种拼写都保留，方便调用方与测试任选其一。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..models import CreateModelsOptions, ModelsImpl, ProviderImpl, create_models
from ..models_generated import MODELS
from ..types import KNOWN_PROVIDERS, Model
from .anthropic import anthropic_provider
from .catalog import PROVIDER_CATALOG_DATA_DIR
from .deepseek import deepseek_provider
from .google import google_provider
from .minimax import minimax_provider
from .minimax_cn import minimax_cn_provider
from .moonshotai import moonshotai_provider
from .moonshotai_cn import moonshotai_cn_provider
from .openai import openai_provider
from .xiaomi import xiaomi_provider

__all__ = [
    "BuiltinProvider",
    "all_providers",
    "builtin_models",
    "builtin_providers",
    "get_builtin_model",
    "get_builtin_model_data_generated_at",
    "get_builtin_models",
    "get_builtin_providers",
]

#: 生成目录中出现的 provider id。
BuiltinProvider = str


def get_builtin_model(provider: str, model_id: str) -> Model | None:
    """读取一条生成的内置 chat 模型（类型化）。

    Args:
        provider: provider id。
        model_id: 模型 id。

    Returns:
        对应的 chat 模型；provider 或模型不存在时返回 ``None``。
    """
    models = MODELS.get(provider)
    return models.get(model_id) if models is not None else None


def get_builtin_providers() -> list[str]:
    """生成目录中的全部 provider id，保持生成顺序。

    Returns:
        provider id 列表。
    """
    return list(MODELS)


def get_builtin_model_data_generated_at() -> int | None:
    """全部内置 provider 目录共用的生成时间戳，单位为 epoch 毫秒。

    Returns:
        从 ``.manifest.json`` 的 ``generatedAt`` 解析出的时间戳；文件缺失或
        字段不可解析时返回 ``None``。
    """
    try:
        with (PROVIDER_CATALOG_DATA_DIR / ".manifest.json").open("r", encoding="utf-8") as handle:
            manifest: dict[str, Any] = json.load(handle)
    except (OSError, ValueError):
        return None
    generated_at = manifest.get("generatedAt")
    if not isinstance(generated_at, str):
        return None
    try:
        return int(datetime.fromisoformat(generated_at.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def get_builtin_models(provider: str) -> list[Model]:
    """``provider`` 的全部生成 chat 模型。

    Args:
        provider: provider id。

    Returns:
        chat 模型列表；provider 不存在时为空列表。
    """
    models = MODELS.get(provider)
    return list(models.values()) if models is not None else []


#: provider id 到工厂函数的登记表。列表顺序由 :data:`pi_ai.types.KNOWN_PROVIDERS`
#: 决定，新增 provider 时两处必须同时出现，否则 :func:`builtin_providers` 直接报错。
_PROVIDER_FACTORIES: dict[str, Callable[[], ProviderImpl]] = {
    "anthropic": anthropic_provider,
    "google": google_provider,
    "openai": openai_provider,
    "deepseek": deepseek_provider,
    "minimax": minimax_provider,
    "minimax-cn": minimax_cn_provider,
    "moonshotai": moonshotai_provider,
    "moonshotai-cn": moonshotai_cn_provider,
    "xiaomi": xiaomi_provider,
}


def builtin_providers() -> list[ProviderImpl]:
    """全部内置 provider，每次调用重新构建。

    Returns:
        按 :data:`~pi_ai.types.KNOWN_PROVIDERS` 顺序新建的 :class:`ProviderImpl` 列表。
    """
    return [_PROVIDER_FACTORIES[provider_id]() for provider_id in KNOWN_PROVIDERS]


def all_providers() -> list[ProviderImpl]:
    """:func:`builtin_providers` 的移植公开名称别名。

    Returns:
        与 :func:`builtin_providers` 相同的 provider 列表。
    """
    return builtin_providers()


def builtin_models(options: CreateModelsOptions | None = None) -> ModelsImpl:
    """注册了全部内置 provider 的 ``Models`` 集合。

    Args:
        options: 透传给 :func:`create_models` 的创建选项。

    Returns:
        已注册全部内置 provider 的 ``Models`` 集合。
    """
    models = create_models(options)
    for provider in builtin_providers():
        models.set_provider(provider)
    return models
