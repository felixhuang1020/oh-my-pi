"""生成的 provider 模型目录。

移植自 ``packages/ai/src/models.generated.ts``。TypeScript 模块为每个 provider
静态 import 一个自动生成的 ``providers/<provider>.models.ts`` shim，并暴露三个
``Record<provider, Record<modelId, model>>`` 映射。这些 shim 已合并进
:mod:`pi_ai.providers.catalog`（见 ``PORTING.md`` §10），因此三个映射以惰性
:class:`~collections.abc.Mapping` 视图的形式暴露，底层复用该缓存加载器，
而非在 import 时急切地重新展平每个 ``data/*.json``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from .providers.catalog import chat_model_catalog
from .types import KNOWN_PROVIDERS, Model

__all__ = [
    "BUILTIN_PROVIDER_IDS",
    "MODELS",
]

#: ``MODELS`` 的键，按生成顺序排列（TypeScript 中的 ``Object.keys(MODELS)``）。
#:
#: 直接派生自 :data:`pi_ai.types.KNOWN_PROVIDERS`：本项目裁剪后只保留其中的九家
#: provider，两处不再各自手写同一份 id 列表。
BUILTIN_PROVIDER_IDS: tuple[str, ...] = tuple(KNOWN_PROVIDERS)


class _CatalogMap(Mapping[str, dict[str, Any]]):
    """基于 JSON 目录的惰性只读 ``{provider: {model_id: model}}`` 视图。

    底层按需调用 :mod:`pi_ai.providers.catalog` 的缓存加载器，避免在 import 时
    重新展平全部 ``data/*.json``。

    Attributes:
        _loader: 按 provider id 加载并展平目录的回调。
        _name: 视图名称，仅用于 ``repr``。
    """

    __slots__ = ("_loader", "_name")

    def __init__(self, name: str, loader: Callable[[str], dict[str, Any]]) -> None:
        """保存视图名称与目录加载器。

        Args:
            name: 视图名称，用于 ``repr``。
            loader: 按 provider id 返回展平目录的回调。
        """
        self._name = name
        self._loader = loader

    def __getitem__(self, provider: str) -> dict[str, Any]:
        """按 provider id 惰性加载其模型目录。

        Args:
            provider: provider id。

        Returns:
            该 provider 的 ``{model_id: model}`` 目录。

        Raises:
            KeyError: 该 provider id 不在 :data:`BUILTIN_PROVIDER_IDS` 中。
        """
        if provider not in BUILTIN_PROVIDER_IDS:
            raise KeyError(provider)
        return self._loader(provider)

    def __iter__(self) -> Iterator[str]:
        """按生成顺序迭代内置 provider id。

        Returns:
            provider id 迭代器。
        """
        return iter(BUILTIN_PROVIDER_IDS)

    def __len__(self) -> int:
        """返回内置 provider 数量。

        Returns:
            provider id 的个数。
        """
        return len(BUILTIN_PROVIDER_IDS)

    def __repr__(self) -> str:
        """返回包含视图名称与 provider 列表的调试字符串。

        Returns:
            ``<名称>(<provider id 列表>)`` 形式的字符串。
        """
        return f"{self._name}({list(BUILTIN_PROVIDER_IDS)!r})"


#: 所有生成的 chat 目录，以 provider id 为键。
MODELS: Mapping[str, dict[str, Model]] = _CatalogMap("MODELS", chat_model_catalog)
