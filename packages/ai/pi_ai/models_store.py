"""按 provider id 作键的持久化模型目录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .types import AnyModel
from .utils.abort import AbortSignal
from .utils.serde import clone

__all__ = ["InMemoryModelsStore", "ModelsStore", "ModelsStoreEntry", "ModelsStoreOperationOptions"]


@dataclass
class ModelsStoreEntry:
    """已持久化的 provider 目录快照。

    Attributes:
        models: 持久化的各类型模型列表。
        last_modified: 来自远端目录 Last-Modified 头的 Unix 时间戳。
        checked_at: 上次完成远端检查的 Unix 时间戳。
        etag: 来自远端目录 ETag 头的不透明校验值，原样存储。
    """

    #: 持久化的各类型模型。
    models: list[AnyModel] = field(default_factory=list)
    #: 远端目录 Last-Modified 头对应的 Unix 时间戳。
    last_modified: int | None = field(default=None, metadata={"alias": "lastModified"})
    #: 上次完成远端检查的 Unix 时间戳。
    checked_at: int | None = field(default=None, metadata={"alias": "checkedAt"})
    #: 远端目录 ETag 头的不透明校验值，原样存储。
    etag: str | None = None


@dataclass
class ModelsStoreOperationOptions:
    """模型存储操作的取消控制。

    Attributes:
        signal: 用于取消正在进行的存储操作的信号；``None`` 表示不可取消。
    """

    signal: AbortSignal | None = None


@runtime_checkable
class ModelsStore(Protocol):
    """按 provider id 作键的持久化模型目录协议。"""

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        """读取某个 provider 的目录快照。

        Args:
            provider_id: 目标 provider id。
            options: 取消控制选项；``None`` 表示不可取消。

        Returns:
            已持久化的目录快照；不存在时为 ``None``。
        """
        ...

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None
    ) -> None:
        """写入某个 provider 的目录快照。

        Args:
            provider_id: 目标 provider id。
            entry: 待写入的目录快照。
            options: 取消控制选项；``None`` 表示不可取消。
        """
        ...

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        """删除某个 provider 的目录快照。

        Args:
            provider_id: 目标 provider id。
            options: 取消控制选项；``None`` 表示不可取消。
        """
        ...


class InMemoryModelsStore:
    """将目录保存在内存中的 :class:`ModelsStore`，读写时均做克隆。"""

    def __init__(self) -> None:
        """初始化空的 provider 目录映射。"""
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        """读取目录快照，返回其克隆副本。

        Args:
            provider_id: 目标 provider id。
            options: 取消控制选项；``None`` 表示不可取消。

        Returns:
            目录快照的克隆；不存在时为 ``None``。

        Raises:
            AbortError: 取消信号已被触发。
        """
        _throw_if_aborted(options)
        entry = self._entries.get(provider_id)
        return clone(entry) if entry is not None else None

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None
    ) -> None:
        """写入目录快照，写入前先做克隆。

        Args:
            provider_id: 目标 provider id。
            entry: 待写入的目录快照。
            options: 取消控制选项；``None`` 表示不可取消。

        Raises:
            AbortError: 取消信号已被触发。
        """
        _throw_if_aborted(options)
        self._entries[provider_id] = clone(entry)

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        """删除目录快照；条目不存在时静默返回。

        Args:
            provider_id: 目标 provider id。
            options: 取消控制选项；``None`` 表示不可取消。

        Raises:
            AbortError: 取消信号已被触发。
        """
        _throw_if_aborted(options)
        self._entries.pop(provider_id, None)


def _throw_if_aborted(options: ModelsStoreOperationOptions | None) -> None:
    """在取消信号已触发时抛出异常。

    Args:
        options: 取消控制选项；``None`` 表示无事可做。

    Raises:
        AbortError: 传入的取消信号已被触发。
    """
    if options is not None and options.signal is not None:
        options.signal.throw_if_aborted()


__all__ += ["AnyModel"]
