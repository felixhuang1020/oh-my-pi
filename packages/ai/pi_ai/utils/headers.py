"""HTTP header 转换辅助函数."""

from __future__ import annotations

from collections.abc import Mapping

from ..types import ProviderHeaders

__all__ = ["headers_to_record", "provider_headers_to_record"]


def headers_to_record(headers: Mapping[str, str]) -> dict[str, str]:
    """把 header 映射复制为普通字典.

    Args:
        headers: 任意 header 映射。

    Returns:
        键与值均保持原样的普通字典副本。
    """
    return {key: value for key, value in headers.items()}


def provider_headers_to_record(*header_sources: ProviderHeaders | None) -> dict[str, str] | None:
    """把多个 provider header 源合并为普通字典，并丢弃被抑制的项.

    值为 ``None`` 的项视为被抑制；按小写归一化后，越靠后的源优先级
    越高，胜出项保留其原始大小写。

    Args:
        *header_sources: 依次尝试的 header 源，``None`` 会被跳过。

    Returns:
        合并后的 header 字典；没有任何有效项时返回 ``None``。
    """
    merged: dict[str, tuple[str, str]] = {}
    for source in header_sources:
        if not source:
            continue
        for name, value in source.items():
            normalized = name.lower()
            merged.pop(normalized, None)
            if value is not None:
                merged[normalized] = (name, value)
    if not merged:
        return None
    return {name: value for name, value in merged.values()}
