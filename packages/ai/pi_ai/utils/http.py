"""provider 适配器共用的 HTTP 客户端构建.

TypeScript 原版调用 ``globalThis.fetch``，并允许调用方注入替换实现。移植版把
同样的注入点落到 :mod:`httpx` 上：调用方提供的 :data:`~pi_ai.types.FetchFunction`
是一个协程 ``(httpx.Request) -> httpx.Response``，会被包装成 transport，
使每个适配器继续使用普通的 ``httpx.AsyncClient``。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from ..types import FetchFunction, ProviderEnv, ProviderHeaders

__all__ = [
    "FetchTransport",
    "build_headers",
    "create_client",
    "provider_env_value",
    "resolve_timeout",
]

#: 默认 HTTP 超时（秒），对齐各 SDK 的十分钟上限。
DEFAULT_TIMEOUT_SECONDS = 600.0


class FetchTransport(httpx.AsyncBaseTransport):
    """把调用方注入的 fetch 可调用对象适配为 :mod:`httpx` transport.

    Attributes:
        _fetch: 调用方注入的 fetch 协程，签名与 ``httpx`` 的请求处理一致。
    """

    def __init__(self, fetch: FetchFunction) -> None:
        """保存 fetch 实现。

        Args:
            fetch: 调用方注入的 fetch 协程 ``(httpx.Request) -> httpx.Response``。
        """
        self._fetch = fetch

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """把 httpx 请求转交给注入的 fetch 实现。

        Args:
            request: httpx 构造的请求。

        Returns:
            注入实现返回的响应。
        """
        return await self._fetch(request)

    async def aclose(self) -> None:
        """关闭底层 fetch 实现（若它暴露 ``aclose``）."""
        closer = getattr(self._fetch, "aclose", None)
        if callable(closer):
            await closer()


def provider_env_value(name: str, env: ProviderEnv | None = None) -> str | None:
    """解析环境变量：先查 provider 作用域的覆盖值，再回落到进程环境.

    Args:
        name: 环境变量名。
        env: provider 作用域的环境变量映射；为 ``None`` 时只查进程环境。

    Returns:
        解析到的非空值；两处都没有时返回 ``None``。
    """
    if env is not None:
        scoped = env.get(name)
        if scoped:
            return scoped
    value = os.environ.get(name)
    return value if value else None


def build_headers(
    *sources: ProviderHeaders | None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """按小写不区分大小写地合并 provider header.

    靠后的源覆盖靠前的源；值为 ``None`` 时彻底抑制该 header ——
    与 :class:`~pi_ai.types.ProviderHeaders` 文档约定的契约一致。

    Args:
        *sources: 依次参与合并的 header 源，靠后者覆盖靠前者。
        base: 排在最前面的基础 header；为 ``None`` 时忽略。

    Returns:
        合并后的 header 字典，键为小写名称。
    """
    merged: dict[str, tuple[str, str | None]] = {}
    for source in (base, *sources):
        if not source:
            continue
        for name, value in source.items():
            merged[name.lower()] = (name, value)
    return {name: value for name, value in merged.values() if value is not None}


def resolve_timeout(timeout_ms: int | None) -> httpx.Timeout:
    """按毫秒预算构造 httpx 超时配置.

    Args:
        timeout_ms: 总超时预算（毫秒）；为 ``None`` 时使用
            :data:`DEFAULT_TIMEOUT_SECONDS`。

    Returns:
        带 30 秒连接超时上限的 httpx 超时配置。
    """
    if timeout_ms is None:
        return httpx.Timeout(DEFAULT_TIMEOUT_SECONDS, connect=30.0)
    seconds = max(0.001, timeout_ms / 1000.0)
    return httpx.Timeout(seconds, connect=min(seconds, 30.0))


def create_client(
    *,
    headers: Mapping[str, str] | None = None,
    timeout_ms: int | None = None,
    fetch: FetchFunction | None = None,
    base_url: str | None = None,
    proxy: str | None = None,
    env: Mapping[str, str] | None = None,
    target_url: str | None = None,
) -> httpx.AsyncClient:
    """构建一个 httpx.AsyncClient，支持移植的请求选项.

    Args:
        headers: 每个请求的默认头部。
        timeout_ms: 单个请求的超时时间（毫秒）。
        fetch: 调用方提供的 fetch 实现，会被包装为 transport。
        base_url: 可选的基础 URL，与相对请求路径合并。
        proxy: 显式的代理 URL，优先级高于环境变量。
        env: 提供程序作用域的环境变量。当同时给出 target_url 时，代理会根据目标 URL 解析。
        target_url: 要请求的 URL，用于代理解析。

    Returns:
        配置好的 httpx.AsyncClient 实例。

    Note:
        默认超时时间为 600 秒（10 分钟），匹配 SDK 的超时上限。
    """
    transport = FetchTransport(fetch) if fetch is not None else None
    resolved_proxy = proxy
    if resolved_proxy is None and target_url is not None:
        resolved_proxy = resolve_proxy_for_target(target_url, dict(env) if env is not None else None)
    return httpx.AsyncClient(
        base_url=base_url or "",
        headers=dict(headers or {}),
        timeout=resolve_timeout(timeout_ms),
        transport=transport,
        proxy=resolved_proxy,
        follow_redirects=True,
    )


def resolve_proxy_for_target(target_url: str, env: Mapping[str, str] | None = None) -> str | None:
    """为 ``target_url`` 解析代理 URL：先查作用域 ``env``，再查进程环境.

    一个带延迟导入缓存的薄封装，避免所有 HTTP 调用点都直接导入代理模块。

    Args:
        target_url: 即将请求的目标 URL。
        env: provider 作用域的环境变量映射。

    Returns:
        解析到的代理 URL；未配置代理时返回 ``None``。
    """
    from .node_http_proxy import resolve_http_proxy_url_for_target

    return resolve_http_proxy_url_for_target(target_url, dict(env) if env is not None else None)


async def iter_response_lines(response: httpx.Response) -> AsyncIterator[str]:
    """逐行迭代响应体，去除行尾换行符.

    Args:
        response: 待读取的 httpx 响应。

    Yields:
        去除行尾换行符后的每一行文本。
    """
    async for line in response.aiter_lines():
        yield line


async def read_error_body(response: httpx.Response, *, limit: int = 4000) -> str:
    """最多读取错误响应体的 ``limit`` 个字符.

    Args:
        response: 出错的 httpx 响应。
        limit: 最多保留的字符数（关键字参数）。

    Returns:
        截断后的响应文本；读取或解码失败时返回空串。
    """
    try:
        text = (await response.aread()).decode(response.encoding or "utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 错误报告自身绝不能抛出
        return ""
    return text[:limit]


def json_body(payload: Any) -> Any:  # pragma: no cover - 仅为便捷再导出
    """原样返回载荷，作为便捷再导出占位。

    Args:
        payload: 任意载荷对象。

    Returns:
        传入的 ``payload`` 本身。
    """
    return payload
