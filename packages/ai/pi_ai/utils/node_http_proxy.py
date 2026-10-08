"""从环境变量解析 HTTP 代理.

移植自 ``packages/ai/src/utils/node-http-proxy.ts``。TypeScript 原版存在的原因是
Node 自身的 fetch/websocket 实现会忽略 ``HTTP_PROXY``/``HTTPS_PROXY``；Python 的
``urllib``/``httpx`` 同样把代理选择留给调用方，因此 provider 适配器通过
:func:`resolve_http_proxy_url_for_target` 决定请求是否应走代理。

查找顺序与 TypeScript 一致：调用方提供的 :data:`~pi_ai.types.ProviderEnv` 覆盖值
优先于进程环境，小写键优先于大写键，``<scheme>_proxy`` 优先于 ``ALL_PROXY``。
``NO_PROXY`` 支持精确主机、后缀、``.domain``/``*.domain`` 通配符、``*``、
方括号中的 IPv6 字面量以及可选端口。SOCKS 与 PAC 代理 URL 会被明确拒绝。
"""

from __future__ import annotations

import json
import re
from urllib.parse import SplitResult, urlsplit

from ..types import ProviderEnv
from .provider_env import get_provider_env_value

__all__ = ["UNSUPPORTED_PROXY_PROTOCOL_MESSAGE", "resolve_http_proxy_url_for_target"]

_DEFAULT_PROXY_PORTS: dict[str, int] = {
    "ftp": 21,
    "gopher": 70,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}

_NO_PROXY_SEPARATOR = re.compile(r"[,\s]")
_PORT_SUFFIX = re.compile(r":\d*$")
_INTEGER_PREFIX = re.compile(r"[+-]?\d+")

UNSUPPORTED_PROXY_PROTOCOL_MESSAGE = (
    "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL."
)


def _get_proxy_env(key: str, env: ProviderEnv | None = None) -> str:
    """解析单个代理环境变量键：先查作用域覆盖值，再查 ``os.environ``.

    Args:
        key: 环境变量名，小写与大写形式都会查找。
        env: 可选的调用方作用域覆盖值。

    Returns:
        命中第一个非空取值；全部未命中时返回空串。
    """
    lowercase_key = key.lower()
    uppercase_key = key.upper()
    return (
        (env.get(lowercase_key) if env else None)
        or (env.get(uppercase_key) if env else None)
        or get_provider_env_value(lowercase_key)
        or get_provider_env_value(uppercase_key)
        or ""
    )


def _parse_int(value: str) -> int | None:
    """等价于 ``Number.parseInt(value, 10)``；用 ``None`` 代替 JavaScript 的 ``NaN``.

    Args:
        value: 待解析的字符串。

    Returns:
        开头的十进制整数；没有整数前缀时返回 ``None``。
    """
    match = _INTEGER_PREFIX.match(value.lstrip())
    return int(match.group()) if match else None


def _parse_proxy_target_url(target_url: str) -> SplitResult | None:
    """解析请求目标 URL；不是绝对 URL 时返回 ``None``.

    Args:
        target_url: 请求目标的绝对 URL 字符串。

    Returns:
        拆分结果；URL 无法解析时返回 ``None``。
    """
    try:
        return urlsplit(target_url)
    except ValueError:
        return None


def _strip_brackets(host: str) -> str:
    """去掉 IPv6 字面量两侧的方括号.

    Args:
        host: 主机名或 ``[::1]`` 形式的 IPv6 字面量。

    Returns:
        去掉方括号后的主机名。
    """
    return host[1:-1] if host.startswith("[") and host.endswith("]") else host


def _parse_no_proxy_entry(entry: str) -> tuple[str, int] | None:
    """把一条 ``NO_PROXY`` 项拆为 ``(host, port)``；缺省端口记为 0.

    Args:
        entry: ``NO_PROXY`` 中以逗号或空白分隔的单条记录。

    Returns:
        主机名与端口；条目为空时返回 ``None``。
    """
    trimmed = entry.strip().lower()
    if not trimmed:
        return None

    if trimmed.startswith("["):
        closing_bracket = trimmed.find("]")
        if closing_bracket != -1:
            host = trimmed[1:closing_bracket]
            rest = trimmed[closing_bracket + 1 :]
            if rest.startswith(":"):
                port = _parse_int(rest[1:])
                return (host, 0 if port is None else port)
            return (host, 0)

    if ":" in trimmed and len(trimmed.split(":")) > 2:
        return (trimmed, 0)

    colon_index = trimmed.rfind(":")
    if colon_index != -1 and colon_index == trimmed.find(":"):
        host = trimmed[:colon_index]
        port = _parse_int(trimmed[colon_index + 1 :])
        if port is not None:
            return (host, port)

    return (trimmed, 0)


def _should_proxy_hostname(hostname: str, port: int, env: ProviderEnv | None = None) -> bool:
    """按 ``NO_PROXY`` 判断目标主机是否应走代理.

    支持精确主机、后缀、``.domain``/``*.domain`` 通配符、``*``、
    方括号中的 IPv6 字面量以及可选端口。

    Args:
        hostname: 目标主机名（不含方括号）。
        port: 目标端口。
        env: 可选的调用方作用域覆盖值。

    Returns:
        目标未被豁免时返回 ``True``。
    """
    no_proxy = _get_proxy_env("no_proxy", env).lower()
    if not no_proxy:
        return True
    if no_proxy == "*":
        return False

    normalized_target_host = _strip_brackets(hostname.lower())

    for entry in _NO_PROXY_SEPARATOR.split(no_proxy):
        parsed = _parse_no_proxy_entry(entry)
        if parsed is None:
            continue
        entry_host, entry_port = parsed

        if entry_port and entry_port != port:
            continue

        domain = _strip_brackets(entry_host)
        if domain.startswith("*."):
            domain = domain[2:]
        elif domain.startswith(".") or domain.startswith("*"):
            domain = domain[1:]

        if not domain:
            continue

        if normalized_target_host == domain or normalized_target_host.endswith(f".{domain}"):
            return False

    return True


def _get_proxy_for_url(target_url: str, env: ProviderEnv | None = None) -> str:
    """返回 ``target_url`` 应使用的代理 URL；需直连时返回 ``""``.

    Args:
        target_url: 请求目标的绝对 URL。
        env: 可选的调用方作用域覆盖值。

    Returns:
        命中的代理 URL；直连时返回空串。
    """
    parsed_url = _parse_proxy_target_url(target_url)
    if parsed_url is None or not parsed_url.scheme or not parsed_url.netloc:
        return ""

    protocol = parsed_url.scheme.split(":", 1)[0]
    hostname = _strip_brackets(parsed_url.hostname or _PORT_SUFFIX.sub("", parsed_url.netloc))
    try:
        port = parsed_url.port
    except ValueError:
        return ""
    port = port or _DEFAULT_PROXY_PORTS.get(protocol) or 0
    if not _should_proxy_hostname(hostname, port, env):
        return ""

    proxy = _get_proxy_env(f"{protocol}_proxy", env) or _get_proxy_env("all_proxy", env)
    if proxy and "://" not in proxy:
        proxy = f"{protocol}://{proxy}"
    return proxy


def resolve_http_proxy_url_for_target(target_url: str, env: ProviderEnv | None = None) -> str | None:
    """解析 ``target_url`` 应使用的 HTTP(S) 代理.

    目标绕过代理时返回 ``None``。与 TypeScript 返回 WHATWG ``URL`` 不同，
    这里返回 ``httpx``（及 websocket 客户端）可直接接受的代理 URL 字符串。

    Args:
        target_url: 请求目标的绝对 URL。
        env: 可选的调用方作用域覆盖值。

    Returns:
        代理 URL；直连时返回 ``None``。

    Raises:
        ValueError: 当代理 URL 无法解析，或协议不是 HTTP/HTTPS 时。
    """
    proxy = _get_proxy_for_url(target_url, env)
    if not proxy:
        return None

    try:
        proxy_url = urlsplit(proxy)
    except ValueError as error:
        raise ValueError(f"Invalid proxy URL {json.dumps(proxy, ensure_ascii=False)}: {error}") from error

    if proxy_url.scheme not in ("http", "https"):
        raise ValueError(f"{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {proxy_url.scheme}:")

    return proxy
