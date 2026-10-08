"""移植自 ``packages/ai/test/node-http-proxy.test.ts``。

测试 :func:`pi_ai.utils.node_http_proxy.resolve_http_proxy_url_for_target`。
上游直接改写 ``process.env``；移植版改为接收作用域 ``env`` 映射并优先于进程环境，
因此上游的每个环境变量用例都通过该映射表达，进程别名排序的用例则用
``monkeypatch.setenv`` 实现。

移植版返回代理 URL 字符串（而非 WHATWG ``URL`` 对象），所以上游的
``?.toString()`` 断言在这里直接比较字符串。
"""

from __future__ import annotations

import pytest

from pi_ai.utils.node_http_proxy import (
    UNSUPPORTED_PROXY_PROTOCOL_MESSAGE,
    resolve_http_proxy_url_for_target,
)

BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "npm_config_http_proxy",
    "npm_config_https_proxy",
    "npm_config_proxy",
    "npm_config_no_proxy",
)


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch):
    """确定性的进程环境：不含任何代理环境变量。"""
    for key in PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_respects_no_proxy_exclusions():
    env = {
        "HTTPS_PROXY": "http://proxy.example:8080",
        "NO_PROXY": "bedrock-runtime.us-east-1.amazonaws.com",
    }

    assert resolve_http_proxy_url_for_target(BEDROCK_URL, env) is None


def test_resolves_http_and_https_proxy_urls():
    env = {"HTTPS_PROXY": "http://proxy.example:8080"}

    assert resolve_http_proxy_url_for_target(BEDROCK_URL, env) == "http://proxy.example:8080"


def test_prefers_scoped_proxy_env_aliases_before_process_env_aliases(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("https_proxy", "http://process-proxy.example:8080")

    assert (
        resolve_http_proxy_url_for_target(BEDROCK_URL, {"HTTPS_PROXY": "http://scoped-proxy.example:8080"})
        == "http://scoped-proxy.example:8080"
    )


def test_rejects_socks_and_pac_proxy_urls_explicitly():
    env = {"HTTPS_PROXY": "socks5://proxy.example:1080"}

    with pytest.raises(ValueError) as exc_info:
        resolve_http_proxy_url_for_target(BEDROCK_URL, env)

    assert UNSUPPORTED_PROXY_PROTOCOL_MESSAGE in str(exc_info.value)


def test_handles_subdomain_wildcards_ipv6_and_ports_in_no_proxy():
    env = {
        "HTTPS_PROXY": "http://proxy.example:8080",
        "NO_PROXY": "example.com, .wildcard.org, *.star.net, ::1, [2001:db8::1], 127.0.0.1:8080",
    }

    assert resolve_http_proxy_url_for_target("https://example.com", env) is None
    assert resolve_http_proxy_url_for_target("https://api.example.com", env) is None
    assert resolve_http_proxy_url_for_target("https://wildcard.org", env) is None
    assert resolve_http_proxy_url_for_target("https://api.wildcard.org", env) is None
    assert resolve_http_proxy_url_for_target("https://star.net", env) is None
    assert resolve_http_proxy_url_for_target("https://api.star.net", env) is None
    assert resolve_http_proxy_url_for_target("https://notexample.com", env) == "http://proxy.example:8080"

    assert resolve_http_proxy_url_for_target("https://[::1]:80", env) is None
    assert resolve_http_proxy_url_for_target("https://[2001:db8::1]", env) is None
    assert resolve_http_proxy_url_for_target("https://127.0.0.1:8080", env) is None
    assert resolve_http_proxy_url_for_target("https://127.0.0.1:3000", env) == "http://proxy.example:8080"
