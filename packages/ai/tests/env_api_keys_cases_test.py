"""移植 ``packages/ai/test/env-api-keys.test.ts``。

上游测试直接改 ``process.env`` 并在 ``afterEach`` 恢复；这里用 ``monkeypatch``
获得同样的隔离。所有断言都针对模块默认（进程环境）路径，与 TypeScript 完全一致。
"""

from __future__ import annotations

import pytest

from pi_ai.env_api_keys import find_env_keys, get_env_api_key

_ALL_TOUCHED = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ALL_TOUCHED:
        monkeypatch.delenv(name, raising=False)


def test_reports_anthropic_auth_token_alongside_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")

    assert find_env_keys("anthropic") == [
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ]
    assert get_env_api_key("anthropic") == "api-key"


def test_does_not_return_anthropic_auth_token_as_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")

    assert find_env_keys("anthropic") == ["ANTHROPIC_AUTH_TOKEN"]
    assert get_env_api_key("anthropic") is None


def test_falls_back_to_anthropic_api_key_for_api_key_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")

    assert get_env_api_key("anthropic") == "api-key"
