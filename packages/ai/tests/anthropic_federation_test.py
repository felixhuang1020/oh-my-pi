"""Anthropic 工作负载身份联邦（workload identity federation）配置解析的测试。

移植自 ``packages/ai/test/anthropic-federation.test.ts``。TypeScript 套件 mock 了
``@anthropic-ai/sdk`` 构造函数；本移植用记录型假实现替换 ``AnthropicMessagesClient``，
从而观察构造选项与请求参数。
"""

from __future__ import annotations

import httpx

import pi_ai.api.anthropic_messages as anthropic_messages
from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.env_api_keys import (
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_FEDERATION_RULE_ID_ENV,
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV,
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV,
    ANTHROPIC_WORKSPACE_ID_ENV,
)
from pi_ai.models import CreateModelsOptions, create_models
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.types import Context, Model, UserMessage
from pi_ai.utils.abort import AbortController
from pi_ai.utils.transcript import normalize_context

NEVER_ABORTED = AbortController().signal

SSE_RESPONSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test",'
    b'"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":1}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)

FEDERATION_ENV = {
    ANTHROPIC_FEDERATION_RULE_ID_ENV: "fdrl_test",
    ANTHROPIC_ORGANIZATION_ID_ENV: "org-test",
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV: "svac_test",
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV: "/tmp/identity.jwt",
}

EXPECTED_CONFIG = {
    "organization_id": "org-test",
    "workspace_id": None,
    "authentication": {
        "type": "oidc_federation",
        "federation_rule_id": "fdrl_test",
        "service_account_id": "svac_test",
        "identity_token": {"source": "file", "path": "/tmp/identity.jwt"},
    },
}

RAW_CONTEXT = Context(system_prompt="System prompt.", messages=[UserMessage(content="Hello", timestamp=0)])
CONTEXT = normalize_context(
    Context(system_prompt="System prompt.", messages=[UserMessage(content="Hello", timestamp=0)])
)

ANTHROPIC_MODEL = Model(
    id="claude-test",
    name="Claude Test",
    api="anthropic-messages",
    provider="anthropic",
    base_url="https://api.anthropic.com",
    reasoning=False,
    input=["text"],
    context_window=100000,
    max_tokens=4096,
)


class FakeAuthContext:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    async def env(self, name: str) -> str | None:
        return self.values.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


class RecordingAnthropicClient:
    """记录构造选项与请求参数，随后返回固定的 SSE 响应。"""

    instances: list["RecordingAnthropicClient"] = []
    created_params: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.config = kwargs.get("config")
        self.api_key = kwargs.get("api_key")
        self.auth_token = kwargs.get("auth_token")
        self.default_headers = dict(kwargs.get("default_headers") or {})
        RecordingAnthropicClient.instances.append(self)

    async def create_message(self, params, *, signal=None, timeout_ms=None) -> httpx.Response:
        RecordingAnthropicClient.created_params.append(params)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE_RESPONSE)


def install_recording_client(monkeypatch) -> None:
    RecordingAnthropicClient.instances = []
    RecordingAnthropicClient.created_params = []
    monkeypatch.setattr(anthropic_messages, "AnthropicMessagesClient", RecordingAnthropicClient)


def resolve_with_env(env: dict[str, str]):
    return anthropic_provider().auth.api_key.resolve(ctx=FakeAuthContext(env), signal=NEVER_ABORTED)


# ---------------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------------


async def test_resolves_the_federation_variables_as_provider_env_with_no_request_auth():
    result = await resolve_with_env(FEDERATION_ENV)
    assert result.auth.api_key is None
    assert result.auth.headers is None
    assert result.env == FEDERATION_ENV
    assert result.source == "workload identity federation"


async def test_passes_anthropic_workspace_id_through_when_set():
    result = await resolve_with_env({**FEDERATION_ENV, ANTHROPIC_WORKSPACE_ID_ENV: "wrkspc_test"})
    assert result.env[ANTHROPIC_WORKSPACE_ID_ENV] == "wrkspc_test"


async def test_is_not_configured_when_a_federation_variable_is_missing():
    partial = {key: value for key, value in FEDERATION_ENV.items() if key != ANTHROPIC_IDENTITY_TOKEN_FILE_ENV}
    assert await resolve_with_env(partial) is None


async def test_treats_anthropic_service_account_id_as_optional_like_the_sdk(monkeypatch):
    partial = {key: value for key, value in FEDERATION_ENV.items() if key != ANTHROPIC_SERVICE_ACCOUNT_ID_ENV}
    result = await resolve_with_env(partial)
    assert result.auth.api_key is None
    assert result.auth.headers is None
    assert result.env == partial
    assert result.source == "workload identity federation"

    install_recording_client(monkeypatch)
    await stream(ANTHROPIC_MODEL, CONTEXT, AnthropicOptions(env=partial)).result()
    assert RecordingAnthropicClient.instances[0].config == {
        **EXPECTED_CONFIG,
        "authentication": {**EXPECTED_CONFIG["authentication"], "service_account_id": None},
    }


async def test_keeps_api_key_and_auth_token_precedence_over_federation():
    api_key = await resolve_with_env({**FEDERATION_ENV, "ANTHROPIC_API_KEY": "api-key"})
    assert api_key.auth.api_key == "api-key"
    assert api_key.source == "ANTHROPIC_API_KEY"

    auth_token = await resolve_with_env({**FEDERATION_ENV, ANTHROPIC_AUTH_TOKEN_ENV: "auth-token"})
    assert auth_token.auth.headers == {"Authorization": "Bearer auth-token"}
    assert auth_token.source == ANTHROPIC_AUTH_TOKEN_ENV


# ---------------------------------------------------------------------------------
# 请求整形
# ---------------------------------------------------------------------------------


async def test_hands_the_sdk_a_federation_config_instead_of_a_key(monkeypatch):
    install_recording_client(monkeypatch)
    await stream(ANTHROPIC_MODEL, CONTEXT, AnthropicOptions(env=FEDERATION_ENV)).result()

    client = RecordingAnthropicClient.instances[0]
    assert client.api_key is None
    assert client.auth_token is None
    assert client.config == EXPECTED_CONFIG
    assert client.default_headers.get("Authorization") is None
    assert "oauth-2025-04-20" not in (RecordingAnthropicClient.created_params[0].get("betas") or [])


async def test_threads_auth_context_federation_variables_through_models(monkeypatch):
    install_recording_client(monkeypatch)
    models = create_models(CreateModelsOptions(auth_context=FakeAuthContext(FEDERATION_ENV)))
    models.set_provider(anthropic_provider())

    await models.stream_simple(ANTHROPIC_MODEL, RAW_CONTEXT).result()

    client = RecordingAnthropicClient.instances[0]
    assert client.api_key is None
    assert client.config == EXPECTED_CONFIG


async def test_lets_an_explicit_api_key_win_over_federation_env(monkeypatch):
    install_recording_client(monkeypatch)
    await stream(
        ANTHROPIC_MODEL, CONTEXT, AnthropicOptions(api_key="explicit-key", env=FEDERATION_ENV)
    ).result()

    client = RecordingAnthropicClient.instances[0]
    assert client.api_key == "explicit-key"
    assert client.config is None


async def test_does_not_federate_other_anthropic_messages_providers(monkeypatch):
    install_recording_client(monkeypatch)
    kimi = Model(
        id="kimi-for-coding",
        name="Kimi For Coding",
        api="anthropic-messages",
        provider="kimi-coding",
        base_url="https://api.kimi.com/coding",
        reasoning=False,
        input=["text"],
        context_window=100000,
        max_tokens=4096,
    )
    message = await stream(kimi, CONTEXT, AnthropicOptions(env=FEDERATION_ENV)).result()

    assert message.stop_reason == "error"
    assert RecordingAnthropicClient.instances == []
