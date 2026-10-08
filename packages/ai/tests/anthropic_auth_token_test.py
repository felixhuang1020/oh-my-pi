"""``ANTHROPIC_AUTH_TOKEN`` 解析与 Anthropic 兼容 User-Agent 的测试。

移植自 ``packages/ai/test/anthropic-auth-token.test.ts``。TypeScript 套件 mock 了
``@anthropic-ai/sdk`` 构造函数；本移植则补丁替换适配器的 ``create_http_client``
接缝，并检查记录下来的 :class:`httpx.Request`（请求头 + JSON body）。
"""

from __future__ import annotations

import json

import httpx

import pi_ai.api.anthropic_messages as anthropic_messages
from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.types import SimpleStreamOptions
from pi_ai.env_api_keys import ANTHROPIC_AUTH_TOKEN_ENV
from pi_ai.models import CreateModelsOptions, create_models
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.types import Context, Model, UserMessage
from pi_ai.utils.abort import AbortController
from pi_ai.utils.pi_user_agent import get_pi_user_agent
from pi_ai.utils.transcript import normalize_context

NEVER_ABORTED = AbortController().signal

SSE_RESPONSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test",'
    b'"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":1}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)

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

KIMI_CODING_MODEL = Model(
    id="kimi-for-coding",
    name="Kimi For Coding",
    api="anthropic-messages",
    provider="moonshotai",
    base_url="https://api.moonshot.ai/anthropic",
    reasoning=False,
    input=["text"],
    context_window=100000,
    max_tokens=4096,
)


class FakeAuthContext:
    """基于 dict 的鸭子类型 ``AuthContext`` 实现。"""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    async def env(self, name: str) -> str | None:
        return self.values.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


def install_transport(monkeypatch) -> list[httpx.Request]:
    """补丁替换适配器的 HTTP 客户端工厂，并记录每一个请求。"""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=SSE_RESPONSE, request=request
        )

    def fake_create_http_client(**kwargs) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=kwargs.get("headers"),
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        )

    monkeypatch.setattr(anthropic_messages, "create_http_client", fake_create_http_client)
    return recorded


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


# ---------------------------------------------------------------------------------
# Anthropic auth token 环境变量
# ---------------------------------------------------------------------------------


async def test_resolves_anthropic_auth_token_as_a_bearer_authorization_header():
    provider = anthropic_provider()
    auth = await provider.auth.api_key.resolve(
        ctx=FakeAuthContext(
            {
                ANTHROPIC_AUTH_TOKEN_ENV: "auth-token",
                "ANTHROPIC_API_KEY": "api-key",
            }
        ),
        signal=NEVER_ABORTED,
    )

    assert auth.auth.headers == {"Authorization": "Bearer auth-token"}
    assert auth.source == ANTHROPIC_AUTH_TOKEN_ENV


async def test_uses_authorization_headers_without_oauth_mode_request_shaping(monkeypatch):
    recorded = install_transport(monkeypatch)
    stream_object = stream(
        ANTHROPIC_MODEL, CONTEXT, AnthropicOptions(headers={"Authorization": "Bearer gateway-token"})
    )
    await stream_object.result()

    request = recorded[0]
    assert "x-api-key" not in request.headers
    assert request.headers.get("authorization") == "Bearer gateway-token"
    assert "oauth-2025-04-20" not in (request.headers.get("anthropic-beta") or "")
    assert _body(request)["system"][0]["text"] == "System prompt."


async def test_threads_auth_context_anthropic_auth_token_through_request_headers(monkeypatch):
    recorded = install_transport(monkeypatch)
    models = create_models(
        CreateModelsOptions(
            auth_context=FakeAuthContext({ANTHROPIC_AUTH_TOKEN_ENV: "ctx-token"})
        )
    )
    models.set_provider(anthropic_provider())

    await models.stream_simple(ANTHROPIC_MODEL, RAW_CONTEXT).result()

    request = recorded[0]
    assert "x-api-key" not in request.headers
    assert request.headers.get("authorization") == "Bearer ctx-token"
    assert "oauth-2025-04-20" not in (request.headers.get("anthropic-beta") or "")
    assert _body(request)["system"][0]["text"] == "System prompt."


async def test_lets_explicit_request_headers_override_anthropic_auth_token(monkeypatch):
    recorded = install_transport(monkeypatch)
    models = create_models(
        CreateModelsOptions(
            auth_context=FakeAuthContext({ANTHROPIC_AUTH_TOKEN_ENV: "ctx-token"})
        )
    )
    models.set_provider(anthropic_provider())

    await models.stream_simple(
        ANTHROPIC_MODEL, RAW_CONTEXT, SimpleStreamOptions(headers={"Authorization": "Bearer explicit-token"})
    ).result()

    assert recorded[0].headers.get("authorization") == "Bearer explicit-token"


# ---------------------------------------------------------------------------------
# Anthropic 兼容 User-Agent
# ---------------------------------------------------------------------------------


async def test_uses_pis_user_agent_by_default_for_anthropic_messages_requests(monkeypatch):
    recorded = install_transport(monkeypatch)
    await stream(ANTHROPIC_MODEL, CONTEXT, AnthropicOptions(api_key="anthropic-key")).result()

    assert recorded[0].headers.get("user-agent") == get_pi_user_agent()


async def test_lets_explicit_headers_override_the_default_anthropic_messages_user_agent(monkeypatch):
    recorded = install_transport(monkeypatch)
    await stream(
        KIMI_CODING_MODEL,
        CONTEXT,
        AnthropicOptions(api_key="kimi-key", headers={"User-Agent": "custom-client"}),
    ).result()

    assert recorded[0].headers.get("user-agent") == "custom-client"


async def test_preserves_explicit_anthropic_beta_header_replacement(monkeypatch):
    recorded = install_transport(monkeypatch)
    await stream(
        ANTHROPIC_MODEL,
        CONTEXT,
        AnthropicOptions(api_key="anthropic-key", headers={"anthropic-beta": "custom-beta"}),
    ).result()

    assert recorded[0].headers.get("anthropic-beta") == "custom-beta"


async def test_preserves_explicit_anthropic_beta_header_suppression(monkeypatch):
    recorded = install_transport(monkeypatch)
    await stream(
        ANTHROPIC_MODEL,
        CONTEXT,
        AnthropicOptions(api_key="anthropic-key", headers={"anthropic-beta": None}),
    ).result()

    assert recorded[0].headers.get("anthropic-beta") is None
