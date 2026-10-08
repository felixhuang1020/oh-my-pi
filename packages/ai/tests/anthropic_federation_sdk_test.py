"""需要真实 Anthropic SDK 的联邦（federation）行为测试。

移植自 ``packages/ai/test/anthropic-federation-sdk.test.ts``。上游第一个用例走 SDK 自己的
OIDC 联邦 token 交换（对伪造 fetch 运行真实的 ``@anthropic-ai/sdk``）。Python 移植仅为
API 对齐而*转发*联邦配置给 SDK 等价客户端，并不实现 OIDC 交换（见
``pi_ai/api/anthropic_messages.py`` 的模块注释），因此该用例被跳过。而断言凭据链
绝不执行的 header 自持鉴权用例可以表达，已移植到下方。
"""

from __future__ import annotations

import httpx
import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.types import Context, Model, UserMessage
from pi_ai.utils.transcript import normalize_context

SSE_RESPONSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test",'
    b'"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":1}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)

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


@pytest.mark.skip(
    reason="the port forwards the federation config for API parity but does not run the "
    "Anthropic SDK's OIDC token exchange, so there is no /v1/oauth/token call to observe"
)
async def test_exchanges_the_identity_token_once_across_requests():
    raise AssertionError("unreachable")


async def test_does_not_run_the_sdk_credential_chain_for_header_owned_auth():
    requests: list[dict[str, str | None]] = []

    async def fetch(request: httpx.Request) -> httpx.Response:
        requests.append(
            {"path": request.url.path, "authorization": request.headers.get("authorization")}
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=SSE_RESPONSE, request=request
        )

    message = await stream(
        ANTHROPIC_MODEL,
        CONTEXT,
        AnthropicOptions(headers={"Authorization": "Bearer auth-token"}, fetch=fetch),
    ).result()

    assert message.stop_reason == "stop"
    assert requests == [{"path": "/v1/messages", "authorization": "Bearer auth-token"}]
