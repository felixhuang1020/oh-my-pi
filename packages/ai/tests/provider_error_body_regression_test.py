"""移植自 ``packages/ai/test/provider-error-body-regression.test.ts``。

针对 issue provider-error-body-passthrough 的分层 provider 回归测试：每层取一个代表
（验收标准 7）——断言最终 ``errorMessage`` 同时携带 HTTP 状态与响应体原因。

重构后 ``openai-completions`` 与 Bedrock provider 均已删除，仅剩四个受支持的 API
（``anthropic-messages``、``openai-responses``、``google-generative-ai``、
``pi-messages``）。上游 mock ``openai`` SDK 与 ``@aws-sdk/client-bedrock-runtime``；
本移植的适配器直接走 HTTP，因此注入失败统一使用 ``fetch=`` 选项（返回带响应体的
403），与各适配器内部的 httpx 调用路径一致。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream_simple as stream_simple_anthropic
from pi_ai.api.google_generative_ai import GoogleOptions
from pi_ai.api.google_generative_ai import stream_simple as stream_simple_google
from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.api.openai_responses import stream_simple as stream_simple_openai_responses
from pi_ai.api.pi_messages import PiMessagesOptions
from pi_ai.api.pi_messages import stream_simple as stream_simple_pi_messages
from pi_ai.types import (
    Context,
    Model,
    ModelCost,
    TextContent,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

_ZERO_COST = ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0)

CONTEXT = normalize_context(
    Context(
        system_prompt="",
        messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
        tools=[],
    )
)


def _model(api: str, provider: str, base_url: str) -> Model:
    """构造受支持 API 的最小 chat 模型。"""
    return Model(
        id="test-model",
        name="Test Model",
        api=api,
        provider=provider,
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=_ZERO_COST,
        context_window=1000,
        max_tokens=100,
    )


ANTHROPIC_MODEL = _model("anthropic-messages", "anthropic", "https://api.anthropic.com")
RESPONSES_MODEL = _model("openai-responses", "openai", "https://api.openai.com/v1")
GOOGLE_MODEL = _model(
    "google-generative-ai", "google", "https://generativelanguage.googleapis.com"
)
PI_MESSAGES_MODEL = _model("pi-messages", "openai", "https://api.openai.com/v1")


def _json_body(payload: Any) -> bytes:
    """按 ``JSON.stringify`` 的方式序列化：紧凑且不做 ASCII 转义。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _JsonFetch:
    """对每个请求都返回 ``status``/``payload``，并记录请求。"""

    def __init__(self, status: int, payload: Any) -> None:
        self._status = status
        self._payload = payload
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status,
            headers={"content-type": "application/json"},
            content=_json_body(self._payload),
            request=request,
        )


def _blocked_fetch() -> _JsonFetch:
    """带网关 WAF 原因的 403 响应体。"""
    return _JsonFetch(403, {"error": {"error": "blocked by gateway WAF"}})


def _assert_status_and_body_surface(output: Any) -> None:
    """断言最终错误同时透出 403 状态与响应体原因。"""
    assert output.stop_reason == "error"
    assert "403" in output.error_message
    assert "blocked by gateway WAF" in output.error_message
    assert output.error_message != "403 status code (no body)"


# ---------------------------------------------------------------------------------
# provider 错误响应体透传（分层回归）
# ---------------------------------------------------------------------------------


async def test_anthropic_messages_surfaces_status_and_body():
    fetch = _blocked_fetch()

    output = await stream_simple_anthropic(
        ANTHROPIC_MODEL, CONTEXT, AnthropicOptions(api_key="test", fetch=fetch, max_retries=0)
    ).result()

    _assert_status_and_body_surface(output)
    assert fetch.requests


async def test_openai_responses_status_only_keeps_the_prefix_and_surfaces_the_body():
    fetch = _blocked_fetch()

    output = await stream_simple_openai_responses(
        RESPONSES_MODEL, CONTEXT, OpenAIResponsesOptions(api_key="test", fetch=fetch, max_retries=0)
    ).result()

    assert output.stop_reason == "error"
    assert "OpenAI API error (403)" in output.error_message
    assert "blocked by gateway WAF" in output.error_message
    assert fetch.requests


async def test_google_generative_ai_surfaces_status_and_body():
    fetch = _blocked_fetch()

    output = await stream_simple_google(
        GOOGLE_MODEL, CONTEXT, GoogleOptions(api_key="test", fetch=fetch, max_retries=0)
    ).result()

    _assert_status_and_body_surface(output)
    assert fetch.requests


async def test_pi_messages_surfaces_status_and_body():
    fetch = _blocked_fetch()

    output = await stream_simple_pi_messages(
        PI_MESSAGES_MODEL, CONTEXT, PiMessagesOptions(api_key="test", fetch=fetch, max_retries=0)
    ).result()

    _assert_status_and_body_surface(output)
    assert fetch.requests
