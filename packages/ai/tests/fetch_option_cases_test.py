"""移植 ``packages/ai/test/fetch-option.test.ts``。

上游把 ``globalThis.fetch`` 替换为会抛异常的 fallback，并断言实际使用的是调用方
传入的 ``fetch`` 选项。Python 中与环境 fallback 对应的是默认
:class:`httpx.AsyncHTTPTransport`，本模块 monkeypatch 它使其一旦被调用即报错；
注入的 ``fetch`` 则通过记录的 :class:`httpx.Request` 对象来观测。

重构删除了 azure-openai-responses / google-vertex / mistral-conversations /
openai-codex-responses / openai-completions 以及图像生成模块，对应用例一并移除；
保留 anthropic-messages、openai-responses、google-generative-ai 与 pi-messages
的 fetch 透传覆盖。

移植版 Google adapter 直接走 HTTP（不依赖 ``@google/genai`` SDK），因此它*接受*
自定义 fetch，而不是像上游那样拒绝它。
"""

from __future__ import annotations

import httpx
import pytest

from pi_ai.api.anthropic_messages import stream_simple as stream_anthropic
from pi_ai.api.google_generative_ai import stream_simple as stream_google
from pi_ai.api.openai_responses import stream_simple as stream_openai_responses
from pi_ai.api.pi_messages import stream_simple as stream_pi_messages
from pi_ai.types import (
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

CONTEXT = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))


def create_model(api: str) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api=api,
        provider="test-provider",
        base_url="https://upstream.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10_000,
        max_tokens=1_000,
    )


class RecordingFetch:
    """上游的 ``custom`` mock：始终以带 JSON body 的 401 响应。"""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            401,
            json={"error": {"message": "upstream rejected request"}},
            headers={"content-type": "application/json"},
            request=request,
        )


@pytest.fixture
def ambient_fetch(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """上游会抛异常的 ``fallback``：默认 transport 绝不能被走到。"""
    calls: list[httpx.Request] = []

    async def handle(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("ambient fetch must not be called")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    return calls


async def test_passes_fetch_through_stream_simple_to_the_anthropic_sdk(
    ambient_fetch: list[httpx.Request],
) -> None:
    custom = RecordingFetch()
    await stream_anthropic(
        create_model("anthropic-messages"),
        CONTEXT,
        SimpleStreamOptions(api_key="test-key", fetch=custom, max_retries=0),
    ).result()

    assert custom.requests
    assert ambient_fetch == []


async def test_passes_fetch_through_stream_simple_to_openai_sdk_adapters(
    ambient_fetch: list[httpx.Request],
) -> None:
    custom = RecordingFetch()
    await stream_openai_responses(
        create_model("openai-responses"),
        CONTEXT,
        SimpleStreamOptions(api_key="test-key", fetch=custom, max_retries=0),
    ).result()

    assert len(custom.requests) == 1
    assert ambient_fetch == []


async def test_uses_fetch_for_pi_messages_http_requests(
    ambient_fetch: list[httpx.Request],
) -> None:
    custom = RecordingFetch()
    await stream_pi_messages(
        create_model("pi-messages"),
        CONTEXT,
        SimpleStreamOptions(api_key="test-key", fetch=custom),
    ).result()

    assert len(custom.requests) == 1
    assert ambient_fetch == []


async def test_allows_google_adapters_to_receive_the_default_fetch_explicitly(
    ambient_fetch: list[httpx.Request],
) -> None:
    # 上游传入的是打桩的环境 fetch；移植版没有全局 fetch，因此忠实的检查是
    # 显式传入的 fetch 被接受并实际使用。
    ambient = RecordingFetch()
    result = await stream_google(
        create_model("google-generative-ai"),
        CONTEXT,
        SimpleStreamOptions(api_key="test-key", fetch=ambient),
    ).result()

    assert len(ambient.requests) == 1
    assert "Custom fetch is not supported" not in (result.error_message or "")
