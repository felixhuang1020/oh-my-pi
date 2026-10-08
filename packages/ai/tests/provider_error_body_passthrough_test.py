"""移植自 ``packages/ai/test/provider-error-body-passthrough.test.ts``。

针对 issue provider-error-body-passthrough 的回归测试：代理/网关后的端点返回非 2xx
时，真实原因在响应体里，而 provider SDK 的报错信息却含糊不清。上游测试让一个 ``openai``
``APIError`` 经由忽略响应体的 provider 传出，断言最终 ``errorMessage`` 同时包含状态码
与响应体中的原因。

重构后 image 生成与 ``openrouter`` provider 均已删除，本移植收窄到幸存的
``openai-responses`` 文本 provider：SDK 形状的 ``APIError``（消息为
``"403 status code (no body)"``，但解析后的响应体挂在 ``error`` 属性上）由
``options.fetch``（异步 ``(httpx.Request) -> httpx.Response``）抛出，而非 mock
``openai`` SDK；断言适配器仍透出真实原因。
"""

from __future__ import annotations

from typing import Any

import httpx

from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.api.openai_responses import stream as stream_openai_responses
from pi_ai.types import (
    Context,
    Model,
    ModelCost,
    TextContent,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

#: 模块私有解析后响应体，与上游 ``{"error":"blocked by gateway WAF"}`` 一致。
_ERROR_BODY: dict[str, Any] = {"error": "blocked by gateway WAF"}

CONTEXT = normalize_context(
    Context(messages=[UserMessage(content=[TextContent(text="Generate a dog")], timestamp=0)])
)


class _SdkApiError(Exception):
    """``openai`` ``APIError`` 的形状：含糊消息 + ``status`` + 解析后 ``error``。"""

    def __init__(self, message: str, status: int, error: Any) -> None:
        super().__init__(message)
        self.name = "APIError"
        self.status = status
        self.error = error


class _RaisingFetch:
    """伪造的 ``options.fetch``，始终抛出预设的 SDK 形状异常。"""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.requests: list[httpx.Request] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raise self._error


def _model() -> Model:
    return Model(
        id="gpt-test",
        name="GPT Test",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        context_window=1000,
        max_tokens=100,
    )


async def test_surfaces_the_parsed_body_reason_instead_of_the_opaque_sdk_message():
    model = _model()
    fetch = _RaisingFetch(_SdkApiError("403 status code (no body)", 403, _ERROR_BODY))

    output = await stream_openai_responses(
        model, CONTEXT, OpenAIResponsesOptions(api_key="test", fetch=fetch, max_retries=0)
    ).result()

    assert output.stop_reason == "error"
    # 应透出状态码。
    assert "403" in output.error_message
    # 响应体中的原因不能被 SDK 的含糊报错吞掉。
    assert "blocked by gateway WAF" in output.error_message
    assert output.error_message != "403 status code (no body)"
    assert fetch.call_count == 1
