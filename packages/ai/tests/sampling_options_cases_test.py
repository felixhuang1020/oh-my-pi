"""移植自 ``packages/ai/test/sampling-options.test.ts``。

通过 ``on_payload`` 注入点捕获 provider 请求体（上游在 ``onPayload`` 中抛出
``PayloadCaptured``），断言适配器实际序列化的采样字段。

上游传入形状为 ``StreamOptions`` 的对象字面量；本移植的适配器使用各自的数据类选项，因此
测试构造相应数据类。

重构删除了 ``openai-completions`` 协议与 ``azure-openai-responses`` 模块；原先覆盖它们的
用例改到同样会把 ``sampling_params`` 合并进请求体的 ``openai-responses`` 上，非 OpenAI
兼容 API 的忽略行为仍由 anthropic-messages 覆盖。
"""

from __future__ import annotations


from typing import Any

import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.compat import stream, stream_simple
from pi_ai.types import (
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StreamOptions,
    UserMessage,
)


class PayloadCaptured(Exception):
    """对应上游用于中断请求的 ``PayloadCaptured`` 错误。"""

    def __init__(self) -> None:
        super().__init__("payload captured")
        self.name = "PayloadCaptured"


_OPTION_TYPES: dict[str, type] = {
    "openai-responses": OpenAIResponsesOptions,
    "anthropic-messages": AnthropicOptions,
}


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=1234)])


def make_model(api: str, sampling_params: dict[str, Any] | None = None) -> Model:
    return Model(
        id="custom-model",
        name="Custom Model",
        api=api,
        provider="custom-provider",
        base_url="http://127.0.0.1:9/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=16384,
        sampling_params=sampling_params,
    )


def _capturing(sink: list[dict[str, Any]], cls: type, **overrides: Any):
    def on_payload(payload: Any, model: Model) -> None:
        sink.append(payload)
        raise PayloadCaptured()

    return cls(api_key="fake-key", on_payload=on_payload, **overrides)


async def capture_payload(model: Model, **options: Any) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []
    await stream(model, make_context(), _capturing(captured, _OPTION_TYPES[model.api], **options)).result()
    assert captured, "Expected payload to be captured before request failure"
    return captured[0]


async def test_merges_request_sampling_params_into_the_request_body() -> None:
    payload = await capture_payload(
        make_model("openai-responses"),
        sampling_params={"top_p": 0.95, "top_k": 0, "min_p": 0},
    )

    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 0
    assert payload["min_p"] == 0


async def test_omits_sampling_params_when_neither_options_nor_model_set_them() -> None:
    payload = await capture_payload(make_model("openai-responses"))

    assert payload.get("temperature") is None
    assert payload.get("top_p") is None


@pytest.mark.parametrize("api", ["openai-responses"])
async def test_applies_model_level_sampling_params_with_request_keys_taking_precedence(api: str) -> None:
    payload = await capture_payload(
        make_model(api, {"top_p": 0.95, "min_p": 0.05}),
        sampling_params={"top_p": 0.5},
    )

    assert payload["top_p"] == 0.5
    assert payload["min_p"] == 0.05


async def test_passes_request_sampling_params_through_stream_simple() -> None:
    captured: list[dict[str, Any]] = []

    options = _capturing(captured, SimpleStreamOptions, sampling_params={"top_p": 0.5})
    await stream_simple(make_model("openai-responses"), make_context(), options).result()

    assert captured and captured[0]["top_p"] == 0.5


async def test_overrides_named_request_fields() -> None:
    payload = await capture_payload(
        make_model("openai-responses"),
        temperature=0,
        sampling_params={"temperature": 1},
    )

    assert payload["temperature"] == 1


async def test_is_ignored_by_non_openai_compatible_apis() -> None:
    payload = await capture_payload(
        make_model("anthropic-messages"),
        sampling_params={"top_p": 0.9, "top_k": 40},
    )

    assert payload.get("top_p") is None
    assert payload.get("top_k") is None


@pytest.mark.parametrize("api", ["openai-responses"])
async def test_compat_stream_accepts_a_bare_stream_options(api: str) -> None:
    captured: list[dict[str, Any]] = []

    def on_payload(payload: Any, model: Model) -> None:
        captured.append(payload)
        raise PayloadCaptured()

    options = StreamOptions(api_key="fake-key", on_payload=on_payload, sampling_params={"top_p": 0.95})
    await stream(make_model(api), make_context(), options).result()

    assert captured and captured[0]["top_p"] == 0.95
