"""移植自 ``cache-retention.test.ts``。

上游套件在任何 HTTP 调用前从 ``onPayload`` 抛异常来截获各 provider 的请求 payload。
Python 适配器暴露同样的 ``on_payload`` 接缝，因此这里的所有用例都保持离线：payload 被
记录，抛出的错误被编码进事件流，由测试排空消费。

上游用 ``ANTHROPIC_API_KEY`` 门控两个 Anthropic 用例、用 ``OPENAI_API_KEY`` 门控两个
OpenAI Responses 用例。这些也永远不会触网——密钥只是让适配器先构建客户端，随后 payload
回调就中止了请求——因此这里改用 ``monkeypatch.setenv`` 直接运行，而非跳过。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream as stream_anthropic
from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.api.openai_responses import stream as stream_openai_responses
from pi_ai.compat import get_model, stream
from pi_ai.types import (
    Context,
    Model,
    ModelCompat,
    ModelCost,
    TranscriptContext,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context


class PayloadCaptured(Exception):
    """上游的 ``PayloadCaptured``：payload 记录完成后终止请求。"""


class PayloadCapture:
    """记录请求 payload 并中止请求，等价于上游的 ``stopAfterPayload``。"""

    def __init__(self) -> None:
        self.payload: Any = None

    def __call__(self, payload: Any, model: Model | None = None) -> None:
        self.payload = payload
        raise PayloadCaptured()


@pytest.fixture(autouse=True)
def _clear_cache_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """上游的 ``beforeEach``/``afterEach``：环境变量开关初始为未设置。"""
    monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)


def _raw_context() -> Context:
    return Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Hello", timestamp=0)],
    )


def _context() -> TranscriptContext:
    return normalize_context(_raw_context())


async def _drain(stream_object: Any) -> None:
    async for _ in stream_object:
        pass


def _proxy(model: Model, **overrides: Any) -> Model:
    """dataclass 模型版的 ``{ ...baseModel, ...overrides }``。"""
    return dataclasses.replace(model, **overrides)


# ---------------------------------------------------------------------------------
# Anthropic provider 场景
# ---------------------------------------------------------------------------------


async def test_anthropic_uses_default_cache_ttl_when_pi_cache_retention_is_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    model = get_model("anthropic", "claude-haiku-4-5")
    capture = PayloadCapture()

    await _drain(stream(model, _raw_context(), AnthropicOptions(on_payload=capture)))

    assert capture.payload is not None
    # system prompt 应带有不带 ttl 的 cache_control
    assert capture.payload["system"] is not None
    assert capture.payload["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_uses_1h_cache_ttl_when_pi_cache_retention_is_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    model = get_model("anthropic", "claude-haiku-4-5")
    capture = PayloadCapture()

    await _drain(stream(model, _raw_context(), AnthropicOptions(on_payload=capture)))

    assert capture.payload is not None
    assert capture.payload["system"] is not None
    assert capture.payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_adds_ttl_for_non_api_anthropic_base_url_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    base_model = get_model("anthropic", "claude-haiku-4-5")
    proxy_model = _proxy(base_model, base_url="https://my-proxy.example.com/v1")
    capture = PayloadCapture()

    await _drain(
        stream_anthropic(
            proxy_model,
            _context(),
            AnthropicOptions(api_key="fake-key", on_payload=capture),
        )
    )

    assert capture.payload is not None
    assert capture.payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_omits_ttl_when_supports_long_cache_retention_is_false() -> None:
    base_model = get_model("anthropic", "claude-haiku-4-5")
    proxy_model = _proxy(
        base_model,
        base_url="https://my-proxy.example.com/v1",
        compat=ModelCompat(supports_long_cache_retention=False),
    )
    capture = PayloadCapture()

    await _drain(
        stream_anthropic(
            proxy_model,
            _context(),
            AnthropicOptions(api_key="fake-key", cache_retention="long", on_payload=capture),
        )
    )

    assert capture.payload is not None
    assert capture.payload["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_omits_cache_control_when_cache_retention_is_none() -> None:
    base_model = get_model("anthropic", "claude-haiku-4-5")
    capture = PayloadCapture()

    await _drain(
        stream_anthropic(
            base_model,
            _context(),
            AnthropicOptions(api_key="fake-key", cache_retention="none", on_payload=capture),
        )
    )

    assert capture.payload is not None
    assert capture.payload["system"][0].get("cache_control") is None


async def test_anthropic_adds_cache_control_to_string_user_messages() -> None:
    base_model = get_model("anthropic", "claude-haiku-4-5")
    capture = PayloadCapture()

    await _drain(
        stream_anthropic(
            base_model,
            _context(),
            AnthropicOptions(api_key="fake-key", on_payload=capture),
        )
    )

    assert capture.payload is not None
    last_message = capture.payload["messages"][-1]
    assert isinstance(last_message["content"], list) is True
    last_block = last_message["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_sets_1h_cache_ttl_when_cache_retention_is_long() -> None:
    base_model = get_model("anthropic", "claude-haiku-4-5")
    capture = PayloadCapture()

    await _drain(
        stream_anthropic(
            base_model,
            _context(),
            AnthropicOptions(api_key="fake-key", cache_retention="long", on_payload=capture),
        )
    )

    assert capture.payload is not None
    assert capture.payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ---------------------------------------------------------------------------------
# OpenAI Responses provider 场景
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gpt-6-luna", "gpt-6-sol"],
)
def test_responses_does_not_enable_cache_warming_from_the_documented_ttl_alone(model_id: str) -> None:
    assert get_model("openai", model_id).prompt_cache is None


async def test_responses_does_not_set_prompt_cache_retention_when_pi_cache_retention_is_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
    model = get_model("openai", "gpt-4o-mini")
    capture = PayloadCapture()

    await _drain(stream(model, _raw_context(), OpenAIResponsesOptions(on_payload=capture)))

    assert capture.payload is not None
    assert capture.payload.get("prompt_cache_retention") is None


async def test_responses_sets_prompt_cache_retention_to_24h_when_pi_cache_retention_is_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
    model = get_model("openai", "gpt-4o-mini")
    capture = PayloadCapture()

    await _drain(stream(model, _raw_context(), OpenAIResponsesOptions(on_payload=capture)))

    assert capture.payload is not None
    assert capture.payload["prompt_cache_retention"] == "24h"


async def test_responses_sets_prompt_cache_retention_for_non_api_openai_base_url_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    base_model = get_model("openai", "gpt-4o-mini")
    proxy_model = _proxy(base_model, base_url="https://my-proxy.example.com/v1")
    capture = PayloadCapture()

    await _drain(
        stream_openai_responses(
            proxy_model,
            _context(),
            OpenAIResponsesOptions(api_key="sk-fake-key", on_payload=capture),
        )
    )

    assert capture.payload is not None
    assert capture.payload["prompt_cache_retention"] == "24h"


async def test_responses_omits_prompt_cache_retention_when_supports_long_cache_retention_is_false() -> None:
    model = _proxy(
        get_model("openai", "gpt-4o-mini"),
        compat=ModelCompat(supports_long_cache_retention=False),
    )
    capture = PayloadCapture()

    await _drain(
        stream_openai_responses(
            model,
            _context(),
            OpenAIResponsesOptions(
                api_key="sk-fake-key",
                cache_retention="long",
                session_id="session-compat-false",
                on_payload=capture,
            ),
        )
    )

    assert capture.payload is not None
    assert capture.payload.get("prompt_cache_retention") is None


async def test_responses_omits_prompt_cache_key_and_disables_implicit_writes_when_cache_retention_is_none() -> None:
    model = get_model("openai", "gpt-5.6-sol")
    capture = PayloadCapture()

    await _drain(
        stream_openai_responses(
            model,
            _context(),
            OpenAIResponsesOptions(
                api_key="sk-fake-key",
                cache_retention="none",
                session_id="session-1",
                on_payload=capture,
            ),
        )
    )

    assert capture.payload is not None
    assert capture.payload.get("prompt_cache_key") is None
    assert capture.payload.get("prompt_cache_retention") is None
    assert capture.payload.get("prompt_cache_options") == {"mode": "explicit"}


async def test_responses_omits_prompt_cache_options_for_models_that_reject_it() -> None:
    model = get_model("openai", "gpt-4o-mini")
    capture = PayloadCapture()

    await _drain(
        stream_openai_responses(
            model,
            _context(),
            OpenAIResponsesOptions(
                api_key="sk-fake-key",
                cache_retention="none",
                session_id="session-1",
                on_payload=capture,
            ),
        )
    )

    assert capture.payload is not None
    assert capture.payload.get("prompt_cache_key") is None
    assert capture.payload.get("prompt_cache_options") is None


@pytest.mark.parametrize(
    ("model_id", "retention", "cache_options"),
    [
        ("gpt-4o-mini", "24h", None),
        ("gpt-6-astra", None, {"ttl": "30m"}),
        ("gpt-6-sol", None, {"ttl": "30m"}),
        ("gpt-6-luna", None, {"ttl": "30m"}),
    ],
)
async def test_responses_uses_the_supported_long_cache_field(
    model_id: str, retention: str | None, cache_options: dict[str, Any] | None
) -> None:
    model = get_model("openai", model_id)
    capture = PayloadCapture()

    await _drain(
        stream_openai_responses(
            model,
            _context(),
            OpenAIResponsesOptions(
                api_key="sk-fake-key",
                cache_retention="long",
                session_id="session-2",
                on_payload=capture,
            ),
        )
    )

    assert capture.payload is not None
    assert capture.payload.get("prompt_cache_key") == "session-2"
    assert capture.payload.get("prompt_cache_retention") == retention
    assert capture.payload.get("prompt_cache_options") == cache_options
