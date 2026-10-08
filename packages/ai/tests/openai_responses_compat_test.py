"""移植自 ``openai-responses-compat.test.ts``。

上游测试对 ``globalThis.fetch`` 打桩以捕获请求头/请求体，并返回 ``data: [DONE]``
（使流在请求体构建完成后报错）。移植版以同样方式使用 :class:`FakeFetch`，
并读取记录下来的 ``httpx.Request``。
"""

from __future__ import annotations

import dataclasses

import pytest

from pi_ai.api.openai_responses import OpenAIResponsesOptions, stream
from pi_ai.compat import get_model
from pi_ai.types import JsonSchemaConstrainedSampling, Model, ModelCompat, Tool

from openai_port_harness import (
    FakeFetch,
    collect_events,
    responses_context,
    sse_response,
    user_message,
)

_DONE_ONLY = sse_response([])


def _context(*, system_prompt: str = "sys", tools=None):
    return responses_context(
        [user_message("hi", timestamp=1)], system_prompt=system_prompt, tools=tools
    )


async def _capture(model: Model, *, context=None, **option_kwargs):
    """发起一次请求，返回捕获到的请求体及记录的请求头。"""
    captured: dict = {}

    def on_payload(params, _model):
        captured.update(params)

    fetch = FakeFetch(_DONE_ONLY)
    await collect_events(
        stream(
            model,
            context if context is not None else _context(),
            OpenAIResponsesOptions(
                api_key="sk-test-key", fetch=fetch, on_payload=on_payload, **option_kwargs
            ),
        )
    )
    return captured, fetch.last_request.headers


# --------------------------------------------------------------------------------------
# provider 默认值
# --------------------------------------------------------------------------------------


async def test_omits_reasoning_when_no_reasoning_is_requested():
    model = get_model("openai", "gpt-5-mini")
    payload, _headers = await _capture(model)

    assert payload is not None
    assert "reasoning" not in payload


async def test_forwards_required_tool_choice():
    model = get_model("openai", "gpt-5.4")
    context = _context(
        system_prompt=None,
        tools=[Tool(name="ping", description="Ping", parameters={"type": "object", "properties": {"value": {"type": "string"}}})],
    )
    payload, _headers = await _capture(model, context=context, tool_choice="required")

    assert payload["tool_choice"] == "required"
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "ping"


async def test_sets_strict_mode_explicitly_for_openai_responses_tools():
    model = get_model("openai", "gpt-5.4")
    context = _context(
        system_prompt=None,
        tools=[
            Tool(
                name="ordinary",
                description="An ordinary tool",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "offset": {"type": "number"}},
                    "required": ["path"],
                },
            ),
            Tool(
                name="constrained",
                description="A constrained tool",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}},
                constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
            ),
        ],
    )
    payload, _headers = await _capture(model, context=context)

    assert model.compat.supports_strict_mode is True
    ordinary, constrained = payload["tools"]
    assert ordinary["name"] == "ordinary"
    assert ordinary["strict"] is False
    assert constrained["name"] == "constrained"
    assert constrained["strict"] is True


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-6-sol",
        "gpt-6-luna",
    ],
)
async def test_sends_none_reasoning_effort_for_openai_models_when_no_reasoning_is_requested(model_id):
    model = get_model("openai", model_id)
    payload, _headers = await _capture(model)

    assert payload["reasoning"] == {"effort": "none"}


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro"],
)
async def test_omits_reasoning_effort_for_openai_models_when_off_is_unsupported(model_id):
    model = get_model("openai", model_id)
    payload, _headers = await _capture(model)

    assert "reasoning" not in payload


async def test_sets_cache_affinity_headers_for_official_openai_responses_requests_with_a_session_id():
    _payload, headers = await _capture(get_model("openai", "gpt-5.4"), session_id="session-123")

    assert headers.get("session_id") == "session-123"
    assert headers.get("x-client-request-id") == "session-123"


async def test_clamps_prompt_cache_key_to_openais_64_character_limit():
    payload, _headers = await _capture(get_model("openai", "gpt-5.4"), session_id="x" * 67)

    assert payload["prompt_cache_key"] == "x" * 64


async def test_sets_cache_affinity_headers_for_proxy_openai_responses_requests_with_a_session_id():
    proxy_model = dataclasses.replace(
        get_model("openai", "gpt-5.4"),
        provider="proxy",
        base_url="https://proxy.example.com/v1",
    )
    _payload, headers = await _capture(proxy_model, session_id="session-123")

    assert headers.get("session_id") == "session-123"
    assert headers.get("x-client-request-id") == "session-123"


async def test_uses_openrouter_session_affinity_header_when_configured():
    proxy_model = dataclasses.replace(
        get_model("openai", "gpt-5.4"),
        provider="proxy",
        base_url="https://proxy.example.com/v1",
        compat=ModelCompat(session_affinity_format="openrouter"),
    )
    payload, headers = await _capture(proxy_model, session_id="session-proxy")

    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") is None
    assert headers.get("x-session-id") == "session-proxy"
    assert payload.get("session_id") is None
    assert payload["prompt_cache_key"] == "session-proxy"


async def test_auto_detects_openrouter_session_affinity_header_for_openrouter_responses_endpoints():
    open_router_model = dataclasses.replace(
        get_model("openai", "gpt-5.4"),
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )
    payload, headers = await _capture(open_router_model, session_id="session-openrouter")

    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") is None
    assert headers.get("x-session-id") == "session-openrouter"
    assert payload.get("session_id") is None
    assert payload["prompt_cache_key"] == "session-openrouter"


async def test_uses_openai_no_session_format_when_configured():
    proxy_model = dataclasses.replace(
        get_model("openai", "gpt-5.4"),
        provider="proxy",
        base_url="https://proxy.example.com/v1",
        compat=ModelCompat(session_affinity_format="openai-nosession"),
    )
    payload, headers = await _capture(proxy_model, session_id="session-proxy")

    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") == "session-proxy"
    assert headers.get("x-session-id") is None
    assert payload.get("session_id") is None
    assert payload["prompt_cache_key"] == "session-proxy"


async def test_can_omit_openai_session_id_header_while_preserving_other_affinity_data():
    proxy_model = dataclasses.replace(
        get_model("openai", "gpt-5.4"),
        provider="proxy",
        base_url="https://proxy.example.com/v1",
        compat=ModelCompat(session_affinity_format="openai-nosession"),
    )
    payload, headers = await _capture(proxy_model, session_id="session-123")

    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") == "session-123"
    assert payload["prompt_cache_key"] == "session-123"


async def test_lets_explicit_headers_override_the_default_openai_cache_affinity_headers():
    _payload, headers = await _capture(
        get_model("openai", "gpt-5.4"),
        session_id="session-123",
        headers={"session_id": "override-session", "x-client-request-id": "override-request"},
    )

    assert headers["session_id"] == "override-session"
    assert headers["x-client-request-id"] == "override-request"


async def test_omits_openai_cache_affinity_headers_when_cache_retention_is_none():
    _payload, headers = await _capture(
        get_model("openai", "gpt-5.4"), cache_retention="none", session_id="session-123"
    )

    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") is None


def _usage_event(response_service_tier: str, token_count: int = 100_000) -> dict:
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "service_tier": response_service_tier,
            "usage": {
                "input_tokens": token_count,
                "output_tokens": token_count,
                "total_tokens": token_count * 2,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    }


@pytest.mark.parametrize(
    ("model_id", "service_tier", "response_service_tier", "multiplier"),
    [
        ("gpt-5.4", "priority", "priority", 2),
        ("gpt-5.5", "priority", "priority", 2.5),
        ("gpt-5.5", "flex", "flex", 0.5),
        # GPT-6 系列即使请求 "priority" 也会上报 "fast" 模式（#10034）
        ("gpt-6-luna", "priority", "fast", 2),
        ("gpt-6-luna", "fast", "fast", 2),
    ],
)
async def test_applies_service_tier_cost_multiplier(model_id, service_tier, response_service_tier, multiplier):
    model = get_model("openai", model_id)
    token_count = 100_000
    token_scale = token_count / 1_000_000
    fetch = FakeFetch(sse_response([_usage_event(response_service_tier, token_count)]))
    result = await stream(
        model,
        _context(),
        OpenAIResponsesOptions(api_key="sk-test-key", fetch=fetch, service_tier=service_tier),
    ).result()

    assert result.usage.cost.input == pytest.approx(model.cost.input * multiplier * token_scale, abs=1e-12)
    assert result.usage.cost.output == pytest.approx(model.cost.output * multiplier * token_scale, abs=1e-12)
    assert result.usage.cost.total == pytest.approx(
        (model.cost.input + model.cost.output) * multiplier * token_scale, abs=1e-12
    )


# --------------------------------------------------------------------------------------
# max_output_tokens 兼容
# --------------------------------------------------------------------------------------


async def test_sends_max_output_tokens_by_default():
    payload, _headers = await _capture(get_model("openai", "gpt-5.4"), max_tokens=1024)

    assert payload["max_output_tokens"] == 1024


async def test_omits_max_output_tokens_when_supports_max_output_tokens_is_false():
    base_model = get_model("openai", "gpt-5.4")
    model = dataclasses.replace(
        base_model, compat=dataclasses.replace(base_model.compat, supports_max_output_tokens=False)
    )
    payload, _headers = await _capture(model, max_tokens=1024)

    assert payload.get("max_output_tokens") is None
