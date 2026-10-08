"""移植自 ``openai-responses-chatgpt-sign-in.test.ts``。"""

from __future__ import annotations

import dataclasses

import pytest

from pi_ai.api.openai_responses import OpenAIResponsesOptions, stream
from pi_ai.types import Model, ModelCompat, ModelCost, TextContent, UserMessage

from openai_port_harness import FakeFetch, json_response, responses_context, stream_result

_MODEL = Model(
    id="gpt-5-mini",
    name="GPT-5 Mini",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=400_000,
    max_tokens=128_000,
)


def _context():
    return responses_context([UserMessage(content=[TextContent(type="text", text="hi")], timestamp=0)])


async def _capture_payload(api_key: str, request_model: Model = _MODEL) -> dict:
    captured: dict = {}

    def on_payload(params, _model):
        captured.update(params)

    fetch = FakeFetch(json_response({"error": {"message": "boom"}}, status=500))
    await stream_result(
        stream(
            request_model,
            _context(),
            OpenAIResponsesOptions(
                api_key=api_key,
                max_tokens=1000,
                temperature=0.5,
                cache_retention="long",
                on_payload=on_payload,
                fetch=fetch,
            ),
        )
    )
    assert captured, "Request payload was not captured"
    return captured


async def test_omits_request_fields_that_token_sharing_rejects():
    payload = await _capture_payload("chatgpt-access-token")

    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert payload.get("prompt_cache_retention") is None


async def test_omits_prompt_cache_options_on_models_with_explicit_prompt_cache_mode():
    explicit_cache_model = dataclasses.replace(
        _MODEL, compat=ModelCompat(supports_explicit_prompt_cache_mode=True)
    )

    sign_in_payload = await _capture_payload("chatgpt-access-token", explicit_cache_model)
    api_key_payload = await _capture_payload("sk-proj-test", explicit_cache_model)

    assert sign_in_payload.get("prompt_cache_options") is None
    assert api_key_payload["prompt_cache_options"] == {"ttl": "30m"}


@pytest.mark.parametrize(
    ("api_key", "request_model"),
    [
        ("sk-proj-test", _MODEL),
        ("gateway-key", dataclasses.replace(_MODEL, base_url="https://gateway.example.com/v1")),
    ],
    ids=["openai-api-keys", "other-openai-compatible-endpoints"],
)
async def test_keeps_those_fields(api_key, request_model):
    payload = await _capture_payload(api_key, request_model)

    assert payload["max_output_tokens"] == 1000
    assert payload["temperature"] == 0.5
    assert payload["prompt_cache_retention"] == "24h"
