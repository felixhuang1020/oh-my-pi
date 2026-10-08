"""移植自 ``openai-responses-usage-limit.test.ts``。"""

from __future__ import annotations

from pi_ai.api.openai_responses import OpenAIResponsesOptions, stream
from pi_ai.types import Model, ModelCost, TextContent, UserMessage

from openai_port_harness import FakeFetch, json_response, responses_context, sse_response, stream_result

_USAGE_LIMIT_ERROR = {
    "code": "subscription_sharing_usage_limit_exceeded",
    "message": "Usage limit reached.",
}

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


async def _get_error_message(fetch: FakeFetch) -> str | None:
    result = await stream_result(
        stream(_MODEL, _context(), OpenAIResponsesOptions(api_key="test", fetch=fetch))
    )
    assert result.stop_reason == "error"
    return result.error_message


async def test_links_to_chatgpt_usage_when_the_request_is_rejected():
    fetch = FakeFetch(
        json_response(
            {"error": {**_USAGE_LIMIT_ERROR, "type": "rate_limit_error"}},
            status=429,
            headers={"content-type": "application/json"},
        )
    )

    error_message = await _get_error_message(fetch)

    assert "subscription_sharing_usage_limit_exceeded" in error_message
    assert "Check your ChatGPT usage: https://chatgpt.com/settings/usage" in error_message


async def test_links_to_chatgpt_usage_when_the_stream_fails():
    event = {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {"id": "resp_failed", "status": "failed", "error": _USAGE_LIMIT_ERROR},
    }
    fetch = FakeFetch(sse_response([("response.failed", event)]))

    error_message = await _get_error_message(fetch)

    assert "subscription_sharing_usage_limit_exceeded: Usage limit reached." in error_message
    assert "Check your ChatGPT usage: https://chatgpt.com/settings/usage" in error_message
