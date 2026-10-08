"""移植自 ``openai-responses-cache-affinity-e2e.test.ts``。

上游套件仅在设置了 ``OPENAI_API_KEY`` 时才运行这个真实 OpenAI 测试。移植版保留同样的
开关：它需要真实凭据与网络访问，而测试套件禁止联网。
"""

from __future__ import annotations

import os

import pytest

from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.compat import complete, get_model
from pi_ai.types import Context, UserMessage

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="live OpenAI e2e; requires OPENAI_API_KEY and network access",
)


async def test_handles_direct_openai_responses_requests_with_aligned_cache_affinity_identifiers():
    model = get_model("openai", "gpt-5.4")
    session_id = "0195d6e4-4cf9-7f44-a2d8-f8f7f49ee9d3"
    context = Context(
        system_prompt="You are a helpful assistant. Reply exactly as requested.",
        messages=[
            UserMessage(
                content="Reply with exactly: openai cache affinity e2e success", timestamp=0
            )
        ],
    )

    response = await complete(
        model,
        context,
        OpenAIResponsesOptions(api_key=os.environ["OPENAI_API_KEY"], session_id=session_id),
    )

    assert response.stop_reason != "error", response.error_message
    assert response.error_message is None
    assert "openai cache affinity e2e success" in "".join(
        block.text for block in response.content if block.type == "text"
    )
