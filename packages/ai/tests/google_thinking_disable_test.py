"""移植 ``packages/ai/test/google-thinking-disable.test.ts``。

这里每个上游用例都是带门控的 live E2E（``describe.skipIf(!process.env.X)``），
会发起真实 API 调用并测量 thinking 是否被禁用。移植版保留完整的断言辅助函数，
并给每个用例加上 ``pytest.mark.skip`` 及其所需的环境变量说明，
确保没有测试会意外访问网络。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from pi_ai.compat import get_model, stream_simple
from pi_ai.types import Context, SimpleStreamOptions, UserMessage


@dataclass
class RunResult:
    thinking_event_count: int
    thinking_char_count: int
    text: str
    output_tokens: int
    content_types: list[str]


@dataclass
class DisableExpectations:
    request_options: dict[str, Any] = field(default_factory=dict)
    min_pongs: int = 35
    max_output_tokens: int | None = None


def make_context() -> Context:
    return Context(
        system_prompt="You are a precise assistant. Follow the requested output format exactly.",
        messages=[
            UserMessage(
                content=(
                    "Before replying, carefully solve 36863 * 5279 internally. Then reply with the word pong "
                    "repeated exactly 40 times, separated by single spaces. Do not add any other text."
                ),
                timestamp=0,
            )
        ],
    )


def count_pongs(text: str) -> int:
    return len(re.findall(r"\bpong\b", text, flags=re.IGNORECASE))


async def run_without_reasoning(model: Any, options: dict[str, Any] | None = None) -> RunResult:
    values = {"max_tokens": 160, "temperature": 0, **(options or {})}
    stream_object = stream_simple(model, make_context(), SimpleStreamOptions(**values))

    thinking_event_count = 0
    thinking_char_count = 0

    async for event in stream_object:
        if event.type in ("thinking_start", "thinking_end"):
            thinking_event_count += 1
        if event.type == "thinking_delta":
            thinking_event_count += 1
            thinking_char_count += len(event.delta)

    response = await stream_object.result()
    assert response.stop_reason == "stop", response.error_message

    text = "".join(block.text for block in response.content if block.type == "text").strip()

    return RunResult(
        thinking_event_count=thinking_event_count,
        thinking_char_count=thinking_char_count,
        text=text,
        output_tokens=response.usage.output,
        content_types=[block.type for block in response.content],
    )


async def expect_thinking_disabled_e2e(model: Any, expectations: DisableExpectations | None = None) -> None:
    expectations = expectations or DisableExpectations()
    result = await run_without_reasoning(model, expectations.request_options)

    assert result.thinking_event_count == 0
    assert result.thinking_char_count == 0
    assert "thinking" not in result.content_types
    assert count_pongs(result.text) >= expectations.min_pongs
    if expectations.max_output_tokens is not None:
        assert result.output_tokens < expectations.max_output_tokens


# ---------------------------------------------------------------------------------
# 网络 E2E（已跳过）
# ---------------------------------------------------------------------------------


def vertex_request_options() -> dict[str, Any]:
    """上游的 ``vertexOptions``：配置了 API key 时显式传入。

    project/location 由 adapter 自身从环境读取，与移植的
    ``resolve_project``/``resolve_location`` 完全一致。
    """
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    return {"api_key": api_key} if api_key else {}


@pytest.mark.skip(reason="network E2E: requires ANTHROPIC_API_KEY")
async def test_anthropic_disables_thinking_for_budget_based_reasoning_models() -> None:
    await expect_thinking_disabled_e2e(
        get_model("anthropic", "claude-sonnet-4-5"),
        DisableExpectations(request_options={"max_tokens": 320, "temperature": 0}),
    )


@pytest.mark.skip(reason="network E2E: requires ANTHROPIC_API_KEY")
async def test_anthropic_disables_thinking_for_adaptive_reasoning_models() -> None:
    await expect_thinking_disabled_e2e(
        get_model("anthropic", "claude-sonnet-4-6"),
        DisableExpectations(request_options={"max_tokens": 320, "temperature": 0}),
    )


@pytest.mark.skip(reason="network E2E: requires GEMINI_API_KEY")
async def test_google_disables_thinking_for_gemini_2_5() -> None:
    await expect_thinking_disabled_e2e(get_model("google", "gemini-2.5-flash"))


@pytest.mark.skip(reason="network E2E: requires GEMINI_API_KEY")
async def test_google_disables_thinking_for_gemini_3x() -> None:
    await expect_thinking_disabled_e2e(get_model("google", "gemini-3-flash-preview"))


@pytest.mark.skip(reason="network E2E: requires GEMINI_API_KEY")
async def test_google_does_not_error_when_thinking_is_off_for_gemini_3_1_pro() -> None:
    await expect_thinking_disabled_e2e(
        get_model("google", "gemini-3.1-pro-preview"),
        DisableExpectations(request_options={"max_tokens": 512}, min_pongs=20),
    )


@pytest.mark.skip(
    reason=(
        "network E2E: requires GOOGLE_CLOUD_API_KEY or "
        "GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT + GOOGLE_CLOUD_LOCATION"
    )
)
async def test_vertex_disables_thinking_for_gemini_2_5() -> None:
    await expect_thinking_disabled_e2e(
        get_model("google-vertex", "gemini-2.5-flash"),
        DisableExpectations(request_options=vertex_request_options()),
    )


@pytest.mark.skip(
    reason=(
        "network E2E: requires GOOGLE_CLOUD_API_KEY or "
        "GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT + GOOGLE_CLOUD_LOCATION"
    )
)
async def test_vertex_disables_thinking_for_gemini_3x() -> None:
    await expect_thinking_disabled_e2e(
        get_model("google-vertex", "gemini-3-flash-preview"),
        DisableExpectations(request_options=vertex_request_options()),
    )


@pytest.mark.skip(reason="network E2E: requires OPENAI_API_KEY")
async def test_openai_disables_thinking_for_responses_reasoning_models() -> None:
    await expect_thinking_disabled_e2e(
        get_model("openai", "gpt-5.4-mini"),
        DisableExpectations(request_options={"temperature": None}),
    )
