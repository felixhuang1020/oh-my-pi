"""移植自 ``xhigh.test.ts``。

上游将整个文件包在 ``describe.skipIf(!process.env.OPENAI_API_KEY)`` 中，驱动真实
OpenAI API。"不支持 xhigh" 的用例失败仅因 provider 拒绝请求：移植的 OpenAI 适配器
会原样转发 ``reasoning_effort="xhigh"``（``_build_params`` 经 ``thinking_level_map``
映射层级且从不抛错），因此没有可断言的离线校验。重构后 ``openai-completions`` 协议
被删除，原先对应的第三个用例（completions 下 xhigh 报错）一并移除，剩余两个 Responses
用例全部跳过，其测试体保留在下方但永不运行。
"""

from __future__ import annotations

import pytest

from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.compat import get_model, stream
from pi_ai.types import Context, UserMessage

_SKIP_REASON = (
    "live provider E2E: upstream gates on process.env.OPENAI_API_KEY; the Python suite is offline"
)


def _make_context() -> Context:
    return Context(messages=[UserMessage(content="What is 12 + 34? Think step by step.", timestamp=0)])


@pytest.mark.skip(reason=_SKIP_REASON)
async def test_gpt_5_5_supports_xhigh_with_openai_responses() -> None:
    model = get_model("openai", "gpt-5.5")
    stream_object = stream(model, _make_context(), OpenAIResponsesOptions(reasoning_effort="xhigh"))
    has_thinking = False

    async for event in stream_object:
        if event.type in ("thinking_start", "thinking_delta"):
            has_thinking = True

    response = await stream_object.result()
    assert response.stop_reason == "stop", f"Error: {response.error_message}"
    assert any(block.type == "text" for block in response.content) is True
    assert has_thinking or any(block.type == "thinking" for block in response.content)


@pytest.mark.skip(reason=_SKIP_REASON)
async def test_gpt_5_mini_errors_with_openai_responses_when_using_xhigh() -> None:
    model = get_model("openai", "gpt-5-mini")
    stream_object = stream(model, _make_context(), OpenAIResponsesOptions(reasoning_effort="xhigh"))

    async for _ in stream_object:
        pass

    response = await stream_object.result()
    assert response.stop_reason == "error"
    assert "xhigh" in (response.error_message or "")
