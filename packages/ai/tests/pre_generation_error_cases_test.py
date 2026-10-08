"""移植 ``packages/ai/test/pre-generation-error.test.ts``（"直接 API 认证"）。

上游测试断言每个适配器的 ``streamSimple`` 在缺少 API key 时**同步**抛出异常。在本移植
中，模块级直接函数 ``stream_simple`` 行为完全一致；而惰性注册表分发路径
（``pi_ai/api/lazy.py``）会把同一失败转为终结性的 ``error`` 事件，由本文件最后一个测试
覆盖。
"""

from __future__ import annotations

import importlib

import pytest

from pi_ai.types import Context, Model, ModelCost
from pi_ai.utils.transcript import normalize_context

#: 上游测试覆盖的每个适配器：``(api id, 实现它的模块)``。
#: 重构后仅保留四种 API；``openai-completions`` 及其余已删除适配器不再纳入。
ADAPTERS = [
    ("anthropic-messages", "pi_ai.api.anthropic_messages"),
    ("google-generative-ai", "pi_ai.api.google_generative_ai"),
    ("openai-responses", "pi_ai.api.openai_responses"),
]


def _model(api: str) -> Model:
    return Model(
        id="test-model",
        name="Test",
        api=api,
        provider="test-provider",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=1_000,
        max_tokens=100,
    )


CONTEXT = normalize_context(Context(messages=[]))


@pytest.mark.parametrize(
    ("api", "module_name"),
    ADAPTERS,
    ids=[api for api, _ in ADAPTERS],
)
def test_throws_synchronously_when_auth_is_missing(api: str, module_name: str) -> None:
    module = importlib.import_module(module_name)
    with pytest.raises(RuntimeError, match="No API key for provider: test-provider"):
        module.stream_simple(_model(api), CONTEXT, None)


async def test_lazy_registry_dispatch_encodes_missing_auth_as_a_stream_error() -> None:
    """移植差异：注册表分发把同样的失败以内联事件的形式上报。"""
    from pi_ai.compat import get_api_provider

    provider = get_api_provider("anthropic-messages")
    assert provider is not None
    model = Model(
        id="test-model",
        name="Test",
        api="anthropic-messages",
        provider="test-provider",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=1_000,
        max_tokens=100,
    )
    message = await provider.stream_simple(model, CONTEXT, None).result()
    assert message.stop_reason == "error"
    assert message.error_message == "No API key for provider: test-provider"
