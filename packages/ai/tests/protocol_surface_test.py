"""本项目裁剪后的协议表面不变量。

这些断言把 fork 的收窄固定下来：四个 KnownApi、九家 KnownProvider、只有 chat
模型、只有 api-key 认证。任何一处回退都会在此失败。
"""

from __future__ import annotations

from pi_ai.models_generated import BUILTIN_PROVIDER_IDS
from pi_ai.providers.all import all_providers, builtin_providers, get_builtin_providers
from pi_ai.types import KNOWN_APIS, KNOWN_PROVIDERS

#: 本 fork 保留的四条 wire protocol。
EXPECTED_APIS = ("openai-responses", "anthropic-messages", "google-generative-ai", "pi-messages")
#: 本 fork 保留的九家 provider。
EXPECTED_PROVIDERS = (
    "anthropic",
    "google",
    "openai",
    "deepseek",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "xiaomi",
)


def test_known_apis_are_exactly_the_four_supported_protocols():
    assert tuple(KNOWN_APIS) == EXPECTED_APIS


def test_known_providers_are_exactly_the_nine_builtins():
    assert tuple(KNOWN_PROVIDERS) == EXPECTED_PROVIDERS


def test_builtin_providers_match_known_providers():
    assert [provider.id for provider in all_providers()] == list(EXPECTED_PROVIDERS)
    assert [provider.id for provider in builtin_providers()] == list(EXPECTED_PROVIDERS)
    assert set(get_builtin_providers()) == set(EXPECTED_PROVIDERS)
    assert tuple(BUILTIN_PROVIDER_IDS) == EXPECTED_PROVIDERS


def test_every_builtin_model_is_a_chat_model_using_a_known_api():
    for provider in all_providers():
        for model in provider.get_models():
            assert model.type == "chat"
            assert model.api in KNOWN_APIS


def test_builtin_providers_only_use_api_key_authentication():
    for provider in all_providers():
        assert provider.auth.api_key is not None
        # OAuth 已随本 fork 删除：ProviderAuth 不再带 oauth 字段。
        assert not hasattr(provider.auth, "oauth")


def test_completions_providers_were_migrated_to_the_responses_api():
    models = {provider.id: {model.api for model in provider.get_models()} for provider in all_providers()}
    for provider in ("deepseek", "moonshotai", "moonshotai-cn", "xiaomi"):
        assert models[provider] == {"openai-responses"}
    # 任何内置 provider 都不再使用 openai-completions。
    assert all("openai-completions" not in apis for apis in models.values())
