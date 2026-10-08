"""已生成模型目录的测试。"""

from __future__ import annotations

from pi_ai.model_catalog import flatten_chat_model_catalog
from pi_ai.providers.catalog import (
    available_provider_catalogs,
    chat_model_catalog,
    load_provider_data,
    provider_models,
)
from pi_ai.types import Model, ModelCompat, ModelCost


def test_every_generated_catalog_is_discoverable():
    catalogs = available_provider_catalogs()
    assert len(catalogs) == 9
    assert "anthropic" in catalogs
    assert "openai" in catalogs


def test_hidden_manifest_is_not_reported_as_a_provider():
    # ``Path.glob`` 会匹配以点开头的文件，因此 ``data/.manifest.json`` 必须被过滤掉。
    assert not [catalog for catalog in available_provider_catalogs() if catalog.startswith(".")]
    assert all("/" not in catalog for catalog in available_provider_catalogs())


def test_anthropic_catalog_flattens_to_chat_models():
    catalog = chat_model_catalog("anthropic")
    assert len(catalog) > 5
    model = catalog["claude-fable-5"]
    assert isinstance(model, Model)
    assert model.provider == "anthropic"
    assert model.api == "anthropic-messages"
    assert model.base_url == "https://api.anthropic.com"
    assert model.reasoning is True
    assert model.context_window > 0
    assert model.max_tokens > 0
    assert model.cost.input > 0


def test_nested_compat_and_limits_are_deserialized():
    model = chat_model_catalog("anthropic")["claude-fable-5"]
    assert isinstance(model.compat, ModelCompat)
    assert model.compat.supports_mid_convo_system_messages is True
    assert model.compat.allowed_fallback_models
    fallback = model.compat.allowed_fallback_models[0]
    assert fallback.provider == "anthropic"
    assert fallback.model
    assert fallback.cost.input > 0
    assert model.input_limits is not None


def test_thinking_level_map_preserves_null_entries():
    model = chat_model_catalog("anthropic")["claude-fable-5"]
    assert model.thinking_level_map is not None
    assert "off" in model.thinking_level_map
    assert model.thinking_level_map["off"] is None
    assert model.thinking_level_map["xhigh"] == "xhigh"


def test_costs_are_model_cost_instances_with_tiers():
    model = chat_model_catalog("openai")["gpt-5.4"]
    assert isinstance(model.cost, ModelCost)
    assert model.cost.output > 0
    if model.cost.tiers:
        assert all(tier.input_tokens_above >= 0 for tier in model.cost.tiers)


def test_completions_providers_are_served_by_the_responses_api():
    for provider in ("deepseek", "moonshotai", "moonshotai-cn", "xiaomi"):
        models = chat_model_catalog(provider)
        assert models
        assert {model.api for model in models.values()} == {"openai-responses"}


def test_provider_models_returns_every_type():
    ids = {model.id for model in provider_models("openai")}
    assert ids
    types = {model.type for model in provider_models("openai")}
    assert types == {"chat"}


def test_unknown_provider_yields_empty_catalog():
    assert load_provider_data("does-not-exist") == {}
    assert chat_model_catalog("does-not-exist") == {}


def test_flatten_helpers_filter_by_type():
    groups = {
        "api-a": {
            "chat:one": {"id": "one", "type": "chat", "name": "One", "api": "api-a", "provider": "p"},
            "image:two": {"id": "two", "type": "image", "name": "Two", "api": "api-a", "provider": "p"},
            "classifier:three": {
                "id": "three",
                "type": "classifier",
                "name": "Three",
                "api": "api-a",
                "provider": "p",
            },
        }
    }
    assert list(flatten_chat_model_catalog("p", groups)) == ["one"]


def test_entry_without_a_type_is_treated_as_chat():
    groups = {"api-a": {"one": {"id": "one", "name": "One", "api": "api-a", "provider": "p"}}}
    assert list(flatten_chat_model_catalog("p", groups)) == ["one"]


def test_entries_without_an_id_are_dropped():
    groups = {"api-a": {"bad": {"name": "No id", "type": "chat"}}}
    assert flatten_chat_model_catalog("p", groups) == {}


def test_catalog_getters_are_cached_but_models_are_reusable():
    first = chat_model_catalog("anthropic")
    second = chat_model_catalog("anthropic")
    assert first is second
