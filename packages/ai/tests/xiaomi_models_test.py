"""Xiaomi MiMo 模型目录内容测试。

移植自 ``packages/ai/test/xiaomi-models.test.ts``。
裁剪后只保留 ``xiaomi`` 本体目录，token-plan 变体已随 refactor 删除。
"""

from __future__ import annotations

import pytest

from pi_ai.compat import get_models

XIAOMI_PROVIDERS = ["xiaomi"]
DEPRECATED_MODEL_IDS = ["mimo-v2-flash", "mimo-v2-omni", "mimo-v2-pro"]
CATALOG_MODEL_IDS = [
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "mimo-v2.5-pro-ultraspeed",
    "mimo-v2.6-flash",
    "mimo-v2.6-pro",
    "mimo-v2.6-pro-ultraspeed",
]


@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
def test_omits_deprecated_models(provider):
    model_ids = [model.id for model in get_models(provider)]
    for model_id in DEPRECATED_MODEL_IDS:
        assert model_id not in model_ids


@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
def test_keeps_catalog_models(provider):
    model_ids = [model.id for model in get_models(provider)]
    for model_id in CATALOG_MODEL_IDS:
        assert model_id in model_ids


@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
def test_lists_exactly_the_catalog_models(provider):
    model_ids = {model.id for model in get_models(provider)}
    assert model_ids == set(CATALOG_MODEL_IDS)
