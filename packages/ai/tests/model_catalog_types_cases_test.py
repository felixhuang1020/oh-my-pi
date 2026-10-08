"""移植 ``packages/ai/test/model-catalog-types.test.ts``。

上游文件主体是 ``expectTypeOf`` 字面量类型断言，运行时在 Python 中没有等价物。
文件中夹带的运行时断言在此移植；纯类型断言只做报告，不伪造。
裁剪后九家 provider 中保留了 ``openai`` 与 ``deepseek`` 这两组 Responses API 目录。
"""

from __future__ import annotations

from pi_ai.providers.catalog import chat_model_catalog


def _openai():
    return chat_model_catalog("openai")


def _deepseek():
    return chat_model_catalog("deepseek")


def test_derives_model_api_id_and_provider_literals_from_grouped_model_data():
    # 上游这里只断言 TypeScript 字面量类型；api/id/provider 的值作为运行时证据断言。
    # 字面量*类型*在 Python 中无法表达。
    deepseek = _deepseek()
    for model_id in ("deepseek-flash", "deepseek-v4-pro"):
        model = deepseek[model_id]
        assert model.api == "openai-responses"
        assert model.id == model_id
        assert model.provider == "deepseek"


def test_routes_openai_gpt_5_6_sol_through_the_responses_api():
    assert _openai()["gpt-5.6-sol"].api == "openai-responses"


# 回归测试，见 https://github.com/earendil-works/pi/issues/9209
def test_routes_all_openai_gpt_models_through_the_responses_api():
    openai = _openai()
    gpt_models = [model for model in openai.values() if model.id.startswith("gpt-")]
    assert len(gpt_models) > 0
    assert all(model.api == "openai-responses" for model in gpt_models)
    assert openai["gpt-6-astra"].api == "openai-responses"
    for model_id in ("gpt-6-sol", "gpt-6-luna"):
        model = openai[model_id]
        assert model.api == "openai-responses"
        assert model.context_window == 272000
        assert model.max_tokens == 128000
        assert model.thinking_level_map is not None
        assert model.thinking_level_map.get("off") == "none"
        assert model.thinking_level_map.get("max") == "max"
