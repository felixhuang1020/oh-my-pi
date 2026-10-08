"""移植自 ``packages/ai/test/model-types.test.ts``。

覆盖运行时模型类型收窄（:func:`pi_ai.utils.model_operations.get_model_type`、
:func:`is_model_type`）、只实现 ``getModels`` 的旧式 provider、内置目录 getter，
以及 ``refresh`` 时丢弃未知类型的已存储/已拉取模型。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pi_ai.auth.types import AuthResult, ModelAuth, ProviderAuth
from pi_ai.compat import get_model as get_compat_model
from pi_ai.models import (
    CreateModelsOptions,
    CreateProviderOptions,
    ModelsRefreshOptions,
    create_models,
    create_provider,
    has_api,
    models_are_equal,
)
from pi_ai.models_store import InMemoryModelsStore, ModelsStoreEntry
from pi_ai.providers.all import get_builtin_model
from pi_ai.providers.faux import (
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_provider,
)
from pi_ai.types import Context, Model, ModelCost, UserMessage
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.model_operations import get_model_type, is_model_type


def chat_model(provider: str, id: str) -> Model:
    return Model(
        id=id,
        name=id,
        api="test-chat",
        provider=provider,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )


class _NoAuth:
    name = "Test"

    async def resolve(self, **_: object) -> AuthResult:
        return AuthResult(auth=ModelAuth())


class _ChatStreams:
    def stream(self, *args: object, **kwargs: object) -> AssistantMessageEventStream:
        return AssistantMessageEventStream()

    def stream_simple(self, *args: object, **kwargs: object) -> AssistantMessageEventStream:
        return AssistantMessageEventStream()


class _HandwrittenProvider:
    """上游手写的 ``Provider`` 字面量：没有 ``getAllModels``。"""

    def __init__(self, faux) -> None:  # noqa: ANN001 - 测试假对象
        self.id = "handwritten"
        self.name = "Handwritten"
        self.auth = faux.provider.auth
        self._faux = faux

    def get_models(self):
        return self._faux.models

    def stream(self, *args: object, **kwargs: object):
        return self._faux.provider.stream(*args, **kwargs)

    def stream_simple(self, *args: object, **kwargs: object):
        return self._faux.provider.stream_simple(*args, **kwargs)


def _handwritten():
    faux = faux_provider(RegisterFauxProviderOptions(provider="handwritten"))
    models = create_models()
    models.set_provider(_HandwrittenProvider(faux))
    return faux, models


async def test_chat_models_without_a_type_work_through_a_handwritten_provider_without_get_all_models() -> None:
    faux, models = _handwritten()

    model = models.get_model("handwritten", faux.models[0].id)
    assert model is not None
    # TypeScript 对 chat 模型的 `type` 保持 undefined；移植版的 dataclass
    # 会把缺失的 type 归一化为 "chat"（BaseModel.type 默认值）。
    assert model.type == "chat"
    assert get_model_type(model) == "chat"
    assert has_api(model, model.api) is True
    assert models_are_equal(model, replace(model, type="chat")) is True
    assert models_are_equal(model, replace(model, type="embedding")) is False

    faux.set_responses([faux_assistant_message("hi")])
    result = await models.complete(
        model, Context(messages=[UserMessage(content="hi", timestamp=0)])
    )
    assert result.stop_reason == "stop"


def test_handwritten_provider_without_get_all_models_lists_all_models() -> None:
    faux, models = _handwritten()

    assert list(models.get_models_of_type("chat", "handwritten")) == list(faux.models)
    assert list(models.get_all_models("handwritten")) == list(faux.models)


def test_narrow_mixed_lists_with_is_model_type() -> None:
    mixed = [
        chat_model("p", "c"),
        replace(chat_model("p", "typed"), type="chat"),
        replace(chat_model("p", "future-embedding"), type="embedding"),
    ]
    assert [model.id for model in mixed if is_model_type(model, "chat")] == ["c", "typed"]
    assert [model.id for model in mixed if is_model_type(model, "embedding")] == [
        "future-embedding"
    ]


def test_builtin_catalog_getters_return_model_shapes_that_can_be_reassigned_within_one_api() -> None:
    # 上游是编译期回归检查：返回类型不能携带字面量模型 id。
    # Python 运行时无类型检查，因此这里改为验证 getter 的运行时形态。
    model = get_builtin_model("openai", "gpt-4o-mini")
    model = get_builtin_model("openai", "gpt-4o")
    compat = get_compat_model("openai", "gpt-4o-mini")
    compat = get_compat_model("openai", "gpt-4o")

    assert model is not None and model.id == "gpt-4o"
    assert compat is not None and compat.id == "gpt-4o"


async def test_stored_and_fetched_models_of_unknown_types_are_dropped_instead_of_failing_the_refresh() -> None:
    models_store = InMemoryModelsStore()
    stored = ModelsStoreEntry(
        models=[
            chat_model("dyn", "stored-chat"),
            replace(chat_model("dyn", "stored-unknown"), type="image"),
            replace(chat_model("dyn", "future-embedding"), type="embedding"),
            replace(chat_model("dyn", "future-video"), type="video"),
        ]
    )
    await models_store.write("dyn", stored)

    fetched: list[object] = []

    async def fetch_models(context) -> list[object]:  # noqa: ANN001 - 测试缝隙
        return fetched

    models = create_models(CreateModelsOptions(models_store=models_store))
    models.set_provider(
        create_provider(
            CreateProviderOptions(
                id="dyn",
                auth=ProviderAuth(api_key=_NoAuth()),
                models=[],
                fetch_models=fetch_models,
                api=_ChatStreams(),
            )
        )
    )

    restored = await models.refresh(ModelsRefreshOptions(providers=["dyn"], allow_network=False))
    assert len(restored.errors) == 0
    assert [model.id for model in models.get_all_models("dyn")] == ["stored-chat"]

    fetched.extend(
        [
            chat_model("dyn", "fetched-chat"),
            replace(chat_model("dyn", "fetched-video"), type="video"),
        ]
    )
    refreshed = await models.refresh(ModelsRefreshOptions(providers=["dyn"]))
    assert len(refreshed.errors) == 0
    assert [model.id for model in models.get_all_models("dyn")] == ["fetched-chat"]
    stored_after = await models_store.read("dyn")
    assert stored_after is not None
    assert [model.id for model in stored_after.models] == ["fetched-chat"]
