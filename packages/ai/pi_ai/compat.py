"""临时兼容入口，保留旧的全局 pi-ai API 接口。

移植自 ``packages/ai/src/compat.ts``：带环境 API key 注入的 api 分发
``stream()``/``complete()``、api 注册表、生成目录读取（``get_model``/``get_models``/
``get_providers``）、faux 注册以及图像生成。现有应用只需把 import 原样切换到
``pi_ai.compat`` 模块；新代码请使用 ``create_models()`` 和 provider 工厂。
上游的 ``compat.ts`` 会随 coding-agent ModelManager 迁移一并删除，本模块也随之废弃。

TypeScript ``export *`` 对同级旧模块（``images``、``image-models``、
``legacy-api-aliases``、各 API 惰性包装器……）的重新导出通过 :func:`__getattr__`
呈现：这些模块一旦落地即可解析，而本模块在 import 时不依赖它们。
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .api.lazy import lazy_api, lazy_load
from .env_api_keys import *  # noqa: F401,F403 - 对应 `export * from "./env-api-keys.ts"`
from .env_api_keys import get_env_api_key
from .models import ModelsImpl
from .providers.all import (
    builtin_models,
    get_builtin_model,
    get_builtin_models,
    get_builtin_providers,
)
from .providers.faux import (
    FauxProviderRegistration,
    RegisterFauxProviderOptions,
    create_faux_core,
)
from .types import (
    AssistantMessage,
    AssistantMessageEventStream,
    Context,
    Model,
    ProviderStreams,
    SimpleStreamOptions,
    StreamOptions,
)
from .utils.transcript import normalize_context

__all__ = [
    "AMBIENT_AUTH_MARKER",
    "ApiProvider",
    "ApiStreamFunction",
    "ApiStreamSimpleFunction",
    "complete",
    "complete_simple",
    "get_api_provider",
    "get_api_providers",
    "get_model",
    "get_models",
    "get_providers",
    "register_api_provider",
    "register_built_in_api_providers",
    "register_faux_provider",
    "reset_api_providers",
    "stream",
    "stream_simple",
    "unregister_api_providers",
]

#: 已注册 api 实现的 ``stream`` 入口。
ApiStreamFunction = Callable[..., AssistantMessageEventStream]
#: 已注册 api 实现的 ``stream_simple`` 入口。
ApiStreamSimpleFunction = Callable[..., AssistantMessageEventStream]


@dataclass
class ApiProvider:
    """以单一 api id 注册的 api 实现。

    Attributes:
        api: 该实现负责的 api id，例如 ``anthropic-messages``。
        stream: 完整的 ``stream`` 入口。
        stream_simple: 仅接收简单选项的 ``stream_simple`` 入口。
    """

    api: str
    stream: ApiStreamFunction
    stream_simple: ApiStreamSimpleFunction


@dataclass
class _RegisteredApiProvider:
    """注册表条目，把 api 实现与其注册者 id 关联起来。

    Attributes:
        provider: 已加 api 守卫的 api 实现。
        source_id: 注册者 id；``None`` 表示不属于任何可批量注销的来源。
    """

    provider: ApiProvider
    source_id: str | None = None


_api_provider_registry: dict[str, _RegisteredApiProvider] = {}


def wrap_stream(api: str, stream: ApiStreamFunction) -> ApiStreamFunction:
    """为 ``stream`` 加守卫，拒绝属于其他 api 的模型。

    Args:
        api: 该实现负责的 api id。
        stream: 原始 ``stream`` 可调用对象。

    Returns:
        先校验 ``model.api`` 再转发的包装函数。
    """

    def wrapped(model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        """校验 api 后转发给原始 ``stream``。

        Args:
            model: 待请求的模型。
            context: 请求上下文。
            options: 流式选项；``None`` 表示使用默认值。

        Returns:
            助手消息事件流。

        Raises:
            RuntimeError: ``model.api`` 与该实现负责的 api 不一致。
        """
        if model.api != api:
            raise RuntimeError(f"Mismatched api: {model.api} expected {api}")
        return stream(model, context, options)

    return wrapped


def wrap_stream_simple(api: str, stream_simple: ApiStreamSimpleFunction) -> ApiStreamSimpleFunction:
    """为 ``stream_simple`` 加守卫，拒绝属于其他 api 的模型。

    Args:
        api: 该实现负责的 api id。
        stream_simple: 原始 ``stream_simple`` 可调用对象。

    Returns:
        先校验 ``model.api`` 再转发的包装函数。
    """

    def wrapped(model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
        """校验 api 后转发给原始 ``stream_simple``。

        Args:
            model: 待请求的模型。
            context: 请求上下文。
            options: 简单流式选项；``None`` 表示使用默认值。

        Returns:
            助手消息事件流。

        Raises:
            RuntimeError: ``model.api`` 与该实现负责的 api 不一致。
        """
        if model.api != api:
            raise RuntimeError(f"Mismatched api: {model.api} expected {api}")
        return stream_simple(model, context, options)

    return wrapped


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    """按 api id 注册（或替换）一个 api 实现。

    Args:
        provider: 待注册的 api 实现；其两个流式入口会被加上 api 守卫。
        source_id: 注册者 id，供 :func:`unregister_api_providers` 批量注销。
    """
    _api_provider_registry[provider.api] = _RegisteredApiProvider(
        provider=ApiProvider(
            api=provider.api,
            stream=wrap_stream(provider.api, provider.stream),
            stream_simple=wrap_stream_simple(provider.api, provider.stream_simple),
        ),
        source_id=source_id,
    )


def get_api_provider(api: str) -> ApiProvider | None:
    """返回已注册的 api 实现（若存在）。

    Args:
        api: 待查询的 api id。

    Returns:
        对应的 api 实现；未注册时为 ``None``。
    """
    entry = _api_provider_registry.get(api)
    return entry.provider if entry is not None else None


def get_api_providers() -> list[ApiProvider]:
    """返回所有已注册的 api 实现。

    Returns:
        按注册顺序排列的 api 实现列表。
    """
    return [entry.provider for entry in _api_provider_registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """移除 ``source_id`` 名下注册的全部 api 实现。

    Args:
        source_id: 注册时传入的来源 id。
    """
    for api, entry in list(_api_provider_registry.items()):
        if entry.source_id == source_id:
            del _api_provider_registry[api]


def clear_api_providers() -> None:
    """清空所有已注册的 api 实现。"""
    _api_provider_registry.clear()


def register_faux_provider(options: RegisterFauxProviderOptions | None = None) -> FauxProviderRegistration:
    """以私有 id 注册一个脚本化的 faux api 实现。

    Args:
        options: faux provider 的初始配置；``None`` 表示使用默认配置。

    Returns:
        可用于读取状态、注入响应并注销该 provider 的注册句柄。
    """
    core = create_faux_core(options)
    source_id = f"faux-provider-{_random_source_suffix()}"
    register_api_provider(
        ApiProvider(api=core.api, stream=core.stream, stream_simple=core.stream_simple),
        source_id,
    )
    return FauxProviderRegistration(
        api=core.api,
        models=core.models,
        get_model=core.get_model,
        state=core.state,
        set_responses=core.set_responses,
        append_responses=core.append_responses,
        get_pending_response_count=core.get_pending_response_count,
        unregister=lambda: unregister_api_providers(source_id),
    )


def _random_source_suffix() -> str:
    """生成 8 位小写字母数字后缀，用于构造唯一的 faux 来源 id。

    Returns:
        长度为 8 的随机小写字母数字字符串。
    """
    return "".join(random.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(8))


#: compat 层默认注册的 ``(api id, ProviderStreams)`` 对。
BUILTIN_APIS: list[tuple[str, ProviderStreams]] = [
    ("anthropic-messages", lazy_api(lazy_load("pi_ai.api.anthropic_messages"))),
    ("openai-responses", lazy_api(lazy_load("pi_ai.api.openai_responses"))),
    ("google-generative-ai", lazy_api(lazy_load("pi_ai.api.google_generative_ai"))),
    ("pi-messages", lazy_api(lazy_load("pi_ai.api.pi_messages"))),
]

_builtin_api_provider_instances: dict[str, ApiProvider | None] = {}


def register_built_in_api_providers() -> None:
    """注册内置 API 实现，且不覆盖已有条目。

    compat 可能在测试或扩展已为某个内置 api id 注册了覆盖实现之后才加载。
    """
    for api, streams in BUILTIN_APIS:
        if get_api_provider(api) is None:
            register_api_provider(
                ApiProvider(api=api, stream=streams.stream, stream_simple=streams.stream_simple)
            )
        _builtin_api_provider_instances[api] = get_api_provider(api)


def reset_api_providers() -> None:
    """清空注册表并重新注册内置 api 实现。"""
    clear_api_providers()
    _builtin_api_provider_instances.clear()
    register_built_in_api_providers()


register_built_in_api_providers()

_compat_models: ModelsImpl = builtin_models()
AMBIENT_AUTH_MARKER = "<authenticated>"

#: 已废弃的静态目录读取。请改用 ``pi_ai.providers.all`` 中的 ``get_builtin_model``。
get_model = get_builtin_model
#: 已废弃的静态目录读取。请改用 ``pi_ai.providers.all`` 中的 ``get_builtin_models``。
get_models = get_builtin_models
#: 已废弃的静态目录读取。请改用 ``pi_ai.providers.all`` 中的 ``get_builtin_providers``。
get_providers = get_builtin_providers


def has_explicit_api_key(api_key: str | None) -> bool:
    """调用方是否提供了非空白的 api key。

    Args:
        api_key: 调用方传入的 api key。

    Returns:
        为非空白字符串时为 ``True``，否则为 ``False``。
    """
    return isinstance(api_key, str) and api_key.strip() != ""


def with_env_api_key(model: Model, options: Any) -> Any:
    """调用方未显式提供时，注入从环境发现的 api key。

    Args:
        model: 待请求的模型，用于确定 provider。
        options: 原始流式选项；``None`` 表示尚未提供任何选项。

    Returns:
        原 ``options``，或在注入环境 key 后得到的新选项对象。
    """
    if has_explicit_api_key(options.api_key if options is not None else None):
        return options
    api_key = get_env_api_key(model.provider, options.env if options is not None else None)
    if not api_key or api_key == AMBIENT_AUTH_MARKER:
        return options
    if options is None:
        return StreamOptions(api_key=api_key)
    return replace(options, api_key=api_key)


def get_builtin_provider_for_model(model: Model) -> Any | None:
    """当 ``model`` 属于 compat 目录时，返回其所属的内置 provider。

    Args:
        model: 待查询的模型。

    Returns:
        匹配的内置 provider；模型来自其他 api 实现或不在 compat 目录中时
        为 ``None``。
    """
    if get_api_provider(model.api) is not _builtin_api_provider_instances.get(model.api):
        return None
    provider = _compat_models.get_provider(model.provider)
    if provider is None:
        return None
    return provider if any(candidate.api == model.api for candidate in provider.get_models()) else None


def resolve_api_provider(api: str) -> ApiProvider:
    """返回已注册的 api 实现，无注册时抛出异常。

    Args:
        api: 待解析的 api id。

    Returns:
        对应的 api 实现。

    Raises:
        RuntimeError: 该 api id 尚无已注册的实现。
    """
    provider = get_api_provider(api)
    if provider is None:
        raise RuntimeError(f"No API provider registered for api: {api}")
    return provider


def stream(model: Model, context: Context, options: Any = None) -> AssistantMessageEventStream:
    """发起 stream 请求，优先经由所属内置 provider 而非裸 api 注册表。

    Args:
        model: 待请求的模型。
        context: 请求上下文。
        options: 流式选项；``None`` 表示使用默认值。

    Returns:
        助手消息事件流。
    """
    transcript = normalize_context(context)
    builtin_provider = get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        return builtin_provider.stream(model, transcript, with_env_api_key(model, options))
    provider = resolve_api_provider(model.api)
    return provider.stream(model, transcript, with_env_api_key(model, options))


async def complete(model: Model, context: Context, options: Any = None) -> AssistantMessage:
    """等待 :func:`stream` 的终态消息。

    Args:
        model: 待请求的模型。
        context: 请求上下文。
        options: 流式选项；``None`` 表示使用默认值。

    Returns:
        流结束后的最终助手消息。
    """
    return await stream(model, context, options).result()


def stream_simple(
    model: Model, context: Context, options: SimpleStreamOptions | None = None
) -> AssistantMessageEventStream:
    """使用 provider 无关的简单选项发起 stream 请求。

    Args:
        model: 待请求的模型。
        context: 请求上下文。
        options: 简单流式选项；``None`` 表示使用默认值。

    Returns:
        助手消息事件流。
    """
    transcript = normalize_context(context)
    builtin_provider = get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        return builtin_provider.stream_simple(model, transcript, with_env_api_key(model, options))
    provider = resolve_api_provider(model.api)
    return provider.stream_simple(model, transcript, with_env_api_key(model, options))


async def complete_simple(
    model: Model, context: Context, options: SimpleStreamOptions | None = None
) -> AssistantMessage:
    """等待 :func:`stream_simple` 的终态消息。

    Args:
        model: 待请求的模型。
        context: 请求上下文。
        options: 简单流式选项；``None`` 表示使用默认值。

    Returns:
        流结束后的最终助手消息。
    """
    return await stream_simple(model, context, options).result()


#: TypeScript compat 层以 ``export *`` 重新导出的模块。
_LAZY_REEXPORT_MODULES: tuple[str, ...] = (
    "pi_ai.api.anthropic_messages",
    "pi_ai.api.google_generative_ai",
    "pi_ai.api.openai_responses",
    "pi_ai.api.pi_messages",
)


def __getattr__(name: str) -> Any:
    """惰性转发 TypeScript compat 层以 ``export *`` 重新导出的名称。

    Args:
        name: 被访问的属性名。

    Returns:
        首个提供该名称的重导出模块中的属性值。

    Raises:
        AttributeError: 所有候选模块都不可导入或都不提供该名称。
    """
    for module_name in _LAZY_REEXPORT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
