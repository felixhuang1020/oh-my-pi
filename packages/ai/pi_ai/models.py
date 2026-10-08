"""provider 注册表：``Models``、``create_provider`` 与模型辅助函数。

移植自 ``packages/ai/src/models.ts``。:class:`Provider` 是具体的运行时单元，
拥有请求行为；:class:`Models` 负责解析认证，并把每个请求委托给拥有该模型的
provider。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from .api.lazy import lazy_stream
from .auth.context import default_provider_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.resolve import AuthResolutionOverrides, resolve_provider_auth
from .auth.types import (
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthOperationOptions,
    AuthResult,
    Credential,
    CredentialStore,
    ProviderAuth,
    ProviderAuthInteractionImpl,
)
from .models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry
from .types import (
    AnyModel,
    AssistantMessage,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    Model,
    ModelCostRates,
    ModelType,
    ProviderHeaders,
    ProviderRequestOptions,
    SimpleStreamOptions,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from .utils.abort import (
    AbortController,
    AbortSignal,
    operation_signal,
    race_with_abort_signal,
)
from .utils.async_utils import maybe_await
from .utils.model_operations import (
    assert_chat_model,
    get_model_type,
    is_model_type,
)
from .utils.models_error import ModelsError
from .utils.transcript import normalize_context

__all__ = [
    "CreateModelsOptions",
    "CreateProviderOptions",
    "Models",
    "ModelsImpl",
    "ModelsPublication",
    "ModelsRefreshOptions",
    "ModelsRefreshResult",
    "MutableModels",
    "Provider",
    "ProviderImpl",
    "RefreshModelsContext",
    "UNSET",
    "calculate_cost",
    "clamp_thinking_level",
    "create_models",
    "create_provider",
    "get_supported_thinking_levels",
    "has_api",
    "models_are_equal",
]


class _Unset:
    """哨兵值，区分“保持存储不变”与显式 ``None``。"""

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        """返回该哨兵类的全局唯一实例。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """返回固定字符串 ``UNSET``，便于日志与调试。"""
        return "UNSET"

    def __bool__(self) -> bool:
        """哨兵在布尔上下文中恒为假，便于以 ``if value:`` 判断是否设置。"""
        return False


#: 标记“本次发布不触及持久化存储”。
UNSET = _Unset()


@dataclass
class ModelsPublication:
    """provider 经代次校验后发布的目录状态。

    Attributes:
        persist: provider 选定的待持久化目录。``UNSET`` 表示存储保持不变；
            ``None`` 表示删除存储。
        update: 可选的同步更新，作用于 provider 私有的内存目录状态。
    """

    #: provider 选定的待持久化目录；``UNSET`` 表示存储不变，``None`` 表示删除。
    persist: Any = UNSET
    #: 可选的同步更新，作用于 provider 私有的内存目录状态。
    update: Callable[[], None] | None = None


@dataclass
class RefreshModelsContext:
    """传给 ``Provider.refresh_models`` 的上下文。

    Attributes:
        credential: 生效的已配置凭据。
        stored: 本次刷新阶段开始前捕获的不可变 provider 作用域目录快照。
        publish: 持久化与内存状态的代次校验发布回调。
        allow_network: 离线/仅缓存初始化期间为 False。
        force: 允许网络访问时，绕过 provider 新鲜度检查立即拉取。
        signal: 始终存在；即便公开刷新调用方省略了可选 signal 也是如此。
    """

    #: 生效的已配置凭据。
    credential: Credential | None = None
    #: 本次刷新阶段开始前捕获的不可变 provider 作用域目录快照。
    stored: ModelsStoreEntry | None = None
    #: 持久化与内存状态的代次校验发布回调。
    publish: Callable[[ModelsPublication], Awaitable[bool]] | None = None
    #: 离线/仅缓存初始化期间为 False。
    allow_network: bool = False
    #: 允许网络访问时，绕过 provider 新鲜度检查立即拉取。
    force: bool | None = None
    #: 始终存在，即使公开刷新调用方省略了其可选 signal。
    signal: AbortSignal = field(default_factory=AbortSignal)


@dataclass
class ModelsRefreshOptions:
    """:meth:`Models.refresh` 的选项。

    Attributes:
        allow_network: 是否允许网络访问。
        providers: 仅刷新这些 provider id；未知与静态 provider 会被忽略。
        force: 允许网络访问时，绕过 provider 新鲜度检查立即拉取。
        signal: 取消信号。
    """

    allow_network: bool | None = None
    #: 仅刷新这些 provider id；未知与静态 provider 会被忽略。
    providers: Sequence[str] | None = None
    #: 允许网络访问时，绕过 provider 新鲜度检查立即拉取。
    force: bool | None = None
    signal: AbortSignal | None = None


@dataclass
class ModelsRefreshResult:
    """:meth:`Models.refresh` 的结果。

    Attributes:
        aborted: 调用方的 signal 是否在刷新过程中被触发。
        errors: 以 provider id 为键的失败原因；成功的 provider 不出现在其中。
    """

    aborted: bool = False
    errors: dict[str, Exception] = field(default_factory=dict)


class ModelsRequestTransforms:
    """Models 层各请求方法所接受的请求变换标记类型。"""

    __slots__ = ()


@runtime_checkable
class Provider(Protocol):
    """具体的运行时 provider：元数据、认证、模型列表与操作。

    Attributes:
        id: provider 的唯一标识。
        name: 用于展示的名称。
        auth: 该 provider 的认证配置。
    """

    id: str
    name: str
    auth: ProviderAuth


@runtime_checkable
class Models(Protocol):
    """provider 的运行时集合，附带认证应用与请求便捷方法。"""

    def get_providers(self) -> Sequence[Provider]:
        """返回当前已注册的全部 provider。"""
        ...


@dataclass
class CreateModelsOptions:
    """:func:`create_models` 的注入点。

    Attributes:
        credentials: 凭据存储；省略时使用进程内的 ``InMemoryCredentialStore``。
        models_store: provider 目录存储；省略时使用 ``InMemoryModelsStore``。
        auth_context: 认证解析上下文；省略时使用默认 provider 认证上下文。
    """

    credentials: CredentialStore | None = None
    models_store: ModelsStore | None = None
    auth_context: AuthContext | None = None


def merge_headers(
    base: ProviderHeaders | None,
    override: ProviderHeaders | None,
) -> ProviderHeaders | None:
    """大小写不敏感地合并 provider 头；调用方的值优先。

    同名键以 ``override`` 为准，并保留 ``override`` 中书写的大小写形式；
    若两者都没有条目则返回 ``None``。

    Args:
        base: 基础头（通常来自认证解析结果）。
        override: 覆盖头（通常来自请求选项）；同名键会替换 ``base`` 中的条目。

    Returns:
        合并后的头字典；两参数均为空时返回 ``None``。
    """
    if not base and not override:
        return None
    merged: ProviderHeaders = dict(base or {})
    for name, value in (override or {}).items():
        lowered = name.lower()
        for existing in [key for key in merged if key.lower() == lowered]:
            del merged[existing]
        merged[name] = value
    return merged


def _provider_models(provider: Any) -> Sequence[AnyModel]:
    """provider 列出的全部模型，回退到 ``get_models()``（``getAllModels?.() ?? getModels()``）。

    Args:
        provider: 任意实现了 ``get_models``，或同时实现了 ``get_all_models`` 的 provider。

    Returns:
        provider 提供的全类型模型序列。
    """
    get_all = getattr(provider, "get_all_models", None)
    return get_all() if callable(get_all) else provider.get_models()


def _known_model_type(model: AnyModel) -> bool:
    """判断模型类型是否属于本版本认识的范围。

    Args:
        model: 待检查的模型。

    Returns:
        类型为 ``chat`` 时为 ``True``。
    """
    return get_model_type(model) == "chat"


def with_known_model_types(entry: ModelsStoreEntry) -> ModelsStoreEntry:
    """丢弃本版本不认识其类型的已存储模型。

    Args:
        entry: 存储目录快照。

    Returns:
        仅保留已知类型模型的新目录快照。
    """
    return replace(entry, models=[model for model in entry.models if _known_model_type(model)])


class ModelsImpl:
    """默认的 :class:`Models` 实现。

    持有 provider 注册表、凭据存储、目录存储与认证上下文，并按 provider
    代次协调刷新与发布。

    Attributes:
        _providers: 以 provider id 为键的注册表。
        _credentials: 凭据存储。
        _models_store: provider 目录存储。
        _auth_context: 认证解析上下文。
        _refresh_generations: 每个 provider 的刷新代次，用于作废过期发布。
        _refresh_controllers: 每个 provider 正在进行的刷新控制器。
        _publication_chains: 每个 provider 的发布串行链。
    """

    def __init__(self, options: CreateModelsOptions | None = None) -> None:
        """由可选的注入选项构造 Models 集合。

        Args:
            options: 注入点；省略时全部使用默认的内存实现。
        """
        options = options or CreateModelsOptions()
        self._providers: dict[str, Any] = {}
        self._credentials: CredentialStore = options.credentials or InMemoryCredentialStore()
        self._models_store: ModelsStore = options.models_store or InMemoryModelsStore()
        self._auth_context: AuthContext = options.auth_context or default_provider_auth_context()
        self._refresh_generations: dict[str, int] = {}
        self._refresh_controllers: dict[str, AbortController] = {}
        self._publication_chains: dict[str, asyncio.Future[Any]] = {}

    # -- provider 注册表 ------------------------------------------------------

    def set_provider(self, provider: Any) -> None:
        """按 id upsert 一个 provider，并取消其进行中的刷新。

        Args:
            provider: 要注册或替换的 provider 对象。
        """
        self._supersede_provider_refresh(provider.id)
        self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        """按 id 移除一个 provider。

        Args:
            id: 要移除的 provider id。
        """
        self._supersede_provider_refresh(id)
        self._providers.pop(id, None)

    def clear_providers(self) -> None:
        """移除所有 provider。"""
        for provider_id in {*self._providers, *self._refresh_controllers}:
            self._supersede_provider_refresh(provider_id)
        self._providers.clear()

    def get_providers(self) -> Sequence[Any]:
        """所有已注册的 provider。

        Returns:
            按注册顺序排列的 provider 列表副本。
        """
        return list(self._providers.values())

    def get_provider(self, id: str) -> Any | None:
        """``id`` 下注册的 provider（若存在）。

        Args:
            id: 要查询的 provider id。

        Returns:
            对应的 provider；未注册时为 ``None``。
        """
        return self._providers.get(id)

    # -- 模型读取 -------------------------------------------------------------

    def get_models(self, provider: str | None = None) -> Sequence[Model]:
        """同步读取单个或全部 provider 的最近已知 chat 模型。

        Args:
            provider: 指定 provider id；为 ``None`` 时汇总全部 provider。

        Returns:
            匹配的 chat 模型序列；provider 未知或读取异常时返回空列表。
        """
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None:
                return []
            try:
                return entry.get_models()
            except Exception:  # noqa: BLE001 - 尽力而为；行为异常的 provider 返回空列表
                return []
        models: list[Any] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_models())
            except Exception:  # noqa: BLE001
                continue
        return models

    def get_all_models(self, provider: str | None = None) -> Sequence[AnyModel]:
        """同步读取单个或全部 provider 的最近已知全类型模型。

        ``get_all_models`` 在 provider 契约中是可选的：在其出现之前编写的
        provider，或只有 chat 模型的 provider，会回退到 ``get_models()``。

        Args:
            provider: 指定 provider id；为 ``None`` 时汇总全部 provider。

        Returns:
            匹配的全类型模型序列；provider 未知或读取异常时返回空列表。
        """
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None:
                return []
            try:
                return _provider_models(entry)
            except Exception:  # noqa: BLE001 - 尽力而为；行为异常的 provider 返回空列表
                return []
        models: list[Any] = []
        for entry in self._providers.values():
            try:
                models.extend(_provider_models(entry))
            except Exception:  # noqa: BLE001
                continue
        return models

    def get_models_of_type(self, type: ModelType, provider: str | None = None) -> Sequence[Any]:
        """同步读取某一类型的最近已知模型。

        Args:
            type: 模型类型；本项目只支持 ``chat``。
            provider: 指定 provider id；为 ``None`` 时汇总全部 provider。

        Returns:
            类型匹配的模型序列。
        """
        return [model for model in self.get_all_models(provider) if is_model_type(model, type)]

    def get_model(self, provider: str, id: str) -> Model | None:
        """基于最近已知列表的同步运行时 chat 模型查找。

        Args:
            provider: provider id。
            id: 模型 id。

        Returns:
            匹配的 chat 模型；未找到时为 ``None``。
        """
        return next((model for model in self.get_models(provider) if model.id == id), None)

    def get_model_of_type(self, type: ModelType, provider: str, id: str) -> Any | None:
        """同步运行时查找某一类型的模型。

        Args:
            type: 模型类型；本项目只支持 ``chat``。
            provider: provider id。
            id: 模型 id。

        Returns:
            类型与 id 均匹配的模型；未找到时为 ``None``。
        """
        return next((model for model in self.get_models_of_type(type, provider) if model.id == id), None)

    # -- 刷新 -----------------------------------------------------------------

    def _supersede_provider_refresh(self, provider_id: str) -> int:
        """作废 provider 的当前刷新并递增其代次。

        Args:
            provider_id: provider id。

        Returns:
            递增后的新代次；后续发布必须携带该代次才会生效。
        """
        generation = self._refresh_generations.get(provider_id, 0) + 1
        self._refresh_generations[provider_id] = generation
        previous = self._refresh_controllers.pop(provider_id, None)
        if previous is not None:
            previous.abort()
        return generation

    def _begin_provider_refresh(self, provider_id: str) -> tuple[int, AbortController]:
        """开启一轮新的 provider 刷新。

        Args:
            provider_id: provider id。

        Returns:
            ``(generation, controller)`` 二元组，用于本次刷新的代次校验与取消。
        """
        generation = self._supersede_provider_refresh(provider_id)
        controller = AbortController()
        self._refresh_controllers[provider_id] = controller
        return generation, controller

    async def _publish_provider_models(
        self,
        provider_id: str,
        generation: int,
        signal: AbortSignal,
        publication: ModelsPublication,
    ) -> bool:
        """按代次校验并串行发布 provider 的目录状态。

        发布串行化：同一 provider 的新发布必须等待上一次发布链结束后才开始。

        Args:
            provider_id: provider id。
            generation: 发起发布时的刷新代次；与当前代次不符时放弃发布。
            signal: 取消信号。
            publication: 待发布的目录状态。

        Returns:
            发布完成且未被取消或作废时为 ``True``。
        """
        previous = self._publication_chains.get(provider_id)

        async def run() -> bool:
            """执行一次候选发布的持久化与内存更新。"""
            if previous is not None:
                try:
                    await previous
                except Exception:  # noqa: BLE001
                    pass
            if signal.aborted or self._refresh_generations.get(provider_id) != generation:
                return False
            if publication.persist is None:
                await self._models_store.delete(provider_id, _store_options(signal))
            elif publication.persist is not UNSET:
                from .utils.serde import clone

                await self._models_store.write(provider_id, clone(publication.persist), _store_options(signal))
            if signal.aborted or self._refresh_generations.get(provider_id) != generation:
                return False
            if publication.update is not None:
                publication.update()
            return True

        queued: asyncio.Task[bool] = asyncio.ensure_future(run())
        tail: asyncio.Future[Any] = asyncio.ensure_future(_swallow_task(queued))
        self._publication_chains[provider_id] = tail
        tail.add_done_callback(lambda finished, pid=provider_id: self._release_publication(pid, finished))
        return await race_with_abort_signal(queued, signal)

    def _release_publication(self, provider_id: str, finished: asyncio.Future[Any]) -> None:
        """在发布链结束后清理对应的链条目。

        Args:
            provider_id: provider id。
            finished: 刚结束的发布链尾部 future。
        """
        if self._publication_chains.get(provider_id) is finished:
            self._publication_chains.pop(provider_id, None)

    async def _run_provider_refresh_phase(
        self,
        provider: Any,
        credential: Credential | None,
        allow_network: bool,
        force: bool | None,
        generation: int,
        signal: AbortSignal,
    ) -> None:
        """执行 provider 刷新的一轮：读取存储快照并调用 ``refresh_models``。

        Args:
            provider: 待刷新的 provider。
            credential: 本轮生效的凭据；离线恢复阶段为 ``None``。
            allow_network: 本轮是否允许访问网络。
            force: 允许网络访问时，是否绕过 provider 新鲜度检查立即拉取。
            generation: 本轮刷新代次。
            signal: 取消信号。
        """
        stored = await self._models_store.read(provider.id, _store_options(signal))
        context = RefreshModelsContext(
            credential=credential,
            stored=with_known_model_types(stored) if stored is not None else None,
            publish=lambda publication: self._publish_provider_models(provider.id, generation, signal, publication),
            allow_network=allow_network,
            force=force if allow_network else None,
            signal=signal,
        )
        await provider.refresh_models(context)

    async def refresh(self, options: ModelsRefreshOptions | None = None) -> ModelsRefreshResult:
        """并发刷新所选的已配置动态 provider。

        Args:
            options: 刷新选项；省略时按默认值允许网络访问并刷新全部 provider。

        Returns:
            刷新结果，包含是否被取消，以及各 provider 的失败原因。
        """
        options = options or ModelsRefreshOptions()
        allow_network = options.allow_network if options.allow_network is not None else True
        caller_signal = operation_signal(options.signal)
        errors: dict[str, Exception] = {}
        if caller_signal.aborted:
            return ModelsRefreshResult(aborted=True, errors=errors)
        selected = set(options.providers) if options.providers else None
        refreshable = [
            provider
            for provider in self._providers.values()
            if getattr(provider, "refresh_models", None) is not None and (selected is None or provider.id in selected)
        ]

        async def refresh_one(provider: Any) -> None:
            """刷新单个 provider，并把异常收集进 ``errors``。

            Args:
                provider: 待刷新的 provider。
            """
            generation, controller = self._begin_provider_refresh(provider.id)
            signal = AbortSignal.any([caller_signal, controller.signal])

            async def operation() -> None:
                """先离线恢复缓存状态，再按需携带凭据访问网络。"""
                stored_credential: Credential | None = None
                credential_error: BaseException | None = None
                try:
                    stored_credential = await self._read_credential(provider.id, signal)
                except BaseException as error:  # noqa: BLE001
                    credential_error = error

                # 在解析认证或访问网络之前，先恢复缓存的 provider 状态。
                await self._run_provider_refresh_phase(provider, stored_credential, False, None, generation, signal)
                if credential_error is not None:
                    raise credential_error
                if not allow_network or signal.aborted:
                    return

                credential = await self._resolve_refresh_credential(provider, stored_credential, signal)
                if credential is None:
                    return
                await self._run_provider_refresh_phase(
                    provider, credential, True, options.force, generation, signal
                )

            try:
                await race_with_abort_signal(operation(), signal)
            except Exception as error:  # noqa: BLE001
                if not signal.aborted:
                    errors[provider.id] = (
                        error
                        if isinstance(error, Exception)
                        else ModelsError("model_source", f"Model refresh failed for {provider.id}", cause=error)
                    )
            finally:
                if self._refresh_controllers.get(provider.id) is controller:
                    self._refresh_controllers.pop(provider.id, None)

        gather = asyncio.gather(
            *(refresh_one(provider) for provider in refreshable), return_exceptions=True
        )
        try:
            await race_with_abort_signal(gather, caller_signal)
        except Exception as error:  # noqa: BLE001
            if not caller_signal.aborted:
                raise
        return ModelsRefreshResult(aborted=caller_signal.aborted, errors=dict(errors))

    async def _resolve_refresh_credential(
        self,
        provider: Any,
        stored: Credential | None,
        signal: AbortSignal,
    ) -> Credential | None:
        """为刷新解析可用凭据。

        Args:
            provider: 目标 provider。
            stored: 已存储的凭据；可能为空。
            signal: 取消信号。

        Returns:
            生效的凭据；无可用凭据或凭据类型不匹配时为 ``None``。
        """
        api_key = provider.auth.api_key
        if api_key is None:
            return None
        credential = stored if stored is not None and stored.type == "api_key" else None
        result = await api_key.resolve(ctx=self._auth_context, credential=credential, signal=signal)
        if result is None:
            return None
        return ApiKeyCredential(type="api_key", key=result.auth.api_key, env=result.env)

    async def _read_credential(self, provider_id: str, signal: AbortSignal) -> Credential | None:
        """读取 provider 的已存储凭据，并把存储异常包装为 :class:`ModelsError`。

        Args:
            provider_id: provider id。
            signal: 取消信号。

        Returns:
            已存储凭据；不存在时为 ``None``。

        Raises:
            ModelsError: 凭据存储读取失败时。
        """
        try:
            return await self._credentials.read(provider_id, _auth_options(signal))
        except Exception as error:  # noqa: BLE001
            raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error) from error

    # -- 认证 -----------------------------------------------------------------

    async def _check_provider_auth(
        self,
        provider: Any,
        credential: Credential | None,
        signal: AbortSignal,
    ) -> AuthCheck | None:
        """检查单个 provider 的认证配置是否完整。

        Args:
            provider: 目标 provider。
            credential: 已存储凭据；可能为空。
            signal: 取消信号。

        Returns:
            认证检查结果；未配置对应认证方式时为 ``None``。

        Raises:
            ModelsError: provider 自有的 ``check`` 抛出异常时。
        """
        api_key = provider.auth.api_key
        if api_key is None:
            return None
        check = getattr(api_key, "check", None)
        if check is not None:
            try:
                return await check(
                    ctx=self._auth_context,
                    credential=credential if credential is not None and credential.type == "api_key" else None,
                    signal=signal,
                )
            except Exception as error:  # noqa: BLE001
                raise ModelsError(
                    "auth", f"API key auth check failed for provider {provider.id}", cause=error
                ) from error
        resolution = await resolve_provider_auth(provider, self._credentials, self._auth_context, AuthResolutionOverrides(signal=signal))
        return AuthCheck(source=resolution.source, type="api_key") if resolution else None

    async def check_auth(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> AuthCheck | None:
        """检查 provider 的认证配置是否完整。

        Args:
            provider_id: provider id。
            options: 操作选项；其 signal 用于取消检查。

        Returns:
            认证检查结果；provider 不存在或未配置时返回 ``None``。
        """
        signal = operation_signal(options.signal if options else None)

        async def check() -> AuthCheck | None:
            """在 signal 保护下执行一次认证检查。"""
            signal.throw_if_aborted()
            provider = self._providers.get(provider_id)
            if provider is None:
                return None
            return await self._check_provider_auth(provider, await self._read_credential(provider_id, signal), signal)

        return await race_with_abort_signal(check(), signal)

    async def _get_authenticated_providers(
        self, provider_id: str | None, signal: AbortSignal
    ) -> list[tuple[Any, Credential | None]]:
        """筛出认证配置完整的 provider 及其凭据。

        Args:
            provider_id: 仅检查该 provider；为 ``None`` 时检查全部 provider。
            signal: 取消信号。

        Returns:
            ``(provider, credential)`` 列表，仅含认证检查通过者。
        """
        signal.throw_if_aborted()
        if provider_id is not None:
            provider = self._providers.get(provider_id)
            providers = [provider] if provider is not None else []
        else:
            providers = self.get_providers()

        results: list[tuple[Any, Credential | None]] = []
        for provider in providers:
            credential = await self._read_credential(provider.id, signal)
            auth = await self._check_provider_auth(provider, credential, signal)
            if auth is not None:
                results.append((provider, credential))
        return results

    async def get_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> Sequence[Model]:
        """其 provider 认证配置完整的 chat 模型。

        Args:
            provider_id: 仅查询该 provider；为 ``None`` 时汇总全部 provider。
            options: 操作选项；其 signal 用于取消查询。

        Returns:
            认证可用的 chat 模型序列。
        """
        signal = operation_signal(options.signal if options else None)

        async def available() -> list[Model]:
            """汇总各已认证 provider 过滤后的 chat 模型。"""
            providers = await self._get_authenticated_providers(provider_id, signal)
            models: list[Model] = []
            for provider, credential in providers:
                provider_models = provider.get_models()
                filter_models = getattr(provider, "filter_models", None)
                models.extend(filter_models(provider_models, credential) if filter_models else provider_models)
            return models

        return await race_with_abort_signal(available(), signal)

    async def get_available_of_type(
        self, type: ModelType, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> Sequence[Any]:
        """其 provider 认证配置完整的某一类型模型。

        Args:
            type: 模型类型；本项目只保留 ``chat``。
            provider_id: 仅查询该 provider；为 ``None`` 时汇总全部 provider。
            options: 操作选项；其 signal 用于取消查询。

        Returns:
            认证可用且类型匹配的模型序列。
        """
        return [model for model in await self.get_all_available(provider_id, options) if is_model_type(model, type)]

    async def get_all_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> Sequence[AnyModel]:
        """其 provider 认证配置完整的全类型模型。

        Args:
            provider_id: 仅查询该 provider；为 ``None`` 时汇总全部 provider。
            options: 操作选项；其 signal 用于取消查询。

        Returns:
            认证可用的全类型模型序列。
        """
        signal = operation_signal(options.signal if options else None)

        async def available() -> list[AnyModel]:
            """汇总各已认证 provider 过滤后的全类型模型。"""
            providers = await self._get_authenticated_providers(provider_id, signal)
            models: list[AnyModel] = []
            for provider, credential in providers:
                provider_models = provider.get_all_models()
                filter_models = getattr(provider, "filter_models", None)
                models.extend(filter_models(provider_models, credential) if filter_models else provider_models)
            return models

        return await race_with_abort_signal(available(), signal)

    async def get_auth(
        self, provider_or_model: str | AnyModel, overrides: AuthResolutionOverrides | None = None
    ) -> AuthResult | None:
        """解析 provider 作用域的认证；传入模型时还合并静态 model headers。

        Args:
            provider_or_model: provider id，或带有 ``provider`` 与 ``headers`` 的模型。
            overrides: 认证解析覆盖项；省略时使用默认解析。

        Returns:
            认证结果；provider 不存在或无可用认证时为 ``None``。
        """
        signal = operation_signal(overrides.signal if overrides else None)
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        effective = AuthResolutionOverrides(
            api_key=overrides.api_key if overrides else None,
            env=overrides.env if overrides else None,
            signal=signal,
        )
        result = await resolve_provider_auth(provider, self._credentials, self._auth_context, effective)
        model_headers = getattr(provider_or_model, "headers", None) if not isinstance(provider_or_model, str) else None
        if result is None or model_headers is None:
            return result
        return AuthResult(
            auth=replace(result.auth, headers=merge_headers(result.auth.headers, model_headers)),
            env=result.env,
            source=result.source,
        )

    async def login(
        self,
        provider_id: str,
        type: str,
        interaction: Any,
        options: Any = None,
    ) -> Credential:
        """执行 provider 自有的 api-key 登录流程并持久化返回的凭据。

        Args:
            provider_id: provider id。
            type: 登录方式；本项目只支持 ``api_key``。
            interaction: 登录交互回调对象，其 ``signal`` 用于取消。
            options: 保留参数；api-key 流程不消费。

        Returns:
            持久化后的凭据；存储未返回结果时回退为登录得到的凭据。

        Raises:
            ModelsError: provider 未知、不支持该登录方式，或凭据写入失败时。
        """
        signal = operation_signal(getattr(interaction, "signal", None))
        signal.throw_if_aborted()
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        if type != "api_key":
            raise ModelsError("auth", f"{provider.name} does not support {type} login")
        method = provider.auth.api_key
        if method is None or getattr(method, "login", None) is None:
            raise ModelsError("auth", f"{provider.name} does not support {type} login")

        interaction_impl = ProviderAuthInteractionImpl(interaction)
        credential = await race_with_abort_signal(method.login(interaction_impl), signal)
        try:
            post = await self._credentials.modify(
                provider_id, lambda _current: _return(credential), _auth_options(signal)
            )
        except Exception as error:  # noqa: BLE001
            signal.throw_if_aborted()
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error) from error
        return post if post is not None else credential

    async def logout(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        """移除 provider 的已存储凭据。

        Args:
            provider_id: provider id。
            options: 操作选项；其 signal 用于取消。

        Raises:
            ModelsError: 凭据存储删除失败时。
        """
        signal = operation_signal(options.signal if options else None)
        signal.throw_if_aborted()
        try:
            await self._credentials.delete(provider_id, _auth_options(signal))
        except Exception as error:  # noqa: BLE001
            signal.throw_if_aborted()
            raise ModelsError("auth", f"Credential store delete failed for {provider_id}", cause=error) from error

    # -- 请求分发 -------------------------------------------------------------

    def _require_provider(self, model: AnyModel) -> Any:
        """取出模型所属的 provider。

        Args:
            model: 任意类型的模型，其 ``provider`` 字段指明目标。

        Returns:
            模型所属的 provider。

        Raises:
            ModelsError: 模型的 provider 未注册时。
        """
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    def _require_chat_provider(self, model: Model) -> Any:
        """校验模型确为 chat 模型，并取出其 provider。

        Args:
            model: 待校验的 chat 模型。

        Returns:
            模型所属的 provider。
        """
        assert_chat_model(model)
        return self._require_provider(model)

    async def _apply_auth(
        self,
        model: Any,
        options: ProviderRequestOptions | None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
        fallback: ProviderRequestOptions | None = None,
    ) -> tuple[Any, Any]:
        """解析认证并合并进请求选项。

        ``fallback`` 是调用方未传选项时使用的兜底对象：TypeScript 从
        ``options ?? {}`` 展开 ``...providerOptions``，因而总会产生一个对象。

        chat 路径刻意使用 :class:`SimpleStreamOptions`——它是 ``StreamOptions``
        的*超集*。adapter 从普通 ``stream()`` 调用中读取仅存在于 simple 的字段
        （如 ``deferred``）是合法的——上游读到的是 ``undefined``——而 Python
        若不提供该字段会抛 ``AttributeError`` 而非得到 ``None``。

        Args:
            model: 待请求的模型。
            options: 调用方传入的请求选项；可为 ``None``。
            transform_headers: 可选的 Models 层头变换回调，最后执行。
            fallback: 调用方未传选项时使用的兜底选项对象。

        Returns:
            ``(request_model, request_options)`` 二元组；前者已注入认证解析出的
            ``base_url``，后者已合并 api key、headers 与 env。

        Raises:
            ModelsError: provider 未注册或未配置认证时。
        """
        self._require_provider(model)
        resolution = await self.get_auth(
            model,
            AuthResolutionOverrides(
                api_key=options.api_key if options else None,
                env=options.env if options else None,
                signal=options.signal if options else None,
            ),
        )
        if resolution is None:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")
        auth = resolution.auth

        # 显式请求选项按字段优先；仅限 Models 层的变换最后执行。
        api_key = options.api_key if options and options.api_key is not None else auth.api_key
        headers = merge_headers(auth.headers, options.headers if options else None)
        if transform_headers is not None:
            headers = await maybe_await(transform_headers(headers or {}))
        env = None
        if resolution.env or (options and options.env):
            env = {**(resolution.env or {}), **(options.env if options and options.env else {})}
        request_model = replace(model, base_url=auth.base_url) if auth.base_url else model
        base = options if options is not None else (fallback or StreamOptions())
        request_options = replace(base, api_key=api_key, headers=headers, env=env)
        return request_model, request_options

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
    ):
        """经由所属 provider stream 一段规范化 transcript。

        Args:
            model: 目标 chat 模型。
            context: 待规范化的对话 context。
            options: 请求选项；省略时使用默认选项。
            transform_headers: 可选的 Models 层头变换回调。

        Returns:
            惰性 stream；认证解析在首次消费时才发生。
        """
        transcript = normalize_context(context)

        async def setup() -> Any:
            """解析认证并委托所属 provider 发起 stream。"""
            provider = self._require_chat_provider(model)
            request_model, request_options = await self._apply_auth(
                model, options, transform_headers, SimpleStreamOptions()
            )
            return provider.stream(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
    ) -> AssistantMessage:
        """等待完整的 assistant 响应。

        Args:
            model: 目标 chat 模型。
            context: 待规范化的对话 context。
            options: 请求选项；省略时使用默认选项。
            transform_headers: 可选的 Models 层头变换回调。

        Returns:
            完整的 assistant 消息。
        """
        return await self.stream(model, context, options, transform_headers).result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
    ):
        """使用 provider 无关的简单选项发起 stream。

        Args:
            model: 目标 chat 模型。
            context: 待规范化的对话 context。
            options: 简单请求选项；省略时使用默认选项。
            transform_headers: 可选的 Models 层头变换回调。

        Returns:
            惰性 stream；认证解析在首次消费时才发生。
        """
        transcript = normalize_context(context)

        async def setup() -> Any:
            """解析认证并委托所属 provider 发起 simple stream。"""
            provider = self._require_chat_provider(model)
            request_model, request_options = await self._apply_auth(
                model, options, transform_headers, SimpleStreamOptions()
            )
            return provider.stream_simple(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
    ) -> AssistantMessage:
        """等待完整的简单响应。

        Args:
            model: 目标 chat 模型。
            context: 待规范化的对话 context。
            options: 简单请求选项；省略时使用默认选项。
            transform_headers: 可选的 Models 层头变换回调。

        Returns:
            完整的 assistant 消息。
        """
        return await self.stream_simple(model, context, options, transform_headers).result()

    def stream_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
    ):
        """stream 此前 deferred 响应的补全结果。

        Args:
            model: 目标 chat 模型。
            handle: 此前 deferred 响应的句柄。
            options: deferred 拉取选项；省略时使用默认选项。
            transform_headers: 可选的 Models 层头变换回调。

        Returns:
            惰性 stream；首次消费时若 provider 不支持 deferred 响应则报错。
        """

        async def setup() -> Any:
            """解析认证并委托所属 provider 补全 deferred 响应。"""
            provider = self._require_chat_provider(model)
            fetch_deferred = getattr(provider, "fetch_deferred", None)
            if fetch_deferred is None:
                raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
            request_model, request_options = await self._apply_auth(
                model, options, transform_headers, SimpleStreamOptions()
            )
            return fetch_deferred(request_model, handle, request_options)

        return lazy_stream(model, setup)

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
    ) -> AssistantMessage:
        """等待此前 deferred 响应的补全结果。

        Args:
            model: 目标 chat 模型。
            handle: 此前 deferred 响应的句柄。
            options: deferred 拉取选项；省略时使用默认选项。
            transform_headers: 可选的 Models 层头变换回调。

        Returns:
            完整的 assistant 消息。
        """
        return await self.stream_deferred(model, handle, options, transform_headers).result()

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
        transform_headers: Callable[[ProviderHeaders], Any] | None = None,
    ) -> None:
        """尽力取消一个 deferred 响应。

        Args:
            model: 目标 chat 模型。
            handle: 待取消的 deferred 响应句柄。
            options: deferred 取消选项；省略时使用默认选项。
            transform_headers: 可选的 Models 层头变换回调。

        Raises:
            ModelsError: provider 不支持 deferred 响应时。
        """
        provider = self._require_chat_provider(model)
        cancel_deferred = getattr(provider, "cancel_deferred", None)
        if cancel_deferred is None:
            raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
        request_model, request_options = await self._apply_auth(
            model, options, transform_headers, ProviderRequestOptions()
        )
        await cancel_deferred(request_model, handle, request_options)


#: :func:`create_models` 返回的可变 :class:`Models` 接口。
MutableModels = ModelsImpl


def create_models(options: CreateModelsOptions | None = None) -> ModelsImpl:
    """构建一个空的 :class:`Models` 集合。

    Args:
        options: 注入点；省略时全部使用默认的内存实现。

    Returns:
        新建的 :class:`ModelsImpl` 实例。
    """
    return ModelsImpl(options)


# --------------------------------------------------------------------------------------
# provider 工厂
# --------------------------------------------------------------------------------------


@dataclass
class CreateProviderOptions:
    """:func:`create_provider` 构建 provider 所需的部件。

    Attributes:
        id: provider id。
        name: 显示名称，缺省时回退为 ``id``。
        base_url: 覆盖默认请求地址。
        headers: 附加到所有请求的静态头。
        auth: 认证配置。
        models: 静态基线 chat 模型。
        fetch_models: 拉取动态模型覆盖层。
        filter_models: 按凭据过滤 chat 模型可用性。
        api: chat 实现：单个实现服务所有 chat 模型，或以 ``model.api`` 为键的映射。
    """

    id: str = ""
    name: str | None = None
    base_url: str | None = None
    headers: ProviderHeaders | None = None
    auth: ProviderAuth = field(default_factory=ProviderAuth)
    #: 静态基线 chat 模型。
    models: Sequence[AnyModel] = field(default_factory=list)
    #: 拉取动态模型覆盖层。
    fetch_models: Callable[[RefreshModelsContext], Awaitable[Sequence[AnyModel]]] | None = None
    #: 按凭据过滤 chat 模型可用性。
    filter_models: Callable[[Sequence[Model], Credential | None], Sequence[Model]] | None = None
    #: chat 实现：单个实现服务所有 chat 模型，或以 ``model.api`` 为键的映射。
    api: Any = None


class ProviderImpl:
    """:func:`create_provider` 产出的 provider。

    Attributes:
        id: provider id。
        name: 展示名称；未提供时回退为 ``id``。
        base_url: 覆盖的默认请求地址。
        headers: 附加到所有请求的静态头。
        auth: 认证配置。
        filter_models: 按凭据过滤 chat 模型可用性的回调。
    """

    def __init__(self, options: CreateProviderOptions) -> None:
        """由部件构造 provider，并按实现能力挂载可选的 deferred 方法。

        Args:
            options: provider 部件。

        Raises:
            ValueError: 未提供任何 ``api`` 实现时。
        """
        single = options.api if hasattr(options.api, "stream") else None
        by_api: dict[str, Any] | None = None if (single or options.api is None) else dict(options.api)
        streams = [single] if single is not None else [entry for entry in (by_api or {}).values() if entry is not None]
        if not streams:
            raise ValueError(f'Provider {options.id}: "api" is required.')

        self.id = options.id
        self.name = options.name or options.id
        self.base_url = options.base_url
        self.headers = options.headers
        self.auth = options.auth
        self._single = single
        self._by_api = by_api
        self._baseline_models: Sequence[AnyModel] = list(options.models)
        self._dynamic_models: list[AnyModel] = []
        self._fetch_models = options.fetch_models
        self.filter_models = options.filter_models

        if any(getattr(entry, "fetch_deferred", None) is not None for entry in streams):
            self.fetch_deferred = self._fetch_deferred
        if any(getattr(entry, "cancel_deferred", None) is not None for entry in streams):
            self.cancel_deferred = self._cancel_deferred

    # -- 模型列表 -------------------------------------------------------------

    def _current_models(self) -> list[AnyModel]:
        """把动态覆盖层合并到静态基线模型之上。

        匹配规则为类型与 id 均相同；命中的基线条目被原地替换，未命中的追加到末尾。

        Returns:
            合并后的全类型模型列表。
        """
        merged = list(self._baseline_models)
        for model in self._dynamic_models:
            index = next(
                (
                    position
                    for position, entry in enumerate(merged)
                    if get_model_type(entry) == get_model_type(model) and entry.id == model.id
                ),
                -1,
            )
            if index >= 0:
                merged[index] = model
            else:
                merged.append(model)
        return merged

    def get_models(self) -> list[Model]:
        """当前已知的 chat 模型。

        Returns:
            基线模型与动态覆盖层合并后的 chat 模型列表。
        """
        return [model for model in self._current_models() if is_model_type(model, "chat")]

    def get_all_models(self) -> list[AnyModel]:
        """当前已知的全类型模型。

        Returns:
            基线模型与动态覆盖层合并后的全类型模型列表。
        """
        return self._current_models()

    def _api_for(self, model: Model) -> Any:
        """取出服务该模型的 api 实现。

        Args:
            model: 目标模型。

        Returns:
            单一实现，或映射中 ``model.api`` 对应的实现；无匹配时为 ``None``。
        """
        if self._single is not None:
            return self._single
        return (self._by_api or {}).get(model.api)

    async def refresh_models(self, context: RefreshModelsContext) -> None:
        """先恢复已存储的覆盖层，再按需拉取新的覆盖层。

        Args:
            context: 刷新上下文，提供存储快照、发布回调与取消信号。
        """
        if context.stored is not None:
            restored = [
                model for model in context.stored.models if model.provider == self.id
            ]

            def apply_restored() -> None:
                """把恢复出的已存储模型装入动态覆盖层。"""
                self._dynamic_models = list(restored)

            published = await context.publish(ModelsPublication(update=apply_restored)) if context.publish else True
            if not published:
                return
        if not context.allow_network or context.signal.aborted:
            return
        assert self._fetch_models is not None
        fetched = await self._fetch_models(context)
        if context.signal.aborted:
            return
        refreshed = [model for model in fetched if _known_model_type(model)]

        def apply_refreshed() -> None:
            """把新拉取的模型装入动态覆盖层。"""
            self._dynamic_models = list(refreshed)

        if context.publish is not None:
            await context.publish(
                ModelsPublication(
                    persist=ModelsStoreEntry(models=list(refreshed), checked_at=int(time.time() * 1000)),
                    update=apply_refreshed,
                )
            )
        else:
            apply_refreshed()

    # -- 分发 -----------------------------------------------------------------

    def _dispatch(self, model: Model, run: Callable[[Any], Any]):
        """把请求分发给服务该模型的 api 实现。

        Args:
            model: 目标模型。
            run: 接收 api 实现并执行具体请求的回调。

        Returns:
            回调的返回值；无可用 api 实现时返回携带错误的惰性 stream。
        """
        streams = self._api_for(model)
        if streams is None:

            async def fail() -> Any:
                """在首次消费 stream 时抛出缺实现错误。"""
                raise ModelsError("stream", f'Provider {self.id} has no API implementation for "{model.api}"')

            return lazy_stream(model, fail)
        return run(streams)

    def stream(self, model: Model, context: TranscriptContext, options: StreamOptions | None = None):
        """stream 一段规范化 transcript。

        Args:
            model: 目标 chat 模型。
            context: 对话 context。
            options: 请求选项；省略时由实现使用默认值。

        Returns:
            惰性 stream。
        """
        return self._dispatch(model, lambda streams: streams.stream(model, context, options))

    def stream_simple(self, model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None):
        """使用 provider 无关的简单选项发起 stream。

        Args:
            model: 目标 chat 模型。
            context: 对话 context。
            options: 简单请求选项；省略时由实现使用默认值。

        Returns:
            惰性 stream。
        """
        return self._dispatch(model, lambda streams: streams.stream_simple(model, context, options))

    def _fetch_deferred(self, model: Model, handle: DeferredHandle, options: DeferredFetchOptions | None = None):
        """补全此前 deferred 的响应。

        Args:
            model: 目标 chat 模型。
            handle: deferred 响应句柄。
            options: deferred 拉取选项；省略时由实现使用默认值。

        Returns:
            惰性 stream；实现不支持 deferred 响应时在首次消费时报错。
        """

        async def setup() -> Any:
            """在首次消费时校验并转发 deferred 拉取请求。"""
            implementation = self._api_for(model)
            fetch_deferred = getattr(implementation, "fetch_deferred", None)
            if fetch_deferred is None:
                raise ModelsError(
                    "provider", f'Provider {self.id} does not support deferred responses for "{model.api}"'
                )
            return fetch_deferred(model, handle, options)

        return lazy_stream(model, setup)

    async def _cancel_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredCancelOptions | None = None
    ) -> None:
        """尽力取消一个 deferred 响应。

        Args:
            model: 目标 chat 模型。
            handle: deferred 响应句柄。
            options: deferred 取消选项；省略时由实现使用默认值。

        Raises:
            ModelsError: 实现不支持取消 deferred 响应时。
        """
        implementation = self._api_for(model)
        cancel_deferred = getattr(implementation, "cancel_deferred", None)
        if cancel_deferred is None:
            raise ModelsError(
                "provider", f'Provider {self.id} cannot cancel deferred responses for "{model.api}"'
            )
        await cancel_deferred(model, handle, options)


def create_provider(options: CreateProviderOptions) -> ProviderImpl:
    """由部件构建 provider。

    单个 ``api`` 负责 stream 所有 chat 模型；``api`` 映射则按 ``model.api``
    分发，模型所属 api 无对应条目时产生 stream 错误。

    Args:
        options: provider 部件。

    Returns:
        新建的 :class:`ProviderImpl` 实例。
    """
    return ProviderImpl(options)


def has_api(model: AnyModel, api: str) -> bool:
    """运行时检查并收窄为由 ``api`` 服务的 chat 模型。

    Args:
        model: 待检查的模型。
        api: 目标 api 标识。

    Returns:
        模型为 chat 类型且 ``model.api`` 等于 ``api`` 时为 ``True``。
    """
    return is_model_type(model, "chat") and model.api == api


def calculate_cost(model: AnyModel, usage: Usage) -> Any:
    """按模型目录价格就地计算 ``usage.cost``。

    以输入 token 总数（含缓存读取与写入）挑选命中的价格档位；
    Anthropic 的 1h 缓存写入按基础输入价的 2 倍计费。

    Args:
        model: 提供价格目录与档位的模型。
        usage: 用量对象；其 ``cost`` 字段会被就地更新。

    Returns:
        更新后的 ``usage.cost``。
    """
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates: ModelCostRates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers or []:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above

    # Anthropic 的 1h 缓存写入按基础输入价的 2 倍计费。
    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    usage.cost.input = (rates.input / 1_000_000) * usage.input
    usage.cost.output = (rates.output / 1_000_000) * usage.output
    usage.cost.cache_read = (rates.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )
    return usage.cost


#: thinking 档位，按努力程度升序排列。
EXTENDED_THINKING_LEVELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def get_supported_thinking_levels(model: Model) -> list[str]:
    """``model`` 接受的 thinking 档位，遵循 ``thinking_level_map``。

    非 reasoning 模型仅支持 ``off``；``xhigh`` 与 ``max`` 必须显式出现在
    ``thinking_level_map`` 中才被认可。

    Args:
        model: 待查询的模型。

    Returns:
        按努力程度升序排列的受支持档位。
    """
    if not model.reasoning:
        return ["off"]

    def supported(level: str) -> bool:
        """判断单个档位是否被该模型接受。

        Args:
            level: 档位名。

        Returns:
            档位受支持时为 ``True``。
        """
        mapped = (model.thinking_level_map or {}).get(level)
        if mapped is None and level in (model.thinking_level_map or {}):
            return False
        if level in ("xhigh", "max"):
            return level in (model.thinking_level_map or {})
        return True

    return [level for level in EXTENDED_THINKING_LEVELS if supported(level)]


def clamp_thinking_level(model: Model, level: str) -> str:
    """返回最接近 ``level`` 的受支持 thinking 档位。

    优先向后寻找更高档位，找不到时再向前回退；未知档位直接回退到最低可用档位。

    Args:
        model: 待查询的模型。
        level: 期望的档位名。

    Returns:
        最接近期望的受支持档位；无可用档位时返回 ``off``。
    """
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    if level not in EXTENDED_THINKING_LEVELS:
        return available[0] if available else "off"
    requested_index = EXTENDED_THINKING_LEVELS.index(level)
    for candidate in EXTENDED_THINKING_LEVELS[requested_index:]:
        if candidate in available:
            return candidate
    for candidate in reversed(EXTENDED_THINKING_LEVELS[:requested_index]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def models_are_equal(a: AnyModel | None, b: AnyModel | None) -> bool:
    """判断两个模型是否为同一词条：类型、id 与 provider 均相同。

    Args:
        a: 第一个模型；可为 ``None``。
        b: 第二个模型；可为 ``None``。

    Returns:
        两者均非空且类型、id、provider 全部一致时为 ``True``。
    """
    if not a or not b:
        return False
    return get_model_type(a) == get_model_type(b) and a.id == b.id and a.provider == b.provider


async def _return(value: Any) -> Any:
    """原样返回入参的异步包装，用作存储修改回调。

    Args:
        value: 待返回的值。

    Returns:
        入参 ``value`` 本身。
    """
    return value


async def _swallow_task(task: asyncio.Task[Any]) -> None:
    """等待任务结束并吞掉其异常，用于生成可挂回调的发布链尾部。

    Args:
        task: 待等待的任务。
    """
    try:
        await asyncio.shield(task)
    except (Exception, asyncio.CancelledError):  # noqa: BLE001
        pass


def _store_options(signal: AbortSignal) -> Any:
    """构造目录存储操作选项。

    Args:
        signal: 取消信号。

    Returns:
        携带该 signal 的存储操作选项。
    """
    from .models_store import ModelsStoreOperationOptions

    return ModelsStoreOperationOptions(signal=signal)


def _auth_options(signal: AbortSignal) -> AuthOperationOptions:
    """构造认证操作选项。

    Args:
        signal: 取消信号。

    Returns:
        携带该 signal 的认证操作选项。
    """
    return AuthOperationOptions(signal=signal)


# 为只从 ``models`` 导入的调用方重新导出。
__all__ += ["ModelsError", "get_model_type", "is_model_type"]

from .utils.models_error import ModelsErrorCode  # noqa: E402

__all__ += ["ModelsErrorCode"]
