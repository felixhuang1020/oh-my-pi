"""pi-ai：``@earendil-works/pi-ai`` 的 Python 移植版。

公开接口与 ``packages/ai/src/index.ts`` 对应：核心数据类型、``Models``
注册表、认证原语、``faux`` 测试 provider，以及无副作用的工具函数。

provider 工厂位于 :mod:`pi_ai.providers`（``all_providers()`` 返回全部内置
provider），API 实现位于 :mod:`pi_ai.api`。本项目只保留四种 chat API
（``openai-responses``、``anthropic-messages``、``google-generative-ai``、
``pi-messages``）、九家 provider，并只支持 api-key 认证。

各 API 的 *option* dataclass 通过 :pep:`562` 模块 ``__getattr__`` 惰性解析：
TypeScript barrel 用 ``export type`` 导入它们（运行时已擦除），而为了几个注解
就 import 数千行的 adapter 会让 ``import pi_ai`` 变得不必要地昂贵。

    >>> from pi_ai import create_models, lazy_api
    >>> from pi_ai.providers.all import all_providers
    >>> models = create_models()
    >>> for provider in all_providers():
    ...     models.set_provider(provider)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# ``types`` 的通配导入刻意排在具名导入之前：同名时后者胜出，公开接口因此始终是
# 下面显式列出的那一个，而不是 ``types`` 里恰好同名的占位符。
from .types import *
from .api.lazy import LazyApi, lazy_api, lazy_load, lazy_stream
from .auth.context import DefaultProviderAuthContext, default_provider_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.helpers import EnvApiKeyAuth, env_api_key_auth
from .auth.resolve import AuthResolutionOverrides, resolve_provider_auth
from .auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthInteraction,
    AuthOperationOptions,
    AuthPrompt,
    AuthPromptSecret,
    AuthPromptText,
    AuthResult,
    Credential,
    CredentialInfo,
    CredentialStore,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from .models import (
    EXTENDED_THINKING_LEVELS,
    CreateModelsOptions,
    CreateProviderOptions,
    Models,
    ModelsError,
    ModelsErrorCode,
    ModelsImpl,
    ModelsPublication,
    ModelsRefreshOptions,
    ModelsRefreshResult,
    MutableModels,
    Provider,
    ProviderImpl,
    RefreshModelsContext,
    calculate_cost,
    clamp_thinking_level,
    create_models,
    create_provider,
    get_model_type,
    get_supported_thinking_levels,
    has_api,
    is_model_type,
    merge_headers,
    models_are_equal,
)
from .models_store import (
    InMemoryModelsStore,
    ModelsStore,
    ModelsStoreEntry,
    ModelsStoreOperationOptions,
)
from .session_resources import (
    SessionResourceCleanup,
    cleanup_session_resources,
    register_session_resource_cleanup,
)
from .utils.abort import (
    AbortController,
    AbortError,
    AbortSignal,
    CombinedAbortSignal,
    combine_abort_signals,
    operation_signal,
    race_with_abort_signal,
)
from .utils.assistant_message_frame import (
    AssistantMessageFrame,
    AssistantMessageFrameEncoder,
    reduce_assistant_message_frames,
)
from .utils.diagnostics import (
    AssistantMessageDiagnostic,
    DiagnosticErrorInfo,
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    extract_diagnostic_error,
    format_thrown_value,
)
from .utils.event_stream import (
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
)
from .utils.json_parse import parse_json_with_repair, parse_streaming_json, repair_json
from .utils.overflow import get_overflow_patterns, is_context_overflow, is_recoverable_length
from .utils.retry import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms,
)
from .utils.serde import from_json, to_json
from .utils.text import content_text, get_system_message_text, render_system_message_update
from .utils.transcript import (
    collapse_system_messages,
    create_initial_system_message,
    declarations_equal,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_declared_tools,
    get_initial_system_message,
    get_tool_state_changes,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
    resolve_transcript,
    resolve_transcript_tools,
    to_tool_declaration,
    without_initial_system_message,
)
from .utils.typebox_helpers import string_enum
from .utils.uuid import uuidv7
from .utils.validation import validate_tool_arguments, validate_tool_call

__version__ = "0.1.0"

#: 惰性解析的公开名称：属性名 -> ``(模块, 原始名称)``。
#:
#: 对应 ``src/index.ts`` 的仅类型 import 及 ``src/providers/all.ts`` 的
#: provider/registry 导出，但 import 时无需付出加载代价。
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # -- 各 API 的 option 类型（上游仅类型导出） -------------------------------------
    "AnthropicEffort": ("pi_ai.api.anthropic_messages", "AnthropicEffort"),
    "AnthropicOptions": ("pi_ai.api.anthropic_messages", "AnthropicOptions"),
    "AnthropicThinkingDisplay": ("pi_ai.api.anthropic_messages", "AnthropicThinkingDisplay"),
    "GoogleApiThinkingLevel": ("pi_ai.api.google_shared", "GoogleApiThinkingLevel"),
    "GoogleOptions": ("pi_ai.api.google_generative_ai", "GoogleOptions"),
    "OpenAIResponsesOptions": ("pi_ai.api.openai_responses", "OpenAIResponsesOptions"),
    "PiMessagesEvent": ("pi_ai.api.pi_messages", "PiMessagesEvent"),
    "PiMessagesOptions": ("pi_ai.api.pi_messages", "PiMessagesOptions"),
    "PiMessagesRewriteImpact": ("pi_ai.api.pi_messages", "PiMessagesRewriteImpact"),
    "ResolvedGoogleThinkingLevel": ("pi_ai.api.google_shared", "ResolvedGoogleThinkingLevel"),
    # -- 内置 provider 注册表（``src/providers/all.ts``） ----------------------------
    "BuiltinProvider": ("pi_ai.providers.all", "BuiltinProvider"),
    "all_providers": ("pi_ai.providers.all", "all_providers"),
    "builtin_models": ("pi_ai.providers.all", "builtin_models"),
    "builtin_providers": ("pi_ai.providers.all", "builtin_providers"),
    "get_builtin_model": ("pi_ai.providers.all", "get_builtin_model"),
    "get_builtin_model_data_generated_at": ("pi_ai.providers.all", "get_builtin_model_data_generated_at"),
    "get_builtin_models": ("pi_ai.providers.all", "get_builtin_models"),
    "get_builtin_providers": ("pi_ai.providers.all", "get_builtin_providers"),
}


#: 本模块急切定义的名称。``_LAZY_EXPORTS`` 中的惰性属性被刻意排除：
#: ``from pi_ai import *`` 不应强制加载 provider adapter，且上游这些名称
#: 本就是 ``import *`` 不会携带的仅类型 import。它们仍可作为属性访问
#: （``pi_ai.AnthropicOptions``），并由 :func:`__dir__` 列出。
__all__ = sorted(
    name
    for name in globals()
    # `annotations` 由 `from __future__ import annotations` 自身绑定。
    if not name.startswith("_") and name not in {"Any", "TYPE_CHECKING", "annotations", "LAZY"} and name not in _LAZY_EXPORTS
)


def __getattr__(name: str) -> Any:
    """首次访问时解析上文惰性导出的名称。

    Args:
        name: 被访问的属性名。

    Returns:
        ``_LAZY_EXPORTS`` 中登记的对应对象。

    Raises:
        AttributeError: 该名称不在 ``_LAZY_EXPORTS`` 中。
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """列出模块全局名称与全部惰性导出名，且不触发实际导入。

    Returns:
        排序后的可用属性名列表。
    """
    return sorted({*globals(), *_LAZY_EXPORTS})


if TYPE_CHECKING:  # pragma: no cover - 仅辅助类型检查，对应 TS 的仅类型 import
    from .api.anthropic_messages import AnthropicEffort, AnthropicOptions, AnthropicThinkingDisplay
    from .api.google_generative_ai import GoogleOptions
    from .api.google_shared import GoogleApiThinkingLevel, ResolvedGoogleThinkingLevel
    from .api.openai_responses import OpenAIResponsesOptions
    from .api.pi_messages import PiMessagesEvent, PiMessagesOptions, PiMessagesRewriteImpact
    from .providers.all import all_providers, builtin_models, builtin_providers
