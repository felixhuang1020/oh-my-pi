"""核心数据模型：模型、消息、stream 事件与操作结果。

对应 ``packages/ai/src/types.ts``。结构化类型是带 snake_case 字段与 camelCase
JSON 别名的 dataclass（见 :mod:`pi_ai.utils.serde`）。TypeScript 写作
``KnownX | (string & {})`` 的开放字符串联合在此为普通 ``str``，闭集则保留为
名为 ``KnownX`` 的 ``StrEnum``。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeAlias, get_args, runtime_checkable

from .utils.abort import AbortSignal
from .utils.diagnostics import AssistantMessageDiagnostic
from .utils.event_stream import AssistantMessageEventStream
from .utils.serde import content_block

__all__ = [
    "Api",
    "AnyModel",
    "AnthropicAllowedFallbackModel",
    "AssistantMessage",
    "AssistantMessageEvent",
    "BaseModel",
    "CacheRetention",
    "ConstrainedSamplingConfig",
    "Context",
    "DeferredCancelOptions",
    "DeferredFetchOptions",
    "DeferredHandle",
    "FetchFunction",
    "GrammarFormat",
    "GrammarVariants",
    "JsonObject",
    "JsonValue",
    "KnownApi",
    "KNOWN_APIS",
    "KNOWN_PROVIDERS",
    "KnownProvider",
    "Message",
    "Model",
    "ModelCompat",
    "ModelCost",
    "ModelCostRates",
    "ModelCostTier",
    "ModelInputLimits",
    "ModelPromptCache",
    "ModelThinkingLevel",
    "ModelType",
    "NestedToolCallRecord",
    "NestedToolCalls",
    "ProviderEnv",
    "ProviderHeaders",
    "ProviderId",
    "ProviderRequestOptions",
    "ProviderResponse",
    "ProviderStreamOptions",
    "ProviderStreams",
    "SessionAffinityFormat",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "SystemMessage",
    "TextContent",
    "ThinkingBudgets",
    "ThinkingContent",
    "ThinkingLevel",
    "ThinkingLevelMap",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolReference",
    "ToolResultMessage",
    "TranscriptContext",
    "Transport",
    "Usage",
    "UserMessage",
]

# --------------------------------------------------------------------------------------
# 标量与别名
# --------------------------------------------------------------------------------------

#: 任何可用 JSON 表示的值。运行时不限制用途，与 TypeScript 一致。
JsonValue: TypeAlias = Any
#: JSON 对象。
JsonObject: TypeAlias = dict[str, Any]

#: 内置 chat API 标识符。
#:
#: 用 ``Literal`` 而非 ``StrEnum``：运行时从不把这些值当值使用，因此该别名是
#: TypeScript 字符串字面量联合的直接 Python 对应——无类、无 import 开销，同时为
#: 类型检查器保留同样的穷举信息。``KNOWN_APIS`` 在运行时发现场景暴露同一集合。
KnownApi: TypeAlias = Literal[
    "openai-responses",
    "anthropic-messages",
    "google-generative-ai",
    "pi-messages",
]

#: 所有内置 chat API id，派生自 :data:`KnownApi`，两者不会漂移。
KNOWN_APIS: tuple[KnownApi, ...] = get_args(KnownApi)

#: chat API id：已知 API 或自定义字符串（上游为 ``KnownApi | (string & {})``）。
Api: TypeAlias = str

#: 内置 provider 标识符。
KnownProvider: TypeAlias = Literal[
    "anthropic",
    "google",
    "openai",
    "deepseek",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "xiaomi",
]

#: 所有内置 provider id，派生自 :data:`KnownProvider`。
KNOWN_PROVIDERS: tuple[KnownProvider, ...] = get_args(KnownProvider)

#: provider 的 id。
ProviderId: TypeAlias = str

#: 简单请求的 provider 无关工具选择。
ToolChoice: TypeAlias = Literal["auto", "none"]

#: 支持该功能的模型所请求的推理努力程度。
#:
#: ``xhigh`` 和 ``max`` 仅受部分模型系列支持；判断具体模型是否支持请使用
#: 该模型的 thinking 档位元数据。
ThinkingLevel: TypeAlias = Literal["minimal", "low", "medium", "high", "xhigh", "max"]

#: 含显式 ``off`` 状态的 thinking 档位。
ModelThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

#: 将 pi thinking 档位映射为 provider/模型特定值；``None`` 表示不支持。
ThinkingLevelMap: TypeAlias = dict[ModelThinkingLevel, str | None]


@dataclass
class ThinkingBudgets:
    """各 thinking 档位的 token 预算（仅基于 token 的 provider）。"""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


#: prompt 缓存保留偏好。
CacheRetention: TypeAlias = Literal["none", "short", "long"]

#: 各保留档位下尽力而为的 prompt 缓存存活秒数。
ModelPromptCache: TypeAlias = dict[str, int]

#: 首选 provider 传输方式。
Transport: TypeAlias = Literal["sse", "websocket", "websocket-cached", "auto"]

#: provider 作用域的环境变量覆盖；其值优先于 ``os.environ``。
ProviderEnv: TypeAlias = dict[str, str]
#: provider 请求头；``None`` 值会抑制同名的 provider 默认值。
ProviderHeaders: TypeAlias = dict[str, str | None]
#: provider HTTP 请求的可选 fetch 实现。
FetchFunction: TypeAlias = Callable[..., Awaitable[Any]]

#: 会话亲和（session affinity）头约定。
SessionAffinityFormat: TypeAlias = Literal["openai", "openai-nosession", "openrouter"]

#: OpenAI 受限采样的 grammar 变体。
GrammarFormat: TypeAlias = Literal["openai_lark", "openai_regex"]

#: 同一 grammar 的 provider 特定编码。
GrammarVariants: TypeAlias = dict[str, str]


# --------------------------------------------------------------------------------------
# 请求选项
# --------------------------------------------------------------------------------------


@dataclass
class ProviderResponse:
    """请求回调所观察到的 HTTP 响应。"""

    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)


def _option_field_names() -> frozenset[str]:
    """请求选项继承体系中任意位置声明的所有字段名。

    基于实时类树计算，使 :meth:`ProviderRequestOptions.__getattr__` 中的容忍
    逻辑无需手工维护列表。
    """
    names: set[str] = set()
    stack: list[type] = [ProviderRequestOptions]
    seen: set[type] = set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        names.update(dataclass_field.name for dataclass_field in fields(cls))
        stack.extend(cls.__subclasses__())
    return frozenset(names)


@dataclass
class ProviderRequestOptions:
    """provider 请求共有的认证、传输与生命周期回调。

    未知的公开属性读作 ``None``。上游的 option 接口是结构化的：adapter 可以读取
    只有它自己更宽的 option 类型才声明的字段（``tool_choice``、``service_tier``、
    ``client``、``deferred``……），而调用方传入的是更窄的基类对象，JavaScript 在
    这里会得到 ``undefined``；Python 则会抛 ``AttributeError``，表现为虚假的
    stream 错误。以下划线开头的名字仍会抛出，使拼写错误和私有属性探测大声失败。
    """

    signal: AbortSignal | None = None
    telemetry_context: Any = None
    api_key: str | None = None
    fetch: FetchFunction | None = None
    env: ProviderEnv | None = None
    on_payload: Callable[..., Any] | None = None
    on_response: Callable[..., Any] | None = None
    headers: ProviderHeaders | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None

    def __getattr__(self, name: str) -> Any:
        """把未声明的*选项*字段读作 ``None``（见类 docstring）。

        仅容忍本继承体系中其他 option 类型的真实字段名。任意其他名字——拼写
        错误，或绝不能泄露给 provider 的 Models 层字段（如 ``transform_headers``）
        ——仍会抛出，因此 ``hasattr`` 继续可用作泄露探测器。

        Args:
            name: 被访问的属性名。

        Returns:
            该字段在本继承体系中存在但未设置时为 ``None``。

        Raises:
            AttributeError: 名字以下划线开头，或不属于任何已知 option 字段。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _option_field_names():
            return None
        raise AttributeError(name)


@dataclass
class StreamOptions(ProviderRequestOptions):
    """所有 provider 的 ``stream`` 都接受的选项。"""

    on_provider_stream_event: Callable[..., Any] | None = None
    temperature: float | None = None
    sampling_params: dict[str, Any] | None = None
    max_tokens: int | None = None
    transport: Transport | str | None = None
    cache_retention: CacheRetention | str | None = None
    session_id: str | None = None
    websocket_connect_timeout_ms: int | None = None
    metadata: dict[str, Any] | None = None
    #: pi 未建模的 provider 特定选项，对应 ``ProviderStreamOptions = StreamOptions & Record<string, unknown>``。
    extra: dict[str, Any] = field(default_factory=dict, repr=False)


#: 含未建模 provider 扩展字段的 stream 选项。
ProviderStreamOptions: TypeAlias = StreamOptions


@dataclass
class DeferredFetchOptions(ProviderRequestOptions):
    """获取 deferred provider 响应的选项。"""

    wait: int | None = None


#: 尽力取消 deferred 响应的选项。
DeferredCancelOptions: TypeAlias = ProviderRequestOptions


@dataclass
class SimpleStreamOptions(StreamOptions):
    """``streamSimple``/``completeSimple`` 接受的统一选项。"""

    tool_choice: ToolChoice | None = None
    reasoning: ThinkingLevel | str | None = None
    deferred: bool | dict[str, Any] | None = None
    thinking_budgets: ThinkingBudgets | None = None


# --------------------------------------------------------------------------------------
# 消息内容
# --------------------------------------------------------------------------------------


@content_block("text")
@dataclass
class TextContent:
    """纯文本块。"""

    text: str = ""
    type: str = "text"
    #: provider 消息元数据（旧版 id 字符串或 ``TextSignatureV1`` JSON）。
    text_signature: str | None = field(default=None, metadata={"alias": "textSignature"})


@content_block("thinking")
@dataclass
class ThinkingContent:
    """推理块，可能被 provider 安全过滤器脱敏。"""

    thinking: str = ""
    type: str = "thinking"
    #: provider 特定的不透明或序列化推理回放数据。
    thinking_signature: str | None = field(default=None, metadata={"alias": "thinkingSignature"})
    #: 为 true 时载荷保存在 ``thinking_signature`` 中，必须原样回放。
    redacted: bool | None = None


@content_block("toolCall")
@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""

    id: str = ""
    name: str = ""
    arguments: JsonObject = field(default_factory=dict)
    type: str = "toolCall"
    #: Google 特有的不透明签名，用于复用思考上下文。
    thought_signature: str | None = field(default=None, metadata={"alias": "thoughtSignature"})
    #: OpenAI Responses 中动态加载或带命名空间工具的命名空间。
    namespace: str | None = None


#: 可出现在 assistant 消息中的块。
AssistantContent: TypeAlias = TextContent | ThinkingContent | ToolCall
#: 可出现在 user 或 tool-result 消息中的块。
UserContent: TypeAlias = TextContent


class StopReason(StrEnum):
    """assistant 响应结束的原因。"""

    PENDING = "pending"
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "toolUse"
    ERROR = "error"
    ABORTED = "aborted"
    DEFERRED = "deferred"


@dataclass
class Cost:
    """单次请求的货币成本，单位美元。"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = field(default=0.0, metadata={"alias": "cacheRead"})
    cache_write: float = field(default=0.0, metadata={"alias": "cacheWrite"})
    total: float = 0.0


@dataclass
class Usage:
    """单次请求的 token 统计。"""

    input: int = 0
    output: int = 0
    cache_read: int = field(default=0, metadata={"alias": "cacheRead"})
    cache_write: int = field(default=0, metadata={"alias": "cacheWrite"})
    total_tokens: int = field(default=0, metadata={"alias": "totalTokens"})
    cost: Cost = field(default_factory=Cost)
    #: ``cache_write`` 中以 1h 保留期写入的部分（仅 Anthropic）。
    cache_write_1h: int | None = field(default=None, metadata={"alias": "cacheWrite1h"})
    #: provider 上报时的推理 token 数，为 ``output`` 的子集。
    reasoning: int | None = None


@dataclass
class DeferredHandle:
    """描述异步 deferred 响应的 provider 令牌。"""

    provider: str = ""
    model_id: str = field(default="", metadata={"alias": "modelId"})
    api: str = ""
    #: provider 令牌，例如 response id，或 batch id 加 row id。
    id: str = ""
    expires_at: int | None = field(default=None, metadata={"alias": "expiresAt"})
    poll_after_ms: int | None = field(default=None, metadata={"alias": "pollAfterMs"})
    #: 重建最终 assistant 消息所需的 provider 转换数据。
    data: JsonValue = None


@dataclass
class NestedToolCallRecord:
    """工具运行期间发起的另一次工具调用。"""

    id: str = ""
    name: str = ""
    status: str = "ok"
    arguments: JsonObject | None = None
    arguments_bytes: int | None = field(default=None, metadata={"alias": "argumentsBytes"})
    duration_ms: int | None = field(default=None, metadata={"alias": "durationMs"})
    error: str | None = None


@dataclass
class NestedToolCalls:
    """工具嵌套调用的有界记录；不记录结果。"""

    calls: list[NestedToolCallRecord] = field(default_factory=list)
    complete: bool = True


@dataclass
class SystemMessage:
    """transcript 中某一时刻的系统指令与工具声明。"""

    content: str | list[TextContent] = ""
    timestamp: int = 0
    role: str = "system"
    #: 命名的有序 prompt 分节，逐字渲染在 ``content`` 之后。
    sections: dict[str, str | None] | None = None
    #: 此刻起可用工具的完整定义。
    tools_added: list[Tool] | None = field(default=None, metadata={"alias": "toolsAdded"})
    #: 此刻起不再可用的工具。
    tools_removed: list[ToolReference] | None = field(default=None, metadata={"alias": "toolsRemoved"})


@dataclass
class UserMessage:
    """一轮用户消息。"""

    content: str | list[UserContent] = ""
    timestamp: int = 0
    role: str = "user"


@dataclass
class AssistantMessage:
    """模型响应，完整或部分。"""

    api: Api = ""
    provider: ProviderId = ""
    model: str = ""
    content: list[AssistantContent] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason | str = field(default=StopReason.PENDING, metadata={"alias": "stopReason"})
    timestamp: int = 0
    role: str = "assistant"
    #: provider 上报的、与所请求 ``model`` 不同的具体模型。
    response_model: str | None = field(default=None, metadata={"alias": "responseModel"})
    #: provider 特定的响应/消息标识符。
    response_id: str | None = field(default=None, metadata={"alias": "responseId"})
    #: 本次响应实际使用的 provider 原生努力档位。
    provider_thinking_level: str | None = field(default=None, metadata={"alias": "providerThinkingLevel"})
    #: agent 循环为本次响应请求的 pi thinking 档位。
    thinking_level: ModelThinkingLevel | str | None = field(default=None, metadata={"alias": "thinkingLevel"})
    #: 针对失败与恢复的脱敏 provider/运行时诊断信息。
    diagnostics: list[AssistantMessageDiagnostic] | None = None
    deferred: DeferredHandle | None = None
    error_message: str | None = field(default=None, metadata={"alias": "errorMessage"})
    raw_stop_reason: str | None = field(default=None, metadata={"alias": "rawStopReason"})
    #: provider 指示模型已显式结束本轮回复。
    end_turn: bool | None = field(default=None, metadata={"alias": "endTurn"})


@dataclass
class ToolResultMessage:
    """执行一次工具调用的结果。"""

    tool_call_id: str = field(default="", metadata={"alias": "toolCallId"})
    tool_name: str = field(default="", metadata={"alias": "toolName"})
    content: list[UserContent] = field(default_factory=list)
    is_error: bool = field(default=False, metadata={"alias": "isError"})
    timestamp: int = 0
    role: str = "toolResult"
    details: JsonValue = None
    #: 工具执行本身的用量；不计入主 LLM 上下文统计。
    usage: Usage | None = None
    #: 该工具对其他工具发起的调用，保留作会话记录。
    nested_calls: NestedToolCalls | None = field(default=None, metadata={"alias": "nestedCalls"})


#: 模型可见的任意消息。
Message: TypeAlias = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


# --------------------------------------------------------------------------------------
# 工具与上下文
# --------------------------------------------------------------------------------------


@dataclass
class JsonSchemaConstrainedSampling:
    """把工具的 JSON Schema 作为服务端受限采样 grammar 提供。"""

    strict: str = "prefer"
    type: str = "json_schema"


@dataclass
class GrammarConstrainedSampling:
    """按格式各提供一份 provider 特定 grammar。"""

    variants: GrammarVariants = field(default_factory=dict)
    type: str = "grammar"


#: 工具的 provider 端受限采样配置。
ConstrainedSamplingConfig: TypeAlias = JsonSchemaConstrainedSampling | GrammarConstrainedSampling


@dataclass
class Tool:
    """向模型展示的工具声明。"""

    name: str = ""
    description: str = ""
    #: 描述工具参数的 JSON Schema。
    parameters: dict[str, Any] = field(default_factory=dict)
    constrained_sampling: ConstrainedSamplingConfig | bool | None = field(
        default=None, metadata={"alias": "constrainedSampling"}
    )


@dataclass
class ToolReference:
    """按名称引用一个工具。"""

    name: str = ""


@dataclass
class Context:
    """调用方提供的请求上下文。

    ``system_prompt`` 和 ``tools`` 是首条 system 消息的简写；
    :func:`pi_ai.utils.transcript.normalize_context` 会把它们折叠为一条。
    """

    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = field(default=None, metadata={"alias": "systemPrompt"})
    tools: list[Tool] | None = None


@dataclass
class TranscriptContext:
    """规范化后的请求上下文：只有 :func:`normalize_context` 会产出这种形态。

    prompt 与工具声明由 transcript 的 system 消息承载，绝不放在上下文本体中。
    """

    messages: list[Message] = field(default_factory=list)

    def __iter__(self):
        """按原顺序迭代上下文中的消息。

        Returns:
            消息列表的迭代器。
        """
        return iter(self.messages)

    def __len__(self) -> int:
        """返回上下文中的消息条数。

        Returns:
            消息条数。
        """
        return len(self.messages)


# --------------------------------------------------------------------------------------
# stream 事件
# --------------------------------------------------------------------------------------


@dataclass
class StartEvent:
    """一次生成已开始。"""

    partial: AssistantMessage
    type: str = "start"


@dataclass
class TextStartEvent:
    """一个文本块已开始。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "text_start"


@dataclass
class TextDeltaEvent:
    """一个文本块增长。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "text_delta"


@dataclass
class TextEndEvent:
    """一个文本块结束，并携带权威内容。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "text_end"


@dataclass
class ThinkingStartEvent:
    """一个思考块已开始。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "thinking_start"


@dataclass
class ThinkingDeltaEvent:
    """一个思考块增长。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "thinking_delta"


@dataclass
class ThinkingEndEvent:
    """一个思考块结束，并携带权威内容。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "thinking_end"


@dataclass
class ToolCallStartEvent:
    """一次工具调用已开始。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "toolcall_start"


@dataclass
class ToolCallDeltaEvent:
    """工具调用的 JSON 参数在增长。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "toolcall_delta"


@dataclass
class ToolCallEndEvent:
    """一次工具调用结束。"""

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    tool_call: ToolCall = field(default_factory=ToolCall, metadata={"alias": "toolCall"})
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "toolcall_end"


@dataclass
class DoneEvent:
    """一次成功的生成已结束。"""

    reason: StopReason | str = StopReason.STOP
    message: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "done"


@dataclass
class ErrorEvent:
    """一次生成以错误或中止结束。"""

    reason: StopReason | str = StopReason.ERROR
    error: AssistantMessage = field(default_factory=AssistantMessage)
    type: str = "error"


#: :class:`~pi_ai.utils.event_stream.AssistantMessageEventStream` 的事件协议。
AssistantMessageEvent: TypeAlias = (
        StartEvent
        | TextStartEvent
        | TextDeltaEvent
        | TextEndEvent
        | ThinkingStartEvent
        | ThinkingDeltaEvent
        | ThinkingEndEvent
        | ToolCallStartEvent
        | ToolCallDeltaEvent
        | ToolCallEndEvent
        | DoneEvent
        | ErrorEvent
)


# --------------------------------------------------------------------------------------
# provider 兼容性标志
# --------------------------------------------------------------------------------------


@dataclass
class ModelCompat:
    """provider adapter 理解的兼容性覆盖项。

    TypeScript 类型是按 API 选择字段的条件类型；运行时它就是一个普通对象，
    本 dataclass 建模的正是该对象。每个 adapter 只读取自己文档中列出的字段。
    """

    # -- 通用兼容开关 ----------------------------------------------------------------
    #: provider 是否支持 ``developer`` 角色（而非 ``system``）。
    supports_developer_role: bool | None = field(default=None, metadata={"alias": "supportsDeveloperRole"})
    #: provider 是否支持带 grammar 格式的 OpenAI 自定义工具。
    supports_openai_grammar_tools: bool | None = field(
        default=None, metadata={"alias": "supportsOpenAIGrammarTools"}
    )
    #: 具体模型是否接受对话中途的 system/developer 消息。
    supports_mid_convo_system_messages: bool | None = field(
        default=None, metadata={"alias": "supportsMidConvoSystemMessages"}
    )
    #: 是否接受对话中途的 ``tool_addition``/``tool_removal`` 块。
    supports_mid_convo_tool_changes: bool | None = field(
        default=None, metadata={"alias": "supportsMidConvoToolChanges"}
    )
    #: provider 是否支持 Anthropic 严格工具 schema。
    supports_strict_tools: bool | None = field(default=None, metadata={"alias": "supportsStrictTools"})
    #: 模型传输是否支持仅 effort 的 system 消息与 thinking 绑定。
    supports_mid_convo_effort: bool | None = field(default=None, metadata={"alias": "supportsMidConvoEffort"})
    #: provider 是否接受按工具设置的 ``eager_input_streaming``。
    supports_eager_tool_input_streaming: bool | None = field(
        default=None, metadata={"alias": "supportsEagerToolInputStreaming"}
    )
    #: 工具定义上的 Anthropic 风格 ``cache_control`` 标记。
    supports_cache_control_on_tools: bool | None = field(
        default=None, metadata={"alias": "supportsCacheControlOnTools"}
    )
    #: 模型是否接受 Anthropic 的 ``temperature`` 请求字段。
    supports_temperature: bool | None = field(default=None, metadata={"alias": "supportsTemperature"})
    #: 无论 model id 为何都强制自适应思考。
    force_adaptive_thinking: bool | None = field(default=None, metadata={"alias": "forceAdaptiveThinking"})
    #: 空思考签名回放为 ``signature: ""`` 而非文本。
    allow_empty_signature: bool | None = field(default=None, metadata={"alias": "allowEmptySignature"})
    #: 是否从 ``options.session_id`` 发送会话亲和数据。
    send_session_affinity_headers: bool | None = field(
        default=None, metadata={"alias": "sendSessionAffinityHeaders"}
    )
    #: 会话亲和头格式。
    session_affinity_format: SessionAffinityFormat | str | None = field(
        default=None, metadata={"alias": "sessionAffinityFormat"}
    )
    #: provider 是否支持长 prompt 缓存保留。
    supports_long_cache_retention: bool | None = field(
        default=None, metadata={"alias": "supportsLongCacheRetention"}
    )
    #: provider 是否支持工具定义中的 ``strict``。
    supports_strict_mode: bool | None = field(default=None, metadata={"alias": "supportsStrictMode"})

    # -- OpenAI Responses 专用兼容字段 -------------------------------------------------
    #: 模型是否支持以消息为锚点的 ``additional_tools`` 输入项。
    supports_additional_tools: bool | None = field(default=None, metadata={"alias": "supportsAdditionalTools"})
    #: 模型是否支持客户端执行的工具搜索。
    supports_tool_search: bool | None = field(default=None, metadata={"alias": "supportsToolSearch"})
    #: 模型是否接受 ``prompt_cache_options``（GPT-5.6+）。
    supports_explicit_prompt_cache_mode: bool | None = field(
        default=None, metadata={"alias": "supportsExplicitPromptCacheMode"}
    )
    #: provider 是否接受 ``max_output_tokens`` 参数。
    supports_max_output_tokens: bool | None = field(
        default=None, metadata={"alias": "supportsMaxOutputTokens"}
    )

    # -- Anthropic Messages 专用兼容字段 ----------------------------------------------
    #: Anthropic 在 ``fallbacks`` 中接受的服务端拒绝回退模型。
    allowed_fallback_models: list[AnthropicAllowedFallbackModel] | None = field(
        default=None, metadata={"alias": "allowedFallbackModels"}
    )


@dataclass
class AnthropicAllowedFallbackModel:
    """允许的服务端拒绝回退目标，附带本地定价元数据。"""

    provider: ProviderId = ""
    model: str = ""
    cost: ModelCost = field(default_factory=lambda: ModelCost())


# --------------------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------------------


@dataclass
class ModelCostRates:
    """每百万 token 的美元价格。"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = field(default=0.0, metadata={"alias": "cacheRead"})
    cache_write: float = field(default=0.0, metadata={"alias": "cacheWrite"})


@dataclass
class ModelCostTier(ModelCostRates):
    """请求级别的定价档位。"""

    input_tokens_above: int = field(default=0, metadata={"alias": "inputTokensAbove"})


@dataclass
class ModelCost(ModelCostRates):
    """基础费率加上可选的请求级别定价档位。"""

    tiers: list[ModelCostTier] | None = None


@dataclass
class ModelInputLimits:
    """provider 的请求大小与图像限制。"""

    max_request_bytes: int | None = field(default=None, metadata={"alias": "maxRequestBytes"})


@dataclass
class BaseModel:
    """所有目录条目共有的字段。"""

    id: str = ""
    name: str = ""
    api: str = ""
    provider: ProviderId = ""
    base_url: str = field(default="", metadata={"alias": "baseUrl"})
    input: list[str] = field(default_factory=list)
    cost: ModelCost = field(default_factory=ModelCost)
    type: ModelType = "chat"
    input_limits: ModelInputLimits | None = field(default=None, metadata={"alias": "inputLimits"})
    headers: dict[str, str] | None = None


@dataclass
class Model(BaseModel):
    """chat 模型，可与 ``stream`` 等方法配合使用。"""

    reasoning: bool = False
    context_window: int = field(default=0, metadata={"alias": "contextWindow"})
    max_tokens: int = field(default=0, metadata={"alias": "maxTokens"})
    thinking_level_map: ThinkingLevelMap | None = field(default=None, metadata={"alias": "thinkingLevelMap"})
    prompt_cache: ModelPromptCache | None = field(default=None, metadata={"alias": "promptCache"})
    sampling_params: dict[str, Any] | None = field(default=None, metadata={"alias": "samplingParams"})
    compat: ModelCompat | None = None


#: 目录条目：本项目只保留 chat 模型。
AnyModel: TypeAlias = Model

#: 目录条目的用途类型。本项目只支持 chat 模型：图片生成与分类模型已随
#: ``ImageApi``/``ClassifierApi`` 一并删除，不再出现在类型面上。
ModelType: TypeAlias = Literal["chat"]


# --------------------------------------------------------------------------------------
# 实现契约
# --------------------------------------------------------------------------------------


@runtime_checkable
class ProviderStreams(Protocol):
    """API 实现模块的统一 stream 契约。"""

    def stream(
            self,
            model: Model,
            context: TranscriptContext,
            options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """以流式方式完成一次模型请求。

        Args:
            model: 目标模型。
            context: 规范化后的请求上下文。
            options: 可选的 stream 选项。

        Returns:
            逐事件产出结果的 assistant message 事件流。
        """
        ...

    def stream_simple(
            self,
            model: Model,
            context: TranscriptContext,
            options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """以流式方式完成一次简化的模型请求。

        Args:
            model: 目标模型。
            context: 规范化后的请求上下文。
            options: 可选的简化 stream 选项。

        Returns:
            逐事件产出结果的 assistant message 事件流。
        """
        ...

    def fetch_deferred(
            self,
            model: Model,
            handle: DeferredHandle,
            options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        """拉取一次延迟响应的结果流。

        Args:
            model: 目标模型。
            handle: 延迟响应的句柄。
            options: 可选的拉取选项。

        Returns:
            逐事件产出结果的 assistant message 事件流。
        """
        ...

    async def cancel_deferred(
            self,
            model: Model,
            handle: DeferredHandle,
            options: DeferredCancelOptions | None = None,
    ) -> None:
        """取消一次尚未完成的延迟响应。

        Args:
            model: 目标模型。
            handle: 延迟响应的句柄。
            options: 可选的取消选项。
        """
        ...


#: 类型化 chat stream 函数的签名。
StreamFunction: TypeAlias = Callable[..., AssistantMessageEventStream]

#: 回放辅助函数可检查的消息序列。
TranscriptMessages: TypeAlias = Sequence[Any]
