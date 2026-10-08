"""``pi-agent`` 的公开类型。

移植自 ``packages/agent/src/types.ts``。所有结构化类型均为 :mod:`dataclasses`
dataclass，字段为 snake_case，其原始 camelCase 线上名记录在字段元数据
（``alias``）中，遵循 ``PORTING.md`` 的移植约定。从不作为数据实例化的结构契约
（``StreamFn``、``ToolCallHooks``）用 :class:`typing.Protocol` 表示。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeAlias

from pi_ai.types import (
    AssistantMessageEvent,
    JsonValue,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)
from pi_ai.utils.transcript import get_current_system_prompt

__all__ = [
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEndEvent",
    "AgentEvent",
    "AgentEventSink",
    "AgentInitialState",
    "AgentLoopConfig",
    "AgentLoopTurnUpdate",
    "AgentMessage",
    "AgentRequestUpdate",
    "AgentStartEvent",
    "AgentState",
    "AgentTool",
    "AgentToolCall",
    "AgentToolCallOutcome",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "AgentTurnAction",
    "AgentTurnContext",
    "AgentTurnDecision",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "FinishTurn",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "PrepareNextTurnContext",
    "PrepareRequest",
    "PrepareRequestContext",
    "QueueMode",
    "StreamFn",
    "ThinkingLevel",
    "ToolCallHooks",
    "ToolExecutionEndEvent",
    "ToolExecutionMode",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]


class StreamFn(Protocol):
    """agent 循环使用的 stream 函数；``Models.stream_simple`` 满足该协议。

    循环传入的是规范化后的会话记录：system prompt 与工具声明始终由记录中的
    system 消息承载，而不是 ``context.system_prompt`` 或 ``context.tools``。

    契约：
    - 对请求/模型/运行时故障不得抛出异常或返回被拒绝的 awaitable。
    - 必须返回 :class:`~pi_ai.utils.event_stream.AssistantMessageEventStream`
      （直接返回或作为 awaitable 返回）。
    - 故障必须编码进返回的 stream：由协议事件加最终的
      :class:`~pi_ai.types.AssistantMessage` 表达，其 ``stop_reason`` 为
      ``"error"`` 或 ``"aborted"``，并带 ``error_message``。
    """

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None = None,
    ) -> Any:
        """针对 ``context`` 为 ``model`` 流式产出一条 assistant 响应。

        Args:
            model: 要调用的模型。
            context: 规范化后的会话记录，system prompt 与工具声明都在其中。
            options: 额外的 stream 选项；为 ``None`` 时使用模型默认设置。

        Returns:
            该次响应的 assistant 消息事件 stream（或它的 awaitable）。
        """
        ...


#: 单条 assistant 消息中各 tool call 的执行方式。
#:
#: - ``"sequential"``：每个 tool call 依次完成准备、执行与收尾，再开始下一个。
#: - ``"parallel"``：tool call 依次准备，随后允许并发执行的工具同时运行。
#:   ``tool_execution_end`` 按工具完成顺序在各自收尾后发出，而 tool-result
#:   消息则按 assistant 中的先后顺序稍后产出。
ToolExecutionMode: TypeAlias = Literal["sequential", "parallel"]


#: 队列 drain 点一次注入多少条排队的 user 消息。
#:
#: - ``"all"``：在该 drain 点取出并注入全部排队消息。
#: - ``"one-at-a-time"``：只取出并注入最旧的一条，其余留待后续 drain 点。
QueueMode: TypeAlias = Literal["all", "one-at-a-time"]


#: 支持思考/推理的模型的 thinking 档位。
#:
#: ``xhigh`` 与 ``max`` 仅少数模型系列支持；判断具体模型是否支持请使用
#: pi-ai 提供的模型 thinking 档位元数据。
ThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


#: assistant 消息产出的单个 tool-call 内容块。
AgentToolCall: TypeAlias = ToolCall


@dataclass
class BeforeToolCallResult:
    """``before_tool_call`` 返回的结果。

    ``block=True`` 阻止工具执行，循环改为产出一条错误 tool result；
    ``reason`` 即该错误结果展示的文本，省略时使用默认的拦截提示。
    ``terminate`` 表示本次调用被拦截后 agent 应在当前工具批次结束时停止；
    只有当批次内每个收尾的 tool result 都置为 true 时才会提前终止。

    Attributes:
        block: 为真时阻止工具执行，循环改为产出一条错误 tool result。
        reason: 工具被拦截时展示给模型的错误文本；省略时使用默认拦截提示。
        terminate: 为真时表示本次拦截后 agent 应在当前工具批次结束时停止。
    """

    block: bool | None = None
    reason: str | None = None
    terminate: bool | None = None


@dataclass
class AfterToolCallResult:
    """``after_tool_call`` 返回的部分覆盖结果。

    合并语义按字段逐一处理：

    - ``content``：提供时整体替换 tool result 的 content 数组
    - ``details``：提供时整体替换 tool result 的 details 值
    - ``is_error``：提供时替换 tool result 的错误标志
    - ``usage``：提供时替换 tool result 的 usage
    - ``terminate``：提供时替换提前终止提示
    - ``structured_content``：提供时替换结构化内容。若提供了 ``content`` 而未提供
      它，结构化内容会被丢弃，因为它可能已与 content 不再匹配；要保留它需与
      ``content`` 一并返回。

    其余省略的字段保留执行所得 tool result 的原值。
    ``content``、``details``、``usage`` 不做深度合并。

    Attributes:
        content: 替换用的 content 数组。
        details: 替换用的 details 值。
        structured_content: 替换用的结构化内容。
        is_error: 替换用的错误标志。
        usage: 替换用的 usage。
        terminate: 替换用的提前终止提示。
    """

    content: list[TextContent] | None = None
    details: Any = None
    structured_content: JsonValue = field(default=None, metadata={"alias": "structuredContent"})
    is_error: bool | None = field(default=None, metadata={"alias": "isError"})
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass
class BeforeToolCallContext:
    """传给 ``before_tool_call`` 的上下文。

    Attributes:
        assistant_message: 发起本次 tool call 的 assistant 消息。
        tool_call: 来自 ``assistant_message.content`` 的原始 tool call 块。
        args: 按目标工具 schema 校验后的参数。
        context: 准备 tool call 时的当前 agent 上下文。
    """

    #: 发起本次 tool call 的 assistant 消息。
    assistant_message: Any = field(default=None, metadata={"alias": "assistantMessage"})
    #: 来自 ``assistant_message.content`` 的原始 tool call 块。
    tool_call: AgentToolCall = field(default_factory=ToolCall, metadata={"alias": "toolCall"})
    #: 按目标工具 schema 校验后的参数。
    args: Any = None
    #: 准备 tool call 时的当前 agent 上下文。
    context: AgentContext | None = None


@dataclass
class AfterToolCallContext:
    """传给 ``after_tool_call`` 的上下文。

    Attributes:
        assistant_message: 发起本次 tool call 的 assistant 消息。
        tool_call: 来自 ``assistant_message.content`` 的原始 tool call 块。
        args: 按目标工具 schema 校验后的参数。
        result: 应用任何 ``after_tool_call`` 覆盖之前的执行结果。
        is_error: 执行结果当前是否被视作错误。
        context: 执行 tool call 时的当前 agent 上下文。
    """

    assistant_message: Any = field(default=None, metadata={"alias": "assistantMessage"})
    tool_call: AgentToolCall = field(default_factory=ToolCall, metadata={"alias": "toolCall"})
    args: Any = None
    #: 应用任何 ``after_tool_call`` 覆盖之前的执行结果。
    result: AgentToolResult | None = None
    #: 执行结果当前是否被视作错误。
    is_error: bool = field(default=False, metadata={"alias": "isError"})
    context: AgentContext | None = None


@dataclass
class AgentTurnContext:
    """传给回合完成回调的上下文。

    Attributes:
        message: 完成本回合的 assistant 消息。
        tool_results: 本回合产出的 tool result 消息。
        context: 追加本回合 assistant 消息与 tool result 之后的当前 agent 上下文。
        new_messages: 若循环在此刻退出，本次调用将返回的消息；prompt 运行包含初始
            prompt 消息，续跑运行不包含运行前已存在的上下文消息。
    """

    #: 完成本回合的 assistant 消息。
    message: Any = None
    #: 本回合产出的 tool result 消息。
    tool_results: list[ToolResultMessage] = field(default_factory=list, metadata={"alias": "toolResults"})
    #: 追加本回合 assistant 消息与 tool result 之后的当前 agent 上下文。
    context: AgentContext | None = None
    #: 若循环在此刻退出，本次调用将返回的消息。prompt 运行包含初始 prompt 消息；
    #: 续跑运行不包含运行前已存在的上下文消息。
    new_messages: list[Any] = field(default_factory=list, metadata={"alias": "newMessages"})


class AgentTurnAction(StrEnum):
    """``AgentTurnDecision`` 的动作。"""

    CONTINUE = "continue"
    END = "end"


@dataclass
class AgentTurnDecision:
    """:data:`FinishTurn` 返回的决策。

    返回 ``None`` 表示维持正常调度。

    Attributes:
        action: 回合结束后要执行的动作；默认 ``CONTINUE``。
    """

    action: AgentTurnAction | str = AgentTurnAction.CONTINUE


#: 在 assistant 回合及其全部 tool-result 消息完成之后、``turn_end`` 之前调用。
#: 正常回合返回 ``CONTINUE`` 会保证再发一次 provider 请求；tool-result、steering
#: 或 follow-up 的调度可以满足该请求且不额外增加请求，否则循环会用当前上下文再
#: 运行一轮。错误与中止响应仍是硬退出。Python 回调可以是同步或异步的。
FinishTurn: TypeAlias = Callable[..., Any]


@dataclass
class AgentLoopTurnUpdate:
    """agent 循环在再次发起 provider 请求前使用的替换运行时状态。

    Attributes:
        context: 下一次 provider 请求的上下文。
        messages: 下一次 provider 请求前要追加的消息，按正常生命周期事件产出。
        model: 下一次 provider 请求使用的模型。
        thinking_level: 下一次 provider 请求的 thinking 档位。
    """

    #: 下一次 provider 请求的上下文。
    context: AgentContext | None = None
    #: 下一次 provider 请求前要追加的消息，按正常生命周期事件产出。
    messages: list[Any] | None = None
    #: 下一次 provider 请求使用的模型。
    model: Model | None = None
    #: 下一次 provider 请求的 thinking 档位。
    thinking_level: ThinkingLevel | str | None = field(default=None, metadata={"alias": "thinkingLevel"})


@dataclass
class PrepareRequestContext:
    """发起会话型 provider 请求前一刻可见的运行时状态。

    Attributes:
        context: 当前 agent 上下文。
        model: 本次请求使用的模型。
        thinking_level: 本次请求使用的 thinking 档位。
    """

    context: AgentContext | None = None
    model: Model = field(default_factory=Model)
    thinking_level: ThinkingLevel | str = field(default="off", metadata={"alias": "thinkingLevel"})


@dataclass
class AgentRequestUpdate:
    """正在准备的 provider 请求的替换运行时状态。

    Attributes:
        context: 替换后的 agent 上下文；为 ``None`` 时沿用原值。
        model: 替换后的模型；为 ``None`` 时沿用原值。
        thinking_level: 替换后的 thinking 档位；为 ``None`` 时沿用原值。
    """

    context: AgentContext | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | str | None = field(default=None, metadata={"alias": "thinkingLevel"})


#: 在每次会话型 provider 请求（含首次）之前立即调用。
#: 运行该回调时，待处理消息已经追加并产出完毕。
PrepareRequest: TypeAlias = Callable[..., Any]


#: :attr:`AgentLoopConfig.prepare_next_turn` 的上下文。
PrepareNextTurnContext: TypeAlias = AgentTurnContext


@dataclass
class AgentLoopConfig(SimpleStreamOptions):
    """:func:`pi_agent.agent_loop.run_agent_loop` 的配置。

    扩展 :class:`~pi_ai.types.SimpleStreamOptions`，与 TypeScript 的
    ``AgentLoopConfig extends SimpleStreamOptions`` 一一对应。

    Attributes:
        model: 本次运行使用的模型。
        convert_to_llm: 把 ``AgentMessage`` 转换为 LLM 兼容消息的可选回调。
        transform_context: ``convert_to_llm`` 之前应用的上下文变换回调。
        get_api_key: 为每次 LLM 调用动态解析 API key 的回调。
        finish_turn: assistant 回合完成后的回合完成回调。
        prepare_request: 每次会话型 provider 请求之前调用的回调。
        prepare_next_turn: ``turn_end`` 之后、下一回合开始之前的回调。
        get_steering_messages: 返回运行途中要注入的 steering 消息的回调。
        get_follow_up_messages: 返回 agent 本将停止时要继续处理的 follow-up 消息的回调。
        tool_execution: 工具执行模式，默认 ``"parallel"``。
        before_tool_call: 工具执行前、参数校验完成后调用的回调。
        after_tool_call: 工具执行结束后、相关事件产出之前调用的回调。
    """

    model: Model = field(default_factory=Model)

    #: 在每次 LLM 调用前，把 ``AgentMessage`` 转换为 LLM 兼容的消息。
    #: 转换目标是 :data:`~pi_ai.types.Message`。
    #:
    #: 契约：不得抛出或拒绝，应返回安全的回退值。抛出会中断底层循环，
    #: 且不会产出正常事件序列。
    convert_to_llm: Callable[..., Any] | None = field(default=None, metadata={"alias": "convertToLlm"})

    #: 可选的上下文变换，在 ``convert_to_llm`` 之前应用。
    #:
    #: 契约：不得抛出或拒绝，应返回原消息或其他安全回退值。
    transform_context: Callable[..., Any] | None = field(default=None, metadata={"alias": "transformContext"})

    #: 为每次 LLM 调用动态解析 API key。
    #:
    #: 契约：不得抛出或拒绝；无可用 key 时返回 ``None``。
    get_api_key: Callable[..., Any] | None = field(default=None, metadata={"alias": "getApiKey"})

    #: 在 assistant 消息及全部 tool-result 消息产出之后、``turn_end`` 之前立即调用。
    finish_turn: FinishTurn | None = field(default=None, metadata={"alias": "finishTurn"})

    #: 在每次会话型 provider 请求（含首次）之前立即调用。该 hook 不轮询队列。
    prepare_request: PrepareRequest | None = field(default=None, metadata={"alias": "prepareRequest"})

    #: 在 ``turn_end`` 之后、循环将继续且下一回合即将开始之前调用。
    prepare_next_turn: Callable[..., Any] | None = field(default=None, metadata={"alias": "prepareNextTurn"})

    #: 返回运行途中要注入会话的 steering 消息。
    #:
    #: 契约：不得抛出或拒绝；无可用消息时返回 ``[]``。
    get_steering_messages: Callable[..., Any] | None = field(default=None, metadata={"alias": "getSteeringMessages"})

    #: 返回 agent 本将停止时要继续处理的 follow-up 消息。
    #:
    #: 契约：不得抛出或拒绝；无可用消息时返回 ``[]``。
    get_follow_up_messages: Callable[..., Any] | None = field(
        default=None, metadata={"alias": "getFollowUpMessages"}
    )

    #: 工具执行模式，默认 ``"parallel"``。
    tool_execution: ToolExecutionMode | str | None = field(default=None, metadata={"alias": "toolExecution"})

    #: 工具执行前、参数校验完成后调用。
    before_tool_call: Callable[..., Any] | None = field(default=None, metadata={"alias": "beforeToolCall"})

    #: 工具执行结束后、``tool_execution_end`` 与 tool-result 消息事件产出之前调用。
    after_tool_call: Callable[..., Any] | None = field(default=None, metadata={"alias": "afterToolCall"})


class CustomAgentMessages:
    """可扩展的自定义应用消息基类。

    TypeScript 应用通过 declaration merging 扩展 ``CustomAgentMessages`` 接口；
    Python 应用只需让对象携带 ``role`` 属性即可走同样的代码路径，
    因此本类仅为镜像 TypeScript 接面而存在。
    """


#: LLM 消息与应用自定义消息的联合。自定义消息是任何带 ``role`` 属性的对象，
#: 所以该别名放宽为 ``Any``。
AgentMessage: TypeAlias = Message | Any


#: 工具产出的最终或部分结果。
@dataclass
class AgentToolResult:
    """工具产出的最终或部分结果。

    Attributes:
        content: 返回给模型的文本内容。
        details: 供日志或 UI 渲染使用的任意结构化细节。
        structured_content: 符合工具 ``output_schema`` 的机器可读结果。
        usage: 最终一次工具执行自身的 usage（若可得）。
        is_error: 无需抛出即可报告失败；模型会把 ``content`` 视为错误结果。
        terminate: 提示 agent 在当前工具批次结束后停止。
    """

    #: 返回给模型的文本内容。
    content: list[TextContent] = field(default_factory=list)
    #: 供日志或 UI 渲染使用的任意结构化细节。
    details: Any = None
    #: 符合工具 ``output_schema`` 的机器可读结果，供程序化调用方使用。
    #: 不会发给模型；面向模型的结果仍是 ``content``。
    structured_content: JsonValue = field(default=None, metadata={"alias": "structuredContent"})
    #: 最终一次工具执行自身的 usage（若可得）。
    usage: Usage | None = None
    #: 无需抛出即可报告失败；模型会把 ``content`` 视为错误结果。
    is_error: bool | None = field(default=None, metadata={"alias": "isError"})
    #: 提示 agent 在当前工具批次结束后停止。
    terminate: bool | None = None


#: 各 hook 运行完毕后 tool call 的最终结果。
@dataclass
class AgentToolCallOutcome:
    """各 hook 运行完毕后 tool call 的最终结果。

    Attributes:
        tool_call: 本次执行的 tool call 块。
        result: 执行与各 hook 处理后得到的结果。
        is_error: 本次执行最终是否被视作错误。
    """

    tool_call: AgentToolCall = field(default_factory=ToolCall, metadata={"alias": "toolCall"})
    result: AgentToolResult = field(default_factory=AgentToolResult)
    is_error: bool = field(default=False, metadata={"alias": "isError"})


#: 工具用于流式产出部分执行更新的回调。
AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult], Any]


@dataclass
class AgentTool(Tool):
    """agent 运行时使用的工具定义。

    ``parameters`` 与 ``output_schema`` 为普通 JSON Schema 字典，
    是 TypeBox ``TSchema`` 在本移植中的替代。

    Attributes:
        label: 供 UI 展示的人类可读标签。
        prepare_arguments: schema 校验前处理原始 tool-call 参数的可选兼容垫片。
        output_schema: 成功结果中 ``structured_content`` 的 JSON Schema。
        execute: 执行 tool call 的回调，签名为 ``execute(tool_call_id, params,
            signal, on_update)``。
        replay: 持久意图已存在但结果未知时的恢复策略。
        execution_mode: 单个工具的执行模式覆盖。
    """

    #: 供 UI 展示的人类可读标签。
    label: str = ""
    #: schema 校验前处理原始 tool-call 参数的可选兼容垫片。
    prepare_arguments: Callable[[Any], Any] | None = field(default=None, metadata={"alias": "prepareArguments"})
    #: 成功结果中 ``structured_content`` 的 JSON Schema。
    output_schema: dict[str, Any] | None = field(default=None, metadata={"alias": "outputSchema"})
    #: 执行 tool call：``execute(tool_call_id, params, signal, on_update)``。
    execute: Callable[..., Any] | None = None
    #: 持久意图已存在但结果未知时的恢复策略。
    replay: str | None = None
    #: 单个工具的执行模式覆盖。
    execution_mode: ToolExecutionMode | str | None = field(default=None, metadata={"alias": "executionMode"})


@dataclass
class AgentContext:
    """传入底层 agent 循环的上下文快照。

    Attributes:
        messages: 模型可见的会话记录。
        tools: 本次运行可执行的工具。
    """

    #: 模型可见的会话记录。
    messages: list[Any] = field(default_factory=list)
    #: 本次运行可执行的工具。
    tools: list[AgentTool] | None = None


@dataclass
class AgentStartEvent:
    """运行开始。

    Attributes:
        type: 事件类型标签，固定为 ``"agent_start"``。
    """

    type: str = "agent_start"


@dataclass
class AgentEndEvent:
    """运行结束；一次运行派发的最后一个事件。

    Attributes:
        messages: 本次运行产出的全部消息。
        type: 事件类型标签，固定为 ``"agent_end"``。
    """

    messages: list[Any] = field(default_factory=list)
    type: str = "agent_end"


@dataclass
class TurnStartEvent:
    """一个回合（一次 assistant 响应加其 tool call/结果）开始。

    Attributes:
        type: 事件类型标签，固定为 ``"turn_start"``。
    """

    type: str = "turn_start"


@dataclass
class TurnEndEvent:
    """一个回合结束。

    Attributes:
        message: 本回合的 assistant 消息。
        tool_results: 本回合产出的 tool result 消息。
        type: 事件类型标签，固定为 ``"turn_end"``。
    """

    message: Any = None
    tool_results: list[ToolResultMessage] = field(default_factory=list, metadata={"alias": "toolResults"})
    type: str = "turn_end"


@dataclass
class MessageStartEvent:
    """一条 system、user、assistant 或 tool-result 消息开始。

    Attributes:
        message: 开始增长的消息。
        type: 事件类型标签，固定为 ``"message_start"``。
    """

    message: Any = None
    type: str = "message_start"


@dataclass
class MessageUpdateEvent:
    """assistant 消息增长；仅在流式期间对 assistant 消息发出。

    Attributes:
        message: 增长中的 assistant 消息。
        assistant_message_event: 触发本次更新的底层 assistant 消息事件。
        type: 事件类型标签，固定为 ``"message_update"``。
    """

    message: Any = None
    assistant_message_event: AssistantMessageEvent | None = field(
        default=None, metadata={"alias": "assistantMessageEvent"}
    )
    type: str = "message_update"


@dataclass
class MessageEndEvent:
    """一条消息结束。

    Attributes:
        message: 结束的消息。
        type: 事件类型标签，固定为 ``"message_end"``。
    """

    message: Any = None
    type: str = "message_end"


@dataclass
class ToolExecutionStartEvent:
    """一个 tool call 开始执行。

    Attributes:
        tool_call_id: 该 tool call 的 id。
        tool_name: 工具名。
        args: 校验后的调用参数。
        type: 事件类型标签，固定为 ``"tool_execution_start"``。
    """

    tool_call_id: str = field(default="", metadata={"alias": "toolCallId"})
    tool_name: str = field(default="", metadata={"alias": "toolName"})
    args: Any = None
    type: str = "tool_execution_start"


@dataclass
class ToolExecutionUpdateEvent:
    """工具流式产出了部分结果。

    Attributes:
        tool_call_id: 该 tool call 的 id。
        tool_name: 工具名。
        args: 校验后的调用参数。
        partial_result: 本次产出的部分结果。
        type: 事件类型标签，固定为 ``"tool_execution_update"``。
    """

    tool_call_id: str = field(default="", metadata={"alias": "toolCallId"})
    tool_name: str = field(default="", metadata={"alias": "toolName"})
    args: Any = None
    partial_result: Any = field(default=None, metadata={"alias": "partialResult"})
    type: str = "tool_execution_update"


@dataclass
class ToolExecutionEndEvent:
    """一个 tool call 执行结束。

    Attributes:
        tool_call_id: 该 tool call 的 id。
        tool_name: 工具名。
        result: 工具产出的结果。
        is_error: 本次执行是否被视作错误。
        type: 事件类型标签，固定为 ``"tool_execution_end"``。
    """

    tool_call_id: str = field(default="", metadata={"alias": "toolCallId"})
    tool_name: str = field(default="", metadata={"alias": "toolName"})
    result: Any = None
    is_error: bool = field(default=False, metadata={"alias": "isError"})
    type: str = "tool_execution_end"


#: :class:`~pi_agent.agent.Agent` 为 UI 更新派发的事件。
#:
#: ``agent_end`` 是一次运行的最后一个事件，但该事件上被 await 的
#: ``Agent.subscribe()`` 监听器仍计入运行结算；要等这些监听器结束后
#: agent 才进入空闲。
AgentEvent: TypeAlias = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)

#: 底层循环的事件 sink；可以是同步或异步的。
AgentEventSink: TypeAlias = Callable[[AgentEvent], Any]


class ToolCallHooks(Protocol):
    """:class:`AgentLoopConfig` 的 ``before_tool_call`` 与 ``after_tool_call`` hook。

    Attributes:
        before_tool_call: 工具执行前、参数校验完成后调用的 hook；``None`` 表示不启用。
        after_tool_call: 工具执行结束后调用的 hook；``None`` 表示不启用。
    """

    before_tool_call: Callable[..., Any] | None
    after_tool_call: Callable[..., Any] | None


class AgentState:
    """公开的 agent 状态。

    ``tools`` 与 ``messages`` 通过访问器属性暴露，实现会在存储前复制被赋值的列表。

    Attributes:
        messages: 会话记录；通过属性访问，赋值时会复制顶层列表。
        tools: 可执行的工具；通过属性访问，赋值时会复制顶层列表。
        model: 当前使用的模型。
        thinking_level: 当前 thinking 档位。
        is_streaming: agent 是否正在处理 prompt 或续跑。
        streaming_message: 当前流式响应的部分 assistant 消息（若有）。
        pending_tool_calls: 正在执行的 tool call id 集合。
        error_message: 最近一次失败或中止的 assistant 回合的错误消息（若有）。
    """

    def __init__(
        self,
        *,
        messages: list[Any],
        tools: list[AgentTool],
        model: Model,
        thinking_level: ThinkingLevel | str,
    ) -> None:
        """初始化 agent 状态。

        Args:
            messages: 初始会话记录。
            tools: 初始可执行工具列表。
            model: 初始模型。
            thinking_level: 初始 thinking 档位。
        """
        self._messages = messages
        self._tools = tools
        self.model: Model = model
        self.thinking_level: ThinkingLevel | str = thinking_level
        #: agent 正在处理 prompt 或续跑时为 True；要等被 await 的 ``agent_end``
        #: 监听器结算后才变回 False。
        self.is_streaming: bool = False
        #: 当前流式响应的部分 assistant 消息（若有）。
        self.streaming_message: Any = None
        #: 正在执行的 tool call id。
        self.pending_tool_calls: set[str] = set()
        #: 最近一次失败或中止的 assistant 回合的错误消息（若有）。
        self.error_message: str | None = None

    @property
    def system_prompt(self) -> str:
        """当前 system prompt，从记录中的 system 消息回放而来。

        只读：要修改 prompt，请追加带 ``content`` 或 ``sections`` 的 system 消息。
        在 ``initial_state`` 中，它用于生成首条 system 消息。

        Returns:
            当前生效的 system prompt 文本。
        """
        return get_current_system_prompt(self._messages)

    @property
    def tools(self) -> list[AgentTool]:
        """可执行的工具。赋新列表时会复制顶层列表。

        Returns:
            当前工具列表。
        """
        return self._tools

    @tools.setter
    def tools(self, tools: list[AgentTool]) -> None:
        """替换可执行的工具列表。

        Args:
            tools: 新的工具列表；存储时会复制顶层列表。
        """
        self._tools = list(tools)

    @property
    def messages(self) -> list[Any]:
        """会话记录。赋新列表时会复制顶层列表。

        Returns:
            当前会话消息列表。
        """
        return self._messages

    @messages.setter
    def messages(self, messages: list[Any]) -> None:
        """替换会话记录。

        Args:
            messages: 新的消息列表；存储时会复制顶层列表。
        """
        self._messages = list(messages)


@dataclass
class AgentInitialState:
    """:class:`~pi_agent.agent.Agent` 的初始状态。

    除非 ``messages`` 已以 system 消息开头，否则 ``system_prompt`` 与 ``tools``
    会成为首条 system 消息。

    Attributes:
        system_prompt: 初始 system prompt；并入首条 system 消息。
        model: 初始模型。
        thinking_level: 初始 thinking 档位。
        tools: 初始可执行工具列表；并入首条 system 消息。
        messages: 初始会话记录。
    """

    system_prompt: str | None = field(default=None, metadata={"alias": "systemPrompt"})
    model: Model | None = None
    thinking_level: ThinkingLevel | str | None = field(default=None, metadata={"alias": "thinkingLevel"})
    tools: list[AgentTool] | None = None
    messages: list[Any] | None = None
