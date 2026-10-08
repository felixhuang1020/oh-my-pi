"""移植自 ``packages/ai/test/transcript-tool-changes.test.ts``。

经 ``on_payload`` 捕获注入点驱动真实适配器（上游在 ``onPayload`` 中抛出
``PayloadCaptured``），断言各传输层如何回放转录中的 system 消息更新与 tool 变更。全程
不发送请求：钩子在 HTTP 调用前中止，流再把失败编码出来。
"""

from __future__ import annotations

from typing import Any

from pi_ai.compat import stream_simple
from pi_ai.types import (
    Context,
    Model,
    ModelCompat,
    ModelCost,
    SimpleStreamOptions,
    SystemMessage,
    Tool,
    ToolReference,
    UserMessage,
)


class PayloadCaptured(Exception):
    """对应上游用于中断请求的 ``PayloadCaptured`` 错误。"""


def tool(name: str) -> Tool:
    # TypeBox 的 ``Type.Object({})`` 对应普通 JSON Schema。
    return Tool(name=name, description=f"{name} tool", parameters={"type": "object", "properties": {}})


BASE_TOOL = tool("base_tool")
LATE_TOOL = tool("late_tool")

CONTEXT = Context(
    messages=[
        SystemMessage(
            content="base prompt",
            sections={"rules": "<rules>\nold rules\n</rules>", "docs": "<docs>\nread docs\n</docs>"},
            tools_added=[BASE_TOOL],
            timestamp=0,
        ),
        UserMessage(content="before", timestamp=1),
        SystemMessage(
            content="updated guidance",
            sections={"rules": "<rules>\nnew rules\n</rules>", "docs": None},
            tools_removed=[ToolReference(name="base_tool")],
            tools_added=[LATE_TOOL],
            timestamp=2,
        ),
    ]
)
ADDITION_CONTEXT = Context(
    messages=[
        SystemMessage(content="base prompt", tools_added=[BASE_TOOL], timestamp=0),
        UserMessage(content="before", timestamp=1),
        SystemMessage(content="updated guidance", tools_added=[LATE_TOOL], timestamp=2),
    ]
)

MODEL_BASE: dict[str, Any] = {
    "base_url": "http://127.0.0.1:9",
    "input": ["text"],
    "cost": ModelCost(),
    "context_window": 100000,
    "max_tokens": 1000,
}


def make_model(
    api: str,
    provider: str,
    *,
    id: str = "claude-opus-5",
    name: str = "Claude Opus 5",
    mid_convo_system_messages: bool | None = None,
    mid_convo_tool_changes: bool | None = None,
    additional_tools: bool | None = None,
    tool_search: bool | None = None,
    reasoning: bool = True,
) -> Model:
    compat = ModelCompat(
        supports_mid_convo_system_messages=mid_convo_system_messages,
        supports_mid_convo_tool_changes=mid_convo_tool_changes,
        supports_additional_tools=additional_tools,
        supports_tool_search=tool_search,
    )
    return Model(
        **MODEL_BASE,
        reasoning=reasoning,
        id=id,
        name=name,
        api=api,
        provider=provider,
        compat=compat,
    )


async def capture_payload(model: Model, context: Context) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def on_payload(payload: Any, _model: Model) -> None:
        captured.update(payload)
        raise PayloadCaptured()

    stream = stream_simple(
        model, context, SimpleStreamOptions(api_key="test-key", on_payload=on_payload)
    )
    await stream.result()
    assert captured, "Expected payload capture"
    return captured


# --------------------------------------------------------------------------------------
# describe("transcript system messages")：转录 system 消息
# --------------------------------------------------------------------------------------


async def test_sends_anthropic_updates_and_tool_changes_in_native_system_messages() -> None:
    model = make_model(
        "anthropic-messages",
        "anthropic",
        mid_convo_system_messages=True,
        mid_convo_tool_changes=True,
    )
    payload = await capture_payload(model, CONTEXT)

    assert "mid-conversation-tool-changes-2026-07-01" in payload["betas"]
    assert [block["text"] for block in payload["system"]] == [
        "base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>",
    ]
    # 初始 tool 保持生效并携带缓存断点；占位符及之后的所有声明都延迟加载；被移除的
    # tool 仍保留在声明中。
    tools = payload["tools"]
    assert [(item["name"], item.get("cache_control"), item.get("defer_loading")) for item in tools] == [
        ("base_tool", {"type": "ephemeral"}, None),
        ("__pi_deferred_placeholder__", None, True),
        ("late_tool", None, True),
    ]
    assert tools[0].get("defer_loading") is None
    assert tools[1].get("cache_control") is None
    assert tools[2].get("cache_control") is None

    update = payload["messages"][-1]
    assert update["role"] == "system"
    content = update["content"]
    assert [block["type"] for block in content] == ["text", "tool_removal", "tool_addition"]
    assert content[1]["tool"]["name"] == "base_tool"
    assert content[2]["tool"]["name"] == "late_tool"
    assert "updated guidance" in content[0]["text"]
    assert "<rules>\nnew rules\n</rules>" in content[0]["text"]
    assert 'Removed system prompt section "docs"' in content[0]["text"]

    # 占位符在任何变更之前就已声明，其脚手架从首个请求起即可被缓存。
    initial = await capture_payload(model, Context(messages=CONTEXT.messages[:2]))
    assert [item["name"] for item in initial["tools"]] == ["base_tool", "__pi_deferred_placeholder__"]


async def test_sends_the_current_anthropic_tool_list_when_native_tool_changes_cannot_express_the_history() -> None:
    model = make_model(
        "anthropic-messages",
        "anthropic",
        mid_convo_system_messages=True,
        mid_convo_tool_changes=True,
    )
    redefined_tool = Tool(
        name="base_tool", description="changed", parameters={"type": "object", "properties": {}}
    )
    fallback_contexts = [
        # 同名重定义：内容块仅按名称引用 tool。
        Context(
            messages=[
                SystemMessage(content="base prompt", tools_added=[BASE_TOOL], timestamp=0),
                SystemMessage(
                    content="updated guidance",
                    tools_removed=[ToolReference(name="base_tool")],
                    tools_added=[redefined_tool],
                    timestamp=2,
                ),
            ]
        ),
        # 没有初始 tool：Anthropic 会拒绝全延迟加载的 tool 列表。
        Context(
            messages=[
                SystemMessage(content="base prompt", timestamp=0),
                SystemMessage(content="updated guidance", tools_added=[redefined_tool], timestamp=2),
            ]
        ),
    ]
    for fallback_context in fallback_contexts:
        payload = await capture_payload(model, fallback_context)
        assert "mid-conversation-tool-changes-2026-07-01" not in (payload.get("betas") or [])
        tools = payload["tools"]
        assert [(item["name"], item["description"], item.get("cache_control")) for item in tools] == [
            ("base_tool", "changed", {"type": "ephemeral"}),
        ]
        assert tools[0].get("defer_loading") is None
        assert [block["type"] for block in payload["messages"][-1]["content"]] == ["text"]


async def test_folds_anthropic_updates_into_the_system_prompt_without_native_support() -> None:
    model = make_model("anthropic-messages", "anthropic", id="claude-sonnet-4-5", name="Claude Sonnet 4.5")
    payload = await capture_payload(model, CONTEXT)

    assert "mid-conversation-tool-changes-2026-07-01" not in (payload.get("betas") or [])
    assert [block["text"] for block in payload["system"]] == [
        "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>",
    ]
    assert [item["name"] for item in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


async def test_requires_both_anthropic_capabilities_for_native_tool_changes() -> None:
    model = make_model(
        "anthropic-messages",
        "anthropic",
        mid_convo_tool_changes=True,
    )
    payload = await capture_payload(model, CONTEXT)

    assert "mid-conversation-tool-changes-2026-07-01" not in (payload.get("betas") or [])
    assert [item["name"] for item in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


async def test_anchors_openai_additions_at_their_developer_message() -> None:
    model = make_model(
        "openai-responses",
        "openai",
        id="gpt-5.4",
        name="GPT-5.4",
        mid_convo_system_messages=True,
        additional_tools=True,
    )
    payload = await capture_payload(model, ADDITION_CONTEXT)

    assert [item["name"] for item in payload["tools"]] == ["base_tool"]
    additional = next(item for item in payload["input"] if item.get("type") == "additional_tools")
    assert [item["name"] for item in additional["tools"]] == ["late_tool"]
    assert [
        item["content"]
        for item in payload["input"]
        if item.get("role") == "developer" and item.get("type") is None
    ] == ["base prompt", "updated guidance"]


async def test_maps_system_message_additions_into_synthetic_tool_search() -> None:
    model = make_model(
        "openai-responses",
        "openai",
        id="gpt-5.4",
        name="GPT-5.4",
        mid_convo_system_messages=True,
        tool_search=True,
    )
    payload = await capture_payload(model, ADDITION_CONTEXT)

    assert [item["name"] for item in payload["tools"]] == ["base_tool"]
    assert "tool_search_call" in [item.get("type") for item in payload["input"]]
    search_output = next(item for item in payload["input"] if item.get("type") == "tool_search_output")
    assert [item["name"] for item in search_output["tools"]] == ["late_tool"]


async def test_folds_openai_updates_into_the_leading_developer_message_without_native_support() -> None:
    model = make_model(
        "openai-responses", "openai", id="gpt-4.1", name="GPT-4.1", additional_tools=True
    )
    payload = await capture_payload(model, CONTEXT)

    assert [item["name"] for item in payload["tools"]] == ["late_tool"]
    assert [item.get("type") or item.get("role") for item in payload["input"]] == ["developer", "user"]
    assert payload["input"][0]["content"] == "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"


async def test_falls_back_to_the_complete_current_tool_state_when_removals_are_unsupported() -> None:
    model = make_model(
        "openai-responses",
        "openai",
        id="gpt-5.4",
        name="GPT-5.4",
        mid_convo_system_messages=True,
        additional_tools=True,
    )
    payload = await capture_payload(model, CONTEXT)

    assert [item["name"] for item in payload["tools"]] == ["late_tool"]
    assert not any(item.get("type") == "additional_tools" for item in payload["input"])
    assert len([item for item in payload["input"] if item.get("role") == "developer"]) == 2
