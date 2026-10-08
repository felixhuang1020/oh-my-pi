"""agent 测试使用的当前时间 tool。

移植自 ``packages/agent/test/utils/get-current-time.ts``。
"""

from __future__ import annotations

from datetime import datetime, timezone as timezone_module
from typing import Any
from zoneinfo import ZoneInfo

from pi_ai.types import TextContent

from pi_agent.types import AgentTool, AgentToolResult

__all__ = [
    "GET_CURRENT_TIME_SCHEMA",
    "get_current_time",
    "get_current_time_tool",
]

GET_CURRENT_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "timezone": {
            "type": "string",
            "description": "Optional timezone (e.g., 'America/New_York', 'Europe/London')",
        }
    },
}


def _utc_timestamp_millis(now: datetime) -> int:
    return int(now.timestamp() * 1000)


async def get_current_time(timezone: str | None = None) -> AgentToolResult:
    """返回当前日期时间，可选按 ``timezone`` 渲染。"""
    now = datetime.now(timezone_module.utc)
    if timezone:
        try:
            local = now.astimezone(ZoneInfo(timezone))
        except Exception as error:  # noqa: BLE001 - 与 TypeScript 的 throw 行为保持一致
            raise ValueError(
                f"Invalid timezone: {timezone}. Current UTC time: {now.isoformat()}"
            ) from error
        text = local.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    else:
        text = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    return AgentToolResult(
        content=[TextContent(text=text)],
        details={"utcTimestamp": _utc_timestamp_millis(now)},
    )


async def _execute(_tool_call_id: str, args: dict[str, Any]) -> AgentToolResult:
    return await get_current_time(args.get("timezone"))


#: 当前时间 tool，对应上游的 ``getCurrentTimeTool``。
get_current_time_tool = AgentTool(
    name="get_current_time",
    label="Current Time",
    description="Get the current date and time",
    parameters=GET_CURRENT_TIME_SCHEMA,
    execute=_execute,
)
