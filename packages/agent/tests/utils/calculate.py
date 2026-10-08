"""agent 测试使用的计算器 tool。

移植自 ``packages/agent/test/utils/calculate.ts``。TypeScript 辅助函数用 ``new Function``
求值；本移植用 Python 的受限 ``eval``，足以覆盖测试用到的算术表达式。
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pi_ai.types import TextContent, Usage

from pi_agent.types import AgentTool, AgentToolResult

__all__ = [
    "calculate",
    "calculate_tool",
    "create_calculate_tool_with_usage",
]

CALCULATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate",
        }
    },
    "required": ["expression"],
}


def calculate(expression: str) -> AgentToolResult:
    """对 ``expression`` 求值并返回其文本渲染。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - 测试辅助函数
    except Exception as error:  # noqa: BLE001 - 与 TypeScript 的 throw 行为保持一致
        raise RuntimeError(str(error) or error.__class__.__name__) from error
    return AgentToolResult(
        content=[TextContent(text=f"{expression} = {result}")],
        details=None,
    )


async def _execute(_tool_call_id: str, args: dict[str, Any]) -> AgentToolResult:
    return calculate(args["expression"])


#: 计算器 tool，对应上游的 ``calculateTool``。
calculate_tool = AgentTool(
    name="calculate",
    label="Calculator",
    description="Evaluate mathematical expressions",
    parameters=CALCULATE_SCHEMA,
    execute=_execute,
)


def create_calculate_tool_with_usage(usage: Usage) -> AgentTool:
    """结果携带 ``usage`` 的计算器，对应上游的 ``createCalculateToolWithUsage``。"""

    async def execute(_tool_call_id: str, args: dict[str, Any]) -> AgentToolResult:
        result = calculate(args["expression"])
        return dataclasses.replace(result, usage=usage)

    return dataclasses.replace(calculate_tool, execute=execute)
