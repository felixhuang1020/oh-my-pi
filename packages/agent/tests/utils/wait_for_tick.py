"""把控制权交还事件循环。

移植自 ``packages/agent/test/utils/wait-for-tick.ts``：TypeScript 辅助函数调度一个零延迟
定时器，Python 的等价实现是 await ``asyncio.sleep(0)``。
"""

from __future__ import annotations

import asyncio

__all__ = ["wait_for_tick"]


async def wait_for_tick() -> None:
    """让其他所有就绪任务各运行一次。"""
    await asyncio.sleep(0)
