"""流式适配器共用的 Server-sent events 解码.

TypeScript 各适配器都在 ``ReadableStream`` 上手写 SSE 状态机；Python 从异步行
迭代器读取同一种线上格式，因此一个解码器即可服务所有适配器，帧规则集中维护
且只在一处测试。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

__all__ = ["SSEDecoder", "SSEEvent", "iter_sse_events"]


@dataclass
class SSEEvent:
    """一条已分发的 server-sent event."""

    #: ``event:`` 字段的值（如存在）。
    event: str | None = None
    #: 各 ``data:`` 字段按协议要求以换行拼接。
    data: str = ""
    #: 最近一次出现的 ``id:`` 字段值。
    id: str | None = None
    #: 最近一次出现的 ``retry:`` 字段值（毫秒）。
    retry: int | None = None
    #: 产生本事件的原始行，供调试与 provider 特定解析使用。
    raw: list[str] = field(default_factory=list)

    def json(self) -> object:
        """把 :attr:`data` 按 JSON 解析，非 JSON 时抛出 :class:`ValueError`."""
        import json

        return json.loads(self.data)


class SSEDecoder:
    """面向直接消费原始分块的调用方的增量 SSE 解码器.

    用 :meth:`feed` 喂入已解码的行，空行分发缓冲的事件；
    :meth:`flush` 在流结束时分发残留的事件。

    Attributes:
        _event: 当前缓冲的 ``event:`` 字段值。
        _data: 当前缓冲的所有 ``data:`` 字段值。
        _raw: 当前事件已消费的原始行，供调试使用。
        _last_id: 最近一次出现的 ``id:`` 字段值，跨事件保留。
        _retry: 最近一次出现的 ``retry:`` 字段值，跨事件保留。
    """

    def __init__(self) -> None:
        """初始化解码器：无缓冲事件，无记住的 ``id`` 与 ``retry``."""
        self._event: str | None = None
        self._data: list[str] = []
        self._raw: list[str] = []
        self._last_id: str | None = None
        self._retry: int | None = None

    def feed(self, line: str) -> list[SSEEvent]:
        """消费一行（不含终止符）并返回任何已分发的事件.

        Args:
            line: SSE 协议中的一行内容，已去除行终止符。

        Returns:
            如果是空行（事件分隔符），则分发缓冲的事件；否则返回空列表。

        Note:
            空行触发事件分发，符合 SSE 协议规范。
        """
        if line == "":
            return self._dispatch()
        self._raw.append(line)
        if line.startswith(":"):
            return []
        name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if name == "event":
            self._event = value
        elif name == "data":
            self._data.append(value)
        elif name == "id":
            self._last_id = value
        elif name == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                pass
        return []

    def flush(self) -> list[SSEEvent]:
        """分发未被空行终止的缓冲事件."""
        return self._dispatch()

    def _dispatch(self) -> list[SSEEvent]:
        """清空缓冲并按当前状态产出至多一个事件。

        Returns:
            含一个事件的列表；无任何缓冲行时返回空列表。

        Note:
            ``id`` 与 ``retry`` 是跨事件保留的字段，只有 ``event`` 与 ``data``
            在分发后被清空。
        """
        if not self._raw:
            self._event = None
            self._data = []
            return []
        event = SSEEvent(
            event=self._event,
            data="\n".join(self._data),
            id=self._last_id,
            retry=self._retry,
            raw=list(self._raw),
        )
        self._event = None
        self._data = []
        self._raw = []
        return [event]


async def iter_sse_events(lines: AsyncIterator[str] | Iterable[str]) -> AsyncIterator[SSEEvent]:
    """从异步（或同步）行迭代器中解码 SSE 事件.

    Args:
        lines: 逐行产出 SSE 文本的异步迭代器或同步可迭代对象。

    Yields:
        解码后的事件；流结束时通过 :meth:`SSEDecoder.flush` 补发残留事件。
    """
    decoder = SSEDecoder()
    if hasattr(lines, "__aiter__"):
        async for line in lines:  # type: ignore[union-attr]  # 该分支已确认可异步迭代
            for event in decoder.feed(line.rstrip("\r\n")):
                yield event
    else:
        for line in lines:  # type: ignore[union-attr]  # 剩余分支按同步可迭代对象处理
            for event in decoder.feed(line.rstrip("\r\n")):
                yield event
    for event in decoder.flush():
        yield event
