"""Super Agent SSE 流 / Super Agent SSE Stream

- :class:`SSEEvent`: SSE 协议单个事件的原始表示, 不做业务 normalize。
- :func:`parse_sse_async`: 从 ``httpx.Response.aiter_lines`` 提取 ``SSEEvent``。
- :class:`InvokeStream`: Phase 1 状态载体 + Phase 2 懒触发的异步流。

故意不引入 ``httpx-sse`` 等额外依赖;约 30 行的解析器足以覆盖
``event / data / id / retry`` 四字段、注释行、多行 ``data:``、流末尾 flush。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

import httpx


@dataclass
class SSEEvent:
    """SSE 协议单个事件的原始表示.

    SDK 不做高阶 normalize, 调用方按需用 :meth:`data_json` 解析。
    """

    event: Optional[str] = None
    data: str = ""
    id: Optional[str] = None
    retry: Optional[int] = None

    def data_json(self) -> Optional[Any]:
        """尝试把 ``data`` 解析为 JSON, 失败或为空返回 ``None``."""
        if not self.data:
            return None
        try:
            return json.loads(self.data)
        except (TypeError, ValueError):
            return None


async def parse_sse_async(
    response: httpx.Response,
) -> AsyncIterator[SSEEvent]:
    """按行解析 SSE, 逐个 yield :class:`SSEEvent`.

    规则:

    - 空行 = 事件边界, flush 当前字段 (允许空 event + 空 data 的空心跳事件被跳过)
    - ``:`` 开头的行 = 注释, 忽略
    - ``field: value`` 形式, ``:`` 后第一个空格被去除
    - 多行 ``data:`` 用 ``\\n`` 拼接
    - ``retry`` 非整数时忽略
    - 未知字段忽略 (向前兼容)
    - 流结束时若仍有未 flush 的字段, flush 一次
    """

    event: Optional[str] = None
    data_lines: List[str] = []
    sse_id: Optional[str] = None
    retry: Optional[int] = None

    def _has_content() -> bool:
        return bool(data_lines) or event is not None or sse_id is not None

    async for raw_line in response.aiter_lines():
        # httpx 的 aiter_lines 已去除换行符; 空字符串表示事件边界
        line = raw_line.rstrip("\r")
        if line == "":
            if _has_content():
                yield SSEEvent(
                    event=event,
                    data="\n".join(data_lines),
                    id=sse_id,
                    retry=retry,
                )
            event = None
            data_lines = []
            sse_id = None
            retry = None
            continue

        if line.startswith(":"):
            continue

        if ":" in line:
            field_name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
        else:
            field_name, value = line, ""

        if field_name == "event":
            event = value
        elif field_name == "data":
            data_lines.append(value)
        elif field_name == "id":
            sse_id = value
        elif field_name == "retry":
            try:
                retry = int(value)
            except (TypeError, ValueError):
                pass
        # 未知字段忽略

    if _has_content():
        yield SSEEvent(
            event=event,
            data="\n".join(data_lines),
            id=sse_id,
            retry=retry,
        )


StreamCallable = Callable[[], Awaitable[AsyncIterator[SSEEvent]]]
"""Phase 2 拉流回调: 返回 (awaitable of) 异步迭代器."""


@dataclass
class InvokeStream:
    """Phase 1 已完成的状态载体, 同时是 Phase 2 SSE 流的异步可迭代器.

    ``await SuperAgent.invoke_async(...)`` 完成后即可读:
      - :attr:`conversation_id`
      - :attr:`session_id`
      - :attr:`stream_url`
      - :attr:`stream_headers`

    只在首次 ``async for`` 或 ``__aiter__`` 调用时才触发 Phase 2 GET。
    """

    conversation_id: str
    session_id: str
    stream_url: str
    stream_headers: Dict[str, str]
    _stream_factory: StreamCallable
    _iterator: Optional[AsyncIterator[SSEEvent]] = field(
        default=None, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    async def _ensure_iterator(self) -> AsyncIterator[SSEEvent]:
        if self._iterator is None:
            self._iterator = await self._stream_factory()
        return self._iterator

    def __aiter__(self) -> "InvokeStream":
        return self

    async def __anext__(self) -> SSEEvent:
        if self._closed:
            raise StopAsyncIteration
        iterator = await self._ensure_iterator()
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise

    async def aclose(self) -> None:
        """提前关闭底层 HTTP 连接, 释放资源."""
        self._closed = True
        iterator = self._iterator
        self._iterator = None
        if iterator is not None:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

    async def __aenter__(self) -> "InvokeStream":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ["SSEEvent", "parse_sse_async", "InvokeStream"]
