"""Super Agent AG-UI 适配器 / Super Agent AG-UI Adapter

把 :class:`InvokeStream` 的原始 :class:`SSEEvent` 解码为 ``ag_ui.core.BaseEvent``
强类型事件 (**client 侧解码**)。

与 ``agentrun/server/agui_protocol.py`` (server 侧编码方向) 是反向关系, 互不依赖:
本文件只消费 ``ag_ui.core`` 的事件类, server 侧则负责产生它们。

使用:

.. code-block:: python

    from agentrun.super_agent.agui import as_agui_events

    async for event in as_agui_events(stream):
        # event 是 ag_ui.core.BaseEvent 子类 (如 TextMessageContentEvent)
        ...

    # 想跳过未知事件 (如过渡期兼容), 传 on_unknown="skip"
    async for event in as_agui_events(stream, on_unknown="skip"):
        ...
"""

from __future__ import annotations

from typing import AsyncGenerator, Dict, Literal, Optional, Type

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    MessagesSnapshotEvent,
    RawEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from .stream import InvokeStream, SSEEvent

_EVENT_TYPE_TO_CLASS: Dict[str, Type[BaseEvent]] = {
    "RUN_STARTED": RunStartedEvent,
    "RUN_FINISHED": RunFinishedEvent,
    "RUN_ERROR": RunErrorEvent,
    "TEXT_MESSAGE_START": TextMessageStartEvent,
    "TEXT_MESSAGE_CONTENT": TextMessageContentEvent,
    "TEXT_MESSAGE_END": TextMessageEndEvent,
    "TOOL_CALL_START": ToolCallStartEvent,
    "TOOL_CALL_ARGS": ToolCallArgsEvent,
    "TOOL_CALL_END": ToolCallEndEvent,
    "TOOL_CALL_RESULT": ToolCallResultEvent,
    "STATE_SNAPSHOT": StateSnapshotEvent,
    "STATE_DELTA": StateDeltaEvent,
    "MESSAGES_SNAPSHOT": MessagesSnapshotEvent,
    "RAW": RawEvent,
    "CUSTOM": CustomEvent,
}


UnknownMode = Literal["raise", "skip"]


def _data_preview(data: str, limit: int = 200) -> str:
    if len(data) <= limit:
        return data
    return data[:limit] + "..."


def decode_sse_to_agui(
    sse_event: SSEEvent,
    *,
    on_unknown: UnknownMode = "raise",
) -> Optional[BaseEvent]:
    """把单个 :class:`SSEEvent` 解码为 AG-UI 事件.

    - 空 ``data`` (keepalive) → 返回 ``None``
    - 未知 ``event`` + ``on_unknown='raise'`` → 抛 ``ValueError``
    - 未知 ``event`` + ``on_unknown='skip'`` → 返回 ``None``
    - ``data`` 不合法 JSON 或 Pydantic 校验失败 → 抛 ``ValueError``
      (不论 ``on_unknown``, 因为这通常是真实错误)
    """
    if sse_event.data == "":
        return None

    event_name = sse_event.event
    if not event_name or event_name not in _EVENT_TYPE_TO_CLASS:
        if on_unknown == "skip":
            return None
        raise ValueError(
            f"Unknown AG-UI event type {event_name!r}; data prefix:"
            f" {_data_preview(sse_event.data)}"
        )

    cls = _EVENT_TYPE_TO_CLASS[event_name]
    try:
        return cls.model_validate_json(sse_event.data)
    except Exception as exc:  # JSONDecodeError, ValidationError, etc.
        raise ValueError(
            f"Failed to decode AG-UI {event_name!r} event: {exc};"
            f" data prefix: {_data_preview(sse_event.data)}"
        ) from exc


async def as_agui_events(
    stream: InvokeStream,
    *,
    on_unknown: UnknownMode = "raise",
) -> AsyncGenerator[BaseEvent, None]:
    """把 :class:`InvokeStream` 中的原始 :class:`SSEEvent` 解码为强类型流.

    无论正常消费结束、中途异常、解码异常, 都保证 ``await stream.aclose()`` 被调用
    以释放 httpx 连接。
    """
    try:
        async for sse_event in stream:
            agui_event = decode_sse_to_agui(sse_event, on_unknown=on_unknown)
            if agui_event is None:
                continue
            yield agui_event
    finally:
        await stream.aclose()


__all__ = [
    "as_agui_events",
    "decode_sse_to_agui",
    "_EVENT_TYPE_TO_CLASS",
]
