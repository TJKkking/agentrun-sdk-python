"""Unit tests for ``agentrun.super_agent.agui``."""

import ast
import json
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

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
import pytest

from agentrun.super_agent import agui as agui_mod
from agentrun.super_agent.agui import (
    _EVENT_TYPE_TO_CLASS,
    as_agui_events,
    decode_sse_to_agui,
)
from agentrun.super_agent.stream import InvokeStream, SSEEvent


def _make_stream_from_sse(events: List[SSEEvent]) -> InvokeStream:
    async def _gen():
        for ev in events:
            yield ev

    async def _factory():
        return _gen()

    return InvokeStream(
        conversation_id="c",
        session_id="s",
        stream_url="https://x.com/s",
        stream_headers={},
        _stream_factory=_factory,
    )


# ─── decode_sse_to_agui ──────────────────────────────────────


def test_decode_sse_text_message_content():
    payload = {
        "type": "TEXT_MESSAGE_CONTENT",
        "messageId": "m1",
        "delta": "hi",
    }
    ev = SSEEvent(event="TEXT_MESSAGE_CONTENT", data=json.dumps(payload))
    result = decode_sse_to_agui(ev)
    assert isinstance(result, TextMessageContentEvent)


def test_decode_sse_run_started():
    payload = {"type": "RUN_STARTED", "threadId": "t1", "runId": "r1"}
    ev = SSEEvent(event="RUN_STARTED", data=json.dumps(payload))
    assert isinstance(decode_sse_to_agui(ev), RunStartedEvent)


def test_decode_sse_run_finished():
    payload = {"type": "RUN_FINISHED", "threadId": "t1", "runId": "r1"}
    ev = SSEEvent(event="RUN_FINISHED", data=json.dumps(payload))
    assert isinstance(decode_sse_to_agui(ev), RunFinishedEvent)


def test_decode_sse_run_error():
    payload = {"type": "RUN_ERROR", "message": "oops"}
    ev = SSEEvent(event="RUN_ERROR", data=json.dumps(payload))
    assert isinstance(decode_sse_to_agui(ev), RunErrorEvent)


def test_decode_sse_text_message_lifecycle():
    cases = [
        (
            "TEXT_MESSAGE_START",
            {
                "type": "TEXT_MESSAGE_START",
                "messageId": "m",
                "role": "assistant",
            },
            TextMessageStartEvent,
        ),
        (
            "TEXT_MESSAGE_CONTENT",
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": "m",
                "delta": "x",
            },
            TextMessageContentEvent,
        ),
        (
            "TEXT_MESSAGE_END",
            {
                "type": "TEXT_MESSAGE_END",
                "messageId": "m",
            },
            TextMessageEndEvent,
        ),
    ]
    for name, payload, cls in cases:
        result = decode_sse_to_agui(
            SSEEvent(event=name, data=json.dumps(payload))
        )
        assert isinstance(result, cls), name


def test_decode_sse_tool_call_lifecycle():
    cases = [
        (
            "TOOL_CALL_START",
            {
                "type": "TOOL_CALL_START",
                "toolCallId": "tc",
                "toolCallName": "fn",
            },
            ToolCallStartEvent,
        ),
        (
            "TOOL_CALL_ARGS",
            {
                "type": "TOOL_CALL_ARGS",
                "toolCallId": "tc",
                "delta": "arg",
            },
            ToolCallArgsEvent,
        ),
        (
            "TOOL_CALL_END",
            {
                "type": "TOOL_CALL_END",
                "toolCallId": "tc",
            },
            ToolCallEndEvent,
        ),
        (
            "TOOL_CALL_RESULT",
            {
                "type": "TOOL_CALL_RESULT",
                "toolCallId": "tc",
                "messageId": "m",
                "content": "r",
            },
            ToolCallResultEvent,
        ),
    ]
    for name, payload, cls in cases:
        result = decode_sse_to_agui(
            SSEEvent(event=name, data=json.dumps(payload))
        )
        assert isinstance(result, cls), name


def test_decode_sse_state_events():
    assert isinstance(
        decode_sse_to_agui(
            SSEEvent(
                event="STATE_SNAPSHOT",
                data=json.dumps({"type": "STATE_SNAPSHOT", "snapshot": {}}),
            )
        ),
        StateSnapshotEvent,
    )
    assert isinstance(
        decode_sse_to_agui(
            SSEEvent(
                event="STATE_DELTA",
                data=json.dumps({"type": "STATE_DELTA", "delta": []}),
            )
        ),
        StateDeltaEvent,
    )
    assert isinstance(
        decode_sse_to_agui(
            SSEEvent(
                event="MESSAGES_SNAPSHOT",
                data=json.dumps({
                    "type": "MESSAGES_SNAPSHOT",
                    "messages": [],
                }),
            )
        ),
        MessagesSnapshotEvent,
    )


def test_decode_sse_raw_and_custom():
    assert isinstance(
        decode_sse_to_agui(
            SSEEvent(
                event="RAW",
                data=json.dumps({"type": "RAW", "event": {"k": "v"}}),
            )
        ),
        RawEvent,
    )
    assert isinstance(
        decode_sse_to_agui(
            SSEEvent(
                event="CUSTOM",
                data=json.dumps({"type": "CUSTOM", "name": "n", "value": 1}),
            )
        ),
        CustomEvent,
    )


def test_decode_sse_empty_data_returns_none():
    assert decode_sse_to_agui(SSEEvent(event=None, data="")) is None
    assert decode_sse_to_agui(SSEEvent(event="RUN_STARTED", data="")) is None


def test_decode_sse_unknown_event_raise():
    with pytest.raises(ValueError) as exc:
        decode_sse_to_agui(SSEEvent(event="UNKNOWN_X", data="{}"))
    assert "UNKNOWN_X" in str(exc.value)
    assert "{}" in str(exc.value)


def test_decode_sse_unknown_event_skip():
    result = decode_sse_to_agui(
        SSEEvent(event="UNKNOWN_X", data="{}"), on_unknown="skip"
    )
    assert result is None


def test_decode_sse_invalid_json_raises():
    with pytest.raises(ValueError) as exc:
        decode_sse_to_agui(
            SSEEvent(event="TEXT_MESSAGE_CONTENT", data="not json")
        )
    assert "TEXT_MESSAGE_CONTENT" in str(exc.value)
    assert "not json" in str(exc.value)


def test_decode_sse_pydantic_validation_failure_raises():
    with pytest.raises(ValueError):
        decode_sse_to_agui(
            SSEEvent(
                event="TEXT_MESSAGE_CONTENT",
                data='{"unrelated":"x"}',
            )
        )


def test_decode_sse_data_with_newlines():
    # Embedded escaped newlines are fine inside JSON
    payload = json.dumps({
        "type": "TEXT_MESSAGE_CONTENT",
        "messageId": "m",
        "delta": "hi\nthere",
    })
    result = decode_sse_to_agui(
        SSEEvent(event="TEXT_MESSAGE_CONTENT", data=payload)
    )
    assert isinstance(result, TextMessageContentEvent)
    assert result.delta == "hi\nthere"


# ─── as_agui_events ──────────────────────────────────────────


async def test_as_agui_events_yields_typed_events():
    events = [
        SSEEvent(
            event="RUN_STARTED",
            data=json.dumps({
                "type": "RUN_STARTED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
        SSEEvent(
            event="TEXT_MESSAGE_CONTENT",
            data=json.dumps({
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": "m",
                "delta": "x",
            }),
        ),
        SSEEvent(
            event="RUN_FINISHED",
            data=json.dumps({
                "type": "RUN_FINISHED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
    ]
    stream = _make_stream_from_sse(events)
    collected: List[BaseEvent] = [ev async for ev in as_agui_events(stream)]
    assert [type(e).__name__ for e in collected] == [
        "RunStartedEvent",
        "TextMessageContentEvent",
        "RunFinishedEvent",
    ]


async def test_as_agui_events_skips_empty_data():
    events = [
        SSEEvent(
            event="RUN_STARTED",
            data=json.dumps({
                "type": "RUN_STARTED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
        SSEEvent(event="RUN_STARTED", data=""),  # keepalive
        SSEEvent(
            event="RUN_FINISHED",
            data=json.dumps({
                "type": "RUN_FINISHED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
    ]
    stream = _make_stream_from_sse(events)
    collected = [ev async for ev in as_agui_events(stream)]
    assert len(collected) == 2


async def test_as_agui_events_unknown_skip_mode():
    events = [
        SSEEvent(
            event="RUN_STARTED",
            data=json.dumps({
                "type": "RUN_STARTED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
        SSEEvent(event="UNKNOWN_X", data="{}"),
        SSEEvent(
            event="RUN_FINISHED",
            data=json.dumps({
                "type": "RUN_FINISHED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
    ]
    stream = _make_stream_from_sse(events)
    collected = [ev async for ev in as_agui_events(stream, on_unknown="skip")]
    assert len(collected) == 2


async def test_as_agui_events_unknown_raise_mode():
    events = [
        SSEEvent(
            event="RUN_STARTED",
            data=json.dumps({
                "type": "RUN_STARTED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
        SSEEvent(event="UNKNOWN_X", data="{}"),
    ]
    stream = _make_stream_from_sse(events)
    it = as_agui_events(stream)
    first = await it.__anext__()
    assert isinstance(first, RunStartedEvent)
    with pytest.raises(ValueError):
        await it.__anext__()


async def test_as_agui_events_closes_stream_on_normal_end():
    events = [
        SSEEvent(
            event="RUN_STARTED",
            data=json.dumps({
                "type": "RUN_STARTED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
    ]
    stream = _make_stream_from_sse(events)
    close = AsyncMock()
    stream.aclose = close
    async for _ in as_agui_events(stream):
        pass
    close.assert_awaited_once()


async def test_as_agui_events_closes_stream_on_consumer_exception():
    """当消费循环里抛异常, 只要消费者用 ``aclosing`` 包裹 (或手动 aclose),
    适配器的 ``finally`` 就能跑到 ``stream.aclose`` — 这是异步生成器清理的
    标准用法 (Python async gen 的清理不会在异常透传时自动同步执行).
    """
    from contextlib import aclosing

    events = [
        SSEEvent(
            event="RUN_STARTED",
            data=json.dumps({
                "type": "RUN_STARTED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
        SSEEvent(
            event="RUN_FINISHED",
            data=json.dumps({
                "type": "RUN_FINISHED",
                "threadId": "t",
                "runId": "r",
            }),
        ),
    ]
    stream = _make_stream_from_sse(events)
    close = AsyncMock()
    stream.aclose = close
    with pytest.raises(RuntimeError):
        async with aclosing(as_agui_events(stream)) as gen:
            async for _ in gen:
                raise RuntimeError("consumer blew up")
    close.assert_awaited_once()


async def test_as_agui_events_closes_stream_on_decode_exception():
    events = [
        SSEEvent(
            event="TEXT_MESSAGE_CONTENT",
            data="not json",
        ),
    ]
    stream = _make_stream_from_sse(events)
    close = AsyncMock()
    stream.aclose = close
    with pytest.raises(ValueError):
        async for _ in as_agui_events(stream):
            pass
    close.assert_awaited_once()


# ─── map completeness / module hygiene ──────────────────────


def test_event_type_to_class_map_completeness():
    required = {
        "RUN_STARTED",
        "RUN_FINISHED",
        "RUN_ERROR",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "STATE_SNAPSHOT",
        "STATE_DELTA",
        "MESSAGES_SNAPSHOT",
        "RAW",
        "CUSTOM",
    }
    assert required <= set(_EVENT_TYPE_TO_CLASS.keys())


def test_no_sync_as_agui_events_export():
    attrs = dir(agui_mod)
    assert "as_agui_events_sync" not in attrs
    assert "sync_as_agui_events" not in attrs


def test_agui_module_only_imports_ag_ui_core():
    src = Path(agui_mod.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("ag_ui"):
                assert (
                    node.module == "ag_ui.core"
                ), f"Disallowed ag-ui import: {node.module}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ag_ui"):
                    assert (
                        alias.name == "ag_ui.core"
                    ), f"Disallowed ag-ui import: {alias.name}"
