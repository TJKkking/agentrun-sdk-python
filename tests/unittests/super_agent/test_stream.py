"""Unit tests for ``agentrun.super_agent.stream``."""

from typing import List
from unittest.mock import MagicMock

import pytest

from agentrun.super_agent.stream import InvokeStream, parse_sse_async, SSEEvent


class _FakeResponse:
    """Replays a pre-canned list of SSE lines via ``aiter_lines``."""

    def __init__(self, lines: List[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


async def _collect(lines: List[str]) -> List[SSEEvent]:
    return [ev async for ev in parse_sse_async(_FakeResponse(lines))]


# ─── parse_sse_async ─────────────────────────────────────────


async def test_parse_sse_simple_event():
    events = await _collect(["event: m", "data: hi", "id: 1", ""])
    assert len(events) == 1
    ev = events[0]
    assert ev.event == "m"
    assert ev.data == "hi"
    assert ev.id == "1"


async def test_parse_sse_multiline_data():
    events = await _collect(["data: a", "data: b", ""])
    assert len(events) == 1
    assert events[0].data == "a\nb"


async def test_parse_sse_comment_ignored():
    events = await _collect([": comment", "data: x", ""])
    assert len(events) == 1
    assert events[0].data == "x"


async def test_parse_sse_unknown_field_ignored():
    events = await _collect(["unknown: v", "data: x", ""])
    assert len(events) == 1
    assert events[0].data == "x"


async def test_parse_sse_retry_invalid_ignored():
    events = await _collect(["retry: not-a-number", "data: x", ""])
    assert len(events) == 1
    assert events[0].retry is None


async def test_parse_sse_retry_valid():
    events = await _collect(["retry: 5000", "data: x", ""])
    assert events[0].retry == 5000


async def test_parse_sse_strip_leading_space_after_colon():
    events = await _collect(["data: hello", ""])
    assert events[0].data == "hello"


async def test_parse_sse_field_without_colon():
    events = await _collect(["data", ""])
    assert len(events) == 1
    assert events[0].data == ""


async def test_parse_sse_flush_at_stream_end():
    events = await _collect(["data: final"])
    assert len(events) == 1
    assert events[0].data == "final"


async def test_parse_sse_multiple_events():
    events = await _collect([
        "event: a",
        "data: 1",
        "",
        "event: b",
        "data: 2",
        "",
    ])
    assert len(events) == 2
    assert events[0].event == "a"
    assert events[1].event == "b"


async def test_parse_sse_empty_line_without_content_skipped():
    # Two consecutive empty lines → no duplicate events
    events = await _collect(["", "data: x", "", ""])
    assert len(events) == 1


# ─── SSEEvent.data_json ──────────────────────────────────────


def test_sse_event_data_json_success():
    ev = SSEEvent(event="x", data='{"k":1}')
    assert ev.data_json() == {"k": 1}


def test_sse_event_data_json_failure():
    assert SSEEvent(event="x", data="not json").data_json() is None


def test_sse_event_data_json_empty():
    assert SSEEvent(event="x", data="").data_json() is None


# ─── InvokeStream ────────────────────────────────────────────


async def _make_stream(events: List[SSEEvent]) -> InvokeStream:
    async def _gen():
        for ev in events:
            yield ev

    async def _factory():
        return _gen()

    return InvokeStream(
        conversation_id="c1",
        session_id="s1",
        stream_url="https://x.com/stream",
        stream_headers={"X-Super-Agent-Session-Id": "s1"},
        _stream_factory=_factory,
    )


async def test_invoke_stream_async_iter():
    events = [
        SSEEvent(event="m", data="1"),
        SSEEvent(event="m", data="2"),
        SSEEvent(event="m", data="3"),
    ]
    stream = await _make_stream(events)
    collected = [ev async for ev in stream]
    assert len(collected) == 3
    assert [ev.data for ev in collected] == ["1", "2", "3"]


async def test_invoke_stream_aclose():
    closed = {"v": False}

    async def _gen():
        try:
            yield SSEEvent(event="m", data="x")
        finally:
            closed["v"] = True

    async def _factory():
        return _gen()

    stream = InvokeStream(
        conversation_id="c",
        session_id="s",
        stream_url="u",
        stream_headers={},
        _stream_factory=_factory,
    )
    # Advance one step to open the iterator
    it = stream.__aiter__()
    await it.__anext__()
    await stream.aclose()
    assert closed["v"] is True
    # After close, iteration is terminated
    with pytest.raises(StopAsyncIteration):
        await it.__anext__()


async def test_invoke_stream_async_with():
    closed = {"v": False}

    async def _gen():
        try:
            yield SSEEvent(event="m", data="x")
        finally:
            closed["v"] = True

    async def _factory():
        return _gen()

    stream = InvokeStream(
        conversation_id="c",
        session_id="s",
        stream_url="u",
        stream_headers={},
        _stream_factory=_factory,
    )
    async with stream as s:
        async for _ev in s:
            break
    assert closed["v"] is True
