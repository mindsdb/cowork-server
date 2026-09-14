from __future__ import annotations

import json
import threading

import pytest

from cowork.api.v1.endpoints import coding as coding_endpoints
from cowork.coding.contracts import TerminalPage, TerminalStatus
from cowork.coding.terminal import TerminalBuffer


def test_terminal_buffer_supports_reconnect_and_completion() -> None:
    buffer = TerminalBuffer("process-1")
    buffer.append("Zmlyc3Q=", "stdout", False)
    buffer.append("c2Vjb25k", "stderr", True)

    first = buffer.page()
    assert [item.seq for item in first.items] == [1, 2]
    assert first.items[1].stream == "stderr"
    assert first.items[1].cap_reached is True
    assert buffer.page(after=1).items[0].seq == 2

    buffer.finish(7, None)
    completed = buffer.wait(after=2, timeout=0.01)
    assert completed.status == TerminalStatus.exited
    assert completed.exit_code == 7


def test_terminal_wait_wakes_when_output_arrives() -> None:
    buffer = TerminalBuffer("process-1")
    result = []
    waiter = threading.Thread(target=lambda: result.append(buffer.wait(0, 1)))
    waiter.start()
    buffer.append("b2s=", "stdout", False)
    waiter.join(timeout=1)

    assert result and result[0].next_seq == 1


@pytest.mark.asyncio
async def test_terminal_stream_completion_preserves_terminal_page_contract(monkeypatch) -> None:
    page = TerminalPage(
        process_id="process-1",
        status=TerminalStatus.exited,
        first_seq=2,
        next_seq=2,
        exit_code=0,
    )

    class Service:
        def get_session(self, session_id: str) -> object:
            assert session_id == "task-1"
            return object()

        def wait_for_terminal(self, session_id: str, after: int, timeout: float) -> TerminalPage:
            assert (session_id, after, timeout) == ("task-1", 2, 15.0)
            return page

    class Request:
        async def is_disconnected(self) -> bool:
            return False

    monkeypatch.setattr(coding_endpoints, "_service", lambda: Service())
    response = await coding_endpoints.stream_terminal(Request(), "task-1", 2)
    chunks = [chunk async for chunk in response.body_iterator]
    frame = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks).decode()
    payload = json.loads(frame.split("data: ", 1)[1].strip())

    assert payload["status"] == "exited"
    assert payload["items"] == []


@pytest.mark.asyncio
async def test_named_terminal_stream_is_scoped_to_its_tab(monkeypatch) -> None:
    page = TerminalPage(
        process_id="process-2",
        status=TerminalStatus.exited,
        first_seq=4,
        next_seq=4,
        exit_code=0,
    )

    class Service:
        def terminal_tab(self, session_id: str, terminal_id: str) -> TerminalPage:
            assert (session_id, terminal_id) == ("task-1", "terminal-2")
            return page

        def wait_for_terminal_tab(
            self,
            session_id: str,
            terminal_id: str,
            after: int,
            timeout: float,
        ) -> TerminalPage:
            assert (session_id, terminal_id, after, timeout) == (
                "task-1", "terminal-2", 4, 15.0,
            )
            return page

    class Request:
        async def is_disconnected(self) -> bool:
            return False

    monkeypatch.setattr(coding_endpoints, "_service", lambda: Service())
    response = await coding_endpoints.stream_terminal_tab(
        Request(), "task-1", "terminal-2", 4
    )
    chunks = [chunk async for chunk in response.body_iterator]
    frame = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks).decode()
    payload = json.loads(frame.split("data: ", 1)[1].strip())

    assert payload["process_id"] == "process-2"
    assert payload["status"] == "exited"
