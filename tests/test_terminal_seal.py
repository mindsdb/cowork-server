"""A turn producer must ALWAYS leave a terminal record (ENG-1717).

Every producer branch in `ResponsesHandler` closes its buffer on its own path,
but the terminal is not truly guaranteed: an exception escaping the
`except Exception` handler (e.g. the error-classification helpers raising) or a
`BaseException` matching neither `except` clause skips `buffer.close()`. The
buffer then stays open with no terminal, the in-process FileStreamBuffer tail
(the desktop path) blocks forever, the client holds its single shared stream
slot, and every later message strands at "Queued". Unlike the duration bound in
RunRegistry (#345), this bites a turn that FAILS FAST — it never reaches the
timeout. `_seal_unterminated_buffer` is the guaranteed-terminal backstop run
from each producer's `finally`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cowork.handlers.responses import _seal_unterminated_buffer


class _FakeBuffer:
    """Faithful to the real buffer contract: `is_closed` gates `close()`'s
    idempotency, `append`/`close` are awaitable, and either may be made to
    raise to model a disk/redis failure mid-seal."""

    def __init__(self, *, closed: bool = False, fail_append: bool = False, fail_close: bool = False) -> None:
        self._closed = closed
        self.appended: list = []
        self.close_reason: str | None = None
        self._fail_append = fail_append
        self._fail_close = fail_close

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def append(self, type_, data):
        if self._fail_append:
            raise RuntimeError("append failed")
        self.appended.append((type_, data))
        return len(self.appended)

    async def close(self, reason, extra=None):
        if self._fail_close:
            raise RuntimeError("close failed")
        self._closed = True
        self.close_reason = reason


def _live():
    return SimpleNamespace(discarded=False)


async def test_seals_an_open_unterminated_buffer():
    buffer = _FakeBuffer()
    await _seal_unterminated_buffer(buffer, _live(), "conv-1")
    # A terminal record is written and a failed frame surfaces a generic error.
    assert buffer.close_reason == "error"
    assert buffer.is_closed is True
    assert len(buffer.appended) == 1
    assert "sse" in buffer.appended[0][1]


async def test_noop_when_already_closed():
    # close() is idempotent on the real buffer; the seal must not append a
    # spurious error frame after a turn already ended cleanly.
    buffer = _FakeBuffer(closed=True)
    await _seal_unterminated_buffer(buffer, _live(), "conv-2")
    assert buffer.appended == []
    assert buffer.close_reason is None


async def test_skips_the_discarded_path():
    # A discarded turn's buffer file was deleted by discard_conversation;
    # closing would recreate it, which the next turn would then tail.
    buffer = _FakeBuffer()
    await _seal_unterminated_buffer(buffer, SimpleNamespace(discarded=True), "conv-3")
    assert buffer.appended == []
    assert buffer.close_reason is None
    assert buffer.is_closed is False


async def test_close_failure_is_swallowed():
    # A seal failure must never mask the original exception propagating out of
    # the producer's `finally`.
    buffer = _FakeBuffer(fail_close=True)
    # Does not raise.
    await _seal_unterminated_buffer(buffer, _live(), "conv-4")


async def test_still_seals_when_the_error_frame_cannot_be_emitted():
    # Even if we can't surface the error frame, the terminal close is what
    # releases the client's stream slot — it must still run.
    buffer = _FakeBuffer(fail_append=True)
    await _seal_unterminated_buffer(buffer, _live(), "conv-5")
    assert buffer.appended == []  # append raised
    assert buffer.close_reason == "error"
    assert buffer.is_closed is True


async def test_seal_carries_a_request_id_when_the_remote_producer_gives_one():
    # The remote path's own correlation id — this is the hardest-failing
    # turn (one that escaped every named except clause), so it's exactly the
    # one a user is most likely to report; it must not be the one case with
    # no reference id.
    buffer = _FakeBuffer()
    await _seal_unterminated_buffer(buffer, _live(), "conv-6", request_id="corr-seal")
    assert "corr-seal" in buffer.appended[0][1]["sse"]


async def test_seal_omits_request_id_for_the_in_process_path():
    # The in-process/direct producers have no correlation id to offer.
    buffer = _FakeBuffer()
    await _seal_unterminated_buffer(buffer, _live(), "conv-7")
    assert "request_id" not in buffer.appended[0][1]["sse"]
