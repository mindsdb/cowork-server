"""Every path that collects tool rows must also hand them to the DB.

Four independent paths reach save_assistant_turn with a turn's tool rows, and
ENG-1808 happened precisely because one of them was added later and forgotten:
the in-process paths carried rows from ENG-742 onward while _produce_remote —
the only path a SaaS deployment ever uses — silently did not.

These properties have no observable form in a unit test (they are about which
call site exists, not about behaviour), which is why this file inspects source,
mirroring tests/test_harness_autopublish_wiring.py. Assertions are normalized
(comments stripped, whitespace collapsed) and check the smallest distinctive
fragment, so a differently-formatted implementation stays green.
"""
from __future__ import annotations

import inspect
import io
import re
import textwrap
import tokenize

import pytest

from cowork.channels import runtime as rt
from cowork.handlers import responses as r


def _norm(obj) -> str:
    """Source with comments removed and whitespace collapsed.

    Comments are stripped through the tokenizer rather than a regex because the
    code here is heavily commented and prose mentioning `tool_rows=` would
    otherwise read as a real call-site argument.
    """
    src = textwrap.dedent(inspect.getsource(obj))
    lines = src.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(src).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col]
    return re.sub(r"\s+", " ", "\n".join(lines))


# `_produce_remote`'s source includes its nested `replies_as_stream_events` and
# `persist`, which is where the remote turn's history flow lives.
PERSISTERS = [
    pytest.param(r.ResponsesHandler._run_turn, id="in-process-streaming"),
    pytest.param(r.ResponsesHandler._collect, id="in-process-non-streaming"),
    pytest.param(r.ResponsesHandler._produce_remote, id="remote"),
    pytest.param(rt.AntonChannelRuntime._run_anton, id="channels"),
]


@pytest.mark.parametrize("persister", PERSISTERS)
def test_collected_rows_are_passed_on(persister):
    """Collecting rows and never handing them over is exactly the ENG-1808 bug.

    Counted rather than matched on `tool_rows=`: `_collect` passes them
    positionally through `_save_assistant_turn`, so a keyword-argument check
    would fail on a path that is in fact correct. Two occurrences means the
    accumulator is both filled and forwarded.
    """
    src = _norm(persister)
    assert "turn_rows" in src, "this path does not collect tool rows at all"
    assert src.count("turn_rows") >= 2, "collected tool rows are never passed on"


@pytest.mark.parametrize("persister", PERSISTERS)
def test_rows_are_kept_out_of_the_replayable_event_log(persister):
    """The client rebuilds its UI from the events log, and tool rows are hidden
    from the UI — so the history payload must never be appended to it.

    Every collector slice-assigns into `turn_rows` and returns/breaks before the
    generic event-append, which is what this fragment pins.
    """
    assert "turn_rows[:]" in _norm(persister)


def test_remote_path_gates_rows_on_a_clean_finish():
    """Rows arrive before the terminal event and persist() also runs on failure
    and cancellation, so the gate is what keeps a torn turn text-only."""
    src = _norm(r.ResponsesHandler._produce_remote)
    assert "def persist(*, clean: bool = False)" in src
    assert "tool_rows=turn_rows if clean else None" in src
    assert "persist(clean=True)" in src
    # Exactly one caller opts in; the failure paths must not.
    assert src.count("persist(clean=True)") == 1


def test_remote_path_sanitizes_before_persisting():
    """The pod is semi-trusted; unvalidated rows must not reach the DB."""
    src = _norm(r.ResponsesHandler._produce_remote)
    assert "sanitize_turn_history_rows(" in src
