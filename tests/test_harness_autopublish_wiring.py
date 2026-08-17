"""Where autopublish hangs in the turn lifecycle.

It must run AFTER the try/finally, alongside the card yields, not inside the
finally: an `await` there is skipped on cancellation, so the property the comment
in harness.py promises ("runs on every exit") would silently not hold. That
property has no other observable form, which is why these tests inspect the source
rather than behavior.

Assertions are normalized (whitespace collapsed) and check the smallest
distinctive fragment, so an equivalent-but-differently-spaced implementation stays
green.
"""
from __future__ import annotations

import inspect
import re

from cowork.harnesses.anton_harness import harness as h


def _src() -> str:
    return re.sub(r"\s+", " ", inspect.getsource(h.AntonHarness.stream_response))


def test_autopublish_is_called_outside_the_finally_block():
    src = _src()
    finally_at = src.index("finally:")
    call_at = src.index("autopublish_project_artifacts(")
    yield_at = src.index("ArtifactCreated(")

    # The call sits between the finally block and the card yields, i.e. in the
    # normal-completion path.
    assert finally_at < call_at < yield_at


def test_finally_block_contains_no_await():
    src = _src()
    # Slice up to the first post-finally statement, not to the autopublish call:
    # the `await` token sits immediately before that call and would otherwise fall
    # inside the slice.
    finally_body = src[src.index("finally:"):src.index("republished =")]
    assert "await " not in finally_body


def test_cards_exclude_self_heal_publishes():
    src = _src()
    assert "cards_for_slugs(" in src
    # An edited (not new) artifact must get a card so its URL reaches the UI
    # without waiting for the next list fetch — hence the union with
    # `republished`. But INTERSECTED with the touched set: `republished` also holds
    # phase-2 self-heal publishes, and carding those would attach unrelated old
    # artifacts to this turn's answer.
    assert "republished &" in src


def test_pre_turn_snapshot_captures_content_mtimes():
    src = _src()
    assert "snapshot_artifact_state(" in src
    assert "before_mtimes" in src
