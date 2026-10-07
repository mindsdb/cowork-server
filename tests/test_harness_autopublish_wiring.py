"""Where autopublish hangs in the turn lifecycle, on BOTH producers.

Two producers reach the same end-of-turn artifact flow, and the deployment that
autopublish exists for uses the one that is easiest to forget: on an org
deployment `AntonHarness.stream_response` refuses outright (turns run on the
remote worker), so `handlers.responses._produce_remote` is the only path that
actually publishes there. The in-process path is what desktop and the tests
exercise. A change that wires up one and not the other looks completely healthy
in every other test.

Both must satisfy the same two properties:

  * an artifact is recorded on every exit. The in-process harness indexes in
    its generator's `finally`, which therefore must contain no `await`, because
    an await there is skipped on cancellation. The remote producer records in
    the end-of-turn step every other exit runs (`end_turn`, which a cancel
    waits for), and after the reply stream on a clean finish, each record a
    database unit;
  * publishing and carding run only in the normal-completion path: after the
    `finally` in process, after the reply stream ends without a raise remotely.

Neither property has an observable form (their whole point is what happens when
a turn is cancelled), which is why these tests inspect the source rather than
behavior. Assertions are normalized (whitespace collapsed) and check the smallest
distinctive fragment, so an equivalent-but-differently-spaced implementation stays
green.
"""
from __future__ import annotations

import inspect
import io
import re
import textwrap
import tokenize

import pytest

from cowork.handlers import responses as r
from cowork.harnesses.anton_harness import harness as h
from cowork.services import task_objects as t


def _norm(obj) -> str:
    """Source with comments removed and whitespace collapsed.

    Comments are stripped through the tokenizer rather than a regex because
    these tests search for bare keywords: the code they inspect is heavily
    commented, and prose explaining why there is no `await` in a finally block
    would otherwise read as an `await` in that finally block.
    """
    src = textwrap.dedent(inspect.getsource(obj))
    lines = src.splitlines()
    # Truncate each commented line at the comment's own column rather than
    # untokenizing: untokenize re-spaces the code it emits, which would break
    # assertions that look for exact fragments like `cards_for_slugs(`.
    for token in tokenize.generate_tokens(io.StringIO(src).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col]
    return re.sub(r"\s+", " ", "\n".join(lines))


# `_run_remote_turn`'s source includes its nested `replies_as_stream_events`
# and `end_turn`, which is where the remote turn's artifact flow lives.
PRODUCERS = [
    pytest.param(h.AntonHarness.stream_response, id="in-process"),
    pytest.param(r.ResponsesHandler._run_remote_turn, id="remote"),
]


def test_publish_is_called_outside_the_finally_block():
    src = _norm(h.AntonHarness.stream_response)
    finally_at = src.index("finally:")
    call_at = src.index("publish_and_card_turn_artifacts(")
    yield_at = src.index("ArtifactCreated(")

    # The call sits between the finally block and the card yields, i.e. in the
    # normal-completion path.
    assert finally_at < call_at < yield_at


def test_finally_block_contains_no_await():
    src = _norm(h.AntonHarness.stream_response)
    # Slice from the finally to the publish call's own `await`, which is the
    # first post-finally statement. `rindex` for the await so the slice stops
    # at the token immediately preceding the call rather than at some earlier
    # one inside the finally, of which there must be none anyway: that is
    # exactly what this asserts.
    finally_body = src[src.index("finally:"):src.rindex("await publish_and_card_turn_artifacts(")]
    assert "await " not in finally_body


def test_indexing_runs_in_the_finally():
    src = _norm(h.AntonHarness.stream_response)
    assert src.index("finally:") < src.index("index_turn_artifacts(")


def test_remote_producer_publishes_only_after_the_reply_stream_ends_cleanly():
    """The record and the publish follow the reply loop, so a raise inside it
    (a failure, a Stop) skips both; the record comes first, because the
    publish reads the owners it writes."""
    src = _norm(r.ResponsesHandler._run_remote_turn)
    loop_at = src.index("async for kind, data in stream_remote_replies(")
    record_at = src.index("changes = await record_artifacts(completed_cleanly=completed_cleanly)")
    publish_at = src.index("await publish_and_card_turn_artifacts(")
    assert loop_at < record_at < publish_at < src.index("ArtifactCreated(")


def test_remote_producer_records_artifacts_on_every_other_exit():
    """Every exit that is not a clean finish ends through end_turn, which
    records the artifacts before it saves the answer, and a deleted turn
    still records them before it drops the rest."""
    src = _norm(r.ResponsesHandler._run_remote_turn)
    assert "await record_artifacts(completed_cleanly=False) saved = await persist(clean=False)" in src
    assert "await record_artifacts(completed_cleanly=False) return" in src


@pytest.mark.parametrize("producer", PRODUCERS)
def test_pre_turn_snapshot_captures_content_mtimes(producer):
    src = _norm(producer)
    assert "snapshot_artifact_state(" in src
    assert "before_mtimes" in src


def test_remote_producer_breaks_rather_than_returns_on_completion():
    """`turn_completed` used to `return`, which would skip the publish/card block
    sitting after the reply loop. The distinction is invisible to every other test:
    a `return` still streams the answer correctly, it just silently never
    publishes."""
    src = _norm(r.ResponsesHandler._run_remote_turn)
    completed_at = src.index('elif kind == "turn_completed":')
    tail = src[completed_at:completed_at + 200]
    assert "break" in tail
    assert "return" not in tail.split("break")[0]


def test_cards_exclude_self_heal_publishes():
    src = _norm(t.publish_and_card_turn_artifacts)
    assert "cards_for_slugs(" in src
    # An edited (not new) artifact must get a card so its URL reaches the UI
    # without waiting for the next list fetch — hence the union with
    # `republished`. But INTERSECTED with the touched set: `republished` also holds
    # phase-2 self-heal publishes, and carding those would attach unrelated old
    # artifacts to this turn's answer.
    assert "republished &" in src
