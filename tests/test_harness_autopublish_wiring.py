"""Where autopublish hangs in the turn lifecycle, on BOTH producers.

Two producers reach the same end-of-turn artifact flow, and the deployment that
autopublish exists for uses the one that is easiest to forget: on an org
deployment `AntonHarness.stream_response` refuses outright (turns run on the
remote worker), so `handlers.responses._produce_remote` is the only path that
actually publishes there. The in-process path is what desktop and the tests
exercise. A change that wires up one and not the other looks completely healthy
in every other test.

Both must satisfy the same two properties:

  * indexing runs on every exit. The in-process harness waits for its cleanup
    through run_to_completion, so cancellation cannot abandon its database unit;
  * publishing and carding run AFTER the finally, in the normal-completion path.

The database and cancellation behavior also have dynamic coverage in
test_turn_pool_connections. These assertions check the wiring and order, with
whitespace normalized so formatting does not change their result.
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


# `_produce_remote`'s source includes its nested `replies_as_stream_events`,
# which is where the remote turn's artifact flow lives.
PRODUCERS = [
    pytest.param(h.AntonHarness.stream_response, id="in-process"),
    pytest.param(r.ResponsesHandler._produce_remote, id="remote"),
]


@pytest.mark.parametrize("producer", PRODUCERS)
def test_publish_is_called_outside_the_finally_block(producer):
    src = _norm(producer)
    finally_at = src.index("finally:")
    call_at = src.index("publish_and_card_turn_artifacts(")
    yield_at = src.index("ArtifactCreated(")

    # The call sits between the finally block and the card yields, i.e. in the
    # normal-completion path.
    assert finally_at < call_at < yield_at


@pytest.mark.parametrize("producer", PRODUCERS)
def test_finally_cleanup_is_protected_from_cancellation(producer):
    src = _norm(producer)
    # Slice from the finally to the publish call's own `await`, which is the
    # first post-finally statement on both paths.
    finally_body = src[src.index("finally:"):src.rindex("await publish_and_card_turn_artifacts(")]
    if producer is h.AntonHarness.stream_response:
        assert "await run_to_completion(record_turn_cleanup())" in finally_body
    else:
        assert "await " not in finally_body


@pytest.mark.parametrize("producer", PRODUCERS)
def test_indexing_runs_in_the_finally(producer):
    src = _norm(producer)
    cleanup = "record_turn_cleanup())" if producer is h.AntonHarness.stream_response else "index_turn_artifacts("
    assert src.index("finally:") < src.index(cleanup)


@pytest.mark.parametrize("producer", PRODUCERS)
def test_pre_turn_snapshot_captures_content_mtimes(producer):
    src = _norm(producer)
    assert "snapshot_artifact_state(" in src
    assert "before_mtimes" in src


def test_remote_producer_breaks_rather_than_returns_on_completion():
    """`turn_completed` used to `return`, which would skip the publish/card block
    now sitting after the try. The distinction is invisible to every other test:
    a `return` still streams the answer correctly, it just silently never
    publishes."""
    src = _norm(r.ResponsesHandler._produce_remote)
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
