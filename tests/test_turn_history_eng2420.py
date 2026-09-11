"""ENG-2420 — the in-process path must id-check its own rows too.

anton could build a `tool_use` block with an EMPTY id. Persisting one poisons
the conversation permanently: every later turn replays it and the provider
rejects the whole request with `400 Invalid 'input[N].call_id': empty string`.

The pod path has been immune since ENG-1808 — `sanitize_turn_history_rows`
already rejects a non-string-or-empty id — while the in-process path assigned
`data["rows"]` straight through. That asymmetry is why the same anton bug was
permanent on desktop and cost the web only one turn's tool detail.

The size budgets are deliberately NOT adopted here; the second half of this
file is what stops someone "simplifying" this into a call to the pod sanitizer.
"""

import copy

from cowork.handlers._turn_history import (
    _MAX_RESULT_BYTES,
    _MAX_TURN_BYTES,
    reject_unreplayable_tool_rows,
    sanitize_turn_history_rows,
)


def _pair(uid="t1", result="BODY"):
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": uid, "name": "scratchpad", "input": {"code": "1"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": uid, "content": result}]},
    ]


def test_a_clean_pair_passes_through_unchanged():
    # Snapshot first: the function returns the SAME list object on success, so
    # `== rows` would compare an object with itself and pass even if a future
    # refactor mutated the rows in place (review: pnewsam on #520).
    rows = _pair()
    before = copy.deepcopy(rows)
    assert reject_unreplayable_tool_rows(rows) == before


def test_an_empty_tool_use_id_drops_the_whole_turn():
    rows = _pair()
    rows[0]["content"][0]["id"] = ""
    assert reject_unreplayable_tool_rows(rows) == []


def test_an_empty_tool_result_id_drops_the_whole_turn():
    rows = _pair()
    rows[1]["content"][0]["tool_use_id"] = ""
    assert reject_unreplayable_tool_rows(rows) == []


def test_a_non_string_id_drops_the_whole_turn():
    """Truthy but unusable — the check is `isinstance(str)`, not `if id`."""
    rows = _pair()
    rows[0]["content"][0]["id"] = {"nested": "id"}
    assert reject_unreplayable_tool_rows(rows) == []


def test_one_bad_pair_drops_the_good_ones_with_it():
    """All-or-nothing: pairing is a property of the SET, so keeping the good
    rows would leave the conversation with an orphan block."""
    rows = _pair("good") + _pair("")
    assert reject_unreplayable_tool_rows(rows) == []


def test_a_non_list_payload_is_refused():
    assert reject_unreplayable_tool_rows("not a list") == []


def test_an_unexpected_role_is_refused():
    assert reject_unreplayable_tool_rows([{"role": "system", "content": []}]) == []


# --- the budgets belong to the POD path, not this one -----------------------

def test_an_oversize_tool_result_is_kept_verbatim():
    """The in-process path has never been subject to the 16 KiB cap, and the
    sanitizer's own docstring says its placeholder fires "on ordinary large
    cell output". Desktop is where large local-data output happens; adopting
    that budget to fix an id bug would silently truncate good tool detail."""
    big = "x" * (_MAX_RESULT_BYTES * 2)
    rows = _pair(result=big)
    before = copy.deepcopy(rows)
    assert reject_unreplayable_tool_rows(rows) == before
    # ...and this is exactly what the pod sanitizer would have done instead:
    assert sanitize_turn_history_rows(rows)[1]["content"][0]["content"] != big


def test_an_oversize_turn_is_kept_whole():
    """Each result stays under the per-result cap so the pod sanitizer has no
    placeholder to shrink it with — the only thing left for it to do is drop
    the turn. The in-process path keeps it."""
    chunk = "y" * (_MAX_RESULT_BYTES // 2)
    rows = []
    for i in range(40):
        rows += _pair(f"t{i}", result=chunk)
    before = copy.deepcopy(rows)
    assert reject_unreplayable_tool_rows(rows) == before
    assert sanitize_turn_history_rows(rows) == []
