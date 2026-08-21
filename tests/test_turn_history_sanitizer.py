"""Validation of turn-history rows arriving from a semi-trusted pod (ENG-1808).

Every rejection drops the WHOLE set, never a single row: tool_use/tool_result
pairing is a property of the set, so a partial drop leaves an orphan and every
later turn in that conversation fails with a 400 from the LLM API.
"""

import pytest

from cowork.handlers._turn_history import (
    _MAX_RESULT_BYTES,
    _MAX_TURN_BYTES,
    sanitize_turn_history_rows,
)


def _pair(uid="t1", result="BODY"):
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": uid, "name": "scratchpad", "input": {"code": "1"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": uid, "content": result}]},
    ]


# --- happy path -------------------------------------------------------------

def test_a_clean_pair_passes_through_unchanged():
    rows = _pair()
    assert sanitize_turn_history_rows(rows) == rows


def test_several_pairs_pass_through():
    rows = _pair("t1") + _pair("t2")
    assert sanitize_turn_history_rows(rows) == rows


def test_two_tool_uses_in_one_assistant_row_pass():
    rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a", "name": "x", "input": {}},
            {"type": "tool_use", "id": "b", "name": "y", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "A"},
            {"type": "tool_result", "tool_use_id": "b", "content": "B"}]},
    ]
    assert sanitize_turn_history_rows(rows) == rows


def test_empty_input_is_empty_output():
    assert sanitize_turn_history_rows([]) == []


# --- structural rejection (whole set) ---------------------------------------

@pytest.mark.parametrize("garbage", [
    None,
    "not a list",
    {"role": "assistant"},
    42,
])
def test_non_list_input_is_rejected(garbage):
    assert sanitize_turn_history_rows(garbage) == []


@pytest.mark.parametrize("bad_row", [
    "not a dict",
    None,
    {"role": "system", "content": []},
    {"role": "assistant", "content": "not a list"},
    {"role": "assistant", "content": []},
    {"content": [{"type": "tool_use", "id": "t1"}]},
])
def test_a_malformed_row_drops_the_whole_set(bad_row):
    """One bad row poisons the set, because keeping the rest could orphan a
    block. The turn degrades to text-only replay."""
    assert sanitize_turn_history_rows(_pair() + [bad_row]) == []


def test_a_text_block_in_a_tool_row_drops_the_set():
    rows = _pair()
    rows[0]["content"].append({"type": "text", "text": "narration"})
    assert sanitize_turn_history_rows(rows) == []


def test_a_tool_result_block_under_the_assistant_role_drops_the_set():
    """Role and block type are coupled: the splitter only ever produces
    assistant/tool_use and user/tool_result."""
    rows = [{"role": "assistant", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "B"}]}]
    assert sanitize_turn_history_rows(rows) == []


def test_a_tool_use_without_an_id_drops_the_set():
    rows = _pair()
    del rows[0]["content"][0]["id"]
    assert sanitize_turn_history_rows(rows) == []


def test_a_tool_result_without_a_tool_use_id_drops_the_set():
    rows = _pair()
    del rows[1]["content"][0]["tool_use_id"]
    assert sanitize_turn_history_rows(rows) == []


# --- pairing ---------------------------------------------------------------

def test_a_tool_use_with_no_result_drops_the_set():
    assert sanitize_turn_history_rows([_pair()[0]]) == []


def test_a_tool_result_with_no_use_drops_the_set():
    assert sanitize_turn_history_rows([_pair()[1]]) == []


def test_mismatched_ids_drop_the_set():
    rows = _pair()
    rows[1]["content"][0]["tool_use_id"] = "other"
    assert sanitize_turn_history_rows(rows) == []


def test_a_duplicated_tool_use_id_drops_the_set():
    """Counter, not set: two tool_use blocks sharing an id against a single
    tool_result would pass a set comparison and replay invalidly."""
    rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            {"type": "tool_use", "id": "t1", "name": "y", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "B"}]},
    ]
    assert sanitize_turn_history_rows(rows) == []


def test_no_orphans_survive_any_accepted_set():
    """The property the set-level policy exists to guarantee."""
    for rows in (_pair(), _pair("a") + _pair("b")):
        clean = sanitize_turn_history_rows(rows)
        uses = {b["id"] for r in clean if r["role"] == "assistant" for b in r["content"]}
        results = {b["tool_use_id"] for r in clean if r["role"] == "user"
                   for b in r["content"]}
        assert uses == results


# --- size caps -------------------------------------------------------------

def test_an_oversized_tool_result_becomes_a_placeholder():
    """The row survives so pairing survives; only the payload is dropped."""
    rows = _pair(result="x" * (_MAX_RESULT_BYTES + 1))
    clean = sanitize_turn_history_rows(rows)
    block = clean[1]["content"][0]
    assert block["tool_use_id"] == "t1"
    assert isinstance(block["content"], str)
    assert "tool output omitted" in block["content"]
    assert str(_MAX_RESULT_BYTES) in block["content"]
    # The tool_use side is untouched.
    assert clean[0] == rows[0]


def test_an_oversized_list_shaped_result_becomes_a_string_placeholder():
    """tool_result content may be a string or a list of blocks; the placeholder
    is always a string, which is valid for both."""
    rows = _pair(result=[{"type": "text", "text": "y" * (_MAX_RESULT_BYTES + 1)}])
    block = sanitize_turn_history_rows(rows)[1]["content"][0]
    assert isinstance(block["content"], str)
    assert "tool output omitted" in block["content"]


def test_a_result_exactly_at_the_cap_is_kept():
    """Boundary: the cap is inclusive. `_wire_size` measures the JSON encoding,
    so a string of N chars serializes to N+2 bytes with its quotes — hence the
    -2."""
    body = "z" * (_MAX_RESULT_BYTES - 2)
    rows = _pair(result=body)
    assert sanitize_turn_history_rows(rows)[1]["content"][0]["content"] == body


def test_a_result_one_byte_over_the_cap_is_replaced():
    body = "z" * (_MAX_RESULT_BYTES - 1)
    block = sanitize_turn_history_rows(_pair(result=body))[1]["content"][0]
    assert "tool output omitted" in block["content"]


def test_a_small_result_is_kept():
    rows = _pair(result="ok")
    assert sanitize_turn_history_rows(rows)[1]["content"][0]["content"] == "ok"


def test_an_oversized_turn_drops_the_set():
    """tool_use.input has no safe placeholder, so it can only push the turn over
    the total budget."""
    fat = {"code": "q" * (_MAX_TURN_BYTES // 2)}
    rows = []
    for i in range(3):
        rows.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "scratchpad", "input": fat}]})
        rows.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}]})
    # ~384 KB of tool_use input against a 256 KB budget, and none of it is
    # placeholder-able.
    assert sanitize_turn_history_rows(rows) == []


def test_placeholders_can_bring_a_turn_back_under_the_budget():
    """Caps are applied BEFORE the total is judged, so a turn that is oversized
    only because of tool OUTPUT survives. Three 128 KB results total ~384 KB
    against a 256 KB budget; after capping they are a few hundred bytes."""
    big = "w" * (_MAX_RESULT_BYTES * 8)
    rows = _pair("a", result=big) + _pair("b", result=big) + _pair("c", result=big)
    clean = sanitize_turn_history_rows(rows)
    assert len(clean) == 6
    assert all("tool output omitted" in r["content"][0]["content"]
               for r in clean if r["role"] == "user")


# --- logging ---------------------------------------------------------------

def test_a_rejection_logs_at_warning_not_info(monkeypatch):
    """staging runs at LOG_LEVEL=WARNING, so an INFO-level notice is invisible —
    a silently dropped turn is exactly what we would need the log for.

    Asserted through the module logger rather than caplog on purpose: cowork's
    setup_logging() calls logging.basicConfig(force=True) at import time, which
    drops pytest's root capture handler. caplog therefore sees nothing in a
    full-suite run (it works in isolation), so it cannot express this.
    """
    from cowork.handlers import _turn_history

    warnings: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(_turn_history.logger, "warning",
                        lambda msg, *a: warnings.append(msg % a if a else msg))
    monkeypatch.setattr(_turn_history.logger, "info",
                        lambda msg, *a: infos.append(msg % a if a else msg))

    assert sanitize_turn_history_rows([_pair()[0]]) == []

    assert len(warnings) == 1
    assert "do not pair" in warnings[0]
    assert infos == []
