"""Preview diagnostics reach the agent prompt without being able to break it."""

from __future__ import annotations

from cowork.services.artifact_revisions import _preview_error_lines


def test_lines_are_numbered_and_carry_the_position():
    lines = _preview_error_lines(
        [{"message": "TypeError: x is not a function", "file": "dashboard.html", "line": 44}],
        "dashboard.html",
    )
    assert lines == ["  1. TypeError: x is not a function — dashboard.html:44"]


def test_a_position_without_a_file_gets_the_source_path():
    # The shim blanks the file for an error in the document's own inline
    # script: about:srcdoc on web and a signed draft URL on desktop name
    # nothing the agent can open, and the server knows the file it will edit.
    lines = _preview_error_lines([{"message": "boom", "file": "", "line": 7}], "static/index.html")
    assert lines == ["  1. boom — static/index.html:7"]


def test_about_srcdoc_is_replaced_too():
    lines = _preview_error_lines(
        [{"message": "boom", "file": "about:srcdoc", "line": 7}], "static/index.html"
    )
    assert lines == ["  1. boom — static/index.html:7"]


def test_an_entry_without_a_position_gets_no_location_at_all():
    # A failed script tag or a CSP violation happened to a URL, not at a line
    # of the source. Naming the source path there would read as "the bug is in
    # static/index.html", which is exactly the wrong place to send the agent.
    lines = _preview_error_lines(
        [{"message": "Failed to load script https://cdn.example/x.js", "file": "", "line": 0}],
        "static/index.html",
    )
    assert lines == ["  1. Failed to load script https://cdn.example/x.js"]


def test_at_most_ten_entries_survive():
    entries = [{"message": f"e{i}", "file": "a.html", "line": 1} for i in range(25)]
    assert len(_preview_error_lines(entries, "a.html")) == 10


def test_a_long_message_is_truncated():
    lines = _preview_error_lines([{"message": "x" * 900, "file": "a.html", "line": 1}], "a.html")
    assert len(lines[0]) < 400


def test_garbage_is_dropped_rather_than_raised():
    # A malformed diagnostic must never fail the repair it rides along with.
    assert _preview_error_lines("not a list", "a.html") == []
    assert _preview_error_lines([None, 5, {"file": "a.html"}], "a.html") == []
    assert _preview_error_lines(None, "a.html") == []
