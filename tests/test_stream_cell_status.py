"""Tests for classify_cell_status — the killed/timeout/ok tag on scratchpad
results so the renderer can show a dead cell as dead, not stuck "running".

Markers mirror the strings anton produces in core/backends/local.py
(timeout / inactivity kill) and format_cell_result / prepare_scratchpad_exec
(`[error]`, empty-code "exec failed").
"""

from __future__ import annotations

import json

from cowork.harnesses.anton_harness.stream_formatter import classify_cell_status


def _cell(*, error=None, stdout="", stderr=""):
    """Mirror json.dumps(asdict(cell)) — the real exec StreamToolResult content."""
    return json.dumps({
        "code": "x",
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "description": "",
        "estimated_time": "",
        "logs": "",
    })


class TestClassifyCellStatusJson:
    """The exec path sends json.dumps(asdict(cell)) — inspect the error field."""

    def test_success_is_ok(self):
        assert classify_cell_status(_cell(error=None, stdout="42")) == "ok"

    def test_timeout_kill(self):
        assert classify_cell_status(_cell(error="Cell timed out after 180s total. Process killed")) == "timeout"

    def test_inactivity_kill(self):
        assert classify_cell_status(_cell(error="Cell killed after 60s of inactivity")) == "timeout"

    def test_runtime_error(self):
        assert classify_cell_status(_cell(error="Traceback (most recent call last)\nNameError: x")) == "error"

    def test_success_with_markers_in_stdout_is_still_ok(self):
        # Regression guard: a successful cell whose OWN output mentions
        # "[error]" or "Cell timed out" must not be misclassified — only the
        # structured error field decides.
        noisy = _cell(error=None, stdout="log row: [error] Cell timed out after ... of inactivity")
        assert classify_cell_status(noisy) == "ok"


class TestClassifyCellStatusFallback:
    """Non-JSON results (dump notebook, other tools) use a best-effort sniff."""

    def test_empty_or_none_is_ok(self):
        assert classify_cell_status("") == "ok"
        assert classify_cell_status(None) == "ok"

    def test_plain_text_ok(self):
        assert classify_cell_status("The notebook has been displayed.") == "ok"

    def test_plain_text_timeout(self):
        assert classify_cell_status("Cell timed out after 180s total") == "timeout"

    def test_plain_text_error_bracket(self):
        assert classify_cell_status("[error]\nNameError") == "error"
