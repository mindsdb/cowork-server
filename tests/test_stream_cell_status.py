"""Tests for classify_cell_status — the killed/timeout/ok tag on scratchpad
results so the renderer can show a dead cell as dead, not stuck "running".

Markers mirror the strings anton produces in core/backends/local.py
(timeout / inactivity kill) and format_cell_result / prepare_scratchpad_exec
(`[error]`, empty-code "exec failed").
"""

from __future__ import annotations

from cowork.harnesses.anton_harness.stream_formatter import classify_cell_status


class TestClassifyCellStatus:
    def test_ok_for_normal_output(self):
        assert classify_cell_status("[output]\n42") == "ok"

    def test_empty_or_none_is_ok(self):
        assert classify_cell_status("") == "ok"
        assert classify_cell_status(None) == "ok"

    def test_timeout_kill(self):
        assert classify_cell_status("Cell timed out after 180s total") == "timeout"

    def test_inactivity_kill(self):
        msg = "Cell killed after 60s of inactivity (no output or progress() calls)"
        assert classify_cell_status(msg) == "timeout"

    def test_runtime_error_is_error(self):
        assert classify_cell_status("[error]\nNameError: name 'x' is not defined") == "error"

    def test_empty_code_drop_is_error(self):
        assert classify_cell_status("Scratchpad exec failed: the `code` argument was empty.") == "error"

    def test_timeout_takes_precedence_over_error_bracket(self):
        # A killed cell's result carries both "[error]" and the timeout text;
        # it should classify as the more specific "timeout".
        assert classify_cell_status("[error]\nCell timed out after 180s total. Process killed") == "timeout"
