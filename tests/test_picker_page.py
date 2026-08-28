"""render_picker_page()'s rendered JS must always report `newFiles` on a
successful result — useGoogleDrivePicker.js (cowork) reads it unconditionally
to scope an attach action to just what was picked this time, not the full
accumulated grant."""
from __future__ import annotations

from cowork.services.connectors.oauth.picker_page import render_picker_page


def _render(**overrides) -> str:
    kwargs = {
        "access_token": "tok",
        "api_key": "key",
        "app_id": "app",
        "account_email": "a@b.com",
        "state": "state123",
    }
    kwargs.update(overrides)
    return render_picker_page(**kwargs)


def test_picked_files_result_includes_new_files():
    page = _render()
    assert "reportResult({ ok: true, files: files, newFiles: files })" in page


def test_cancel_and_load_failure_results_include_empty_new_files():
    page = _render()
    assert page.count("reportResult({ ok: true, files: [], newFiles: [] })") == 3


def test_error_result_has_no_files_field():
    page = _render()
    assert "ok: false, reason:" in page
    assert "reportResult({ ok: false, reason:" in page
