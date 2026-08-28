"""render_picker_page()'s rendered JS must always report `newFiles` on a
successful result — useGoogleDrivePicker.js (cowork) reads it unconditionally
to scope an attach action to just what was picked this time, not the full
accumulated grant."""
from __future__ import annotations

import html
import json

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


def test_every_data_attr_value_is_html_escaped_including_file_ids():
    """Regression: file_ids is JSON-encoded before it's an HTML attribute
    value, but json.dumps() output is JSON-safe, not HTML-attribute-safe —
    its own quote characters still need html.escape(), or a crafted file id
    breaks out of the data-file-ids attribute."""
    hostile = 'a" onmouseover=alert(1) x="'
    page = _render(state='s" onx="1', file_ids=[hostile])

    # The raw, unescaped breakout must never appear — its quotes have to be
    # &quot;, not literal ", or onmouseover becomes a live tag attribute.
    assert hostile not in page
    assert '"1"' not in page  # would appear only if the state attribute broke out unescaped

    # The escaped attribute must still decode + JSON.parse back to the
    # original value, the way the client's dataset/JSON.parse read does.
    segment = page.split("data-file-ids=")[1]
    quoted = segment[1:segment.index('"', 1)]
    decoded = html.unescape(quoted)
    assert json.loads(decoded) == [hostile]
