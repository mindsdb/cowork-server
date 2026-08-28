"""render_picker_page()'s rendered JS must always report `newFiles` on a
successful result — useGoogleDrivePicker.js (cowork) reads it unconditionally
to scope an attach action to just what was picked this time, not the full
accumulated grant."""
from __future__ import annotations

import html
import json
from html.parser import HTMLParser

from cowork.services.connectors.oauth.picker_page import (
    is_valid_drive_file_ids,
    render_picker_page,
)


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


class _StatusDivAttrs(HTMLParser):
    """Collects the attributes of the first <div id="status"> tag using a
    real HTML tokenizer — reports exactly what a browser would treat as a
    live attribute vs. inert text, unlike a substring search."""

    def __init__(self):
        super().__init__()
        self.attrs = None

    def handle_starttag(self, tag, attrs):
        if tag == "div" and self.attrs is None and ("id", "status") in attrs:
            self.attrs = dict(attrs)


def _status_div_attrs(page: str) -> dict:
    parser = _StatusDivAttrs()
    parser.feed(page)
    assert parser.attrs is not None, 'expected a <div id="status"> tag'
    return parser.attrs


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
    breaks out of the data-file-ids attribute.

    Parses the real tag with an HTML tokenizer rather than substring-
    matching the page text: json.dumps() already backslash-escapes the
    hostile string's own quotes, so the raw payload never appears verbatim
    in the page either way — a plain `hostile not in page` check can't tell
    an escaped attribute from a broken-out one. An actual parser can: if the
    payload broke out, `onmouseover` shows up as a live attribute key.
    """
    hostile = 'a" onmouseover=alert(1) x="'
    page = _render(state='s" onx="1', file_ids=[hostile])

    attrs = _status_div_attrs(page)

    assert "onmouseover" not in attrs
    assert set(attrs) == {
        "class", "id", "data-state", "data-access-token", "data-api-key",
        "data-app-id", "data-account-email", "data-file-ids",
    }

    # The escaped attributes must still decode back to their original
    # values, the way the client's dataset/JSON.parse read does.
    assert html.unescape(attrs["data-state"]) == 's" onx="1'
    assert json.loads(html.unescape(attrs["data-file-ids"])) == [hostile]


def test_is_valid_drive_file_ids():
    """Mirrors desktop's isValidDriveFileIds (drive-picker-service.ts) —
    same shape, same allowed/rejected cases."""
    assert is_valid_drive_file_ids([]) is True
    assert is_valid_drive_file_ids(["1a2B3c_-4d", "AbCd_1234-XYZ"]) is True

    assert is_valid_drive_file_ids("abc") is False
    assert is_valid_drive_file_ids(None) is False
    assert is_valid_drive_file_ids([123]) is False
    assert is_valid_drive_file_ids(["ok", "</script><script>evil()</script>"]) is False
    assert is_valid_drive_file_ids(["has space"]) is False
    assert is_valid_drive_file_ids(["../etc/passwd"]) is False
