"""published_owner_state returns the raw owner-side .published.json entry."""

import json
from unittest import mock

from cowork.services import publish as svc


def test_published_owner_state_returns_access_fields(tmp_path):
    art = tmp_path / "app"
    art.mkdir(parents=True)
    (art / "index.html").write_text("<html></html>")
    (art / ".published.json").write_text(json.dumps({
        "index.html": {"report_id": "r", "url": "u", "published": True,
                        "mode": "password", "requires_password": True,
                        "access_password": "s3cret", "pwd_version": 2},
    }))
    with mock.patch.object(svc, "resolve_artifact_path", return_value=art / "index.html"):
        state = svc.published_owner_state(str(art / "index.html"))
    assert state["mode"] == "password"
    assert state["access_password"] == "s3cret"


def test_published_owner_state_blank_when_absent(tmp_path):
    art = tmp_path / "none"
    art.mkdir()
    with mock.patch.object(svc, "resolve_artifact_path", return_value=art):
        assert svc.published_owner_state(str(art)) == {}
