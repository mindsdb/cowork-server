"""The password hash anton sent is kept owner-side and handed back.

anton salts every hash, so without the previous entry the same password gets a
new hash on every publish. The artifact content host fingerprints the stored
hash to revoke viewers' password grants, so autopublish, which re-publishes on
every agent turn, would otherwise ask every viewer for the password again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import cowork.services.publish as publish_mod
from cowork.db.scoped import LOCAL_SCOPE

HASH = "pbkdf2_sha256$200000$c2FsdHNhbHRzYWx0c2FsdA==$ZGVyaXZlZGtleQ=="
PUBLISH_URL = "https://view.example"


@pytest.fixture
def page(tmp_path, monkeypatch):
    """A legacy loose HTML file: no metadata.json, so no identity lookup."""
    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(publish_mod, "vault_for_scope", lambda scope: object())
    art = tmp_path / "artifacts"
    art.mkdir()
    f = art / "report.html"
    f.write_text("<html><body>hi</body></html>", encoding="utf-8")
    return f


@pytest.fixture
def publish_calls(monkeypatch):
    calls: list[dict] = []

    def fake_publish(path, **kwargs):
        calls.append(kwargs)
        out = {"view_url": "https://view.example/a/r1", "report_id": "r1", "md5": "m1"}
        if (kwargs.get("access") or {}).get("mode") == "password":
            out["password_hash"] = HASH
        return out

    monkeypatch.setattr("anton.publisher.publish", fake_publish)
    return calls


def _entry(page: Path) -> dict:
    return json.loads((page.parent / ".published.json").read_text(encoding="utf-8"))[page.name]


def _publish(page: Path, access: dict) -> dict:
    return publish_mod.publish_artifact(
        page, artifacts_base=page.parent, api_key="k", publish_url=PUBLISH_URL, access=access,
    )


def test_the_sent_hash_is_stored_next_to_the_plaintext(page, publish_calls):
    out = _publish(page, {"mode": "password", "password": "hunter2"})

    entry = _entry(page)
    assert entry["access_password"] == "hunter2"
    assert entry["password_hash"] == HASH
    assert publish_calls[0]["previous_access"] is None
    # Owner-side state only, never part of the API response.
    assert "password_hash" not in out["result"]


def test_the_previous_entry_is_handed_to_the_publisher(page, publish_calls):
    (page.parent / ".published.json").write_text(json.dumps({
        "report.html": {
            "report_id": "r1", "url": "https://view.example/a/r1", "published": True,
            "mode": "password", "requires_password": True,
            "access_password": "hunter2", "pwd_version": 1, "password_hash": HASH,
        },
    }), encoding="utf-8")

    _publish(page, {"mode": "password", "password": "hunter2"})

    assert publish_calls[0]["report_id"] == "r1"
    assert publish_calls[0]["previous_access"]["password_hash"] == HASH
    assert _entry(page)["password_hash"] == HASH


def test_leaving_password_mode_drops_the_stored_hash(page, publish_calls):
    _publish(page, {"mode": "password", "password": "hunter2"})
    _publish(page, {"mode": "public"})

    entry = _entry(page)
    assert entry["mode"] == "public"
    assert "password_hash" not in entry
    assert "access_password" not in entry


def test_update_hands_the_stored_hash_back(page, publish_calls, monkeypatch):
    _publish(page, {"mode": "password", "password": "hunter2"})
    monkeypatch.setattr(
        publish_mod, "desktop_artifact_and_base", lambda raw, *, scope: publish_mod.DesktopArtifact(page, page.parent)
    )
    monkeypatch.setattr(publish_mod, "desktop_publish_credential", lambda: ("k", PUBLISH_URL))

    publish_mod.update_artifact(str(page), scope=LOCAL_SCOPE)

    assert publish_calls[1]["access"] == {"mode": "password", "password": "hunter2"}
    assert publish_calls[1]["previous_access"]["password_hash"] == HASH
    assert _entry(page)["password_hash"] == HASH


def test_unpublish_keeps_the_stored_hash(page, publish_calls, monkeypatch):
    _publish(page, {"mode": "password", "password": "hunter2"})
    monkeypatch.setattr("anton.publisher.unpublish", lambda identifier, **kwargs: None)

    publish_mod.unpublish_artifact(page, artifacts_base=page.parent, api_key="k", publish_url=PUBLISH_URL)

    entry = _entry(page)
    assert entry["published"] is False
    assert entry["password_hash"] == HASH


def test_the_real_publisher_resends_the_hash_while_the_password_is_unchanged(page, monkeypatch):
    """anton's own publish(), with only the HTTP call replaced.

    The other tests fake the publisher, so they cannot see an anton that ignores
    `previous_access` or stops returning `password_hash`.
    """
    sent: list[dict] = []

    def fake_request(url, api_key, method="GET", payload=None, verify=True):
        sent.append(json.loads(payload))
        return json.dumps({"view_url": "https://view.example/a/r1", "report_id": "r1", "md5": "m1"})

    monkeypatch.setattr("anton.publisher.minds_request", fake_request)

    _publish(page, {"mode": "password", "password": "hunter2"})
    _publish(page, {"mode": "password", "password": "hunter2"})
    _publish(page, {"mode": "password", "password": "changed"})

    hashes = [body["access"]["password_hash"] for body in sent]
    assert hashes[0] == hashes[1]
    assert hashes[2] != hashes[1]
    assert _entry(page)["password_hash"] == hashes[2]
    assert all("password" not in body["access"] for body in sent)
