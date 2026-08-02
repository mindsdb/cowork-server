"""Tests for the cowork-server publish access handling (ENG-322).

Covers the body model, the pure access/version resolver, and the owner-side
state read back for pre-filling the publish dialog.
"""

import json
from pathlib import Path

from anton.publish_access import normalize_emails as _normalize_emails
from anton.publish_access import resolve_access as _resolve_access
from cowork.api.v1.endpoints.publish import _PublishBody
from cowork.services.artifacts import _published_access_for


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


def test_publish_body_restricted():
    b = _PublishBody.model_validate(
        {"path": "/tmp/a", "access": {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True}}
    )
    assert b.access.mode == "restricted"
    assert b.access.emails == ["a@x.com"]
    assert b.access.org_allowed is True


def test_publish_body_back_compat_password():
    b = _PublishBody.model_validate({"path": "/tmp/a", "password": "p"})
    assert b.password == "p"
    assert b.access is None


def test_publish_body_access_password():
    b = _PublishBody.model_validate({"path": "/tmp/a", "access": {"mode": "password", "password": "p"}})
    assert b.access.mode == "password"
    assert b.access.password == "p"


# ---------------------------------------------------------------------------
# _resolve_access
# ---------------------------------------------------------------------------


def test_resolve_public_default():
    eff, pwd_v, acc_v, owner = _resolve_access(None, None, None)
    assert eff == {"mode": "public"}
    assert owner == {"mode": "public", "requires_password": False}


def test_resolve_legacy_password():
    eff, pwd_v, acc_v, owner = _resolve_access("hunter2", None, None)
    assert eff == {"mode": "password", "password": "hunter2"}
    assert pwd_v == 1
    assert owner["mode"] == "password"
    assert owner["requires_password"] is True
    assert owner["access_password"] == "hunter2"
    assert owner["pwd_version"] == 1


def test_resolve_empty_password_degrades_to_public():
    eff, *_ = _resolve_access("   ", None, None)
    assert eff == {"mode": "public"}


def test_resolve_password_change_bumps_pwd_version():
    prev = {"mode": "password", "access_password": "old", "pwd_version": 2}
    _, pwd_v, _, owner = _resolve_access("new", None, prev)
    assert pwd_v == 3
    assert owner["pwd_version"] == 3


def test_resolve_password_unchanged_keeps_pwd_version():
    prev = {"mode": "password", "access_password": "same", "pwd_version": 2}
    _, pwd_v, _, _ = _resolve_access("same", None, prev)
    assert pwd_v == 2


def test_resolve_restricted_normalizes():
    eff, _, acc_v, owner = _resolve_access(
        None, {"mode": "restricted", "emails": [" A@x.com ", "a@x.com"], "org_allowed": True}, None
    )
    assert eff == {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True}
    assert acc_v == 1
    assert owner["mode"] == "restricted"
    assert owner["emails"] == ["a@x.com"]
    assert owner["org_allowed"] is True
    assert owner["access_version"] == 1


def test_resolve_restricted_change_bumps_access_version():
    prev = {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": False, "access_version": 2}
    _, _, acc_v, _ = _resolve_access(
        None, {"mode": "restricted", "emails": ["a@x.com", "b@x.com"], "org_allowed": False}, prev
    )
    assert acc_v == 3


def test_resolve_restricted_unchanged_keeps_access_version():
    prev = {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True, "access_version": 2}
    _, _, acc_v, _ = _resolve_access(
        None, {"mode": "restricted", "emails": [" A@x.com "], "org_allowed": True}, prev
    )
    assert acc_v == 2


def test_resolve_restricted_empty_degrades_to_public():
    eff, *_ = _resolve_access(None, {"mode": "restricted", "emails": [], "org_allowed": False}, None)
    assert eff == {"mode": "public"}


def test_resolve_explicit_public_clears_prior():
    prev = {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True, "access_version": 5}
    eff, _, _, owner = _resolve_access(None, {"mode": "public"}, prev)
    assert eff == {"mode": "public"}
    assert owner == {"mode": "public", "requires_password": False}


def test_normalize_emails():
    assert _normalize_emails([" A@X.com ", "a@x.com", "B@Y.com"]) == ["a@x.com", "b@y.com"]


# ---------------------------------------------------------------------------
# _published_access_for
# ---------------------------------------------------------------------------


def _write_published(folder: Path, entry: dict, name: str = "index.html") -> Path:
    (folder / ".published.json").write_text(json.dumps({name: entry}), encoding="utf-8")
    return folder / name


def test_published_access_restricted(tmp_path: Path):
    primary = _write_published(
        tmp_path,
        {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True, "access_version": 2},
    )
    out = _published_access_for(tmp_path, primary)
    assert out["accessMode"] == "restricted"
    assert out["accessEmails"] == ["a@x.com"]
    assert out["orgAllowed"] is True
    assert out["accessProtected"] is False


def test_published_access_password(tmp_path: Path):
    primary = _write_published(
        tmp_path, {"mode": "password", "requires_password": True, "access_password": "s3cret", "pwd_version": 1}
    )
    out = _published_access_for(tmp_path, primary)
    assert out["accessMode"] == "password"
    assert out["accessProtected"] is True
    assert out["accessPassword"] == "s3cret"


def test_published_access_legacy_password_without_mode(tmp_path: Path):
    primary = _write_published(tmp_path, {"requires_password": True, "access_password": "old"})
    out = _published_access_for(tmp_path, primary)
    assert out["accessMode"] == "password"
    assert out["accessProtected"] is True


def test_published_access_public_default(tmp_path: Path):
    out = _published_access_for(tmp_path, tmp_path / "index.html")
    assert out["accessMode"] == "public"
    assert out["accessProtected"] is False
    assert out["accessEmails"] == []


# ---------------------------------------------------------------------------
# Unified .published.json target/key convention (anton == cowork)
# ---------------------------------------------------------------------------


def test_resolve_publish_target_matches_anton(tmp_path):
    from anton.publish_access import resolve_publish_target
    root = tmp_path / ".anton" / "artifacts"
    art = root / "sales"
    art.mkdir(parents=True)
    (art / "metadata.json").write_text('{"type": "html-report", "primary": "report.html"}')
    (art / "report.html").write_text("<html></html>")
    _t, pub_dir, key, is_fs = resolve_publish_target(art, [root])
    assert (pub_dir, key, is_fs) == (art, "report.html", False)
