"""The OAuth state file holds a connector's client secret, so only its user can
read it.

OAuthStateStore writes the file through dir_atomic_write, which renames a new
0600 file over the old one, and creates a missing folder 0700. A folder that
already exists keeps its mode, since it can be a home folder the user chose.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from cowork.services.connectors.oauth.state import OAuthStateStore

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")


@pytest.fixture(autouse=True)
def _common_umask():
    # Under the common 022 umask a plain write leaves the file readable by
    # every account. A runner with a stricter umask would hide that.
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _start_sign_in(store: OAuthStateStore) -> None:
    store.set_pending(
        "google",
        state="state-1",
        verifier="verifier-1",
        redirect_uri="http://127.0.0.1:26866/api/v1/connectors/oauth/callback",
        started_at="2026-01-01T00:00:00+00:00",
        client_id="client-id",
        client_secret="client-secret",
    )


def test_a_fresh_save_is_private(tmp_path):
    path = tmp_path / "oauth_state.json"
    store = OAuthStateStore(str(path))

    _start_sign_in(store)

    assert _mode(path) == 0o600
    assert store.get_pending("google")["clientSecret"] == "client-secret"


def test_an_existing_world_readable_file_ends_private_after_a_save(tmp_path):
    path = tmp_path / "oauth_state.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)
    store = OAuthStateStore(str(path))

    _start_sign_in(store)

    assert _mode(path) == 0o600
    assert store.get_pending("google")["clientSecret"] == "client-secret"


def test_a_missing_folder_is_created_private(tmp_path):
    path = tmp_path / "home" / "oauth_state.json"

    _start_sign_in(OAuthStateStore(str(path)))

    assert _mode(path.parent) == 0o700
    assert _mode(path) == 0o600


def test_an_existing_folder_keeps_its_mode(tmp_path):
    folder = tmp_path / "home"
    folder.mkdir()
    folder.chmod(0o755)

    _start_sign_in(OAuthStateStore(str(folder / "oauth_state.json")))

    assert _mode(folder) == 0o755
