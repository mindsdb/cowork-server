"""The OAuth state file holds a connector's client secret, so only its user can
read it.

OAuthStateStore writes the file through dir_atomic_write, which renames a new
0600 file over the name, so a link or a second name planted there is replaced,
never written through. The file's own folder, when missing, is created 0700;
folders above it take the default mode. A folder that already exists keeps its
mode, since it can be a home folder the user chose.

Each change reads the whole file and writes it back, so a read that fails for
any reason but a missing file raises rather than reading as empty, and one
lock serializes the changes in the process.
"""
from __future__ import annotations

import errno
import logging
import os
import stat
import threading
from pathlib import Path

import pytest

from cowork.services.connectors.oauth import state as state_module
from cowork.services.connectors.oauth.state import OAuthStateStore

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes and links only")


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
    return stat.S_IMODE(path.lstat().st_mode)


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


@posix_only
def test_a_fresh_save_is_private(tmp_path):
    path = tmp_path / "oauth_state.json"
    store = OAuthStateStore(str(path))

    _start_sign_in(store)

    assert _mode(path) == 0o600
    assert store.get_pending("google")["clientSecret"] == "client-secret"


@posix_only
def test_an_existing_world_readable_file_ends_private_after_a_save(tmp_path):
    path = tmp_path / "oauth_state.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)
    store = OAuthStateStore(str(path))

    _start_sign_in(store)

    assert _mode(path) == 0o600
    assert store.get_pending("google")["clientSecret"] == "client-secret"


@posix_only
def test_a_link_at_the_name_is_replaced_not_written_through(tmp_path):
    # A save that wrote through the name and then tightened it would put the
    # secret into the link's target and leave the name a link.
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    outside.chmod(0o644)
    home = tmp_path / "home"
    home.mkdir()
    path = home / "oauth_state.json"
    path.symlink_to(outside)

    _start_sign_in(OAuthStateStore(str(path)))

    assert outside.read_text(encoding="utf-8") == "outside"
    assert _mode(outside) == 0o644
    assert stat.S_ISREG(path.lstat().st_mode)
    assert _mode(path) == 0o600
    assert "client-secret" in path.read_text(encoding="utf-8")


@posix_only
def test_a_second_name_for_the_file_keeps_its_old_bytes_and_mode(tmp_path):
    path = tmp_path / "oauth_state.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)
    twin = tmp_path / "twin.json"
    os.link(path, twin)

    _start_sign_in(OAuthStateStore(str(path)))

    assert twin.read_text(encoding="utf-8") == "{}"
    assert _mode(twin) == 0o644
    assert _mode(path) == 0o600


@posix_only
def test_a_missing_folder_is_created_private(tmp_path):
    path = tmp_path / "home" / "oauth_state.json"

    _start_sign_in(OAuthStateStore(str(path)))

    assert _mode(path.parent) == 0o700
    assert _mode(path) == 0o600


@posix_only
def test_an_existing_folder_keeps_its_mode(tmp_path):
    folder = tmp_path / "home"
    folder.mkdir()
    folder.chmod(0o755)

    _start_sign_in(OAuthStateStore(str(folder / "oauth_state.json")))

    assert _mode(folder) == 0o755


@posix_only
def test_a_missing_folder_takes_the_default_mode_on_windows(tmp_path, monkeypatch):
    # On Windows, Python turns a 0o700 mkdir into an owner-only access list
    # that drops what the profile passes down, so the store asks for none.
    monkeypatch.setattr(state_module, "_IS_WINDOWS", True)
    path = tmp_path / "home" / "oauth_state.json"

    _start_sign_in(OAuthStateStore(str(path)))

    assert _mode(path.parent) == 0o755


def test_a_failed_read_raises_and_leaves_the_file_unchanged(tmp_path, monkeypatch):
    # Read as empty, a failed read would let this change save a file that
    # holds only its own entry, dropping the other services' state.
    path = tmp_path / "oauth_state.json"
    store = OAuthStateStore(str(path))
    _start_sign_in(store)
    before = path.read_bytes()
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == path:
            raise OSError(errno.EMFILE, "Too many open files")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    with pytest.raises(OSError) as raised:
        store.set_pending(
            "github",
            state="state-2",
            verifier="verifier-2",
            redirect_uri="http://127.0.0.1:26866/api/v1/connectors/oauth/github/callback",
            started_at="2026-01-01T00:00:00+00:00",
        )

    assert raised.value.errno == errno.EMFILE
    assert path.read_bytes() == before


def test_a_path_with_no_file_name_logs_a_warning(tmp_path, caplog):
    path = tmp_path / "missing" / ".."

    with caplog.at_level(logging.WARNING, logger=state_module.__name__):
        _start_sign_in(OAuthStateStore(str(path)))

    warnings = [
        record
        for record in caplog.records
        if record.name == state_module.__name__ and record.levelno == logging.WARNING
    ]
    assert [record.msg for record in warnings] == ["Could not save OAuth state to %s: %s"]


def test_a_change_waits_for_one_already_saving(tmp_path, monkeypatch):
    # The first change holds its save open. A second change that read the file
    # meanwhile would save it without the first one's outcome, and the first
    # save would then drop the second's.
    store = OAuthStateStore(str(tmp_path / "oauth_state.json"))
    saving = threading.Event()
    release = threading.Event()
    real_save = OAuthStateStore._save
    gated = [True]

    def gated_save(self, *args, **kwargs):
        if gated[0]:
            gated[0] = False
            saving.set()
            release.wait(timeout=10)
        real_save(self, *args, **kwargs)

    monkeypatch.setattr(OAuthStateStore, "_save", gated_save)
    first_change = threading.Thread(target=store.set_outcome, args=("state-1", {"status": "success"}))
    second_change = threading.Thread(target=store.set_outcome, args=("state-2", {"status": "error"}))

    first_change.start()
    try:
        assert saving.wait(timeout=10)
        second_change.start()
        # Without the lock the second change reads and saves before this
        # wait ends, ahead of the first change's save.
        second_change.join(timeout=0.5)
    finally:
        release.set()
        first_change.join(timeout=10)
    second_change.join(timeout=10)

    assert store.get_outcome("state-1") == {"status": "success"}
    assert store.get_outcome("state-2") == {"status": "error"}
