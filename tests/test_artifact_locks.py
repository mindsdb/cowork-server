"""Per-slug publish locks.

Two turns in one project can reconcile the same slug at once; for a new artifact
that would mint two report_ids and orphan one. The lock lives outside the artifact
folders so it can never be mistaken for artifact content, and it expires by TTL
rather than being released when a publish wait times out.
"""
from __future__ import annotations

import json
import os
import time

from cowork.services import artifact_locks as locks
from cowork.services.artifacts import _user_files, content_mtime


def test_acquire_then_second_attempt_is_refused(tmp_path):
    assert locks.acquire(tmp_path, "dash", ttl_s=60) is True
    assert locks.acquire(tmp_path, "dash", ttl_s=60) is False


def test_release_lets_the_next_caller_in(tmp_path):
    locks.acquire(tmp_path, "dash", ttl_s=60)
    locks.release(tmp_path, "dash")
    assert locks.acquire(tmp_path, "dash", ttl_s=60) is True


def test_expired_lock_is_stolen(tmp_path):
    locks.acquire(tmp_path, "dash", ttl_s=60)
    lock_file = tmp_path / locks.LOCKS_DIRNAME / "dash.lock"
    old = time.time() - 3600
    os.utime(lock_file, (old, old))

    assert locks.acquire(tmp_path, "dash", ttl_s=60) is True


def test_lock_of_one_slug_does_not_block_another(tmp_path):
    assert locks.acquire(tmp_path, "one", ttl_s=60) is True
    assert locks.acquire(tmp_path, "two", ttl_s=60) is True


def test_release_of_absent_lock_is_silent(tmp_path):
    locks.release(tmp_path, "never-held")  # must not raise


def test_lock_does_not_leak_into_artifact_content(tmp_path):
    folder = tmp_path / "dash"
    folder.mkdir()
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "dash", "type": "html-app"}))
    before_files = {p.name for p in _user_files(folder)}
    before_mtime = content_mtime(folder)

    locks.acquire(tmp_path, "dash", ttl_s=60)

    assert {p.name for p in _user_files(folder)} == before_files
    assert content_mtime(folder) == before_mtime


def test_locks_dir_is_not_mistaken_for_an_artifact(tmp_path):
    locks.acquire(tmp_path, "dash", ttl_s=60)
    lock_dir = tmp_path / locks.LOCKS_DIRNAME
    # The artifact enumerator gates on metadata.json; the locks dir has none.
    assert not (lock_dir / "metadata.json").exists()


def test_unwritable_base_reports_not_acquired(tmp_path):
    # Failing to take the lock must read as "skip this slug this turn" (safe,
    # the next turn retries), never as "go ahead and publish".
    base = tmp_path / "ro"
    base.mkdir()
    base.chmod(0o500)
    try:
        assert locks.acquire(base, "dash", ttl_s=60) is False
    finally:
        base.chmod(0o700)
