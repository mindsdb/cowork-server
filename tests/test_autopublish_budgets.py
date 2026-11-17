"""The autopublish time budgets must stay consistent.

publish_artifact runs in a thread that outlives asyncio.wait_for. It may keep
polling an async publish job and then write .published.json; the slug lock
must still be held when that happens, or another turn could start a second
publish of the same artifact.
"""
import asyncio
import os
import threading
import time
from unittest.mock import patch

import pytest

from cowork.services import artifact_autopublish as ap
from cowork.services import artifact_locks as locks


@pytest.mark.parametrize(
    ("timeout_s", "lock_ttl", "job_budget"),
    [(30.0, 90.0, 45.0), (ap.DEFAULT_TIMEOUT_S, 180.0, 135.0), (120.0, 360.0, 315.0)],
)
def test_budgets_for_supported_timeouts(timeout_s, lock_ttl, job_budget):
    # The default budget fits a heavy fullstack build, which anton expects to
    # take one to two minutes.
    assert ap.lock_ttl_for(timeout_s) == lock_ttl
    assert ap.job_budget_for(lock_ttl) == job_budget
    # Counted from the refresh at the accepted job, the poll and the work
    # after it leave the lock half a minute to spare.
    assert lock_ttl - job_budget - ap.PUBLISH_THREAD_SLACK_S >= 30.0


def test_publish_one_passes_job_budget_and_records_polling_phase(tmp_path):
    seen = {}
    release = threading.Event()
    assert locks.acquire(tmp_path, "slug", ttl_s=180)
    lock_file = tmp_path / locks.LOCKS_DIRNAME / "slug.lock"
    # A slow upload: the lock would expire during the poll without a refresh.
    old = time.time() - 170
    os.utime(lock_file, (old, old))

    def polling_publish(folder, **kw):
        seen.update(kw)
        kw["on_job_accepted"]({"job_id": "j"})
        release.wait(5.0)

    records = []
    abandoned: set[str] = set()

    async def run():
        # wait_for's clock starts with the call, so the timeout leaves the
        # thread a full second to report the accepted job first. The release
        # comes before asyncio.run waits for its executor's threads.
        try:
            return await ap._publish_one(
                tmp_path, "slug", "key", "https://p", 1.0, scope=None, job_budget_s=42.0, abandoned=abandoned,
            )
        finally:
            release.set()

    with patch.object(ap, "publish_artifact", polling_publish), \
            patch.object(ap, "_record", lambda result, **f: records.append((result, f))):
        ok = asyncio.run(run())
    assert ok is False
    assert seen["job_budget_s"] == 42.0
    assert records[-1][0] == "timeout" and records[-1][1]["phase"] == "polling"
    assert abandoned == {"slug"}
    assert time.time() - lock_file.stat().st_mtime < 5
