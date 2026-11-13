"""The autopublish time budgets must stay consistent.

publish_artifact runs in a thread that outlives asyncio.wait_for. It may keep
polling an async publish job and then write .published.json; the slug lock
must still be held when that happens, or another turn could start a second
publish of the same artifact.
"""
import asyncio
import threading
from unittest.mock import patch

import pytest

from cowork.services import artifact_autopublish as ap


def test_default_job_budget_fits_a_slow_dependency_install():
    # anton expects a heavy fullstack build to take one to two minutes.
    assert ap.job_budget_for(ap.lock_ttl_for(ap.DEFAULT_TIMEOUT_S)) == 135.0


@pytest.mark.parametrize("timeout_s", [30.0, ap.DEFAULT_TIMEOUT_S, 120.0])
def test_invariant_holds_for_supported_timeouts(timeout_s):
    # Counted from the lock's acquire: the wait_for window runs alongside the
    # thread, so it is not part of the sum.
    ttl = ap.lock_ttl_for(timeout_s)
    budget = ap.job_budget_for(ttl)
    assert ap.PUBLISH_POST_TIMEOUT_S + budget + ap.PUBLISH_THREAD_SLACK_S <= ttl
    assert budget > 0


def test_publish_one_passes_job_budget_and_records_polling_phase(tmp_path):
    seen = {}
    release = threading.Event()

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
