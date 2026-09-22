"""The three autopublish time budgets must stay consistent (ENG-1580).

publish_artifact runs in a thread that outlives asyncio.wait_for. It may keep
polling an async publish job and then write .published.json; the slug lock
must still be held when that happens, or another turn could start a second
publish of the same artifact.
"""
from unittest.mock import patch

import pytest

from cowork.services import artifact_autopublish as ap


def test_default_job_budget_is_90s():
    assert ap.job_budget_for(ap.lock_ttl_for(ap.DEFAULT_TIMEOUT_S), ap.DEFAULT_TIMEOUT_S) == 90.0


@pytest.mark.parametrize("timeout_s", [30.0, ap.DEFAULT_TIMEOUT_S, 120.0])
def test_invariant_holds_for_supported_timeouts(timeout_s):
    ttl = ap.lock_ttl_for(timeout_s)
    assert timeout_s + ap.PUBLISH_POST_TIMEOUT_S + ap.job_budget_for(ttl, timeout_s) <= ttl
    assert ap.job_budget_for(ttl, timeout_s) > 0


def test_publish_one_passes_job_budget_and_records_polling_phase(tmp_path):
    import asyncio
    import threading

    seen = {}
    accepted = threading.Event()

    def slow_publish(folder, **kw):
        seen.update(kw)
        kw["on_job_accepted"]({"job_id": "j"})
        accepted.set()
        # Outlive wait_for by a wide margin; the event above removes the race
        # between the thread start and the timeout.
        threading.Event().wait(1.0)

    records = []

    async def run():
        task = asyncio.create_task(
            ap._publish_one(tmp_path, "slug", "key", "https://p", 0.3, scope=None, job_budget_s=42.0)
        )
        await asyncio.to_thread(accepted.wait, 5.0)
        return await task

    with patch.object(ap, "publish_artifact", slow_publish), \
            patch.object(ap, "_record", lambda result, **f: records.append((result, f))):
        ok = asyncio.run(run())
    assert ok is False
    assert seen["job_budget_s"] == 42.0
    assert records[-1][0] == "timeout" and records[-1][1]["phase"] == "polling"
