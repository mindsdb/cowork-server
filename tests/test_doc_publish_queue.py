"""The org republish queue on fakeredis: debounce with a cap, one replica per
claim, retries that never push back a fresher edit, and slots that come back."""
from __future__ import annotations

import asyncio
import logging

import fakeredis
import fakeredis.aioredis as fakeaioredis
import pytest

from cowork.docs import publish_queue as pq
from cowork.services.live_publish import DocPublishJob, Outcome, PublishResult

JOB = DocPublishJob(artifact_id="a1", project_id="p1", artifacts_base="/base", slug="plan", org_id="org-1")


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _client(server):
    return fakeaioredis.FakeRedis(server=server, decode_responses=True)


def test_a_repeat_edit_moves_the_due_time_up_to_the_cap(server):
    async def scenario():
        r = _client(server)
        await pq.enqueue(JOB, client=r, now=100.0)
        first = await r.zscore(pq.DUE_KEY, "a1")
        await pq.enqueue(JOB, client=r, now=110.0)
        second = await r.zscore(pq.DUE_KEY, "a1")
        await pq.enqueue(JOB, client=r, now=129.0)
        capped = await r.zscore(pq.DUE_KEY, "a1")
        fields = await r.hgetall(pq.doc_key("a1"))
        ttl = await r.ttl(pq.doc_key("a1"))
        return first, second, capped, fields, ttl

    first, second, capped, fields, ttl = asyncio.run(scenario())
    assert (first, second, capped) == (104.0, 114.0, 130.0)
    assert fields["slug"] == "plan" and fields["org_id"] == "org-1" and "user_id" not in fields
    assert 0 < ttl <= pq.HASH_TTL_S


def test_a_redis_error_does_not_reach_the_edit():
    class Broken:
        def pipeline(self, **kw):
            raise ConnectionError("down")

    asyncio.run(pq.enqueue(JOB, client=Broken(), now=1.0))


class Recorder:
    """A publish_doc_page stand-in: answers from a script and frees the slot."""

    def __init__(self, *answers, hold: asyncio.Event | None = None):
        self.answers = list(answers)
        self.jobs = []
        self.hold = hold

    async def __call__(self, job, *, on_done):
        self.jobs.append(job)
        try:
            if self.hold is not None:
                await self.hold.wait()
            answer = self.answers.pop(0) if self.answers else Outcome.PUBLISHED
            if isinstance(answer, BaseException):
                raise answer
            return answer if isinstance(answer, PublishResult) else PublishResult(answer)
        finally:
            on_done()


async def _settle(worker):
    for _ in range(50):
        if not worker._tasks:
            return
        await asyncio.sleep(0.01)


def test_of_two_replicas_one_publishes(server):
    async def scenario():
        await pq.enqueue(JOB, client=_client(server), now=0.0)
        a, b = Recorder(), Recorder()
        wa = pq.DocPublishWorker(_client(server), publish=a, clock=lambda: 10.0)
        wb = pq.DocPublishWorker(_client(server), publish=b, clock=lambda: 10.0)
        await asyncio.gather(wa.tick(), wb.tick())
        await _settle(wa)
        await _settle(wb)
        return len(a.jobs) + len(b.jobs), wa.free_slots, wb.free_slots

    assert asyncio.run(scenario()) == (1, pq.SLOTS, pq.SLOTS)


def test_an_edit_during_a_publish_is_published_after_it(server):
    async def scenario():
        r = _client(server)
        await pq.enqueue(JOB, client=r, now=0.0)
        hold = asyncio.Event()
        rec = Recorder(hold=hold)
        w = pq.DocPublishWorker(_client(server), publish=rec, clock=lambda: 10.0)
        await w.tick()
        await asyncio.sleep(0)
        await pq.enqueue(JOB, client=r, now=10.0)
        hold.set()
        await _settle(w)
        w._clock = lambda: 20.0
        await w.tick()
        await _settle(w)
        return len(rec.jobs)

    assert asyncio.run(scenario()) == 2


def test_a_retry_keeps_the_due_time_a_fresher_edit_set(server):
    async def scenario():
        r = _client(server)
        await pq.enqueue(JOB, client=r, now=0.0)
        hold = asyncio.Event()
        rec = Recorder(Outcome.RETRY, hold=hold)
        w = pq.DocPublishWorker(_client(server), publish=rec, clock=lambda: 10.0)
        await w.tick()
        await asyncio.sleep(0)
        await pq.enqueue(JOB, client=r, now=10.0)  # due 14
        hold.set()
        await _settle(w)
        return await r.zscore(pq.DUE_KEY, "a1"), await r.hget(pq.doc_key("a1"), "attempts")

    assert asyncio.run(scenario()) == (14.0, "1")


def test_retries_back_off_and_stop_after_the_last_attempt(server):
    async def scenario():
        r = _client(server)
        await pq.enqueue(JOB, client=r, now=0.0)
        rec = Recorder(*([Outcome.RETRY] * pq.MAX_ATTEMPTS))
        now = {"t": 10.0}
        w = pq.DocPublishWorker(_client(server), publish=rec, clock=lambda: now["t"])
        dues = []
        for _ in range(pq.MAX_ATTEMPTS):
            await w.tick()
            await _settle(w)
            due = await r.zscore(pq.DUE_KEY, "a1")
            dues.append(None if due is None else round(due - now["t"], 6))
            now["t"] = (due or now["t"]) + 0.1
        return dues, await r.hget(pq.doc_key("a1"), "attempts")

    dues, attempts = asyncio.run(scenario())
    assert dues == [5.0, 15.0, 60.0, 300.0, None] and attempts is None


def test_a_permanent_error_drops_the_entry_and_a_temporary_one_retries(server):
    from cowork.services.publish import PublishRejected, PublishTooLarge

    async def scenario(error):
        r = _client(server)
        await r.flushall()
        await pq.enqueue(JOB, client=r, now=0.0)
        w = pq.DocPublishWorker(_client(server), publish=Recorder(error), clock=lambda: 10.0)
        await w.tick()
        await _settle(w)
        return await r.zscore(pq.DUE_KEY, "a1"), w.free_slots

    too_large = PublishTooLarge.__new__(PublishTooLarge)
    assert asyncio.run(scenario(too_large)) == (None, pq.SLOTS)
    assert asyncio.run(scenario(PublishRejected("network"))) == (15.0, pq.SLOTS)


@pytest.mark.parametrize("fields", [{}, {"attempts": "2"}])
def test_an_entry_without_its_fields_is_skipped(server, fields):
    async def scenario():
        r = _client(server)
        await r.zadd(pq.DUE_KEY, {"gone": 1.0})
        if fields:
            await r.hset(pq.doc_key("gone"), mapping=fields)
        rec = Recorder()
        w = pq.DocPublishWorker(_client(server), publish=rec, clock=lambda: 10.0)
        await w.tick()
        await _settle(w)
        return rec.jobs, await r.zscore(pq.DUE_KEY, "gone"), w.free_slots

    assert asyncio.run(scenario()) == ([], None, pq.SLOTS)


def test_a_retry_keeps_the_entry_expiring(server):
    async def scenario():
        r = _client(server)
        await pq.enqueue(JOB, client=r, now=0.0)
        w = pq.DocPublishWorker(_client(server), publish=Recorder(Outcome.RETRY), clock=lambda: 10.0)
        await w.tick()
        await _settle(w)
        return await r.ttl(pq.doc_key("a1"))

    assert 0 < asyncio.run(scenario()) <= pq.HASH_TTL_S


def test_no_more_than_the_free_slots_are_claimed(server):
    async def scenario():
        r = _client(server)
        for i in range(pq.SLOTS + 2):
            await pq.enqueue(DocPublishJob(f"a{i}", "p1", "/base", f"s{i}", "org-1"), client=r, now=0.0)
        hold = asyncio.Event()
        rec = Recorder(hold=hold)
        w = pq.DocPublishWorker(_client(server), publish=rec, clock=lambda: 10.0)
        await w.tick()
        await asyncio.sleep(0)
        claimed, left = len(rec.jobs), await r.zcard(pq.DUE_KEY)
        hold.set()
        await _settle(w)
        return claimed, left, w.free_slots

    assert asyncio.run(scenario()) == (pq.SLOTS, 2, pq.SLOTS)


def test_stopping_puts_claimed_entries_back_and_waits_for_enqueues(server, monkeypatch):
    async def scenario():
        r = _client(server)
        monkeypatch.setattr(pq, "get_redis", lambda: r)
        monkeypatch.setattr(pq, "get_backend", lambda: "redis")
        await pq.enqueue(JOB, client=r, now=0.0)
        rec = Recorder(hold=asyncio.Event())
        w = pq.DocPublishWorker(r, publish=rec, clock=lambda: 10.0)
        await w.tick()
        await asyncio.sleep(0)
        assert pq.schedule_enqueue(DocPublishJob("b1", "p1", "/base", "other", "org-1"))
        await pq.stop_doc_publish_worker(w, budget_s=2.0)
        return await r.zscore(pq.DUE_KEY, "a1") is not None, await r.zscore(pq.DUE_KEY, "b1") is not None

    assert asyncio.run(scenario()) == (True, True)


def test_the_worker_starts_only_on_an_org_deployment_with_redis(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(pq, "get_app_settings", lambda: SimpleNamespace(tenancy_mode="local"))
    assert pq.start_doc_publish_worker() is None
    monkeypatch.setattr(pq, "get_app_settings", lambda: SimpleNamespace(tenancy_mode="org"))
    monkeypatch.setattr(pq, "get_backend", lambda: "file")
    assert pq.start_doc_publish_worker() is None


def test_the_log_fields_survive_the_json_formatter(caplog):
    from cowork.common.structured_logging import EVENT_FIELDS

    caplog.set_level(logging.DEBUG, logger=pq.logger.name)
    pq._record(PublishResult(Outcome.SKIPPED, "owner_unknown"), JOB, attempts=1, error_type="X")
    record = caplog.records[-1]
    extras = set(record.__dict__) - set(logging.makeLogRecord({}).__dict__) - {"message", "asctime"}
    assert extras and extras <= EVENT_FIELDS, extras - EVENT_FIELDS
    assert {k: getattr(record, k) for k in ("event", "outcome", "reason", "artifact_slug")} == {
        "event": "doc_publish", "outcome": "skipped", "reason": "owner_unknown", "artifact_slug": "plan"}
    assert record.levelno == logging.WARNING


def test_a_doc_its_owner_unpublished_logs_quietly(caplog):
    caplog.set_level(logging.DEBUG, logger=pq.logger.name)
    pq._record(PublishResult(Outcome.SKIPPED, "unpublished"), JOB)
    pq._record(PublishResult(Outcome.UNCHANGED), JOB)
    assert [r.levelno for r in caplog.records[-2:]] == [logging.DEBUG, logging.DEBUG]
