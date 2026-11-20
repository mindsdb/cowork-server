"""publish_doc_page: the checks before a key is minted, the md5 cut, the first
publish's owner-only access, and letting go of the slot, lock and key."""
from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import cowork.services.publish as publish
from cowork.db.units import FilesystemBusy
from cowork.docs.service import create_doc
from cowork.docs.store import DOC_HTML
from cowork.services import live_publish as lp
from cowork.services.artifact_access import (
    ArtifactAccessRefused,
    ArtifactAccessUnavailable,
    ArtifactAudienceRejected,
)
from cowork.services.artifact_autopublish import AUTOPUBLISH_ACCESS
from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable


_REAL_OWNER_OF = lp._owner_of


class FakeKey:
    minted: list["FakeKey"] = []

    def __init__(self, user_id, org_id, *, min_ttl_s, workspace_id=None):
        self.user_id = user_id
        self.revoked = False
        FakeKey.minted.append(self)

    async def get(self):
        return "key"

    async def revoke(self):
        self.revoked = True


@pytest.fixture
def released(monkeypatch):
    """Slugs whose publish lock was let go."""
    slugs = []

    async def release(base, slug, *, scope):
        slugs.append(slug)

    monkeypatch.setattr(lp, "_release", release)
    return slugs


@pytest.fixture
def events(monkeypatch):
    """Events the hub was asked to publish."""
    seen = []
    monkeypatch.setattr(lp.hub, "publish", lambda topic, event, **kw: seen.append(event))
    return seen


@pytest.fixture
def doc(tmp_path, monkeypatch, released):
    FakeKey.minted = []
    monkeypatch.setattr(lp, "PublishKey", FakeKey)
    monkeypatch.setattr(lp, "_publish_url", lambda scope: "https://publish.test")
    monkeypatch.setattr(lp, "_active_workspace_id", lambda scope: None)

    # File store and settings calls run inline: the test has no org file store routing.
    async def inline_efs(fn, *, scope, deadline=True):
        return fn()

    async def inline_threads(fn):
        return fn()

    monkeypatch.setattr(lp, "run_efs", inline_efs)
    monkeypatch.setattr(lp, "run_on_unit_threads", inline_threads)

    async def owner(job, scope):
        return "owner-1"

    monkeypatch.setattr(lp, "_owner_of", owner)
    return create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")


@pytest.fixture
def late_upload(monkeypatch):
    """An upload that outlasts the publish timeout until the test opens the gate; it fails when asked to."""
    monkeypatch.setattr(lp, "PUBLISH_TIMEOUT_S", 0.05)

    def install(*, fail: bool = False) -> threading.Event:
        gate = threading.Event()

        def upload(*a, **k):
            gate.wait(5)
            if fail:
                raise RuntimeError("boom")
            return {"url": "u"}

        monkeypatch.setattr(lp, "publish_artifact", upload)
        return gate

    return install


def _job(made, **over):
    fields = dict(artifact_id=made.artifact_id, project_id="p1", artifacts_base=str(made.folder.parent),
                  slug=made.folder.name, org_id="org-1")
    fields.update(over)
    return lp.DocPublishJob(**fields)


def _mark_published(folder: Path, *, md5: str, published: bool = True) -> None:
    record = {DOC_HTML: {"url": "https://view.test/x", "report_id": "r1", "last_md5": md5, "published": published}}
    (folder / ".published.json").write_text(json.dumps(record))


async def _until(pred, *, tries=100, delay=0.02):
    for _ in range(tries):
        if pred():
            return
        await asyncio.sleep(delay)


def _late_failures(caplog):
    return [r for r in caplog.records if getattr(r, "outcome", None) == "late_failure"]


async def _run(job):
    done = []
    result = await lp.publish_doc_page(job, on_done=lambda: done.append(1))
    await lp.drain_cleanup(timeout=2.0)
    return result, done


def test_a_first_publish_is_owner_only_and_keyed_to_the_owner(doc, events, monkeypatch):
    calls = []
    monkeypatch.setattr(lp, "publish_artifact", lambda folder, **kw: calls.append(kw) or {"url": "u"})
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.PUBLISHED) and done == [1]
    assert calls[0]["access"] == AUTOPUBLISH_ACCESS and calls[0]["background"] is True
    assert calls[0]["scope"].user_id == "owner-1" and calls[0]["project_id"] == "p1"
    assert FakeKey.minted[0].user_id == "owner-1" and FakeKey.minted[0].revoked
    assert events == [{"type": "doc.published", "rev": 0}]


def test_an_unchanged_page_mints_no_key_and_uploads_nothing(doc, monkeypatch):
    monkeypatch.setattr(lp, "compute_publish_md5", lambda page, **kw: "same")
    _mark_published(doc.folder, md5="same")
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: pytest.fail("uploaded"))
    result, done = asyncio.run(_run(_job(doc)))
    assert result.outcome is lp.Outcome.UNCHANGED and done == [1] and FakeKey.minted == []


def test_a_changed_page_keeps_the_access_its_owner_chose(doc, monkeypatch):
    monkeypatch.setattr(lp, "compute_publish_md5", lambda page, **kw: "new")
    _mark_published(doc.folder, md5="old")
    kept = object()
    monkeypatch.setattr(lp, "kept_access", lambda entry, *, org_mode: kept)
    calls = []
    monkeypatch.setattr(lp, "publish_artifact", lambda folder, **kw: calls.append(kw) or {"url": "u"})
    result, _ = asyncio.run(_run(_job(doc)))
    assert result.outcome is lp.Outcome.PUBLISHED and calls[0]["access"] is kept


def test_a_page_its_owner_unpublished_is_left_alone(doc, monkeypatch):
    _mark_published(doc.folder, md5="old", published=False)
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: pytest.fail("uploaded"))
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.SKIPPED, "unpublished") and done == [1]


@pytest.mark.parametrize("over, reason", [
    ({"slug": "../other"}, "bad_slug"),
    ({"slug": ".hidden"}, "bad_slug"),
    ({"slug": "a/b"}, "bad_slug"),
    ({"slug": "missing"}, "not_a_doc"),
    ({"artifact_id": "00000000-0000-0000-0000-000000000000"}, "id_mismatch"),
])
def test_a_record_that_names_another_folder_is_skipped(doc, monkeypatch, over, reason):
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: pytest.fail("uploaded"))
    result, done = asyncio.run(_run(_job(doc, **over)))
    assert result == lp.PublishResult(lp.Outcome.SKIPPED, reason) and done == [1] and FakeKey.minted == []


def test_an_unknown_owner_is_skipped_before_a_key(doc, monkeypatch):
    async def nobody(job, scope):
        return None

    monkeypatch.setattr(lp, "_owner_of", nobody)
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.SKIPPED, "owner_unknown") and done == [1]
    assert FakeKey.minted == []


def test_a_busy_lock_answers_retry_within_a_second_and_frees_the_slot(doc, monkeypatch):
    monkeypatch.setattr(lp, "acquire", lambda base, slug, *, ttl_s: False)
    monkeypatch.setattr(lp, "LOCK_WAIT_S", 0.2)
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.RETRY, "lock_busy") and done == [1] and FakeKey.minted == []


def test_no_key_answers_retry_and_lets_the_lock_go(doc, released, monkeypatch):
    class NoKey(FakeKey):
        async def get(self):
            return None

    monkeypatch.setattr(lp, "PublishKey", NoKey)
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.RETRY, "no_key") and done == [1]
    assert released == [doc.folder.name]


def test_on_done_comes_after_the_lock_is_released(doc, monkeypatch):
    order = []

    async def release(base, slug, *, scope):
        order.append("release")

    monkeypatch.setattr(lp, "_release", release)
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: {"url": "u"})

    async def scenario():
        await lp.publish_doc_page(_job(doc), on_done=lambda: order.append("done"))
        await lp.drain_cleanup(timeout=2.0)

    asyncio.run(scenario())
    assert order == ["release", "done"]


def test_after_a_timeout_the_slot_lock_and_key_wait_for_the_thread(doc, released, late_upload):
    gate = late_upload()

    async def scenario():
        done = []
        with pytest.raises(asyncio.TimeoutError):
            await lp.publish_doc_page(_job(doc), on_done=lambda: done.append(1))
        held = (list(done), list(released), FakeKey.minted[0].revoked)
        gate.set()
        await _until(lambda: done)
        await lp.drain_cleanup(timeout=2.0)
        return held, done

    held, done = asyncio.run(scenario())
    assert held == ([], [], False)
    assert done == [1] and released == [doc.folder.name] and FakeKey.minted[0].revoked


def test_an_upload_error_reaches_the_caller_after_letting_go(doc, monkeypatch):
    def too_large(*a, **k):
        raise _bare(publish.PublishTooLarge)

    monkeypatch.setattr(lp, "publish_artifact", too_large)

    async def scenario():
        done = []
        with pytest.raises(publish.PublishTooLarge):
            await lp.publish_doc_page(_job(doc), on_done=lambda: done.append(1))
        await _until(lambda: done)
        await lp.drain_cleanup(timeout=2.0)
        return done

    assert asyncio.run(scenario()) == [1]


def test_the_upload_thread_sees_the_callers_context(doc, monkeypatch):
    marker = contextvars.ContextVar("marker", default=None)
    seen = []
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: seen.append(marker.get()) or {"url": "u"})

    async def scenario():
        marker.set("request-1")
        await lp.publish_doc_page(_job(doc), on_done=lambda: None)
        await lp.drain_cleanup(timeout=2.0)

    asyncio.run(scenario())
    assert seen == ["request-1"]


def _bare(exc_type):
    return exc_type.__new__(exc_type)


@pytest.mark.parametrize("exc, permanent", [
    (_bare(publish.AccessModeChanged), True),
    (_bare(publish.ShareOrgRequired), True),
    (_bare(publish.PublishTooLarge), True),
    (_bare(publish.PublishOwnedByOtherUser), True),
    (_bare(publish.ComparisonPublishRefused), True),
    (ArtifactAccessRefused("x"), True),
    (_bare(ArtifactAudienceRejected), True),
    (_bare(ProductPermissionDenied), True),
    (_bare(publish.PublishRejected), False),
    (_bare(publish.PublisherUnavailable), False),
    (ArtifactAccessUnavailable("x"), False),
    (RuntimeError("Publishing failed: x"), False),
    (ValueError("a partial read"), False),
    (asyncio.TimeoutError(), False),
    (_bare(ProductPermissionUnavailable), False),
], ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else str(value))
def test_which_errors_are_permanent(exc, permanent):
    assert lp.is_permanent(exc) is permanent


@pytest.mark.parametrize("resolution, owner", [
    (ValueError("Unknown project"), None),
    (None, None),
    (SimpleNamespace(unknown=True, owner_user_id=None), None),
    (SimpleNamespace(unknown=False, owner_user_id=42), "42"),
], ids=["unknown_project", "base_not_the_project_root", "unknown_resolution", "known_owner_as_a_string"])
def test_owner_of(doc, monkeypatch, resolution, owner):
    job = _job(doc)

    def owners(*a, **k):
        if isinstance(resolution, Exception):
            raise resolution
        return None if resolution is None else {job.slug: resolution}

    async def inline_db(fn, *, scope):
        return fn(object())

    monkeypatch.setattr(lp, "run_db", inline_db)
    monkeypatch.setattr(lp, "_artifact_owners", owners)
    assert asyncio.run(_REAL_OWNER_OF(job, lp.TenantScope(org_mode=True, org_id="org-1"))) == owner


def test_an_upload_finishing_after_the_timeout_still_tells_the_editors(doc, events, late_upload):
    gate = late_upload()

    async def scenario():
        with pytest.raises(asyncio.TimeoutError):
            await lp.publish_doc_page(_job(doc), on_done=lambda: None)
        before = list(events)
        gate.set()
        await _until(lambda: events)
        await lp.drain_cleanup(timeout=2.0)
        return before

    before = asyncio.run(scenario())
    assert before == [] and events == [{"type": "doc.published", "rev": 0}]


def test_an_upload_failing_after_the_timeout_logs_and_publishes_nothing(doc, events, late_upload, caplog):
    gate = late_upload(fail=True)

    async def scenario():
        with pytest.raises(asyncio.TimeoutError):
            await lp.publish_doc_page(_job(doc), on_done=lambda: None)
        gate.set()
        await _until(lambda: _late_failures(caplog))
        await lp.drain_cleanup(timeout=2.0)

    with caplog.at_level("WARNING", logger=lp.logger.name):
        asyncio.run(scenario())
    late = _late_failures(caplog)
    assert len(late) == 1 and late[0].levelname == "WARNING"
    assert late[0].error_type == "RuntimeError" and late[0].artifact_slug == doc.folder.name
    assert events == []


def test_a_failing_announcement_still_lets_go_of_the_slot_and_lock(doc, released, monkeypatch):
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: {"url": "u"})

    def broken(topic, event, **kw):
        raise RuntimeError("hub down")

    monkeypatch.setattr(lp.hub, "publish", broken)

    async def scenario():
        done = []
        await lp.publish_doc_page(_job(doc), on_done=lambda: done.append(1))
        await _until(lambda: done)
        await lp.drain_cleanup(timeout=2.0)
        return done

    assert asyncio.run(scenario()) == [1] and released == [doc.folder.name]


def test_an_upload_failing_after_the_caller_was_cancelled_logs_and_publishes_nothing(doc, events, monkeypatch,
                                                                                     caplog):
    entered = threading.Event()
    gate = threading.Event()

    def fail_late(*a, **k):
        entered.set()
        gate.wait(5)
        raise RuntimeError("boom")

    monkeypatch.setattr(lp, "publish_artifact", fail_late)

    async def scenario():
        caller = asyncio.create_task(lp.publish_doc_page(_job(doc), on_done=lambda: None))
        await _until(entered.is_set, tries=250)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        gate.set()
        await _until(lambda: _late_failures(caplog))
        await lp.drain_cleanup(timeout=2.0)

    with caplog.at_level("WARNING", logger=lp.logger.name):
        asyncio.run(scenario())
    late = _late_failures(caplog)
    assert len(late) == 1 and late[0].levelname == "WARNING"
    assert events == []


def test_a_busy_file_store_while_taking_the_lock_answers_lock_busy(doc, monkeypatch):
    # The inline run_efs passes the busy store up the way the real one refuses a slot.
    def busy(base, slug, *, ttl_s):
        raise FilesystemBusy(limiter="run_efs")

    monkeypatch.setattr(lp, "acquire", busy)
    monkeypatch.setattr(lp, "LOCK_WAIT_S", 0.2)
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.RETRY, "lock_busy") and done == [1] and FakeKey.minted == []


def test_a_cancel_after_the_lock_was_taken_lets_it_go(doc, released, monkeypatch):
    monkeypatch.setattr(lp, "acquire", lambda base, slug, *, ttl_s: True)

    # The real run_efs waits for a take that already runs, then passes the cancel on.
    async def cancelled_after_the_take(fn, *, scope, deadline=True):
        result = fn()
        if result is True:
            raise asyncio.CancelledError
        return result

    monkeypatch.setattr(lp, "run_efs", cancelled_after_the_take)

    async def scenario():
        done = []
        with pytest.raises(asyncio.CancelledError):
            await lp.publish_doc_page(_job(doc), on_done=lambda: done.append(1))
        return done

    assert asyncio.run(scenario()) == [1] and released == [doc.folder.name] and FakeKey.minted == []


def test_an_empty_id_never_matches_a_doc_without_one(doc, monkeypatch):
    monkeypatch.setattr(lp, "load_doc", lambda folder: {})
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: pytest.fail("uploaded"))
    result, done = asyncio.run(_run(_job(doc, artifact_id="")))
    assert result == lp.PublishResult(lp.Outcome.SKIPPED, "id_mismatch") and done == [1]


def test_the_project_root_is_checked_before_any_file_is_read(doc, monkeypatch):
    async def nobody(job, scope):
        return None

    monkeypatch.setattr(lp, "_owner_of", nobody)
    monkeypatch.setattr(lp, "_doc_folder", lambda job: pytest.fail("read the doc folder"))
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.SKIPPED, "owner_unknown") and done == [1]


def test_a_bad_slug_is_refused_before_the_owner_lookup(doc, monkeypatch):
    async def no_lookup(job, scope):
        pytest.fail("looked up the owner")

    monkeypatch.setattr(lp, "_owner_of", no_lookup)
    result, done = asyncio.run(_run(_job(doc, slug="../other")))
    assert result == lp.PublishResult(lp.Outcome.SKIPPED, "bad_slug") and done == [1]
