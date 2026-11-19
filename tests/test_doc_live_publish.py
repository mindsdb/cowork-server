"""publish_doc_page: the checks before a key is minted, the md5 cut, the first
publish's owner-only access, and letting go of the slot, lock and key."""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

from cowork.docs.service import create_doc
from cowork.docs.store import DOC_HTML
from cowork.services import live_publish as lp
from cowork.services.artifact_autopublish import AUTOPUBLISH_ACCESS


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
def doc(tmp_path, monkeypatch):
    FakeKey.minted = []
    created = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    made = SimpleNamespace(folder=created.folder, artifact_id=created.artifact_id)
    monkeypatch.setattr(lp, "PublishKey", FakeKey)
    monkeypatch.setattr(lp, "_publish_url", lambda scope: "https://publish.test")
    monkeypatch.setattr(lp, "_active_workspace_id", lambda scope: None)

    # File store and settings calls run inline: the test has no org file store routing.
    async def inline_efs(fn, *, scope, deadline=True):
        return fn()

    async def inline_cleanup(fn, *, scope):
        return fn()

    async def inline_threads(fn):
        return fn()

    monkeypatch.setattr(lp, "run_efs", inline_efs)
    monkeypatch.setattr(lp, "run_efs_cleanup", inline_cleanup)
    monkeypatch.setattr(lp, "run_on_unit_threads", inline_threads)

    async def owner(job, scope):
        return "owner-1"

    monkeypatch.setattr(lp, "_owner_of", owner)
    events = []
    monkeypatch.setattr(lp.hub, "publish", lambda topic, event, **kw: events.append(event))
    made.events = events
    return made


def _job(made, **over):
    fields = dict(artifact_id=made.artifact_id, project_id="p1", artifacts_base=str(made.folder.parent),
                  slug=made.folder.name, org_id="org-1")
    fields.update(over)
    return lp.DocPublishJob(**fields)


def _mark_published(folder: Path, *, md5: str, published: bool = True) -> None:
    record = {DOC_HTML: {"url": "https://view.test/x", "report_id": "r1", "last_md5": md5, "published": published}}
    (folder / ".published.json").write_text(json.dumps(record))


async def _run(job):
    done = []
    result = await lp.publish_doc_page(job, on_done=lambda: done.append(1))
    await lp.drain_cleanup(timeout=2.0)
    return result, done


def test_a_first_publish_is_owner_only_and_keyed_to_the_owner(doc, monkeypatch):
    calls = []
    monkeypatch.setattr(lp, "publish_artifact", lambda folder, **kw: calls.append(kw) or {"url": "u"})
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.PUBLISHED) and done == [1]
    assert calls[0]["access"] == AUTOPUBLISH_ACCESS and calls[0]["background"] is True
    assert calls[0]["scope"].user_id == "owner-1" and calls[0]["project_id"] == "p1"
    assert FakeKey.minted[0].user_id == "owner-1" and FakeKey.minted[0].revoked
    assert doc.events == [{"type": "doc.published", "rev": 0}]


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


def test_no_key_answers_retry_and_lets_the_lock_go(doc, monkeypatch):
    released = []
    monkeypatch.setattr(lp, "release", lambda base, slug: released.append(slug))

    class NoKey(FakeKey):
        async def get(self):
            return None

    monkeypatch.setattr(lp, "PublishKey", NoKey)
    result, done = asyncio.run(_run(_job(doc)))
    assert result == lp.PublishResult(lp.Outcome.RETRY, "no_key") and done == [1]
    assert released == [doc.folder.name]


def test_on_done_comes_after_the_lock_is_released(doc, monkeypatch):
    order = []
    monkeypatch.setattr(lp, "release", lambda base, slug: order.append("release"))
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: {"url": "u"})

    async def scenario():
        await lp.publish_doc_page(_job(doc), on_done=lambda: order.append("done"))
        await lp.drain_cleanup(timeout=2.0)

    asyncio.run(scenario())
    assert order == ["release", "done"]


def test_after_a_timeout_the_slot_lock_and_key_wait_for_the_thread(doc, monkeypatch):
    gate = threading.Event()
    released = []
    monkeypatch.setattr(lp, "release", lambda base, slug: released.append(slug))
    monkeypatch.setattr(lp, "PUBLISH_TIMEOUT_S", 0.05)
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: gate.wait(5) or {"url": "u"})

    async def scenario():
        done = []
        with pytest.raises(asyncio.TimeoutError):
            await lp.publish_doc_page(_job(doc), on_done=lambda: done.append(1))
        held = (list(done), list(released), FakeKey.minted[0].revoked)
        gate.set()
        for _ in range(100):
            if done:
                break
            await asyncio.sleep(0.02)
        await lp.drain_cleanup(timeout=2.0)
        return held, done, released

    held, done, released_after = asyncio.run(scenario())
    assert held == ([], [], False)
    assert done == [1] and released_after == [doc.folder.name] and FakeKey.minted[0].revoked


def test_an_upload_error_reaches_the_caller_after_letting_go(doc, monkeypatch):
    from cowork.services.publish import PublishTooLarge

    def too_large(*a, **k):
        raise PublishTooLarge.__new__(PublishTooLarge)

    monkeypatch.setattr(lp, "publish_artifact", too_large)

    async def scenario():
        done = []
        with pytest.raises(PublishTooLarge):
            await lp.publish_doc_page(_job(doc), on_done=lambda: done.append(1))
        for _ in range(100):
            if done:
                break
            await asyncio.sleep(0.02)
        await lp.drain_cleanup(timeout=2.0)
        return done

    assert asyncio.run(scenario()) == [1]


def test_the_upload_thread_sees_the_callers_context(doc, monkeypatch):
    import contextvars

    marker = contextvars.ContextVar("marker", default=None)
    seen = []
    monkeypatch.setattr(lp, "publish_artifact", lambda *a, **k: seen.append(marker.get()) or {"url": "u"})

    async def scenario():
        marker.set("request-1")
        await lp.publish_doc_page(_job(doc), on_done=lambda: None)
        await lp.drain_cleanup(timeout=2.0)

    asyncio.run(scenario())
    assert seen == ["request-1"]


@pytest.mark.parametrize("name, permanent", [
    ("AccessModeChanged", True), ("ShareOrgRequired", True), ("PublishTooLarge", True),
    ("PublishOwnedByOtherUser", True), ("ComparisonPublishRefused", True),
    ("PublishRejected", False), ("PublisherUnavailable", False),
])
def test_which_publish_errors_are_permanent(name, permanent):
    import cowork.services.publish as publish

    exc_type = getattr(publish, name)
    exc = exc_type.__new__(exc_type)
    assert lp.is_permanent(exc) is permanent


def test_other_errors_are_permanent_or_not_by_type():
    from cowork.services.artifact_access import (
        ArtifactAccessRefused,
        ArtifactAccessUnavailable,
        ArtifactAudienceRejected,
    )
    from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable

    assert lp.is_permanent(ArtifactAccessRefused("x"))
    assert lp.is_permanent(ArtifactAudienceRejected.__new__(ArtifactAudienceRejected))
    assert not lp.is_permanent(ArtifactAccessUnavailable("x"))
    assert not lp.is_permanent(RuntimeError("Publishing failed: x"))
    assert not lp.is_permanent(ValueError("a partial read"))
    assert not lp.is_permanent(asyncio.TimeoutError())
    assert lp.is_permanent(ProductPermissionDenied.__new__(ProductPermissionDenied))
    assert not lp.is_permanent(ProductPermissionUnavailable.__new__(ProductPermissionUnavailable))
