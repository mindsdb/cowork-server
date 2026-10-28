"""Owner-side share controls for org (Cloud) artifacts — ENG-2316.

Handlers are called directly, which is this suite's convention for the
workspace routes (the identity gate has its own over-HTTP file).

What matters here is the boundary: setting an audience is a re-publish, so it
must go through the org-mode publish path autopublish already proves works, and
it must be refused for anyone but the owner. The GET exists because the artifact
CARD deliberately withholds `accessEmails`/`accessPassword` in org mode — one
artifacts root is shared by the whole org, so a card cannot tell owner from
co-member. This route can.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import artifact_workspace as aw
from cowork.db.scoped import TenantScope
from cowork.services import artifact_locks

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
ARTIFACT_ID = "11111111-1111-1111-1111-111111111111"


class _Session:
    scope = ORG_SCOPE


@pytest.fixture
def folder(tmp_path):
    base = tmp_path / "org-1" / "proj" / ".anton" / "artifacts"
    f = base / "rep"
    f.mkdir(parents=True)
    (f / "report.html").write_text("<html></html>")
    (f / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    return f


@pytest.fixture
def as_owner(monkeypatch, folder):
    """Owner resolution succeeds and yields this folder."""
    monkeypatch.setattr(
        aw, "_owner_workspace",
        lambda session, project_ref, artifact_id: (
            SimpleNamespace(project_id="proj-1"), folder, {"slug": "rep"}, {},
        ),
    )
    return folder


@pytest.fixture
def publish_calls(monkeypatch):
    calls = []

    def fake_publish(artifact, *, artifacts_base, api_key, publish_url,
                     password=None, access=None, scope=None, project_id=None):
        calls.append({
            "folder": artifact, "artifacts_base": artifacts_base, "api_key": api_key,
            "publish_url": publish_url, "access": access, "scope": scope,
            "project_id": project_id,
        })
        return {"status": "ok", "url": "https://view.example/r/1"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fake_publish)
    return calls


@pytest.fixture
def publish_context(monkeypatch):
    """Org publish endpoint and a turn key; yields the list of revoke calls."""
    monkeypatch.setattr(
        "cowork.services.artifact_autopublish._publish_url",
        lambda scope: "https://api.staging.mindshub.ai",
    )
    revoked = []

    class FakeKey:
        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return "turnkey-1"

        async def revoke(self):
            revoked.append(True)

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey", FakeKey)
    return revoked


pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("granted_product_permissions")]


def _upload_service_answers(monkeypatch, status: int, reason: str, body: bytes) -> None:
    """The real publish path runs, and the upload service answers `status`."""
    import io
    from urllib.error import HTTPError

    def reject(*args, **kwargs):
        raise HTTPError("https://publish.example/upload", status, reason, {}, io.BytesIO(body))

    monkeypatch.setattr("anton.publisher.publish", reject)
    monkeypatch.setattr("cowork.services.publish.vault_for_scope", lambda scope: object())
    monkeypatch.setattr(
        "cowork.services.artifact_authorization_identity.publish_authorization_key",
        lambda *args: "owner/artifact",
    )


def _lock_is_free(folder) -> bool:
    free = artifact_locks.acquire(folder.parent, folder.name, ttl_s=60)
    if free:
        artifact_locks.release(folder.parent, folder.name)
    return free


async def test_setting_access_republishes_with_the_chosen_audience(
    as_owner, publish_calls, publish_context,
):
    access = {"mode": "restricted", "emails": ["a@example.com"], "org_allowed": False}

    out = await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access=access), _Session(),
    )

    assert out["url"] == "https://view.example/r/1"
    assert publish_calls[0]["access"] == access
    # The org-mode credential and endpoint, not the desktop provider settings.
    assert publish_calls[0]["api_key"] == "turnkey-1"
    assert publish_calls[0]["publish_url"] == "https://api.staging.mindshub.ai"
    # `vault_for_scope` RAISES on an org deployment without a scope, so dropping
    # this kwarg would fail every share rather than read the wrong vault.
    assert publish_calls[0]["scope"] is ORG_SCOPE
    assert publish_calls[0]["artifacts_base"] == as_owner.parent
    assert publish_calls[0]["project_id"] == "proj-1"


async def test_sharing_publicly_is_passed_through_verbatim(
    as_owner, publish_calls, publish_context,
):
    """The publisher owns the access schema; this route must not reinterpret it."""
    await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
    )

    assert publish_calls[0]["access"] == {"mode": "public"}


async def test_audience_change_waits_for_an_inflight_live_sync(
    as_owner, publish_calls, publish_context,
):
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    task = asyncio.create_task(
        aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "restricted"}), _Session(),
        )
    )

    try:
        await asyncio.sleep(0.05)
        assert publish_calls == []
    finally:
        artifact_locks.release(as_owner.parent, as_owner.name)

    await task
    assert publish_calls[0]["access"] == {"mode": "restricted"}
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    artifact_locks.release(as_owner.parent, as_owner.name)


@pytest.mark.parametrize("status", [403, 503])
async def test_share_preserves_mint_failure_after_waiting_for_publish_lock(
    as_owner, publish_calls, monkeypatch, status,
):
    from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable

    error_type = ProductPermissionDenied if status == 403 else ProductPermissionUnavailable
    admitted = asyncio.Event()

    async def allow_initial_check(scope, permission):
        admitted.set()

    async def reject_mint(**kwargs):
        raise error_type()

    monkeypatch.setattr(aw, "require_product_permission", allow_initial_check)
    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", reject_mint)
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60)
    task = asyncio.create_task(aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
    ))
    try:
        await admitted.wait()
    finally:
        artifact_locks.release(as_owner.parent, as_owner.name)
    with pytest.raises(error_type) as error:
        await task
    assert error.value.status_code == status
    assert publish_calls == []
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60)
    artifact_locks.release(as_owner.parent, as_owner.name)


@pytest.mark.parametrize("status", [403, 503])
async def test_share_preserves_authority_failure_from_the_artifact_consumer(
    as_owner, publish_context, monkeypatch, status,
):
    body = (
        {"code": "permission_denied"} if status == 403
        else {"error": "Auth unavailable", "code": "auth_unavailable"}
    )
    _upload_service_answers(monkeypatch, status, "Rejected", json.dumps(body).encode())
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert error.value.status_code == status
    assert error.value.detail["code"] == ("permission_denied" if status == 403 else "permission_unavailable")
    assert not (as_owner / ".published.json").exists()
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60)
    artifact_locks.release(as_owner.parent, as_owner.name)


async def test_a_non_owner_cannot_change_access(monkeypatch, publish_calls, publish_context):
    """`_owner_workspace` is the boundary — a reviewer holds a read grant, and
    must not be able to widen the audience with it."""
    def refuse(*_a, **_kw):
        raise HTTPException(status_code=403, detail="Not the owner")

    monkeypatch.setattr(aw, "_owner_workspace", refuse)

    with pytest.raises(HTTPException) as excinfo:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )

    assert excinfo.value.status_code == 403
    assert publish_calls == []


async def test_access_is_not_changed_when_no_publish_credential_is_available(
    as_owner, publish_calls, monkeypatch,
):
    """A share that cannot be performed must fail loudly, not report success."""
    monkeypatch.setattr(
        "cowork.services.artifact_autopublish._publish_url",
        lambda scope: "https://api.staging.mindshub.ai",
    )

    class NoKey:
        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return ""

        async def revoke(self):
            pass

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey", NoKey)

    with pytest.raises(HTTPException) as excinfo:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )

    assert excinfo.value.status_code == 503
    assert publish_calls == []


async def test_owner_reads_back_the_email_list_a_card_would_withhold(as_owner):
    """Pre-filling "selected users" needs the emails the org card strips."""
    (as_owner / ".published.json").write_text(json.dumps({
        "report.html": {
            "report_id": "rid", "url": "u", "published": True,
            "mode": "restricted", "emails": ["a@example.com"],
            "org_allowed": False, "owner_only": False,
        },
    }))

    out = await aw.artifact_access("proj", ARTIFACT_ID, _Session())

    assert out["accessMode"] == "restricted"
    assert out["accessEmails"] == ["a@example.com"]


async def test_identity_service_failure_does_not_report_a_successful_share(
    as_owner, publish_context, monkeypatch,
):
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    def unavailable(*_args, **_kwargs):
        raise ArtifactAccessUnavailable("Could not establish artifact authorization")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", unavailable)
    with pytest.raises(HTTPException) as excinfo:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert excinfo.value.status_code == 503
    assert not (as_owner / ".published.json").exists()
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    artifact_locks.release(as_owner.parent, as_owner.name)


async def test_owner_reads_back_an_owner_only_artifact(as_owner):
    (as_owner / ".published.json").write_text(json.dumps({
        "report.html": {
            "report_id": "rid", "url": "u", "published": True,
            "mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True,
        },
    }))

    out = await aw.artifact_access("proj", ARTIFACT_ID, _Session())

    assert out["ownerOnly"] is True
    assert out["accessEmails"] == []


async def test_share_of_a_report_owned_by_another_account_is_409(as_owner, publish_context, monkeypatch):
    """The org Share path says so instead of publishing a copy under a new URL."""
    from anton.publisher import ArtifactOwnedByOtherUserError

    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs)
        raise ArtifactOwnedByOtherUserError()

    monkeypatch.setattr("anton.publisher.publish", reject)
    monkeypatch.setattr("cowork.services.publish.vault_for_scope", lambda scope: object())
    monkeypatch.setattr(
        "cowork.services.artifact_authorization_identity.publish_authorization_key",
        lambda *args: "owner/artifact",
    )
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert error.value.status_code == 409
    assert error.value.detail == {
        "code": "artifact_owned_by_other_user",
        "message": "This artifact was published by another owner.",
    }
    assert len(calls) == 1
    assert not (as_owner / ".published.json").exists()
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    artifact_locks.release(as_owner.parent, as_owner.name)


async def test_share_reports_an_unmarked_upload_503_as_the_service_being_unavailable(
    as_owner, publish_context, monkeypatch,
):
    _upload_service_answers(monkeypatch, 503, "Service Unavailable", b'{"error": "Service Unavailable"}')
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert error.value.status_code == 503
    assert error.value.detail == {
        "code": "publish_unavailable",
        "message": "The publishing service is temporarily unavailable. Try again in a moment.",
    }
    assert not (as_owner / ".published.json").exists()
    assert _lock_is_free(as_owner)


async def test_a_too_large_artifact_is_413_with_its_code(as_owner, publish_context, monkeypatch):
    _upload_service_answers(monkeypatch, 413, "Payload Too Large", b'{"message": "Request Too Long"}')
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert error.value.status_code == 413
    assert error.value.detail == {
        "code": "artifact_too_large",
        "message": "This artifact is too large to publish. Remove or shrink large files in it, such as PDFs or images.",
    }
    assert not (as_owner / ".published.json").exists()
    assert _lock_is_free(as_owner)


async def test_a_rejected_publish_shows_the_upload_service_text(as_owner, publish_context, monkeypatch):
    from cowork.services.publish import PublishRejected

    def reject(*args, **kwargs):
        raise PublishRejected("Publishing failed: This organization is archived or still being provisioned.")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", reject)
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert error.value.status_code == 502
    assert error.value.detail == "Publishing failed: This organization is archived or still being provisioned."
    assert _lock_is_free(as_owner)


@pytest.mark.parametrize("local_error", [
    RuntimeError("Publishing failed: [Errno 2] No such file: '/data/org-1/proj/x.pdf'"),
    PermissionError("[Errno 13] Permission denied: '/data/org-1/proj/.published.json'"),
])
async def test_a_local_publish_failure_hides_its_text_but_logs_it(
    as_owner, publish_context, monkeypatch, caplog, local_error,
):
    import logging

    def fail(*args, **kwargs):
        raise local_error

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fail)
    with caplog.at_level(logging.WARNING, logger=aw.logger.name):
        with pytest.raises(HTTPException) as error:
            await aw.set_artifact_access(
                "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
            )
    assert error.value.status_code == 502
    assert error.value.detail == "Publishing failed. Try again in a moment."
    record = next(r for r in caplog.records if "Could not change access for artifact" in r.getMessage())
    assert record.exc_info is not None
    assert _lock_is_free(as_owner)


async def test_a_missing_publisher_is_503(as_owner, publish_context, monkeypatch):
    from cowork.services.publish import PublisherUnavailable

    def fail(*args, **kwargs):
        raise PublisherUnavailable("Anton publisher is unavailable")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fail)
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert error.value.status_code == 503
    assert error.value.detail == "Anton publisher is unavailable"


async def test_the_turn_key_is_revoked_after_a_share(as_owner, publish_calls, publish_context):
    await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
    )
    assert publish_context == [True]


async def test_the_turn_key_is_revoked_after_a_failed_share(as_owner, publish_context, monkeypatch):
    from cowork.services.publish import PublishRejected

    def reject(*args, **kwargs):
        raise PublishRejected("Publishing failed: nope")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", reject)
    with pytest.raises(HTTPException):
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
        )
    assert publish_context == [True]


async def test_the_turn_key_is_kept_while_a_timed_out_publish_still_runs(
    as_owner, publish_context, monkeypatch,
):
    import threading

    finish = threading.Event()

    def slow(*args, **kwargs):
        finish.wait(5)
        return {"status": "ok", "url": "https://view.example/r/1"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", slow)
    monkeypatch.setattr(aw, "_LIVE_PUBLISH_TIMEOUT_S", 0.05)
    try:
        with pytest.raises(HTTPException) as error:
            await aw.set_artifact_access(
                "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
            )
    finally:
        finish.set()
    assert error.value.detail == "Publishing timed out. Try again in a moment."
    assert publish_context == []
