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

import json

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import artifact_workspace as aw
from cowork.db.scoped import TenantScope

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
        lambda session, project_ref, artifact_id: (object(), folder, {"slug": "rep"}, {}),
    )
    return folder


@pytest.fixture
def publish_calls(monkeypatch):
    calls = []

    def fake_publish(artifact, *, artifacts_base, api_key, publish_url,
                     password=None, access=None, scope=None):
        calls.append({
            "folder": artifact, "artifacts_base": artifacts_base, "api_key": api_key,
            "publish_url": publish_url, "access": access, "scope": scope,
        })
        return {"status": "ok", "url": "https://view.example/r/1"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fake_publish)
    return calls


@pytest.fixture
def publish_context(monkeypatch):
    monkeypatch.setattr(
        "cowork.services.artifact_autopublish._publish_url",
        lambda scope: "https://api.staging.mindshub.ai",
    )

    class FakeKey:
        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return "turnkey-1"

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey", FakeKey)


pytestmark = pytest.mark.asyncio


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


async def test_sharing_publicly_is_passed_through_verbatim(
    as_owner, publish_calls, publish_context,
):
    """The publisher owns the access schema; this route must not reinterpret it."""
    await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(),
    )

    assert publish_calls[0]["access"] == {"mode": "public"}


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
