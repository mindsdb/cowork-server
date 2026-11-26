"""Stateful full-stack apps (metadata.json `type: "fullstack-stateful-app"`)
can't be published or shared for now (ENG-232): a published backend runs on
MindsHub's shared runner, where one owner's backend could read another's
credentials from a warm process. They still build and run in the desktop app.
Every publish path refuses them, the same way it refuses a widget (ENG-4684);
a stateless full-stack app publishes as before."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import cowork.services.artifact_autopublish  # noqa: F401  (see test_artifact_access_endpoints)
from cowork.api.v1.endpoints import artifact_workspace as aw
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.harnesses.anton_harness.tools import (
    COWORK_PUBLISH_PROMPT,
    SELF_HOSTED_PUBLISH_PROMPT,
)
from cowork.services import artifact_autopublish, publish
from cowork.services.artifact_metadata import (
    STATEFUL_APP_PUBLISH_REFUSAL,
    WIDGET_PUBLISH_REFUSAL,
    ArtifactMetadata,
    is_stateful_app,
    publish_refusal,
)
from cowork.services.artifact_permissions import (
    artifact_capabilities,
    with_display_limits,
)
from cowork.services.artifacts import html_artifacts

ORG_SCOPE = TenantScope(
    org_mode=True,
    org_id="0a0a0a0a-0000-4000-8000-000000000001",
    user_id="0b0b0b0b-0000-4000-8000-000000000001",
)
ARTIFACT_ID = "11111111-1111-1111-1111-111111111111"


def test_only_the_stateful_type_is_refused():
    assert is_stateful_app("fullstack-stateful-app")
    for other in (
        "fullstack-stateless-app",
        "html-app",
        "Fullstack-Stateful-App",
        "",
        None,
        ["fullstack-stateful-app"],
    ):
        assert not is_stateful_app(other)
    assert (
        publish_refusal(artifact_type="fullstack-stateful-app")
        == STATEFUL_APP_PUBLISH_REFUSAL
    )
    assert (
        publish_refusal(display="widget", artifact_type="fullstack-stateful-app")
        == WIDGET_PUBLISH_REFUSAL
    )
    assert publish_refusal(artifact_type="fullstack-stateless-app") is None


def test_capabilities_say_a_stateful_app_cant_be_published_or_shared():
    caps = artifact_capabilities(
        SimpleNamespace(scope=LOCAL_SCOPE),
        None,
        "notes",
        artifact_type="fullstack-stateful-app",
    )
    assert caps["canPublish"] is False and caps["canShare"] is False
    assert caps["publishRefusal"] == STATEFUL_APP_PUBLISH_REFUSAL
    # It still opens, edits and runs.
    assert caps["canEdit"] is True
    plain = artifact_capabilities(
        SimpleNamespace(scope=LOCAL_SCOPE),
        None,
        "board",
        artifact_type="fullstack-stateless-app",
    )
    assert (
        "canPublish" not in plain
        and "canShare" not in plain
        and "publishRefusal" not in plain
    )
    assert with_display_limits({"canEdit": True}, None, "fullstack-stateless-app") == {
        "canEdit": True
    }


def _app(base, slug, artifact_type):
    folder = base / slug
    (folder / "static").mkdir(parents=True)
    (folder / "backend.py").write_text("handler = None\n")
    (folder / "static" / "index.html").write_text("<html><body>app</body></html>")
    (folder / "metadata.json").write_text(
        json.dumps({"slug": slug, "type": artifact_type})
    )
    return folder


@pytest.fixture
def stateful_folder(tmp_path):
    base = tmp_path / ORG_SCOPE.org_id / "proj" / ".anton" / "artifacts"
    return _app(base, "notes", "fullstack-stateful-app")


def test_publish_artifact_refuses_a_stateful_app_before_anything_is_built(
    stateful_folder, monkeypatch
):
    def no_publisher(*_a, **_kw):
        raise AssertionError("nothing is built for a stateful app")

    monkeypatch.setattr(publish, "_a_hard_linked_file", no_publisher)

    with pytest.raises(publish.StatefulAppPublishRefused) as refused:
        publish.publish_artifact(
            stateful_folder,
            artifacts_base=stateful_folder.parent,
            api_key="k",
            publish_url="https://api.mindshub.ai",
        )
    # A ValueError: 400 on the REST paths, a failed tool result for the agent.
    assert isinstance(refused.value, ValueError)
    assert str(refused.value) == STATEFUL_APP_PUBLISH_REFUSAL


def test_autopublish_and_the_publishable_list_skip_a_stateful_app_but_not_a_stateless_one(
    stateful_folder, monkeypatch
):
    base = stateful_folder.parent
    decision = artifact_autopublish.needs_publish(stateful_folder, base)
    assert decision.reason == artifact_autopublish.REASON_NOT_PUBLISHABLE

    stateless = _app(base, "board", "fullstack-stateless-app")
    assert (
        artifact_autopublish.needs_publish(stateless, base).reason
        != artifact_autopublish.REASON_NOT_PUBLISHABLE
    )

    monkeypatch.setattr(
        "cowork.services.artifacts._scan_artifact_dirs", lambda scope: [base]
    )
    paths = [row["path"] for row in html_artifacts(scope=None)]
    assert not any("/notes/" in path for path in paths)
    assert any("/board/" in path for path in paths)


@pytest.fixture
def as_stateful_owner(monkeypatch, stateful_folder):
    monkeypatch.setattr(
        aw,
        "_owner_workspace",
        lambda scope, project_ref, artifact_id: _resolved(stateful_folder),
    )

    def never(*_a, **_kw):
        raise AssertionError("a stateful app is never published")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", never)
    return stateful_folder


async def _resolved(folder):
    return (
        SimpleNamespace(project_id="proj-1"),
        folder,
        ArtifactMetadata.model_validate(
            {"slug": "notes", "type": "fullstack-stateful-app"}
        ),
        {},
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("granted_product_permissions")
async def test_publish_changes_refuses_a_stateful_app(as_stateful_owner):
    with pytest.raises(HTTPException) as refused:
        await aw.republish_artifact("proj", ARTIFACT_ID, ORG_SCOPE, principal=None)
    assert refused.value.status_code == 400
    assert refused.value.detail == STATEFUL_APP_PUBLISH_REFUSAL


@pytest.mark.asyncio
@pytest.mark.usefixtures("granted_product_permissions")
async def test_sharing_refuses_a_stateful_app(as_stateful_owner):
    body = aw._AccessBody(access={"mode": "public"})
    with pytest.raises(HTTPException) as refused:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, body, ORG_SCOPE, SimpleNamespace(), principal=None
        )
    assert refused.value.status_code == 400
    assert refused.value.detail == STATEFUL_APP_PUBLISH_REFUSAL


def test_the_desktop_agent_hears_stateful_apps_dont_publish():
    assert "fullstack-stateful-app" in COWORK_PUBLISH_PROMPT
    # The self-hosted publisher refuses every full-stack app already.
    assert "fullstack-stateful-app" not in SELF_HOSTED_PUBLISH_PROMPT
