"""The access anton's publisher is handed, from each caller of publish_artifact.

Every caller builds the publish's access as an ArtifactAccessRequest: the
desktop publish route and the agent's publish tool from what they were asked,
a re-publish from what the artifact keeps (kept_access), and the Share
dialog's save as the person chose it. publish_artifact turns it into anton's
shape once, at the call into anton's resolve_access
(cowork.services.publish._plan_access, through artifact_audience.for_publisher).
The desktop route's and the tool's cases run the real publish_artifact with
resolve_access recorded and the upload faked; the MindsHub-mode cases run
_plan_access, the step of publish_artifact that hands anton its access. Each
pins the dict anton receives for each mode.
"""

from __future__ import annotations

import asyncio
import copy
import json
from unittest import mock

import pytest
from anton.publish_access import resolve_access

from cowork.api.v1.endpoints import publish as routes
from cowork.db.scoped import LOCAL_SCOPE
from cowork.harnesses.anton_harness import tools as htools
from cowork.models.artifact_owner_choice import AccessMode
from cowork.schemas.artifact_access import ArtifactAccessRequest
from cowork.services import publish
from cowork.services.artifact_autopublish import AUTOPUBLISH_ACCESS
from tests._org_sharing import BOB, TEAM_ENG, TEAM_OPS, artifact_rule

PUBLISH_URL = "https://api.staging.mindshub.ai"


@pytest.fixture
def artifact(tmp_path):
    folder = tmp_path / "artifacts" / "rep"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html>v1</html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    return folder


@pytest.fixture
def anton_access(monkeypatch):
    """Each access anton's resolve_access is handed, in order; the upload
    itself answers like the publish lambda."""
    handed: list = []

    def recorded(password, access, previous):
        handed.append(copy.deepcopy(access))
        return resolve_access(password, access, previous)

    monkeypatch.setattr(publish, "_resolve_access", recorded)
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda _source, **kwargs: {"view_url": "https://view.example/r/1", "report_id": "rid-1", "md5": "m"},
    )
    return handed


@pytest.mark.parametrize(
    ("asked", "handed"),
    [
        (
            {"mode": "public"},
            {"mode": "public", "emails": [], "org_allowed": False, "owner_only": False},
        ),
        (
            {"mode": "password", "password": "hunter2"},
            {"mode": "password", "password": "hunter2", "emails": [], "org_allowed": False, "owner_only": False},
        ),
        (
            {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True},
            {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True, "owner_only": False},
        ),
        (
            {"mode": "restricted", "emails": [], "owner_only": True},
            {"mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True},
        ),
    ],
    ids=["public", "password", "restricted", "owner-only"],
)
def test_the_desktop_publish_route_hands_anton_each_mode(artifact, anton_access, monkeypatch, asked, handed):
    monkeypatch.setattr(
        routes, "_desktop_context", lambda raw, *, scope: (artifact, artifact.parent, "desktop-key", PUBLISH_URL)
    )

    asyncio.run(routes.publish_artifact(routes._PublishBody(path=str(artifact), access=asked), LOCAL_SCOPE))

    assert anton_access == [handed]


def test_the_desktop_publish_route_without_access_hands_anton_none(artifact, anton_access, monkeypatch):
    """A bare top-level password publishes password-protected: anton reads
    it beside an access of None."""
    monkeypatch.setattr(
        routes, "_desktop_context", lambda raw, *, scope: (artifact, artifact.parent, "desktop-key", PUBLISH_URL)
    )

    asyncio.run(routes.publish_artifact(routes._PublishBody(path=str(artifact), password="pw"), LOCAL_SCOPE))

    assert anton_access == [None]
    assert json.loads((artifact / ".published.json").read_text())["index.html"]["mode"] == "password"


@pytest.mark.parametrize(
    ("tool_input", "kept", "handed"),
    [
        ({"access_mode": "public"}, None, {"mode": "public"}),
        ({"access_mode": "password", "password": "hunter2"}, None, {"mode": "password", "password": "hunter2"}),
        (
            {"access_mode": "restricted", "emails": [" A@X.com ", "a@x.com"], "org_allowed": True},
            None,
            {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True, "owner_only": False},
        ),
        (
            {"access_mode": "restricted", "emails": [], "owner_only": True},
            None,
            {"mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True},
        ),
        (
            {},
            {"report_id": "r", "mode": "password", "requires_password": True, "access_password": "old"},
            {"mode": "password", "password": "old"},
        ),
        (
            {},
            {"report_id": "r", "mode": "restricted", "emails": ["a@x.com"], "org_allowed": False, "owner_only": False},
            {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": False, "owner_only": False},
        ),
        ({}, {}, {"mode": "public"}),
    ],
    ids=["public", "password", "restricted", "owner-only", "keeps-password", "keeps-restricted", "first-publish"],
)
def test_the_agents_publish_tool_hands_anton_each_mode(artifact, anton_access, monkeypatch, tool_input, kept, handed):
    """The tool's explicit fields, else the access the artifact keeps, else
    public."""
    monkeypatch.setattr(
        publish, "desktop_publish_context",
        lambda raw, *, scope: (artifact, artifact.parent, "desktop-key", PUBLISH_URL),
    )
    session = mock.Mock()
    session._workspace.base = str(artifact.parent)
    session._workspace.artifacts_dir = artifact.parent
    with mock.patch.object(htools, "_published_owner_state", return_value=kept or {}):
        asyncio.run(htools._cowork_publish_or_preview(
            session, {"file_path": str(artifact / "index.html"), "action": "publish", **tool_input},
        ))

    assert anton_access == [handed]


@pytest.mark.parametrize(
    ("access", "stored", "handed"),
    [
        # A live sync or an autopublish of a restricted entry (kept_access):
        # who it is shared with is the rule's.
        (
            lambda: publish.kept_access({"mode": "restricted", "access_version": 2}, org_mode=True),
            artifact_rule(emails=["a@else.com"], org=True, teams=[TEAM_OPS]),
            {"mode": "restricted", "owner_only": True, "emails": ["a@else.com"], "org_allowed": True},
        ),
        # A first autopublish: the owner alone.
        (
            lambda: AUTOPUBLISH_ACCESS,
            artifact_rule(),
            {"mode": "restricted", "emails": [], "owner_only": True, "org_allowed": False},
        ),
        # A save naming teams and people alone reaches anton as owner-only;
        # this server writes the teams and people to MindsHub itself.
        (
            lambda: ArtifactAccessRequest(
                mode=AccessMode.RESTRICTED, allowed_emails=[], org_allowed=False, allowed_team_ids=[TEAM_ENG],
                allowed_user_ids=[BOB],
            ),
            artifact_rule(),
            {"mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True},
        ),
        # A save of everyone and an outside address, its notify left out.
        (
            lambda: ArtifactAccessRequest.model_validate({
                "mode": "restricted", "org_allowed": True, "allowed_emails": ["e@else.com"],
                "notify": {"emails": ["e@else.com"]},
            }),
            artifact_rule(),
            {"mode": "restricted", "org_allowed": True, "emails": ["e@else.com"]},
        ),
        # A publish that isn't restricted reads no rule and keeps its fields.
        (lambda: ArtifactAccessRequest(mode=AccessMode.PASSWORD, password="pw"), None, {"mode": "password", "password": "pw"}),
        (lambda: ArtifactAccessRequest(mode=AccessMode.PUBLIC), None, {"mode": "public"}),
    ],
    ids=["kept-restricted", "first-autopublish", "teams-and-people", "everyone-and-emails", "password", "public"],
)
def test_a_publish_in_mindshub_mode_hands_anton_each_mode(anton_access, access, stored, handed):
    """``access`` is built in the test, so a caller's model that can't be
    built fails its own case."""
    publish._plan_access(None, access(), None, stored=stored)

    assert anton_access == [handed]
