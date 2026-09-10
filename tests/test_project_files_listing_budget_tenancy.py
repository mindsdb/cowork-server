"""The listing cap counts files the caller may see, not candidates walked.

A project is shared by the whole organization, but `conversations/<id>/` under
it is one member's private workspace. Those files are discarded by
`_conversation_workspace_ok` after the walk has produced them, so budgeting
candidates let a co-member's large workspace spend the entire listing on rows
that never ship — and the shared files were then silently absent.

Goes through the real HTTP stack: the budget is spent in the route, and a
handler-level test in local mode cannot see the difference because
`_conversation_workspace_ok` returns True unconditionally there.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlmodel import Session

import cowork.api.v1.endpoints.project_files as pf
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
OWNER = "11111111-1111-4111-8111-111111111111"
PEER = "33333333-3333-4333-8333-333333333333"

PEER_HEADERS = {"X-User-Id": PEER, "X-Organization-Id": ORG}
PROJECT = "listing-budget"

#: Small enough to blow through with a handful of files, so the test does not
#: have to create thousands.
CAP = 5


@pytest.fixture(scope="module")
def client():
    saved = {
        k: os.environ.get(k)
        for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")
    }
    os.environ["COWORK_TENANCY_MODE"] = "org"
    os.environ["COWORK_IDENTITY_ENFORCE"] = "enforce"
    get_app_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient

        from cowork.server import create_app

        # No `with`: skipping the lifespan skips boot migrations, which would
        # collide with the schema conftest already created.
        test_client = TestClient(create_app())
        try:
            yield test_client
        finally:
            test_client.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_app_settings.cache_clear()


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("listing-budget") / PROJECT
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = Project(
            id=uuid.uuid4(), name=PROJECT, path=str(root), org_id=ORG
        )
        conversation = Conversation(
            id=uuid.uuid4(),
            topic="the owner's chat",
            project_id=project.id,
            org_id=ORG,
            created_by=OWNER,
        )
        session.add(project)
        session.add(conversation)
        session.commit()
        conversation_id = conversation.id

    # The walk is breadth-first, so the shared file has to sit DEEPER than the
    # private ones for the budget to reach the private ones first. The private
    # workspace is at depth 3 and the shared file at depth 4.
    workspace = root / "conversations" / str(conversation_id)
    workspace.mkdir(parents=True)
    for i in range(CAP * 4):
        (workspace / f"private{i:03d}.txt").write_text("not the peer's")
    shared = root / "docs" / "2026" / "09"
    shared.mkdir(parents=True)
    (shared / "shared.txt").write_text("everyone may read this")

    yield {"root": root, "conversation_id": conversation_id}


def test_a_peers_private_workspace_does_not_spend_the_peers_listing(
    client, tree, monkeypatch
):
    monkeypatch.setattr(pf, "_MAX_LISTED_FILES", CAP)

    res = client.get(f"/api/v1/projects/{PROJECT}/files", headers=PEER_HEADERS)

    assert res.status_code == 200, res.text
    paths = {f["path"] for f in res.json()["files"]}
    # The file the peer is entitled to, which budgeting candidates dropped.
    assert "docs/2026/09/shared.txt" in paths
    # And none of the owner's, which is the pre-existing tenancy guarantee.
    assert not [p for p in paths if p.startswith("conversations/")]


def test_the_owner_still_sees_their_own_workspace(client, tree):
    owner_headers = {"X-User-Id": OWNER, "X-Organization-Id": ORG}

    res = client.get(f"/api/v1/projects/{PROJECT}/files", headers=owner_headers)

    assert res.status_code == 200, res.text
    paths = {f["path"] for f in res.json()["files"]}
    assert [p for p in paths if p.startswith("conversations/")]
