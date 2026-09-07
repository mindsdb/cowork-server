"""Organisation isolation through the cowork-server application HTTP stack.

Boots create_app() in org mode with identity enforcement and exercises trusted
gateway-style identity headers through middleware, dependencies, services, and
the database.

For each covered resource, an organisation probing another organisation's
identifier must receive the same application response as it receives for a
nonexistent identifier. Destructive probes must also leave the owning
organisation's resource unchanged.

This suite assumes the identity headers were supplied by a trusted gateway. It
does not test gateway header spoofing or Kubernetes/S3 isolation.

The other axis, one member of an organisation against another member of the
same one, lives beside it rather than here: tests/test_project_files_tenancy.py
for the file and preview routes, tests/test_artifacts_api_tenancy.py and
tests/test_artifact_roots.py for live artifacts, and the per-service
tests/test_*_tenancy.py files for conversations, files, schedules and settings.
"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

from cowork.common.settings.app_settings import get_app_settings

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"
USER_A_PEER = "33333333-3333-4333-8333-333333333333"
USER_A_ADMIN = "44444444-4444-4444-8444-444444444444"

A = {
    "X-User-Id": USER_A,
    "X-User-Email": "creator@example.com",
    "X-Organization-Id": ORG_A,
}
B = {"X-User-Id": USER_B, "X-Organization-Id": ORG_B}
A_PEER = {
    "X-User-Id": USER_A_PEER,
    "X-User-Email": "peer@example.com",
    "X-Organization-Id": ORG_A,
}
A_OTHER_ADMIN = {
    "X-User-Id": USER_A_ADMIN,
    "X-User-Email": "admin@example.com",
    "X-Organization-Id": ORG_A,
    "X-User-Roles": "manage-organization",
}


@pytest.fixture(scope="module")
def client():
    saved = {k: os.environ.get(k) for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")}
    os.environ["COWORK_TENANCY_MODE"] = "org"
    os.environ["COWORK_IDENTITY_ENFORCE"] = "enforce"
    get_app_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from cowork.server import create_app

        # No `with`: skipping the lifespan skips boot migrations (they would
        # collide with the schema conftest already created) and never starts
        # background workers in this suite.
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


def _normalized(res, probe_id: str) -> bytes:
    """Response body with the caller-supplied probe id blanked, so bodies that
    echo the (attacker-chosen) id still compare equal across probes."""
    return res.content.replace(probe_id.encode(), b"<ID>")


def _assert_same_response(cross, missing, cross_id: str, missing_id: str, *, status: int = 404):
    assert cross.status_code == missing.status_code == status, (
        f"cross={cross.status_code} {cross.text!r} missing={missing.status_code} {missing.text!r}"
    )
    assert _normalized(cross, cross_id) == _normalized(missing, missing_id)
    assert cross.headers.get("content-type") == missing.headers.get("content-type")


def _same_as_missing(client, method: str, path_for, real_id: str, *, status: int = 404):
    """Org B probing org A's resource must answer exactly like probing a valid
    nonexistent id - the no-differential-leak property."""
    missing_id = str(uuid4())
    cross = client.request(method, path_for(real_id), headers=B)
    missing = client.request(method, path_for(missing_id), headers=B)
    _assert_same_response(cross, missing, real_id, missing_id, status=status)
    return cross


def _same_as_missing_body(client, method: str, url: str, body_for, real_id: str, *,
                          status: int = 404, headers=B):
    """Body-based variant: the foreign id travels in the JSON payload."""
    missing_id = str(uuid4())
    cross = client.request(method, url, headers=headers, json=body_for(real_id))
    missing = client.request(method, url, headers=headers, json=body_for(missing_id))
    _assert_same_response(cross, missing, real_id, missing_id, status=status)
    return cross


def _project(client, headers) -> str:
    res = client.post("/api/v1/projects/", json={"name": f"p-{uuid4().hex[:6]}"}, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()["id"]


def _conversation(client, headers, topic="t") -> str:
    """Project + conversation for `headers`' org. (A fresh org has no GENERAL
    project without the lifespan seed, so a bare conversation create 404s.)"""
    created = client.post("/api/v1/conversations/",
                          json={"topic": topic, "projectId": _project(client, headers)},
                          headers=headers)
    assert created.status_code == 201, created.text
    return created.json()["id"]


# -- identity enforcement -----------------------------------------------------

def test_no_identity_is_401(client):
    # Header parsing/edge cases live in test_principal.py (middleware-level);
    # these prove create_app() actually mounts enforcement in org mode.
    res = client.get("/api/v1/projects/")
    assert res.status_code == 401, res.text


def test_user_without_org_is_401(client):
    res = client.get("/api/v1/projects/", headers={"X-User-Id": USER_A})
    assert res.status_code == 401, res.text


def test_org_without_user_is_401(client):
    res = client.get("/api/v1/projects/", headers={"X-Organization-Id": ORG_A})
    assert res.status_code == 401, res.text


# -- projects -----------------------------------------------------------------

def test_projects_isolated(client):
    pid = _project(client, A)

    b_list = client.get("/api/v1/projects/", headers=B)
    assert b_list.status_code == 200, b_list.text
    b_rows = b_list.json()
    assert isinstance(b_rows, list)
    assert pid not in [p["id"] for p in b_rows]

    a_before = [p for p in client.get("/api/v1/projects/", headers=A).json() if p["id"] == pid]
    _same_as_missing(client, "DELETE", lambda i: f"/api/v1/projects/{i}", pid)
    a_after = [p for p in client.get("/api/v1/projects/", headers=A).json() if p["id"] == pid]
    assert a_after == a_before  # row unchanged, not merely present


def test_shared_resource_authorization_and_contract_through_http(client):
    """Exercise same-org policy through middleware, DI, routes, and storage."""
    project_name = f"shared-{uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/projects/",
        json={"name": project_name},
        headers=A,
    )
    assert created.status_code == 201, created.text
    project = created.json()
    project_id = project["id"]
    assert project["attribution"]["createdBy"] == {
        "userId": USER_A,
        "email": "creator@example.com",
    }
    assert project["capabilities"] == {
        "canRename": True,
        "canDelete": True,
        "canEditInstructions": True,
    }

    denied = client.patch(
        f"/api/v1/projects/{project_id}",
        json={"name": f"{project_name}-peer"},
        headers=A_PEER,
    )
    assert denied.status_code == 403, denied.text

    # Echoing the current name must remain a selection-only update for a peer.
    selected = client.patch(
        f"/api/v1/projects/{project_id}",
        json={"name": project_name, "isActive": True},
        headers=A_PEER,
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["name"] == project_name

    renamed_name = f"{project_name}-admin"
    renamed = client.patch(
        f"/api/v1/projects/{project_id}",
        json={"name": renamed_name},
        headers=A_OTHER_ADMIN,
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["attribution"]["lastModifiedBy"]["userId"] == USER_A_ADMIN

    skill_label = f"http skill {uuid4().hex[:8]}"
    skill_created = client.post(
        "/api/v1/skills/",
        json={"label": skill_label, "instructions": "Initial skill"},
        headers=A,
    )
    assert skill_created.status_code == 201, skill_created.text
    skill = skill_created.json()
    assert skill["attribution"]["createdBy"]["userId"] == USER_A
    assert set(skill["capabilities"]) == {"canEdit", "canDelete", "canDisable"}
    denied = client.put(
        f"/api/v1/skills/{skill['id']}",
        json={"instructions": "Peer overwrite"},
        headers=A_PEER,
    )
    assert denied.status_code == 403, denied.text
    skill_updated = client.put(
        f"/api/v1/skills/{skill['id']}",
        json={"enabled": False},
        headers=A_OTHER_ADMIN,
    )
    assert skill_updated.status_code == 200, skill_updated.text
    assert (
        skill_updated.json()["attribution"]["lastModifiedBy"]["userId"] == USER_A_ADMIN
    )

    memory_created = client.put(
        "/api/v1/memory/",
        headers=A,
        json={
            "scope": "project",
            "category": "rules",
            "content": "Creator memory",
            "project_id": project_id,
        },
    )
    assert memory_created.status_code == 200, memory_created.text
    assert memory_created.json()["attribution"]["createdBy"]["userId"] == USER_A
    denied = client.put(
        "/api/v1/memory/",
        headers=A_PEER,
        json={
            "scope": "project",
            "category": "rules",
            "content": "Peer memory",
            "project_id": project_id,
        },
    )
    assert denied.status_code == 403, denied.text
    memory_updated = client.put(
        "/api/v1/memory/",
        headers=A_OTHER_ADMIN,
        json={
            "scope": "project",
            "category": "rules",
            "content": "Admin memory",
            "project_id": project_id,
        },
    )
    assert memory_updated.status_code == 200, memory_updated.text
    assert memory_updated.json()["capabilities"] == {
        "canEdit": True,
        "canDelete": True,
    }

    instructions_url = f"/api/v1/projects/{renamed_name}/files/.anton/anton.md"
    instructions_created = client.put(
        instructions_url,
        json={"content": "Creator instructions"},
        headers=A,
    )
    assert instructions_created.status_code == 200, instructions_created.text
    assert instructions_created.json()["attribution"]["createdBy"]["userId"] == USER_A
    denied = client.put(
        instructions_url,
        json={"content": "Peer instructions"},
        headers=A_PEER,
    )
    assert denied.status_code == 403, denied.text
    instructions_updated = client.put(
        instructions_url,
        json={"content": "Admin instructions"},
        headers=A_OTHER_ADMIN,
    )
    assert instructions_updated.status_code == 200, instructions_updated.text
    assert (
        instructions_updated.json()["attribution"]["lastModifiedBy"]["userId"]
        == USER_A_ADMIN
    )
    assert instructions_updated.json()["capabilities"] == {
        "canEdit": True,
        "canDelete": True,
    }

    general = next(
        row
        for row in client.get("/api/v1/projects/", headers=A_PEER).json()
        if row["name"] == "general"
    )
    general_selected = client.patch(
        f"/api/v1/projects/{general['id']}",
        json={"name": "general", "isActive": True},
        headers=A_PEER,
    )
    assert general_selected.status_code == 200, general_selected.text


# -- conversations --------------------------------------------------------------

def test_conversations_isolated(client):
    cid = _conversation(client, A, topic="a-secret")

    b_list = client.get("/api/v1/conversations/?project=all", headers=B)
    assert b_list.status_code == 200, b_list.text
    payload = b_list.json()
    assert "conversations" in payload
    assert cid not in [c["id"] for c in payload["conversations"]]

    _same_as_missing(client, "GET", lambda i: f"/api/v1/conversations/{i}", cid)
    _same_as_missing(client, "GET", lambda i: f"/api/v1/conversations/{i}/items", cid)
    _same_as_missing(client, "DELETE", lambda i: f"/api/v1/conversations/{i}", cid)

    a_read = client.get(f"/api/v1/conversations/{cid}", headers=A)
    assert a_read.status_code == 200, a_read.text
    assert a_read.json()["title"] == "a-secret"  # unchanged after B's probes


def test_conversation_create_under_foreign_project(client):
    a_pid = _project(client, A)

    _same_as_missing_body(
        client, "POST", "/api/v1/conversations/",
        lambda i: {"topic": "intruder", "projectId": i}, a_pid,
    )

    # Nothing was created under A's project.
    a_convs = client.get(f"/api/v1/conversations/?project_id={a_pid}", headers=A).json()
    assert "intruder" not in [c["title"] for c in a_convs["conversations"]]


def test_conversation_move_to_foreign_project(client):
    a_pid = _project(client, A)
    b_cid = _conversation(client, B, topic="b-own")
    b_home = client.get(f"/api/v1/conversations/{b_cid}", headers=B).json()["projectId"]

    _same_as_missing_body(
        client, "POST", f"/api/v1/conversations/{b_cid}/move",
        lambda i: {"projectId": i}, a_pid,
    )

    # B's conversation never moved; A's project gained nothing.
    assert client.get(f"/api/v1/conversations/{b_cid}", headers=B).json()["projectId"] == b_home
    a_convs = client.get(f"/api/v1/conversations/?project_id={a_pid}", headers=A).json()
    assert b_cid not in [c["id"] for c in a_convs["conversations"]]


# -- schedules -------------------------------------------------------------------

def _schedule_body(project_id: str, title="a-sched") -> dict:
    return {"title": title, "prompt": "do it", "cadence": "daily",
            "nextRunAt": "2027-01-01T00:00:00Z", "projectId": project_id}


def test_schedules_isolated(client):
    created = client.post("/api/v1/schedules/", headers=A, json=_schedule_body(_project(client, A)))
    assert created.status_code == 201, created.text
    sid = created.json()["id"]

    b_list = client.get("/api/v1/schedules/", headers=B)
    assert b_list.status_code == 200, b_list.text
    payload = b_list.json()
    assert "schedules" in payload
    assert sid not in [s["id"] for s in payload["schedules"]]

    _same_as_missing(client, "GET", lambda i: f"/api/v1/schedules/{i}", sid)
    _same_as_missing(client, "DELETE", lambda i: f"/api/v1/schedules/{i}", sid)
    _same_as_missing(client, "POST", lambda i: f"/api/v1/schedules/{i}/pause", sid)

    a_read = client.get(f"/api/v1/schedules/{sid}", headers=A)
    assert a_read.status_code == 200, a_read.text
    assert a_read.json()["enabled"] is True  # B's pause attempt changed nothing


def test_schedule_create_under_foreign_project(client):
    # Foreign-parent write: found (and fixed) create_schedule accepting any
    # project_id unanchored - B could attach schedules to A's project.
    a_pid = _project(client, A)

    _same_as_missing_body(
        client, "POST", "/api/v1/schedules/",
        lambda i: _schedule_body(i, title="intruder-sched"), a_pid,
    )

    a_scheds = client.get(f"/api/v1/schedules/?project_id={a_pid}", headers=A).json()
    assert "intruder-sched" not in [s["title"] for s in a_scheds["schedules"]]


def test_schedule_repoint_to_foreign_project(client):
    a_pid = _project(client, A)
    b_sched = client.post("/api/v1/schedules/", headers=B, json=_schedule_body(_project(client, B)))
    assert b_sched.status_code == 201, b_sched.text
    b_sid = b_sched.json()["id"]
    b_home = b_sched.json()["projectId"]

    _same_as_missing_body(
        client, "PUT", f"/api/v1/schedules/{b_sid}",
        lambda i: {"projectId": i}, a_pid,
    )

    assert client.get(f"/api/v1/schedules/{b_sid}", headers=B).json()["projectId"] == b_home


# -- files -----------------------------------------------------------------------

def test_files_isolated(client):
    created = client.post(
        "/api/v1/files/", headers=A,
        files={"file": ("a.txt", b"org-a-bytes", "text/plain")},
        data={"purpose": "assistants"},
    )
    assert created.status_code == 201, created.text
    fid = created.json()["id"]

    b_list = client.get("/api/v1/files/", headers=B)
    assert b_list.status_code == 200, b_list.text
    payload = b_list.json()
    assert "data" in payload
    assert fid not in [f["id"] for f in payload["data"]]

    _same_as_missing(client, "GET", lambda i: f"/api/v1/files/{i}", fid)
    _same_as_missing(client, "GET", lambda i: f"/api/v1/files/{i}/content", fid)
    _same_as_missing(client, "DELETE", lambda i: f"/api/v1/files/{i}", fid)

    a_content = client.get(f"/api/v1/files/{fid}/content", headers=A)
    assert a_content.status_code == 200, a_content.text
    assert a_content.content == b"org-a-bytes"  # bytes unchanged after B's probes


# -- memory (two-org check for the Jul-23 scoping fix) -----------------------------

def test_memory_isolated_across_orgs(client):
    pid = _project(client, A)

    put = client.put("/api/v1/memory/", headers=A, json={
        "scope": "project", "category": "rules", "content": "org-a secret memory",
        "project_id": pid,
    })
    assert put.status_code == 200, put.text

    # B reading A's project memory: byte-equivalent to a nonexistent project.
    missing_id = str(uuid4())
    cross = client.get(f"/api/v1/memory/?project_id={pid}", headers=B)
    missing = client.get(f"/api/v1/memory/?project_id={missing_id}", headers=B)
    _assert_same_response(cross, missing, pid, missing_id, status=cross.status_code)
    assert b"org-a secret memory" not in cross.content

    # B writing to A's project memory: fails like a nonexistent project
    # (endpoint maps the service ValueError to 400 for both).
    _same_as_missing_body(
        client, "PUT", "/api/v1/memory/",
        lambda i: {"scope": "project", "category": "rules",
                   "content": "b-overwrite", "project_id": i},
        pid, status=400,
    )

    # A's memory is unchanged after B's read+write attempts.
    a_read = client.get(f"/api/v1/memory/?project_id={pid}", headers=A)
    assert a_read.status_code == 200, a_read.text
    rules = [m for m in a_read.json() if m["category"] == "rules" and m["scope"] == "project"]
    assert [m["content"].strip() for m in rules] == ["org-a secret memory"]  # store appends \n


# -- streaming surfaces --------------------------------------------------------------

class _LiveHandle:
    """Duck-typed RunHandle registered in the real process-global registry.

    A real RunHandle would need an asyncio task, but the lifespan-less
    TestClient runs each request in its own event loop, so a task created here
    could never be awaited by the endpoint (cross-loop). The registry state is
    real; only the task is stubbed. Registry authz logic itself is covered in
    test_streaming_tenancy.
    """

    class _Buffer:
        latest_seq = 3

    def __init__(self, conversation_id: str, org_id: str, user_id: str) -> None:
        self.conversation_id = conversation_id
        self.org_id = org_id
        self.user_id = user_id
        self.turn_id = 1
        self.buffer = self._Buffer()
        self.cancelled = False

    @property
    def is_running(self) -> bool:
        return not self.cancelled

    async def cancel(self) -> bool:
        self.cancelled = True
        return True


def test_streaming_surfaces_isolated(client):
    from cowork.streaming import registry

    cid = _conversation(client, A, topic="live")
    handle = _LiveHandle(cid, ORG_A, USER_A)
    registry._by_cid[cid] = handle
    try:
        # B cancelling A's LIVE run: 404, identical to an unknown id, and the
        # run stays active.
        missing_id = str(uuid4())
        cross = client.post("/api/v1/responses/cancel", headers=B, json={"conversation_id": cid})
        missing = client.post("/api/v1/responses/cancel", headers=B,
                              json={"conversation_id": missing_id})
        _assert_same_response(cross, missing, cid, missing_id)
        assert not handle.cancelled and handle.is_running

        # B's in-flight list never shows A's run.
        b_res = client.get("/api/v1/responses/in-flight-list", headers=B)
        assert b_res.status_code == 200, b_res.text
        b_payload = b_res.json()
        assert "in_flight" in b_payload
        assert cid not in {r["conversation_id"] for r in b_payload["in_flight"]}

        # A sees and cancels its own run through the same real stack.
        a_payload = client.get("/api/v1/responses/in-flight-list", headers=A).json()
        assert cid in {r["conversation_id"] for r in a_payload["in_flight"]}
        cancelled = client.post("/api/v1/responses/cancel", headers=A, json={"conversation_id": cid})
        assert cancelled.status_code == 200, cancelled.text
        assert handle.cancelled
    finally:
        registry._by_cid.pop(cid, None)


def _setting(client, headers, key):
    rows = client.get("/api/v1/settings/", headers=headers).json()
    return next(s for s in rows if s["key"] == key)


# Org-admin variants: org-key settings writes require manage-organization.
A_ADMIN = {**A, "X-User-Roles": "manage-organization"}
B_ADMIN = {**B, "X-User-Roles": "manage-organization"}


def test_settings_writes_are_org_isolated(client):
    # org A's admin stores a provider credential
    assert client.put("/api/v1/settings/openai_api_key", json={"value": "sk-A"}, headers=A_ADMIN).status_code == 200
    # org B must NOT resolve it (reads fall back to global rows only, never A's org row)
    assert _setting(client, B, "openai_api_key")["is_set"] is False
    assert _setting(client, A, "openai_api_key")["is_set"] is True

    # org B setting the same key writes its OWN row, doesn't touch A's
    assert client.put("/api/v1/settings/openai_api_key", json={"value": "sk-B"}, headers=B_ADMIN).status_code == 200
    assert _setting(client, A, "openai_api_key")["is_set"] is True
    assert _setting(client, B, "openai_api_key")["is_set"] is True

    # logout is a no-op in org mode: one member signing out must not wipe the
    # keys the whole org runs on.
    res = client.post("/api/v1/settings/logout", headers=A)
    assert res.status_code == 200 and res.json()["deleted"] == []
    assert _setting(client, A, "openai_api_key")["is_set"] is True
    assert _setting(client, B, "openai_api_key")["is_set"] is True


def test_org_settings_writes_require_admin_role(client):
    # a plain member (no manage-organization role) cannot change org config...
    assert client.put("/api/v1/settings/minds_url", json={"value": "http://x"}, headers=A).status_code == 403
    assert client.put("/api/v1/settings/", json={"values": {"minds_url": "http://x"}}, headers=A).status_code == 403
    assert client.delete("/api/v1/settings/openai_api_key", headers=A).status_code == 403
    # ...but personal preferences remain open to every member
    assert client.put("/api/v1/settings/tone", json={"value": "casual"}, headers=A).status_code == 200
    # and an admin can write org config
    assert client.put("/api/v1/settings/minds_url", json={"value": "http://x"}, headers=A_ADMIN).status_code == 200
    # skipped sentinel values ("***"/None) don't trip the gate for members
    assert client.put(
        "/api/v1/settings/", json={"values": {"openai_api_key": "***", "greeting": "hi"}}, headers=A
    ).status_code == 200
