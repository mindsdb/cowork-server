"""Project file routes: what one member of an org may fetch from another's chat.

A project is shared by the whole organization, but `conversations/<id>/` under it
is one member's private workspace. `read`, `write` and `delete` have always run
`_require_workspace_access`; `files-raw`, `preview-mount-file` and `preview-asset`
did not, so a member could download or preview a co-member's conversation bytes.

These go through the real HTTP stack rather than calling the handlers, because
two of the three defects were about what a *route* forgot to call, and because
the preview token has to survive a round trip to be worth testing at all.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
USER_A = "11111111-1111-4111-8111-111111111111"
# The axis the org filter does not cover: a second member of the SAME org.
USER_A2 = "33333333-3333-4333-8333-333333333333"
USER_B = "22222222-2222-4222-8222-222222222222"

A = {"X-User-Id": USER_A, "X-Organization-Id": ORG_A}
A2 = {"X-User-Id": USER_A2, "X-Organization-Id": ORG_A}
B = {"X-User-Id": USER_B, "X-Organization-Id": ORG_B}

PROJECT = "files-tenancy"
NOT_FOUND = {"detail": "File not found"}


@pytest.fixture(scope="module")
def client():
    saved = {k: os.environ.get(k) for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")}
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
    """One shared project holding one private conversation workspace.

    `path` is set explicitly rather than left to the storage root, so this
    fixture exercises the routes rather than the path-resolution rules that
    `test_projects_tenancy.py` already covers.
    """
    root = tmp_path_factory.mktemp("files-tenancy") / PROJECT
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = Project(id=uuid.uuid4(), name=PROJECT, path=str(root), org_id=ORG_A)
        conversation = Conversation(
            id=uuid.uuid4(),
            topic="private chat",
            project_id=project.id,
            org_id=ORG_A,
            created_by=USER_A,
        )
        session.add(project)
        session.add(conversation)
        session.commit()
        conversation_id = conversation.id

    workspace = root / "conversations" / str(conversation_id)
    workspace.mkdir(parents=True)
    (workspace / "report.txt").write_text("A's private notes")
    (workspace / "page.html").write_text("<html><body>private</body></html>")
    (workspace / "sidecar.css").write_text("body{color:red}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "shared.txt").write_text("everyone may read this")

    yield {"root": root, "conversation_id": conversation_id, "workspace": workspace}


def _raw(client, headers, rel):
    return client.get(f"/api/v1/projects/{PROJECT}/files-raw/{rel}", headers=headers)


def _mount(client, headers, rel):
    return client.post(
        "/api/v1/projects/preview-mount-file",
        json={"name": PROJECT, "path": rel},
        headers=headers,
    )


# ── files-raw ─────────────────────────────────────────────────────────────

def test_files_raw_serves_the_owner_their_own_conversation_file(client, tree):
    res = _raw(client, A, f"conversations/{tree['conversation_id']}/report.txt")

    assert res.status_code == 200
    assert res.text == "A's private notes"


def test_files_raw_refuses_another_members_conversation_file(client, tree):
    res = _raw(client, A2, f"conversations/{tree['conversation_id']}/report.txt")

    assert res.status_code == 404
    # Byte-identical to a genuine miss on the same route, so the refusal does not
    # confirm that the file is there.
    assert res.json() == NOT_FOUND
    assert res.json() == _raw(client, A2, "conversations/%s/nope.txt" % uuid.uuid4()).json()


def test_files_raw_keeps_shared_project_files_readable_by_every_member(client, tree):
    """The gate must not overshoot: everything outside `conversations/` is a
    shared project file and stays readable by the whole org."""
    res = _raw(client, A2, "shared.txt")

    assert res.status_code == 200
    assert res.text == "everyone may read this"


# ── preview-mount-file ────────────────────────────────────────────────────

def test_preview_mount_refuses_another_members_conversation_file(client, tree):
    res = _mount(client, A2, f"conversations/{tree['conversation_id']}/page.html")

    assert res.status_code == 404
    assert res.json() == NOT_FOUND


def test_preview_mount_and_asset_work_for_the_owner(client, tree):
    mounted = _mount(client, A, f"conversations/{tree['conversation_id']}/page.html")
    assert mounted.status_code == 200
    token = mounted.json()["token"]

    res = client.get(f"/api/v1/projects/preview-asset/{token}/page.html", headers=A)
    assert res.status_code == 200
    assert "private" in res.text

    # The mount is a directory, so sibling assets the page pulls resolve too.
    sidecar = client.get(f"/api/v1/projects/preview-asset/{token}/sidecar.css", headers=A)
    assert sidecar.status_code == 200


# ── the preview token ─────────────────────────────────────────────────────

def test_the_token_is_random_rather_than_derived_from_the_path(client, tree):
    """It used to be `sha256(parent)[:16]`, which anyone who could guess the
    storage layout could compute without ever calling the mount route. Two
    mounts of the same file returning the same string is that bug."""
    rel = f"conversations/{tree['conversation_id']}/page.html"

    first = _mount(client, A, rel).json()["token"]
    second = _mount(client, A, rel).json()["token"]

    assert first != second
    assert len(first) >= 32


def test_a_captured_token_is_useless_to_another_member(client, tree):
    token = _mount(client, A, f"conversations/{tree['conversation_id']}/page.html").json()["token"]

    res = client.get(f"/api/v1/projects/preview-asset/{token}/page.html", headers=A2)

    assert res.status_code == 404
    assert client.get(f"/api/v1/projects/preview-asset/{token}/page.html", headers=A).status_code == 200


def test_a_captured_token_is_useless_in_another_org(client, tree):
    """The token was the one capability in the system that crossed the tenant
    wall: the route took no session at all, so whoever held the string read the
    bytes.

    The SAME user id with the other org's id, deliberately, because
    `readable_by` is a conjunction and changing both axes at once lets the user
    comparison alone satisfy the assertion. One human holds one `user_id` across
    every org they belong to and the switcher changes only the org header, so
    this is the request the org half actually has to refuse. Same discipline as
    `test_sources_for_scope_covers_only_own_org` in test_artifact_roots.py.
    """
    token = _mount(client, A, f"conversations/{tree['conversation_id']}/page.html").json()["token"]

    same_user_other_org = {"X-User-Id": USER_A, "X-Organization-Id": ORG_B}
    res = client.get(
        f"/api/v1/projects/preview-asset/{token}/page.html", headers=same_user_other_org
    )

    assert res.status_code == 404
    # And the org half is what refused it, not the absence of the file.
    assert client.get(f"/api/v1/projects/preview-asset/{token}/page.html", headers=A).status_code == 200


def test_a_captured_token_is_useless_to_a_stranger(client, tree):
    """Different user AND different org, the shape a leaked URL actually takes."""
    token = _mount(client, A, f"conversations/{tree['conversation_id']}/page.html").json()["token"]

    assert client.get(f"/api/v1/projects/preview-asset/{token}/page.html", headers=B).status_code == 404


def test_a_token_dies_when_its_ttl_runs_out(client, tree, monkeypatch):
    from cowork.api.v1.endpoints import project_files

    monkeypatch.setattr(project_files, "PREVIEW_TOKEN_TTL_SECONDS", 0)
    token = _mount(client, A, f"conversations/{tree['conversation_id']}/page.html").json()["token"]

    res = client.get(f"/api/v1/projects/preview-asset/{token}/page.html", headers=A)

    assert res.status_code == 404


def test_minting_drops_records_that_have_expired(client, tree, monkeypatch):
    """Nothing else removes an entry, so without this the registry grows for the
    life of the process."""
    from cowork.api.v1.endpoints import project_files

    monkeypatch.setattr(project_files, "PREVIEW_TOKEN_TTL_SECONDS", 0)
    stale = _mount(client, A, f"conversations/{tree['conversation_id']}/page.html").json()["token"]
    assert stale in project_files._PROJECT_PREVIEW_MOUNTS

    monkeypatch.setattr(project_files, "PREVIEW_TOKEN_TTL_SECONDS", 30 * 60)
    _mount(client, A, f"conversations/{tree['conversation_id']}/page.html")

    assert stale not in project_files._PROJECT_PREVIEW_MOUNTS


def test_the_registry_is_capped_so_live_mounts_cannot_grow_without_bound(client, tree, monkeypatch):
    """The old token was `sha256(parent)`, so repeated mounts of one directory
    reused one key. A random token per mint inserts every time and only the TTL
    evicts, so a member looping the mount route grows the dict for 30 minutes.
    """
    from cowork.api.v1.endpoints import project_files

    monkeypatch.setattr(project_files, "PREVIEW_MOUNT_LIMIT", 8)
    project_files._PROJECT_PREVIEW_MOUNTS.clear()

    for _ in range(40):
        _mount(client, A, f"conversations/{tree['conversation_id']}/page.html")

    assert len(project_files._PROJECT_PREVIEW_MOUNTS) <= 8


# ── the mounted directory is not a way out of the workspace ───────────────
#
# A mount grants a DIRECTORY. `preview_mount_file` gates the file it was handed,
# so the gate is satisfied by any shared file, and the directory that then gets
# mounted can sit above every member's private workspace. All three of these
# served another member's bytes before the `serves` check existed.

def test_a_mount_at_the_project_root_cannot_read_into_a_workspace(client, tree):
    """A2 may write and preview a shared project file. That must not become a
    read of A's conversation directory."""
    assert client.put(
        f"/api/v1/projects/{PROJECT}/files/shared-page.html",
        json={"content": "<html>mine</html>"},
        headers=A2,
    ).status_code == 200

    token = _mount(client, A2, "shared-page.html").json()["token"]

    # The mount itself still works on what it was taken for.
    assert client.get(f"/api/v1/projects/preview-asset/{token}/shared-page.html", headers=A2).status_code == 200

    stolen = client.get(
        f"/api/v1/projects/preview-asset/{token}/conversations/{tree['conversation_id']}/report.txt",
        headers=A2,
    )
    assert stolen.status_code == 404
    assert "private notes" not in stolen.text


def test_a_mount_at_the_project_root_cannot_read_a_workspaces_artifacts(client, tree):
    """Same door, the other P0 behind it: live artifacts live under
    `conversations/<id>/.anton/artifacts`, so this route bypassed the roots
    resolver's owner filter entirely."""
    art = tree["workspace"] / ".anton" / "artifacts" / "deck"
    art.mkdir(parents=True, exist_ok=True)
    (art / "index.html").write_text("<html>A's private artifact</html>")
    client.put(
        f"/api/v1/projects/{PROJECT}/files/shared-page.html",
        json={"content": "<html>mine</html>"},
        headers=A2,
    )

    token = _mount(client, A2, "shared-page.html").json()["token"]

    res = client.get(
        f"/api/v1/projects/preview-asset/{token}/"
        f"conversations/{tree['conversation_id']}/.anton/artifacts/deck/index.html",
        headers=A2,
    )
    assert res.status_code == 404
    assert "private artifact" not in res.text


def test_a_mount_on_conversations_itself_cannot_read_a_workspace(client, tree):
    """`_conversation_workspace_ok` treats `conversations/<not-a-uuid>` as a
    shared file, so an .html placed directly under `conversations/` passes the
    gate and mounts the directory every workspace hangs off."""
    assert client.put(
        f"/api/v1/projects/{PROJECT}/files/conversations/door.html",
        json={"content": "<html>door</html>"},
        headers=A2,
    ).status_code == 200

    token = _mount(client, A2, "conversations/door.html").json()["token"]

    stolen = client.get(
        f"/api/v1/projects/preview-asset/{token}/{tree['conversation_id']}/report.txt",
        headers=A2,
    )
    assert stolen.status_code == 404
    assert "private notes" not in stolen.text


def test_a_root_mount_reaches_no_workspace_at_all_not_even_the_owners(client, tree):
    """The refusal is not an ownership decision, because `preview_asset` holds no
    session: a mount reaches its own workspace or none. A owns this conversation
    and is still refused from a root-level mount.

    That costs nothing real. A preview's sub-assets sit beside its entry file, so
    the owner previewing their own workspace mounts from inside it, which
    `test_preview_mount_and_asset_work_for_the_owner` covers. Keeping the rule
    session-free is what stops this route opening a connection per sub-asset.
    """
    client.put(
        f"/api/v1/projects/{PROJECT}/files/shared-page.html",
        json={"content": "<html>mine</html>"},
        headers=A,
    )
    token = _mount(client, A, "shared-page.html").json()["token"]

    res = client.get(
        f"/api/v1/projects/preview-asset/{token}/conversations/{tree['conversation_id']}/report.txt",
        headers=A,
    )
    assert res.status_code == 404


def test_the_pinned_open_refuses_a_symlinked_component(tmp_path):
    """What closes the check-then-open window, tested at the helper because the
    routes cannot show it.

    `_require_workspace_access` decides ownership from a resolved path and the
    sink then re-opens that path by name, which hands the whole chain back to
    the kernel to walk again. A pod mounts its own workspace read-write, so
    between the two it can swap a directory component for a link into another
    member's tree, or another org's.

    A route-level test cannot reproduce that: `_safe_relpath` resolves first, so
    any link planted BEFORE the request is collapsed out of the path and the
    swap that matters is the one landing in a window a test cannot schedule.
    What is testable is the property that makes the window harmless, which is
    that the open refuses a symlinked component instead of following it.
    """
    from cowork.api.v1.endpoints.project_files import _pinned_fd

    base = tmp_path / "project"
    (base / "real").mkdir(parents=True)
    (base / "real" / "notes.md").write_text("mine")
    (base / "victim").mkdir()
    (base / "victim" / "notes.md").write_text("theirs")

    # The honest chain opens.
    with _pinned_fd(base / "real" / "notes.md", base, os.O_RDONLY) as fd:
        assert os.read(fd, 64) == b"mine"

    # Swap the directory component for a link, the way a pod would. The path
    # still lands inside the project, so a containment check would allow it.
    (base / "real").rename(base / "real-moved")
    (base / "real").symlink_to(base / "victim", target_is_directory=True)

    with pytest.raises(OSError):
        with _pinned_fd(base / "real" / "notes.md", base, os.O_RDONLY):
            pass


def test_a_refused_component_becomes_a_404_not_a_500(tmp_path):
    """The OSError has to reach the client as the same 404 a genuine miss
    returns. A 500 on this path would be an existence oracle and a page.
    """
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.project_files import _pinned_regular_file

    base = tmp_path / "project"
    (base / "victim").mkdir(parents=True)
    (base / "victim" / "notes.md").write_text("theirs")
    (base / "real").symlink_to(base / "victim", target_is_directory=True)

    with pytest.raises(HTTPException) as err:
        _pinned_regular_file(base / "real" / "notes.md", base)

    assert err.value.status_code == 404
    assert err.value.detail == "File not found"

    # A directory where a file was asked for takes the same exit.
    with pytest.raises(HTTPException) as err:
        _pinned_regular_file(base / "victim", base)
    assert err.value.status_code == 404


# ── desktop ───────────────────────────────────────────────────────────────

def test_desktop_preview_is_unaffected(tmp_path, monkeypatch):
    """One user, no organization, nothing to compare a mount against. The gate
    and the scope check both have to be inert here or the desktop preview breaks.
    """
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        root = tmp_path / "desktop-proj"
        (root / "conversations" / "c1").mkdir(parents=True)
        (root / "conversations" / "c1" / "page.html").write_text("<html>desktop</html>")

        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as session:
            session.add(
                Project(id=uuid.uuid4(), name="desktop-proj", path=str(root), org_id=None)
            )
            session.commit()

        client = TestClient(create_app())
        mounted = client.post(
            "/api/v1/projects/preview-mount-file",
            json={"name": "desktop-proj", "path": "conversations/c1/page.html"},
        )
        assert mounted.status_code == 200
        token = mounted.json()["token"]

        res = client.get(f"/api/v1/projects/preview-asset/{token}/page.html")
        assert res.status_code == 200
        assert "desktop" in res.text

        raw = client.get("/api/v1/projects/desktop-proj/files-raw/conversations/c1/page.html")
        assert raw.status_code == 200
    finally:
        get_app_settings.cache_clear()


def test_the_project_directory_is_real(tree):
    """Guards the fixture itself: every refusal above would also be produced by a
    project whose files were never written."""
    assert Path(tree["workspace"] / "report.txt").is_file()
