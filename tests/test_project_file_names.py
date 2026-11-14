"""Which names a project's ordinary files carry, and what follows from them:
cloud routing and the cloud copy, a recipient's view of a shared project,
the routes that add a file to a project's root, and the project file a
cloud turn's figure names."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from _fakes import stored_file
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope

ORG = "7a1c2d3e-4f5a-4b6c-8d7e-9f0a1b2c3d82"
USER = "8b2d3e4f-5a6b-4c7d-9e8f-0a1b2c3d4e82"
SCOPE = TenantScope(org_mode=True, org_id=ORG, user_id=USER)
MEMBER = {"X-User-Id": USER, "X-Organization-Id": ORG}

posix_only = pytest.mark.skipif(os.name == "nt", reason="org mode runs on POSIX only")


def _write(root: Path, rel: str, content: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def project(org_deployment) -> Path:
    """A project folder of the organization, named for this test alone."""
    from cowork.services.projects import projects_root

    folder = projects_root(scope=SCOPE) / f"names-{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    return folder


def _copied(project: Path, conv) -> dict[str, str]:
    from cowork.services.project_file_copy import NOT_COPIED_FILENAME

    root = project / "conversations" / str(conv) / "project-files"
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != NOT_COPIED_FILENAME
    }


# ── The server's folder names below the project root ────────────────────


@posix_only
@pytest.mark.parametrize(
    "rel", ["docs/skills/guide.md", "Research/Memory/porter.md", "archive/conversations/2024.md", "memory/notes.md"],
)
def test_a_users_folder_named_like_the_servers_holds_files_a_cloud_turn_gets(project, rel):
    """The server keeps conversations/ and skills/ at a project's root only,
    and project memory in .anton/memory, so a folder of the user's with one
    of those names elsewhere holds ordinary files: a cloud turn routes to
    the agent for the file, and its copy holds it."""
    from cowork.services.files import project_holds_ordinary_files
    from cowork.services.project_file_copy import ProjectFileCopyLimits, ProjectFilesOutcome, stage_project_files

    _write(project, rel, "the porter kept the keys")
    conv = uuid4()

    holds = project_holds_ordinary_files(project)
    staged = stage_project_files(
        project, conv, scope=SCOPE,
        limits=ProjectFileCopyLimits(max_file_bytes=1000, max_files=10, max_total_bytes=10_000, seconds=30.0),
    )

    assert holds is True
    assert (staged.outcome, staged.files) == (ProjectFilesOutcome.STAGED, 1)
    assert _copied(project, conv) == {rel: "the porter kept the keys"}


@pytest.mark.parametrize(
    ("parts", "ordinary"),
    [
        (("conversations", "c", "notes.md"), False),
        (("Skills", "s", "SKILL.md"), False),
        (("docs", "skills", "guide.md"), True),
        (("archive", "Conversations", "2024.md"), True),
        (("memory", "notes.md"), True),
        (("src", "node_modules", "x.js"), False),
        (("data", "__pycache__", "m.pyc"), False),
        (("docs", "desktop.ini"), False),
    ],
)
def test_the_servers_folders_are_not_ordinary_at_the_root_alone(parts, ordinary):
    from cowork.services.project_file_view import ordinary_path

    assert ordinary_path(parts) is ordinary


# ── A recipient's view of a shared project ──────────────────────────────


@contextmanager
def _shared_project_at(monkeypatch, root: Path):
    """SharedProjectFiles reading ``root`` as the shared project's folder."""
    from cowork.common.paths import pinned_dir
    from cowork.services import shared_projects

    @contextmanager
    def directory(self, project_id):
        with pinned_dir(root) as pinned:
            yield pinned

    monkeypatch.setattr(shared_projects.SharedProjectFiles, "directory", directory)
    yield shared_projects.SharedProjectFiles(None)


def test_a_recipients_listing_never_walks_into_a_folder_it_hides(tmp_path, monkeypatch):
    """A folder whose name a recipient can't see is not walked into, so its
    files spend none of the listing's entry budget and a file after it is
    still listed."""
    from cowork.services import shared_projects

    root = tmp_path / "project"
    for index in range(20):
        _write(root, f"Q3: raw/{index}.csv")
    _write(root, "zz-notes.md", "notes")
    monkeypatch.setattr(shared_projects, "MAX_LIST_ENTRIES", 10)

    with _shared_project_at(monkeypatch, root) as files:
        listed = files.list_files(uuid4())

    assert [file.path for file in listed.files] == ["zz-notes.md"]
    assert listed.truncated is False


def test_a_recipients_view_still_hides_the_servers_folder_names_at_any_depth(tmp_path, monkeypatch):
    """The copy's rule keeps the server's folder names at the project root
    only and memory nowhere; a recipient's view keeps hiding all three
    everywhere, as it did before the copy shared its rule."""
    from cowork.services.shared_projects import SharedProjectFiles

    root = tmp_path / "project"
    for rel in ("docs/skills/guide.md", "Research/Memory/porter.md", "archive/conversations/2024.md"):
        _write(root, rel)
    _write(root, "report.txt")

    with _shared_project_at(monkeypatch, root) as files:
        listed = files.list_files(uuid4())

    assert [file.path for file in listed.files] == ["report.txt"]
    with pytest.raises(FileNotFoundError):
        SharedProjectFiles(None).read_file(uuid4(), "docs/skills/guide.md", limit=100)


# ── Adding a file to a project's root ───────────────────────────────────


@posix_only
def test_an_upload_a_cloud_turn_would_never_read_is_refused(org_deployment):
    """On cloud, an upload whose name the copy always leaves out (a file
    the OS leaves, a cache folder's name or a server folder's name at the
    root) is refused with a plain reason, and nothing is written. A memory
    file is ordinary, so it lands."""
    from cowork.server import create_app
    from cowork.services.project_file_view import NOT_A_PROJECT_FILE_NAME

    client = TestClient(create_app())
    created = client.post("/api/v1/projects/", json={"name": f"upload-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert created.status_code == 201, created.text
    name, project = created.json()["name"], Path(created.json()["path"])
    names = ["desktop.ini", "Thumbs.db", "node_modules", "Skills", "Memory", "notes.md"]

    uploaded = client.post(
        f"/api/v1/projects/{name}/files/upload",
        files=[("files", (upload, b"x", "application/octet-stream")) for upload in names],
        headers=MEMBER,
    )

    assert uploaded.status_code == 200, uploaded.text
    results = {result["name"]: result for result in uploaded.json()["results"]}
    for refused in ("desktop.ini", "Thumbs.db", "node_modules", "Skills"):
        assert (results[refused]["ok"], results[refused]["error"]) == (False, NOT_A_PROJECT_FILE_NAME)
        assert not (project / refused).exists()
    assert results["Memory"]["ok"] and results["notes.md"]["ok"]


@pytest.fixture
def org_session(tmp_path, monkeypatch):
    """A member's session in an organization, on a database of the test's
    own."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield ScopedSession(session, SCOPE)
    get_app_settings.cache_clear()


def _attachment(scoped: ScopedSession, filename: str):
    from cowork.services.files import attachment_purpose

    return stored_file(
        scoped=scoped, filename=filename, content_type="text/csv", data=b"a,b\n1,2\n",
        purpose=attachment_purpose(str(uuid4())),
    )


@posix_only
def test_moving_a_hidden_attachment_to_the_project_drops_its_leading_dots(org_session):
    """A dot name would be a project file the listing hides and no cloud
    turn reads, so the move takes the leading dots off, and the next cloud
    turn's copy holds the file."""
    from cowork.api.v1.endpoints.compat import stubs
    from cowork.services.files import FileService
    from cowork.services.project_file_copy import ProjectFileCopyLimits, stage_project_files
    from cowork.services.projects import ProjectService

    project = ProjectService(org_session).create_project(f"move-{uuid4().hex[:8]}")
    attachment = _attachment(org_session, ".budget.csv")

    moved = stubs._move_attachment_to_project(org_session, project_name=project.name, attachment_id=attachment.id)

    assert moved.project_path == "budget.csv"
    assert (Path(project.path) / "budget.csv").read_bytes() == b"a,b\n1,2\n"
    with pytest.raises(ValueError):
        FileService(org_session).get_file_content(attachment.id)
    conv = uuid4()
    stage_project_files(
        project.path, conv, scope=SCOPE,
        limits=ProjectFileCopyLimits(max_file_bytes=1000, max_files=10, max_total_bytes=10_000, seconds=30.0),
    )
    assert _copied(Path(project.path), conv) == {"budget.csv": "a,b\n1,2\n"}


@posix_only
@pytest.mark.parametrize("filename", ["Thumbs.db", ".desktop.ini", "node_modules", "..skills"])
def test_moving_an_attachment_no_cloud_turn_would_read_is_refused_and_keeps_it(org_session, filename):
    """A name that is still not ordinary once its leading dots are off is
    refused with a plain reason before anything is copied, so the
    attachment stays."""
    from cowork.api.v1.endpoints.compat import stubs
    from cowork.services.files import FileService
    from cowork.services.project_file_view import NOT_A_PROJECT_FILE_NAME
    from cowork.services.projects import ProjectService

    project = ProjectService(org_session).create_project(f"refuse-{uuid4().hex[:8]}")
    attachment = _attachment(org_session, filename)

    with pytest.raises(HTTPException) as refused:
        stubs._move_attachment_to_project(org_session, project_name=project.name, attachment_id=attachment.id)

    assert (refused.value.status_code, refused.value.detail) == (409, NOT_A_PROJECT_FILE_NAME)
    assert FileService(org_session).get_file_row(attachment.id).id == attachment.id
    assert [entry.name for entry in Path(project.path).iterdir() if not entry.name.startswith(".")] == []


# ── The project file a cloud turn's figure names ────────────────────────


@pytest.fixture
def pod_project(tmp_path: Path):
    """A project holding q3.csv, and a conversation whose pod reads its copy."""
    root = tmp_path / "projects" / "general"
    conv = uuid4()
    _write(root, "data/q3.csv", "q,sales\n3,10\n")
    _write(root / "conversations" / str(conv) / "project-files", "data/q3.csv", "q,sales\n3,10\n")
    return root, conv


def _in_pod(root: Path, conv):
    from cowork.harnesses.anton_harness.agent_files import AgentFiles

    return AgentFiles.in_cloud_pod(root, conversation_id=conv, workspace_rel_path="projects/general")


def test_a_copied_project_file_names_the_project_file(pod_project):
    """The pod reads the project's files from its copy, which the project's
    listing hides; a figure drawn from one names the project file, by a
    relative or an absolute path, even after the copy is pruned."""
    from cowork.harnesses.anton_harness.agent_files import NamedFile

    root, conv = pod_project
    files = _in_pod(root, conv)

    assert files.named_file("project-files/data/q3.csv") == NamedFile(path="data/q3.csv")
    assert files.named_file("/workspace/project-files/data/q3.csv") == NamedFile(path="data/q3.csv")
    (root / "conversations" / str(conv) / "project-files" / "data" / "q3.csv").unlink()
    assert files.named_file("project-files/data/q3.csv") == NamedFile(path="data/q3.csv")


def test_a_copy_names_no_project_file_that_is_not_an_ordinary_one_there(pod_project, tmp_path):
    """A copy whose project file is gone, a second name for a file, a link
    or a name the copy never makes names nothing, and neither does what a
    copy removed."""
    root, conv = pod_project
    copy = root / "conversations" / str(conv) / "project-files"
    files = _in_pod(root, conv)
    _write(copy, "gone.csv")
    _write(copy, "twice.csv")
    os.link(_write(tmp_path, "elsewhere.csv"), root / "twice.csv")
    _write(copy, "linked.csv")
    (root / "linked.csv").symlink_to(root / "data" / "q3.csv")
    _write(copy, ".not-copied.json", "{}")
    _write(root / "conversations" / str(conv) / ".project-files-removed", "a1/q3.csv")

    assert files.named_file("project-files/gone.csv") is None
    assert files.named_file("project-files/twice.csv") is None
    assert files.named_file("project-files/linked.csv") is None
    assert files.named_file("project-files/.not-copied.json") is None
    assert files.named_file(".project-files-removed/a1/q3.csv") is None
