"""Model comparisons: two sides of one task, each an ordinary conversation in a
hidden sandbox project, contained until the user continues with one."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.comparison import ComparisonSide
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services.projects import COMPARISON_SANDBOX_PREFIX, ProjectService


class _StubHarness:
    id = "stub"

    def __init__(self):
        self.calls: list[dict] = []

    def stream_response(self, **kwargs):
        self.calls.append(kwargs)
        return None

    async def formatter(self, stream, model, event_sink):
        event_sink(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "ok"},
        )
        if False:
            yield


@pytest.fixture()
def harness():
    return _StubHarness()


@pytest.fixture()
def client(harness):
    from cowork.server import create_app

    with patch("cowork.handlers.responses.get_harness", return_value=harness):
        yield TestClient(create_app())


def _create(client, *, title="Build a sales dashboard", source_project_id=None, sides=None):
    body = {
        "title": title,
        "sides": sides or [
            {"model": "kimi", "reasoningEffort": None},
            {"model": "qwen", "reasoningEffort": "xhigh"},
        ],
    }
    if source_project_id is not None:
        body["sourceProjectId"] = str(source_project_id)
    return client.post("/api/v1/comparisons/", json=body)


def _project(project_id) -> Project:
    session = get_open_session()
    try:
        return session.get(Project, UUID(str(project_id)))
    finally:
        session.close()


def _conversation(conversation_id) -> Conversation | None:
    session = get_open_session()
    try:
        return session.get(Conversation, UUID(str(conversation_id)))
    finally:
        session.close()


def _handle(*, running: bool):
    """A real RunHandle, so `is_running` has the shape production code sees."""
    from cowork.streaming.registry import RunHandle

    return RunHandle(
        conversation_id="c", turn_id=0, buffer=None,
        task=SimpleNamespace(done=lambda: not running),
    )


def _real_project(client, name: str) -> dict:
    r = client.post("/api/v1/projects/", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


def test_create_makes_two_sides_each_in_its_own_sandbox(client):
    r = _create(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert [s["label"] for s in body["sides"]] == ["a", "b"]
    assert body["verdict"] is None

    a, b = body["sides"]
    assert a["projectId"] != b["projectId"]
    for side, model, effort in ((a, "kimi", None), (b, "qwen", "xhigh")):
        assert side["model"] == model
        assert side["reasoningEffort"] == effort
        assert side["turnCount"] == 0
        project = _project(side["projectId"])
        assert project.name.startswith(COMPARISON_SANDBOX_PREFIX)
        assert Path(project.path).is_dir()
        conversation = _conversation(side["conversationId"])
        assert conversation.project_id == project.id
        assert conversation.model == model
        assert conversation.reasoning_effort == effort
        # Pinned to Anton: the model pick only drives the Anton harness.
        assert conversation.harness == "anton"


def test_both_sandboxes_read_the_same_to_the_agent(client):
    # The harness writes the project's label into the system prompt; the two
    # sides' prompts must differ by nothing but the model.
    body = _create(client, title="Same label").json()
    labels = {_project(s["projectId"]).display_name for s in body["sides"]}
    assert labels == {"Same label"}


def test_create_rejects_anything_but_two_sides(client):
    r = client.post(
        "/api/v1/comparisons/",
        json={"title": "x", "sides": [{"model": "kimi"}]},
    )
    assert r.status_code == 422


def test_create_rejects_a_sandbox_as_the_source(client):
    first = _create(client).json()
    r = _create(client, source_project_id=first["sides"][0]["projectId"])
    assert r.status_code == 400, r.text


def test_create_from_an_unknown_project_is_404(client):
    r = _create(client, source_project_id=uuid4())
    assert r.status_code == 404, r.text


def test_sandboxes_and_their_tasks_are_hidden_from_lists(client):
    body = _create(client, title="hidden-from-lists").json()
    project_ids = {s["projectId"] for s in body["sides"]}
    conversation_ids = {s["conversationId"] for s in body["sides"]}

    listed = {p["id"] for p in client.get("/api/v1/projects/").json()}
    assert not project_ids & listed

    all_tasks = client.get("/api/v1/conversations/", params={"project": "all", "limit": 500}).json()
    assert not conversation_ids & {c["id"] for c in all_tasks["conversations"]}

    results = client.get("/api/v1/search", params={"q": "hidden-from-lists"}).json()["results"]
    assert not conversation_ids & {r["id"] for r in results}

    # Still reachable directly: the Compare screen opens each side.
    side = body["sides"][0]
    assert client.get(f"/api/v1/conversations/{side['conversationId']}").status_code == 200
    in_project = client.get("/api/v1/conversations/", params={"project_id": side["projectId"]}).json()
    assert side["conversationId"] in {c["id"] for c in in_project["conversations"]}


def test_a_project_whose_name_only_resembles_the_prefix_stays_listed(client):
    # `_` is a LIKE wildcard. Unescaped, the cross-project filter would also
    # hide a real project named `acomparison-...`.
    project = _real_project(client, "acomparison-quarterly")
    r = client.post("/api/v1/conversations/", json={"title": "kept", "projectId": project["id"]})
    assert r.status_code == 201, r.text
    all_tasks = client.get("/api/v1/conversations/", params={"project": "all", "limit": 500}).json()
    assert r.json()["id"] in {c["id"] for c in all_tasks["conversations"]}


def test_sandbox_artifacts_are_hidden_from_the_unfiltered_list_only(client):
    body = _create(client).json()
    side = body["sides"][0]
    slug = f"side-dashboard-{uuid4().hex[:8]}"
    folder = Path(_project(side["projectId"]).path) / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<h1>side</h1>")
    (folder / "metadata.json").write_text('{"type": "html-app", "primary": "index.html"}')

    def titles(cards):
        return {card.get("title") for card in cards}

    own = client.get("/api/v1/artifacts/", params={"project_id": side["projectId"]}).json()
    assert slug in titles(own)
    everything = client.get("/api/v1/artifacts/").json()
    assert slug not in titles(everything)


def _seed_source(root: Path) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "sales.csv").write_text("region,total\nwest,10\n")
    (root / "notes.md").write_text("# notes")
    (root / ".anton" / "memory").mkdir(parents=True, exist_ok=True)
    (root / ".anton" / "anton.md").write_text("Always use EUR.")
    (root / ".anton" / "memory" / "rules.md").write_text("- prefer bar charts")
    (root / ".anton" / "artifacts" / "live-report").mkdir(parents=True, exist_ok=True)
    (root / ".anton" / "artifacts" / "live-report" / ".published.json").write_text("{}")
    (root / ".anton" / "episodes").mkdir(parents=True, exist_ok=True)
    (root / ".anton" / "episodes" / "e.jsonl").write_text("{}")


def test_create_from_a_project_copies_its_files_into_both_sides(client, tmp_path):
    source = _real_project(client, "copy-source")
    root = Path(source["path"])
    _seed_source(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("not yours")
    os.symlink(outside, root / "link.txt")

    body = _create(client, source_project_id=source["id"]).json()
    assert body["sourceProjectLabel"] == "copy-source"
    for side in body["sides"]:
        dest = Path(_project(side["projectId"]).path)
        assert (dest / "data" / "sales.csv").read_text() == "region,total\nwest,10\n"
        assert (dest / "notes.md").is_file()
        assert (dest / ".anton" / "anton.md").read_text() == "Always use EUR."
        assert (dest / ".anton" / "memory" / "rules.md").is_file()
        # Artifacts carry publish records and ids; episodes are history.
        assert not (dest / ".anton" / "artifacts" / "live-report").exists()
        assert not (dest / ".anton" / "episodes").exists()
        # A link is never followed.
        assert not (dest / "link.txt").exists()
    # The source is untouched.
    assert (root / ".anton" / "artifacts" / "live-report" / ".published.json").is_file()


@pytest.fixture()
def hosted(tmp_path, monkeypatch):
    """An org-mode session on its own in-memory database, as a hosted member."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine

    from cowork.db.scoped import TenantScope

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    session = ScopedSession(Session(engine), TenantScope(True, org, "alice"))
    try:
        yield SimpleNamespace(session=session, org=org)
    finally:
        session.close()
        get_app_settings.cache_clear()


def test_a_hosted_side_reads_the_source_projects_files_from_its_own_workspace(hosted, tmp_path):
    from cowork.services.comparisons import ComparisonService, SideSpec
    from cowork.services.files import stage_project_instructions

    source_root = tmp_path / "source"
    _seed_source(source_root)
    source = Project(name="source", path=str(source_root))
    hosted.session.add(source)
    hosted.session.commit()

    comparison = ComparisonService(hosted.session).create_comparison(
        title="t", sides=[SideSpec("kimi"), SideSpec("qwen")], source_project_id=source.id
    )

    for side in comparison.sides:
        sandbox = Path(hosted.session.get(Project, side.project_id).path)
        # A hosted turn sees only this folder.
        workspace = sandbox / "conversations" / str(side.conversation_id)
        assert (workspace / "data" / "sales.csv").read_text() == "region,total\nwest,10\n"
        assert (workspace / "notes.md").is_file()
        assert not (sandbox / "data").exists()
        # Instructions stay at the root, where each turn stages them from.
        assert stage_project_instructions(sandbox, side.conversation_id)
        assert (workspace / ".anton" / "anton.md").read_text() == "Always use EUR."
        assert (sandbox / ".anton" / "memory" / "rules.md").is_file()
        assert not (workspace / ".anton" / "memory").exists()
    # A hosted carry moves the workspace whole, so nothing is hashed or kept.
    assert comparison.copied_files is None


def test_a_desktop_comparison_keeps_one_manifest_and_the_history_list_does_not_load_it(client):
    import sqlalchemy as sa

    from cowork.models.comparison import Comparison
    from cowork.services.comparisons import ComparisonService

    source = _real_project(client, "manifest-source")
    _seed_source(Path(source["path"]))
    body = _create(client, source_project_id=source["id"]).json()

    session = get_open_session()
    try:
        comparison = session.get(Comparison, UUID(body["id"]))
        assert set(comparison.copied_files) == {"data/sales.csv", "notes.md"}
        session.expunge_all()
        listed = [c for c in ComparisonService(ScopedSession(session, LOCAL_SCOPE)).list_comparisons(limit=200) if c.id == comparison.id]
        assert "copied_files" in sa.inspect(listed[0]).unloaded
    finally:
        session.close()


def test_copy_skips_member_workspaces_only_on_a_hosted_deployment(tmp_path):
    from cowork.services.comparisons import copy_project_files

    src = tmp_path / "src"
    (src / "conversations" / "someone-else").mkdir(parents=True)
    (src / "conversations" / "someone-else" / "private.txt").write_text("private")
    for org_mode, expected in ((True, False), (False, True)):
        dest = tmp_path / f"dest-{org_mode}"
        dest.mkdir()
        copy_project_files(src, dest, org_mode=org_mode)
        assert (dest / "conversations" / "someone-else" / "private.txt").exists() is expected


def test_copy_does_not_descend_through_a_linked_directory(tmp_path):
    from cowork.services.comparisons import copy_project_files

    other = tmp_path / "other-org"
    other.mkdir()
    (other / "secret.txt").write_text("secret")
    src = tmp_path / "src"
    src.mkdir()
    os.symlink(other, src / "planted")
    dest = tmp_path / "dest"
    dest.mkdir()
    assert copy_project_files(src, dest, org_mode=True) == 0
    assert not (dest / "planted").exists()


def test_a_project_too_big_to_copy_is_refused_and_leaves_nothing_behind(client, monkeypatch):
    from cowork.services import comparisons

    source = _real_project(client, "too-big")
    root = Path(source["path"])
    for i in range(3):
        (root / f"f{i}.txt").write_text("x")
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 2)

    before = {p.name for p in ProjectService(ScopedSession(get_open_session(), LOCAL_SCOPE)).list_projects()}
    r = _create(client, source_project_id=source["id"])
    assert r.status_code == 413, r.text
    after = {p.name for p in ProjectService(ScopedSession(get_open_session(), LOCAL_SCOPE)).list_projects()}
    assert after == before


def test_copy_budget_counts_bytes_as_read(tmp_path, monkeypatch):
    from cowork.services import comparisons

    src = tmp_path / "src"
    src.mkdir()
    (src / "big.bin").write_bytes(b"x" * 10)
    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.setattr(comparisons, "_COPY_MAX_BYTES", 9)
    with pytest.raises(comparisons.ProjectTooLargeToCopyError):
        comparisons.copy_project_files(src, dest, org_mode=False)


def test_verdicts_are_kept_per_turn_and_the_latest_is_the_verdict(client):
    cid = _create(client).json()["id"]
    r = client.put(f"/api/v1/comparisons/{cid}/verdicts/0", json={"winner": "a"})
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "a"

    r = client.put(f"/api/v1/comparisons/{cid}/verdicts/1", json={"winner": "b"})
    assert r.json()["verdict"] == "b"
    # Changing an earlier turn's verdict rewrites that turn, not the latest.
    r = client.put(f"/api/v1/comparisons/{cid}/verdicts/0", json={"winner": "tie"})
    body = r.json()
    assert [(v["turnIndex"], v["winner"]) for v in body["verdicts"]] == [(0, "tie"), (1, "b")]
    assert body["verdict"] == "b"


def test_verdict_rejects_an_unknown_winner(client):
    cid = _create(client).json()["id"]
    r = client.put(f"/api/v1/comparisons/{cid}/verdicts/0", json={"winner": "both"})
    assert r.status_code == 422


def test_verdict_rejects_a_negative_turn(client):
    cid = _create(client).json()["id"]
    r = client.put(f"/api/v1/comparisons/{cid}/verdicts/-1", json={"winner": "a"})
    assert r.status_code == 400


def test_list_and_get(client):
    cid = _create(client, title="listed one").json()["id"]
    listed = client.get("/api/v1/comparisons/").json()["comparisons"]
    assert cid in {c["id"] for c in listed}
    assert client.get(f"/api/v1/comparisons/{cid}").json()["title"] == "listed one"
    assert client.get(f"/api/v1/comparisons/{uuid4()}").status_code == 404


def test_continue_moves_the_side_into_a_real_project(client):
    body = _create(client).json()
    side_a = body["sides"][0]
    project = _real_project(client, "continue-target")

    r = client.post(
        f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": project["id"]}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"conversationId": side_a["conversationId"], "projectId": project["id"], "carriedAll": True}
    assert str(_conversation(side_a["conversationId"]).project_id) == project["id"]

    after = client.get(f"/api/v1/comparisons/{body['id']}").json()
    continued = after["sides"][0]
    assert continued["continuedAt"] is not None
    assert continued["continuedTurnCount"] == 0
    # It is an ordinary task now: listed across projects again.
    all_tasks = client.get("/api/v1/conversations/", params={"project": "all", "limit": 500}).json()
    assert side_a["conversationId"] in {c["id"] for c in all_tasks["conversations"]}

    again = client.post(
        f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": project["id"]}
    )
    assert again.status_code == 409


def test_continue_refuses_a_sandbox_as_the_destination(client):
    body = _create(client).json()
    r = client.post(
        f"/api/v1/comparisons/{body['id']}/sides/a/continue",
        json={"projectId": body["sides"][1]["projectId"]},
    )
    assert r.status_code == 400, r.text


def test_continue_refuses_while_the_side_is_running(client):
    body = _create(client).json()
    project = _real_project(client, "busy-target")
    running = _handle(running=True)
    with patch("cowork.streaming.registry.registry.get", return_value=running):
        r = client.post(
            f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": project["id"]}
        )
    assert r.status_code == 409


def test_continue_rejects_an_unknown_side(client):
    body = _create(client).json()
    project = _real_project(client, "side-c-target")
    r = client.post(
        f"/api/v1/comparisons/{body['id']}/sides/c/continue", json={"projectId": project["id"]}
    )
    assert r.status_code == 422


def test_delete_removes_the_sandboxes_but_not_a_continued_task(client):
    body = _create(client).json()
    a, b = body["sides"]
    project = _real_project(client, "delete-target")
    # Something the side wrote outside its artifacts.
    (Path(_project(a["projectId"]).path) / "analysis.csv").write_text("region,total\n")
    client.post(
        f"/api/v1/comparisons/{body['id']}/sides/a/continue",
        json={"projectId": project["id"], "modelLabel": "Kimi"},
    )
    b_path = Path(_project(b["projectId"]).path)

    assert client.delete(f"/api/v1/comparisons/{body['id']}").status_code == 204
    assert client.get(f"/api/v1/comparisons/{body['id']}").status_code == 404
    assert _project(b["projectId"]) is None
    assert not b_path.exists()
    assert _conversation(b["conversationId"]) is None
    # The continued side lives on as a task, with its work in its project.
    assert _conversation(a["conversationId"]) is not None
    carried = Path(project["path"]) / "Build a sales dashboard (Kimi)" / "analysis.csv"
    assert carried.read_text() == "region,total\n"


def test_continue_brings_what_the_side_made_or_changed_into_the_project(client, tmp_path):
    source = _real_project(client, "carry-source")
    _seed_source(Path(source["path"]))
    body = _create(client, source_project_id=source["id"]).json()
    a = body["sides"][0]
    sandbox = Path(_project(a["projectId"]).path)
    (sandbox / "notes.md").write_text("# notes, edited by the side")
    (sandbox / "scripts").mkdir()
    (sandbox / "scripts" / "clean.py").write_text("print('clean')")
    outside = tmp_path / "secret.txt"
    outside.write_text("not the side's")
    os.symlink(outside, sandbox / "linked.txt")
    target = _real_project(client, "carry-target")
    (Path(target["path"]) / "notes.md").write_text("# the project's own notes")

    r = client.post(
        f"/api/v1/comparisons/{body['id']}/sides/a/continue",
        json={"projectId": target["id"], "modelLabel": "Claude Opus 5.5"},
    )

    assert r.status_code == 200, r.text
    folder = Path(target["path"]) / "Build a sales dashboard (Claude Opus 5.5)"
    assert (folder / "notes.md").read_text() == "# notes, edited by the side"
    assert (folder / "scripts" / "clean.py").read_text() == "print('clean')"
    # An unchanged copy of the source isn't brought back, and a link is never followed.
    assert not (folder / "data").exists()
    assert not (folder / "linked.txt").exists()
    assert not (folder / ".anton").exists()
    # The project's own files are untouched.
    assert (Path(target["path"]) / "notes.md").read_text() == "# the project's own notes"
    # Everything the task needs is in its project, so the sandbox goes.
    assert _project(a["projectId"]) is None
    assert not sandbox.exists()


def _make_artifact(project_root: Path, slug: str, conversation_id: str) -> Path:
    import json

    folder = project_root / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"provenance": [{"conversation": conversation_id}]}))
    (folder / "index.html").write_text("<h1>report</h1>")
    return folder


def test_an_artifact_that_cannot_be_moved_keeps_the_sandbox_and_a_retry_moves_it(client, monkeypatch):
    from cowork.services import task_objects

    body = _create(client).json()
    a = body["sides"][0]
    sandbox = Path(_project(a["projectId"]).path)
    _make_artifact(sandbox, "report", a["conversationId"])
    target = _real_project(client, "artifact-target")
    url = f"/api/v1/comparisons/{body['id']}/sides/a/continue"
    real_move = task_objects.shutil.move

    def refuse(src, dst):
        raise PermissionError(13, "Permission denied", src)

    monkeypatch.setattr(task_objects.shutil, "move", refuse)
    first = client.post(url, json={"projectId": target["id"]})

    assert first.status_code == 200, first.text
    assert first.json()["carriedAll"] is False
    assert (sandbox / ".anton" / "artifacts" / "report" / "index.html").is_file()
    assert _project(a["projectId"]) is not None
    assert client.get(f"/api/v1/comparisons/{body['id']}").json()["sides"][0]["carryIncomplete"] is True

    monkeypatch.setattr(task_objects.shutil, "move", real_move)
    again = client.post(url, json={"projectId": target["id"]})

    assert again.json()["carriedAll"] is True
    assert (Path(target["path"]) / ".anton" / "artifacts" / "report" / "index.html").is_file()
    assert _project(a["projectId"]) is None
    assert client.get(f"/api/v1/comparisons/{body['id']}").json()["sides"][0]["carryIncomplete"] is False


def test_continue_with_nothing_new_makes_no_folder(client):
    source = _real_project(client, "untouched-source")
    _seed_source(Path(source["path"]))
    body = _create(client, source_project_id=source["id"]).json()
    target = _real_project(client, "untouched-target")

    client.post(f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": target["id"]})

    assert sorted(p.name for p in Path(target["path"]).iterdir() if not p.name.startswith(".")) == []


def test_continue_keeps_the_sandbox_when_the_work_cannot_all_be_carried(client, monkeypatch):
    from cowork.services import comparisons

    body = _create(client).json()
    a = body["sides"][0]
    sandbox = Path(_project(a["projectId"]).path)
    for i in range(3):
        (sandbox / f"out{i}.csv").write_text("x")
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 1)
    target = _real_project(client, "partial-target")

    r = client.post(f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": target["id"]})

    assert r.status_code == 200, r.text
    assert (sandbox / "out2.csv").is_file()
    assert _project(a["projectId"]) is not None
    assert client.delete(f"/api/v1/comparisons/{body['id']}").status_code == 204
    # Still kept after the comparison is deleted: it holds work the project doesn't have.
    assert (sandbox / "out2.csv").is_file()


def test_a_file_that_cannot_be_copied_keeps_the_sandbox(tmp_path):
    from cowork.services.comparisons import copy_side_changes

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    sandbox.mkdir()
    destination.mkdir()
    (sandbox / "ok.txt").write_text("ok")
    unreadable = sandbox / "locked.txt"
    unreadable.write_text("x")
    unreadable.chmod(0)
    try:
        result = copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})
    finally:
        unreadable.chmod(0o644)

    assert (result.files, result.complete) == (1, False)


def test_a_desktop_side_brings_back_a_folder_named_conversations(client):
    body = _create(client).json()
    a = body["sides"][0]
    sandbox = Path(_project(a["projectId"]).path)
    # Only a hosted project keeps task workspaces in `conversations/`; on the
    # desktop that name is the user's.
    (sandbox / "conversations").mkdir()
    (sandbox / "conversations" / "interview.md").write_text("notes")
    target = _real_project(client, "conversations-target")

    r = client.post(
        f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": target["id"], "modelLabel": "Kimi"}
    )

    assert r.status_code == 200, r.text
    carried = Path(target["path"]) / "Build a sales dashboard (Kimi)" / "conversations" / "interview.md"
    assert carried.read_text() == "notes"


def test_a_folder_that_cannot_be_read_keeps_the_sandbox(tmp_path):
    from cowork.services.comparisons import copy_side_changes

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    (sandbox / "locked").mkdir(parents=True)
    destination.mkdir()
    (sandbox / "locked" / "results.csv").write_text("x")
    (sandbox / "ok.txt").write_text("ok")
    (sandbox / "locked").chmod(0)
    try:
        result = copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})
    finally:
        (sandbox / "locked").chmod(0o755)

    assert (result.files, result.complete) == (1, False)


def test_a_folder_deeper_than_the_walk_keeps_the_sandbox(tmp_path, monkeypatch):
    from cowork.services import comparisons

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    (sandbox / "a" / "b").mkdir(parents=True)
    destination.mkdir()
    (sandbox / "a" / "b" / "deep.txt").write_text("x")
    monkeypatch.setattr(comparisons, "_COPY_MAX_DEPTH", 1)

    assert comparisons.copy_side_changes(sandbox, destination, "Run (Kimi)", copied={}).complete is False


def test_a_copy_cut_short_leaves_no_partial_file_in_the_project(tmp_path, monkeypatch):
    from cowork.services import comparisons

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    sandbox.mkdir()
    destination.mkdir()
    (sandbox / "big.csv").write_text("0123456789")
    monkeypatch.setattr(comparisons, "_COPY_CHUNK", 2)
    monkeypatch.setattr(comparisons, "_COPY_MAX_BYTES", 5)

    result = comparisons.copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})

    assert result.complete is False
    assert list((destination / "Run (Kimi)").iterdir()) == []


def test_a_partial_carry_says_so_and_a_retry_finishes_it_in_the_same_folder(client, monkeypatch):
    from cowork.services import comparisons

    body = _create(client).json()
    a = body["sides"][0]
    sandbox = Path(_project(a["projectId"]).path)
    for i in range(3):
        (sandbox / f"out{i}.csv").write_text(f"x{i}")
    target = _real_project(client, "retry-target")
    url = f"/api/v1/comparisons/{body['id']}/sides/a/continue"
    request = {"projectId": target["id"], "modelLabel": "Kimi"}

    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 1)
    first = client.post(url, json=request)
    assert first.status_code == 200, first.text
    assert first.json()["carriedAll"] is False
    assert _project(a["projectId"]) is not None

    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 2000)
    again = client.post(url, json=request)

    assert again.status_code == 200, again.text
    assert again.json() == {"conversationId": a["conversationId"], "projectId": target["id"], "carriedAll": True}
    folders = [p.name for p in Path(target["path"]).iterdir() if not p.name.startswith(".")]
    assert folders == ["Build a sales dashboard (Kimi)"]
    folder = Path(target["path"]) / folders[0]
    assert sorted(p.read_text() for p in folder.iterdir()) == ["x0", "x1", "x2"]
    # Everything is in the project now, so the sandbox goes, and the side is done.
    assert _project(a["projectId"]) is None
    assert client.post(url, json=request).status_code == 409


def test_a_retry_never_overwrites_a_carried_file_the_user_has_since_changed(client, monkeypatch):
    from cowork.services import comparisons

    body = _create(client).json()
    a = body["sides"][0]
    sandbox = Path(_project(a["projectId"]).path)
    (sandbox / "a.csv").write_text("side")
    (sandbox / "b.csv").write_text("side")
    target = _real_project(client, "edited-target")
    url = f"/api/v1/comparisons/{body['id']}/sides/a/continue"
    request = {"projectId": target["id"], "modelLabel": "Kimi"}
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 1)
    assert client.post(url, json=request).json()["carriedAll"] is False
    folder = Path(target["path"]) / "Build a sales dashboard (Kimi)"
    (folder / "a.csv").write_text("edited in the project")
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 2000)

    again = client.post(url, json=request)

    # The user's copy wins and counts as carried, so an edit can't keep every
    # retry incomplete and the sandbox around for good.
    assert again.json()["carriedAll"] is True
    assert (folder / "a.csv").read_text() == "edited in the project"
    assert (folder / "b.csv").read_text() == "side"
    assert _project(a["projectId"]) is None


def test_a_partly_carried_side_says_so_and_where_a_retry_goes(client, monkeypatch):
    from cowork.services import comparisons

    body = _create(client).json()
    sandbox = Path(_project(body["sides"][0]["projectId"]).path)
    for i in range(2):
        (sandbox / f"out{i}.csv").write_text(f"x{i}")
    target = _real_project(client, "flag-target")
    url = f"/api/v1/comparisons/{body['id']}/sides/a/continue"

    def sides():
        [listed] = [c["sides"] for c in client.get("/api/v1/comparisons/?limit=200").json()["comparisons"] if c["id"] == body["id"]]
        fetched = client.get(f"/api/v1/comparisons/{body['id']}").json()["sides"]
        assert listed == fetched
        return {s["label"]: (s["carryIncomplete"], s["continuedProjectId"]) for s in fetched}

    assert sides() == {"a": (False, None), "b": (False, None)}
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 1)
    client.post(url, json={"projectId": target["id"]})
    # Read back as stored, not from the Continue response, so it holds after a reload.
    assert sides() == {"a": (True, target["id"]), "b": (False, None)}

    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 2000)
    assert client.post(url, json={"projectId": target["id"]}).json()["carriedAll"] is True
    assert sides() == {"a": (False, target["id"]), "b": (False, None)}


def test_the_history_reads_continued_sides_in_one_query_per_table_however_many(client):
    from sqlalchemy import event

    for i in range(3):
        body = _create(client, title=f"page {i}").json()
        target = _real_project(client, f"page-target-{i}")
        client.post(f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": target["id"]})
    session = get_open_session()
    engine = session.get_bind()
    session.close()
    statements: list[str] = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        page = client.get("/api/v1/comparisons/?limit=200").json()["comparisons"]
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert sum(1 for c in page for side in c["sides"] if side["continuedAt"]) >= 3
    assert sum(1 for s in statements if "FROM projects" in s) == 1
    assert sum(1 for s in statements if "FROM conversations" in s) == 1


def test_a_fully_carried_side_is_not_marked_incomplete(client):
    body = _create(client).json()
    (Path(_project(body["sides"][0]["projectId"]).path) / "out.csv").write_text("x")
    target = _real_project(client, "complete-target")

    client.post(f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": target["id"]})

    side = client.get(f"/api/v1/comparisons/{body['id']}").json()["sides"][0]
    assert (side["carryIncomplete"], side["continuedProjectId"]) == (False, target["id"])


def test_a_file_that_appears_while_its_copy_is_written_is_never_replaced(tmp_path, monkeypatch):
    from cowork.services import comparisons

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    sandbox.mkdir()
    destination.mkdir()
    (sandbox / "report.md").write_text("from the side")
    real_copy = comparisons._copy_file

    def copy_then_race(src, dst, name, budget, **kwargs):
        real_copy(src, dst, name, budget, **kwargs)
        # A sync client or editor writes the same name between the check and the publish.
        (destination / "Run (Kimi)" / name).write_text("written meanwhile")

    monkeypatch.setattr(comparisons, "_copy_file", copy_then_race)

    result = comparisons.copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})

    folder = destination / "Run (Kimi)"
    assert (folder / "report.md").read_text() == "written meanwhile"
    assert sorted(p.name for p in folder.iterdir()) == ["report.md"]
    assert result.complete is True


# EINVAL and EACCES: how Windows reports a FAT/exFAT drive or a network share.
@pytest.mark.parametrize("errno_name", ["EXDEV", "EPERM", "ENOTSUP", "EINVAL", "EACCES"])
def test_without_hard_links_a_file_is_still_carried_and_never_replaces_one(tmp_path, monkeypatch, errno_name):
    import errno

    from cowork.services import comparisons

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    sandbox.mkdir()
    destination.mkdir()
    (sandbox / "new.md").write_text("new")
    (sandbox / "raced.md").write_text("from the side")

    def no_links(*args, **kwargs):
        raise OSError(getattr(errno, errno_name), "no hard links here")

    monkeypatch.setattr(comparisons.os, "link", no_links)
    real_copy = comparisons._copy_file

    def copy_then_race(src, dst, name, budget, **kwargs):
        real_copy(src, dst, name, budget, **kwargs)
        if name == "raced.md":
            (destination / "Run (Kimi)" / name).write_text("written meanwhile")

    monkeypatch.setattr(comparisons, "_copy_file", copy_then_race)

    result = comparisons.copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})

    folder = destination / "Run (Kimi)"
    assert (folder / "new.md").read_text() == "new"
    assert (folder / "raced.md").read_text() == "written meanwhile"
    assert sorted(p.name for p in folder.iterdir()) == ["new.md", "raced.md"]
    assert result.complete is True


def test_an_entry_that_cannot_be_stat_ed_keeps_the_sandbox(tmp_path, monkeypatch):
    from cowork.services import comparisons

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    sandbox.mkdir()
    destination.mkdir()
    (sandbox / "ok.txt").write_text("ok")
    (sandbox / "vanishing.txt").write_text("x")
    real_lstat = comparisons.dir_lstat

    def lstat(directory, name):
        if name == "vanishing.txt":
            raise PermissionError(name)
        return real_lstat(directory, name)

    monkeypatch.setattr(comparisons, "dir_lstat", lstat)

    result = comparisons.copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})

    assert (result.files, result.complete) == (1, False)


def test_a_folder_whose_listing_fails_keeps_the_sandbox(tmp_path, monkeypatch):
    from cowork.services import comparisons

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    (sandbox / "sub").mkdir(parents=True)
    destination.mkdir()
    (sandbox / "sub" / "inner.txt").write_text("x")
    (sandbox / "ok.txt").write_text("ok")
    real_scandir = comparisons.dir_scandir

    def scandir(directory):
        if directory.path.name == "sub":
            raise OSError("listing failed")
        return real_scandir(directory)

    monkeypatch.setattr(comparisons, "dir_scandir", scandir)

    result = comparisons.copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})

    assert (result.files, result.complete) == (1, False)


def test_a_hosted_workspace_that_is_a_link_is_not_followed(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination, elsewhere = tmp_path / "sandbox", tmp_path / "project", tmp_path / "elsewhere"
    (sandbox / "conversations").mkdir(parents=True)
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("not the side's")
    (sandbox / "conversations" / str(side.conversation_id)).symlink_to(elsewhere)
    destination.mkdir()

    assert _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True) is False
    assert not (destination / "conversations").exists()
    assert (elsewhere / "secret.txt").read_text() == "not the side's"


def test_a_hosted_sandbox_whose_conversations_folder_is_a_link_is_not_followed(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination, elsewhere = tmp_path / "sandbox", tmp_path / "project", tmp_path / "elsewhere"
    sandbox.mkdir()
    destination.mkdir()
    (elsewhere / str(side.conversation_id)).mkdir(parents=True)
    (elsewhere / str(side.conversation_id) / "secret.txt").write_text("not the side's")
    (sandbox / "conversations").symlink_to(elsewhere)

    assert _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True) is False
    assert (elsewhere / str(side.conversation_id) / "secret.txt").read_text() == "not the side's"
    assert not (destination / "conversations" / str(side.conversation_id)).exists()


def test_a_hosted_project_whose_conversations_folder_is_a_link_is_not_followed(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination, elsewhere = tmp_path / "sandbox", tmp_path / "project", tmp_path / "elsewhere"
    (sandbox / "conversations" / str(side.conversation_id)).mkdir(parents=True)
    (sandbox / "conversations" / str(side.conversation_id) / "report.md").write_text("the side's")
    destination.mkdir()
    elsewhere.mkdir()
    (destination / "conversations").symlink_to(elsewhere)

    assert _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True) is False
    assert list(elsewhere.iterdir()) == []
    assert (sandbox / "conversations" / str(side.conversation_id) / "report.md").is_file()


def test_a_carried_folder_swapped_for_a_link_before_it_is_opened_is_not_followed(tmp_path, monkeypatch):
    import shutil

    from cowork.services import comparisons

    sandbox, destination, elsewhere = tmp_path / "s", tmp_path / "d", tmp_path / "elsewhere"
    sandbox.mkdir()
    destination.mkdir()
    elsewhere.mkdir()
    (sandbox / "out.csv").write_text("x")
    real_budget = comparisons._CopyBudget

    def swap_then_budget(*args, **kwargs):
        # Between the folder being made and its open, something replaces it.
        shutil.rmtree(destination / "F")
        (destination / "F").symlink_to(elsewhere)
        return real_budget(*args, **kwargs)

    monkeypatch.setattr(comparisons, "_CopyBudget", swap_then_budget)

    with pytest.raises(OSError):
        comparisons.copy_side_changes(sandbox, destination, "F", copied={})
    assert list(elsewhere.iterdir()) == []


def test_a_hosted_workspace_already_in_the_project_is_merged_without_overwriting(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination = tmp_path / "sandbox", tmp_path / "project"
    workspace = sandbox / "conversations" / str(side.conversation_id)
    (workspace / "out").mkdir(parents=True)
    (workspace / "out" / "chart.png").write_bytes(b"side")
    (workspace / "notes.md").write_text("side")
    target = destination / "conversations" / str(side.conversation_id)
    target.mkdir(parents=True)
    # The task ran in the project after a partial carry, so its workspace exists there.
    (target / "notes.md").write_text("written in the project")

    for _ in range(2):  # the same on every retry
        assert _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True) is True
    assert (target / "notes.md").read_text() == "written in the project"
    assert (target / "out" / "chart.png").read_bytes() == b"side"


def test_a_hosted_merge_brings_the_sides_agent_state_but_not_its_scratchpad_environments(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination = tmp_path / "sandbox", tmp_path / "project"
    workspace = sandbox / "conversations" / str(side.conversation_id)
    (workspace / ".anton" / "episodes").mkdir(parents=True)
    (workspace / ".anton" / "episodes" / "1.jsonl").write_text("side episode")
    (workspace / ".anton" / "anton.md").write_text("side instructions")
    (workspace / ".anton" / "scratchpad-venvs" / "sales" / "bin").mkdir(parents=True)
    (workspace / ".anton" / "scratchpad-venvs" / "sales" / "bin" / "python").write_text("venv")
    (workspace / "notes.md").write_text("side")
    target = destination / "conversations" / str(side.conversation_id)
    (target / ".anton").mkdir(parents=True)
    (target / ".anton" / "anton.md").write_text("edited in the project")

    assert _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True) is True
    assert (target / ".anton" / "episodes" / "1.jsonl").read_text() == "side episode"
    assert (target / ".anton" / "anton.md").read_text() == "edited in the project"
    assert (target / "notes.md").read_text() == "side"
    assert not (target / ".anton" / "scratchpad-venvs").exists()


def test_a_desktop_carry_still_leaves_the_projects_agent_state_out(tmp_path):
    from cowork.services.comparisons import copy_side_changes

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    (sandbox / ".anton" / "episodes").mkdir(parents=True)
    (sandbox / ".anton" / "episodes" / "1.jsonl").write_text("episode")
    (sandbox / "report.md").write_text("work")
    destination.mkdir()

    result = copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})

    assert result.complete is True
    assert sorted(p.name for p in (destination / "Run (Kimi)").iterdir()) == ["report.md"]


@pytest.mark.parametrize("kind", ["link", "file"])
def test_a_hosted_target_that_is_not_a_folder_is_left_alone(tmp_path, kind):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination, elsewhere = tmp_path / "sandbox", tmp_path / "project", tmp_path / "elsewhere"
    workspace = sandbox / "conversations" / str(side.conversation_id)
    workspace.mkdir(parents=True)
    (workspace / "notes.md").write_text("side")
    (destination / "conversations").mkdir(parents=True)
    elsewhere.mkdir()
    target = destination / "conversations" / str(side.conversation_id)
    if kind == "link":
        target.symlink_to(elsewhere)
    else:
        target.write_text("not a folder")

    assert _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True) is False
    assert [p.name for p in (destination / "conversations").iterdir()] == [str(side.conversation_id)]
    assert list(elsewhere.iterdir()) == []
    assert (workspace / "notes.md").read_text() == "side"


def test_a_hosted_side_with_no_workspace_has_nothing_to_carry(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "project").mkdir()

    assert _carry_side_work(
        side, sandbox=tmp_path / "sandbox", destination=tmp_path / "project", folder_name="x", org_mode=True
    ) is True


def test_a_side_whose_kept_sandbox_is_gone_stops_offering_the_rest(client, monkeypatch):
    from cowork.services import comparisons

    body = _create(client).json()
    a = body["sides"][0]
    sandbox = Path(_project(a["projectId"]).path)
    for i in range(2):
        (sandbox / f"out{i}.csv").write_text("x")
    target = _real_project(client, "gone-target")
    url = f"/api/v1/comparisons/{body['id']}/sides/a/continue"
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 1)
    client.post(url, json={"projectId": target["id"]})
    session = get_open_session()
    try:
        ProjectService(ScopedSession(session, LOCAL_SCOPE)).delete_project(UUID(a["projectId"]))
    finally:
        session.close()

    assert client.post(url, json={"projectId": target["id"]}).status_code == 409
    assert client.get(f"/api/v1/comparisons/{body['id']}").json()["sides"][0]["carryIncomplete"] is False


def test_a_hosted_merge_that_cannot_copy_everything_keeps_the_sandbox(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination = tmp_path / "sandbox", tmp_path / "project"
    workspace = sandbox / "conversations" / str(side.conversation_id)
    workspace.mkdir(parents=True)
    locked = workspace / "locked.csv"
    locked.write_text("x")
    (destination / "conversations" / str(side.conversation_id)).mkdir(parents=True)
    locked.chmod(0)
    try:
        carried = _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True)
    finally:
        locked.chmod(0o644)

    assert carried is False


def test_an_unreadable_side_file_is_not_carried_by_a_same_named_project_file(tmp_path):
    from cowork.services.comparisons import copy_side_changes

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    sandbox.mkdir()
    (destination / "Run (Kimi)").mkdir(parents=True)
    (destination / "Run (Kimi)" / "report.md").write_text("the user's")
    unreadable = sandbox / "report.md"
    unreadable.write_text("the side's")
    unreadable.chmod(0)
    try:
        result = copy_side_changes(sandbox, destination, "Run (Kimi)", copied={}, into="Run (Kimi)")
    finally:
        unreadable.chmod(0o644)

    # The side's version could not be read, so it can't be shown to be there.
    assert result.complete is False
    assert (destination / "Run (Kimi)" / "report.md").read_text() == "the user's"


def test_a_digest_never_reads_through_a_link(tmp_path):
    from cowork.common.paths import pinned_dir
    from cowork.services.comparisons import _file_digest

    (tmp_path / "real.txt").write_text("x")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")

    with pinned_dir(tmp_path) as d:
        assert _file_digest(d, "real.txt") is not None
        assert _file_digest(d, "link.txt") is None


def test_a_file_copy_never_reads_through_a_link(tmp_path):
    from cowork.common.paths import pinned_dir
    from cowork.services.comparisons import _copy_file, _CopyBudget

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (tmp_path / "outside.txt").write_text("outside")
    (src / "link.txt").symlink_to(tmp_path / "outside.txt")
    budget = _CopyBudget()

    with pinned_dir(src) as s, pinned_dir(dst) as d:
        _copy_file(s, d, "link.txt", budget)

    assert (budget.files, list(dst.iterdir())) == (0, [])


def test_a_retry_into_a_different_project_is_refused(client, monkeypatch):
    from cowork.services import comparisons

    body = _create(client).json()
    sandbox = Path(_project(body["sides"][0]["projectId"]).path)
    for i in range(2):
        (sandbox / f"out{i}.csv").write_text("x")
    first, other = _real_project(client, "first-target"), _real_project(client, "other-target")
    url = f"/api/v1/comparisons/{body['id']}/sides/a/continue"
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 1)
    client.post(url, json={"projectId": first["id"]})

    assert client.post(url, json={"projectId": other["id"]}).status_code == 409


def test_a_long_model_name_is_shortened_rather_than_refused():
    from cowork.schemas.comparisons import ComparisonContinueRequest

    body = ComparisonContinueRequest.model_validate({"projectId": str(uuid4()), "modelLabel": "x" * 5000})

    assert body.model_label == "x" * 80


def test_a_long_model_name_still_continues(client):
    body = _create(client).json()
    (Path(_project(body["sides"][0]["projectId"]).path) / "out.csv").write_text("x")
    target = _real_project(client, "long-label-target")
    label = "Claude Opus 5.5 (1M context) " * 5

    r = client.post(
        f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": target["id"], "modelLabel": label}
    )

    assert r.status_code == 200, r.text
    [folder] = [p.name for p in Path(target["path"]).iterdir() if not p.name.startswith(".")]
    assert folder == "Build a sales dashboard (Claude Opus 5.5 (1M context) Claude Opus)"


def test_a_hosted_side_takes_its_whole_workspace_along(tmp_path):
    from cowork.services.comparisons import _carry_side_work

    side = SimpleNamespace(conversation_id=uuid4())
    sandbox, destination = tmp_path / "sandbox", tmp_path / "project"
    workspace = sandbox / "conversations" / str(side.conversation_id)
    workspace.mkdir(parents=True)
    (workspace / "session.pkl").write_bytes(b"state")
    destination.mkdir()

    assert _carry_side_work(side, sandbox=sandbox, destination=destination, folder_name="x", org_mode=True)
    assert (destination / "conversations" / str(side.conversation_id) / "session.pkl").read_bytes() == b"state"
    assert not workspace.exists()


def test_the_carried_folder_is_named_for_the_comparison_and_side():
    from cowork.services.comparisons import _carried_folder_name

    assert _carried_folder_name("Build a dashboard\nfrom sales", "Claude Opus 5.5") == "Build a dashboard (Claude Opus 5.5)"
    # Only ever one path component.
    assert _carried_folder_name("../../etc/passwd", "a/b:c") == "etc passwd (a b c)"
    assert _carried_folder_name("", "") == "Comparison"


def test_a_second_carry_into_the_same_project_gets_its_own_folder(tmp_path):
    from cowork.services.comparisons import copy_side_changes

    sandbox, destination = tmp_path / "s", tmp_path / "d"
    sandbox.mkdir()
    destination.mkdir()
    (sandbox / "a.txt").write_text("a")
    (destination / "Run (Kimi)").mkdir()

    result = copy_side_changes(sandbox, destination, "Run (Kimi)", copied={})
    assert (result.folder, result.files, result.complete) == ("Run (Kimi) 2", 1, True)
    assert (destination / "Run (Kimi) 2" / "a.txt").read_text() == "a"


def test_delete_frees_the_sandboxes_scratchpad_slots_and_backends(client):
    from unittest.mock import MagicMock

    from cowork.services import artifacts, scratchpad_runtime

    body = _create(client).json()
    a, b = body["sides"]
    elsewhere = _real_project(client, "unrelated")
    sandbox_paths = [_project(side["projectId"]).path for side in (a, b)]
    procs = {}
    for path in [*sandbox_paths, elsewhere["path"]]:
        scratchpad_runtime._pads[scratchpad_runtime._key("dash", path)] = object()
        procs[path] = MagicMock(returncode=None)
        artifacts._LAUNCHED_BACKENDS[str(Path(path).resolve())] = {"dash": {"proc": procs[path]}}
    try:
        assert client.delete(f"/api/v1/comparisons/{body['id']}").status_code == 204

        for path in sandbox_paths:
            assert scratchpad_runtime.get("dash", workspace_path=path) is None
            procs[path].terminate.assert_called_once()
        # Another project's preview is untouched.
        assert scratchpad_runtime.get("dash", workspace_path=elsewhere["path"]) is not None
        procs[elsewhere["path"]].terminate.assert_not_called()
    finally:
        scratchpad_runtime._pads.clear()
        artifacts._LAUNCHED_BACKENDS.clear()


def test_delete_refuses_while_a_side_is_running(client):
    body = _create(client).json()
    running = _handle(running=True)
    with patch("cowork.streaming.registry.registry.get", return_value=running):
        assert client.delete(f"/api/v1/comparisons/{body['id']}").status_code == 409
    assert client.get(f"/api/v1/comparisons/{body['id']}").status_code == 200


def _turn(client, conversation_id, **extra):
    return client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False, "conversation": conversation_id, **extra},
    )


def test_a_side_turn_runs_on_the_sides_model_whatever_the_client_sends(client, harness):
    side = _create(client).json()["sides"][1]
    blocked = [{"engine": "slack", "name": "team"}]
    with patch("cowork.handlers.responses.blocked_connections", AsyncMock(return_value=blocked)):
        r = _turn(
            client,
            side["conversationId"],
            model="something-else",
            reasoning_effort="low",
            disabled_connections=[{"engine": "postgres", "name": "warehouse"}],
        )
    assert r.status_code == 200, r.text
    call = harness.calls[0]
    assert call["model"] == "qwen"
    assert call["reasoning_effort"] == "xhigh"
    # The client's own pick stays off, and the messaging connector joins it.
    assert call["disabled_connections"] == [
        {"engine": "postgres", "name": "warehouse"},
        {"engine": "slack", "name": "team"},
    ]
    assert call["trace_metadata"]["response_route_reason"] == "comparison_side"


def test_a_side_turn_fails_closed_when_connections_cannot_be_listed(client, harness):
    side = _create(client).json()["sides"][0]
    with patch(
        "cowork.handlers.responses.blocked_connections", AsyncMock(side_effect=RuntimeError("auth down"))
    ):
        r = _turn(client, side["conversationId"])
    assert r.status_code == 503
    assert harness.calls == []


def test_an_ordinary_turn_is_untouched(client, harness):
    project = _real_project(client, "ordinary")
    conv = client.post("/api/v1/conversations/", json={"title": "t", "projectId": project["id"]}).json()
    lister = AsyncMock(return_value=[{"engine": "slack", "name": "team"}])
    with patch("cowork.handlers.responses.blocked_connections", lister):
        r = _turn(client, conv["id"], model="picked", reasoning_effort="low")
    assert r.status_code == 200, r.text
    call = harness.calls[0]
    assert call["model"] == "picked"
    assert call["reasoning_effort"] == "low"
    assert call["disabled_connections"] is None
    assert call["trace_metadata"]["response_route_reason"] != "comparison_side"
    lister.assert_not_called()


def test_a_continued_side_is_an_ordinary_task(client, harness):
    body = _create(client).json()
    project = _real_project(client, "continued-turns")
    client.post(f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": project["id"]})
    with patch("cowork.handlers.responses.blocked_connections", AsyncMock(return_value=[])) as lister:
        r = _turn(client, body["sides"][0]["conversationId"], model="now-mine")
    assert r.status_code == 200, r.text
    assert harness.calls[0]["model"] == "now-mine"
    lister.assert_not_called()


def test_ineligible_reason_puts_a_side_first():
    from cowork.handlers.response_routing import ineligible_reason

    assert ineligible_reason(
        has_non_text_input=True, has_attachments=True, has_disabled_connections=True, is_comparison_side=True
    ) == "comparison_side"
    assert ineligible_reason(
        has_non_text_input=False, has_attachments=False, has_disabled_connections=False
    ) is None


def test_blocked_connections_are_the_messaging_ones(monkeypatch):
    from cowork.services import comparisons
    from cowork.services.connectors import connections

    listed = [
        SimpleNamespace(engine="slack", name="team"),
        SimpleNamespace(engine="gmail", name="me"),
        SimpleNamespace(engine="postgres", name="warehouse"),
        SimpleNamespace(engine="hubspot", name="crm"),
        SimpleNamespace(engine="google_calendar", name="work"),
        SimpleNamespace(engine="pagerduty", name="oncall"),
        SimpleNamespace(engine="doordash", name="lunch"),
        SimpleNamespace(engine="shippo", name="labels"),
        SimpleNamespace(engine="google_analytics_4", name="site"),
        SimpleNamespace(engine="my-custom-thing", name="x"),
    ]
    monkeypatch.setattr(connections.ConnectionsService, "list", lambda self: listed)
    import asyncio

    blocked = [item["engine"] for item in asyncio.run(comparisons.blocked_connections(LOCAL_SCOPE))]
    # Reaching people, acting in the world, and anything the catalog can't vouch for.
    assert blocked == ["slack", "gmail", "google_calendar", "pagerduty", "doordash", "shippo", "my-custom-thing"]


def test_blocked_connections_read_auths_list_on_a_hosted_deployment(monkeypatch):
    import asyncio

    from cowork.db.scoped import TenantScope
    from cowork.services import comparisons

    lister = AsyncMock(return_value=[{"engine": "outlook", "name": "o"}, {"engine": "snowflake", "name": "s"}])
    monkeypatch.setattr("cowork.turnqueue.auth_keys.list_active_connections", lister)
    scope = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
    assert asyncio.run(comparisons.blocked_connections(scope)) == [{"engine": "outlook", "name": "o"}]
    assert lister.await_args.kwargs["org_id"] == "org-1"
    assert lister.await_args.kwargs["user_id"] == "user-1"


def test_every_blocked_category_exists_in_the_catalog():
    # A renamed category would silently turn a messaging connector back on.
    from cowork.services.comparisons import BLOCKED_CONNECTOR_CATEGORIES
    from cowork.services.connectors.specs._registry import registry

    categories = {spec.category for spec in registry.list_connectors()}
    assert BLOCKED_CONNECTOR_CATEGORIES <= categories


def test_every_blocked_engine_exists_in_the_catalog():
    # A renamed engine would silently turn a connector that reaches people back on.
    from cowork.services.comparisons import BLOCKED_CONNECTOR_ENGINES
    from cowork.services.connectors.specs._registry import registry

    assert all(registry.get_connector(engine) is not None for engine in BLOCKED_CONNECTOR_ENGINES)


def test_a_side_never_writes_memory_in_process():
    from cowork.harnesses.anton_harness.harness import _memory_mode

    settings = SimpleNamespace(memory_enabled=True, memory_mode="autopilot")
    side = SimpleNamespace(project=SimpleNamespace(name=f"{COMPARISON_SANDBOX_PREFIX}abc"))
    ordinary = SimpleNamespace(project=SimpleNamespace(name="reports"))
    assert _memory_mode(settings, side) == "off"
    assert _memory_mode(settings, ordinary) == "autopilot"
    assert _memory_mode(SimpleNamespace(memory_enabled=False, memory_mode="autopilot"), ordinary) == "off"


def test_a_side_never_writes_memory_remotely(client):
    from cowork.handlers.responses import ResponsesHandler

    side = _create(client).json()["sides"][0]
    project = _real_project(client, "remote-memory")
    conv = client.post("/api/v1/conversations/", json={"title": "t", "projectId": project["id"]}).json()
    session = ScopedSession(get_open_session(), LOCAL_SCOPE)
    with patch("cowork.handlers.responses.apply_turn_memory", return_value=1) as apply:
        ResponsesHandler._persist_turn_memory(session, UUID(side["conversationId"]), [{"x": 1}], None)
        apply.assert_not_called()
        ResponsesHandler._persist_turn_memory(session, UUID(conv["id"]), [{"x": 1}], None)
        apply.assert_called_once()


def test_publishing_from_a_side_is_refused_before_anything_else(tmp_path):
    from cowork.services.publish import ComparisonPublishRefused, publish_artifact

    base = tmp_path / f"{COMPARISON_SANDBOX_PREFIX}abc" / ".anton" / "artifacts"
    with pytest.raises(ComparisonPublishRefused) as excinfo:
        publish_artifact(base / "report", artifacts_base=base, api_key="", publish_url="https://x")
    # The agent tool treats an "api key" message as "go configure a key".
    assert "api key" not in str(excinfo.value).lower()

    ordinary = tmp_path / "reports" / ".anton" / "artifacts"
    with pytest.raises(ValueError, match="requires an API key"):
        publish_artifact(ordinary / "report", artifacts_base=ordinary, api_key="", publish_url="https://x")


def test_autopublish_skips_a_side(tmp_path, monkeypatch):
    import asyncio

    from cowork.db.scoped import TenantScope
    from cowork.services import artifact_autopublish

    monkeypatch.setattr(artifact_autopublish, "_is_enabled", lambda scope: True)
    candidates = []
    monkeypatch.setattr(artifact_autopublish, "_candidate_slugs", lambda base: candidates.append(base) or [])
    scope = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")

    side_base = tmp_path / f"{COMPARISON_SANDBOX_PREFIX}abc" / ".anton" / "artifacts"
    assert asyncio.run(artifact_autopublish.autopublish_project_artifacts(side_base, scope, touched=set(), project_id="p-side")) == set()
    assert candidates == []

    ordinary = tmp_path / "reports" / ".anton" / "artifacts"
    monkeypatch.setattr(artifact_autopublish, "_publish_url", lambda scope: "https://x")
    monkeypatch.setattr(artifact_autopublish, "_active_workspace_id", lambda scope: None)
    asyncio.run(artifact_autopublish.autopublish_project_artifacts(ordinary, scope, touched=set(), project_id="p-real"))
    assert candidates == [ordinary]


def test_a_comparison_is_invisible_to_another_member(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    try:
        from sqlalchemy.pool import StaticPool
        from sqlmodel import Session, SQLModel, create_engine

        from cowork.db.scoped import TenantScope
        from cowork.services.comparisons import ComparisonNotFoundError, ComparisonService, SideSpec

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        SQLModel.metadata.create_all(engine)
        org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        alice = ComparisonService(ScopedSession(Session(engine), TenantScope(True, org, "alice")))
        bob = ComparisonService(ScopedSession(Session(engine), TenantScope(True, org, "bob")))

        comparison = alice.create_comparison(title="mine", sides=[SideSpec("kimi"), SideSpec("qwen")])
        assert comparison.org_id == org and comparison.created_by == "alice"
        assert all(side.org_id == org and side.created_by == "alice" for side in comparison.sides)
        assert bob.list_comparisons() == []
        with pytest.raises(ComparisonNotFoundError):
            bob.get_comparison(comparison.id)
        with pytest.raises(ComparisonNotFoundError):
            bob.delete_comparison(comparison.id)
        assert [c.id for c in alice.list_comparisons()] == [comparison.id]
    finally:
        get_app_settings.cache_clear()


def test_same_slug_backends_in_two_projects_are_tracked_apart(tmp_path, monkeypatch):
    import asyncio

    from cowork.services import artifacts

    seen: list[dict] = []

    async def fake_launch(*, slug, artifact_folder, scratchpad_pool, tracked_backends, **_):
        seen.append(tracked_backends)
        tracked_backends[slug] = {"proc": None, "port": 1}
        return {"port": 5000 + len(seen), "pid": 1}

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", fake_launch)
    roots = {}
    for name in ("one", "two"):
        root = tmp_path / name
        folder = root / ".anton" / "artifacts" / "dash"
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text("{}")
        roots[name] = (root, folder)
    monkeypatch.setattr(artifacts, "_registered_project_dirs", lambda: {r.resolve() for r, _ in roots.values()})
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})

    for _root, folder in roots.values():
        running, _detail, _port = asyncio.run(artifacts._launch_backend_locked(folder, "dash"))
        assert running
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert set(artifacts._LAUNCHED_BACKENDS) == {str(r.resolve()) for r, _ in roots.values()}


def test_scratchpad_pool_keys_by_workspace(tmp_path, monkeypatch):
    from cowork.services import scratchpad_runtime

    made = []
    monkeypatch.setattr(scratchpad_runtime, "_pads", {})
    monkeypatch.setattr(scratchpad_runtime, "_resolve_coding", lambda **kw: ("p", "m", "k", "b"))
    monkeypatch.setattr(
        scratchpad_runtime,
        "_make_runtime",
        lambda name, **kw: made.append((name, kw["workspace_path"])) or object(),
    )
    one = scratchpad_runtime.get_or_create("dash", workspace_path=str(tmp_path / "one"))
    two = scratchpad_runtime.get_or_create("dash", workspace_path=str(tmp_path / "two"))
    assert one is not two
    assert scratchpad_runtime.get_or_create("dash", workspace_path=str(tmp_path / "one")) is one
    assert [name for name, _ in made] == ["dash", "dash"]
    assert scratchpad_runtime.list_pads() == ["dash", "dash"]


def test_sandboxes_are_hidden_from_search_and_memories(client):
    body = _create(client).json()
    side = body["sides"][0]
    sandbox = _project(side["projectId"])
    folder = Path(sandbox.path) / ".anton" / "artifacts" / "searchable-side"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<h1>x</h1>")
    (folder / "metadata.json").write_text('{"type": "html-app", "primary": "index.html"}')

    # Search scores projects on name and path, so query the sandbox's own name.
    results = client.get("/api/v1/search", params={"q": sandbox.name}).json()["results"]
    assert sandbox.name not in {r["id"] for r in results if r["type"] == "project"}
    results = client.get("/api/v1/search", params={"q": "searchable-side"}).json()["results"]
    assert not [r for r in results if r["type"] == "artifact"]

    memories = client.get("/api/v1/memory/").json()
    listed = {m.get("projectId") or m.get("project_id") for m in memories}
    assert side["projectId"] not in listed
    # Asked for by id, a side's memory still answers.
    own = client.get("/api/v1/memory/", params={"project_id": side["projectId"]}).json()
    assert side["projectId"] in {m.get("projectId") or m.get("project_id") for m in own}


def test_a_side_turn_runs_on_anton_whatever_the_client_picks(harness):
    from cowork.harnesses.base import _registry, register
    from cowork.server import create_app

    @register
    class _Other:
        id = "other"
        label = "Other"

    asked: list[str] = []
    try:
        with patch(
            "cowork.handlers.responses.get_harness",
            side_effect=lambda name: asked.append(name) or harness,
        ), patch("cowork.handlers.responses.blocked_connections", AsyncMock(return_value=[])):
            client = TestClient(create_app())
            side = _create(client).json()["sides"][0]
            r = _turn(client, side["conversationId"], harness="other")
    finally:
        _registry.pop("other", None)
    assert r.status_code == 200, r.text
    assert asked == ["anton"]


def test_backend_launches_lock_per_artifact_not_per_slug(tmp_path, monkeypatch):
    import asyncio

    from cowork.services import artifacts

    monkeypatch.setattr(artifacts, "_BACKEND_LAUNCH_LOCKS", {})
    monkeypatch.setattr(artifacts, "_org_mode", lambda: False)
    monkeypatch.setattr(artifacts, "_probe_port", lambda port, **kw: False)

    async def fake_locked(artifact_dir, slug):
        return True, "launched", 1

    monkeypatch.setattr(artifacts, "_launch_backend_locked", fake_locked)
    folders = [tmp_path / name / ".anton" / "artifacts" / "dash" for name in ("one", "two")]
    for folder in folders:
        folder.mkdir(parents=True)
        asyncio.run(artifacts._ensure_backend_running(folder, 1))
    assert set(artifacts._BACKEND_LAUNCH_LOCKS) == {str(f.resolve()) for f in folders}


def test_delete_never_removes_a_project_that_is_not_a_sandbox(client):
    body = _create(client).json()
    real = _real_project(client, "must-survive")
    session = get_open_session()
    try:
        from sqlmodel import select

        side = session.exec(
            select(ComparisonSide).where(ComparisonSide.conversation_id == UUID(body["sides"][0]["conversationId"]))
        ).one()
        side.project_id = UUID(real["id"])
        session.add(side)
        session.commit()
    finally:
        session.close()
    assert client.delete(f"/api/v1/comparisons/{body['id']}").status_code == 204
    assert _project(real["id"]) is not None


def test_a_continued_side_shows_the_turns_it_was_compared_on(client):
    from cowork.models.message import Message

    body = _create(client).json()
    conversation_id = UUID(body["sides"][0]["conversationId"])
    session = get_open_session()
    try:
        session.add(Message(conversation_id=conversation_id, role="user", content="q", seq=1))
        # A tool call as a turn really stores it: the call on an assistant row,
        # its result on a USER row. Neither is returned by the transcript API.
        session.add(Message(conversation_id=conversation_id, role="assistant",
                            content=[{"type": "tool_use", "id": "t1", "name": "web_search", "input": {}}], seq=2))
        session.add(Message(conversation_id=conversation_id, role="user",
                            content=[{"type": "tool_result", "tool_use_id": "t1", "content": "found"}], seq=3))
        session.add(Message(conversation_id=conversation_id, role="assistant", content="a", seq=4))
        session.commit()
    finally:
        session.close()
    project = _real_project(client, "count-target")
    client.post(f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": project["id"]})

    session = get_open_session()
    try:
        session.add(Message(conversation_id=conversation_id, role="user", content="later", seq=5))
        session.commit()
    finally:
        session.close()
    side = client.get(f"/api/v1/comparisons/{body['id']}").json()["sides"][0]
    assert side["continuedTurnCount"] == 1
    assert side["turnCount"] == 1


def test_a_sandbox_gets_the_desktop_skill_links(client):
    with patch("cowork.services.skill_links.reconcile_project") as reconcile:
        body = _create(client).json()
    linked = {Path(call.args[0]) for call in reconcile.call_args_list}
    assert {Path(_project(s["projectId"]).path) for s in body["sides"]} <= linked


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_copy_skips_a_fifo_swapped_in_after_the_check(tmp_path, monkeypatch):
    from cowork.services import comparisons

    src = tmp_path / "src"
    src.mkdir()
    os.mkfifo(src / "pipe")
    dest = tmp_path / "dest"
    dest.mkdir()
    real_lstat = comparisons.dir_lstat
    regular = os.stat(tmp_path)  # any stat; mode replaced below

    def lying_lstat(d, name):
        st = real_lstat(d, name)
        if name == "pipe":
            import stat as st_mod

            return os.stat_result((st_mod.S_IFREG | 0o644,) + tuple(regular)[1:])
        return st

    monkeypatch.setattr(comparisons, "dir_lstat", lying_lstat)
    assert comparisons.copy_project_files(src, dest, org_mode=False) == 0
    assert not (dest / "pipe").exists()


def test_the_migration_builds_what_the_models_declare(tmp_path, monkeypatch):
    import sqlite3

    from sqlalchemy import create_engine
    from sqlmodel import SQLModel

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.migrations import run_schema_migrations

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    try:
        db_path = tmp_path / "migrated.db"
        uri = f"sqlite:///{db_path}"
        run_schema_migrations(create_engine(uri), uri)
        with sqlite3.connect(db_path) as connection:
            for table in ("comparisons", "comparison_sides", "comparison_verdicts"):
                migrated = {row[1]: (row[3] == 1) for row in connection.execute(f"pragma table_info({table})")}
                declared = {c.name: (not c.nullable) for c in SQLModel.metadata.tables[table].columns}
                assert migrated == declared, table
                indexes = {row[1] for row in connection.execute(f"pragma index_list({table})") if not row[1].startswith("sqlite_autoindex")}
                declared_indexes = {i.name for i in SQLModel.metadata.tables[table].indexes}
                assert indexes == declared_indexes, table
    finally:
        get_app_settings.cache_clear()


def test_copy_refuses_a_directory_swapped_for_a_link_after_the_check(tmp_path, monkeypatch):
    import stat as st_mod

    from cowork.services import comparisons

    other = tmp_path / "other-org"
    other.mkdir()
    (other / "secret.txt").write_text("secret")
    src = tmp_path / "src"
    src.mkdir()
    os.symlink(other, src / "planted")
    dest = tmp_path / "dest"
    dest.mkdir()
    real_lstat = comparisons.dir_lstat

    def lying_lstat(d, name):
        st = real_lstat(d, name)
        if name == "planted":
            return os.stat_result((st_mod.S_IFDIR | 0o755,) + tuple(st)[1:])
        return st

    monkeypatch.setattr(comparisons, "dir_lstat", lying_lstat)
    comparisons.copy_project_files(src, dest, org_mode=True)
    assert not (dest / "planted" / "secret.txt").exists()


def test_a_failed_cleanup_does_not_hide_why_create_failed(client, monkeypatch):
    from cowork.services import comparisons

    source = _real_project(client, "cleanup-fails")
    for i in range(3):
        (Path(source["path"]) / f"f{i}.txt").write_text("x")
    monkeypatch.setattr(comparisons, "_COPY_MAX_FILES", 2)

    def refuse(self, project_id, **kwargs):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(ProjectService, "delete_project", refuse)
    r = _create(client, source_project_id=source["id"])
    assert r.status_code == 413, r.text


def test_a_continue_that_fails_part_way_can_be_retried(client):
    body = _create(client).json()
    project = _real_project(client, "retry-target")
    url = f"/api/v1/comparisons/{body['id']}/sides/a/continue"
    with patch(
        "cowork.services.conversations.ConversationService.update_conversation",
        side_effect=RuntimeError("db blip"),
    ):
        with pytest.raises(RuntimeError):
            client.post(url, json={"projectId": project["id"]})
    r = client.post(url, json={"projectId": project["id"]})
    assert r.status_code == 200, r.text
    assert str(_conversation(body["sides"][0]["conversationId"]).project_id) == project["id"]


def test_a_finished_turn_does_not_block_continue_or_delete(client):
    # The registry keeps a handle after its turn ends, so "there is a handle"
    # must not read as "a turn is running".
    body = _create(client).json()
    project = _real_project(client, "finished-target")
    with patch("cowork.streaming.registry.registry.get", return_value=_handle(running=False)):
        r = client.post(
            f"/api/v1/comparisons/{body['id']}/sides/a/continue", json={"projectId": project["id"]}
        )
        assert r.status_code == 200, r.text
        assert client.delete(f"/api/v1/comparisons/{body['id']}").status_code == 204
