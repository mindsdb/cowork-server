"""An artifact's revision journal reads and writes only its own files.

The journal, `.revisions` in the artifact folder, is a folder the agent's code
can write. A cell locked by the code sandbox can rewrite the manifest, a
pending source write or a repair record, and hard-link any file it can read,
its project's `.anton/.env` included. None of that may make the server read,
write or serve a file outside the journal. A planted blob name answers as a
missing revision does, a hard-linked blob is refused, a repair id names no file
outside `repairs/`, and a journal folder that is a link reads as no journal and
is replaced by a real folder on the first write, with its target untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from cowork.services import artifact_journal, artifact_revisions
from cowork.services.artifact_draft_review import draft_review_grant, enable_draft_review
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_journal import (
    MAX_JOURNAL_FILE_BYTES,
    JournalFileRefused,
    read_journal_file,
    write_journal_file,
)
from cowork.services.artifact_revisions import (
    MAX_SOURCE_BYTES,
    RevisionValidationError,
    active_agent_repair,
    agent_repair_detail,
    capture_agent_revision,
    create_agent_repair,
    current_source,
    finalize_agent_repair,
    list_revisions,
    revision_with_content,
    save_source,
)
from test_personal_partition_artifacts import (  # noqa: F401 (stack, two_people: fixtures)
    ALICE,
    BOB,
    TOKEN,
    A,
    B,
    _artifact,
    _artifact_id,
    _client,
    _general,
    _general_id,
    stack,
    two_people,
)

FIXTURES = Path(__file__).parent / "fixtures" / "artifact_revisions"
CASES = yaml.safe_load((FIXTURES / "planted_content_hashes.yaml").read_text(encoding="utf-8"))
CASE_IDS = [case["id"] for case in CASES]

SECRET = "OPENAI_API_KEY=sk-own-7f3a"
OTHER_SECRET = "OPENAI_API_KEY=sk-other-91c2"
HOME_SECRET = "server-home-secret-55d0"
SECRETS = (SECRET, OTHER_SECRET, HOME_SECRET)
FIRST = "# First\n"
MISSING = (404, {"detail": "Revision not found"})


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_secret(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _files_below(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class Layout:
    """One artifact in its project, beside the files a planted name could reach."""

    folder: Path
    metadata: dict
    artifact_id: str
    secret: Path
    other_secret: Path
    home_file: Path

    @property
    def journal(self) -> Path:
        return self.folder / ".revisions"


@pytest.fixture
def layout(tmp_path) -> Layout:
    project = tmp_path / "alice"
    folder = project / ".anton" / "artifacts" / "report"
    folder.mkdir(parents=True)
    metadata = {
        "id": "a1b2c3d4",
        "slug": "report",
        "name": "Report",
        "type": "document",
        "primary": "brief.md",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "brief.md").write_text(FIRST, encoding="utf-8")
    artifact_id, metadata = ensure_full_id(folder, metadata)
    return Layout(
        folder=folder,
        metadata=metadata,
        artifact_id=artifact_id,
        secret=_write_secret(project / ".anton" / ".env", SECRET),
        other_secret=_write_secret(tmp_path / "bob" / ".anton" / ".env", OTHER_SECRET),
        home_file=_write_secret(tmp_path / "home" / "server-secret.txt", HOME_SECRET),
    )


def _planted_hash(
    value: object, *, folder: Path, secret: Path, other_secret: Path, home_file: Path, real: str
) -> object:
    """A YAML case's contentHash with its placeholders filled in."""
    if not isinstance(value, str):
        return value
    filled = {
        "{secret}": str(secret),
        "{other_secret}": str(other_secret),
        "{other_relative}": os.path.relpath(other_secret, folder / ".revisions" / "blobs"),
        "{home_file}": str(home_file),
        "{real_upper}": real.upper(),
        "{real_short}": real[:-1],
        "{real}": real,
    }
    for placeholder, text in filled.items():
        value = value.replace(placeholder, text)
    return value


def _case_hash(case: dict, layout: Layout, *, real: str) -> object:
    return _planted_hash(
        case["content_hash"],
        folder=layout.folder,
        secret=layout.secret,
        other_secret=layout.other_secret,
        home_file=layout.home_file,
        real=real,
    )


def _plant_revision(folder: Path, *, content_hash: object) -> str:
    """Add a revision naming ``content_hash`` before the head, as the agent's
    code can, and return its id. The head stays the server's own."""
    manifest_path = folder / ".revisions" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planted = {**manifest["revisions"][-1], "id": str(uuid4()), "contentHash": content_hash}
    manifest["revisions"].insert(0, planted)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return planted["id"]


def _plant_ready_repair(folder: Path, *, base_revision_id: str, revision_id: str, path: str) -> str:
    """A repair record the agent's code wrote, ready for its owner's review."""
    repair_id = str(uuid4())
    repairs = folder / ".revisions" / "repairs"
    repairs.mkdir(parents=True, exist_ok=True)
    record = {
        "id": repair_id,
        "path": path,
        "status": "ready",
        "baseRevisionId": base_revision_id,
        "revisionId": revision_id,
        "commentThreadId": "thread-1",
        "conversationId": str(uuid4()),
        "createdAt": "2026-11-02T10:00:00+00:00",
        "updatedAt": "2026-11-02T10:00:00+00:00",
    }
    (repairs / f"{repair_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return repair_id


def _plant_pending(
    folder: Path, *, content_hash: object, path: str, base_revision_id: str, before: bytes
) -> Path:
    """A pending source write the agent's code wrote, shaped like the server's."""
    entry = {
        "id": str(uuid4()),
        "number": 99,
        "path": path,
        "contentHash": content_hash,
        "baseRevisionId": base_revision_id,
        "summary": "Edited artifact",
        "actor": {"kind": "manual", "id": None},
    }
    pending = folder / ".revisions" / "pending-source-write.json"
    pending.write_text(
        json.dumps({"version": 1, "path": path, "beforeContentHash": _sha(before), "revision": entry}),
        encoding="utf-8",
    )
    return pending


def _link_as_blob(folder: Path, secret: Path) -> str:
    """Hard-link ``secret`` into blobs/ under its own sha256, as a locked cell
    can: it may read its project's `.anton/.env` and make a second name for it.
    Returns that hash."""
    content_hash = _sha(secret.read_bytes())
    blobs = folder / ".revisions" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    os.link(secret, blobs / content_hash)
    return content_hash


def _victim(tmp_path: Path) -> Path:
    """Another artifact whose journal holds revisions, a ready repair, a comment
    and a draft-review grant, for a link to point at."""
    folder = tmp_path / "bob" / ".anton" / "artifacts" / "plan"
    folder.mkdir(parents=True)
    metadata = {"id": "e5f6a7b8", "slug": "plan", "name": "Plan", "type": "document", "primary": "plan.md"}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "plan.md").write_text("# Bob's plan\n", encoding="utf-8")
    artifact_id, metadata = ensure_full_id(folder, metadata)
    head = current_source(folder, metadata, artifact_id)["revision"]
    _plant_ready_repair(folder, base_revision_id=head["id"], revision_id=head["id"], path=head["path"])
    write_journal_file(folder, "comments.json", content=b'[{"id": "t1", "status": "open"}]')
    enable_draft_review(folder, org_id="org-1", enabled_by="bob")
    return folder


# ── a planted blob name ──────────────────────────────────────────────────────


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_a_planted_blob_name_reads_no_file(layout, case):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    planted = _plant_revision(layout.folder, content_hash=_case_hash(case, layout, real=head["contentHash"]))

    with pytest.raises(FileNotFoundError, match="^Revision not found$"):
        revision_with_content(layout.folder, planted)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_repair_compare_and_reject_read_no_planted_blob(layout, case):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    planted = _plant_revision(layout.folder, content_hash=_case_hash(case, layout, real=head["contentHash"]))
    repair_id = _plant_ready_repair(
        layout.folder, base_revision_id=planted, revision_id=head["id"], path=head["path"]
    )

    with pytest.raises(FileNotFoundError, match="^Revision not found$"):
        agent_repair_detail(layout.folder, repair_id)
    with pytest.raises(FileNotFoundError, match="^Revision not found$"):
        finalize_agent_repair(layout.folder, layout.metadata, layout.artifact_id, repair_id, "rejected")
    assert (layout.folder / "brief.md").read_text(encoding="utf-8") == FIRST


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_recovery_abandons_a_pending_write_naming_a_planted_blob(layout, case):
    """Before, recovery opened the name first and then answered one of two
    errors, which told the caller whether any server path exists."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    pending = _plant_pending(
        layout.folder,
        content_hash=_case_hash(case, layout, real=head["contentHash"]),
        path=head["path"],
        base_revision_id=head["id"],
        before=FIRST.encode(),
    )

    reread = current_source(layout.folder, layout.metadata, layout.artifact_id)

    assert reread["content"] == FIRST
    assert not pending.exists()


# ── a blob with a second name ────────────────────────────────────────────────


def test_a_blob_with_a_second_name_is_refused(layout):
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    planted = _plant_revision(layout.folder, content_hash=_link_as_blob(layout.folder, layout.secret))

    with pytest.raises(FileNotFoundError, match="^Revision not found$"):
        revision_with_content(layout.folder, planted)


def test_recovery_copies_no_hard_linked_blob_into_the_source(layout):
    """The blob's name matches its bytes, so only the one-name rule stops it."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    pending = _plant_pending(
        layout.folder,
        content_hash=_link_as_blob(layout.folder, layout.secret),
        path=head["path"],
        base_revision_id=head["id"],
        before=FIRST.encode(),
    )

    reread = current_source(layout.folder, layout.metadata, layout.artifact_id)

    assert reread["content"] == FIRST
    assert (layout.folder / "brief.md").read_text(encoding="utf-8") == FIRST
    assert not pending.exists()
    assert layout.secret.read_text(encoding="utf-8") == SECRET


def test_a_save_replaces_a_planted_blob_at_its_name(layout):
    """Before, a planted blob made the save fail after it wrote the source,
    which left the pending record behind for every later request to trip on."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    edited = "# Edited\n"
    planted = layout.journal / "blobs" / _sha(edited.encode())
    os.link(layout.secret, planted)

    saved = save_source(
        layout.folder,
        layout.metadata,
        layout.artifact_id,
        content=edited,
        expected_revision_id=head["id"],
    )

    assert revision_with_content(layout.folder, saved["revision"]["id"])["content"] == edited
    assert planted.read_text(encoding="utf-8") == edited
    assert layout.secret.read_text(encoding="utf-8") == SECRET
    assert os.stat(layout.secret).st_nlink == 1


# ── a planted repair id ──────────────────────────────────────────────────────


@pytest.mark.parametrize("planted_id", ["../../../../escaped", "{outside}"], ids=["relative", "absolute"])
def test_a_planted_repair_id_writes_no_file_outside_the_journal(layout, tmp_path, planted_id):
    """At turn end, the capture finishes this turn's queued repairs and writes
    each back by its id. A record the agent's code wrote must not pick that
    file name: before, these ids made the server write `escaped.json` in
    `.anton` and in a folder it created outside the project."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    outside = tmp_path / "outside" / "escaped"
    record = {
        "id": planted_id.replace("{outside}", str(outside)),
        "status": "queued",
        "conversationId": "turn-1",
        "path": "other.md",
        "baseRevisionId": head["id"],
        "createdAt": "2026-11-02T10:00:00+00:00",
    }
    repairs = layout.journal / "repairs"
    repairs.mkdir()
    planted = [repairs / f"{uuid4()}.json", repairs / "escaped.json"]
    for path in planted:
        path.write_text(json.dumps(record), encoding="utf-8")
    (layout.folder / "brief.md").write_text("# Agent edit\n", encoding="utf-8")

    captured = capture_agent_revision(layout.folder, conversation_id="turn-1")

    assert captured is not None and captured["actor"]["kind"] == "agent"
    assert not (layout.folder.parent.parent / "escaped.json").exists()
    assert not outside.parent.exists()
    assert all(json.loads(path.read_text(encoding="utf-8")) == record for path in planted)


# ── a journal folder that is a link ──────────────────────────────────────────


def test_a_linked_journal_reads_as_no_journal(layout, tmp_path):
    victim = _victim(tmp_path)
    before = _files_below(victim / ".revisions")
    layout.journal.symlink_to(victim / ".revisions", target_is_directory=True)

    assert read_journal_file(layout.folder, "comments.json") is None
    assert draft_review_grant(layout.folder) is None
    assert active_agent_repair(layout.folder) is None
    assert list_revisions(layout.folder) == []
    assert _files_below(victim / ".revisions") == before
    assert layout.journal.is_dir() and not layout.journal.is_symlink()


@pytest.mark.parametrize("linked", [".revisions", ".revisions/blobs", ".revisions/repairs"])
def test_writes_replace_a_linked_journal_folder_not_its_target(layout, tmp_path, linked):
    victim = _victim(tmp_path)
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    (layout.journal / "repairs").mkdir()
    link = layout.folder / linked
    shutil.rmtree(link)
    link.symlink_to(victim / linked, target_is_directory=True)
    before = _files_below(victim / ".revisions")

    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    saved = save_source(
        layout.folder,
        layout.metadata,
        layout.artifact_id,
        content="# Edited\n",
        expected_revision_id=head["id"],
    )
    create_agent_repair(
        layout.folder,
        layout.metadata,
        layout.artifact_id,
        expected_revision_id=saved["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "Fix the title"}],
        conversation_id=str(uuid4()),
    )
    write_journal_file(layout.folder, "comments.json", content=b"[]")
    enable_draft_review(layout.folder, org_id="org-2", enabled_by="alice")

    assert _files_below(victim / ".revisions") == before
    assert link.is_dir() and not link.is_symlink()
    assert revision_with_content(layout.folder, saved["revision"]["id"])["content"] == "# Edited\n"


def test_a_file_at_the_journals_name_is_replaced_by_a_folder(layout):
    os.link(layout.secret, layout.journal)

    source = current_source(layout.folder, layout.metadata, layout.artifact_id)

    assert source["content"] == FIRST
    assert layout.journal.is_dir()
    assert layout.secret.read_text(encoding="utf-8") == SECRET
    assert os.stat(layout.secret).st_nlink == 1


@pytest.mark.parametrize("deployment", ["personal_deployment", "org_deployment"])
def test_the_turn_end_capture_writes_nothing_through_a_linked_journal(layout, tmp_path, request, deployment):
    """In org mode a remote worker runs the agent's code with no sandbox, and
    the server runs this capture when the turn ends."""
    request.getfixturevalue(deployment)
    victim = _victim(tmp_path)
    before = _files_below(victim / ".revisions")
    layout.journal.symlink_to(victim / ".revisions", target_is_directory=True)
    (layout.folder / "brief.md").write_text("# Agent edit\n", encoding="utf-8")

    captured = capture_agent_revision(layout.folder, conversation_id=str(uuid4()))

    assert captured is not None and captured["actor"]["kind"] == "agent"
    assert _files_below(victim / ".revisions") == before
    assert [revision["id"] for revision in list_revisions(layout.folder)] == [captured["id"]]


# ── reads with a bound ───────────────────────────────────────────────────────


def test_a_sparse_journal_file_is_refused_before_it_is_read(layout, monkeypatch):
    """A cell can `truncate` a journal file to any size without writing it."""
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    with (layout.journal / "manifest.json").open("r+b") as stream:
        stream.truncate(MAX_JOURNAL_FILE_BYTES + 1)
    reads: list[int] = []
    monkeypatch.setattr(artifact_journal, "read_bounded", lambda fd, size: reads.append(size) or [])

    with pytest.raises(JournalFileRefused):
        read_journal_file(layout.folder, "manifest.json")
    with pytest.raises(RevisionValidationError, match="unreadable"):
        list_revisions(layout.folder)
    assert reads == []


def test_a_sparse_source_is_refused_before_it_is_read(layout, monkeypatch):
    with (layout.folder / "brief.md").open("r+b") as stream:
        stream.truncate(MAX_SOURCE_BYTES + 1)
    reads: list[int] = []
    monkeypatch.setattr(artifact_revisions, "read_bounded", lambda fd, size: reads.append(size) or [])

    with pytest.raises(RevisionValidationError, match="too large"):
        current_source(layout.folder, layout.metadata, layout.artifact_id)
    assert reads == []


# ── the routes, on a stack where several people sign in ──────────────────────


def _partition_layout(dash: Path, tmp_path: Path) -> Layout:
    """A's artifact `dash`, with A's and B's `.anton/.env` and a file in the
    server's home folder."""
    return Layout(
        folder=dash,
        metadata={},
        artifact_id=_artifact_id(dash),
        secret=_write_secret(_general(ALICE) / ".anton" / ".env", SECRET),
        other_secret=_write_secret(_general(BOB) / ".anton" / ".env", OTHER_SECRET),
        home_file=_write_secret(tmp_path / "home" / "server-secret.txt", HOME_SECRET),
    )


def _workspace(scope, folder: Path) -> str:
    return f"/api/v1/artifacts/workspace/{_general_id(scope)}/{_artifact_id(folder)}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_revision_routes_answer_a_planted_blob_as_a_missing_revision(two_people, tmp_path, case):
    dash, _notes = two_people
    layout = _partition_layout(dash, tmp_path)
    client = _client()
    workspace = _workspace(ALICE, dash)
    opened = client.get(workspace, headers=A)
    assert opened.status_code == 200, opened.text
    head = opened.json()["revision"]
    original = (dash / "index.html").read_text(encoding="utf-8")
    planted = _plant_revision(dash, content_hash=_case_hash(case, layout, real=head["contentHash"]))
    repair_id = _plant_ready_repair(dash, base_revision_id=planted, revision_id=head["id"], path=head["path"])

    answers = [
        client.get(f"{workspace}/revisions/{planted}", headers=A),
        client.post(f"{workspace}/revisions/{planted}/restore", json={"expectedRevisionId": head["id"]}, headers=A),
        client.get(f"{workspace}/agent-repairs/{repair_id}", headers=A),
        client.post(f"{workspace}/agent-repairs/{repair_id}/decision", json={"status": "rejected"}, headers=A),
    ]

    for answer in answers:
        assert (answer.status_code, answer.json()) == MISSING
        assert not any(text in answer.text for text in (*SECRETS, TOKEN))
    served = client.get("/api/v1/artifacts/serve/general/dash/index.html", headers=A)
    assert served.status_code == 200
    assert not any(text in served.text for text in SECRETS)
    assert (dash / "index.html").read_text(encoding="utf-8") == original
    assert client.get(f"{workspace}/revisions/{head['id']}", headers=A).json()["content"] == original


def test_the_routes_and_recovery_refuse_a_hard_linked_blob(two_people, tmp_path):
    dash, _notes = two_people
    layout = _partition_layout(dash, tmp_path)
    client = _client()
    workspace = _workspace(ALICE, dash)
    head = client.get(workspace, headers=A).json()["revision"]
    original = (dash / "index.html").read_text(encoding="utf-8")
    content_hash = _link_as_blob(dash, layout.secret)
    planted = _plant_revision(dash, content_hash=content_hash)

    revision = client.get(f"{workspace}/revisions/{planted}", headers=A)
    restore = client.post(f"{workspace}/revisions/{planted}/restore", json={"expectedRevisionId": head["id"]}, headers=A)
    _plant_pending(
        dash, content_hash=content_hash, path=head["path"], base_revision_id=head["id"], before=original.encode()
    )
    reopened = client.get(workspace, headers=A)
    served = client.get("/api/v1/artifacts/serve/general/dash/index.html", headers=A)

    assert (revision.status_code, revision.json()) == MISSING
    assert (restore.status_code, restore.json()) == MISSING
    assert (reopened.status_code, reopened.json()["content"]) == (200, original)
    assert served.status_code == 200 and SECRET not in served.text
    assert (dash / "index.html").read_text(encoding="utf-8") == original


def test_a_linked_journal_reaches_no_one_elses_revisions_or_comments(two_people):
    """A `.revisions` left as a link to B's artifact's journal, from before the
    sandbox: A reads no journal through it and writes nothing to B's."""
    dash, _notes = two_people
    bobs = _artifact(_general(BOB), "plan", title="Bob's plan")
    client = _client()
    assert client.get(_workspace(BOB, bobs), headers=B).status_code == 200
    bobs_comments = f"/api/v1/artifact-comments/artifact/{_artifact_id(bobs)}/threads"
    assert client.post(bobs_comments, json={"text": "Bob's private note"}, headers=B).status_code == 200
    before = _files_below(bobs / ".revisions")
    (dash / ".revisions").symlink_to(bobs / ".revisions", target_is_directory=True)
    alices_comments = f"/api/v1/artifact-comments/artifact/{_artifact_id(dash)}/threads"

    listed = client.get(alices_comments, headers=A)
    revisions = client.get(f"{_workspace(ALICE, dash)}/revisions", headers=A)
    posted = client.post(alices_comments, json={"text": "Alice's note"}, headers=A)

    assert (listed.status_code, listed.json()["threads"]) == (200, [])
    assert (revisions.status_code, revisions.json()) == (200, {"revisions": []})
    assert posted.status_code == 200
    assert _files_below(bobs / ".revisions") == before
    assert (dash / ".revisions").is_dir() and not (dash / ".revisions").is_symlink()
    assert "Bob's private note" not in client.get(alices_comments, headers=A).text
