"""An artifact's revision journal reads and writes only its own files.

The journal, `.revisions` in the artifact folder, is a folder the agent's code
can write. A cell locked by the code sandbox can rewrite the manifest, a
pending source write or a repair record, and hard-link any file it can read,
its project's `.anton/.env` included. None of that may make the server read,
write or serve a file outside the journal. A planted blob name answers as a
missing revision does, a hard-linked blob is refused, a repair id names no file
outside `repairs/`, and a journal folder that is a link reads as no journal and
is replaced by a real folder on the first write, with its target untouched.
An open that fails for any other reason is an error, never an empty journal.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from cowork.api.v1.endpoints.artifact_workspace import _not_found_detail
from cowork.services import artifact_journal, artifact_revisions, local_artifact_comments
from cowork.services.artifact_draft_review import draft_review_grant, enable_draft_review
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_journal import (
    MAX_JOURNAL_FILE_BYTES,
    MAX_JOURNAL_RECORD_BYTES,
    JournalFileRefused,
    journal_file_mtime_ns,
    read_journal_file,
    remove_journal_file,
    write_journal_file,
)
from cowork.services.artifact_revision_records import AgentRepair, is_repair_id
from cowork.services.artifact_revisions import (
    MAX_SOURCE_BYTES,
    RevisionConflict,
    RevisionValidationError,
    active_agent_repair,
    agent_repair_detail,
    cancel_agent_repair,
    capture_agent_revision,
    create_agent_repair,
    current_source,
    finalize_agent_repair,
    has_queued_agent_repair,
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
JOURNAL = yaml.safe_load((FIXTURES / "journal_cases.yaml").read_text(encoding="utf-8"))
UNENCODABLE = JOURNAL["values_json_cannot_encode"]
UNENCODABLE_IDS = [case["id"] for case in UNENCODABLE]

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


def _unencodable(case: dict) -> object:
    """A values_json_cannot_encode case's value: empty lists nested ``depth``
    deep, or its ``text``."""
    if "text" in case:
        return case["text"]
    value: list = []
    for _ in range(case["depth"] - 1):
        value = [value]
    return value


def _plant_revision(folder: Path, *, content_hash: object, record: dict | None = None) -> object:
    """Add a revision naming ``content_hash`` before the head, as the agent's
    code can, and return its id. ``record`` fields replace the head's. The head
    stays the server's own."""
    manifest_path = folder / ".revisions" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planted = {**manifest["revisions"][-1], "id": str(uuid4()), "contentHash": content_hash, **(record or {})}
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
def test_a_planted_blob_name_reads_no_file(layout, case, monkeypatch):
    """A name that isn't 64 lowercase hex characters is never opened, not
    only refused by a later check."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    planted = _plant_revision(layout.folder, content_hash=_case_hash(case, layout, real=head["contentHash"]))
    opened: list[tuple[str, ...]] = []
    real_read = artifact_revisions.read_journal_file

    def spy(folder, *parts, **kwargs):
        opened.append(parts)
        return real_read(folder, *parts, **kwargs)

    monkeypatch.setattr(artifact_revisions, "read_journal_file", spy)

    with pytest.raises(FileNotFoundError, match="^Revision not found$"):
        revision_with_content(layout.folder, planted)
    assert not [parts for parts in opened if parts[0] == "blobs"]


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
    """Recovery checks the name before it opens anything, so every planted
    name answers alike and none tells the caller whether a server path
    exists."""
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
    """A save over a planted blob replaces the name with the server's own
    bytes and finishes, so no pending record is left for later requests to
    trip on."""
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
    each back by its id. A record the agent's code wrote can't pick that file
    name, so no `escaped.json` lands in `.anton` or in a folder outside the
    project."""
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
    # Read before any lock: the lock replaces the link with a real folder.
    assert artifact_revisions._read_manifest(layout.folder).revisions == []
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
    write_journal_file(layout.folder, "comments.json", content=b"[]")
    with (layout.journal / "comments.json").open("r+b") as stream:
        stream.truncate(MAX_JOURNAL_FILE_BYTES + 1)
    reads: list[int] = []
    monkeypatch.setattr(artifact_journal, "read_bounded", lambda fd, size: reads.append(size) or [])

    with pytest.raises(JournalFileRefused):
        read_journal_file(layout.folder, "comments.json")
    assert reads == []


def test_a_manifest_past_its_cap_is_refused_before_it_is_read(layout, monkeypatch):
    """Valid JSON padded with spaces past the manifest's cap: a whole-file read
    would parse it, so only the cap refuses it."""
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    manifest = layout.journal / "manifest.json"
    manifest.write_bytes(manifest.read_bytes().ljust(artifact_revisions._MAX_MANIFEST_BYTES + 1))
    assert json.loads(manifest.read_bytes())["revisions"]
    reads: list[int] = []
    monkeypatch.setattr(artifact_journal, "read_bounded", lambda fd, size: reads.append(size) or [])

    with pytest.raises(RevisionValidationError, match="unreadable"):
        list_revisions(layout.folder)
    assert reads == []


def test_a_pending_write_past_its_cap_is_abandoned_unread(layout):
    """The record names a real blob and is valid JSON padded with spaces, so a
    read would finish it and write the blob into the source."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    recovered = b"# Recovered\n"
    write_journal_file(layout.folder, "blobs", _sha(recovered), content=recovered)
    pending = _plant_pending(
        layout.folder,
        content_hash=_sha(recovered),
        path=head["path"],
        base_revision_id=head["id"],
        before=FIRST.encode(),
    )
    pending.write_bytes(pending.read_bytes().ljust(MAX_JOURNAL_RECORD_BYTES + 1))

    reread = current_source(layout.folder, layout.metadata, layout.artifact_id)

    assert reread["content"] == FIRST
    assert not pending.exists()


def test_a_sparse_source_is_refused_before_it_is_read(layout, monkeypatch):
    with (layout.folder / "brief.md").open("r+b") as stream:
        stream.truncate(MAX_SOURCE_BYTES + 1)
    reads: list[int] = []
    monkeypatch.setattr(artifact_revisions, "read_bounded", lambda fd, size: reads.append(size) or [])

    with pytest.raises(RevisionValidationError, match="too large"):
        current_source(layout.folder, layout.metadata, layout.artifact_id)
    assert reads == []


# ── an open that couldn't be made ────────────────────────────────────────────


def _open_fails(monkeypatch, failure: dict) -> dict:
    """Make opening the folder ``failure["at"]`` fail with its errno while the
    returned switch is on. The switch starts off."""
    code = getattr(errno, failure["errno"])
    switch = {"on": False}
    if failure["at"] == "journal_folder":
        real_open = artifact_journal.open_pinned_child

        def open_child(parent, name, **kwargs):
            if switch["on"] and name == ".revisions":
                raise OSError(code, os.strerror(code))
            return real_open(parent, name, **kwargs)

        monkeypatch.setattr(artifact_journal, "open_pinned_child", open_child)
    else:
        real_pin = artifact_journal.pinned_dir

        @contextmanager
        def pin(base, **kwargs):
            if switch["on"]:
                raise OSError(code, os.strerror(code))
            with real_pin(base, **kwargs) as pinned:
                yield pinned

        monkeypatch.setattr(artifact_journal, "pinned_dir", pin)
    return switch


def _failing_inside(monkeypatch, switch: dict, module, name: str) -> None:
    """Turn the switch on only while ``module.name`` runs, so the open fails
    in that read and nowhere else, the lock's own open included."""
    real = getattr(module, name)

    def failing(*args, **kwargs):
        switch["on"] = True
        try:
            return real(*args, **kwargs)
        finally:
            switch["on"] = False

    monkeypatch.setattr(module, name, failing)


def _comments(layout: Layout, monkeypatch) -> local_artifact_comments.LocalArtifactComments:
    """The local comment service for ``layout``'s artifact, found without a
    scan of the server's artifacts roots."""
    monkeypatch.setattr(local_artifact_comments, "artifacts_sources_for_scan", lambda **_kwargs: [])
    monkeypatch.setattr(
        local_artifact_comments, "resolve_artifact_folder", lambda _sources, _id: (None, layout.folder, None)
    )
    return local_artifact_comments.LocalArtifactComments(layout.artifact_id, scope=None)


def _history(layout: Layout) -> dict:
    """Four revisions and two comments: the journal an empty read would lose.
    Returns the head revision."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    for number in range(2, 5):
        head = save_source(
            layout.folder,
            layout.metadata,
            layout.artifact_id,
            content=f"# Version {number}\n",
            expected_revision_id=head["id"],
        )["revision"]
    threads = [{"id": "t1", "status": "open"}, {"id": "t2", "status": "open"}]
    write_journal_file(layout.folder, "comments.json", content=json.dumps(threads).encode())
    return head


def _write_after_a_read(operation: str, layout: Layout, head: dict, monkeypatch) -> None:
    if operation == "open_editor":
        current_source(layout.folder, layout.metadata, layout.artifact_id)
    elif operation == "save":
        (layout.folder / "brief.md").write_text("# Outside edit\n", encoding="utf-8")
        save_source(
            layout.folder,
            layout.metadata,
            layout.artifact_id,
            content="# Mine\n",
            expected_revision_id=head["id"],
        )
    elif operation == "turn_end_capture":
        # The capture answers None for any failure, so the turn still ends.
        (layout.folder / "brief.md").write_text("# Agent edit\n", encoding="utf-8")
        assert capture_agent_revision(layout.folder, conversation_id=str(uuid4())) is None
    elif operation == "comment_create":
        _comments(layout, monkeypatch).create({"text": "New note"})
    else:
        raise AssertionError(f"unknown operation {operation}")


@pytest.mark.parametrize("operation", JOURNAL["writers_after_a_read"])
@pytest.mark.parametrize(
    "failure", JOURNAL["transient_open_failures"], ids=[case["id"] for case in JOURNAL["transient_open_failures"]]
)
def test_an_open_that_fails_never_reads_as_an_empty_journal(layout, monkeypatch, operation, failure):
    """Read as empty, the next write would replace the whole history or every
    comment with one entry."""
    head = _history(layout)
    before = _files_below(layout.journal)
    switch = _open_fails(monkeypatch, failure)
    _failing_inside(monkeypatch, switch, artifact_revisions, "read_journal_json")
    _failing_inside(monkeypatch, switch, local_artifact_comments, "read_journal_json")

    if operation == "turn_end_capture":
        _write_after_a_read(operation, layout, head, monkeypatch)
    else:
        with pytest.raises((OSError, RevisionValidationError, HTTPException)):
            _write_after_a_read(operation, layout, head, monkeypatch)

    assert _files_below(layout.journal) == before


@pytest.mark.parametrize(
    "failure", JOURNAL["transient_open_failures"], ids=[case["id"] for case in JOURNAL["transient_open_failures"]]
)
def test_recovery_keeps_a_pending_write_whose_blob_it_could_not_read(layout, monkeypatch, failure):
    """A real crash leaves the blob and the record; a failed read is no reason
    to drop the record, and the next request finishes it."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    recovered = b"# Recovered\n"
    write_journal_file(layout.folder, "blobs", _sha(recovered), content=recovered)
    pending = _plant_pending(
        layout.folder,
        content_hash=_sha(recovered),
        path=head["path"],
        base_revision_id=head["id"],
        before=FIRST.encode(),
    )
    switch = _open_fails(monkeypatch, failure)
    _failing_inside(monkeypatch, switch, artifact_revisions, "read_journal_file")

    with pytest.raises(OSError):
        current_source(layout.folder, layout.metadata, layout.artifact_id)
    assert pending.exists()

    monkeypatch.undo()
    assert current_source(layout.folder, layout.metadata, layout.artifact_id)["content"] == recovered.decode()
    assert not pending.exists()


# ── a journal file that is a link ────────────────────────────────────────────


@pytest.mark.parametrize(
    "case", JOURNAL["linked_journal_files"], ids=[case["id"] for case in JOURNAL["linked_journal_files"]]
)
def test_a_journal_file_that_is_a_link_is_refused_and_replaced(layout, tmp_path, monkeypatch, caplog, case):
    """Left from before the sandbox, or made by a remote worker: no read goes
    through it, the lockless readers included, and the next write replaces the
    link, never its target."""
    victim = _victim(tmp_path)
    write_journal_file(victim, "pending-source-write.json", content=b'{"version": 1}')
    victims_repair = next((victim / ".revisions" / "repairs").glob("*.json")).stem
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    parts = tuple(case["name"].replace("{repair}", victims_repair).split("/"))
    link = layout.journal.joinpath(*parts)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(victim / ".revisions" / Path(*parts))
    before = _files_below(victim / ".revisions")

    with pytest.raises(JournalFileRefused):
        read_journal_file(layout.folder, *parts)
    if case["id"] == "manifest":
        with pytest.raises(RevisionValidationError, match="unreadable"):
            artifact_revisions._read_manifest(layout.folder)
    elif case["id"] == "comments":
        with pytest.raises(HTTPException) as refused:
            _comments(layout, monkeypatch).list_payload("all")
        assert refused.value.status_code == 500
    elif case["id"] == "draft-review-grant":
        assert draft_review_grant(layout.folder) is None
    elif case["id"] == "repair":
        assert active_agent_repair(layout.folder) is None
    elif case["id"] == "pending-source-write":
        with caplog.at_level(logging.WARNING, logger=artifact_revisions.logger.name):
            current_source(layout.folder, layout.metadata, layout.artifact_id)
        reasons = [
            record.reason for record in caplog.records
            if record.name == artifact_revisions.logger.name
            and getattr(record, "event", None) == "artifact_pending_write_abandoned"
        ]
        assert reasons == ["unreadable"]
    write_journal_file(layout.folder, *parts, content=b"{}")

    assert link.is_file() and not link.is_symlink()
    assert _files_below(victim / ".revisions") == before


@pytest.mark.parametrize("writer", JOURNAL["first_writers_on_a_linked_journal"])
def test_the_first_writer_on_a_linked_journal_replaces_the_link(layout, tmp_path, monkeypatch, writer):
    """Each writer runs before anything else has replaced the link, so none
    relies on another's lock having done it."""
    victim = _victim(tmp_path)
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    shutil.rmtree(layout.journal)
    layout.journal.symlink_to(victim / ".revisions", target_is_directory=True)
    before = _files_below(victim / ".revisions")

    if writer == "draft_review_grant":
        enable_draft_review(layout.folder, org_id="org-2", enabled_by="alice")
    elif writer == "comment":
        _comments(layout, monkeypatch).create({"text": "Alice's note"})
    elif writer == "agent_repair":
        with pytest.raises(RevisionConflict):
            create_agent_repair(
                layout.folder,
                layout.metadata,
                layout.artifact_id,
                expected_revision_id=head["id"],
                comment_thread_id="thread-2",
                selector=None,
                thread=[{"text": "Fix the title"}],
                conversation_id=str(uuid4()),
            )
    elif writer == "save":
        with pytest.raises(RevisionConflict):
            save_source(
                layout.folder,
                layout.metadata,
                layout.artifact_id,
                content="# Edited\n",
                expected_revision_id=head["id"],
            )
    else:
        raise AssertionError(f"unknown writer {writer}")

    assert _files_below(victim / ".revisions") == before
    assert layout.journal.is_dir() and not layout.journal.is_symlink()


def test_an_artifact_folder_that_is_a_link_has_no_journal(layout, tmp_path):
    victim = _victim(tmp_path)
    before = _files_below(victim / ".revisions")
    linked = layout.folder.parent / "linked"
    linked.symlink_to(victim, target_is_directory=True)

    assert read_journal_file(linked, "comments.json") is None
    assert draft_review_grant(linked) is None
    assert active_agent_repair(linked) is None
    with pytest.raises(OSError):
        write_journal_file(linked, "comments.json", content=b"[]")
    assert _files_below(victim / ".revisions") == before


# ── journal files the server never wrote ─────────────────────────────────────


def test_a_blob_whose_bytes_do_not_match_its_name_is_refused(layout):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    (layout.journal / "blobs" / head["contentHash"]).write_text("# Rewritten by the agent\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="^Revision not found$"):
        revision_with_content(layout.folder, head["id"])


def test_a_blob_past_the_source_cap_answers_as_missing(layout):
    """Its bytes match its name, so only the cap refuses it."""
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    large = b"x" * (MAX_SOURCE_BYTES + 1)
    (layout.journal / "blobs" / _sha(large)).write_bytes(large)
    planted = _plant_revision(layout.folder, content_hash=_sha(large))

    with pytest.raises(FileNotFoundError, match="^Revision not found$"):
        revision_with_content(layout.folder, planted)


def test_a_save_over_a_source_past_the_cap_is_refused(layout):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    (layout.folder / "brief.md").write_bytes(b"x" * (MAX_SOURCE_BYTES + 1))

    with pytest.raises(RevisionValidationError, match="too large"):
        save_source(layout.folder, layout.metadata, layout.artifact_id, content=FIRST, expected_revision_id=head["id"])


def test_a_journal_file_with_a_second_name_is_refused_and_logged(layout, tmp_path, caplog):
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    os.link(layout.journal / "manifest.json", tmp_path / "manifest-backup.json")

    with caplog.at_level(logging.WARNING, logger=artifact_journal.logger.name):
        with pytest.raises(RevisionValidationError, match="unreadable"):
            list_revisions(layout.folder)

    [record] = [record for record in caplog.records if record.name == artifact_journal.logger.name]
    assert (record.event, record.reason, record.artifact_slug) == (
        "artifact_journal_file_refused",
        "second_name",
        "report",
    )


@pytest.mark.parametrize(
    "case", JOURNAL["malformed_pending_writes"], ids=[case["id"] for case in JOURNAL["malformed_pending_writes"]]
)
def test_recovery_abandons_a_pending_write_the_server_never_writes(layout, case):
    """Each record names a real blob, so a recovery that trusted it would write
    the blob into the source."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    recovered = b"# Recovered\n"
    write_journal_file(layout.folder, "blobs", _sha(recovered), content=recovered)
    pending = _plant_pending(
        layout.folder,
        content_hash=_sha(recovered),
        path=head["path"],
        base_revision_id=head["id"],
        before=FIRST.encode(),
    )
    if "raw" in case:
        pending.write_text(case["raw"], encoding="utf-8")
    elif "nested" in case:
        pending.write_text("[" * case["nested"], encoding="utf-8")
    else:
        record = json.loads(pending.read_text(encoding="utf-8"))
        overrides = case["record"]
        record["version"] = overrides.get("version", record["version"])
        record["path"] = overrides.get("path", record["path"])
        record["revision"]["path"] = overrides.get("revision_path", record["revision"]["path"])
        record["revision"]["baseRevisionId"] = overrides.get(
            "base_revision_id", record["revision"]["baseRevisionId"]
        )
        pending.write_text(json.dumps(record), encoding="utf-8")

    reread = current_source(layout.folder, layout.metadata, layout.artifact_id)

    assert reread["content"] == FIRST
    assert (layout.folder / "brief.md").read_text(encoding="utf-8") == FIRST
    assert not pending.exists()


# ── repair records ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("repair_id", "canonical"),
    [(repair_id, True) for repair_id in JOURNAL["repair_ids"]["canonical"]]
    + [(repair_id, False) for repair_id in JOURNAL["repair_ids"]["refused"]],
)
def test_only_a_canonical_uuid_names_a_repair(repair_id, canonical):
    assert is_repair_id(repair_id) is canonical


@pytest.mark.parametrize("repair_id", JOURNAL["repair_ids"]["refused"])
def test_a_repair_record_refuses_an_id_the_server_never_mints(repair_id):
    """The file-name filter, the read and the write each check the id too; this
    pins the model's own check, which every repair the server holds passes."""
    with pytest.raises(ValidationError):
        AgentRepair.model_validate({"id": repair_id, "path": "brief.md", "status": "queued"})


@pytest.mark.parametrize("repair_id", JOURNAL["repair_ids"]["refused"])
def test_a_record_named_by_a_non_canonical_id_is_left_alone_at_turn_end(layout, repair_id):
    """Its id matches its file name, so only the UUID check skips it. Loaded,
    its write-back would be refused and the turn's edit would get no
    revision."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    repairs = layout.journal / "repairs"
    repairs.mkdir()
    record = {
        "id": repair_id,
        "status": "queued",
        "conversationId": "turn-1",
        "path": "other.md",
        "baseRevisionId": head["id"],
        "createdAt": "2026-11-02T10:00:00+00:00",
    }
    planted = repairs / f"{repair_id}.json"
    planted.write_text(json.dumps(record), encoding="utf-8")
    (layout.folder / "brief.md").write_text("# Agent edit\n", encoding="utf-8")

    captured = capture_agent_revision(layout.folder, conversation_id="turn-1")

    assert captured is not None and captured["actor"]["kind"] == "agent"
    assert json.loads(planted.read_text(encoding="utf-8")) == record


def test_a_deeply_nested_repair_record_is_skipped(layout):
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    depth = next(case["nested"] for case in JOURNAL["malformed_pending_writes"] if "nested" in case)
    repairs = layout.journal / "repairs"
    repairs.mkdir()
    (repairs / f"{uuid4()}.json").write_text("[" * depth, encoding="utf-8")
    (layout.folder / "brief.md").write_text("# Agent edit\n", encoding="utf-8")

    assert has_queued_agent_repair(layout.folder, "turn-1") is False
    captured = capture_agent_revision(layout.folder, conversation_id="turn-1")
    assert captured is not None and captured["actor"]["kind"] == "agent"


def test_repair_records_are_read_one_at_a_time(layout):
    """A scan holds one parsed record at a time, so a pile of large records the
    agent's code planted costs the server one record's memory, not the pile's."""
    pile = JOURNAL["repair_pile"]
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    repairs = layout.journal / "repairs"
    repairs.mkdir()
    padding = ",".join(["{}"] * pile["objects_per_record"])

    def plant() -> None:
        for _ in range(pile["records_per_pile"]):
            repair_id = str(uuid4())
            (repairs / f"{repair_id}.json").write_text(
                f'{{"id": "{repair_id}", "status": "done", "pad": [{padding}]}}', encoding="utf-8"
            )

    def peak() -> int:
        tracemalloc.start()
        try:
            assert has_queued_agent_repair(layout.folder, "turn-1") is False
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    plant()
    one_pile = peak()
    plant()
    two_piles = peak()

    assert two_piles < one_pile * pile["peak_growth_bound"]


def test_a_scan_holds_no_second_copy_of_a_record_that_passes_its_model(layout):
    """Each read checks that the record can be written back, which must not
    cost a second copy of it: a large repair the agent's code planted costs a
    scan about what parsing it does."""
    large = JOURNAL["large_repair"]
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    repairs = layout.journal / "repairs"
    repairs.mkdir()
    repair_id = str(uuid4())
    padding = ",".join(["{}"] * large["objects_per_record"])
    text = f'{{"id": "{repair_id}", "path": "brief.md", "status": "ready", "thread": [{padding}]}}'
    (repairs / f"{repair_id}.json").write_text(text, encoding="utf-8")

    def peak(work) -> int:
        tracemalloc.start()
        try:
            work()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    parsed = peak(lambda: json.loads(text))
    scanned = peak(lambda: has_queued_agent_repair(layout.folder, "turn-1"))

    assert active_agent_repair(layout.folder)["id"] == repair_id
    assert scanned < parsed * large["peak_over_parse_bound"]


# ── records that fail their model ────────────────────────────────────────────


def _filled(record: dict, layout: Layout, *, real: str) -> dict:
    """A YAML case's record fields with their placeholders filled in."""
    return {key: _case_hash({"content_hash": value}, layout, real=real) for key, value in record.items()}


def _skipped_entry_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == artifact_revisions.logger.name
        and record.levelno == logging.WARNING
        and getattr(record, "event", None) == "artifact_revision_entries_skipped"
    ]


def _assert_left_out_then_dropped(layout: Layout, *, head: dict, planted: object, caplog) -> None:
    """The planted entry is left out of the history, with one warning, and the
    next write of the manifest drops it."""
    with caplog.at_level(logging.WARNING, logger=artifact_revisions.logger.name):
        listed = list_revisions(layout.folder)
    warnings = _skipped_entry_warnings(caplog)
    saved = save_source(
        layout.folder, layout.metadata, layout.artifact_id, content="# Second\n", expected_revision_id=head["id"]
    )["revision"]

    assert [revision["id"] for revision in listed] == [head["id"]]
    assert len(warnings) == 1
    manifest = json.loads((layout.journal / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in manifest["revisions"]] == [head["id"], saved["id"]]
    assert planted not in {entry["id"] for entry in manifest["revisions"]}


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_a_planted_revision_is_left_out_of_the_history_and_dropped_on_the_next_write(layout, case, caplog):
    """RevisionEntry refuses the name before any reader sees it. _blob_bytes
    refuses the same names again where they would open a file, so this is the
    test that fails when the model's check goes."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    planted = _plant_revision(layout.folder, content_hash=_case_hash(case, layout, real=head["contentHash"]))

    _assert_left_out_then_dropped(layout, head=head, planted=planted, caplog=caplog)


@pytest.mark.parametrize(
    "case",
    JOURNAL["revisions_failing_their_model"],
    ids=[case["id"] for case in JOURNAL["revisions_failing_their_model"]],
)
def test_a_revision_that_fails_its_model_on_another_field_is_left_out_and_dropped(layout, case, caplog):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    planted = _plant_revision(layout.folder, content_hash=head["contentHash"], record=case["record"])

    _assert_left_out_then_dropped(layout, head=head, planted=planted, caplog=caplog)


@pytest.mark.parametrize("case", UNENCODABLE, ids=UNENCODABLE_IDS)
def test_a_revision_holding_a_value_json_cannot_encode_is_left_out_and_dropped(layout, case, caplog):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    planted = _plant_revision(
        layout.folder, content_hash=head["contentHash"], record={"note": _unencodable(case)}
    )

    _assert_left_out_then_dropped(layout, head=head, planted=planted, caplog=caplog)


@pytest.mark.parametrize("case", UNENCODABLE, ids=UNENCODABLE_IDS)
def test_a_manifest_holding_a_value_json_cannot_encode_reads_as_invalid(layout, case):
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    manifest_path = layout.journal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps({**manifest, "note": _unencodable(case)}), encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(RevisionValidationError, match="^Artifact revision history is invalid$"):
        list_revisions(layout.folder)
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    "case",
    JOURNAL["pending_writes_failing_their_model"],
    ids=[case["id"] for case in JOURNAL["pending_writes_failing_their_model"]],
)
def test_recovery_abandons_a_pending_write_that_fails_its_model(layout, case):
    """The source already holds the revision's bytes, so a recovery that
    accepted the record would add its revision to the history."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    recovered = b"# Recovered\n"
    write_journal_file(layout.folder, "blobs", _sha(recovered), content=recovered)
    pending = _plant_pending(
        layout.folder,
        content_hash=_sha(recovered),
        path=head["path"],
        base_revision_id=head["id"],
        before=FIRST.encode(),
    )
    record = {**json.loads(pending.read_text(encoding="utf-8")), **_filled(case["record"], layout, real=head["contentHash"])}
    pending.write_text(json.dumps(record), encoding="utf-8")
    (layout.folder / "brief.md").write_bytes(recovered)

    listed = list_revisions(layout.folder)

    assert [revision["id"] for revision in listed] == [head["id"]]
    assert not pending.exists()


@pytest.mark.parametrize(
    "case",
    JOURNAL["repairs_failing_their_model"],
    ids=[case["id"] for case in JOURNAL["repairs_failing_their_model"]],
)
def test_a_repair_that_fails_its_model_is_left_alone_and_answers_as_missing(layout, case):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    repair_id = str(uuid4())
    repairs = layout.journal / "repairs"
    repairs.mkdir()
    record = {
        "id": repair_id,
        "path": head["path"],
        "baseRevisionId": head["id"],
        "commentThreadId": "thread-1",
        "conversationId": "turn-1",
        "status": "queued",
        "revisionId": None,
        "createdAt": "2026-11-02T10:00:00+00:00",
        "updatedAt": "2026-11-02T10:00:00+00:00",
        **case["record"],
    }
    planted = repairs / f"{repair_id}.json"
    planted.write_text(json.dumps(record), encoding="utf-8")
    (layout.folder / "brief.md").write_text("# Agent edit\n", encoding="utf-8")

    captured = capture_agent_revision(layout.folder, conversation_id="turn-1")

    assert captured is not None and captured["actor"]["kind"] == "agent"
    assert json.loads(planted.read_text(encoding="utf-8")) == record
    with pytest.raises(FileNotFoundError, match="^Agent repair not found$"):
        agent_repair_detail(layout.folder, repair_id)


@pytest.mark.parametrize(
    "case",
    JOURNAL["repairs_failing_their_model"],
    ids=[case["id"] for case in JOURNAL["repairs_failing_their_model"]],
)
def test_a_repair_refuses_a_value_set_after_the_read(case):
    repair = AgentRepair.model_validate({"id": str(uuid4()), "path": "brief.md", "status": "queued"})

    with pytest.raises(ValidationError):
        for field, value in case["record"].items():
            setattr(repair, field, value)


@pytest.mark.parametrize("case", UNENCODABLE, ids=UNENCODABLE_IDS)
def test_a_pending_write_holding_a_value_json_cannot_encode_is_abandoned(layout, case):
    """A recovery that accepted the record would write the source, then fail
    to write the manifest and keep the record, on every later read."""
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    recovered = b"# Recovered\n"
    write_journal_file(layout.folder, "blobs", _sha(recovered), content=recovered)
    pending = _plant_pending(
        layout.folder,
        content_hash=_sha(recovered),
        path=head["path"],
        base_revision_id=head["id"],
        before=FIRST.encode(),
    )
    record = json.loads(pending.read_text(encoding="utf-8"))
    record["revision"]["note"] = _unencodable(case)
    pending.write_text(json.dumps(record), encoding="utf-8")

    reread = current_source(layout.folder, layout.metadata, layout.artifact_id)

    assert reread["content"] == FIRST
    assert (layout.folder / "brief.md").read_text(encoding="utf-8") == FIRST
    assert not pending.exists()


@pytest.mark.parametrize("case", UNENCODABLE, ids=UNENCODABLE_IDS)
def test_a_repair_holding_a_value_json_cannot_encode_is_left_alone_and_answers_as_missing(layout, case):
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    repair_id = _plant_ready_repair(
        layout.folder, base_revision_id=head["id"], revision_id=head["id"], path=head["path"]
    )
    planted = layout.journal / "repairs" / f"{repair_id}.json"
    record = json.loads(planted.read_text(encoding="utf-8"))
    planted.write_text(json.dumps({**record, "thread": [_unencodable(case)]}), encoding="utf-8")
    before = planted.read_bytes()

    assert active_agent_repair(layout.folder) is None
    with pytest.raises(FileNotFoundError, match="^Agent repair not found$"):
        agent_repair_detail(layout.folder, repair_id)
    with pytest.raises(FileNotFoundError, match="^Agent repair not found$"):
        cancel_agent_repair(layout.folder, repair_id, discard_ready=True)
    assert planted.read_bytes() == before


def test_a_revision_another_build_wrote_keeps_its_keys_through_a_write(layout):
    other = JOURNAL["records_from_another_build"]
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    manifest_path = layout.journal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    written = {
        **{key: value for key, value in manifest["revisions"][-1].items() if key not in other["revision_without"]},
        **other["undeclared"],
    }
    manifest["revisions"][-1] = written
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    listed = list_revisions(layout.folder)
    save_source(
        layout.folder, layout.metadata, layout.artifact_id, content="# Second\n", expected_revision_id=head["id"]
    )

    kept = json.loads(manifest_path.read_text(encoding="utf-8"))["revisions"][0]
    assert list(listed[0].items()) == list(written.items())
    assert list(kept.items()) == list(written.items())


def test_a_repair_another_build_wrote_keeps_its_keys_through_a_write(layout):
    other = JOURNAL["records_from_another_build"]
    head = current_source(layout.folder, layout.metadata, layout.artifact_id)["revision"]
    repair_id = create_agent_repair(
        layout.folder,
        layout.metadata,
        layout.artifact_id,
        expected_revision_id=head["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Fix the title"}],
        conversation_id=str(uuid4()),
    )["repair"]["id"]
    path = layout.journal / "repairs" / f"{repair_id}.json"
    written = {
        **{
            key: value
            for key, value in json.loads(path.read_text(encoding="utf-8")).items()
            if key not in other["repair_without"]
        },
        **other["undeclared"],
    }
    path.write_text(json.dumps(written), encoding="utf-8")

    cancelled = cancel_agent_repair(layout.folder, repair_id)
    detail = agent_repair_detail(layout.folder, repair_id)["repair"]

    kept = json.loads(path.read_text(encoding="utf-8"))
    expected = {**written, "status": "cancelled", "updatedAt": kept["updatedAt"]}
    assert list(kept.items()) == list(expected.items())
    assert list(cancelled.items()) == list(expected.items())
    assert detail == {**expected, "superseded": False}


# ── names ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", JOURNAL["names_with_more_than_one_part"])
def test_the_journal_refuses_a_name_with_more_than_one_part(layout, name):
    current_source(layout.folder, layout.metadata, layout.artifact_id)
    calls = [
        lambda: read_journal_file(layout.folder, name),
        lambda: read_journal_file(layout.folder, name, "manifest.json"),
        lambda: write_journal_file(layout.folder, name, content=b"{}"),
        lambda: remove_journal_file(layout.folder, name),
        lambda: journal_file_mtime_ns(layout.folder, name),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="direct-child"):
            call()


def test_a_404_carries_no_server_path():
    """An OS error names the path it failed on; the service's own messages
    don't."""
    gone = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "/srv/cowork/alice/.anton/artifacts/report")

    assert _not_found_detail(gone) == "Artifact not found"
    assert _not_found_detail(FileNotFoundError("Revision not found")) == "Revision not found"


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
