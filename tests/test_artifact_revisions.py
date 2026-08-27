from __future__ import annotations

import json
from uuid import UUID

import pytest

from cowork.services import artifact_identity as identity_service
from cowork.services import artifact_revisions as revision_service
from cowork.services.artifact_identity import (
    artifact_key,
    ensure_stable_id,
    resolve_artifact_folder,
)
from cowork.services.artifacts import ProjectArtifacts
from cowork.services.artifact_revisions import (
    RevisionConflict,
    active_agent_repair,
    agent_repair_detail,
    cancel_agent_repair,
    capture_agent_revision,
    create_agent_repair,
    current_source,
    current_workspace,
    finalize_agent_repair,
    list_revisions,
    revision_with_content,
    save_source,
)


@pytest.fixture
def artifact(tmp_path):
    folder = tmp_path / "my-artifact"
    folder.mkdir()
    metadata = {
        "id": "a1b2c3d4",
        "slug": "my-artifact",
        "createdAt": "2026-08-25T12:00:00+00:00",
        "name": "My artifact",
        "type": "document",
        "primary": "brief.md",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "brief.md").write_text("# First\n", encoding="utf-8")
    stable_id, metadata = ensure_stable_id(folder, metadata)
    return folder, metadata, stable_id


def test_legacy_identity_is_persisted_and_is_comment_key(artifact):
    folder, _metadata, stable_id = artifact

    persisted = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))

    assert persisted["stableId"] == stable_id
    assert str(UUID(stable_id)) == stable_id
    assert artifact_key(stable_id) == f"artifact/{stable_id}"


def test_invalid_stable_identity_fails_closed_without_rewriting(tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    metadata = {"id": "legacy", "createdAt": "2026-08-25", "stableId": "not-a-uuid"}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="stable identity is invalid"):
        ensure_stable_id(folder, metadata)

    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8")) == metadata


def test_legacy_identity_backfill_merges_latest_metadata(tmp_path):
    folder = tmp_path / "legacy"
    folder.mkdir()
    stale = {"id": "legacy", "createdAt": "2026-08-25", "name": "Before"}
    latest = {**stale, "name": "After", "description": "Concurrent update"}
    (folder / "metadata.json").write_text(json.dumps(latest), encoding="utf-8")

    stable_id, merged = ensure_stable_id(folder, stale)

    assert merged["stableId"] == stable_id
    assert merged["name"] == "After"
    assert merged["description"] == "Concurrent update"


def test_legacy_identity_backfill_preserves_metadata_mtime(tmp_path):
    """Channel delivery (artifacts_since) reads metadata.json's mtime as "this
    turn touched the artifact". A backfill is bookkeeping, not an update — if it
    refreshed the mtime, the first card build after an upgrade would deliver
    every legacy artifact to the chat as though it were new."""
    import os

    folder = tmp_path / "legacy"
    folder.mkdir()
    metadata = {"id": "legacy", "createdAt": "2026-08-25", "name": "Old"}
    path = folder / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    past_ns = path.stat().st_mtime_ns - 3_600_000_000_000  # one hour ago
    os.utime(path, ns=(past_ns, past_ns))

    ensure_stable_id(folder, metadata)

    assert path.stat().st_mtime_ns == past_ns
    assert json.loads(path.read_text(encoding="utf-8"))["stableId"]


def test_stable_identity_resolution_reuses_the_container_index(artifact, monkeypatch):
    folder, _metadata, stable_id = artifact
    unrelated = folder.parent / "unrelated"
    unrelated.mkdir()
    (unrelated / "metadata.json").write_text(json.dumps({
        "id": "unrelated",
        "createdAt": "2026-08-25T12:01:00+00:00",
        "stableId": "22222222-2222-4222-8222-222222222222",
    }), encoding="utf-8")
    source = ProjectArtifacts(folder.parent, None, "test")
    identity_service._clear_identity_indexes()
    original = identity_service.ensure_stable_id
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(identity_service, "ensure_stable_id", counted)

    assert resolve_artifact_folder([source], stable_id)[1] == folder
    first_lookup_calls = calls
    assert resolve_artifact_folder([source], stable_id)[1] == folder

    assert first_lookup_calls == 3  # two indexed folders + target revalidation
    assert calls == first_lookup_calls + 1  # hot lookup reads only its target


def test_stable_identity_miss_refreshes_existing_folder_metadata(artifact):
    folder, _metadata, _stable_id = artifact
    pending = folder.parent / "pending"
    pending.mkdir()
    source = ProjectArtifacts(folder.parent, None, "test")
    wanted = "33333333-3333-4333-8333-333333333333"
    identity_service._clear_identity_indexes()

    with pytest.raises(FileNotFoundError):
        resolve_artifact_folder([source], wanted)

    # Adding metadata inside `pending` may leave the parent artifacts directory
    # clock untouched; a cached negative result must not hide the new artifact.
    (pending / "metadata.json").write_text(
        json.dumps({
            "id": "pending",
            "createdAt": "2026-08-25T12:02:00+00:00",
            "stableId": wanted,
        }),
        encoding="utf-8",
    )

    assert resolve_artifact_folder([source], wanted)[1] == pending


def test_stable_identity_refreshes_when_cached_identity_moves(artifact):
    folder, _metadata, stable_id = artifact
    replacement = folder.parent / "replacement"
    replacement.mkdir()
    replacement_id = "44444444-4444-4444-8444-444444444444"
    replacement_metadata = {
        "id": "replacement",
        "createdAt": "2026-08-25T12:03:00+00:00",
        "stableId": replacement_id,
    }
    replacement_path = replacement / "metadata.json"
    replacement_path.write_text(json.dumps(replacement_metadata), encoding="utf-8")
    source = ProjectArtifacts(folder.parent, None, "test")
    identity_service._clear_identity_indexes()

    assert resolve_artifact_folder([source], stable_id)[1] == folder

    original_path = folder / "metadata.json"
    original_metadata = json.loads(original_path.read_text(encoding="utf-8"))
    original_path.write_text(
        json.dumps({**original_metadata, "stableId": replacement_id}),
        encoding="utf-8",
    )
    replacement_path.write_text(
        json.dumps({**replacement_metadata, "stableId": stable_id}),
        encoding="utf-8",
    )

    assert resolve_artifact_folder([source], stable_id)[1] == replacement


def test_manual_save_is_atomic_and_records_revision(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)

    saved = save_source(
        folder,
        metadata,
        stable_id,
        content="# Revised\n",
        expected_revision_id=initial["revision"]["id"],
        actor_kind="manual",
        actor_id="user-1",
        summary="Updated title",
    )

    assert (folder / "brief.md").read_text(encoding="utf-8") == "# Revised\n"
    assert saved["revision"]["number"] == 2
    assert saved["revision"]["actor"] == {"kind": "manual", "id": "user-1"}
    assert [r["number"] for r in list_revisions(folder)] == [2, 1]
    assert revision_with_content(folder, initial["revision"]["id"])["content"] == "# First\n"


def test_workspace_snapshot_bundles_source_and_filtered_history(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    save_source(
        folder,
        metadata,
        stable_id,
        content="# Revised\n",
        expected_revision_id=initial["revision"]["id"],
    )

    snapshot = current_workspace(folder, metadata, stable_id)

    assert snapshot["content"] == "# Revised\n"
    assert snapshot["revision"] == snapshot["revisions"][0]
    assert [revision["number"] for revision in snapshot["revisions"]] == [2, 1]


def test_interrupted_save_recovers_source_and_attribution(artifact, monkeypatch):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    write_manifest = revision_service._write_manifest
    failed = False

    def fail_once(folder_arg, payload):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated manifest interruption")
        return write_manifest(folder_arg, payload)

    monkeypatch.setattr(revision_service, "_write_manifest", fail_once)
    with pytest.raises(OSError, match="simulated manifest interruption"):
        save_source(
            folder,
            metadata,
            stable_id,
            content="# Recovered\n",
            expected_revision_id=initial["revision"]["id"],
            actor_kind="manual",
            actor_id="user-1",
            summary="Recovered edit",
        )

    assert (folder / ".revisions" / "pending-source-write.json").is_file()
    recovered = current_source(folder, metadata, stable_id)

    assert recovered["content"] == "# Recovered\n"
    assert recovered["revision"]["actor"] == {"kind": "manual", "id": "user-1"}
    assert recovered["revision"]["summary"] == "Recovered edit"
    assert not (folder / ".revisions" / "pending-source-write.json").exists()


def test_stale_save_returns_conflict_without_overwriting(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    first = save_source(
        folder,
        metadata,
        stable_id,
        content="winner",
        expected_revision_id=initial["revision"]["id"],
    )

    with pytest.raises(RevisionConflict) as exc:
        save_source(
            folder,
            metadata,
            stable_id,
            content="stale overwrite",
            expected_revision_id=initial["revision"]["id"],
        )

    assert exc.value.current["id"] == first["revision"]["id"]
    assert (folder / "brief.md").read_text(encoding="utf-8") == "winner"


def test_out_of_band_change_is_captured_before_conflict(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    (folder / "brief.md").write_text("agent changed this", encoding="utf-8")

    with pytest.raises(RevisionConflict) as exc:
        save_source(
            folder,
            metadata,
            stable_id,
            content="manual change",
            expected_revision_id=initial["revision"]["id"],
        )

    assert exc.value.current["actor"]["kind"] == "system"
    assert revision_with_content(folder, exc.value.current["id"])["content"] == "agent changed this"
    assert (folder / "brief.md").read_text(encoding="utf-8") == "agent changed this"


def test_revision_journal_is_private_housekeeping(artifact):
    folder, metadata, stable_id = artifact
    current_source(folder, metadata, stable_id)

    assert (folder / ".revisions" / "manifest.json").is_file()
    assert not any(p.name.endswith(".tmp") for p in (folder / ".revisions").rglob("*"))


def test_revision_retention_prunes_unreferenced_blobs(artifact):
    folder, metadata, stable_id = artifact
    current = current_source(folder, metadata, stable_id)
    for number in range(82):
        current = save_source(
            folder,
            metadata,
            stable_id,
            content=f"# Revision {number}\n",
            expected_revision_id=current["revision"]["id"],
        )

    assert len(list_revisions(folder)) == 80
    assert len(list((folder / ".revisions" / "blobs").iterdir())) == 80


def test_agent_repair_carries_context_and_requires_compare_before_accept(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    thread = [
        {"author": {"email": "reviewer@example.com"}, "text": "Use a clearer title"},
        {"author": {"email": "owner@example.com"}, "text": "Keep it concise"},
    ]

    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector="h1:nth-of-type(1)",
        thread=thread,
        conversation_id="conversation-1",
    )

    with pytest.raises(ValueError, match="already has an agent repair"):
        create_agent_repair(
            folder,
            metadata,
            stable_id,
            expected_revision_id=initial["revision"]["id"],
            comment_thread_id="thread-2",
            selector=None,
            thread=[{"text": "Another simultaneous change"}],
            conversation_id="conversation-2",
        )

    assert stable_id in requested["prompt"]
    assert initial["revision"]["id"] in requested["prompt"]
    assert "h1:nth-of-type(1)" in requested["prompt"]
    assert "Use a clearer title" in requested["prompt"]
    with pytest.raises(ValueError, match="not ready"):
        finalize_agent_repair(
            folder,
            metadata,
            stable_id,
            requested["repair"]["id"],
            "accepted",
        )

    (folder / "brief.md").write_text("# Clear title\n", encoding="utf-8")
    revision = capture_agent_revision(folder, conversation_id="conversation-1")
    detail = agent_repair_detail(folder, requested["repair"]["id"])

    assert revision["commentThreadIds"] == ["thread-1"]
    assert detail["repair"]["status"] == "ready"
    assert detail["compare"]["before"]["content"] == "# First\n"
    assert detail["compare"]["after"]["content"] == "# Clear title\n"
    with pytest.raises(ValueError, match="awaiting review"):
        create_agent_repair(
            folder,
            metadata,
            stable_id,
            expected_revision_id=detail["repair"]["revisionId"],
            comment_thread_id="thread-2",
            selector=None,
            thread=[{"text": "Do another change"}],
            conversation_id="conversation-2",
        )
    assert finalize_agent_repair(
        folder,
        metadata,
        stable_id,
        requested["repair"]["id"],
        "accepted",
    )["status"] == "accepted"


def test_queued_agent_repair_can_be_cancelled_when_turn_does_not_start(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )

    cancelled = cancel_agent_repair(folder, requested["repair"]["id"])

    assert cancelled["status"] == "cancelled"
    assert cancel_agent_repair(folder, cancelled["id"])["status"] == "cancelled"
    replacement = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "Try again"}],
        conversation_id="conversation-2",
    )
    assert replacement["repair"]["status"] == "queued"


def test_active_agent_repair_survives_viewer_navigation(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Change the title"}],
        conversation_id="conversation-1",
    )

    assert active_agent_repair(folder)["id"] == requested["repair"]["id"]
    (folder / "brief.md").write_text("# Changed\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    assert active_agent_repair(folder)["status"] == "ready"
    finalize_agent_repair(
        folder,
        metadata,
        stable_id,
        requested["repair"]["id"],
        "accepted",
    )
    assert active_agent_repair(folder) is None


def test_reject_agent_repair_restores_source_as_a_new_revision(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Try a different title"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    agent_revision = capture_agent_revision(folder, conversation_id="conversation-1")

    decided = finalize_agent_repair(
        folder,
        metadata,
        stable_id,
        requested["repair"]["id"],
        "rejected",
        actor_id="owner-1",
    )

    assert decided["status"] == "rejected"
    assert (folder / "brief.md").read_text(encoding="utf-8") == "# First\n"
    restored = current_source(folder, metadata, stable_id)["revision"]
    assert restored["baseRevisionId"] == agent_revision["id"]
    assert restored["actor"] == {"kind": "manual", "id": "owner-1"}
    assert restored["commentThreadIds"] == ["thread-1"]


def test_rejected_repair_retry_finishes_interrupted_status_write(artifact, monkeypatch):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Try a different title"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    write_repair = revision_service._write_repair
    failed = False

    def fail_once(folder_arg, repair):
        nonlocal failed
        if not failed and repair.get("status") == "rejected":
            failed = True
            raise OSError("simulated repair status interruption")
        return write_repair(folder_arg, repair)

    monkeypatch.setattr(revision_service, "_write_repair", fail_once)
    with pytest.raises(OSError, match="simulated repair status interruption"):
        finalize_agent_repair(
            folder,
            metadata,
            stable_id,
            requested["repair"]["id"],
            "rejected",
            actor_id="owner-1",
        )

    decided = finalize_agent_repair(
        folder,
        metadata,
        stable_id,
        requested["repair"]["id"],
        "rejected",
        actor_id="owner-1",
    )

    assert decided["status"] == "rejected"
    assert (folder / "brief.md").read_text(encoding="utf-8") == "# First\n"
    assert len([
        revision for revision in list_revisions(folder)
        if revision["summary"] == "Rejected agent suggestion"
    ]) == 1


def test_agent_repair_decision_refuses_a_changed_head(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Try a different title"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    ready = agent_repair_detail(folder, requested["repair"]["id"])["repair"]
    current = current_source(folder, metadata, stable_id)
    save_source(
        folder,
        metadata,
        stable_id,
        content="# Owner follow-up\n",
        expected_revision_id=current["revision"]["id"],
    )

    with pytest.raises(RevisionConflict):
        finalize_agent_repair(
            folder,
            metadata,
            stable_id,
            ready["id"],
            "accepted",
        )

    assert agent_repair_detail(folder, ready["id"])["repair"]["status"] == "ready"


def test_agent_repair_finishes_when_agent_makes_no_change(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please check this"}],
        conversation_id="conversation-1",
    )

    unchanged = capture_agent_revision(folder, conversation_id="conversation-1")
    detail = agent_repair_detail(folder, requested["repair"]["id"])

    assert unchanged["id"] == initial["revision"]["id"]
    assert detail["repair"]["status"] == "no_change"
    assert detail["repair"]["revisionId"] is None


def test_agent_repair_reports_conflict_when_base_moves_during_turn(artifact):
    folder, metadata, stable_id = artifact
    initial = current_source(folder, metadata, stable_id)
    requested = create_agent_repair(
        folder,
        metadata,
        stable_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    save_source(
        folder,
        metadata,
        stable_id,
        content="# Owner update\n",
        expected_revision_id=initial["revision"]["id"],
    )
    (folder / "brief.md").write_text("# Agent update\n", encoding="utf-8")

    agent_revision = capture_agent_revision(folder, conversation_id="conversation-1")
    detail = agent_repair_detail(folder, requested["repair"]["id"])

    assert detail["repair"]["status"] == "conflict"
    assert detail["repair"]["revisionId"] == agent_revision["id"]
    assert agent_revision["commentThreadIds"] == []
