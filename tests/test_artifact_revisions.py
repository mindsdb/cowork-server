from __future__ import annotations

import json
from uuid import UUID

import pytest

from cowork.services import artifact_identity as identity_service
from cowork.services import artifact_revisions as revision_service
from cowork.services.artifact_identity import (
    artifact_key,
    ensure_full_id,
    resolve_artifact_folder,
)
from cowork.services.artifacts import ProjectArtifacts
from cowork.services.artifact_revisions import (
    RepairAlreadyPending,
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
    release_repairs_for_comment,
    revision_with_content,
    save_source,
)


def _full_id(value: str) -> str:
    """A dashed UUID spelled the way `metadata.json` stores an id."""
    return UUID(value).hex


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
    artifact_id, metadata = ensure_full_id(folder, metadata)
    return folder, metadata, artifact_id


def test_legacy_identity_is_widened_persisted_and_is_comment_key(artifact):
    folder, _metadata, artifact_id = artifact

    persisted = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))

    assert persisted["id"] == artifact_id
    assert UUID(artifact_id).hex == artifact_id
    # The old eight characters stay the prefix, so `<name>-<id[:8]>` folders
    # keep addressing the same artifact after the widening.
    assert artifact_id[:8] == "a1b2c3d4"
    assert artifact_key(artifact_id) == f"artifact/{UUID(artifact_id)}"


def test_widening_matches_antons_in_memory_derivation(artifact):
    """anton never persists the widened id, so the two derivations have to agree
    or a turn and a card build would mint different identities."""
    from anton.core.artifacts.models import Artifact

    folder, metadata, artifact_id = artifact
    legacy = {
        **metadata,
        "id": "a1b2c3d4",
        "description": "",
        "updatedAt": metadata["createdAt"],
    }

    assert Artifact.model_validate(legacy).id == artifact_id


def test_persisted_stable_id_is_adopted_as_the_artifact_id(tmp_path):
    """Records from the two-field era already keyed published versions, auth
    rules and comment threads by `stableId`; re-deriving would orphan them."""
    folder = tmp_path / "two-field"
    folder.mkdir()
    minted = "55555555-5555-4555-8555-555555555555"
    metadata = {"id": "a1b2c3d4", "createdAt": "2026-08-25", "stableId": minted}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    artifact_id, merged = ensure_full_id(folder, metadata)

    assert artifact_id == _full_id(minted)
    assert merged["id"] == artifact_id
    assert "stableId" not in merged
    assert "stableId" not in json.loads(
        (folder / "metadata.json").read_text(encoding="utf-8")
    )


def test_id_less_metadata_widens_from_its_long_slug(tmp_path):
    """A record with no `id` falls back to the slug, and slugs run to 64
    characters. Reading that width as "malformed UUID" would raise, and
    `card_for_folder` drops an artifact whose identity raises — the artifact
    would vanish from every listing instead of getting an identity."""
    folder = tmp_path / "q3-launch-readiness-for-the-emea-region-rollout"
    folder.mkdir()
    metadata = {"slug": folder.name, "createdAt": "2026-08-25", "name": "Q3"}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    artifact_id, merged = ensure_full_id(folder, metadata)

    assert UUID(artifact_id).hex == artifact_id
    assert merged["id"] == artifact_id
    # Same value on a second, independent read.
    assert ensure_full_id(folder)[0] == artifact_id


def test_invalid_identity_fails_closed_without_rewriting(tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    metadata = {"id": "d0d1d2d3", "createdAt": "2026-08-25", "stableId": "not-a-uuid"}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="identity is invalid"):
        ensure_full_id(folder, metadata)

    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8")) == metadata


def test_legacy_identity_widening_merges_latest_metadata(tmp_path):
    folder = tmp_path / "legacy"
    folder.mkdir()
    stale = {"id": "b0b1b2b3", "createdAt": "2026-08-25", "name": "Before"}
    latest = {**stale, "name": "After", "description": "Concurrent update"}
    (folder / "metadata.json").write_text(json.dumps(latest), encoding="utf-8")

    artifact_id, merged = ensure_full_id(folder, stale)

    assert merged["id"] == artifact_id
    assert merged["name"] == "After"
    assert merged["description"] == "Concurrent update"


def test_legacy_identity_widening_preserves_metadata_mtime(tmp_path):
    """Channel delivery (artifacts_since) reads metadata.json's mtime as "this
    turn touched the artifact". Widening an id is bookkeeping, not an update —
    if it refreshed the mtime, the first card build after an upgrade would
    deliver every legacy artifact to the chat as though it were new."""
    import os

    folder = tmp_path / "legacy"
    folder.mkdir()
    metadata = {"id": "c0c1c2c3", "createdAt": "2026-08-25", "name": "Old"}
    path = folder / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    past_ns = path.stat().st_mtime_ns - 3_600_000_000_000  # one hour ago
    os.utime(path, ns=(past_ns, past_ns))

    artifact_id, _merged = ensure_full_id(folder, metadata)

    assert path.stat().st_mtime_ns == past_ns
    assert json.loads(path.read_text(encoding="utf-8"))["id"] == artifact_id


def test_a_symlinked_metadata_is_not_read_through(tmp_path):
    """An artifact folder is agent-writable, so `metadata.json` can be swapped
    for a link. Following it would make identity resolution read a file of the
    writer's choosing; refusing reads as "this folder has no usable identity",
    which is what every caller already handles."""
    import pytest as _pytest

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"id": "f" * 32, "createdAt": "2026-08-25"}), encoding="utf-8")
    folder = tmp_path / "planted"
    folder.mkdir()
    (folder / "metadata.json").symlink_to(outside)

    with _pytest.raises(OSError):
        ensure_full_id(folder)


def test_widening_does_not_retimestamp_a_symlink_target(tmp_path):
    """The mtime restore is the one part of the write a planted link could still
    steer: `os.replace` acts on the link, but `os.utime` would follow it."""
    import os

    outside = tmp_path / "outside.txt"
    outside.write_text("untouched", encoding="utf-8")
    target_ns = outside.stat().st_mtime_ns - 7_200_000_000_000  # two hours ago
    os.utime(outside, ns=(target_ns, target_ns))

    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "metadata.json").symlink_to(outside)
    metadata = {"id": "c0c1c2c3", "createdAt": "2026-08-25", "name": "Old"}

    # Metadata supplied by the caller, so the read is skipped and the write path
    # runs against the link — the shape this guard is about.
    try:
        ensure_full_id(folder, metadata)
    except (OSError, ValueError):
        pass

    assert outside.read_text(encoding="utf-8") == "untouched"
    assert outside.stat().st_mtime_ns == target_ns


def test_identity_resolution_reuses_the_container_index(artifact, monkeypatch):
    folder, _metadata, artifact_id = artifact
    unrelated = folder.parent / "unrelated"
    unrelated.mkdir()
    (unrelated / "metadata.json").write_text(json.dumps({
        "id": _full_id("22222222-2222-4222-8222-222222222222"),
        "createdAt": "2026-08-25T12:01:00+00:00",
    }), encoding="utf-8")
    source = ProjectArtifacts(folder.parent, None, "test")
    identity_service._clear_identity_indexes()
    original = identity_service.ensure_full_id
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(identity_service, "ensure_full_id", counted)

    assert resolve_artifact_folder([source], artifact_id)[1] == folder
    first_lookup_calls = calls
    assert resolve_artifact_folder([source], artifact_id)[1] == folder

    assert first_lookup_calls == 3  # two indexed folders + target revalidation
    assert calls == first_lookup_calls + 1  # hot lookup reads only its target


def test_identity_revalidation_receives_the_server_index_value(monkeypatch, tmp_path):
    """A request UUID only matches the index in memory; the filesystem-facing
    validator receives the distinct string object supplied by that index."""
    server_id = _full_id("77777777-7777-4777-8777-777777777777")
    request_id = (server_id + "x")[:-1]
    folder = tmp_path / "artifact"
    source = ProjectArtifacts(tmp_path, None, "test")
    seen = []

    monkeypatch.setattr(
        identity_service,
        "_indexed_identities",
        lambda _source, **_kwargs: ((server_id,), False),
    )

    def revalidate(_source, indexed_id):
        seen.append(indexed_id)
        return ((folder, {"id": server_id}),)

    monkeypatch.setattr(identity_service, "_indexed_artifacts", revalidate)

    assert resolve_artifact_folder([source], request_id)[1] == folder
    assert seen == [server_id]
    assert seen[0] is server_id


def test_identity_miss_refreshes_existing_folder_metadata(artifact):
    folder, _metadata, _artifact_id = artifact
    pending = folder.parent / "pending"
    pending.mkdir()
    source = ProjectArtifacts(folder.parent, None, "test")
    wanted = _full_id("33333333-3333-4333-8333-333333333333")
    identity_service._clear_identity_indexes()

    with pytest.raises(FileNotFoundError):
        resolve_artifact_folder([source], wanted)

    # Adding metadata inside `pending` may leave the parent artifacts directory
    # clock untouched; a cached negative result must not hide the new artifact.
    (pending / "metadata.json").write_text(
        json.dumps({
            "id": wanted,
            "createdAt": "2026-08-25T12:02:00+00:00",
        }),
        encoding="utf-8",
    )

    assert resolve_artifact_folder([source], wanted)[1] == pending


def test_identity_refreshes_when_cached_identity_moves(artifact):
    folder, _metadata, artifact_id = artifact
    replacement = folder.parent / "replacement"
    replacement.mkdir()
    replacement_id = _full_id("44444444-4444-4444-8444-444444444444")
    replacement_metadata = {
        "id": replacement_id,
        "createdAt": "2026-08-25T12:03:00+00:00",
    }
    replacement_path = replacement / "metadata.json"
    replacement_path.write_text(json.dumps(replacement_metadata), encoding="utf-8")
    source = ProjectArtifacts(folder.parent, None, "test")
    identity_service._clear_identity_indexes()

    assert resolve_artifact_folder([source], artifact_id)[1] == folder

    original_path = folder / "metadata.json"
    original_metadata = json.loads(original_path.read_text(encoding="utf-8"))
    original_path.write_text(
        json.dumps({**original_metadata, "id": replacement_id}),
        encoding="utf-8",
    )
    replacement_path.write_text(
        json.dumps({**replacement_metadata, "id": artifact_id}),
        encoding="utf-8",
    )

    assert resolve_artifact_folder([source], artifact_id)[1] == replacement


def test_manual_save_is_atomic_and_records_revision(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)

    saved = save_source(
        folder,
        metadata,
        artifact_id,
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
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    save_source(
        folder,
        metadata,
        artifact_id,
        content="# Revised\n",
        expected_revision_id=initial["revision"]["id"],
    )

    snapshot = current_workspace(folder, metadata, artifact_id)

    assert snapshot["content"] == "# Revised\n"
    assert snapshot["revision"] == snapshot["revisions"][0]
    assert [revision["number"] for revision in snapshot["revisions"]] == [2, 1]


def test_interrupted_save_recovers_source_and_attribution(artifact, monkeypatch):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
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
            artifact_id,
            content="# Recovered\n",
            expected_revision_id=initial["revision"]["id"],
            actor_kind="manual",
            actor_id="user-1",
            summary="Recovered edit",
        )

    assert (folder / ".revisions" / "pending-source-write.json").is_file()
    recovered = current_source(folder, metadata, artifact_id)

    assert recovered["content"] == "# Recovered\n"
    assert recovered["revision"]["actor"] == {"kind": "manual", "id": "user-1"}
    assert recovered["revision"]["summary"] == "Recovered edit"
    assert not (folder / ".revisions" / "pending-source-write.json").exists()


def test_stale_save_returns_conflict_without_overwriting(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    first = save_source(
        folder,
        metadata,
        artifact_id,
        content="winner",
        expected_revision_id=initial["revision"]["id"],
    )

    with pytest.raises(RevisionConflict) as exc:
        save_source(
            folder,
            metadata,
            artifact_id,
            content="stale overwrite",
            expected_revision_id=initial["revision"]["id"],
        )

    assert exc.value.current["id"] == first["revision"]["id"]
    assert (folder / "brief.md").read_text(encoding="utf-8") == "winner"


def test_out_of_band_change_is_captured_before_conflict(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    (folder / "brief.md").write_text("agent changed this", encoding="utf-8")

    with pytest.raises(RevisionConflict) as exc:
        save_source(
            folder,
            metadata,
            artifact_id,
            content="manual change",
            expected_revision_id=initial["revision"]["id"],
        )

    assert exc.value.current["actor"]["kind"] == "system"
    assert revision_with_content(folder, exc.value.current["id"])["content"] == "agent changed this"
    assert (folder / "brief.md").read_text(encoding="utf-8") == "agent changed this"


def test_revision_journal_is_private_housekeeping(artifact):
    folder, metadata, artifact_id = artifact
    current_source(folder, metadata, artifact_id)

    assert (folder / ".revisions" / "manifest.json").is_file()
    assert not any(p.name.endswith(".tmp") for p in (folder / ".revisions").rglob("*"))


def test_revision_retention_prunes_unreferenced_blobs(artifact):
    folder, metadata, artifact_id = artifact
    current = current_source(folder, metadata, artifact_id)
    for number in range(82):
        current = save_source(
            folder,
            metadata,
            artifact_id,
            content=f"# Revision {number}\n",
            expected_revision_id=current["revision"]["id"],
        )

    assert len(list_revisions(folder)) == 80
    assert len(list((folder / ".revisions" / "blobs").iterdir())) == 80


def test_agent_repair_carries_context_and_requires_compare_before_accept(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    thread = [
        {"author": {"email": "reviewer@example.com"}, "text": "Use a clearer title"},
        {"author": {"email": "owner@example.com"}, "text": "Keep it concise"},
    ]

    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
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
            artifact_id,
            expected_revision_id=initial["revision"]["id"],
            comment_thread_id="thread-2",
            selector=None,
            thread=[{"text": "Another simultaneous change"}],
            conversation_id="conversation-2",
        )

    assert artifact_id in requested["prompt"]
    assert initial["revision"]["id"] in requested["prompt"]
    assert "h1:nth-of-type(1)" in requested["prompt"]
    assert "Use a clearer title" in requested["prompt"]
    with pytest.raises(ValueError, match="not ready"):
        finalize_agent_repair(
            folder,
            metadata,
            artifact_id,
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
            artifact_id,
            expected_revision_id=detail["repair"]["revisionId"],
            comment_thread_id="thread-2",
            selector=None,
            thread=[{"text": "Do another change"}],
            conversation_id="conversation-2",
        )
    assert finalize_agent_repair(
        folder,
        metadata,
        artifact_id,
        requested["repair"]["id"],
        "accepted",
    )["status"] == "accepted"


def test_superseded_ready_repair_stops_blocking_but_stays_decidable(artifact):
    """Once the owner edits past the agent's revision the suggestion can no
    longer gate the artifact, but it still holds real agent work, so it keeps
    its `ready` status and remains acceptable."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    agent_revision = capture_agent_revision(folder, conversation_id="conversation-1")
    assert agent_repair_detail(folder, requested["repair"]["id"])["repair"]["status"] == "ready"

    save_source(
        folder,
        metadata,
        artifact_id,
        content="# Owner title\n",
        expected_revision_id=agent_revision["id"],
    )

    replacement = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "A second review"}],
        conversation_id="conversation-2",
    )

    assert replacement["repair"]["status"] == "queued"
    assert agent_repair_detail(folder, requested["repair"]["id"])["repair"]["status"] == "ready"


def test_repair_whose_primary_drifted_is_finished_not_stranded(artifact):
    """capture_agent_revision reconciles against the primary it resolves now.
    If metadata["primary"] moved after the handoff was minted, the repair sits
    on a path this capture will never look at, and nothing else can finish it."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    assert requested["repair"]["path"] == "brief.md"

    # The turn repoints the artifact at a different editable file.
    (folder / "summary.md").write_text("# Summary\n", encoding="utf-8")
    moved = {**metadata, "primary": "summary.md"}
    (folder / "metadata.json").write_text(json.dumps(moved), encoding="utf-8")

    capture_agent_revision(folder, conversation_id="conversation-1")

    stranded = json.loads(
        (folder / ".revisions" / "repairs" / f"{requested['repair']['id']}.json")
        .read_text(encoding="utf-8")
    )
    assert stranded["status"] == "conflict"


def test_editing_during_a_turn_does_not_block_the_next_repair(artifact):
    """A queued repair whose base has moved can only land on conflict, so it
    should not hold the path until the TTL expires - an owner editing while a
    turn runs is a normal action, not a wedge."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    save_source(
        folder,
        metadata,
        artifact_id,
        content="# Owner edit mid-turn\n",
        expected_revision_id=initial["revision"]["id"],
    )

    replacement = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "A second review"}],
        conversation_id="conversation-2",
    )

    assert replacement["repair"]["status"] == "queued"
    stranded = json.loads(
        (folder / ".revisions" / "repairs" / f"{requested['repair']['id']}.json")
        .read_text(encoding="utf-8")
    )
    assert stranded["status"] == "conflict"


def test_an_unreadable_timestamp_does_not_block_forever(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    record_path = folder / ".revisions" / "repairs" / f"{requested['repair']['id']}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["createdAt"] = "not-a-date"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    replacement = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "A second review"}],
        conversation_id="conversation-2",
    )

    assert replacement["repair"]["status"] == "queued"


def test_a_superseded_repair_stops_being_the_active_one(artifact):
    """Nothing moves a superseded repair on, so without a preference it stays
    the artifact's active repair for good and its notice never clears."""
    folder, metadata, artifact_id = artifact
    first = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "First review"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent one\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    save_source(
        folder,
        metadata,
        artifact_id,
        content="# Owner\n",
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
    )
    assert active_agent_repair(folder, "brief.md")["id"] == first["repair"]["id"]

    second = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "Second review"}],
        conversation_id="conversation-2",
    )
    (folder / "brief.md").write_text("# Agent two\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-2")

    # The one the owner can act on wins while it is live.
    active = active_agent_repair(folder, "brief.md")
    assert active["id"] == second["repair"]["id"]
    assert active["superseded"] is False


def test_stale_queued_repair_stops_blocking_and_is_finished(artifact):
    """A turn killed between minting the handoff and starting the agent used to
    gate the path forever, because only capture_agent_revision could finish a
    queued repair."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    stranded = folder / ".revisions" / "repairs" / f"{requested['repair']['id']}.json"
    record = json.loads(stranded.read_text(encoding="utf-8"))
    record["createdAt"] = "2020-01-01T00:00:00+00:00"
    stranded.write_text(json.dumps(record), encoding="utf-8")

    replacement = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "A second review"}],
        conversation_id="conversation-2",
    )

    assert replacement["repair"]["status"] == "queued"
    assert json.loads(stranded.read_text(encoding="utf-8"))["status"] == "no_change"


def test_queued_agent_repair_can_be_cancelled_when_turn_does_not_start(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
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
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "Try again"}],
        conversation_id="conversation-2",
    )
    assert replacement["repair"]["status"] == "queued"


def test_ready_repair_can_be_discarded_and_frees_the_path(artifact):
    """Accept and reject were the only exits from ready, so an owner who had
    already dealt with the feedback another way had no way out at all."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    repair_id = requested["repair"]["id"]
    assert agent_repair_detail(folder, repair_id)["repair"]["status"] == "ready"

    # An old client posting no intent keeps the queued-only refusal.
    with pytest.raises(ValueError, match="Only a queued agent repair"):
        cancel_agent_repair(folder, repair_id)

    discarded = cancel_agent_repair(folder, repair_id, discard_ready=True)
    assert discarded["status"] == "discarded"
    assert cancel_agent_repair(folder, repair_id, discard_ready=True)["status"] == "discarded"

    replacement = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "A second review"}],
        conversation_id="conversation-2",
    )
    assert replacement["repair"]["status"] == "queued"


def test_resolving_a_comment_releases_its_ready_repair(artifact):
    """Resolving the comment is the obvious thing to do once the change has
    been eyeballed, and it used to strand the repair in ready forever."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    assert agent_repair_detail(folder, requested["repair"]["id"])["repair"]["status"] == "ready"

    released = release_repairs_for_comment(folder, "thread-1")

    assert [r["status"] for r in released] == ["discarded"]
    replacement = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-2",
        selector=None,
        thread=[{"text": "A second review"}],
        conversation_id="conversation-2",
    )
    assert replacement["repair"]["status"] == "queued"


def test_releasing_leaves_other_threads_and_decided_repairs_alone(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )

    assert release_repairs_for_comment(folder, "thread-other") == []
    assert agent_repair_detail(folder, requested["repair"]["id"])["repair"]["status"] == "queued"

    # A queued repair is left to its own turn: capture_agent_revision only
    # reconciles records that are still queued, so finishing one here would let
    # the agent's edit land with nothing tracking it and no comparison to
    # review it in.
    assert release_repairs_for_comment(folder, "thread-1") == []
    assert agent_repair_detail(folder, requested["repair"]["id"])["repair"]["status"] == "queued"


def test_resolving_mid_turn_leaves_the_agents_edit_reviewable(artifact):
    """Resolving while the turn is still running must not finish the repair:
    capture_agent_revision only reconciles queued records, so the edit would
    land with no ready state and no comparison behind it."""
    folder, metadata, artifact_id = artifact
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )

    release_repairs_for_comment(folder, "thread-1")

    # The turn finishes afterwards, as it would in the product.
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")

    detail = agent_repair_detail(folder, requested["repair"]["id"])
    assert detail["repair"]["status"] == "ready"
    assert detail["compare"]["after"]["content"] == "# Agent title\n"


def test_blocked_repair_names_the_comment_it_is_waiting_on(artifact):
    """The guard used to answer a bare string, so the viewer could not offer a
    way out of the state it described."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )

    with pytest.raises(RepairAlreadyPending) as excinfo:
        create_agent_repair(
            folder,
            metadata,
            artifact_id,
            expected_revision_id=initial["revision"]["id"],
            comment_thread_id="thread-2",
            selector=None,
            thread=[{"text": "Another change"}],
            conversation_id="conversation-2",
        )

    assert excinfo.value.repair["id"] == requested["repair"]["id"]
    assert excinfo.value.repair["commentThreadId"] == "thread-1"


def test_active_agent_repair_survives_viewer_navigation(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
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
        artifact_id,
        requested["repair"]["id"],
        "accepted",
    )
    assert active_agent_repair(folder) is None


def test_reject_agent_repair_restores_source_as_a_new_revision(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
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
        artifact_id,
        requested["repair"]["id"],
        "rejected",
        actor_id="owner-1",
    )

    assert decided["status"] == "rejected"
    assert (folder / "brief.md").read_text(encoding="utf-8") == "# First\n"
    restored = current_source(folder, metadata, artifact_id)["revision"]
    assert restored["baseRevisionId"] == agent_revision["id"]
    assert restored["actor"] == {"kind": "manual", "id": "owner-1"}
    assert restored["commentThreadIds"] == ["thread-1"]


def test_rejected_repair_retry_finishes_interrupted_status_write(artifact, monkeypatch):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
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
            artifact_id,
            requested["repair"]["id"],
            "rejected",
            actor_id="owner-1",
        )

    decided = finalize_agent_repair(
        folder,
        metadata,
        artifact_id,
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


@pytest.fixture
def repair_behind_a_moved_head(artifact):
    """A ready repair whose revision the owner has since edited past - the
    state every accept and reject used to fail in, permanently."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Try a different title"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    ready = agent_repair_detail(folder, requested["repair"]["id"])["repair"]
    save_source(
        folder,
        metadata,
        artifact_id,
        content="# Owner follow-up\n",
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
    )
    head = current_source(folder, metadata, artifact_id)["revision"]["id"]
    return folder, metadata, artifact_id, ready, head


def test_detail_reports_supersession_so_the_viewer_need_not_infer_it(artifact):
    """A caller that infers supersession from its own copy of head cannot tell
    a repair the artifact moved past from one whose revision is simply newer
    than the copy it holds. Both routes answer the same computed flag."""
    folder, metadata, artifact_id = artifact
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")

    # The agent's revision IS head here, so nothing has superseded it.
    assert agent_repair_detail(folder, requested["repair"]["id"])["repair"]["superseded"] is False

    save_source(
        folder,
        metadata,
        artifact_id,
        content="# Owner title\n",
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
    )

    assert agent_repair_detail(folder, requested["repair"]["id"])["repair"]["superseded"] is True


def test_active_repair_reports_whether_the_artifact_moved_past_it(repair_behind_a_moved_head):
    """The viewer decides whether to open the comparison from this flag, so it
    has to distinguish a pending decision from one the artifact overtook."""
    folder, _metadata, _artifact_id, ready, _head = repair_behind_a_moved_head

    active = active_agent_repair(folder, ready["path"])

    assert active["id"] == ready["id"]
    assert active["status"] == "ready"
    assert active["superseded"] is True


def test_active_repair_ignores_other_paths(artifact):
    """The create guard is path filtered, so a repair on another file must not
    reach the viewer opening this one."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )

    assert active_agent_repair(folder, "brief.md")["commentThreadId"] == "thread-1"
    assert active_agent_repair(folder, "other.md") is None
    assert active_agent_repair(folder)["commentThreadId"] == "thread-1"


def test_accept_survives_a_moved_head(repair_behind_a_moved_head):
    """Accepting keeps the agent's revision, which is already in history; it
    writes no content, so head having moved does not make it unsafe - as long
    as the owner confirmed against the head they were shown."""
    folder, metadata, artifact_id, ready, head = repair_behind_a_moved_head

    decided = finalize_agent_repair(
        folder, metadata, artifact_id, ready["id"], "accepted",
        expected_head_revision_id=head,
    )

    assert decided["status"] == "accepted"
    # The owner's later edit is untouched.
    assert (folder / "brief.md").read_text(encoding="utf-8") == "# Owner follow-up\n"


def test_accept_is_idempotent(repair_behind_a_moved_head):
    """A lost response or a double-click must not report a failure for a
    decision that already landed."""
    folder, metadata, artifact_id, ready, head = repair_behind_a_moved_head
    finalize_agent_repair(
        folder, metadata, artifact_id, ready["id"], "accepted",
        expected_head_revision_id=head,
    )

    again = finalize_agent_repair(folder, metadata, artifact_id, ready["id"], "accepted")

    assert again["status"] == "accepted"


def test_accept_refuses_a_superseded_repair_the_owner_did_not_confirm(
    repair_behind_a_moved_head,
):
    """Accepting resolves the review comment, so it must not happen against a
    head the owner never saw."""
    folder, metadata, artifact_id, ready, _head = repair_behind_a_moved_head

    with pytest.raises(RevisionConflict, match="changed after the agent's edit"):
        finalize_agent_repair(folder, metadata, artifact_id, ready["id"], "accepted")

    assert agent_repair_detail(folder, ready["id"])["repair"]["status"] == "ready"


def test_accept_refuses_when_the_owner_undid_the_agents_work(artifact):
    """Restoring the pre-agent content leaves the agent's revision in history,
    so "still in the manifest" is not enough on its own: accepting would close
    the review comment with the change not applied."""
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Try a different title"}],
        conversation_id="conversation-1",
    )
    (folder / "brief.md").write_text("# Agent title\n", encoding="utf-8")
    capture_agent_revision(folder, conversation_id="conversation-1")
    ready = agent_repair_detail(folder, requested["repair"]["id"])["repair"]
    save_source(
        folder,
        metadata,
        artifact_id,
        content="# First\n",
        expected_revision_id=current_source(folder, metadata, artifact_id)["revision"]["id"],
    )

    with pytest.raises(RevisionConflict, match="changed after the agent's edit"):
        finalize_agent_repair(folder, metadata, artifact_id, ready["id"], "accepted")

    assert (folder / "brief.md").read_text(encoding="utf-8") == "# First\n"
    assert agent_repair_detail(folder, ready["id"])["repair"]["status"] == "ready"


def test_accept_refuses_when_the_agent_revision_left_history(repair_behind_a_moved_head):
    """History is pruned past MAX_REVISIONS, so there is a real case where the
    agent's revision is gone and there is nothing left to keep."""
    folder, metadata, artifact_id, ready, _head = repair_behind_a_moved_head
    manifest_path = folder / ".revisions" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revisions"] = [
        entry for entry in manifest["revisions"] if entry["id"] != ready["revisionId"]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no longer in this artifact's history"):
        finalize_agent_repair(folder, metadata, artifact_id, ready["id"], "accepted")


def test_reject_refuses_a_head_the_user_did_not_confirm(repair_behind_a_moved_head):
    """Rejecting restores the pre-agent content over whatever is there now, so
    it stays guarded - including against a head that moved after the confirm."""
    folder, metadata, artifact_id, ready, _head = repair_behind_a_moved_head

    with pytest.raises(RevisionConflict):
        finalize_agent_repair(
            folder,
            metadata,
            artifact_id,
            ready["id"],
            "rejected",
            expected_head_revision_id="a-head-that-moved-on",
        )

    assert agent_repair_detail(folder, ready["id"])["repair"]["status"] == "ready"
    assert (folder / "brief.md").read_text(encoding="utf-8") == "# Owner follow-up\n"


def test_reject_restores_when_the_user_confirmed_the_current_head(repair_behind_a_moved_head):
    folder, metadata, artifact_id, ready, head = repair_behind_a_moved_head

    decided = finalize_agent_repair(
        folder,
        metadata,
        artifact_id,
        ready["id"],
        "rejected",
        expected_head_revision_id=head,
    )

    assert decided["status"] == "rejected"
    assert (folder / "brief.md").read_text(encoding="utf-8") == "# First\n"


def test_reject_without_a_confirmed_head_keeps_the_strict_rule(repair_behind_a_moved_head):
    """An older client sends no head, and must keep refusing rather than
    silently restoring over an edit its user never saw."""
    folder, metadata, artifact_id, ready, _head = repair_behind_a_moved_head

    with pytest.raises(RevisionConflict):
        finalize_agent_repair(folder, metadata, artifact_id, ready["id"], "rejected")


def test_agent_repair_finishes_when_agent_makes_no_change(artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
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
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)
    requested = create_agent_repair(
        folder,
        metadata,
        artifact_id,
        expected_revision_id=initial["revision"]["id"],
        comment_thread_id="thread-1",
        selector=None,
        thread=[{"text": "Please change this"}],
        conversation_id="conversation-1",
    )
    save_source(
        folder,
        metadata,
        artifact_id,
        content="# Owner update\n",
        expected_revision_id=initial["revision"]["id"],
    )
    (folder / "brief.md").write_text("# Agent update\n", encoding="utf-8")

    agent_revision = capture_agent_revision(folder, conversation_id="conversation-1")
    detail = agent_repair_detail(folder, requested["repair"]["id"])

    assert detail["repair"]["status"] == "conflict"
    assert detail["repair"]["revisionId"] == agent_revision["id"]
    assert agent_revision["commentThreadIds"] == []
