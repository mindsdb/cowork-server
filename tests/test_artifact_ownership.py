"""Who owns an artifact in organization mode (ENG-2961).

Project-level roots (ENG-2056) are shared by every member of a project, so the
owner of an artifact there is a server-written attribution row. Legacy
per-conversation roots keep path-derived ownership.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services import artifact_ownership as ownership
from cowork.services import task_objects
from cowork.services.artifacts import ProjectArtifacts


def _engine():
    return get_engine(get_app_settings().database.uri)


@contextmanager
def scoped(org_id: str | None, user_id: str | None = None):
    with Session(_engine()) as raw:
        if org_id is None:
            yield ScopedSession(raw, LOCAL_SCOPE)
        else:
            yield ScopedSession(
                raw, TenantScope(org_mode=True, org_id=org_id, user_id=user_id)
            )


def make_project(tmp_path: Path, org_id: str | None) -> Project:
    path = tmp_path / f"proj-{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    project = Project(id=uuid4(), name=path.name, path=str(path), org_id=org_id)
    with Session(_engine()) as raw:
        raw.add(project)
        raw.commit()
        raw.refresh(project)
        raw.expunge(project)
    return project


def make_conversation(project: Project, owner: str | None) -> UUID:
    conversation_id = uuid4()
    with Session(_engine()) as raw:
        raw.add(
            Conversation(
                id=conversation_id,
                topic="task",
                project_id=project.id,
                org_id=project.org_id,
                created_by=owner,
            )
        )
        raw.commit()
    return conversation_id


def project_root(project: Project) -> ProjectArtifacts:
    parts = (".anton", "artifacts")
    base = Path(project.path).joinpath(*parts)
    base.mkdir(parents=True, exist_ok=True)
    return ProjectArtifacts(
        base=base,
        project_id=str(project.id),
        project_name=project.name,
        trusted_anchor=Path(project.path),
        root_parts=parts,
    )


def legacy_root(project: Project, conversation_part: str) -> ProjectArtifacts:
    parts = ("conversations", conversation_part, ".anton", "artifacts")
    base = Path(project.path).joinpath(*parts)
    base.mkdir(parents=True, exist_ok=True)
    return ProjectArtifacts(
        base=base,
        project_id=str(project.id),
        project_name=project.name,
        trusted_anchor=Path(project.path),
        root_parts=parts,
    )


@pytest.fixture
def org_id() -> str:
    return str(uuid4())


@pytest.fixture
def users() -> tuple[str, str]:
    return str(uuid4()), str(uuid4())


pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


def test_resource_key_is_plain_when_it_fits_and_hashed_when_it_does_not():
    project_id = uuid4()
    assert ownership.artifact_resource_key(project_id, "report") == f"{project_id}/report"
    long_slug = "x" * 255
    key = ownership.artifact_resource_key(project_id, long_slug)
    assert key.startswith(f"{project_id}/sha256:")
    assert len(key) <= 255
    assert key == ownership.artifact_resource_key(project_id, long_slug)


def test_project_root_without_a_row_is_unknown(tmp_path, org_id, users):
    source = project_root(make_project(tmp_path, org_id))
    with scoped(org_id, users[0]) as session:
        resolution = ownership.resolve_artifact_owner(session, source, "report")
    assert resolution == ownership.OwnerResolution(None, "unknown")
    assert resolution.unknown


def test_recorded_owner_resolves_and_first_writer_wins(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    with scoped(org_id) as session:
        assert ownership.record_artifact_owner(
            session, project.id, "report", users[0], action="create"
        ) == users[0]
    with scoped(org_id) as session:
        assert ownership.record_artifact_owner(
            session, project.id, "report", users[1], action="backfill"
        ) == users[0]
    with scoped(org_id, users[1]) as session:
        assert ownership.resolve_artifact_owner(session, source, "report") == (
            ownership.OwnerResolution(users[0], "recorded")
        )


def test_record_with_an_empty_owner_writes_nothing(tmp_path, org_id):
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        assert ownership.record_artifact_owner(
            session, project.id, "report", None, action="create"
        ) is None
        assert ownership.resolve_artifact_owner(
            session, project_root(project), "report"
        ).unknown


def test_a_255_character_slug_is_recorded(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    slug = "s" * 255
    with scoped(org_id) as session:
        ownership.record_artifact_owner(session, project.id, slug, users[0], action="create")
        assert ownership.resolve_artifact_owner(
            session, project_root(project), slug
        ).owner_user_id == users[0]


def test_owner_rows_are_invisible_to_another_org(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        ownership.record_artifact_owner(session, project.id, "report", users[0], action="create")
    with scoped(str(uuid4()), users[0]) as session:
        assert ownership.resolve_artifact_owner(
            session, project_root(project), "report"
        ).unknown


def test_legacy_root_resolves_from_its_conversation(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, users[0])
    source = legacy_root(project, str(conversation_id))
    with scoped(org_id, users[0]) as session:
        assert ownership.resolve_artifact_owner(session, source, "old") == (
            ownership.OwnerResolution(users[0], "legacy_path")
        )


def test_legacy_root_naming_another_projects_conversation_is_unknown(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    other = make_project(tmp_path, org_id)
    conversation_id = make_conversation(other, users[0])
    source = legacy_root(project, str(conversation_id))
    with scoped(org_id, users[0]) as session:
        assert ownership.resolve_artifact_owner(session, source, "old").unknown


def test_malformed_legacy_root_is_unknown_without_raising(tmp_path, org_id, users):
    source = legacy_root(make_project(tmp_path, org_id), "not-a-uuid")
    with scoped(org_id, users[0]) as session:
        assert ownership.resolve_artifact_owner(session, source, "old").unknown


def test_local_mode_returns_the_scope_user(tmp_path):
    project = make_project(tmp_path, None)
    with scoped(None) as session:
        resolution = ownership.resolve_artifact_owner(session, project_root(project), "a")
    assert resolution == ownership.OwnerResolution(None, "local")
    assert not resolution.unknown


def test_batch_resolution_matches_single_resolution(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    with scoped(org_id) as session:
        ownership.record_artifact_owner(session, project.id, "a", users[0], action="create")
        ownership.record_artifact_owner(session, project.id, "b", users[1], action="create")
    with scoped(org_id, users[0]) as session:
        batch = ownership.resolve_artifact_owners(session, source, ["a", "b", "c"])
    assert batch == {
        "a": ownership.OwnerResolution(users[0], "recorded"),
        "b": ownership.OwnerResolution(users[1], "recorded"),
        "c": ownership.OwnerResolution(None, "unknown"),
    }


def test_rekey_moves_ownership_to_the_new_project(tmp_path, org_id, users):
    source_project = make_project(tmp_path, org_id)
    dest_project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        ownership.record_artifact_owner(session, source_project.id, "a", users[0], action="create")
    with scoped(org_id, users[0]) as session:
        assert ownership.rekey_artifact_owner(
            session, project_root(source_project), "a", dest_project.id, "a-2",
            actor_id=users[0],
        ) is True
        assert ownership.resolve_artifact_owner(
            session, project_root(dest_project), "a-2"
        ).owner_user_id == users[0]
        assert ownership.resolve_artifact_owner(
            session, project_root(source_project), "a"
        ).unknown


def test_forget_removes_ownership_only_for_project_roots(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, users[0])
    with scoped(org_id) as session:
        ownership.record_artifact_owner(session, project.id, "a", users[0], action="create")
    with scoped(org_id, users[0]) as session:
        # A legacy artifact with the same slug must not drop the project-root row.
        assert ownership.forget_artifact_owner(
            session, legacy_root(project, str(conversation_id)), "a", actor_id=users[0]
        ) is False
        assert ownership.forget_artifact_owner(
            session, project_root(project), "a", actor_id=users[0]
        ) is True
        assert ownership.resolve_artifact_owner(session, project_root(project), "a").unknown


def write_artifact(base: Path, slug: str, conversation=None) -> Path:
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text("<html></html>")
    meta = {"id": uuid4().hex, "slug": slug, "type": "html-app"}
    if conversation is not None:
        meta["provenance"] = [{"conversation": str(conversation), "turns": []}]
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


def test_provenance_origin_reads_the_first_conversation(tmp_path):
    conversation = uuid4()
    folder = write_artifact(tmp_path, "a", conversation)
    assert ownership.provenance_origin(folder) == conversation


@pytest.mark.parametrize("raw", ["{ not json", json.dumps({"provenance": "x"}), json.dumps([])])
def test_provenance_origin_of_unreadable_metadata_is_none(tmp_path, raw):
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "metadata.json").write_text(raw)
    assert ownership.provenance_origin(folder) is None


def test_provenance_origin_refuses_a_symlinked_metadata_file(tmp_path):
    real = write_artifact(tmp_path / "elsewhere", "real", uuid4())
    folder = tmp_path / "a"
    folder.mkdir()
    os.symlink(real / "metadata.json", folder / "metadata.json")
    assert ownership.provenance_origin(folder) is None


def test_turn_created_slugs_keeps_only_this_turns_own_artifacts(tmp_path):
    conversation, sibling = uuid4(), uuid4()
    write_artifact(tmp_path, "old", conversation)
    before = {"old"}
    write_artifact(tmp_path, "mine", str(conversation).upper())  # spelling differs
    write_artifact(tmp_path, "sibling", sibling)
    write_artifact(tmp_path, "handmade")  # no provenance
    assert ownership.turn_created_slugs(tmp_path, before, conversation) == {"mine"}


def test_unfinished_turn_keeps_unattributed_but_not_foreign_artifacts(tmp_path, caplog):
    # A Stop between anton's metadata write and its provenance append leaves a
    # folder with no provenance; only a non-clean exit may claim it.
    conversation, sibling = uuid4(), uuid4()
    write_artifact(tmp_path, "mine", conversation)
    write_artifact(tmp_path, "half_written")  # no provenance
    write_artifact(tmp_path, "sibling", sibling)
    with caplog.at_level("INFO", logger=ownership.__name__):
        kept = ownership.turn_created_slugs(
            tmp_path, set(), conversation, accept_unattributed=True
        )
    assert kept == {"mine", "half_written"}
    assert "artifact_attribution accepted slug=half_written reason=unfinished_turn" in caplog.text


def test_clean_turn_still_drops_unattributed_artifacts(tmp_path):
    conversation = uuid4()
    write_artifact(tmp_path, "half_written")
    assert ownership.turn_created_slugs(tmp_path, set(), conversation) == set()


def test_turn_created_slugs_uses_the_given_after_listing(tmp_path, monkeypatch):
    conversation = uuid4()
    write_artifact(tmp_path, "mine", conversation)
    write_artifact(tmp_path, "unlisted", conversation)

    def _no_listing(_base):
        raise AssertionError("the caller's listing must be reused")

    monkeypatch.setattr(task_objects, "snapshot_artifact_slugs", _no_listing)
    assert ownership.turn_created_slugs(
        tmp_path, set(), conversation, after={"mine"}
    ) == {"mine"}


def test_project_root_source_is_the_shared_project_root(tmp_path, org_id):
    project = make_project(tmp_path, org_id)
    source = ownership.project_root_source(project)
    assert source.base == Path(project.path) / ".anton" / "artifacts"
    assert source.project_id == str(project.id)
    assert source.project_name == project.name
    assert source.trusted_anchor == Path(project.path)
    assert ownership.is_project_root(source)


def test_turn_created_slugs_never_raises(tmp_path):
    assert ownership.turn_created_slugs(object(), set(), uuid4()) == set()
    assert ownership.turn_created_slugs(tmp_path, set(), "not-a-uuid") == set()


def test_index_turn_artifacts_records_the_creator_as_owner(tmp_path, org_id, users):
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, users[0])
    source = project_root(project)
    before, before_mtimes = task_objects.snapshot_artifact_state(source.base)
    write_artifact(source.base, "fresh", conversation_id)
    scope = TenantScope(org_mode=True, org_id=org_id, user_id=users[0])
    with Session(_engine()) as raw:
        session = ScopedSession(raw, scope)
        conversation = session.get(Conversation, conversation_id)
        new, touched, turn_scope = task_objects.index_turn_artifacts(
            conversation, conversation_id, project.id, source.base,
            before, before_mtimes,
            tracked_new=ownership.turn_created_slugs(source.base, before, conversation_id),
        )
    assert new == ["fresh"]
    assert turn_scope == scope
    with scoped(org_id, users[1]) as session:
        assert ownership.resolve_artifact_owner(session, source, "fresh") == (
            ownership.OwnerResolution(users[0], "recorded")
        )


@pytest.mark.parametrize("completed_cleanly", [True, False])
def test_index_turn_artifacts_attributes_by_provenance_with_one_listing(
    tmp_path, monkeypatch, completed_cleanly
):
    """The producers' path: the root is listed once, and the provenance filter
    keeps this turn's own folder, drops a sibling's, and keeps an unattributed
    one only when the turn did not complete cleanly."""
    conversation_id = uuid4()
    write_artifact(tmp_path, "mine", conversation_id)
    write_artifact(tmp_path, "sibling", uuid4())
    write_artifact(tmp_path, "handmade")
    listings = []
    real_listing = task_objects.snapshot_artifact_slugs

    def _counted(base):
        listings.append(base)
        return real_listing(base)

    monkeypatch.setattr(task_objects, "snapshot_artifact_slugs", _counted)
    monkeypatch.setattr(task_objects, "_recover_turn_scope", lambda _c: None)
    monkeypatch.setattr(task_objects, "_index_new_slugs", lambda *a: None)
    new, touched, _scope = task_objects.index_turn_artifacts(
        None, conversation_id, None, tmp_path, set(), {},
        attribute_by_provenance=True, completed_cleanly=completed_cleanly,
    )
    expected = ["mine"] if completed_cleanly else ["handmade", "mine"]
    assert new == expected
    assert touched == set(expected)
    assert len(listings) == 1
