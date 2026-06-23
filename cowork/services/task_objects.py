from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.task_object import TaskObject
from cowork.services.files import FileService, attachment_purpose

logger = logging.getLogger(__name__)

KIND_ARTIFACT = "artifact"
KIND_FILE = "file"


def _artifacts_base(project: Project) -> Path:
    """A project's on-disk artifacts root (`<project>/.anton/artifacts`)."""
    return Path(project.path) / ".anton" / "artifacts"


def _artifact_owner(folder: Path) -> str | None:
    """The conversation id that first created this artifact, read from its
    metadata `provenance` (written by the shared ArtifactStore for every
    harness). The creating conversation is the first provenance entry."""
    meta = folder / "metadata.json"
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    provenance = data.get("provenance") or []
    if not isinstance(provenance, list) or not provenance:
        return None
    first = provenance[0]
    if isinstance(first, dict):
        owner = first.get("conversation")
        return str(owner) if owner else None
    return None


class TaskObjectService:
    """Indexes the artifacts/files a task owns and relocates them when the
    task moves to another project."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── indexing ──────────────────────────────────────────────────────

    def index_artifact(self, conversation_id: UUID, project_id: UUID, slug: str) -> None:
        """Upsert an artifact row (idempotent on conversation+ref)."""
        if not slug:
            return
        existing = self.session.exec(
            select(TaskObject).where(
                TaskObject.conversation_id == conversation_id,
                TaskObject.kind == KIND_ARTIFACT,
                TaskObject.ref == slug,
            )
        ).first()
        if existing is not None:
            if existing.project_id != project_id:
                existing.project_id = project_id
                self.session.add(existing)
                self.session.commit()
            return
        self.session.add(
            TaskObject(
                conversation_id=conversation_id,
                project_id=project_id,
                kind=KIND_ARTIFACT,
                ref=slug,
            )
        )
        self.session.commit()

    def reconcile_conversation(self, conversation: Conversation, project: Project) -> list[TaskObject]:
        """Scan the project's artifacts and index any folder this
        conversation created (per on-disk provenance) that isn't already
        tracked. Makes the table complete for artifacts produced by any
        harness, or before this index existed. Returns the conversation's
        artifact rows."""
        base = _artifacts_base(project)
        if base.is_dir():
            for folder in base.iterdir():
                if not folder.is_dir():
                    continue
                if _artifact_owner(folder) == str(conversation.id):
                    self.index_artifact(conversation.id, project.id, folder.name)
        return list(
            self.session.exec(
                select(TaskObject).where(
                    TaskObject.conversation_id == conversation.id,
                    TaskObject.kind == KIND_ARTIFACT,
                )
            ).all()
        )

    # ── moving ────────────────────────────────────────────────────────

    def relocate_to_project(
        self,
        conversation: Conversation,
        source: Project,
        dest: Project,
    ) -> dict:
        """Move everything the task owns from `source` to `dest`:
          • artifact folders are physically moved into the destination's
            artifacts tree (prefixed on a name collision; `.published.json`
            rides along inside the folder so the public URL is preserved);
          • attachment files are re-tagged to the destination project
            (their bytes live outside any project dir, so no file move).
        Best-effort: a failure on one object is logged and skipped rather
        than aborting the whole move. Returns counts.
        """
        moved_artifacts = self._relocate_artifacts(conversation, source, dest)
        relinked_files = self._relink_files(conversation, source, dest)
        return {"artifacts": moved_artifacts, "files": relinked_files}

    def _relocate_artifacts(self, conversation: Conversation, source: Project, dest: Project) -> int:
        rows = self.reconcile_conversation(conversation, source)
        if not rows:
            return 0
        src_base = _artifacts_base(source)
        dest_base = _artifacts_base(dest)
        dest_base.mkdir(parents=True, exist_ok=True)
        moved = 0
        for row in rows:
            src_folder = src_base / row.ref
            if not src_folder.is_dir():
                # Folder gone (deleted/already moved) — just retarget the row.
                row.project_id = dest.id
                self.session.add(row)
                continue
            dest_slug = self._unique_slug(dest_base, row.ref, conversation.id)
            try:
                shutil.move(str(src_folder), str(dest_base / dest_slug))
            except OSError:
                logger.warning("Could not move artifact %r to project %r", row.ref, dest.name, exc_info=True)
                continue
            row.project_id = dest.id
            row.ref = dest_slug
            self.session.add(row)
            moved += 1
        self.session.commit()
        return moved

    @staticmethod
    def _unique_slug(dest_base: Path, slug: str, conversation_id: UUID) -> str:
        """Avoid clobbering an artifact already in the destination by
        prefixing with a short task id, then a numeric suffix if needed."""
        if not (dest_base / slug).exists():
            return slug
        prefixed = f"{str(conversation_id)[:8]}-{slug}"[:255]
        if not (dest_base / prefixed).exists():
            return prefixed
        i = 2
        while (dest_base / f"{prefixed}-{i}").exists():
            i += 1
        return f"{prefixed}-{i}"[:255]

    def _relink_files(self, conversation: Conversation, source: Project, dest: Project) -> int:
        old_purpose = attachment_purpose(source.name, str(conversation.id))
        new_purpose = attachment_purpose(dest.name, str(conversation.id))
        if old_purpose == new_purpose:
            return 0
        return FileService(self.session).relink_purpose(old_purpose, new_purpose)


# ── run-boundary attribution ──────────────────────────────────────────────
# Anton runs with its own episodic session id and never tags artifacts with
# the cowork conversation_id, so provenance can't tell us which task created
# which artifact. Instead cowork-server (which DOES know the conversation it's
# running) snapshots the project's artifact folders before a turn and records
# any that appear afterward as owned by that conversation. Harness-agnostic
# and needs no agent change.

def snapshot_artifact_slugs(artifacts_base) -> set[str]:
    """The set of artifact folder names under a project's artifacts dir."""
    base = Path(artifacts_base)
    if not base.is_dir():
        return set()
    return {
        child.name
        for child in base.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file()
    }


def finalize_turn_artifacts(conversation_id, project_id, artifacts_base, before: set[str]) -> list[dict]:
    """End-of-turn artifact handling, from a SINGLE artifacts-dir diff.

    For every artifact folder that appeared during the turn this:
      • indexes it as owned by this conversation (so it relocates with the
        task and shows in the artifacts panel), and
      • builds its inline-chat card payload.

    Because both come from the same diff and the same per-folder card builder
    that the artifacts list uses (`services.artifacts.card_for_folder`), the
    inline cards, the artifacts panel, and the move/index can never disagree
    about what a turn produced or how an artifact opens.

    Returns the card payloads (``[]`` when nothing new). Best-effort: indexing
    and card-building are each guarded, so neither can break a turn.
    """
    after = snapshot_artifact_slugs(artifacts_base)
    new = sorted(after - set(before or ()))
    if not new:
        return []

    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.session import get_engine, get_session_factory

        factory = get_session_factory(get_engine(get_app_settings().database.uri))
        with factory() as session:
            svc = TaskObjectService(session)
            for slug in new:
                svc.index_artifact(conversation_id, project_id, slug)
    except Exception:
        logger.warning("Could not index artifacts created this turn", exc_info=True)

    from cowork.services.artifacts import card_for_folder

    base = Path(artifacts_base)
    cards: list[dict] = []
    for slug in new:
        try:
            card = card_for_folder(base / slug)
        except Exception:
            logger.warning("Could not build inline card for artifact %r", slug, exc_info=True)
            continue
        if card is not None:
            cards.append(card)
    return cards
