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


# ── skill-draft attribution ────────────────────────────────────────────────
# A skill the agent builds for the user (via the `skill-creator` skill) must NOT
# auto-persist to the skills store and must NOT surface as an artifact — the
# user explicitly Saves or Downloads it. We stage drafts under
# `<project>/.anton/skill_drafts/<slug>/` (a sibling of `.anton/artifacts`, both
# under the already-off-limits `.anton/` dir, so a draft is invisible to BOTH
# the artifacts scan and skill-discovery) and surface each as a self-contained
# `response.skill_created` event. Mirrors the artifact snapshot/diff above, but
# is deliberately NOT indexed as a TaskObject — a draft is transient until Save.

# A skill folder is a draft iff it holds the canonical SKILL.md filename.
_DRAFT_FILE_MAX = 200_000  # per sibling file; skills are small text — cap defensively


def snapshot_skill_drafts(drafts_base) -> set[str]:
    """The set of skill-draft folder names under `.anton/skill_drafts`."""
    from anton.core.tools.skill_format import SKILL_FILE

    base = Path(drafts_base)
    if not base.is_dir():
        return set()
    return {
        child.name
        for child in base.iterdir()
        if child.is_dir() and (child / SKILL_FILE).is_file()
    }


def snapshot_stray_skills(project_skills_dir) -> set[str]:
    """The set of *real* (non-symlink) skill folders under `<project>/skills`.

    Every legitimately-enabled skill is a SYMLINK into the canonical store
    (see services.skill_links). So a real directory with a SKILL.md is a skill
    the agent wrote directly — the auto-save leak we must not persist. We diff
    this set around the turn and relocate any newcomer into a draft.

    ponytail: symlink-vs-real is the discriminator (POSIX-accurate). On Windows,
    a non-privileged symlink can fall back to a copy/junction so is_symlink()
    may miss it and mis-flag an enabled skill as stray. Upgrade path if Windows
    relocation misfires: compare realpath against the canonical skills root.
    """
    from anton.core.tools.skill_format import SKILL_FILE

    base = Path(project_skills_dir)
    if not base.is_dir():
        return set()
    return {
        child.name
        for child in base.iterdir()
        if child.is_dir() and not child.is_symlink() and (child / SKILL_FILE).is_file()
    }


def _unique_draft_dir(drafts_base: Path, slug: str) -> Path:
    """A non-colliding destination folder inside `drafts_base` for `slug`."""
    dest = drafts_base / slug
    if not dest.exists():
        return dest
    i = 2
    while (drafts_base / f"{slug}-{i}").exists():
        i += 1
    return drafts_base / f"{slug}-{i}"


def _skill_draft_payload(folder: Path) -> dict | None:
    """Build a SELF-CONTAINED skill-draft payload from a staged folder.

    Carries everything the UI needs to render the card/modal AND to Save
    (POST /skills) or Download offline — so replay-on-reload needs zero staging
    files. Reuses the shared `parse_skill_dir` + `Skill` model rather than
    re-parsing YAML.
    """
    from anton.core.tools.skill_format import SKILL_FILE, parse_skill_dir

    from cowork.models.skill import Skill

    skill_md_path = folder / SKILL_FILE
    if not skill_md_path.is_file():
        return None
    try:
        agent = parse_skill_dir(folder)
    except Exception:
        logger.warning("Could not parse skill draft %r", folder.name, exc_info=True)
        return None
    if agent is None:
        return None

    skill = Skill.model_construct(**dict(agent))
    slug = folder.name
    raw_md = skill_md_path.read_text(encoding="utf-8", errors="replace")

    # Sibling text files (multi-file skills). Skip binaries — skills are text;
    # a binary sibling is out of scope (download falls back to SKILL.md only).
    files: list[dict] = []
    for child in sorted(folder.iterdir()):
        if child.name == SKILL_FILE or not child.is_file():
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        files.append({"name": child.name, "text": text[:_DRAFT_FILE_MAX]})

    return {
        "slug": slug,
        "label": skill.name or slug,
        "name": skill.display_name or skill.name or slug,
        "description": skill.description or "",
        "instructions": skill.instructions or "",
        "skill_md": raw_md[:_DRAFT_FILE_MAX],  # cap like sibling files — keep the SSE payload bounded
        "files": files,
    }


def finalize_turn_skill_drafts(project_path, before_drafts: set[str], before_strays: set[str]) -> list[dict]:
    """End-of-turn skill-draft handling from a single dir diff.

    1. Relocate any NEW stray (non-symlink) skill folder the agent wrote into
       `<project>/skills` over into the drafts dir — kills the auto-save leak
       even if the prompt/tool routing failed. Moving it in makes it show up in
       the drafts diff below, so it travels the same card path.
    2. Diff the drafts dir and return a self-contained payload per new draft.

    Returns `[]` when nothing new. Best-effort: every step is guarded so a draft
    can never break a turn.
    """
    base = Path(project_path)
    drafts_base = base / ".anton" / "skill_drafts"
    skills_dir = base / "skills"

    # 1. Relocate stray auto-saved skills into drafts.
    try:
        new_strays = sorted(snapshot_stray_skills(skills_dir) - set(before_strays or ()))
        if new_strays:
            drafts_base.mkdir(parents=True, exist_ok=True)
        for slug in new_strays:
            try:
                shutil.move(str(skills_dir / slug), str(_unique_draft_dir(drafts_base, slug)))
            except OSError:
                logger.warning("Could not relocate stray skill %r into drafts", slug, exc_info=True)
    except Exception:
        logger.warning("Stray-skill relocation failed", exc_info=True)

    # 2. Diff drafts, build payloads, then remove each folder — the payload is
    # self-contained so staging files are not needed after this point.
    after = snapshot_skill_drafts(drafts_base)
    new = sorted(after - set(before_drafts or ()))
    payloads: list[dict] = []
    for slug in new:
        folder = drafts_base / slug
        payload = _skill_draft_payload(folder)
        if payload is not None:
            payloads.append(payload)
            try:
                shutil.rmtree(folder)
            except OSError:
                logger.warning("Could not remove skill draft folder %r", slug, exc_info=True)
    return payloads
