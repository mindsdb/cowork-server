from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from uuid import UUID


from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope, scope_of_session
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.task_object import TaskObject

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

    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    # ── indexing ──────────────────────────────────────────────────────

    def index_artifact(self, conversation_id: UUID, project_id: UUID, slug: str) -> None:
        """Upsert an artifact row (idempotent on conversation+ref).

        Anchors both roots through the scope first — TaskObject has no org
        column, so parent visibility IS the tenancy check."""
        if not slug:
            return
        if (
            self.session.get(Conversation, conversation_id) is None
            or self.session.get(Project, project_id) is None
        ):
            raise ValueError("Conversation or project not found in scope")
        existing = self.session.exec(
            self.session.select(TaskObject).where(
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
                self.session.select(TaskObject).where(
                    TaskObject.conversation_id == conversation.id,
                    TaskObject.kind == KIND_ARTIFACT,
                )
            ).all()
        )

    # ── deleting ──────────────────────────────────────────────────────

    def delete_for_conversation(self, conversation: Conversation) -> int:
        """Drop every object row a conversation owns, so deleting a chat (or
        clearing its history) doesn't leave dangling rows that keep surfacing
        artifacts the task no longer has. Takes the LOADED conversation (not an
        id) so the parent was necessarily fetched through the caller's scope.
        Stages the deletes on the session; the caller commits as part of its
        own transaction. Returns the count.

        Only the index is removed — on-disk artifact folders and attached file
        bytes are left untouched (work products, managed separately)."""
        rows = self.session.exec(
            self.session.select(TaskObject).where(TaskObject.conversation_id == conversation.id)
        ).all()
        for row in rows:
            self.session.delete(row)
        return len(rows)

    # ── moving ────────────────────────────────────────────────────────

    def relocate_to_project(
        self,
        conversation: Conversation,
        source: Project,
        dest: Project,
    ) -> int:
        """Move everything the task owns from `source` to `dest`:
          • artifact folders are physically moved into the destination's
            artifacts tree (prefixed on a name collision; `.published.json`
            rides along inside the folder so the public URL is preserved);
          • attachment files need no re-tagging: purposes are keyed by the
            conversation id, not the project (ENG-338), so they follow the
            conversation automatically (their bytes live outside any project
            dir, so no file move either).
        Best-effort: a failure on one object is logged and skipped rather
        than aborting the whole move. Returns the moved-artifact count.
        """
        return self._relocate_artifacts(conversation, source, dest)

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


def snapshot_artifact_state(artifacts_base) -> tuple[set[str], dict[str, int]]:
    """Pre-turn snapshot: folder names PLUS each one's content mtime.

    Names alone only reveal artifacts the turn CREATED. The autopublish
    reconciler also needs the ones it EDITED, and those are invisible to a name
    diff, so the snapshot carries `content_mtime` per slug and the diff compares
    both.

    Granularity is whole seconds (`content_mtime` truncates), so an artifact
    edited within the same second as this snapshot is not reported as touched and
    falls through to the reconciler's self-heal phase instead — published, just a
    turn later.
    """
    from cowork.services.artifacts import content_mtime

    base = Path(artifacts_base)
    slugs = snapshot_artifact_slugs(base)
    mtimes: dict[str, int] = {}
    for slug in slugs:
        try:
            mtimes[slug] = content_mtime(base / slug)
        except OSError:
            mtimes[slug] = 0
    return slugs, mtimes


def _recover_turn_scope(conversation) -> TenantScope | None:
    """The tenant scope for post-turn work, or None when there is none.

    Primary source is the scope the conversation's session was wrapped with —
    the ORIGINAL authorization context, never one derived from row data.
    Fallback is the ambient scope bound at the turn boundary
    (`use_settings_scope` in handlers.responses), which survives a detached or
    expired session. Returning None is the fail-safe: callers skip their work
    rather than inventing a tenant.
    """
    from sqlalchemy.orm import object_session

    try:
        _sess = object_session(conversation)
    except Exception:
        _sess = None
    scope = scope_of_session(_sess) if _sess is not None else None
    if scope is not None:
        return scope
    try:
        from cowork.common.settings.user_settings import current_settings_scope

        return current_settings_scope()
    except Exception:
        return None


def _index_new_slugs(conversation_id, project_id, slugs: list[str], scope: TenantScope | None) -> None:
    """Attribute freshly appeared artifacts to this conversation. Best-effort."""
    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.session import get_engine, get_session_factory

        if scope is None:
            # Never invent a scope: local mode passes through; org mode without
            # the caller's scope skips indexing (fail-safe).
            if get_app_settings().tenancy_mode == "org":
                logger.warning(
                    "artifact indexing skipped: no tenant scope provided in org mode (conversation %s)",
                    conversation_id,
                )
                return
            scope = LOCAL_SCOPE
        factory = get_session_factory(get_engine(get_app_settings().database.uri))
        with factory() as session:
            svc = TaskObjectService(ScopedSession(session, scope))
            for slug in slugs:
                svc.index_artifact(conversation_id, project_id, slug)
    except Exception:
        logger.warning("Could not index artifacts created this turn", exc_info=True)


def index_turn_artifacts(
    conversation,
    conversation_id,
    project_id,
    artifacts_base,
    before: set[str],
    before_mtimes: dict[str, int],
) -> tuple[list[str], set[str], TenantScope | None]:
    """End-of-turn artifact bookkeeping from a SINGLE artifacts-dir diff.

    Returns (new_slugs, touched_slugs, scope):
      • new_slugs — folders that appeared during the turn; each is indexed as
        owned by this conversation so it relocates with the task and shows in
        the artifacts panel;
      • touched_slugs — new_slugs plus every pre-existing slug whose content
        mtime grew, i.e. what this turn actually wrote. The autopublish
        reconciler publishes these first;
      • scope — the tenant scope for post-turn work (see _recover_turn_scope).

    conversation_id/project_id are captured by the caller while the row is
    unambiguously attached (not read here, to avoid depending on the session
    still being live/unexpired in this end-of-turn path).

    Never raises. This runs in a turn's `finally`, so an exception here would
    replace the turn's real outcome; on any internal failure it degrades to
    ([], set(), None) and the next turn picks the work up.
    """
    try:
        from cowork.services.artifacts import content_mtime

        base = Path(artifacts_base)
        after = snapshot_artifact_slugs(base)
        new = sorted(after - set(before or ()))
        touched = set(new)
        for slug in after:
            previous = (before_mtimes or {}).get(slug)
            if previous is None:
                continue  # appeared this turn — already in `new`
            try:
                if content_mtime(base / slug) > previous:
                    touched.add(slug)
            except OSError:
                continue
        scope = _recover_turn_scope(conversation)
        if new:
            _index_new_slugs(conversation_id, project_id, new, scope)
        return new, touched, scope
    except Exception:
        logger.warning("index_turn_artifacts failed", exc_info=True)
        return [], set(), None


def cards_for_slugs(
    artifacts_base,
    slugs: list[str],
    *,
    project_id: str | None = None,
    project_name: str = "",
) -> list[dict]:
    """Inline-chat card payloads for the given slugs, order preserved.

    Uses the same per-folder card builder as the artifacts list
    (`services.artifacts.card_for_folder`), so inline cards and the panel can
    never disagree about what a turn produced or how an artifact opens.

    `project_id`/`project_name` are passed through to the card because that is
    how the client addresses an artifact in org mode (project + slug); a card
    without them would fall back to the path-based endpoints, which org mode
    fails closed. Best-effort per slug: an unreadable artifact is skipped.
    """
    from cowork.services.artifacts import card_for_folder

    base = Path(artifacts_base)
    cards: list[dict] = []
    for slug in slugs:
        try:
            card = card_for_folder(
                base / slug, len(cards),
                project_id=project_id, project_name=project_name,
            )
        except Exception:
            logger.warning("Could not build inline card for artifact %r", slug, exc_info=True)
            continue
        if card is not None:
            cards.append(card)
    return cards


def finalize_turn_artifacts(conversation, conversation_id, project_id, artifacts_base, before: set[str]) -> list[dict]:
    """Index this turn's new artifacts and return their cards.

    Kept as the pre-split entry point for harnesses that do not participate in
    autopublish (hermes_harness sets `supports_org_mode = False`). Callers that
    need `touched` or the tenant scope use `index_turn_artifacts` directly.

    `before_mtimes` is empty here on purpose: without a pre-turn mtime snapshot
    `touched` degenerates to "the new slugs", which is all this entry point's
    callers need — they do not publish.
    """
    new, _touched, _scope = index_turn_artifacts(
        conversation, conversation_id, project_id, artifacts_base, before, {},
    )
    return cards_for_slugs(artifacts_base, new)


async def publish_and_card_turn_artifacts(
    artifacts_base,
    *,
    new_slugs: list[str],
    touched_slugs: set[str],
    scope,
    project_id: str | None = None,
    project_name: str = "",
) -> list[dict]:
    """Reconcile publishes for this turn, then build the cards to emit.

    The second half of the end-of-turn artifact flow; `index_turn_artifacts` is
    the first half and produces the three arguments. They are separate because
    only this half may await: indexing has to run in a turn's `finally` (so an
    artifact is recorded even on error or Stop), and an `await` there is skipped
    on cancellation.

    Shared by both producers. The in-process harness reaches it through
    `AntonHarness.stream_response`; on an org deployment that harness refuses to
    run at all and the turn happens on the remote worker, so
    `handlers.responses._produce_remote` calls this against the same shared
    artifacts tree the worker wrote to.

    Cards cover what THIS turn produced or touched. `republished` also carries
    phase-two self-heal publishes — older artifacts from earlier conversations —
    and the stream reducer dedupes only within one message, so including them
    would attach last week's artifacts to this answer.
    """
    from cowork.services.artifact_autopublish import autopublish_project_artifacts

    republished = await autopublish_project_artifacts(
        artifacts_base, scope, touched=set(touched_slugs),
    )
    carded = set(new_slugs) | (republished & set(touched_slugs))
    return cards_for_slugs(
        artifacts_base, sorted(carded),
        project_id=project_id, project_name=project_name,
    )


# ── skill-draft attribution ────────────────────────────────────────────────
# A skill the agent builds for the user (via the `skill-creator` skill) must NOT
# auto-persist to the skills store and must NOT surface as an artifact — the
# user explicitly Saves or Downloads it. We stage drafts under
# `<project>/.anton/skill_drafts/<slug>/` (a sibling of `.anton/artifacts`, both
# under the already-off-limits `.anton/` dir, so a draft is invisible to BOTH
# the artifacts scan and skill-discovery) and surface each as a self-contained
# `response.skill_created` event. Mirrors the artifact snapshot/diff above, but
# is deliberately NOT indexed as a TaskObject — a draft persists on disk until
# the user Saves or dismisses it.

# A skill folder is a draft iff it holds the canonical SKILL.md filename.
_DRAFT_FILE_MAX = 200_000  # per sibling file; skills are small text — cap defensively


def _draft_content_hash(skill_md_path: Path) -> str:
    """SHA-256 of a draft's SKILL.md bytes ('' if unreadable).

    ponytail: hashes SKILL.md only, not sibling files — a sibling-only edit
    won't re-emit a card. Skills are refined by rewriting SKILL.md, so this
    covers the real case; upgrade to a whole-folder hash if sibling-only edits
    must surface.
    """
    try:
        return hashlib.sha256(skill_md_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def snapshot_skill_drafts(drafts_base) -> dict[str, str]:
    """Map each skill-draft folder name under `.anton/skill_drafts` to a content
    hash of its SKILL.md.

    Keying on content (not just the folder name) lets the turn-end diff re-emit
    a card when a draft is refined in place across turns, not only when a new
    folder first appears.
    """
    from anton.core.tools.skill_format import SKILL_FILE

    base = Path(drafts_base)
    if not base.is_dir():
        return {}
    snapshot: dict[str, str] = {}
    for child in base.iterdir():
        skill_md = child / SKILL_FILE
        if child.is_dir() and skill_md.is_file():
            snapshot[child.name] = _draft_content_hash(skill_md)
    return snapshot


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


def _seed_draft_from_store(folder: Path, slug: str) -> None:
    """Copy a saved skill's SKILL.md + sibling files into an empty draft folder.

    Lets "edit an existing skill" start from the stored version (Save then
    upserts it back) instead of the agent hunting for and mutating the live
    store in place. Best-effort: a missing store entry or copy failure just
    leaves the folder empty (a fresh draft), never raises into the turn.

    ponytail: copies top-level files only (skills are flat text) — mirrors
    `_skill_draft_payload`, which also skips subdirs.

    `slug` is agent-supplied: validate it, and key the store by the turn's
    ambient scope. Org mode with no scope bound seeds nothing rather than
    reading the unkeyed root.
    """
    from anton.core.tools.skill_format import SKILL_FILE, validate_name

    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.common.settings.user_settings import current_settings_scope
        from cowork.db.scoped import scoped_storage_root

        validate_name(slug)
        settings = get_app_settings()
        scope = current_settings_scope()
        if settings.tenancy_mode == "org" and (scope is None or not scope.org_mode):
            return
        src = scoped_storage_root(Path(settings.skill.root_dir), scope, store="skills") / slug
    except Exception:
        return
    # Skip symlinks (dir and children): the store is org-shared, so a link could
    # dereference into another org's files (copy2/is_file follow symlinks).
    if src.is_symlink() or not (src / SKILL_FILE).is_file():
        return
    src_resolved = src.resolve()
    try:
        for child in src.iterdir():
            if child.is_symlink() or not child.is_file():
                continue
            if child.resolve().parent != src_resolved:
                logger.warning("Skill draft seed %r: skipping out-of-tree file %r", slug, child.name)
                continue
            shutil.copy2(child, folder / child.name)
    except OSError:
        logger.warning("Could not seed skill draft %r from store", slug, exc_info=True)


# LLM-facing contract for the `create_skill_draft` tool, shared verbatim by both
# harnesses (hermes registers it in run_agent's registry, anton as a ToolDef) so
# the tool reads identically regardless of agent.
CREATE_SKILL_DRAFT_DESCRIPTION = (
    "Claim a staging folder for a skill you are building or improving for the "
    "user (e.g. while running the skill-creator skill). Call this BEFORE writing "
    "the skill; it returns {slug, path, skill_file} — write your SKILL.md to "
    "`skill_file` (and any sibling files into `path`). If a skill with this name "
    "is already saved, the folder is pre-filled with its current contents so you "
    "edit from the saved version. The skill is staged, NOT saved: it surfaces as "
    "a card the user explicitly saves or downloads. NEVER write a skill into the "
    "project `skills/` directory and NEVER use `create_artifact` for a skill."
)

CREATE_SKILL_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The skill's name (becomes its slug, e.g. 'competitive-analysis').",
        },
        "description": {
            "type": "string",
            "description": "Optional one-line trigger description shown on the card.",
        },
    },
    "required": ["name"],
}


def stage_skill_draft(drafts_root, name: str) -> dict:
    """Claim a draft folder for `name` under `drafts_root`; return its location.

    If a saved skill with the same slug exists AND the draft folder isn't already
    populated, seed it with the stored skill (see `_seed_draft_from_store`) so an
    edit starts from the saved version. Returns `{slug, path, skill_file}`, or
    `{error}` on invalid input.

    Shared by both harnesses' `create_skill_draft` tool so the claim + seed
    behaviour is identical regardless of agent.
    """
    from anton.core.tools.skill_format import SKILL_FILE, normalize_name

    name = str(name or "").strip()
    if not name:
        return {"error": "`name` is required."}
    slug = normalize_name(name)
    if not slug:
        return {"error": "`name` must contain at least one alphanumeric character."}

    folder = Path(drafts_root) / slug
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / SKILL_FILE).is_file():
        _seed_draft_from_store(folder, slug)
    return {
        "slug": slug,
        "path": str(folder),
        "skill_file": str(folder / SKILL_FILE),
    }


def finalize_turn_skill_drafts(project_path, before_drafts: dict[str, str], before_strays: set[str]) -> list[dict]:
    """End-of-turn skill-draft handling from a content diff of the drafts dir.

    1. Relocate any NEW stray (non-symlink) skill folder the agent wrote into
       `<project>/skills` over into the drafts dir — kills the auto-save leak
       even if the prompt/tool routing failed. Moving it in makes it show up in
       the drafts diff below, so it travels the same card path.
    2. Diff the drafts dir by SKILL.md content and return a self-contained
       payload for every draft that is NEW or CHANGED since the turn started.

    Draft folders PERSIST — they are not swept here, so an unsaved skill stays on
    disk and later refinements re-emit an updated card (cleanup happens on
    Save/Dismiss, not per turn). Returns `[]` when nothing new or changed.
    Best-effort: every step is guarded so a draft can never break a turn.
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

    # 2. Diff drafts by content: emit a payload for every draft whose SKILL.md is
    # new or changed since the turn started. Folders are KEPT on disk so a draft
    # survives across turns and later refinements re-emit an updated card.
    #
    # ponytail: drafts are removed only on Save (client sweeps via
    # DELETE .../skill_drafts/{slug}) or when the whole project is deleted. An
    # unsaved draft the user abandons is never GC'd — it lingers under
    # `.anton/skill_drafts/`. TODO: age-based GC or a "Unsaved skills" panel with
    # Dismiss to reap orphans (a deleted conversation can orphan its only card).
    before = dict(before_drafts or {})
    after = snapshot_skill_drafts(drafts_base)
    payloads: list[dict] = []
    for slug in sorted(after):
        if after[slug] == before.get(slug):
            continue  # unchanged since turn start → no card
        payload = _skill_draft_payload(drafts_base / slug)
        if payload is not None:
            payloads.append(payload)
    return payloads
