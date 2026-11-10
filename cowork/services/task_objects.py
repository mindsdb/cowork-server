from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError

from cowork.common.logger import log_context
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope, tenant_scope_required
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.task_object import TaskObject
from cowork.services.artifact_identity import ArtifactListing, ListedIds
from cowork.services.artifact_metadata import ArtifactMetadata
from cowork.turnqueue.models import TurnCompleted

if TYPE_CHECKING:
    from cowork.schemas.artifact_card import ArtifactCard

logger = logging.getLogger(__name__)

KIND_ARTIFACT = "artifact"
KIND_FILE = "file"


def _artifacts_base(project: Project) -> Path:
    """A project's on-disk artifacts root (`<project>/.anton/artifacts`)."""
    return Path(project.path) / ".anton" / "artifacts"


def _artifact_owner(folder: Path) -> str | None:
    """The conversation id that first created this artifact, read from its
    metadata `provenance`. Derivation lives in `services.artifacts` — the same
    value ships to the client as the card's `originConversationId`, so the two
    must never disagree about which conversation owns an artifact."""
    meta = folder / "metadata.json"
    if not meta.is_file():
        return None
    try:
        data = ArtifactMetadata.parse(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    from cowork.services.artifacts import origin_conversation_id

    return origin_conversation_id(data) or None


class TaskObjectService:
    """Indexes the artifacts/files a task owns and relocates them when the
    task moves to another project."""

    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    @property
    def _org_mode(self) -> bool:
        return bool(getattr(self.session.scope, "org_mode", False))

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
        # Org mode (ENG-2961, D8): provenance is agent-writable on the shared
        # project root, so it must not create index rows that a relocation would
        # then act on. Rows there come only from the end of the creating turn.
        if not self._org_mode and base.is_dir():
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

    def delete_for_project(self, project_id: UUID) -> int:
        """Drop every object row that still points at a project being deleted.

        A task that moved out without its objects, or whose objects could not
        all follow it, leaves rows here under a conversation in another
        project. The project's own conversations do not reach those, and the
        foreign key into projects has no cascade. Stages the deletes; the
        caller commits. Returns the count."""
        rows = self.session.exec(
            self.session.select(TaskObject).where(TaskObject.project_id == project_id)
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
        """Move the task's artifact folders and retarget its index rows.

        Org mode (ENG-2961, D8): only folders owned by the task's creator move,
        and their owner rows follow them. A row whose folder is already gone is
        retargeted without an ownership check or a "not moved" log line — there
        is nothing on disk to move or to protect.
        """
        rows = self.reconcile_conversation(conversation, source)
        if not rows:
            return 0
        src_base = _artifacts_base(source)
        dest_base = _artifacts_base(dest)
        dest_base.mkdir(parents=True, exist_ok=True)
        org_mode = self._org_mode
        # Lazy, like artifact_ownership below. With live apps on, an app's
        # shares, received mark and launches name its folder by project and
        # slug, so they follow the folder.
        from cowork.code_sandbox import live_apps_on

        live_apps = live_apps_on()
        if live_apps:
            from cowork.services.live_apps import follow_moved_app
        creator = str(conversation.created_by) if conversation.created_by else None
        src_root = None
        resolutions: dict = {}
        if org_mode:
            # Lazy: artifact_ownership imports this module back (lazily too).
            from cowork.services.artifact_ownership import (
                project_root_source,
                rekey_artifact_owner,
                resolve_artifact_owners,
            )

            src_root = project_root_source(source)
            # One query for every row, before the loop.
            resolutions = resolve_artifact_owners(
                self.session, src_root, [row.ref for row in rows]
            )
        rekeys: list[tuple[str, str]] = []
        moved_folders: list[tuple[Path, Path]] = []
        moved = 0
        for row in rows:
            src_folder = src_base / row.ref
            if not src_folder.is_dir():
                # Folder gone (deleted/already moved) — just retarget the row.
                row.project_id = dest.id
                self.session.add(row)
                continue
            if org_mode:
                # D8: a task carries only what its creator owns. Anything else
                # indexed to it (someone else's, or ownerless) stays put, and its
                # row keeps pointing at the source project.
                resolution = resolutions[row.ref]
                if not creator or resolution.owner_user_id != creator:
                    logger.warning(
                        "artifact not moved with task: project=%s slug=%s owner_state=%s",
                        source.id, row.ref, resolution.state,
                    )
                    continue
            dest_slug = self._unique_slug(dest_base, row.ref, conversation.id)
            try:
                shutil.move(str(src_folder), str(dest_base / dest_slug))
            except OSError:
                logger.warning("Could not move artifact %r to project %r", row.ref, dest.name, exc_info=True)
                continue
            if org_mode:
                rekeys.append((row.ref, dest_slug))
            moved_folders.append((src_folder, dest_base / dest_slug))
            if live_apps:
                follow_moved_app(
                    self.session,
                    partition_id=str(source.org_id),
                    source_project_id=source.id,
                    old_slug=row.ref,
                    dest_project_id=dest.id,
                    new_slug=dest_slug,
                )
            row.project_id = dest.id
            row.ref = dest_slug
            self.session.add(row)
            moved += 1
        self.session.commit()
        # Each moved folder's location row follows it (org mode). Plain values
        # for the log line, as the owner rekeys below keep.
        if rekeys:
            from cowork.services.artifact_locations import move_locations

            source_id, dest_id, conversation_id = source.id, dest.id, conversation.id
            try:
                move_locations(self.session, project_id=source_id, moved=dict(rekeys), to_project_id=dest_id)
            except Exception:
                self.session.rollback()
                logger.warning(
                    "Could not move the locations of artifacts to project %s", dest_id, exc_info=True,
                    extra=log_context(project_id=dest_id, conversation_id=conversation_id),
                )
        # A published page keeps its link: .published.json moved with the
        # folder, and its owner record follows. A failed rekey is logged; the
        # owner's next publish then gets a new link, as a copy would.
        from cowork.services.published_page_ownership import rekey_published_page_paths

        for old_folder, new_folder in moved_folders:
            rekey_published_page_paths(old_folder, new_folder)
        # After the index commit: each rekey commits on its own, and must not
        # commit a half-applied batch of index rows along with it.
        if rekeys:
            actor = str(self.session.scope.user_id or creator)
            # Plain values: a failed rekey commit expires every loaded row,
            # and reading one back needs the session the failure broke.
            dest_id, dest_name, conversation_id = dest.id, dest.name, conversation.id
            for old_slug, new_slug in rekeys:
                try:
                    rekey_artifact_owner(
                        self.session, src_root, old_slug, dest_id, new_slug, actor_id=actor
                    )
                except Exception:
                    # The record names the folder at its new address: a name
                    # collision gives it a new slug in the destination.
                    logger.warning(
                        "Could not move the owner of artifact %r to project %r",
                        old_slug,
                        dest_name,
                        exc_info=True,
                        extra=log_context(
                            project_id=dest_id, conversation_id=conversation_id, artifact_slug=new_slug,
                        ),
                    )
                    # The next rekey and the caller's own writes need a
                    # usable session.
                    self.session.rollback()
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
# cowork-server snapshots the project's artifact folders before a turn and
# attributes the ones that appear afterward to the turn. Since anton #399 the
# artifact tools also write the cowork conversation id first in each folder's
# `provenance`; the remote producers use it to drop folders a concurrent
# sibling turn created (`artifact_ownership.turn_created_slugs`). Provenance is
# agent-writable, so it only ever narrows a claim; ownership itself is a
# server-written attribution row (ENG-2961).

def snapshot_artifact_state(artifacts_base) -> tuple[set[str], dict[str, int]]:
    """Pre-turn snapshot of folder names and nanosecond content mtimes.

    Names alone only reveal artifacts the turn CREATED. The autopublish
    reconciler also needs the ones it EDITED, and those are invisible to a name
    diff, so the snapshot carries `content_mtime` per slug and the diff compares
    both.

    Nanosecond precision closes the former whole-second gap without hashing the
    full primary source of every artifact before and after every agent turn.

    An entry the listing could not stat (artifact_folders_below) existed
    before the turn, so it is in the snapshot, and the diff never takes it
    for a folder the turn made; its mtime is unknown, 0, as for a folder
    whose mtime fails to read. A diff that reads its content at the turn end
    (turn_artifact_changes) then counts it as edited, so the turn records a
    revision of it rather than miss an edit. A listing of the folder itself
    that fails raises, since an empty snapshot would let the diff take every
    folder for one the turn made.
    """
    from cowork.services.artifact_identity import artifact_folders_below
    from cowork.services.artifacts import content_mtime_ns

    base = Path(artifacts_base)
    listing = artifact_folders_below(base=base)
    if listing.base_error is not None:
        raise listing.base_error
    mtimes: dict[str, int] = dict.fromkeys(listing.unreadable, 0)
    for slug in listing.found:
        try:
            mtimes[slug] = content_mtime_ns(base / slug)
        except OSError:
            mtimes[slug] = 0
    return set(listing.found | listing.unreadable), mtimes


class ListedLocationsUnchecked(Exception):
    """record_new_artifacts could not check the locations of the folders a
    full listing found (record_listed_folders). The index, owner and
    location writes it made before that check have committed."""


def record_new_artifacts(
    session: ScopedSession,
    *,
    conversation_id,
    project_id,
    slugs: list[str],
    creator: str | None,
    artifact_ids: Mapping[str, str] | None = None,
    listed_ids: ListedIds | None = None,
) -> None:
    """Attribute freshly appeared artifacts to this conversation, in the
    caller's session. Best-effort.

    `listed_ids` is what reading the id of every folder of the project
    found, when the diff listed them all (ArtifactChanges.listed_ids); in
    org mode each folder whose id's row names another folder of the project
    marks the id conflicted, except when the listing read the row's folder
    and found it gone or holding another id (a rename): that row is stale,
    and the id's next lookup moves it (record_listed_folders). That check
    runs last, and when it fails it raises ListedLocationsUnchecked, once
    the writes before it have committed, so the caller can ask for the
    project's self-heal to check the folders again.

    Two records: the `task_objects` row, and in org mode (ENG-2961) the owner
    row naming ``creator``, the conversation's creator, since a project-level
    root is shared by every member and the owner has to be written down when
    the turn ends rather than read from the path later. A missed owner write
    leaves the artifact `unknown`, never fails the turn.

    Each slug's write is tried on its own and rolled back when it fails, so
    one failure does not drop the slugs after it. A failure is logged at
    ERROR with the conversation, the project and the slug as record
    attributes, since the caller never sees it and a database error's
    sanitized line keeps only those.
    """
    org_mode = bool(getattr(session.scope, "org_mode", False))
    # A channel conversation has no creator, and its turn end checks the
    # listed folders even when it created none, so warn only when a created
    # folder is left without an owner.
    if org_mode and not creator and slugs:
        logger.warning(
            "artifact owner not recorded: conversation %s has no creator", conversation_id
        )
    svc = TaskObjectService(session)
    # Independent of the owner writes below: a failed index row must not
    # leave the artifact ownerless as well.
    for slug in slugs:
        try:
            svc.index_artifact(conversation_id, project_id, slug)
        except Exception:
            session.rollback()
            logger.error(
                "Could not index an artifact",
                exc_info=True,
                extra=log_context(project_id=project_id, conversation_id=conversation_id, artifact_slug=slug),
            )
    if org_mode and creator and project_id:
        from cowork.services.artifact_ownership import record_artifact_owner

        for slug in slugs:
            try:
                record_artifact_owner(session, project_id, slug, creator, action="create")
            except Exception:
                session.rollback()
                logger.error(
                    "Could not record the owner of an artifact",
                    exc_info=True,
                    extra=log_context(project_id=project_id, conversation_id=conversation_id, artifact_slug=slug),
                )
    if org_mode and project_id and artifact_ids:
        from cowork.services.artifact_locations import Location, record_made_folder

        # A turn creates its folders in the project's shared root and sees
        # only those, so it writes no location row; a folder whose id has a
        # row naming another folder of the project marks the id conflicted.
        for slug in slugs:
            if slug not in artifact_ids:
                continue
            try:
                record_made_folder(
                    session, artifact_id=artifact_ids[slug], location=Location(UUID(str(project_id)), None, slug),
                )
            except Exception:
                session.rollback()
                logger.error(
                    "Could not record the location of an artifact",
                    exc_info=True,
                    extra=log_context(project_id=project_id, conversation_id=conversation_id, artifact_slug=slug),
                )
    if org_mode and project_id and listed_ids is not None and listed_ids.ids:
        from cowork.services.artifact_locations import record_listed_folders

        # The diff listed the whole folder, so a copy no turn end claims,
        # such as one a shell `cp -r` makes, is among these folders. Nothing
        # else checks them unless the caller asks for the project's
        # self-heal, so a failed check reaches it.
        try:
            record_listed_folders(session, project_id=UUID(str(project_id)), listing=listed_ids)
        except Exception as exc:
            session.rollback()
            raise ListedLocationsUnchecked from exc


@dataclass(frozen=True)
class ArtifactChanges:
    """What one turn did to its project's artifacts directory."""

    # Folders the turn created, sorted. Each is indexed as this
    # conversation's, so it relocates with the task and shows in the
    # artifacts panel.
    created: list[str]
    # The created folders plus pre-existing ones the turn wrote to. The
    # autopublish reconciler publishes these first.
    touched: set[str]
    # The id each created folder holds, read on the file store with the diff,
    # so recording the folder's location row needs no file store call.
    created_ids: dict[str, str] = field(default_factory=dict)
    # The folders among `repair_slugs` the diff ruled on: it found no
    # artifact folder there, or it opened the folder and, after its captures,
    # found no repair in it still queued to this conversation. A folder it
    # could not read, or whose queued repair it could not finish, is left
    # out, so its queued_agent_repairs row stays for the next turn end. Empty
    # when the diff had no repair rows.
    checked_repairs: set[str] = field(default_factory=set)
    # What reading the id of every folder the diff listed found, when it
    # listed the whole folder because the turn named none (a pod that sends
    # no list): the id each folder holds, the folders it could not stat or
    # read, and whether it could not list the folder at all. Recording the
    # turn then marks an id conflicted for a copy no turn end claims
    # (record_new_artifacts), and a folder it could not read asks for the
    # project's location check (request_self_heal_for_unread_folders). Empty
    # when the diff opened only named folders.
    listed_ids: ListedIds = field(default_factory=lambda: ListedIds(ids={}, unreadable=frozenset()))


def tracked_artifacts_from_reply(data: Mapping) -> set[str] | None:
    """The folder names a remote turn's artifact tools touched, from its
    ``turn_completed`` reply (``TurnCompleted``), or None when the pod sent no
    usable list: an older pod, or a list that isn't one of names, holds a
    name longer than ``MAX_ARTIFACT_NAME_CHARS``, which no folder has, or
    holds more than ``MAX_ARTIFACTS_TOUCHED``, which is not one turn's own
    work, so the diff covers the whole folder. Untrusted. The diff opens each name as
    one folder directly below the pinned artifacts base
    (artifact_folders_named), so a name holding ``..``, a separator or NUL, a
    link, or a folder whose ``metadata.json`` isn't a plain file is left out."""
    try:
        touched = TurnCompleted.model_validate(data).artifacts_touched
    except ValidationError:
        return None
    return set(touched) if touched is not None else None


def turn_artifact_changes(
    *,
    conversation_id,
    artifacts_base,
    before: set[str],
    before_mtimes: dict[str, int],
    tracked_new: set[str] | None = None,
    tracked_edits: set[str] | None = None,
    attribute_by_provenance: bool = False,
    completed_cleanly: bool = True,
    repair_slugs: set[str] | None = None,
) -> ArtifactChanges:
    """What the turn created and touched, from the directory as it stands now
    against ``before``, with an agent revision captured for every touched
    folder. The filesystem half of the end-of-turn bookkeeping: it reads and
    writes no database row, so a caller can record the result in a unit of
    its own (record_new_artifacts).

    The two `tracked_*` bounds are what the turn's own artifact tools reported
    touching, and they are what makes attribution correct: every conversation
    in a project shares one artifacts directory, so a folder that merely
    APPEARED during this turn may well be a concurrent sibling turn's work
    (ENG-1933). Only the caller knows what its own tools did.

      • `tracked_new` — slugs this turn may claim as CREATED. Without it any
        folder that appeared is claimed, which is the bug above.
      • `tracked_edits` — slugs this turn may claim as EDITED. None means fall
        back to "any pre-existing folder whose content mtime grew", which is
        what a harness without an open-for-editing tool has to do: it cannot
        know which artifact the agent wrote to. Narrower exposure than
        `tracked_new` (two turns editing the same pre-existing artifact at
        once, rather than the everyday case of both creating something), but
        not zero.

    Both default to None — a pure directory diff, the pre-existing behaviour.
    That is only safe where no concurrent turn can write the directory. The org
    deployment used to qualify (each conversation's pod had its own artifacts
    dir), but its pods now share the project-level base, so the remote
    producer's pure diff carries the same concurrent-sibling caveat as the
    desktop. A remote pod names the artifacts its tools touched on
    `turn_completed`, and the remote producers pass that list as both tracked
    sets; an older pod sends none, and every other exit has none.

    With both tracked sets given, the diff opens only the folders they name,
    and those in `repair_slugs`, each as one folder directly below the pinned
    base (artifact_folders_named), so a link or a name that isn't one folder
    there is never claimed or read. A named folder it could not read is left
    out, and logged. Otherwise it lists the whole folder
    (artifact_folders_below), and then also reads every folder's id
    (`listed_ids`). An entry that listing could not stat is in
    `listed_ids.unreadable`, and a listing of the folder itself that fails
    sets `listed_ids.base_unreadable`, unless that folder is a link, which
    the id reader refuses (listed_folder_ids). Either way the diff then sees
    no folder at all, so the turn claims nothing.

    `attribute_by_provenance` is for the remote producers. Without
    `tracked_new` (the pod sent no list), a folder that appeared is claimed
    only if its `metadata.json` provenance names this conversation
    (`artifact_ownership.turn_created_slugs`), and, when the turn did not
    complete cleanly, also if it has no provenance yet. With `tracked_new` it
    only drops: a named folder whose provenance names another conversation is
    not claimed, since anton also names a sibling's folder whose files changed
    during one of this turn's cells. `conversation_id` is then a UUID.

    `repair_slugs` are the folders recorded as holding an agent repair queued
    to this conversation (queued_agent_repairs). With both tracked sets given,
    they are the only other folders that can be in the answer; the result's
    `checked_repairs` names the repair folders it ruled on: one that holds no
    artifact folder, or one it read that holds no repair still queued to this
    conversation once the captures ran. With the tracked sets but no
    `repair_slugs` (the rows couldn't be read), the diff lists the whole
    folder only to find a repair queued to this conversation. Without the
    tracked sets it checks every folder it listed.
    """
    from cowork.services.artifacts import content_mtime_ns
    from cowork.services.artifact_identity import (
        artifact_folders_below,
        artifact_folders_named,
        artifact_ids_below,
        listed_folder_ids,
    )
    from cowork.services.artifact_revisions import (
        capture_agent_revision,
        has_queued_agent_repair,
    )

    base = Path(artifacts_base)
    conversation_key = str(conversation_id)
    checked_repairs: set[str] = set()
    # Folders checked for a repair queued to this conversation besides those
    # in `after`: every folder, when the repair rows couldn't be read.
    repair_candidates: set[str] = set()
    whole_listing: ArtifactListing | None = None
    if tracked_new is None or tracked_edits is None:
        whole_listing = artifact_folders_below(base=base)
        after = set(whole_listing.found)
    else:
        tracked = set(tracked_new) | set(tracked_edits)
        named = artifact_folders_named(base=base, slugs=tracked | set(repair_slugs or ()))
        after = named.found
        lost = named.unreadable & tracked
        if lost:
            # The turn records nothing for these folders, and in org mode
            # nothing records them later, so say which ones.
            logger.warning(
                "the turn end could not read folders its tools touched, so it leaves them out",
                extra=log_context(conversation_id=conversation_id, artifact_slugs=sorted(lost)),
            )
        if repair_slugs is not None:
            checked_repairs = set(repair_slugs) - named.unreadable
        else:
            # A folder this listing cannot read is not checked, and no row is
            # dropped without repair rows to drop.
            repair_candidates = set(artifact_folders_below(base=base).found)
    appeared = after - set(before or ())
    if attribute_by_provenance:
        from cowork.services.artifact_ownership import provenance_origin, turn_created_slugs

        if tracked_new is None:
            tracked_new = turn_created_slugs(
                base, before, conversation_id,
                after=after, accept_unattributed=not completed_cleanly,
            )
        else:
            # anton also names a sibling's folder whose files changed during
            # one of this turn's cells, so a named folder whose provenance
            # names another conversation is not this turn's. One with no
            # provenance is still claimed.
            own = UUID(conversation_key)
            claimed: set[str] = set()
            for slug in set(tracked_new) & appeared:
                if provenance_origin(base / slug) in (None, own):
                    claimed.add(slug)
            tracked_new = claimed
    # Intersected with `after` throughout, so a slug the agent opened and
    # then deleted can't produce a card for a folder that is gone.
    new = sorted(appeared if tracked_new is None else appeared & set(tracked_new))
    touched = set(new)
    editable = after if tracked_edits is None else (after & set(tracked_edits))
    for slug in editable:
        previous = (before_mtimes or {}).get(slug)
        if previous is None:
            continue  # appeared this turn — `new` above already ruled on it
        try:
            if content_mtime_ns(base / slug) > previous:
                touched.add(slug)
        except OSError:
            continue
    for slug in touched:
        capture_agent_revision(base / slug, conversation_id=conversation_key)
    # A repair turn that makes no source change must still leave the
    # polling state. Check only artifacts with a handoff explicitly bound
    # to this conversation; unrelated queued work is never consumed.
    for slug in (after | repair_candidates) - touched:
        try:
            queued = has_queued_agent_repair(base / slug, conversation_key)
        except OSError:
            # A repairs journal that can't be read holds back only that
            # folder's capture; the check below keeps its row.
            continue
        if queued:
            capture_agent_revision(base / slug, conversation_id=conversation_key)
    # A capture can leave a repair queued (the artifact's lock stayed held,
    # or a read failed), so a repair folder the diff read counts as ruled on
    # only once a read finds nothing in it still queued to this
    # conversation. Otherwise its row stays for the next turn end.
    for slug in checked_repairs & after:
        try:
            still_queued = has_queued_agent_repair(base / slug, conversation_key)
        except OSError:
            still_queued = True
        if still_queued:
            checked_repairs.discard(slug)
    # Read once the captures ran, which can write a folder's id. The listing
    # covers the created folders too, so a row naming one of them is never
    # taken for a row whose folder is gone (record_listed_folders).
    return ArtifactChanges(
        created=new, touched=touched, created_ids=artifact_ids_below(base=base, slugs=new).ids,
        checked_repairs=checked_repairs,
        listed_ids=(
            listed_folder_ids(base=base, listing=whole_listing)
            if whole_listing is not None
            else ListedIds(ids={}, unreadable=frozenset())
        ),
    )


def cards_for_slugs(
    artifacts_base,
    slugs: list[str],
    *,
    scope: TenantScope | None,
    project_id: str | None = None,
    project_name: str = "",
    project_label: str = "",
) -> list[ArtifactCard]:
    """Inline-chat card payloads for the given slugs, order preserved.

    Uses the same per-folder card builder as the artifacts list
    (`services.artifacts.card_for_folder`), so inline cards and the panel can
    never disagree about what a turn produced or how an artifact opens.

    `project_id`/`project_name` are passed through to the card because that is
    how the client addresses an artifact in org mode (project + slug); a card
    without them would fall back to the path-based endpoints, which org mode
    fails closed. Best-effort per slug: an unreadable artifact is skipped.

    `project_name` must be the project's `name`: the card's serve URL is built
    from it, and the serve route resolves that segment by name
    (`services.artifacts._project_artifacts_base`). `project_label` is what the
    card shows instead, when the caller has one
    (`services.projects.display_label`). `scope` is the turn's; it only finds
    the serve URL's project by scanning when no `project_name` is given.
    """
    from cowork.services.artifacts import card_for_folder

    base = Path(artifacts_base)
    cards: list[ArtifactCard] = []
    for slug in slugs:
        try:
            card = card_for_folder(
                base / slug, len(cards),
                scope=scope,
                project_id=project_id, project_name=project_name,
                project_label=project_label,
                # Without this the inline card's serveUrl is rediscovered by
                # scanning the projects root, which cannot find a project
                # pointed at a folder the user chose: the artifacts panel
                # would offer "open in browser" and the turn card would not.
                # Identical for an allocated project, whose directory name and
                # project name are the same string.
                artifacts_base=base,
            )
        except Exception:
            logger.warning("Could not build inline card for artifact %r", slug, exc_info=True)
            continue
        if card is not None:
            cards.append(card)
    return cards


async def publish_and_card_turn_artifacts(
    artifacts_base,
    *,
    new_slugs: list[str],
    touched_slugs: set[str],
    scope,
    project_id: str | None = None,
    project_name: str = "",
    project_label: str = "",
    only_touched: bool = False,
) -> list[ArtifactCard]:
    """Reconcile publishes for this turn, then build the cards to emit.

    The second half of the end-of-turn artifact flow. The first half records
    what the turn created and produces the arguments. The in-process harness
    protects its cleanup with run_to_completion and records the
    turn_artifact_changes result in a database unit, including on error or
    Stop. The remote producer uses the same two artifact helpers.

    Shared by both producers. The in-process harness reaches it through
    `AntonHarness.stream_response`; on an org deployment that harness refuses to
    run at all and the turn happens on the remote worker, so
    `handlers.responses._produce_remote` calls this against the same shared
    artifacts tree the worker wrote to.

    Cards cover what THIS turn produced or touched. In org mode, `republished`
    also carries phase-two self-heal publishes — older artifacts from earlier
    conversations — and the stream reducer dedupes only within one message, so
    an edited (not new) artifact only cards there once its republish this turn
    actually succeeds; that keeps a self-heal from attaching last week's
    artifact to this answer.

    Outside org mode nothing is ever published — `autopublish_project_artifacts`
    is a no-op there — so `republished` is always empty, and gating touched
    (non-new) slugs on it would mean a local edit never cards at all.
    `cards_for_slugs` builds a card straight from the local folder regardless of
    publish state, so every slug this turn touched is cardable there.

    `only_touched` is for a remote turn whose pod named the artifacts its
    tools touched: the reconciler then opens and publishes only the turn's
    own folders. The rest of the project waits for its scheduled self-heal,
    which checks every artifact's location row whether or not autopublish is
    on, and publishes the rest of the project while it is. The producer asks
    for it as its record starts, not here
    (`artifact_autopublish.request_turn_end_self_heal`), so a turn end that
    never reaches this publish, after a Stop or a failed diff, still asks.
    """
    from cowork.services.artifact_autopublish import autopublish_project_artifacts

    republished = await autopublish_project_artifacts(
        artifacts_base, scope, touched=set(touched_slugs), project_id=project_id, only_touched=only_touched,
    )
    if getattr(scope, "org_mode", False):
        carded = set(new_slugs) | (republished & set(touched_slugs))
    else:
        carded = set(touched_slugs)
    from cowork.db.units import run_efs

    return await run_efs(
        partial(
            cards_for_slugs,
            artifacts_base, sorted(carded),
            scope=scope,
            project_id=project_id, project_name=project_name, project_label=project_label,
        ),
        scope=scope or LOCAL_SCOPE,
        deadline=False,
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


def remote_skill_draft_result(entry: dict) -> tuple[dict | None, list[str]]:
    """Card payload for a draft a remote pod reported, plus any drop reasons.

    The pod sends `{"slug", "files": {name: text}}`. This materializes it into a
    throwaway folder and reuses `_skill_draft_payload`, so a remote card is
    byte-identical to the desktop one instead of a second parser drifting from it.

    This is the trust boundary — the pod's payload is untrusted wire data:
    - the slug must be a valid skill name, or it could name any folder
    - filenames must be plain basenames, since `a/../../b` would escape the temp
      dir and let a turn write anywhere the server process can
    A bad entry is dropped and logged; one draft must never fail the turn. The
    reason is also returned (not just logged) so the caller can surface it in
    the chat stream — a lost draft, or a sibling file quietly excluded from an
    otherwise-saved one, must not be invisible to the user.

    Returns ``(payload, reasons)``. ``payload`` is None when the whole draft
    was rejected. ``reasons`` can be non-empty even when ``payload`` succeeded
    (e.g. one unsafe sibling file skipped out of several).
    """
    import tempfile

    from anton.core.tools.skill_format import SKILL_FILE, validate_name

    reasons: list[str] = []
    slug = entry.get("slug") if isinstance(entry, dict) else None
    files = entry.get("files") if isinstance(entry, dict) else None
    if not isinstance(slug, str) or not isinstance(files, dict):
        logger.warning("Remote skill draft: malformed entry, dropping it")
        return None, ["the agent sent a malformed draft; nothing was saved."]
    try:
        validate_name(slug)
    except ValueError:
        logger.warning("Remote skill draft: invalid slug %r, dropping it", slug)
        return None, [f"{slug!r} is not a valid skill name; nothing was saved."]
    try:
        with tempfile.TemporaryDirectory(prefix="cowork-remote-draft-") as tmp:
            folder = Path(tmp) / slug
            folder.mkdir()
            for name, text in files.items():
                if not isinstance(name, str) or not isinstance(text, str):
                    continue
                if name != Path(name).name or name in (".", ".."):
                    logger.warning("Remote skill draft %r: dropping unsafe filename %r", slug, name)
                    reasons.append(f"{slug!r}: skipped file {name!r} (unsafe path).")
                    continue
                (folder / name).write_text(text, encoding="utf-8")
            payload = _skill_draft_payload(folder)
    except (OSError, ValueError):
        logger.warning("Remote skill draft %r: could not build a card", slug, exc_info=True)
        return None, [f"{slug!r} could not be saved (internal error)."]
    # Covers every reason the shared builder rejects a folder — missing or
    # unparseable SKILL.md — without restating its preconditions here.
    if payload is None:
        logger.warning("Remote skill draft %r: no usable %s, dropping it", slug, SKILL_FILE)
        reasons.append(f"{slug!r}: no usable {SKILL_FILE}; nothing was saved.")
    return payload, reasons


def remote_skill_draft_payload(entry: dict) -> dict | None:
    """Card payload for a draft a remote pod reported, or None to drop it.

    Thin wrapper over `remote_skill_draft_result` for callers that only need
    the card and don't stream drop reasons anywhere (e.g. tests); see that
    docstring for the drop rules.
    """
    return remote_skill_draft_result(entry)[0]


def _seed_draft_from_store(folder: Path, slug: str) -> None:
    """Copy a saved skill's SKILL.md + sibling files into an empty draft folder.

    Lets "edit an existing skill" start from the stored version (Save then
    upserts it back) instead of the agent hunting for and mutating the live
    store in place. Best-effort: a missing store entry or copy failure just
    leaves the folder empty (a fresh draft), never raises into the turn.

    ponytail: copies top-level files only (skills are flat text) — mirrors
    `_skill_draft_payload`, which also skips subdirs.

    `slug` is agent-supplied: validate it, and key the store by the turn's
    ambient scope. A partitioned deployment with no scope bound, and org mode
    with an unpartitioned one, seed nothing rather than reading the unkeyed root.
    """
    from anton.core.tools.skill_format import SKILL_FILE, validate_name

    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.common.settings.user_settings import current_settings_scope
        from cowork.db.scoped import scoped_storage_root

        validate_name(slug)
        settings = get_app_settings()
        scope = current_settings_scope()
        if scope is None:
            if tenant_scope_required():
                return
        elif settings.tenancy_mode == "org" and not scope.org_mode:
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


# LLM-facing contract for the `create_skill_draft` tool. Registered by anton as
# a ToolDef; kept harness-neutral so any agent reads the same contract.
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
