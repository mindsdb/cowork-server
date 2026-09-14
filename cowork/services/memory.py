from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Iterator, Literal
from uuid import UUID

from fastapi import HTTPException, status
from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import (
    GlobalMemoryStore,
    PROJECT_SLOTS,
    ProjectMemoryStore,
    SlotRead,
)
from cowork.models.project import Project
from cowork.principal import Principal
from cowork.schemas.memory import MemoryResponse, MemoryScope
from cowork.schemas.shared_resources import (
    MutableResourceCapabilities,
    ResourceAttribution,
)
from cowork.services.shared_resources import (
    PROJECT,
    PROJECT_MEMORY,
    SharedResourceAccess,
    project_memory_resource_key,
    project_resource_key,
)

if TYPE_CHECKING:
    from anton.core.memory.hippocampus import Hippocampus


logger = logging.getLogger(__name__)


@contextmanager
def _project_slot_coordination(
    session: ScopedSession,
    access: SharedResourceAccess,
    project_id: UUID | None,
    category: MemorySlot,
) -> Iterator[tuple[Project, ProjectMemoryStore, str]]:
    """Pin a project's current path, then one memory slot, for a mutation.

    Project rename/delete take the parent lock first. Every contained mutation
    follows the same order and refreshes the row only after acquiring it, so a
    waiter resumes against the renamed path instead of recreating the old one.
    """
    if project_id is None:
        raise ValueError("project_id is required for project-scoped memory.")
    with access.coordination_lock(PROJECT, project_resource_key(project_id)):
        project = session.get(Project, project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found.")
        if session.scope.org_mode:
            session.refresh(project)
        store = ProjectMemoryStore(Path(project.path))
        key = project_memory_resource_key(project.id, category.value)
        with access.coordination_lock(PROJECT_MEMORY, key):
            yield project, store, key


def _slot_holds_content(state: SlotRead) -> bool:
    """Whether a slot holds something a first-author claim must not overwrite.

    Blank bytes count as unwritten, which is what lets the first real writer
    claim a slot anton created empty. A slot that cannot be read still occupies
    the name and still hides an authorship nobody can establish, so it counts as
    written rather than free.
    """
    return not state.readable or bool(state.content.strip())


def _restore_project_slot(
    store: ProjectMemoryStore,
    slot: MemorySlot,
    snapshot: SlotRead,
) -> None:
    """Put back the exact bytes a failed mutation replaced.

    A snapshot taken from a slot that could not be read holds no bytes to put
    back. Writing it anyway would truncate whatever is still there, so the
    failed mutation stands and the caller sees the original error.
    """
    if not snapshot.readable:
        logger.warning(
            "Project memory slot %s had no readable snapshot to restore",
            slot.value,
        )
        return
    store.restore_exact(slot, snapshot.content, existed=snapshot.exists)


def build_turn_memory(
    scope: TenantScope, project_path: str | None = None
) -> dict[str, dict[str, str]]:
    """Memory slots for a remote turn: ``{"global": {slot: text}, "project": {...}}``.

    Org-keyed through the passed `scope` — the remote producer binds no ambient
    scope, so reading it from there would hit the unkeyed root.
    """
    out: dict[str, dict[str, str]] = {}

    global_store = GlobalMemoryStore(scope=scope)
    if global_slots := {
        s.value: text for s in MemorySlot if (text := global_store.read(s).strip())
    }:
        out["global"] = global_slots

    if project_path:
        store = ProjectMemoryStore(Path(project_path))
        if slots := {
            s.value: text for s in PROJECT_SLOTS if (text := store.read(s).strip())
        }:
            out["project"] = slots

    return out


class _WireEngram(BaseModel):
    """One memory entry as a pod may send it. A pod runs model-authored code, so
    entries are untrusted and anything unparseable is skipped.

    The literals mirror anton's ``Engram`` (pinned by
    test_engram_vocabulary_matches_anton) and each is load-bearing: ``scope``
    decides personal vs shared-with-the-team, so an unrecognised value would
    quietly publish a personal note to the whole org, and ``source``/``confidence``
    are interpolated raw into the ``<!-- ... -->`` metadata tail.

    ``max_length`` on text is a sanity bound on what reaches disk, not a content
    limit: a stored slot is re-read into every later turn's payload, but anton
    decides what actually enters a prompt. There is deliberately no cap on entries
    per turn — it would lose real memories while an abusive pod just sends the
    maximum every turn.
    """

    model_config = ConfigDict(extra="ignore")

    text: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)
    ]
    kind: Literal["always", "never", "when", "lesson", "profile"]
    scope: Literal["global", "project"] = "global"
    topic: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    source: Literal["user", "llm", "consolidation"] = "llm"

    @field_validator("topic")
    @classmethod
    def _bound_topic(cls, value: str) -> str:
        # Truncated, not rejected: an over-long topic is a mislabelled memory, and
        # the memory is the part worth keeping. Hippocampus slugifies the charset.
        return value[:64]

    @property
    def targets_project(self) -> bool:
        """Identity is global-only; everything else honours the declared scope."""
        return self.kind != "profile" and self.scope == "project"

    @property
    def slot(self) -> MemorySlot:
        return MemorySlot.LESSONS if self.kind == "lesson" else MemorySlot.RULES


def _one_line(text: str) -> str:
    """Neutralize forged structure before it reaches the slot files, which store
    one entry per line with an HTML-comment metadata tail. Spaces and tabs inside
    a line are the user's formatting and are preserved.

    Mirrors Hippocampus._entry_text deliberately: this must hold whichever anton
    version is pinned, since the wire data is untrusted and the pin can lag.
    """
    text = text.replace("<!--", "<!-").replace("-->", "->")
    return " ".join(line.strip() for line in text.splitlines()).strip()


def _validated_engrams(entries: list[dict]) -> list[_WireEngram]:
    """The entries worth writing, with their text already neutralized.

    A pod's payload is untrusted, so anything unparseable, and anything whose
    text is only forged structure, is skipped here rather than at a slot file.
    """
    validated: list[_WireEngram] = []
    for raw in entries:
        try:
            entry = _WireEngram.model_validate(raw)
        except ValidationError:
            continue
        text = _one_line(entry.text)
        if not text:
            continue
        validated.append(entry.model_copy(update={"text": text}))
    return validated


def _encode_engram(hippocampus: "Hippocampus", entry: _WireEngram) -> None:
    """Write one entry through anton's own encoder for its kind."""
    if entry.kind == "profile":
        hippocampus.rewrite_identity([entry.text])
    elif entry.kind == "lesson":
        hippocampus.encode_lesson(entry.text, topic=entry.topic, source=entry.source)
    else:
        hippocampus.encode_rule(
            entry.text,
            kind=entry.kind,
            confidence=entry.confidence,
            source=entry.source,
        )


def _drop_project_engram(reason: str, key: str) -> bool:
    """Record why a shared entry was refused, and report it as not applied."""
    logger.warning("Dropped a remote-turn project memory entry for %s: %s", key, reason)
    return False


def apply_turn_memory(
    scope: TenantScope,
    project_path: str | None,
    entries: list[dict],
    *,
    access: SharedResourceAccess | None = None,
    project_id: UUID | None = None,
) -> int:
    """Persist engrams a remote turn asked to remember; returns how many passed
    validation (Hippocampus dedupes, so a write may still be a no-op).

    The pod reports intent — this is the only place a cloud turn's memory is
    written, and it goes through the caller's `scope`, so a compromised pod can't
    reach another org. Writes via anton's own Hippocampus rather than reproducing
    its markdown format.

    The caller's private global memory is written first and on its own. The two
    tiers answer to different owners: a shared project slot the caller may not
    touch, or one whose write fails, must not cost that caller the personal half
    of the same turn.
    """
    from anton.core.memory.hippocampus import Hippocampus

    validated = _validated_engrams(entries)

    global_hc = Hippocampus(GlobalMemoryStore(scope=scope).root)
    applied = 0
    for entry in validated:
        if entry.targets_project:
            continue
        _encode_engram(global_hc, entry)
        applied += 1

    project_entries = [entry for entry in validated if entry.targets_project]
    if not project_entries:
        return applied
    if project_path is None:
        logger.warning(
            "Dropped %d remote-turn project memory entr(ies): the turn has no project",
            len(project_entries),
        )
        return applied

    project_store = ProjectMemoryStore(Path(project_path))
    if not scope.org_mode:
        project_hc = Hippocampus(project_store.root)
        for entry in project_entries:
            _encode_engram(project_hc, entry)
            applied += 1
        return applied

    # Cloud turns must carry the request's trusted identity all the way to this
    # detached write. A bare TenantScope has no admin role, so it is not enough
    # to authorize a shared mutation.
    if access is None or project_id is None or not access.has_trusted_actor:
        logger.warning(
            "Dropped %d remote-turn project memory entr(ies): "
            "the turn carries no trusted organization identity",
            len(project_entries),
        )
        return applied
    for entry in project_entries:
        if _apply_project_engram(access, project_id, entry):
            applied += 1
    return applied


def _apply_project_engram(
    access: SharedResourceAccess,
    project_id: UUID,
    entry: _WireEngram,
) -> bool:
    """Persist one shared entry under the slot's authorship gate.

    Returns whether it reached disk. A slot the caller may not write is dropped
    and logged rather than raised, so one refused entry costs only itself.
    """
    from anton.core.memory.hippocampus import Hippocampus

    slot = entry.slot
    with ExitStack() as coordination:
        try:
            _project, store, key = coordination.enter_context(
                _project_slot_coordination(
                    access.session,
                    access,
                    project_id,
                    slot,
                )
            )
            before = store.read_state(slot)
            access.recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: _slot_holds_content(store.read_state(slot)),
            )
            attribution_exists = access.has_attribution(PROJECT_MEMORY, key)
            claimed = None
            claim_token = None
            if _slot_holds_content(before) or attribution_exists:
                if not access.can_change(access.creator_id(PROJECT_MEMORY, key)):
                    return _drop_project_engram(
                        "the turn's actor is not its author",
                        key,
                    )
                if access.claim_is_pending(PROJECT_MEMORY, key):
                    return _drop_project_engram(
                        "another write is establishing it",
                        key,
                    )
                attribution = coordination.enter_context(
                    access.mutation_lock(
                        PROJECT_MEMORY,
                        key,
                        resource_exists=lambda: _slot_holds_content(
                            store.read_state(slot)
                        ),
                    )
                )
                if attribution is None or not access.can_change(
                    attribution.created_by_id
                ):
                    return _drop_project_engram(
                        "the turn's actor is not its author",
                        key,
                    )
                before = store.read_state(slot)
            else:
                claimed, claim_token = access.reserve_claim(PROJECT_MEMORY, key)
                if claimed is None or not access.can_change(claimed.created_by_id):
                    return _drop_project_engram("another member became its author", key)
                if claim_token is None:
                    if claimed.pending_claim_token:
                        return _drop_project_engram(
                            "another write is establishing it",
                            key,
                        )
                    attribution = coordination.enter_context(
                        access.mutation_lock(
                            PROJECT_MEMORY,
                            key,
                            resource_exists=lambda: _slot_holds_content(
                                store.read_state(slot)
                            ),
                        )
                    )
                    if attribution is None or not access.can_change(
                        attribution.created_by_id
                    ):
                        return _drop_project_engram(
                            "the turn's actor is not its author",
                            key,
                        )
                    before = store.read_state(slot)
        except Exception:
            access.session.rollback()
            raise

        try:
            _encode_engram(Hippocampus(store.root), entry)
        except Exception:
            try:
                _restore_project_slot(store, slot, before)
            except Exception:
                logger.exception(
                    "Could not restore project memory after a failed write"
                )
            if claim_token is not None and claimed is not None:
                try:
                    access.release_claim(claimed, claim_token=claim_token)
                except Exception:
                    logger.exception("Could not release a failed project-memory claim")
            raise

        try:
            after = store.read_state(slot)
            if after.content != before.content or after.exists != before.exists:
                if claim_token is not None and claimed is not None:
                    finalized = access.finalize_claim(
                        claimed,
                        claim_token,
                        action="create",
                    )
                    if finalized is None:
                        raise RuntimeError(
                            "Project-memory claim changed before it could be finalized"
                        )
                else:
                    # Attributed slots, legacy slots, and a race whose winner
                    # finalized before this write all record the current actor
                    # as an editor.
                    access.record_update(
                        PROJECT_MEMORY,
                        key,
                        action="update",
                    )
            elif claim_token is not None and claimed is not None:
                access.release_claim(claimed, claim_token=claim_token)
        except Exception:
            try:
                _restore_project_slot(store, slot, before)
            except Exception:
                logger.exception(
                    "Could not restore project memory after an audit failure"
                )
            if claim_token is not None and claimed is not None:
                try:
                    access.release_claim(claimed, claim_token=claim_token)
                except Exception:
                    logger.exception("Could not release a failed project-memory claim")
            raise
    return True


class MemoryService:
    def __init__(
        self,
        session: ScopedSession,
        principal: Principal | None = None,
    ) -> None:
        self.session = session
        self.access = SharedResourceAccess(session, principal)
        # Org mode → per-(org, user); the store keys itself (see GlobalMemoryStore).
        self._global_store = GlobalMemoryStore(scope=session.scope)

    async def get_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        return self.get_memory_sync(scope, category, project_id)

    def get_memory_sync(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        scope = MemoryScope(scope)
        category = MemorySlot(category)
        state = self._read_for_response(scope, category, project_id)
        return self._response(
            scope,
            category,
            state.content,
            project_id,
            read_verified=state.readable,
        )

    async def update_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        return self.update_memory_sync(scope, category, content, project_id)

    def update_memory_sync(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        scope = MemoryScope(scope)
        category = MemorySlot(category)
        self._write(scope, category, content, project_id)
        return self._response(scope, category, content, project_id)

    async def delete_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> None:
        self.delete_memory_sync(scope, category, project_id)

    def delete_memory_sync(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> None:
        scope = MemoryScope(scope)
        category = MemorySlot(category)
        self._delete(scope, category, project_id)

    async def list_memory(self, project_id: UUID | None = None) -> list[MemoryResponse]:
        return self.list_memory_sync(project_id)

    def list_memory_sync(
        self,
        project_id: UUID | None = None,
    ) -> list[MemoryResponse]:
        items: list[MemoryResponse] = []

        for category in MemorySlot:
            items.append(
                self._response(
                    MemoryScope.global_,
                    category,
                    self._global_store.read(category),
                    None,
                )
            )

        projects = list(self.session.exec(self.session.select(Project)).all())
        if project_id is not None:
            projects = [p for p in projects if p.id == project_id]

        for project in projects:
            store = ProjectMemoryStore(Path(project.path))
            for category in PROJECT_SLOTS:
                state = store.read_state(category)
                items.append(
                    self._response(
                        MemoryScope.project,
                        category,
                        state.content,
                        project.id,
                        read_verified=state.readable,
                    )
                )

        if project_id is not None:
            items = [
                item
                for item in items
                if item.scope == MemoryScope.global_ or item.project_id == project_id
            ]

        return items

    def _read(
        self, scope: MemoryScope, category: MemorySlot, project_id: UUID | None
    ) -> str:
        if scope == MemoryScope.global_:
            return self._global_store.read(category)
        project = self._resolve_project(project_id)
        return ProjectMemoryStore(Path(project.path)).read(category)

    def _read_for_response(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None,
    ) -> SlotRead:
        if scope == MemoryScope.global_:
            content = self._global_store.read(category)
            return SlotRead(exists=bool(content), readable=True, content=content)
        project = self._resolve_project(project_id)
        # An unreadable slot keeps the historical fail-soft read body, but the
        # response must not advertise a mutation capability when authorization
        # could not establish whether a legacy shared slot exists.
        return ProjectMemoryStore(Path(project.path)).read_state(category)

    def _write(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None,
    ) -> None:
        if scope == MemoryScope.global_:
            self._global_store.write(category, content)
            return
        self.access.require_actor()
        with _project_slot_coordination(
            self.session,
            self.access,
            project_id,
            category,
        ) as (_project, store, key):
            if not self.session.scope.org_mode:
                store.write(category, content)
                return

            # This read fails closed on purpose: a slot whose bytes cannot be
            # read is a slot whose author cannot be established, and a write
            # must not proceed on the assumption that it is empty.
            exists, content_before = store.read_checked(category)
            before = SlotRead(exists=exists, readable=True, content=content_before)
            self.access.recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: _slot_holds_content(store.read_state(category)),
            )
            attribution_exists = self.access.has_attribution(PROJECT_MEMORY, key)
            if _slot_holds_content(before) or attribution_exists:
                self.access.require_change(
                    self.access.creator_id(PROJECT_MEMORY, key),
                    detail="Only the project-memory author or an organization admin can edit this slot",
                )
                self._write_existing_project_slot(store, category, key, content)
                return
            if not content.strip():
                return

            claimed, claim_token = self.access.reserve_claim(PROJECT_MEMORY, key)
            if claimed is None:
                raise RuntimeError("Project-memory ownership could not be established")
            self.access.require_change(
                claimed.created_by_id,
                detail="Another member became this project-memory slot's author",
            )
            if claim_token is None:
                if claimed.pending_claim_token:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Another write is establishing this project-memory slot",
                    )
                self._write_existing_project_slot(store, category, key, content)
                return

            try:
                store.write(category, content)
                finalized = self.access.finalize_claim(
                    claimed,
                    claim_token,
                    action="create",
                )
                if finalized is None:
                    raise RuntimeError(
                        "Project-memory claim changed before it could be finalized"
                    )
            except Exception:
                try:
                    _restore_project_slot(store, category, before)
                except Exception:
                    logger.exception(
                        "Could not restore project memory after a failed mutation"
                    )
                try:
                    self.access.session.rollback()
                    self.access.release_claim(claimed, claim_token=claim_token)
                except Exception:
                    logger.exception("Could not release a failed project-memory claim")
                raise

    def _write_existing_project_slot(
        self,
        store: ProjectMemoryStore,
        category: MemorySlot,
        key: str,
        content: str,
    ) -> None:
        with self.access.mutation_lock(
            PROJECT_MEMORY,
            key,
            resource_exists=lambda: _slot_holds_content(store.read_state(category)),
        ) as attribution:
            if attribution is None:
                raise RuntimeError("Project-memory mutation lock was not established")
            self.access.require_change(
                attribution.created_by_id,
                detail="Only the project-memory author or an organization admin can edit this slot",
            )
            before = store.read_state(category)
            try:
                store.write(category, content)
                self.access.record_update(
                    PROJECT_MEMORY,
                    key,
                    action="clear" if not content.strip() else "update",
                )
            except Exception:
                try:
                    _restore_project_slot(store, category, before)
                except Exception:
                    logger.exception(
                        "Could not restore project memory after a failed mutation"
                    )
                raise

    def _delete(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None,
    ) -> None:
        if scope == MemoryScope.global_:
            self._global_store.delete(category)
            return
        self.access.require_actor()
        with _project_slot_coordination(
            self.session,
            self.access,
            project_id,
            category,
        ) as (_project, store, key):
            if not self.session.scope.org_mode:
                store.delete(category)
                return
            # A delete never decodes the slot: a symlink squatting the name, or
            # bytes that are not UTF-8, still gets unlinked for whoever owns the
            # slot, and is still refused to everyone else. Reading it as content
            # would only turn a squatted slot into a failed request.
            before = store.read_state(category)
            self.access.recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: _slot_holds_content(store.read_state(category)),
            )
            attribution_exists = self.access.has_attribution(PROJECT_MEMORY, key)
            if not _slot_holds_content(before) and not attribution_exists:
                store.delete(category)
                return
            self.access.require_change(
                self.access.creator_id(PROJECT_MEMORY, key),
                detail="Only the project-memory author or an organization admin can delete this slot",
            )
            with self.access.mutation_lock(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: _slot_holds_content(store.read_state(category)),
            ) as attribution:
                if attribution is None:
                    raise RuntimeError(
                        "Project-memory mutation lock was not established"
                    )
                self.access.require_change(
                    attribution.created_by_id,
                    detail="Only the project-memory author or an organization admin can delete this slot",
                )
                before = store.read_state(category)
                try:
                    store.delete(category)
                    self.access.record_update(
                        PROJECT_MEMORY,
                        key,
                        action="clear",
                    )
                except Exception:
                    try:
                        _restore_project_slot(store, category, before)
                    except Exception:
                        logger.exception(
                            "Could not restore project memory after a failed clear"
                        )
                    raise

    def _resolve_project(self, project_id: UUID | None) -> Project:
        if project_id is None:
            raise ValueError("project_id is required for project-scoped memory.")
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found.")
        return project

    def _response(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None,
        *,
        read_verified: bool = True,
    ) -> MemoryResponse:
        if scope == MemoryScope.global_:
            # Private per-member memory: nobody else can see it, so there is no
            # shared authorship to report and nothing to gate.
            return MemoryResponse(
                scope=scope,
                category=category,
                content=content,
                project_id=None,
                attribution=ResourceAttribution(
                    created_by=None,
                    last_modified_by=None,
                    last_modified_at=None,
                ),
                capabilities=MutableResourceCapabilities(
                    can_edit=True,
                    can_delete=True,
                ),
            )

        if project_id is None:
            raise ValueError("project_id is required for project-scoped memory.")
        key = project_memory_resource_key(project_id, category.value)
        store = ProjectMemoryStore(Path(self._resolve_project(project_id).path))

        if (
            self.session.scope.org_mode
            and read_verified
            and self.access.has_trusted_actor
        ):
            # A read takes no resource lock of its own. It reports the content
            # the caller already read, and recovery only reaches for the
            # coordination lock once a claim's lease has actually run out, so a
            # listing costs neither a lock nor a connection per slot.
            self.access.recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: _slot_holds_content(store.read_state(category)),
            )
        pending = self.access.claim_is_pending(PROJECT_MEMORY, key)
        creator_id = self.access.creator_id(PROJECT_MEMORY, key)
        attribution_exists = self.access.has_attribution(PROJECT_MEMORY, key)
        can_change = self.access.can_change(creator_id)
        # An empty, never-authored slot is open for a member to become its
        # first author. A non-empty unattributed slot is legacy/admin-only.
        if (
            read_verified
            and creator_id is None
            and not content.strip()
            and not attribution_exists
        ):
            can_change = self.access.has_trusted_actor
        # A live reservation is not authorship yet. It cannot advertise a
        # capability to its reserver (or an admin) that PUT will reject.
        if not read_verified or pending:
            can_change = False
        return MemoryResponse(
            scope=scope,
            category=category,
            content=content,
            project_id=project_id,
            # A pending claim is hidden by `attribution` itself; an unattributed
            # slot falls back to the file's own mtime.
            attribution=self.access.attribution(
                PROJECT_MEMORY,
                key,
                fallback_modified_at=store.modified_at(category),
            ),
            capabilities=MutableResourceCapabilities(
                can_edit=can_change,
                can_delete=can_change,
            ),
        )
