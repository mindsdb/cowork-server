from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Iterator, Literal
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
)
from cowork.models.project import Project
from cowork.principal import Principal
from cowork.schemas.memory import MemoryResponse, MemoryScope
from cowork.services.shared_resources import (
    PROJECT,
    PROJECT_MEMORY,
    SharedResourceAccess,
    project_memory_resource_key,
    project_resource_key,
)


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


def _restore_project_slot(
    store: ProjectMemoryStore,
    slot: MemorySlot,
    content: str,
    *,
    existed: bool,
) -> None:
    store.restore_exact(slot, content, existed=existed)


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


def _one_line(text: str) -> str:
    """Neutralize forged structure before it reaches the slot files, which store
    one entry per line with an HTML-comment metadata tail. Spaces and tabs inside
    a line are the user's formatting and are preserved.

    Mirrors Hippocampus._entry_text deliberately: this must hold whichever anton
    version is pinned, since the wire data is untrusted and the pin can lag.
    """
    text = text.replace("<!--", "<!-").replace("-->", "->")
    return " ".join(line.strip() for line in text.splitlines()).strip()


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
    """
    from anton.core.memory.hippocampus import Hippocampus

    global_hc = Hippocampus(GlobalMemoryStore(scope=scope).root)
    project_store = ProjectMemoryStore(Path(project_path)) if project_path else None
    project_hc = Hippocampus(project_store.root) if project_store else None

    applied = 0
    for raw in entries:
        try:
            entry = _WireEngram.model_validate(raw)
        except ValidationError:
            continue

        text = _one_line(entry.text)
        # Identity is global-only; everything else honours the declared scope.
        project_entry = entry.kind != "profile" and entry.scope == "project"
        target = project_hc if project_entry else global_hc
        if not text or target is None:
            continue

        slot = MemorySlot.LESSONS if entry.kind == "lesson" else MemorySlot.RULES
        attribution_exists = False
        claim_token = None
        claimed = None
        before = ""
        before_exists = False
        key = ""
        mutation_context = None
        coordination_context: ExitStack | None = None
        if project_entry and scope.org_mode:
            # Cloud turns must carry the request's trusted identity all the way
            # to this detached write.  A bare TenantScope has no admin role, so
            # it is not enough to authorize a shared mutation.
            if (
                access is None
                or project_id is None
                or project_store is None
                or not access.has_trusted_actor
            ):
                continue
            coordination_context = ExitStack()
            _project, project_store, key = coordination_context.enter_context(
                _project_slot_coordination(
                    access.session,
                    access,
                    project_id,
                    slot,
                )
            )
            project_hc = Hippocampus(project_store.root)
            target = project_hc
            before_exists, before = project_store.read_checked(slot)
            has_meaningful_content = bool(before.strip())
            access.recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: bool(
                    project_store.read_checked(slot)[1].strip()
                ),
            )
            creator_id = access.creator_id(PROJECT_MEMORY, key)
            attribution_exists = access.has_attribution(PROJECT_MEMORY, key)
            if has_meaningful_content or attribution_exists:
                if not access.can_change(creator_id):
                    coordination_context.close()
                    continue
                if access.claim_is_pending(PROJECT_MEMORY, key):
                    coordination_context.close()
                    continue
                mutation_context = access.mutation_lock(
                    PROJECT_MEMORY,
                    key,
                    resource_exists=lambda: bool(
                        project_store.read_checked(slot)[1].strip()
                    ),
                )
                attribution = mutation_context.__enter__()
                if attribution is None or not access.can_change(
                    attribution.created_by_id
                ):
                    mutation_context.__exit__(None, None, None)
                    coordination_context.close()
                    continue
                try:
                    before_exists, before = project_store.read_checked(slot)
                except Exception:
                    mutation_context.__exit__(None, None, None)
                    coordination_context.close()
                    raise
            else:
                claimed, claim_token = access.reserve_claim(
                    PROJECT_MEMORY,
                    key,
                )
                if claimed is None or not access.can_change(claimed.created_by_id):
                    coordination_context.close()
                    continue
                if claim_token is None:
                    if claimed.pending_claim_token:
                        coordination_context.close()
                        continue
                    mutation_context = access.mutation_lock(
                        PROJECT_MEMORY,
                        key,
                        resource_exists=lambda: bool(
                            project_store.read_checked(slot)[1].strip()
                        ),
                    )
                    attribution = mutation_context.__enter__()
                    if attribution is None or not access.can_change(
                        attribution.created_by_id
                    ):
                        mutation_context.__exit__(None, None, None)
                        coordination_context.close()
                        continue
                    try:
                        before_exists, before = project_store.read_checked(slot)
                    except Exception:
                        mutation_context.__exit__(None, None, None)
                        coordination_context.close()
                        raise

        try:
            if entry.kind == "profile":
                target.rewrite_identity([text])
            elif entry.kind == "lesson":
                target.encode_lesson(text, topic=entry.topic, source=entry.source)
            else:
                target.encode_rule(
                    text,
                    kind=entry.kind,
                    confidence=entry.confidence,
                    source=entry.source,
                )
        except Exception:
            if project_entry and scope.org_mode and project_store is not None:
                try:
                    _restore_project_slot(
                        project_store,
                        slot,
                        before,
                        existed=before_exists,
                    )
                except Exception:
                    logger.exception(
                        "Could not restore project memory after a failed write"
                    )
            if claim_token is not None and claimed is not None:
                try:
                    access.release_claim(claimed, claim_token=claim_token)
                except Exception:
                    logger.exception("Could not release a failed project-memory claim")
            if mutation_context is not None:
                mutation_context.__exit__(None, None, None)
            if coordination_context is not None:
                coordination_context.close()
            raise
        if project_entry and scope.org_mode and project_store is not None:
            try:
                after_exists, after = project_store.read_checked(slot)
                if after != before or after_exists != before_exists:
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
                        # Attributed slots, legacy slots, and a race whose
                        # winner finalized before this write all record the
                        # current actor as an editor.
                        access.record_update(
                            PROJECT_MEMORY,
                            key,
                            action="update",
                        )
                elif claim_token is not None and claimed is not None:
                    access.release_claim(claimed, claim_token=claim_token)
            except Exception:
                try:
                    _restore_project_slot(
                        project_store,
                        slot,
                        before,
                        existed=before_exists,
                    )
                except Exception:
                    logger.exception(
                        "Could not restore project memory after an audit failure"
                    )
                if claim_token is not None and claimed is not None:
                    try:
                        access.release_claim(claimed, claim_token=claim_token)
                    except Exception:
                        logger.exception(
                            "Could not release a failed project-memory claim"
                        )
                if mutation_context is not None:
                    mutation_context.__exit__(None, None, None)
                if coordination_context is not None:
                    coordination_context.close()
                raise
            if mutation_context is not None:
                mutation_context.__exit__(None, None, None)
        if coordination_context is not None:
            coordination_context.close()
        applied += 1

    return applied


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
        scope = MemoryScope(scope)
        category = MemorySlot(category)
        content, read_verified = self._read_for_response(scope, category, project_id)
        return self._response(
            scope,
            category,
            content,
            project_id,
            read_verified=read_verified,
        )

    async def update_memory(
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
        scope = MemoryScope(scope)
        category = MemorySlot(category)
        self._delete(scope, category, project_id)

    async def list_memory(self, project_id: UUID | None = None) -> list[MemoryResponse]:
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
                try:
                    _exists, content = store.read_checked(category)
                    read_verified = True
                except (OSError, UnicodeError):
                    content = ""
                    read_verified = False
                items.append(
                    self._response(
                        MemoryScope.project,
                        category,
                        content,
                        project.id,
                        read_verified=read_verified,
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
    ) -> tuple[str, bool]:
        if scope == MemoryScope.global_:
            return self._global_store.read(category), True
        project = self._resolve_project(project_id)
        try:
            _exists, content = ProjectMemoryStore(Path(project.path)).read_checked(
                category
            )
        except (OSError, UnicodeError):
            # Preserve the historical fail-soft read body, but do not advertise
            # a mutation capability when authorization could not establish
            # whether a legacy shared slot exists.
            return "", False
        return content, True

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

            before_exists, before = store.read_checked(category)
            has_meaningful_content = bool(before.strip())
            self.access.recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: bool(
                    store.read_checked(category)[1].strip()
                ),
            )
            attribution_exists = self.access.has_attribution(PROJECT_MEMORY, key)
            if has_meaningful_content or attribution_exists:
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
                    _restore_project_slot(
                        store,
                        category,
                        before,
                        existed=before_exists,
                    )
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
            resource_exists=lambda: bool(store.read_checked(category)[1].strip()),
        ) as attribution:
            if attribution is None:
                raise RuntimeError("Project-memory mutation lock was not established")
            self.access.require_change(
                attribution.created_by_id,
                detail="Only the project-memory author or an organization admin can edit this slot",
            )
            before_exists, before = store.read_checked(category)
            try:
                store.write(category, content)
                self.access.record_update(
                    PROJECT_MEMORY,
                    key,
                    action="clear" if not content.strip() else "update",
                )
            except Exception:
                try:
                    _restore_project_slot(
                        store,
                        category,
                        before,
                        existed=before_exists,
                    )
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
            before_exists, before = store.read_checked(category)
            has_meaningful_content = bool(before.strip())
            self.access.recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: bool(
                    store.read_checked(category)[1].strip()
                ),
            )
            attribution_exists = self.access.has_attribution(PROJECT_MEMORY, key)
            if not has_meaningful_content and not attribution_exists:
                store.delete(category)
                return
            self.access.require_change(
                self.access.creator_id(PROJECT_MEMORY, key),
                detail="Only the project-memory author or an organization admin can delete this slot",
            )
            with self.access.mutation_lock(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: bool(
                    store.read_checked(category)[1].strip()
                ),
            ) as attribution:
                if attribution is None:
                    raise RuntimeError(
                        "Project-memory mutation lock was not established"
                    )
                self.access.require_change(
                    attribution.created_by_id,
                    detail="Only the project-memory author or an organization admin can delete this slot",
                )
                before_exists, before = store.read_checked(category)
                try:
                    store.delete(category)
                    self.access.record_update(
                        PROJECT_MEMORY,
                        key,
                        action="clear",
                    )
                except Exception:
                    try:
                        _restore_project_slot(
                            store,
                            category,
                            before,
                            existed=before_exists,
                        )
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
            return MemoryResponse(
                scope=scope,
                category=category,
                content=content,
                project_id=None,
                attribution={
                    "createdBy": None,
                    "lastModifiedBy": None,
                    "lastModifiedAt": None,
                },
                capabilities={"canEdit": True, "canDelete": True},
            )

        if project_id is None:
            raise ValueError("project_id is required for project-scoped memory.")
        key = project_memory_resource_key(project_id, category.value)

        def build_project_response(
            store: ProjectMemoryStore,
            current_content: str,
            current_read_verified: bool,
            *,
            recover: bool,
        ) -> MemoryResponse:
            if recover:
                self.access.recover_stale_claim(
                    PROJECT_MEMORY,
                    key,
                    resource_exists=lambda: bool(
                        store.read_checked(category)[1].strip()
                    ),
                )
            pending = self.access.claim_is_pending(PROJECT_MEMORY, key)
            creator_id = self.access.creator_id(PROJECT_MEMORY, key)
            attribution_exists = self.access.has_attribution(PROJECT_MEMORY, key)
            can_change = self.access.can_change(creator_id)
            # An empty, never-authored slot is open for a member to become its
            # first author. A non-empty unattributed slot is legacy/admin-only.
            if (
                current_read_verified
                and creator_id is None
                and not current_content.strip()
                and not attribution_exists
            ):
                can_change = self.access.has_trusted_actor
            # A live reservation is not authorship yet. It cannot advertise a
            # capability to its reserver (or an admin) that PUT will reject.
            if not current_read_verified or pending:
                can_change = False
            modified_at = None
            try:
                slot_path = store.root / (
                    "lessons.md" if category == MemorySlot.LESSONS else "rules.md"
                )
                if slot_path.is_file():
                    modified_at = datetime.fromtimestamp(
                        slot_path.stat().st_mtime,
                        tz=timezone.utc,
                    )
            except OSError:
                pass
            attribution = (
                {
                    "createdBy": None,
                    "lastModifiedBy": None,
                    "lastModifiedAt": None,
                }
                if pending
                else self.access.attribution(
                    PROJECT_MEMORY,
                    key,
                    fallback_modified_at=modified_at,
                )
            )
            return MemoryResponse(
                scope=scope,
                category=category,
                content=current_content,
                project_id=project_id,
                attribution=attribution,
                capabilities={"canEdit": can_change, "canDelete": can_change},
            )

        if (
            self.session.scope.org_mode
            and read_verified
            and self.access.has_trusted_actor
        ):
            with _project_slot_coordination(
                self.session,
                self.access,
                project_id,
                category,
            ) as (_project, store, _key):
                try:
                    _exists, content = store.read_checked(category)
                except (OSError, UnicodeError):
                    content = ""
                    read_verified = False
                return build_project_response(
                    store,
                    content,
                    read_verified,
                    recover=read_verified,
                )

        project = self._resolve_project(project_id)
        store = ProjectMemoryStore(Path(project.path))
        return build_project_response(
            store,
            content,
            read_verified,
            recover=False,
        )
