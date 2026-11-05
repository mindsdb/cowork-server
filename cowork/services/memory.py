from __future__ import annotations

import logging
import os
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Iterator, Literal, NamedTuple
from uuid import UUID

from fastapi import HTTPException, status
from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)

from cowork.common.paths import dir_lstat, dir_rmtree, dir_unlink, opened_subdir_nofollow, pinned_dir
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import (
    ScopedSession,
    TenantScope,
    storage_segment,
    user_store_segments,
    user_store_tiers_segments,
)
from cowork.harnesses.memory.policy import MEMORY_OFF_KEPT, resolve_memory_policy
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import (
    MEMORY_STORE,
    PROJECT_TIERS_DIRNAME,
    GlobalMemoryStore,
    PROJECT_SLOTS,
    MemoryStore,
    ProjectMemoryStore,
    SlotRead,
    deployment_memory_root,
    deployment_memory_store,
    nested_project_memory_store,
    project_memory_store,
    project_tier_ids,
)
from cowork.managed import managed_folder
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
        store = _project_store(scope=session.scope, project=project)
        key = project_memory_resource_key(project.id, category.value)
        with access.coordination_lock(PROJECT_MEMORY, key):
            yield project, store, key


def _project_store(*, scope: TenantScope, project: Project) -> ProjectMemoryStore:
    """The project tier ``scope`` reads and writes for ``project``
    (project_memory_store): a person's own tier inside their global tier where
    memory is per person, the deployment's tier inside its global tier where
    memory is shared, the operator view included, and the project's
    ``.anton/memory`` for the operator view where memory is per person, on the
    desktop and in hosted org mode."""
    return project_memory_store(
        scope=scope, project_path=Path(project.path), project_id=project.id
    )


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


class PersonMemoryResetError(ValueError):
    """A reset named no person whose memory is their own to remove."""


@dataclass(frozen=True)
class PersonMemoryReset:
    """What ``reset_person_memory`` removed."""

    # The person's global tier, which holds every project tier of theirs in a
    # personal partition.
    root: Path
    # False when the person had no memory yet, so nothing was there to remove.
    removed: bool


def reset_person_memory(scope: TenantScope) -> PersonMemoryReset:
    """Remove all of one person's learned memory.

    Keyed by ``scope`` alone, so an operator command and a later self-service
    route reach the same folder. A personal partition keeps every project tier
    under the person's global tier (project_memory_store), so removing that
    one folder clears both tiers. In org mode the person's own memory is the
    global tier; project memory there is shared per project and stays.

    Refuses a scope whose memory is not one person's: no user in scope, the
    operator view, the desktop, or a deployment that keeps memory shared,
    where the folder below would be everyone's base tier.
    """
    if not (scope.partitioned_for(PersonalResource.MEMORY) and scope.user_id):
        raise PersonMemoryResetError(
            "Memory reset needs a person whose memory is kept per person"
        )
    root = GlobalMemoryStore(scope=scope).root
    try:
        # Pinned without following a link at `users`, then removed by name
        # under that handle: a link planted at the person's own folder is
        # unlinked, never followed into another tier.
        with pinned_dir(root.parent, nofollow_base=True) as users:
            entry = dir_lstat(users, root.name)
            if stat.S_ISDIR(entry.st_mode):
                dir_rmtree(users, root.name)
            else:
                dir_unlink(users, root.name)
    except FileNotFoundError:
        return PersonMemoryReset(root=root, removed=False)
    logger.info("memory: removed a person's memory at %s", root)
    return PersonMemoryReset(root=root, removed=True)


def _real_child_dirs(path: Path) -> list[str]:
    """The names of the directories in ``path``, links left out; none when
    ``path`` is missing."""
    try:
        with os.scandir(path) as entries:
            return sorted(entry.name for entry in entries if entry.is_dir(follow_symlinks=False))
    except (FileNotFoundError, NotADirectoryError):
        return []


class PersonTier(NamedTuple):
    """One person's memory tier: their partition and their sub."""

    partition: str
    sub: str


def _remove_tier(*, base: Path, parents: tuple[str, ...], name: str) -> bool:
    """Remove the tier ``name`` in ``base/<parents...>``; False when it is missing.

    Every level below ``base`` opens without following a link
    (opened_subdir_nofollow), and a link in the tier's place is unlinked, never
    followed: the agent's code runs as the server's user and could plant one.
    """
    try:
        with opened_subdir_nofollow(base, *parents) as folder:
            if stat.S_ISDIR(dir_lstat(folder, name).st_mode):
                dir_rmtree(folder, name)
            else:
                dir_unlink(folder, name)
    except FileNotFoundError:
        return False
    return True


def remove_project_memory_tiers(*, scope: TenantScope, project_id: UUID) -> int:
    """Remove a deleted project's memory tiers; return how many.

    On a deployment that trusts identity headers a project's tiers can live
    inside global tiers (NestedProjectMemoryStore), outside the project's
    folder, so deleting the project would leave them behind: a person's tier
    listed nowhere, and the deployment's tier listed to every person while
    memory is shared. Both kinds go whichever way COWORK_PERSONAL_RESOURCES
    treats memory now, since a tier from before a switch stays on disk: the
    deployment's tier for the project, and each person's. Where projects are
    personal only the deleter can have a tier of their own; where they are
    shared, or the operator view deletes, every person's tier for that project
    id goes. Everywhere else project memory sits in the project's own folder
    and goes with it.

    The scope's ids are checked before anything is removed. The deployment's
    tier goes first, and an error there is raised. A person's tier that can't
    be removed, such as one behind a link planted in their chain, is logged
    and skipped, so it never keeps another person's tier.
    """
    settings = get_app_settings()
    if not settings.trust_identity_headers:
        return 0
    name = str(UUID(str(project_id)))
    shared = Path(settings.storage.shared_root)
    # <shared>/<partition>/memory/users/<sub>, the layout scoped_user_storage_root
    # gives a person's global tier (user_store_segments).
    if scope.partitioned_for(PersonalResource.PROJECTS):
        # The scope's ids come from the request, so each must be one folder
        # name, refused here before any tier goes; the walk below takes names
        # that are already folders.
        people = (
            [PersonTier(storage_segment(scope.org_id, what="partition"), storage_segment(scope.user_id, what="user"))]
            if scope.org_id and scope.user_id
            else []
        )
    else:
        people = [
            PersonTier(partition, sub)
            for partition in _real_child_dirs(shared)
            for sub in _real_child_dirs(shared.joinpath(*user_store_tiers_segments(partition, store=MEMORY_STORE)))
        ]
    removed = 0
    if _remove_tier(base=deployment_memory_root(), parents=(PROJECT_TIERS_DIRNAME,), name=name):
        removed += 1
    for tier in people:
        parents = (
            *user_store_segments(tier.partition, tier.sub, store=MEMORY_STORE),
            PROJECT_TIERS_DIRNAME,
        )
        try:
            if _remove_tier(base=shared, parents=parents, name=name):
                removed += 1
        except OSError:
            logger.warning(
                "memory: could not remove the tier of deleted project %s in %s",
                name,
                shared.joinpath(*parents),
                exc_info=True,
            )
    if removed:
        logger.info("memory: removed %d memory tier(s) of deleted project %s", removed, name)
    return removed


@dataclass(frozen=True)
class DeploymentMemoryClear:
    """What ``clear_deployment_memory`` removed."""

    # The deployment's own global tier, which holds its project tiers.
    root: Path
    # How many global slots and project tiers were there to remove.
    removed: int


def clear_deployment_memory() -> DeploymentMemoryClear:
    """Remove everything learned in the deployment's own memory tiers.

    Those are the one store every person shares where COWORK_PERSONAL_RESOURCES
    leaves out memory: the global tier's slots, and each project tier in its
    ``projects/`` folder (project_tier_ids). Only those slot files and project
    folders go, so no person's tier, which sits in their own partition, and
    nothing of the operator's managed folder is touched. A link named like a
    project tier is no tier: it stays, and so does what it points at.

    The project tiers are found before anything is removed, so a link in place
    of ``projects/``, which project_tier_ids refuses, stops the clear with
    nothing removed.
    """
    store = deployment_memory_store()
    root = store.root
    tier_ids = project_tier_ids(global_root=root)
    removed = 0
    for slot in MemorySlot:
        if store.read_state(slot).exists:
            store.delete(slot)
            removed += 1
    for tier_id in tier_ids:
        if _remove_tier(base=root, parents=(PROJECT_TIERS_DIRNAME,), name=str(tier_id)):
            removed += 1
    if removed:
        logger.info("memory: cleared the deployment's shared memory at %s", root)
    return DeploymentMemoryClear(root=root, removed=removed)


# Why the memory API refuses a person's change on a deployment that trusts
# identity headers (cowork.api.v1.endpoints.memory).
PERSON_EDIT_REFUSED = "Memory saves on its own as you work, so it can't be edited here."
SHARED_DELETE_REFUSED = (
    "Memory is one store everyone shares on this deployment (COWORK_PERSONAL_RESOURCES "
    "leaves it out), so no one can delete it here."
)
OWN_RESET_REFUSED = "Resetting memory needs a person whose memory is kept per person."
MANAGED_REFUSED = "The deployment's memory rules are read-only: the operator sets them."


def is_person(scope: TenantScope) -> bool:
    """Whether ``scope`` is a signed-in person on a deployment that trusts
    identity headers (cowork.db.scoped.scope_for_identity), rather than the
    operator view, the desktop or hosted org mode."""
    return scope.personal and bool(scope.user_id)


def person_owns_memory(scope: TenantScope) -> bool:
    """Whether ``scope`` is a person whose memory is their own: memory is kept
    per person (COWORK_PERSONAL_RESOURCES lists it), so their tiers are their
    own folders (GlobalMemoryStore, project_memory_store), never the
    deployment's one store."""
    return is_person(scope) and scope.partitioned_for(PersonalResource.MEMORY)


def person_may_reset_memory(scope: TenantScope) -> bool:
    """Whether DELETE /api/v1/memory/all removes ``scope``'s memory: a person
    whose memory is their own, while memory is on."""
    return person_owns_memory(scope) and not resolve_memory_policy(scope=scope).off


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
        self._refuse_while_memory_off(refused="Reading it is refused.")
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
        self._refuse_while_memory_off(refused="Changing it is refused.")
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
        self._refuse_while_memory_off(refused="Deleting it is refused.")
        self._delete(scope, category, project_id)

    def reset_own_memory_sync(self) -> PersonMemoryReset:
        """Remove all of the caller's learned memory with reset_person_memory,
        the reset the operator's clear runs (cowork.ops.memory).

        Only a person whose memory is their own may (person_owns_memory):
        anyone else gets 403, so the desktop and hosted org mode keep no
        reset of their own and no person ever reaches the deployment's shared
        store. While memory is off it answers 409, as the operator's clear
        refuses then, so the kept memory stays as it is.
        """
        if not person_owns_memory(self.session.scope):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=OWN_RESET_REFUSED)
        self._refuse_while_memory_off(refused="Resetting it is refused.")
        return reset_person_memory(self.session.scope)

    async def list_memory(self, project_id: UUID | None = None) -> list[MemoryResponse]:
        return self.list_memory_sync(project_id)

    def list_memory_sync(
        self,
        project_id: UUID | None = None,
    ) -> list[MemoryResponse]:
        # While memory is off on a deployment that trusts identity headers,
        # what is stored stays unread (cowork.harnesses.memory.policy).
        policy = resolve_memory_policy(scope=self.session.scope)
        if policy.off:
            return []

        items = [
            self._global_response(category=category, content=self._global_store.read(category))
            for category in MemorySlot
        ]
        items.extend(_managed_responses())

        projects = list(self.session.exec(self.session.select(Project)).all())
        visible = {project.id for project in projects}
        stores: dict[UUID, ProjectMemoryStore]
        if policy.shared:
            # The deployment's own tiers, one store every scope reads where
            # memory is shared (reads_deployment_tiers): every project tier on
            # disk, whoever's project it is, beside each project the caller
            # can see. A tier is listed by its id alone, so no other person's
            # project row is read and no project name reaches the response.
            stores = {
                tier_id: nested_project_memory_store(scope=self.session.scope, project_id=tier_id)
                for tier_id in sorted(
                    set(project_tier_ids(global_root=self._global_store.root)) | visible, key=str
                )
            }
        else:
            # Everywhere else each project the caller can see keeps its tier
            # where project_memory_store puts it: the caller's own tier where
            # memory is per person, so the listing holds nothing of anyone
            # else, and the project's .anton/memory for the operator view
            # there, on the desktop and in hosted org mode (shared per project
            # there).
            stores = {
                project.id: _project_store(scope=self.session.scope, project=project)
                for project in projects
            }
        if project_id is not None:
            stores = {tier_id: store for tier_id, store in stores.items() if tier_id == project_id}

        for tier_id, store in stores.items():
            for category in PROJECT_SLOTS:
                state = store.read_state(category)
                items.append(
                    self._project_response(
                        category=category,
                        content=state.content,
                        project_id=tier_id,
                        store=store,
                        read_verified=state.readable,
                        writable=tier_id in visible,
                    )
                )
        return items

    def _read(
        self, scope: MemoryScope, category: MemorySlot, project_id: UUID | None
    ) -> str:
        if scope == MemoryScope.global_:
            return self._global_store.read(category)
        project = self._resolve_project(project_id)
        return _project_store(scope=self.session.scope, project=project).read(category)

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
        return _project_store(scope=self.session.scope, project=project).read_state(category)

    def _write(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None,
    ) -> None:
        _refuse_managed(scope)
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
        _refuse_managed(scope)
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
            return self._global_response(category=category, content=content)
        if project_id is None:
            raise ValueError("project_id is required for project-scoped memory.")
        store = _project_store(scope=self.session.scope, project=self._resolve_project(project_id))
        return self._project_response(
            category=category,
            content=content,
            project_id=project_id,
            store=store,
            read_verified=read_verified,
        )

    def _refuse_while_memory_off(self, *, refused: str) -> None:
        """Answer 409 while memory is off on a deployment that trusts identity
        headers, so stored memory stays as it is, unread
        (cowork.harnesses.memory.policy). ``refused`` names what is refused."""
        if resolve_memory_policy(scope=self.session.scope).off:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"{MEMORY_OFF_KEPT} {refused}",
            )

    def _global_response(self, *, category: MemorySlot, content: str) -> MemoryResponse:
        # The global tier records no per-slot authorship. Where memory is
        # shared (reads_deployment_tiers) it is the deployment's one store that
        # every person reads, and the memory API refuses a person's change to
        # it. Everywhere else it is the caller's own.
        return MemoryResponse(
            scope=MemoryScope.global_,
            category=category,
            content=content,
            project_id=None,
            attribution=ResourceAttribution(
                created_by=None,
                last_modified_by=None,
                last_modified_at=None,
            ),
            capabilities=self._capabilities(can_change=True),
        )

    def _capabilities(self, *, can_change: bool) -> MutableResourceCapabilities:
        """What the caller may do to one learned slot it may otherwise change.

        A person on a deployment that trusts identity headers never edits: their
        agent saves memory on its own, and a hand-written rule would shape every
        later answer. They delete a slot only while memory is their own
        (person_owns_memory). The memory API refuses the rest
        (cowork.api.v1.endpoints.memory). Everyone else edits and deletes what
        ``can_change`` allows.
        """
        scope = self.session.scope
        if not is_person(scope):
            return MutableResourceCapabilities(can_edit=can_change, can_delete=can_change)
        return MutableResourceCapabilities(
            can_edit=False,
            can_delete=can_change and person_owns_memory(scope),
        )

    def _project_response(
        self,
        *,
        category: MemorySlot,
        content: str,
        project_id: UUID,
        store: ProjectMemoryStore,
        read_verified: bool = True,
        writable: bool = True,
    ) -> MemoryResponse:
        """One project slot's response, built from its tier's id and store.

        ``writable`` is False for a shared tier whose project the caller can't
        see: a write resolves the project first (_project_slot_coordination),
        so it would refuse.
        """
        key = project_memory_resource_key(project_id, category.value)

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
        if not read_verified or pending or not writable:
            can_change = False
        return MemoryResponse(
            scope=MemoryScope.project,
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
            capabilities=self._capabilities(can_change=can_change),
        )


def _refuse_managed(scope: MemoryScope) -> None:
    """Answer 403 to a change of the managed folder's slots, which nothing in
    Cowork writes (cowork.managed)."""
    if scope == MemoryScope.managed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=MANAGED_REFUSED)


def _managed_responses() -> list[MemoryResponse]:
    """The managed folder's memory slots that hold anything, for the listing,
    read-only for every caller.

    Every turn reads them ahead of the learned tiers (managed_hc, passed by the
    anton harness), so the listing shows the caller the rules their answers
    follow. None without a managed folder.
    """
    folder = managed_folder()
    if folder is None:
        return []
    store = MemoryStore(folder.memory_dir)
    return [
        MemoryResponse(
            scope=MemoryScope.managed,
            category=category,
            content=content,
            project_id=None,
            attribution=ResourceAttribution(
                created_by=None,
                last_modified_by=None,
                last_modified_at=None,
            ),
            capabilities=MutableResourceCapabilities(can_edit=False, can_delete=False),
        )
        for category in MemorySlot
        if (content := store.read(category)).strip()
    ]
