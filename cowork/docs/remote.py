"""The doc tools of a remote (pod) turn, run on this server.

The pod forwards each call (cowork.turnqueue.host_tools); this module binds
those calls to the doc executor (cowork.docs.tools) with what a remote turn
needs on top of a desktop one: the pod's paths are untrusted, since this
server sees every organization's files.
"""

from __future__ import annotations

import copy
import logging
import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cowork.common.logger import log_context
from cowork.common.paths import O_NOFOLLOW, O_NONBLOCK, dir_open, opened_subdir_nofollow, read_bounded
from cowork.common.settings.user_settings import use_settings_scope
from cowork.db.units import DatabaseBusy, run_efs, unit_session
from cowork.docs.images import MAX_IMAGE_BYTES
from cowork.docs.model import DocError
from cowork.docs.prompt import DOC_SKILL_LABEL, DOC_SKILL_MD
from cowork.docs.store import doc_lock, doc_subfolder, read_metadata
from cowork.docs.tools import TOO_BIG_TEXT, DocToolContext, remote_doc_tool_specs, run_doc_tool
from cowork.services.artifact_identity import _write_metadata
from cowork.services.artifact_ownership import artifact_root_for_base, record_artifact_owner
from cowork.services.artifact_permissions import artifact_capabilities
from cowork.schemas.turn_skills import TurnSkillEntry, TurnSkills
from cowork.turnqueue.host_tools import MAX_RESULT_BYTES, EmitStep, HostCall, HostToolOutcome, HostToolRunner

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from cowork.db.scoped import TenantScope
    from cowork.harnesses.anton_harness.agent_files import AgentFiles

logger = logging.getLogger(__name__)


def pod_image_reader(files: AgentFiles | None) -> Callable[[str], bytes]:
    """Read an image the pod names by its own path: placed in the project by
    the pod's mounts (AgentFiles.pod_project_parts), then opened from the
    project folder one component at a time with O_NOFOLLOW, so a link the
    agent's code plants anywhere on the way is refused rather than followed.
    Raises OSError for anything it will not read."""

    def read(src: str) -> bytes:
        parts = files.pod_project_parts(src) if files is not None else None
        if not parts:
            raise OSError("not a path in this turn's project")
        *dirs, name = parts
        try:
            with opened_subdir_nofollow(files.project_root, *dirs) as pinned:
                fd = dir_open(pinned, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
                try:
                    found = os.fstat(fd)
                    if not stat.S_ISREG(found.st_mode) or found.st_size > MAX_IMAGE_BYTES:
                        raise OSError("not an image file within the size limit")
                    return b"".join(read_bounded(fd, found.st_size))
                finally:
                    os.close(fd)
        except ValueError as exc:
            # An embedded NUL or similar: the path is unreadable, not a bug.
            raise OSError(str(exc)) from exc

    return read


NOT_OWNER_MESSAGE = (
    "only the doc's owner can edit it, as in the editor. read_doc and list_docs still work; "
    "say in chat what you would change."
)
BUSY_MESSAGE = "the server is busy and nothing was changed; call edit_doc again in a moment."


def owner_check(*, scope: TenantScope, project_id: str | None, artifacts_base: Path) -> Callable[[Path], None] | None:
    """The edit check the docs API makes (_resolve(write=True), the check
    require_artifact_owner makes): in org mode one project's artifacts tree is
    shared by every member, and only a doc's owner may edit it, through the
    editor or through their agent. None outside org mode."""
    if not getattr(scope, "org_mode", False):
        return None

    def check(folder: Path) -> None:
        may_edit = False
        if project_id:
            try:
                with unit_session(scope=scope) as session:
                    source = artifact_root_for_base(session, project_id, artifacts_base)
                    may_edit = source is not None and artifact_capabilities(session, source, folder.name)["canEdit"]
            except ValueError:
                may_edit = False
            except DatabaseBusy:
                # Nothing was written yet, so say so rather than "check its effect".
                raise DocError("busy", BUSY_MESSAGE) from None
        if not may_edit:
            raise DocError("not_owner", NOT_OWNER_MESSAGE)

    return check


def record_turn_provenance(folder: Path, *, conversation_id: str, turn_index: int, summary: str) -> None:
    """Append this turn to the doc's ``provenance``, in the shape anton's
    ArtifactStore.record_turn writes, once per turn: a later call of the same
    turn leaves the entry as the first one wrote it.

    The pod writes this folder, and this server sees every organization's
    files, so the folder and its ``metadata.json`` are opened without
    following links (a link or a second name there is refused, not read),
    the document is replaced rather than written through, and README.md is
    not written at all. Under the doc's lock, so a write of the editor's
    can't land in between. Raises DocError, OSError or ValueError when the
    metadata can't be read or written.
    """
    with doc_lock(folder), doc_subfolder(folder) as pinned:
        metadata = read_metadata(pinned)
        document = metadata.to_document()
        provenance = document.get("provenance")
        provenance = copy.deepcopy(provenance) if isinstance(provenance, list) else []
        entry = next((p for p in provenance if isinstance(p, dict) and p.get("conversation") == conversation_id),
                     None)
        if entry is None:
            entry = {"conversation": conversation_id, "title": None, "turns": []}
            provenance.append(entry)
        turns = entry.get("turns")
        if not isinstance(turns, list):
            turns = entry["turns"] = []
        if any(isinstance(t, dict) and t.get("index") == turn_index for t in turns):
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        turns.append({"index": turn_index, "timestamp": now, "summary": summary, "files_touched": []})
        _write_metadata(pinned, metadata.with_value("provenance", provenance).with_value("updatedAt", now))


class RemoteDocWrites:
    """What a remote turn's doc tools wrote, recorded from the worker thread
    right after each write, in this order: the owner of a doc a call created,
    the turn's provenance in the doc's metadata, and the folder for the turn
    end to claim. A call whose task is cancelled after the write still leaves
    all three: the thread runs on. The turn end claims a new folder only when
    its provenance names this conversation."""

    def __init__(
        self,
        *,
        scope: TenantScope,
        conversation_id: str,
        project_id: str | None,
        creator: str | None,
        artifacts_base: Path,
        turn_index: int,
    ) -> None:
        self._scope = scope
        self._conversation_id = conversation_id
        self._project_id = project_id
        self._creator = creator
        self._artifacts_base = Path(artifacts_base)
        self._turn_index = turn_index
        self._lock = threading.Lock()
        self._slugs: set[str] = set()
        self._provenance_recorded: set[str] = set()
        self._by_call: dict[str, str] = {}

    def written(self, call_id: str, slug: str, tool: str) -> None:
        created = tool == "create_doc"
        if created and getattr(self._scope, "org_mode", False) and self._project_id and self._creator:
            # Unrecorded, every edit the person makes before the turn ends would be refused.
            try:
                with unit_session(scope=self._scope) as session:
                    record_artifact_owner(session, self._project_id, slug, self._creator, action="create")
            except Exception:
                logger.warning("could not record the owner of a doc a remote turn created", exc_info=True,
                               extra=log_context(conversation_id=self._conversation_id, artifact_slug=slug))
        with self._lock:
            provenance_done = slug in self._provenance_recorded
        if not provenance_done:
            try:
                record_turn_provenance(self._artifacts_base / slug, conversation_id=self._conversation_id,
                                       turn_index=self._turn_index,
                                       summary="Created doc" if created else "Edited doc")
            except Exception:
                logger.warning("could not record a remote turn's provenance on a doc", exc_info=True,
                               extra=log_context(conversation_id=self._conversation_id, artifact_slug=slug))
            else:
                with self._lock:
                    self._provenance_recorded.add(slug)
        with self._lock:
            self._slugs.add(slug)
            self._by_call[call_id] = slug

    def slugs(self) -> set[str]:
        with self._lock:
            return set(self._slugs)

    def slug_for(self, call_id: str) -> str | None:
        with self._lock:
            return self._by_call.get(call_id)


def remote_doc_context(
    *,
    scope: TenantScope,
    conversation_id: str,
    project_id: str | None,
    project_name: str,
    artifacts_base: Path,
    files: AgentFiles | None,
) -> DocToolContext:
    """One remote turn's doc tool context, on the folder its pod writes. Its
    file work runs on the file store's threads (run_efs), with the slot limits
    and the per-person share the docs API's own calls have; ``deadline=False``,
    as the work is bookkeeping a refusal would lose. A cancel waits for a
    started call to return, so the write and what follows it still land."""

    async def on_file_store(fn):
        return await run_efs(fn, scope=scope, deadline=False)

    return DocToolContext(
        artifacts_base=artifacts_base,
        conversation_id=conversation_id,
        project_id=project_id,
        project_name=project_name,
        project_label=project_name,
        owner_check=owner_check(scope=scope, project_id=project_id, artifacts_base=artifacts_base),
        image_reader=pod_image_reader(files),
        max_result_bytes=MAX_RESULT_BYTES,
        run_in_thread=on_file_store,
    )


@dataclass(frozen=True)
class RemoteDocTools:
    """A remote turn's doc tools: the runner its relay loop hands the pod's
    calls to, what the calls wrote, and the context they share."""

    runner: HostToolRunner
    writes: RemoteDocWrites
    context: DocToolContext
    specs: list[dict[str, Any]]


def with_docs_skill(skills: TurnSkills | None) -> TurnSkills:
    """The turn's skills plus the ``doc-tools`` skill, which unlocks every doc
    tool but list_docs in the pod. It replaces a person's skill of the same
    name, and the producer never sheds it (turnqueue.producer._shed_skills)."""
    return {**(skills or {}), DOC_SKILL_LABEL: TurnSkillEntry(files={"SKILL.md": DOC_SKILL_MD})}


def start_remote_doc_tools(
    *,
    scope: TenantScope,
    correlation_id: str,
    conversation_id: str,
    turn_index: int,
    artifacts_base: Path,
    project_id: str | None,
    project_name: str,
    creator: str | None,
    files: AgentFiles | None,
    r: aioredis.Redis | None = None,
) -> RemoteDocTools:
    context = remote_doc_context(scope=scope, conversation_id=conversation_id, project_id=project_id,
                                 project_name=project_name, artifacts_base=artifacts_base, files=files)
    writes = RemoteDocWrites(scope=scope, conversation_id=conversation_id, project_id=project_id,
                             creator=creator, artifacts_base=artifacts_base, turn_index=turn_index)

    async def execute(call: HostCall, emit_step: EmitStep) -> HostToolOutcome:
        # The card and the doc's page read the turn's settings.
        with use_settings_scope(scope):
            result = await run_doc_tool(context, call.tool, call.input, emit_step,
                                        on_written=partial(writes.written, call.call_id))
        return HostToolOutcome(result.text, ok=result.ok)

    specs = remote_doc_tool_specs()
    runner = HostToolRunner(
        correlation_id=correlation_id,
        timeouts={spec["name"]: spec["timeout_s"] for spec in specs},
        execute=execute,
        too_big_text=TOO_BIG_TEXT,
        r=r,
    )
    return RemoteDocTools(runner=runner, writes=writes, context=context, specs=specs)
