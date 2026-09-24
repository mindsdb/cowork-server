"""Model comparisons: one task, two models, side by side.

Each side is an ordinary conversation in its own hidden sandbox project, so the
turn path, streaming, history and artifacts need nothing new to run it. What a
comparison adds is containment, enforced here and at the few places a turn can
reach outside its project:

* the model and effort a side was created with are the ones every turn uses;
* connectors whose job is contacting people are switched off;
* memory is read but never written;
* nothing is published, and the fast-answer router is skipped so both sides
  are measured on the agent.

A conversation is contained exactly while its project is a sandbox. That is
the one test, rather than "has a side row": continuing a side moves its
conversation into a real project, and from then on it is an ordinary task.
"""

from __future__ import annotations

import logging
import os
import stat as stat_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import func

from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_lstat,
    dir_mkdir,
    dir_open,
    dir_scandir,
    open_pinned_child,
    pinned_dir,
)
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.comparison import Comparison, ComparisonSide, ComparisonVerdict
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.services.projects import (
    ProjectNotFoundError,
    ProjectService,
    display_label,
    is_comparison_sandbox,
)

logger = logging.getLogger(__name__)

SIDE_LABELS = ("a", "b")
#: Every side runs on Anton: the model pick is what is being compared, and it
#: only drives the Anton harness. A Coding Mode default must not turn a
#: comparison into two runs of the same external CLI.
COMPARISON_HARNESS = "anton"
VERDICT_WINNERS = frozenset({"a", "b", "tie", "neither"})

#: Connector categories whose purpose is reaching people: chat, email,
#: outreach sequences, campaigns, support replies, meeting invites, signature
#: requests. Two agents running the same task would each send the message, so
#: they are off in a comparison. Every other category stays on -- building a
#: dashboard from the CRM or the warehouse is the comparison people run.
BLOCKED_CONNECTOR_CATEGORIES = frozenset(
    {"communication", "sales-engagement", "marketing", "support", "scheduling", "documents"}
)

# What a copied project may hold. Generous for documents and data files; it
# exists so a comparison started from a project with a vendored dependency tree
# fails fast instead of copying it twice.
_COPY_MAX_FILES = 2000
_COPY_MAX_BYTES = 256 * 1024 * 1024
_COPY_MAX_DEPTH = 32
_COPY_CHUNK = 1024 * 1024
#: From a project's `.anton`, only what shapes how the agent works there. Not
#: `artifacts`: each folder carries its artifact id and publish record, and a
#: copy would give a sandbox the identity of a live published artifact.
_ANTON_KEPT = frozenset({"anton.md", "memory"})


class ComparisonNotFoundError(LookupError):
    pass


class ComparisonConflictError(RuntimeError):
    """The request is valid but the comparison's state does not allow it now."""


class ProjectTooLargeToCopyError(ValueError):
    pass


@dataclass(frozen=True)
class SideSpec:
    model: str
    reasoning_effort: str | None = None


class ComparisonService:
    """Comparisons are personal, like conversations: the org filter comes from
    the scoped session and the owner filter is applied here."""

    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _own(self):
        stmt = self.session.select(Comparison)
        if self.session.scope.org_mode:
            stmt = stmt.where(Comparison.created_by == self.session.scope.user_id)
        return stmt

    def list_comparisons(self, limit: int = 50) -> list[Comparison]:
        return list(
            self.session.exec(
                self._own().order_by(Comparison.created_at.desc(), Comparison.id).limit(limit)
            ).all()
        )

    def get_comparison(self, comparison_id: UUID) -> Comparison:
        comparison = self.session.exec(self._own().where(Comparison.id == comparison_id)).first()
        if comparison is None:
            raise ComparisonNotFoundError("Comparison not found")
        return comparison

    def side(self, comparison: Comparison, label: str) -> ComparisonSide:
        for side in comparison.sides:
            if side.label == label:
                return side
        raise ComparisonNotFoundError("Comparison side not found")

    def turn_count(self, side: ComparisonSide) -> int:
        """Messages in the side's conversation that the comparison shows."""
        count = self.session.exec(
            self.session.select(Message)
            .where(Message.conversation_id == side.conversation_id)
            .with_only_columns(func.count())
        ).one()
        if side.continued_turn_count is not None:
            return min(count, side.continued_turn_count)
        return count

    def contained_side(self, conversation: Conversation) -> ComparisonSide | None:
        """The side a conversation is, while it is still contained.

        None for every ordinary conversation, and for a side once it has been
        continued into a real project.
        """
        # getattr: the turn path is also driven with conversation-shaped
        # stand-ins, and a containment check must not be what raises.
        project = getattr(conversation, "project", None)
        if project is None or not is_comparison_sandbox(getattr(project, "name", None)):
            return None
        return self.session.exec(
            self.session.select(ComparisonSide).where(
                ComparisonSide.conversation_id == conversation.id
            )
        ).first()

    def create_comparison(
        self,
        *,
        title: str,
        sides: list[SideSpec],
        source_project_id: UUID | None = None,
    ) -> Comparison:
        if len(sides) != len(SIDE_LABELS):
            raise ValueError("A comparison has exactly two sides")
        for spec in sides:
            if not spec.model or not spec.model.strip():
                raise ValueError("Each side needs a model")

        projects = ProjectService(self.session)
        source: Project | None = None
        if source_project_id is not None:
            source = projects.get_project(source_project_id)
            if is_comparison_sandbox(source.name):
                raise ValueError("Start a comparison from a project, not from another comparison")

        title = (title or "").strip()[:255] or "Untitled comparison"
        created: list[Project] = []
        try:
            comparison = Comparison(
                title=title,
                source_project_id=source.id if source else None,
                source_project_label=display_label(source)[:255] if source else None,
            )
            # Both sides get the same label: the harness writes it into the
            # system prompt, and the prompts should differ by nothing but the
            # model. The source project's name keeps the agent's framing the
            # same as in the real project.
            sandbox_label = display_label(source) if source else title
            for label, spec in zip(SIDE_LABELS, sides):
                sandbox = projects.create_comparison_sandbox(sandbox_label)
                created.append(sandbox)
                if source is not None:
                    copy_project_files(
                        Path(source.path), Path(sandbox.path), org_mode=self.session.scope.org_mode
                    )
                conversation = Conversation(
                    topic=title,
                    project_id=sandbox.id,
                    harness=COMPARISON_HARNESS,
                    model=spec.model.strip(),
                    reasoning_effort=spec.reasoning_effort,
                )
                self.session.add(conversation)
                self.session.flush()
                comparison.sides.append(
                    ComparisonSide(
                        label=label,
                        model=spec.model.strip(),
                        reasoning_effort=spec.reasoning_effort,
                        project_id=sandbox.id,
                        conversation_id=conversation.id,
                    )
                )
            self.session.add(comparison)
            self.session.commit()
        except Exception:
            self.session.rollback()
            for sandbox in created:
                try:
                    projects.delete_project(sandbox.id)
                except Exception:
                    logger.exception("Could not remove comparison sandbox %s after a failed create", sandbox.id)
            raise
        self.session.refresh(comparison)
        return comparison

    def record_verdict(self, comparison_id: UUID, *, turn_index: int, winner: str) -> ComparisonVerdict:
        if winner not in VERDICT_WINNERS:
            raise ValueError(f"winner must be one of {sorted(VERDICT_WINNERS)}")
        if turn_index < 0:
            raise ValueError("turn_index must be zero or more")
        comparison = self.get_comparison(comparison_id)
        existing = next((v for v in comparison.verdicts if v.turn_index == turn_index), None)
        if existing is not None:
            existing.winner = winner
            verdict = existing
        else:
            verdict = ComparisonVerdict(comparison_id=comparison.id, turn_index=turn_index, winner=winner)
            comparison.verdicts.append(verdict)
        self.session.add(comparison)
        self.session.commit()
        self.session.refresh(verdict)
        return verdict

    def continue_side(self, comparison_id: UUID, label: str, destination_project_id: UUID) -> Conversation:
        """Turn one side into an ordinary task in a real project.

        Reuses the task move: the conversation moves and its artifacts move
        with it. The side row stays, with the message count at the moment of
        continuing, so the comparison keeps showing what was compared.
        """
        from cowork.services.conversations import ConversationService
        from cowork.services.task_objects import TaskObjectService
        from cowork.streaming.registry import registry

        comparison = self.get_comparison(comparison_id)
        side = self.side(comparison, label)
        if side.continued_at is not None:
            raise ComparisonConflictError("This side was already continued")
        destination = ProjectService(self.session).get_project(destination_project_id)
        if is_comparison_sandbox(destination.name):
            raise ValueError("Continue into a project, not into a comparison")
        handle = registry.get(str(side.conversation_id))
        if handle is not None and handle.is_running():
            raise ComparisonConflictError("Wait for this side to finish its turn, or stop it, then continue")

        conversations = ConversationService(self.session)
        conversation = conversations.get_conversation(side.conversation_id)
        turn_count = self.turn_count(side)

        # Move first, mark second, so a failure part-way is retryable: a retry
        # finds the conversation already in the destination (nothing left to
        # relocate) and only records the mark. Marking first would answer 409
        # to the retry while the conversation was still in its sandbox.
        source = conversation.project
        if source is not None and source.id != destination.id:
            TaskObjectService(self.session).relocate_to_project(conversation, source, destination)
        conversation = conversations.update_conversation(conversation.id, project_id=destination.id)

        side.continued_turn_count = turn_count
        side.continued_at = datetime.now(timezone.utc)
        self.session.add(side)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def delete_comparison(self, comparison_id: UUID) -> None:
        """Remove the comparison and both sandboxes with everything in them.

        A side that was continued lives on as a task in its real project; only
        its now-empty sandbox goes.
        """
        from cowork.streaming.registry import registry

        comparison = self.get_comparison(comparison_id)
        for side in comparison.sides:
            handle = registry.get(str(side.conversation_id))
            if side.continued_at is None and handle is not None and handle.is_running():
                raise ComparisonConflictError("Stop both sides before deleting the comparison")
        project_ids = [side.project_id for side in comparison.sides]
        self.session.delete(comparison)
        self.session.commit()
        projects = ProjectService(self.session)
        for project_id in project_ids:
            try:
                project = projects.get_project(project_id)
            except ProjectNotFoundError:
                continue
            if not is_comparison_sandbox(project.name):
                continue
            try:
                projects.delete_project(project_id)
            except Exception:
                # The comparison is gone either way; a leftover sandbox is
                # hidden from every list and is only disk.
                logger.exception("Could not remove comparison sandbox %s", project_id)


def _blocked(engine: str | None) -> bool:
    from cowork.services.connectors.specs._registry import registry

    spec = registry.get_connector(engine or "")
    return spec is not None and spec.category in BLOCKED_CONNECTOR_CATEGORIES


async def blocked_connections(scope: TenantScope) -> list[dict]:
    """The caller's connections a comparison side must not use, as the
    `{engine, name}` pairs the turn path's `disabled` list already takes.

    Listed from where each surface reads connections: the local vault on
    desktop, auth's turn-key connection list for a hosted turn. Raises when the
    list cannot be read, so a caller fails the turn rather than running a side
    that might still reach a messaging app.
    """
    if scope.org_mode:
        from cowork.common.settings.app_settings import TurnQueueSettings
        from cowork.turnqueue.auth_keys import list_active_connections

        items = await list_active_connections(
            org_id=scope.org_id, user_id=scope.user_id, settings=TurnQueueSettings()
        )
    else:
        from cowork.services.connectors.connections import ConnectionsService

        items = [{"engine": c.engine, "name": c.name} for c in ConnectionsService(scope).list()]
    return [
        {"engine": item.get("engine"), "name": item.get("name")}
        for item in items
        if _blocked(item.get("engine"))
    ]


@dataclass
class _CopyBudget:
    files: int = 0
    bytes: int = 0

    def take_file(self) -> None:
        self.files += 1
        if self.files > _COPY_MAX_FILES:
            raise ProjectTooLargeToCopyError(
                f"This project has more than {_COPY_MAX_FILES} files, too many to copy into a comparison"
            )

    def take_bytes(self, count: int) -> None:
        self.bytes += count
        if self.bytes > _COPY_MAX_BYTES:
            raise ProjectTooLargeToCopyError(
                f"This project holds more than {_COPY_MAX_BYTES // (1024 * 1024)} MB, too much to copy into a comparison"
            )


def copy_project_files(source_root: Path, dest_root: Path, *, org_mode: bool) -> int:
    """Copy a project's files into a sandbox. Returns the number of files copied.

    Regular files and directories only, and never through a link: every
    component is opened relative to its pinned parent with `O_NOFOLLOW`. On a
    hosted deployment agents write into project trees on shared storage, so a
    link planted there must not turn this copy into a read of another org's
    files.

    Left out: the project's `.anton` state other than its instructions and
    memory, and on a hosted deployment the per-conversation workspaces under
    `conversations/`, which belong to other members.
    """
    budget = _CopyBudget()
    with pinned_dir(source_root) as src, pinned_dir(dest_root) as dst:
        _copy_children(src, dst, budget, depth=0, top_level=True, org_mode=org_mode)
    return budget.files


def _copy_children(
    src: PinnedDir, dst: PinnedDir, budget: _CopyBudget, *, depth: int, top_level: bool, org_mode: bool
) -> None:
    if depth > _COPY_MAX_DEPTH:
        return
    with dir_scandir(src) as scan:
        names = sorted(entry.name for entry in scan)
    for name in names:
        if top_level and org_mode and name == "conversations":
            continue
        try:
            st = dir_lstat(src, name)
        except OSError:
            continue
        # lstat reports a link as a link, never as a directory or a file, so
        # links fall through both branches. The opens below repeat the refusal
        # for an entry swapped after this check.
        if stat_module.S_ISDIR(st.st_mode):
            if top_level and name == ".anton":
                _copy_anton_dir(src, dst, budget, org_mode=org_mode)
                continue
            _copy_subdir(src, dst, name, budget, depth=depth, org_mode=org_mode)
        elif stat_module.S_ISREG(st.st_mode):
            _copy_file(src, dst, name, budget)


def _copy_subdir(
    src: PinnedDir, dst: PinnedDir, name: str, budget: _CopyBudget, *, depth: int, org_mode: bool
) -> None:
    try:
        child_src = open_pinned_child(src, name)
    except OSError:
        return
    try:
        try:
            dir_mkdir(dst, name)
        except FileExistsError:
            pass
        child_dst = open_pinned_child(dst, name)
        try:
            _copy_children(child_src, child_dst, budget, depth=depth + 1, top_level=False, org_mode=org_mode)
        finally:
            child_dst.close()
    finally:
        child_src.close()


def _copy_anton_dir(src: PinnedDir, dst: PinnedDir, budget: _CopyBudget, *, org_mode: bool) -> None:
    try:
        anton_src = open_pinned_child(src, ".anton")
    except OSError:
        return
    try:
        try:
            dir_mkdir(dst, ".anton")
        except FileExistsError:
            pass
        anton_dst = open_pinned_child(dst, ".anton")
        try:
            with dir_scandir(anton_src) as scan:
                names = sorted(entry.name for entry in scan if entry.name in _ANTON_KEPT)
            for name in names:
                try:
                    st = dir_lstat(anton_src, name)
                except OSError:
                    continue
                if stat_module.S_ISDIR(st.st_mode):
                    _copy_subdir(anton_src, anton_dst, name, budget, depth=1, org_mode=org_mode)
                elif stat_module.S_ISREG(st.st_mode):
                    _copy_file(anton_src, anton_dst, name, budget)
        finally:
            anton_dst.close()
    finally:
        anton_src.close()


def _copy_file(src: PinnedDir, dst: PinnedDir, name: str, budget: _CopyBudget) -> None:
    binary = getattr(os, "O_BINARY", 0)
    # O_NONBLOCK so an entry swapped for a FIFO after the lstat cannot park
    # this thread in open() waiting for a writer; it has no effect on reading
    # a regular file.
    nonblock = getattr(os, "O_NONBLOCK", 0)
    try:
        fd_in = dir_open(src, name, os.O_RDONLY | O_NOFOLLOW | nonblock | binary)
    except OSError:
        return
    try:
        # Re-checked on the open descriptor: the entry could have been swapped
        # for a FIFO or device after the lstat.
        if not stat_module.S_ISREG(os.fstat(fd_in).st_mode):
            return
        budget.take_file()
        fd_out = dir_open(dst, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | binary, 0o644)
        try:
            while True:
                chunk = os.read(fd_in, _COPY_CHUNK)
                if not chunk:
                    break
                # Counted as read, not from the stat, so a file that grows
                # mid-copy is still bounded.
                budget.take_bytes(len(chunk))
                view = memoryview(chunk)
                while view:
                    written = os.write(fd_out, view)
                    view = view[written:]
        finally:
            os.close(fd_out)
    finally:
        os.close(fd_in)
