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

import errno
import hashlib
import logging
import os
import re
import stat as stat_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_link,
    dir_lstat,
    dir_mkdir,
    dir_open,
    dir_scandir,
    dir_unlink,
    open_pinned_child,
    pinned_dir,
)
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.comparison import Comparison, ComparisonSide, ComparisonVerdict
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.schemas.responses import Role
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

#: Connector categories whose purpose is reaching people or acting in the
#: world: chat, email, outreach sequences, campaigns, support replies, meeting
#: invites, signature requests, rides and orders, shipping labels. Two agents
#: running the same task would each do it, so they are off in a comparison.
#: Other categories stay on -- building a dashboard from the CRM or the
#: warehouse is the comparison people run.
BLOCKED_CONNECTOR_CATEGORIES = frozenset(
    {
        "communication", "sales-engagement", "marketing", "support", "scheduling", "documents",
        "mobility", "logistics",
    }
)
#: Connectors that reach people from a category that otherwise stays on: a
#: calendar event invites its attendees, an incident pages whoever is on call,
#: a status page posts publicly.
BLOCKED_CONNECTOR_ENGINES = frozenset({"google_calendar", "pagerduty", "opsgenie", "statuspage"})

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
class ContinuedSide:
    conversation: Conversation
    #: False when some of the side's work stayed in its sandbox; continuing
    #: again into the same project carries the rest.
    carried_all: bool


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

    def list_comparisons(self, limit: int = 50, offset: int = 0) -> list[Comparison]:
        return list(
            self.session.exec(
                self._own().order_by(Comparison.created_at.desc(), Comparison.id).offset(offset).limit(limit)
            ).all()
        )

    def record_usage(self, comparison: Comparison, usage: dict) -> None:
        """Keep each side's cost as just read, for the history list."""
        for side in comparison.sides:
            read = usage.get(side.label)
            if read is None or not read.available:
                continue
            side.usage_snapshot = {
                "estimated_cost_usd": read.estimated_cost_usd,
                "tokens": read.input_tokens + read.output_tokens + read.cached_input_tokens + read.cache_write_tokens,
                "partial": bool(read.truncated or read.unpriced_calls),
            }
            self.session.add(side)
        self.session.commit()

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
        """Turns in the side's conversation that the comparison shows.

        A turn is a user message the transcript API returns. A turn's tool
        calls are stored as rows too, and each tool result is a user-role row,
        so counting rows or user rows would count every tool call as a turn.
        """
        from cowork.services.conversations import _is_tool_row

        contents = self.session.exec(
            self.session.select(Message)
            .where(Message.conversation_id == side.conversation_id, Message.role == Role.user)
            .with_only_columns(Message.content)
        ).all()
        count = sum(1 for content in contents if not _is_tool_row(content))
        if side.continued_turn_count is not None:
            return min(count, side.continued_turn_count)
        return count

    def continued_project_id(self, side: ComparisonSide) -> UUID | None:
        """The project a continued side's task is in: where carrying the rest
        of its work must go."""
        if side.continued_at is None:
            return None
        conversation = self.session.exec(
            self.session.select(Conversation).where(Conversation.id == side.conversation_id)
        ).first()
        return conversation.project_id if conversation is not None else None

    def turn_starts(self, side: ComparisonSide) -> list[datetime]:
        """When each of the side's turns started: its user messages, in order.

        Every turn, including any after the side was continued; the caller decides
        how many the comparison owns.
        """
        from cowork.services.conversations import _is_tool_row

        messages = self.session.exec(
            self.session.select(Message)
            .where(Message.conversation_id == side.conversation_id, Message.role == Role.user)
            .order_by(Message.seq, Message.created_at)
        ).all()
        return [m.created_at for m in messages if m.created_at is not None and not _is_tool_row(m.content)]

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
                copied: dict[str, str] = {}
                if source is not None:
                    copy_project_files(
                        Path(source.path), Path(sandbox.path), org_mode=self.session.scope.org_mode, manifest=copied
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
                        copied_files=copied,
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

    def continue_side(
        self, comparison_id: UUID, label: str, destination_project_id: UUID, *, model_label: str | None = None
    ) -> ContinuedSide:
        """Turn one side into an ordinary task in a real project.

        Reuses the task move: the conversation moves and its artifacts move
        with it. The side's other work comes too (see `_carry_side_work`). The
        side row stays, with the message count at the moment of continuing, so
        the comparison keeps showing what was compared.

        When some of the work could not be carried, the sandbox is kept and
        asking again, into the same project, carries the rest.
        """
        from cowork.services.conversations import ConversationService
        from cowork.services.task_objects import TaskObjectService
        from cowork.streaming.registry import registry

        comparison = self.get_comparison(comparison_id)
        side = self.side(comparison, label)
        kept_sandbox = self._kept_sandbox(side) if side.continued_at is not None else None
        if side.continued_at is not None and kept_sandbox is None:
            if side.carry_incomplete:
                # The sandbox went some other way; there is nothing left to offer.
                side.carry_incomplete = False
                self.session.add(side)
                self.session.commit()
            raise ComparisonConflictError("This side was already continued")
        destination = ProjectService(self.session).get_project(destination_project_id)
        if is_comparison_sandbox(destination.name):
            raise ValueError("Continue into a project, not into a comparison")
        handle = registry.get(str(side.conversation_id))
        if handle is not None and handle.is_running:
            raise ComparisonConflictError("Wait for this side to finish its turn, or stop it, then continue")

        conversations = ConversationService(self.session)
        conversation = conversations.get_conversation(side.conversation_id)
        if kept_sandbox is not None:
            return self._finish_carry(side, comparison, conversation, kept_sandbox, destination, model_label)
        turn_count = self.turn_count(side)

        # Move first, mark second, so a failure part-way is retryable: a retry
        # finds the conversation already in the destination (nothing left to
        # relocate) and only records the mark. Marking first would answer 409
        # to the retry while the conversation was still in its sandbox.
        source = conversation.project
        emptied_sandbox = None
        carried_all = True
        if source is not None and source.id != destination.id:
            TaskObjectService(self.session).relocate_to_project(conversation, source, destination)
            carried_all = _carry_side_work(
                side,
                sandbox=Path(source.path),
                destination=Path(destination.path),
                folder_name=_carried_folder_name(comparison.title, model_label or side.model),
                org_mode=self.session.scope.org_mode,
            )
            if carried_all and is_comparison_sandbox(source.name):
                emptied_sandbox = source
        conversation = conversations.update_conversation(conversation.id, project_id=destination.id)

        side.carry_incomplete = not carried_all
        side.continued_turn_count = turn_count
        side.continued_at = datetime.now(timezone.utc)
        self.session.add(side)
        self.session.commit()
        self.session.refresh(conversation)
        # Everything the task needs now lives in its project, so the sandbox
        # would only be an unreachable copy. Kept when anything couldn't be
        # carried.
        if emptied_sandbox is not None:
            self._remove_sandbox(emptied_sandbox)
        return ContinuedSide(conversation=conversation, carried_all=carried_all)

    def _kept_sandbox(self, side: ComparisonSide) -> Project | None:
        """The continued side's sandbox, still there because not all of its
        work could be carried."""
        try:
            sandbox = ProjectService(self.session).get_project(side.project_id)
        except ProjectNotFoundError:
            return None
        return sandbox if is_comparison_sandbox(sandbox.name) else None

    def _finish_carry(
        self,
        side: ComparisonSide,
        comparison: Comparison,
        conversation: Conversation,
        sandbox: Project,
        destination: Project,
        model_label: str | None,
    ) -> ContinuedSide:
        if conversation.project_id != destination.id:
            raise ComparisonConflictError("This side was already continued into another project")
        carried_all = _carry_side_work(
            side,
            sandbox=Path(sandbox.path),
            destination=Path(destination.path),
            folder_name=_carried_folder_name(comparison.title, model_label or side.model),
            org_mode=self.session.scope.org_mode,
        )
        side.carry_incomplete = not carried_all
        self.session.add(side)
        self.session.commit()
        if carried_all:
            self._remove_sandbox(sandbox)
        return ContinuedSide(conversation=conversation, carried_all=carried_all)

    def _remove_sandbox(self, sandbox: Project) -> None:
        _release_project_runtime(sandbox.path)
        try:
            ProjectService(self.session).delete_project(sandbox.id)
        except Exception:
            logger.exception("Could not remove a continued side's sandbox %s", sandbox.id)

    def delete_comparison(self, comparison_id: UUID) -> None:
        """Remove the comparison and the sandboxes of the sides not continued.

        A continued side's sandbox is kept if it is still there: Continue removes
        it once the side's work is in its project, and keeps it only when
        something could not be carried, which deleting the comparison must not
        destroy.
        """
        from cowork.streaming.registry import registry

        comparison = self.get_comparison(comparison_id)
        for side in comparison.sides:
            handle = registry.get(str(side.conversation_id))
            if side.continued_at is None and handle is not None and handle.is_running:
                raise ComparisonConflictError("Stop both sides before deleting the comparison")
        project_ids = [side.project_id for side in comparison.sides if side.continued_at is None]
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
            _release_project_runtime(project.path)
            try:
                projects.delete_project(project_id)
            except Exception:
                # The comparison is gone either way; a leftover sandbox is
                # hidden from every list and is only disk.
                logger.exception("Could not remove comparison sandbox %s", project_id)


_FOLDER_UNSAFE = re.compile(r'[\x00-\x1f/\\:*?"<>|]+')


def _folder_part(text: str, limit: int) -> str:
    first_line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    cleaned = " ".join(_FOLDER_UNSAFE.sub(" ", first_line).split()).strip(" .")
    return cleaned[:limit].rstrip(" .")


def _carried_folder_name(title: str, model_label: str) -> str:
    """`Build a dashboard (Claude Opus 5.5)`: which comparison, and which side."""
    name = _folder_part(title, 60) or "Comparison"
    label = _folder_part(model_label, 40)
    return f"{name} ({label})" if label else name


def _carry_side_work(
    side: ComparisonSide, *, sandbox: Path, destination: Path, folder_name: str, org_mode: bool
) -> bool:
    """Bring what a continued side worked on into the real project.

    Hosted: a task works in its own `conversations/<id>` folder, workspace and
    scratchpad session included, so that folder moves as a whole, to where a
    task in the destination would have had it. When the task already has that
    folder there (it ran in the project after a partial carry), the side's
    files are merged into it instead, never replacing one already there.

    Desktop: a task works in the project's own folder, and the sandbox also
    holds the copy of the source project the comparison started from. Only
    what the side created or changed is copied, into one folder named for the
    comparison and the side, so nothing in the real project is overwritten.

    True when everything was carried, so the sandbox holds nothing the task
    still needs. The folder used is recorded on the side, so carrying again
    after a partial carry finishes the same folder.
    """
    if org_mode:
        workspace = sandbox / "conversations" / str(side.conversation_id)
        target = destination / "conversations" / str(side.conversation_id)
        if not workspace.exists():
            return True
        if workspace.is_symlink() or not workspace.is_dir() or target.is_symlink():
            return False
        if target.exists():
            if not target.is_dir():
                return False
            try:
                merged = copy_side_changes(workspace, target.parent, target.name, copied={}, into=target.name)
            except OSError:
                logger.exception("Could not merge a continued side's workspace into its project")
                return False
            return merged.complete
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(workspace, target)
        except OSError:
            logger.exception("Could not move a continued side's workspace into its project")
            return False
        return True
    try:
        result = copy_side_changes(
            sandbox, destination, folder_name, copied=side.copied_files or {}, into=side.carried_folder
        )
    except OSError:
        logger.exception("Could not copy a continued side's work into its project")
        return False
    if result.folder is not None:
        side.carried_folder = result.folder
    return result.complete


@dataclass
class CarryResult:
    #: The folder in the project the work went into; None when there was nothing to carry.
    folder: str | None
    #: Files written this time.
    files: int
    #: Every changed file is in the folder, and nothing in the sandbox went unexamined.
    complete: bool


def copy_side_changes(
    sandbox_root: Path,
    destination_root: Path,
    folder_name: str,
    *,
    copied: dict[str, str],
    into: str | None = None,
) -> CarryResult:
    """Copy the sandbox files that are not an unchanged copy of the source into
    one folder of `destination_root`, named `folder_name` unless that is taken.
    With none to copy, no folder is made.

    `into` is the folder an earlier, partial carry used: carrying again
    finishes it, skips a file already there with the same content, and never
    overwrites one that differs, since the user may have edited it since.

    Same rules as `copy_project_files`: regular files and directories only,
    never through a link, and within the same budget. Left out: `.anton`
    (artifacts move separately; scratchpad environments are rebuilt on use).
    """
    changed, examined_all = _changed_files(sandbox_root, copied)
    if not changed:
        return CarryResult(folder=into, files=0, complete=examined_all)
    if into is not None and _is_plain_dir(destination_root / into):
        name = into
    else:
        name, n = folder_name, 2
        while (destination_root / name).exists():
            name, n = f"{folder_name} {n}", n + 1
        (destination_root / name).mkdir()
    budget = _CopyBudget()
    failed = 0
    with pinned_dir(sandbox_root) as src_root, pinned_dir(destination_root / name) as dst_root:
        for rel, digest in changed.items():
            parts = rel.split("/")
            src_dirs, dst_dirs = [], []
            try:
                src, dst = src_root, dst_root
                for part in parts[:-1]:
                    src = open_pinned_child(src, part)
                    src_dirs.append(src)
                    try:
                        dir_mkdir(dst, part)
                    except FileExistsError:
                        pass
                    dst = open_pinned_child(dst, part)
                    dst_dirs.append(dst)
                if not _carry_file(src, dst, parts[-1], digest, budget):
                    failed += 1
            except ProjectTooLargeToCopyError:
                logger.warning("A continued side's work was larger than the copy budget; the rest stays in its sandbox")
                failed += 1
                break
            except OSError:
                failed += 1
            finally:
                for d in reversed(src_dirs + dst_dirs):
                    d.close()
    return CarryResult(folder=name, files=budget.files, complete=examined_all and failed == 0)


def _is_plain_dir(path: Path) -> bool:
    try:
        return stat_module.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


#: link() errors that mean "no hard links here", not "this file failed".
_NO_HARD_LINKS = frozenset(
    code for code in (getattr(errno, n, None) for n in ("EXDEV", "EPERM", "ENOTSUP", "EOPNOTSUPP", "EMLINK")) if code
)


def _carry_file(src: PinnedDir, dst: PinnedDir, name: str, digest: str | None, budget: _CopyBudget) -> bool:
    """Put one changed file in the folder. True when the folder holds it, or
    already holds something of the user's under that name, which wins.

    Written under a temporary name, then published with a hard link, which
    fails rather than replace a file that appeared meanwhile (a rename would
    replace it). So a copy cut short never leaves a partial file, and nothing
    in the project is overwritten.
    """
    if digest is None:
        return False
    try:
        dir_lstat(dst, name)
    except FileNotFoundError:
        pass
    else:
        return True
    temp = f".carry-{os.urandom(8).hex()}.partial"
    before = budget.files
    try:
        _copy_file(src, dst, name, budget, dst_name=temp)
        if budget.files == before:
            return False
        try:
            dir_link(dst, temp, name)
        except FileExistsError:
            return True
        except OSError as e:
            if e.errno not in _NO_HARD_LINKS:
                raise
            return _publish_by_exclusive_copy(dst, temp, name)
        return True
    finally:
        try:
            dir_unlink(dst, temp)
        except OSError:
            pass


def _publish_by_exclusive_copy(dst: PinnedDir, temp: str, name: str) -> bool:
    """Without hard links: create `name` exclusively and copy the finished
    temporary file into it, so an existing name is still never replaced."""
    binary = getattr(os, "O_BINARY", 0)
    try:
        fd_out = dir_open(dst, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | binary, 0o644)
    except FileExistsError:
        return True
    try:
        fd_in = dir_open(dst, temp, os.O_RDONLY | O_NOFOLLOW | binary)
        try:
            while chunk := os.read(fd_in, _COPY_CHUNK):
                view = memoryview(chunk)
                while view:
                    view = view[os.write(fd_out, view):]
        finally:
            os.close(fd_in)
    except BaseException:
        os.close(fd_out)
        # Created exclusively above, so the name is this copy's own.
        try:
            dir_unlink(dst, name)
        except OSError:
            pass
        raise
    os.close(fd_out)
    return True


def _changed_files(sandbox_root: Path, copied: dict[str, str]) -> tuple[dict[str, str | None], bool]:
    """Sandbox files whose content is not in `copied`, by relative path, with
    their SHA-256 (None when unreadable); and whether every entry could be
    examined. Something the walk could not look into may be work, so it
    keeps the sandbox."""
    changed: dict[str, str | None] = {}
    examined_all = True

    def walk(directory: PinnedDir, rel: str, depth: int) -> None:
        nonlocal examined_all
        with dir_scandir(directory) as scan:
            names = sorted(entry.name for entry in scan)
        if names and depth > _COPY_MAX_DEPTH:
            examined_all = False
            return
        for name in names:
            if not rel and name == ".anton":
                continue
            try:
                st = dir_lstat(directory, name)
            except OSError:
                examined_all = False
                continue
            path = f"{rel}{name}"
            if stat_module.S_ISDIR(st.st_mode):
                try:
                    child = open_pinned_child(directory, name)
                except OSError:
                    examined_all = False
                    continue
                try:
                    walk(child, f"{path}/", depth + 1)
                except OSError:
                    examined_all = False
                finally:
                    child.close()
            elif stat_module.S_ISREG(st.st_mode):
                digest = _file_digest(directory, name)
                # A file that can't be read can't be shown to be the original
                # copy, so it counts as changed and its failed copy is noticed.
                if digest is None or copied.get(path) != digest:
                    changed[path] = digest

    with pinned_dir(sandbox_root) as root:
        walk(root, "", 0)
    return changed, examined_all


def _file_digest(directory: PinnedDir, name: str) -> str | None:
    binary = getattr(os, "O_BINARY", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    try:
        fd = dir_open(directory, name, os.O_RDONLY | O_NOFOLLOW | nonblock | binary)
    except OSError:
        return None
    try:
        if not stat_module.S_ISREG(os.fstat(fd).st_mode):
            return None
        digest = hashlib.sha256()
        while chunk := os.read(fd, _COPY_CHUNK):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _release_project_runtime(project_path: str | None) -> None:
    """Stop a sandbox's backend previews and free its scratchpad slots.

    Each project holds its own, and the pool is capped: without this, a few
    deleted comparisons would use up every slot until the server restarts.
    """
    if not project_path:
        return
    from cowork.services import scratchpad_runtime
    from cowork.services.artifacts import stop_project_backends

    stop_project_backends(project_path)
    scratchpad_runtime.release_workspace(project_path)


def _blocked(engine: str | None) -> bool:
    from cowork.services.connectors.specs._registry import registry

    spec = registry.get_connector(engine or "")
    # A connector the catalog doesn't describe (a custom one) could do anything,
    # so it is off too.
    if spec is None:
        return True
    return spec.category in BLOCKED_CONNECTOR_CATEGORIES or engine in BLOCKED_CONNECTOR_ENGINES


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
    #: Relative path -> SHA-256 of each file copied, when the caller wants them.
    manifest: dict[str, str] | None = None

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


def copy_project_files(
    source_root: Path, dest_root: Path, *, org_mode: bool, manifest: dict[str, str] | None = None
) -> int:
    """Copy a project's files into a sandbox. Returns the number of files copied.

    Regular files and directories only, and never through a link: every
    component is opened relative to its pinned parent with `O_NOFOLLOW`. On a
    hosted deployment agents write into project trees on shared storage, so a
    link planted there must not turn this copy into a read of another org's
    files.

    Left out: the project's `.anton` state other than its instructions and
    memory, and on a hosted deployment the per-conversation workspaces under
    `conversations/`, which belong to other members.

    `manifest`, when given, is filled with each copied file's relative path
    and SHA-256 (the `.anton` files it keeps are not listed).
    """
    budget = _CopyBudget(manifest=manifest)
    with pinned_dir(source_root) as src, pinned_dir(dest_root) as dst:
        _copy_children(src, dst, budget, depth=0, top_level=True, org_mode=org_mode, rel="")
    return budget.files


def _copy_children(
    src: PinnedDir, dst: PinnedDir, budget: _CopyBudget, *, depth: int, top_level: bool, org_mode: bool,
    rel: str | None,
) -> None:
    """``rel`` is this directory's path within the copy, or None where files are
    not recorded (the kept ``.anton`` state)."""
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
            _copy_subdir(
                src, dst, name, budget, depth=depth, org_mode=org_mode, rel=None if rel is None else f"{rel}{name}/"
            )
        elif stat_module.S_ISREG(st.st_mode):
            _copy_file(src, dst, name, budget, rel=None if rel is None else f"{rel}{name}")


def _copy_subdir(
    src: PinnedDir, dst: PinnedDir, name: str, budget: _CopyBudget, *, depth: int, org_mode: bool, rel: str | None = None
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
            _copy_children(
                child_src, child_dst, budget, depth=depth + 1, top_level=False, org_mode=org_mode, rel=rel
            )
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


def _copy_file(
    src: PinnedDir, dst: PinnedDir, name: str, budget: _CopyBudget, *, rel: str | None = None, dst_name: str | None = None
) -> None:
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
        fd_out = dir_open(dst, dst_name or name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | binary, 0o644)
        digest = hashlib.sha256() if budget.manifest is not None and rel is not None else None
        try:
            while True:
                chunk = os.read(fd_in, _COPY_CHUNK)
                if not chunk:
                    break
                if digest is not None:
                    digest.update(chunk)
                # Counted as read, not from the stat, so a file that grows
                # mid-copy is still bounded.
                budget.take_bytes(len(chunk))
                view = memoryview(chunk)
                while view:
                    written = os.write(fd_out, view)
                    view = view[written:]
        finally:
            os.close(fd_out)
        if digest is not None:
            budget.manifest[rel] = digest.hexdigest()
    finally:
        os.close(fd_in)
