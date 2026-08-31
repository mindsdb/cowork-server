from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import case, func
from sqlalchemy import select as sa_select

from cowork.common.paths import (
    dir_lstat,
    dir_rmtree,
    dir_unlink,
    opened_subdir_nofollow,
)
from cowork.db.scoped import ScopedSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.schemas.responses import Role
from cowork.services.channel_bindings import ChannelBindingService
from cowork.services.schedules import ScheduleService
from cowork.services.scratchpad_sessions import remove_conversation_sessions
from cowork.services.task_objects import TaskObjectService

# created_at is only second-precision, so rows of turns in the same second
# would otherwise interleave. `seq` is a per-conversation monotonic ordinal
# (see _next_seq) that resolves those ties deterministically. The role tiebreak
# only matters for legacy rows (all seq 0), keeping user before assistant; id is
# the final tiebreak.
_MESSAGE_ORDER = (
    Message.created_at,
    Message.seq,
    case((Message.role == Role.user, 0), else_=1),
    Message.id,
)


def _is_tool_row(content) -> bool:
    """True for a history-only tool block-row (all blocks are tool_use /
    tool_result). These carry prior tool calls for LLM-history replay and are
    hidden from the UI, which renders tool activity from message events."""
    return (
        isinstance(content, list)
        and len(content) > 0
        and all(
            isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result")
            for block in content
        )
    )


logger = logging.getLogger(__name__)

# ENG-1992: swapped in for an image content block a provider permanently
# rejected (a schema/shape mismatch, not a moderation refusal), so it must
# read as a removal notice, not as if the user said this.
_IMAGE_PLACEHOLDER_TEXT = (
    "[An image here could not be sent to the model and was removed "
    "automatically so this conversation could continue. Re-share it if you "
    "still need it referenced.]"
)


def _strip_image_blocks(content):
    """Replace image content blocks with a text placeholder, recursing into
    tool_result blocks' own nested content (a tool can return an image, e.g.
    a screenshot).

    Returns `(content, changed)` — `content` is the exact same object when
    nothing needed stripping, so a caller can skip a write when `changed` is
    False. Used to repair a conversation whose stored history contains an
    image block a provider permanently rejected (ENG-1992's
    ContentValidationError) — once removed, replay just works again, with no
    special-casing needed on future turns.
    """
    if not isinstance(content, list):
        return content, False

    changed = False
    new_blocks = []
    for block in content:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue
        if block.get("type") == "image":
            new_blocks.append({"type": "text", "text": _IMAGE_PLACEHOLDER_TEXT})
            changed = True
            continue
        if block.get("type") == "tool_result":
            nested, nested_changed = _strip_image_blocks(block.get("content"))
            if nested_changed:
                new_blocks.append({**block, "content": nested})
                changed = True
                continue
        new_blocks.append(block)
    return (new_blocks if changed else content), changed


def _skill_created_slug(event_data) -> str | None:
    """The draft slug of a persisted `response.skill_created` event, else None."""
    if (
        not isinstance(event_data, dict)
        or event_data.get("type") != "response.skill_created"
    ):
        return None
    skill = event_data.get("skill")
    slug = skill.get("slug") if isinstance(skill, dict) else None
    return slug if isinstance(slug, str) and slug else None


def _sweep_skill_drafts(session, project_id, slugs: set[str]) -> None:
    """Remove the on-disk skill drafts for `slugs` in a project's drafts dir.

    Called when a turn holding a skill card is deleted: the card's event is gone,
    so the draft would otherwise be orphaned. Best-effort — a missing project or
    folder is a no-op. Confined to direct children of the drafts dir.
    """
    project = session.get(Project, project_id)
    if project is None or not project.path:
        return
    # `<project>/.anton/skill_drafts` sits under the agent-writable tree, so a
    # planted symlink at `.anton`, `skill_drafts`, or the slug could redirect
    # this delete into another org. Pin the dir by O_NOFOLLOW descriptor and
    # rmtree each slug relative to it, never following a link (see
    # opened_subdir_nofollow). slug comes from an event, so reject anything that
    # is not a single path component before handing it to the kernel.
    try:
        with opened_subdir_nofollow(Path(project.path), ".anton", "skill_drafts") as d:
            for slug in slugs:
                if (
                    os.sep in slug
                    or (os.altsep and os.altsep in slug)
                    or slug in {"", ".", ".."}
                ):
                    continue
                try:
                    st = dir_lstat(d, slug)
                except FileNotFoundError:
                    continue
                try:
                    if stat.S_ISLNK(st.st_mode):
                        dir_unlink(d, slug)  # drop the link only, never follow it
                    elif stat.S_ISDIR(st.st_mode):
                        dir_rmtree(d, slug)
                except OSError:
                    logger.warning(
                        "Could not sweep skill draft %r on turn delete",
                        slug,
                        exc_info=True,
                    )
    except OSError:
        # No drafts dir (or a symlink squatting `.anton`/`skill_drafts`): nothing to sweep.
        return


def _discard_conversation_streams(conversation_id) -> None:
    """Drop a conversation's stale stream buffers + handle after a turn delete.

    A turn's buffer is keyed by `turn_id = len(messages)`; truncating the history
    makes the next turn reuse a deleted turn's buffer and replay the old answer
    instead of generating. Streaming owns the buffers — we just ask it to discard.
    Best-effort: never breaks the delete.
    """
    try:
        from cowork.streaming import discard_conversation

        discard_conversation(conversation_id)
    except Exception:
        logger.warning(
            "Could not discard streams for conversation %s",
            conversation_id,
            exc_info=True,
        )


class ConversationService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _default_project_id(self) -> UUID | None:
        """The caller's default project. Imported lazily: projects imports this
        module's models, so a top-level import would cycle."""
        from cowork.services.projects import ProjectService

        return ProjectService(self.session).default_project_id()

    def _next_seq(self, conversation_id: UUID) -> int:
        """Next per-conversation ordinal: max(seq) + 1, or 0 when empty.

        Keeps `seq` monotonic across the whole conversation so message order
        never depends on created_at's second-level resolution.

        ponytail: one extra query per persist, and max+1 races only if a single
        conversation runs two concurrent turns — which it can't today (turns are
        serialized). The id tiebreak in _MESSAGE_ORDER still bounds the fallout
        if that ever changes.
        """
        # ORDER BY seq DESC LIMIT 1 (not func.max): expressible through the
        # scoped .select() API, so no escape hatch to the raw session.
        last = self.session.exec(
            self.session.select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.seq.desc())
            .limit(1)
        ).first()
        return 0 if last is None else last.seq + 1

    def save_user_message(
        self,
        conversation_id: UUID,
        content,
        created_at: datetime | None = None,
        *,
        pending: bool = False,
    ) -> Message:
        """Persist a user message with the next monotonic `seq` (see _next_seq).

        `pending=True` marks the message in-flight (ENG-1231): the streaming path
        persists it at turn start so a mid-turn refresh shows the question, but it
        is kept out of replayed LLM history (get_ordered_messages) until the turn
        ends and finalize_pending clears the flag.
        """
        message = Message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            seq=self._next_seq(conversation_id),
            created_at=created_at,
            pending=pending,
        )
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        return message

    def finalize_pending(
        self, conversation_id: UUID, message_id: UUID | None = None
    ) -> None:
        """Clear the in-flight flag at turn end (ENG-1231), so the finished turn
        rejoins replayed LLM history.

        Pass `message_id` to finalize only that turn's row — the streaming
        producers do this so a completing turn cannot silently absorb an unrelated
        pending row that was stranded by a hard crash (killed between the pending
        persist and finalize) on an earlier turn. Such an orphan stays pending
        (still shown in the UI, still excluded from history) instead of being
        folded into history as a question with no answer.

        With no `message_id`, clears every pending row for the conversation — a
        defensive/idempotent form for callers that just want a clean slate.

        Idempotent: a no-op when nothing matches.
        """
        stmt = (
            self.session.select(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.pending == True)  # noqa: E712 — SQL boolean column, not Python identity
        )
        if message_id is not None:
            stmt = stmt.where(Message.id == message_id)
        pending_rows = self.session.exec(stmt).all()
        if not pending_rows:
            return
        for message in pending_rows:
            message.pending = False
            self.session.add(message)
        self.session.commit()

    def repair_image_content(self, conversation_id: UUID) -> list[UUID]:
        """Strip image content blocks from every stored message in a
        conversation, replacing each with a text placeholder.

        Called when a turn dies on ContentValidationError (ENG-1992): the
        provider permanently rejected some image block in history, and
        retrying identically fails identically forever, because the
        translation that produced the bad block runs fresh from this same
        stored data on every call. Fixing the DATA once, here, rather than
        special-casing replay means every future turn just works — no flag,
        no per-turn filtering to maintain.

        Scans every message (including pending/tool-only rows — a poisoned
        image could be in either) rather than trying to identify "the" one
        culprit from the provider's error: that needs mapping a request-
        relative index (e.g. Responses' "input[70]") back to a specific
        stored message, which isn't reliable across providers or dialects.
        Structurally finding every image block is deterministic and safe
        instead.

        Returns the ids of messages that were actually changed — empty if
        none needed it (e.g. the failure turned out not to be image-shaped
        after all, so there's nothing here to fix).
        """
        messages = self.get_ordered_messages(conversation_id, include_pending=True)
        repaired: list[UUID] = []
        for message in messages:
            new_content, changed = _strip_image_blocks(message.content)
            if not changed:
                continue
            message.content = new_content
            self.session.add(message)
            repaired.append(message.id)
        if repaired:
            self.session.commit()
        return repaired

    def last_message_at(self, conversation_id: UUID) -> datetime | None:
        """Timestamp of the most recent message, or None for an empty
        conversation. This is the real "last activity" — the stored
        `conversation.modified_at` only moves on rename/move, never on a turn
        (ENG-961), so it must be derived from the messages themselves. The
        `messages(conversation_id, created_at)` index makes this an index seek.
        """
        last = self.session.exec(
            self.session.select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(1)
        ).first()
        return last.created_at if last is not None else None

    def list_conversations(
        self,
        project_id: UUID | None = None,
        limit: int = 50,
        all_projects: bool = False,
    ) -> list[Conversation]:
        """Conversations most-recently-*active* first. Ordering by derived
        activity (not `created_at`) is what keeps the `limit` window holding
        recently-*used* tasks rather than recently-*created* ones (ENG-961).

        The order key is a correlated MAX(message.created_at) subquery rather
        than a join+group_by so the statement stays a single-entity select —
        the scoped session's org filter still applies and `exec()` returns
        clean Conversation rows. Empty conversations coalesce to their own
        `created_at` so a NULL max can't sort them to the end.
        """
        last_activity = func.coalesce(
            sa_select(func.max(Message.created_at))
            .where(Message.conversation_id == Conversation.id)
            .correlate(Conversation)
            .scalar_subquery(),
            Conversation.created_at,
        )
        stmt = self.session.select(Conversation)
        # Conversations are personal: the scoped session enforces the org, but
        # user_id has no automatic scoping (see PinService), so without this
        # every org member's tasks show in everyone's list.
        if self.session.scope.org_mode:
            stmt = stmt.where(Conversation.created_by == self.session.scope.user_id)
        if not all_projects:
            stmt = stmt.where(
                Conversation.project_id == (project_id or self._default_project_id())
            )
        # created_at then id break ties deterministically so equal-activity rows
        # (e.g. two empty conversations) keep a stable order across polls.
        stmt = stmt.order_by(
            last_activity.desc(), Conversation.created_at.desc(), Conversation.id
        ).limit(limit)
        return list(self.session.exec(stmt).all())

    def list_conversations_with_activity(
        self,
        project_id: UUID | None = None,
        limit: int = 50,
        all_projects: bool = False,
    ) -> list[tuple[Conversation, datetime]]:
        """`list_conversations` paired with each row's last-activity timestamp
        for serialization. The value reuses `last_message_at` (an index seek on
        the ENG-961 composite index); callers that don't need the value (e.g.
        search indexing) use `list_conversations` and skip the per-row lookup.
        """
        convs = self.list_conversations(
            project_id=project_id, limit=limit, all_projects=all_projects
        )
        return [
            (conv, self.last_message_at(conv.id) or conv.created_at) for conv in convs
        ]

    def _owned(self, conversation_id: UUID) -> Conversation | None:
        """Fetch a conversation only if it belongs to the caller.

        Conversations are personal. The scoped session enforces the org, but
        user_id has no automatic scoping (see PinService) and a bare
        session.get by PK bypasses even the org filter — so every by-id access
        must go through here or a member can read/rename/delete another
        member's chat by guessing its id. Local mode has one user, so no owner
        filter applies.
        """
        stmt = self.session.select(Conversation).where(Conversation.id == conversation_id)
        if self.session.scope.org_mode:
            stmt = stmt.where(Conversation.created_by == self.session.scope.user_id)
        return self.session.exec(stmt).first()

    def owned_ids(self, conversation_ids: Iterable[UUID]) -> set[UUID]:
        """The subset of `conversation_ids` the caller owns, in one query.

        Same rule as `_owned`, batched. The artifact roots resolver walks one
        directory per conversation and holds every id at once, so without this
        a plain list request would issue a SELECT per directory.
        """
        ids = list(conversation_ids)
        if not ids:
            return set()
        stmt = self.session.select(Conversation).where(Conversation.id.in_(ids))
        if self.session.scope.org_mode:
            stmt = stmt.where(Conversation.created_by == self.session.scope.user_id)
        return {row.id for row in self.session.exec(stmt).all()}

    def get_conversation(self, conversation_id: UUID) -> Conversation:
        conversation = self._owned(conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found")
        return conversation

    def create_conversation(
        self,
        topic: str,
        project_id: UUID | None = None,
        conversation_id: UUID | None = None,
        harness: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Conversation:
        """`conversation_id` lets the caller adopt a client-allocated id —
        the composer allocates one up front so attachments can be uploaded
        against it before the first stream creates the conversation."""
        # Anchor the parent: the target project must be visible in scope —
        # otherwise org A could link a conversation to org B's project and
        # leak its name/path through serialization.
        target_project_id = project_id or self._default_project_id()
        if self.session.get(Project, target_project_id) is None:
            raise ValueError("Project not found")
        conversation = Conversation(
            topic=topic,
            project_id=target_project_id,
            harness=harness,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if conversation_id is not None:
            conversation.id = conversation_id
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def project_by_name(self, name: str | None) -> Project | None:
        if not name:
            return None
        # Delegate so the `general` self-heal lives in one place.
        # Lazy import: projects imports this module.
        from cowork.services.projects import ProjectService

        return ProjectService(self.session).get_or_provision_by_name_or_none(name)

    def update_conversation(
        self,
        conversation_id: UUID,
        topic: str | None = None,
        project_id: UUID | None = None,
    ) -> Conversation:
        conversation = self._owned(conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found")
        if topic is not None:
            conversation.topic = topic
        if project_id is not None:
            # Anchor the move target: the project must be visible in scope.
            if self.session.get(Project, project_id) is None:
                raise ValueError("Project not found")
            conversation.project_id = project_id
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def update_history_compaction(
        self,
        conversation_id: UUID,
        summary: str,
        cutoff_message_id: UUID,
    ) -> None:
        """Persist anton's latest compacted history summary + cutoff.

        Best-effort: silently no-ops if the conversation is gone (this runs
        from a turn's cleanup path, after the turn's real outcome is settled).
        """
        conversation = self._owned(conversation_id)
        if conversation is None:
            return
        conversation.history_summary = summary
        conversation.history_summary_cutoff_id = cutoff_message_id
        self.session.add(conversation)
        self.session.commit()

    def delete_conversation(self, conversation_id: UUID) -> bool:
        """Owner-scoped delete for the request path: a member can only delete
        their own conversation."""
        conversation = self._owned(conversation_id)
        if conversation is None:
            return False
        return self._delete_conversation(conversation)

    def delete_conversation_row(self, conversation: Conversation) -> bool:
        """Owner-AGNOSTIC cascade delete, given an already-authorized row.

        Used by ProjectService.delete_project, which is an intentional org-wide
        cleanup: the project's conversations may belong to several members, and
        skipping the foreign ones would orphan their messages/events/task
        objects/attachment bytes (the ENG-701 orphaning the cascade exists to
        prevent). The caller has already scoped the fetch to the org."""
        return self._delete_conversation(conversation)

    def _delete_conversation(self, conversation: Conversation) -> bool:
        conversation_id = conversation.id
        messages = self.session.exec(
            self.session.select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(*_MESSAGE_ORDER)
        ).all()
        for message in messages:
            for event in self.session.exec(
                self.session.select(MessageEvent).where(
                    MessageEvent.message_id == message.id
                )
            ).all():
                self.session.delete(event)
            self.session.delete(message)
        # Drop the conversation's object index too — otherwise the rows
        # outlive the conversation as orphans pointing at artifacts no
        # task owns anymore.
        TaskObjectService(self.session).delete_for_conversation(conversation)
        # Drop the conversation's uploaded attachments (rows + bytes) — they're
        # keyed by conversation id and would otherwise orphan in the file store
        # forever, invisible in any UI (ENG-701). Stage the row deletes into
        # THIS transaction (single commit below, so a crash can't leave a
        # half-deleted "ghost" conversation), then unlink the bytes only after
        # the commit succeeds.
        from cowork.services.files import (
            FileService,
            attachment_purpose,
            remove_conversation_workspace_dir,
            unlink_file_dirs,
        )

        attachment_dirs = FileService(self.session).delete_by_purpose(
            attachment_purpose(str(conversation_id))
        )
        # Three more tables point at this conversation, and unlike the rows above
        # they are not the conversation's own data: a schedule and its runs record
        # the conversation a run produced, and a channel binding records the one
        # its external chat is pinned to. No foreign key in this schema declares
        # an `ondelete`, so Postgres refuses the delete below while any of them
        # still points here, and SQLite (desktop, and the whole test suite) runs
        # with foreign keys off and orphans them instead. Each owning service
        # releases its own link and keeps its rows, staged into this transaction.
        ScheduleService(self.session).release_conversation(conversation_id)
        ChannelBindingService(self.session).release_conversation(conversation_id)
        # anton snapshots the scratchpad namespace to
        # `<project>/.anton/scratchpad-sessions/<conversation_id>/` so variables survive
        # the pad process being replaced each turn (ENG-1124). Nothing else prunes those
        # — only a whole-project delete would — so they accumulate one directory per
        # conversation, and a namespace can hold the injected DS_* credentials (ENG-392),
        # making a stale snapshot data-at-rest rather than just disk.
        # Resolve the project path HERE, while the conversation is still attached; the
        # removal happens after the commit for the same reason as the attachment bytes.
        session_project_path = (
            conversation.project.path if conversation.project is not None else None
        )
        # Its buffers and turn-index entry outlive the rows otherwise: on the
        # Redis backend /in-flight keeps naming a turn whose conversation is
        # gone, and a reused turn_id would replay a deleted turn's answer.
        _discard_conversation_streams(conversation_id)
        self.session.delete(conversation)
        self.session.commit()
        unlink_file_dirs(attachment_dirs)
        remove_conversation_sessions(session_project_path, conversation_id)
        # Also drop the per-conversation workspace (staged attachments +
        # instructions on the shared mount) so it doesn't orphan there.
        remove_conversation_workspace_dir(session_project_path, conversation_id)
        return True

    def delete_turn(self, conversation_id: UUID, turn_index: int) -> int:
        """Delete a turn and everything after it.

        turn_index is 0-based counting only VISIBLE assistant messages —
        hidden tool rows (tool_use / tool_result) are skipped so the index
        matches the UI's, which never shows them. The turn's opening user
        message and all subsequent messages (including this turn's tool rows)
        are removed. Returns the number of messages deleted.
        """
        conversation = self.get_conversation(conversation_id)  # raises if not found
        messages = list(
            self.session.exec(
                self.session.select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(*_MESSAGE_ORDER)
            ).all()
        )
        # Find the Nth visible assistant message (0-based).
        assistant_count = -1
        cut_from = None
        for i, m in enumerate(messages):
            if m.role.value == "assistant" and not _is_tool_row(m.content):
                assistant_count += 1
                if assistant_count == turn_index:
                    # Walk back over this turn's hidden tool rows (the
                    # tool_result row is role=user, so a plain i-1 check would
                    # stop on it and orphan the real user input + tool_use).
                    cut_from = i
                    j = i - 1
                    while j >= 0 and _is_tool_row(messages[j].content):
                        cut_from = j
                        j -= 1
                    # Include the user message that opened the turn.
                    if j >= 0 and messages[j].role.value == "user":
                        cut_from = j
                    break
        if cut_from is None:
            raise ValueError(f"Turn {turn_index} not found")
        to_delete = messages[cut_from:]
        swept_slugs: set[str] = set()
        for msg in to_delete:
            for event in self.session.exec(
                self.session.select(MessageEvent).where(
                    MessageEvent.message_id == msg.id
                )
            ).all():
                slug = _skill_created_slug(event.event_data)
                if slug:
                    swept_slugs.add(slug)
                self.session.delete(event)
            self.session.delete(msg)
        # Clearing the whole history (truncate from turn 0) is the UI's
        # "delete chat history". When nothing remains, the conversation no
        # longer owns anything it produced — drop its object index so a
        # cleared chat doesn't keep resurfacing old artifacts. A partial
        # truncation leaves the index alone (rows aren't turn-scoped, and
        # surviving turns may still reference the artifact).
        if cut_from == 0:
            TaskObjectService(self.session).delete_for_conversation(conversation)
        self.session.commit()
        # After the rows are gone, reap the on-disk drafts whose only card lived
        # in a deleted turn (their `skill_created` events were just removed).
        if swept_slugs:
            _sweep_skill_drafts(self.session, conversation.project_id, swept_slugs)
        # Drop stale stream buffers so a resend regenerates instead of replaying
        # a deleted turn (turn_id == message count collides after truncation).
        _discard_conversation_streams(conversation_id)
        # Rewinding history must rewind the scratchpad too. The namespace snapshot is at
        # the state the *deleted* turns left it in, so without this a resend reloads
        # variables created by a turn the user just removed — the visible history and the
        # agent's actual state would disagree. Cheapest correct answer is to drop the
        # snapshot: the agent then rebuilds from the surviving history, which is exactly
        # what the user asked for by truncating.
        project = self.session.get(Project, conversation.project_id)
        remove_conversation_sessions(project.path if project else None, conversation_id)
        return len(to_delete)

    def save_assistant_turn(
        self,
        conversation_id: UUID,
        text: str,
        events: list[dict],
        harness: str | None = None,
        tool_rows: list[dict] | None = None,
    ) -> None:
        """Persist an assistant turn.

        `tool_rows` are the turn's tool block-messages ({role, content} with
        `tool_use` / `tool_result` blocks). They are written as their own rows
        AHEAD of the visible assistant message so the next turn's history
        replays a valid tool_use → tool_result → text sequence. All rows share
        one commit (hence one `created_at`); `seq` fixes their order, since the
        role tiebreak in _MESSAGE_ORDER would otherwise sort tool_result (user)
        ahead of tool_use (assistant). Hidden from the UI by `get_messages`.
        """
        # Persist when there's body text OR any events — an artifact-only turn
        # (the agent writes a file and says little/nothing) carries no text but
        # emits a `response.artifact_created` event, and that event must survive
        # reload so the inline card replays identically.
        if not text and not events and not tool_rows:
            return
        # Anchor the write to a parent loaded through THIS session's scope —
        # detached writers (producer) call this on a fresh session, and the
        # conversation may be gone or out-of-scope by now.
        self.get_conversation(conversation_id)
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=text,
            harness=harness,
        )
        ordered_rows = [
            Message(
                conversation_id=conversation_id,
                role=row["role"],
                content=row["content"],
                harness=harness,
            )
            for row in (tool_rows or [])
        ]
        ordered_rows.append(assistant_msg)
        base_seq = self._next_seq(conversation_id)
        for offset, message in enumerate(ordered_rows):
            message.seq = base_seq + offset
            self.session.add(message)
        self.session.commit()
        self.session.refresh(assistant_msg)

        for event_seq, event_data in enumerate(events):
            self.session.add(
                MessageEvent(
                    message_id=assistant_msg.id,
                    sequence_number=event_seq,
                    event_data=event_data,
                )
            )
        if events:
            self.session.commit()
            # A skill card supersedes its earlier versions: when this turn emits a
            # `skill_created`, drop the same slug's earlier skill_created events so
            # history holds ONE card per skill (the latest). Keeps the on-disk
            # single-draft-per-slug model consistent with the transcript, and makes
            # the "latest card" durable across reload (not just a render-time dedup).
            new_slugs = {s for s in (_skill_created_slug(e) for e in events) if s}
            if new_slugs:
                self._supersede_skill_cards(
                    conversation_id, assistant_msg.id, new_slugs
                )

    def _supersede_skill_cards(
        self, conversation_id: UUID, keep_message_id: UUID, slugs: set[str]
    ) -> None:
        """Delete earlier `skill_created` events (for `slugs`) in this conversation,
        keeping only the one on `keep_message_id`.

        ponytail: scans the conversation's message events in Python (JSON slug
        isn't portably queryable in SQL). Bounded — runs only on a skill-emitting
        turn, which is rare; upgrade to an indexed column if skills get chatty.
        """
        msg_ids = [
            m.id
            for m in self.session.exec(
                self.session.select(Message).where(
                    Message.conversation_id == conversation_id
                )
            ).all()
        ]
        if not msg_ids:
            return
        deleted = False
        for event in self.session.exec(
            self.session.select(MessageEvent).where(
                MessageEvent.message_id.in_(msg_ids)
            )
        ).all():
            if event.message_id == keep_message_id:
                continue
            if _skill_created_slug(event.event_data) in slugs:
                self.session.delete(event)
                deleted = True
        if deleted:
            self.session.commit()

    def get_ordered_messages(
        self, conversation_id: UUID, *, include_pending: bool = False
    ) -> list[Message]:
        """All messages of a conversation in canonical order (see
        _MESSAGE_ORDER). Includes history-only tool rows — harnesses replay
        them into the LLM context; use get_messages for the UI-facing view.

        Excludes the in-flight pending user message by default (ENG-1231): this is
        the LLM-history read, and the current turn's input arrives separately, so
        replaying it here would double-feed it. Pass include_pending=True only if a
        caller genuinely needs the not-yet-finalized row."""
        # Anchor the parent: Message has no org_id, so tenancy comes from
        # resolving the conversation through the scoped session — a foreign
        # id must answer like a nonexistent one, not leak another org's
        # history (the remote-turn replay path passes ids from the wire).
        self.get_conversation(conversation_id)  # raises if not found
        stmt = self.session.select(Message).where(
            Message.conversation_id == conversation_id
        )
        if not include_pending:
            stmt = stmt.where(Message.pending == False)  # noqa: E712 — SQL boolean column, not Python identity
        return list(self.session.exec(stmt.order_by(*_MESSAGE_ORDER)).all())

    def get_messages(self, conversation_id: UUID) -> list[dict]:
        self.get_conversation(conversation_id)  # raises if not found
        messages = self.session.exec(
            self.session.select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(*_MESSAGE_ORDER)
        ).all()
        result = []
        for message in messages:
            if _is_tool_row(message.content):
                continue  # history-only tool row — not shown in the chat
            events = self.session.exec(
                self.session.select(MessageEvent)
                .where(MessageEvent.message_id == message.id)
                .order_by(MessageEvent.sequence_number)
            ).all()
            item = {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
                "events": [e.event_data for e in events],
            }
            if message.harness:
                item["harness"] = message.harness
            result.append(item)
        return result
