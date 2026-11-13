from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import stat
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from functools import partial
from typing import Any, NamedTuple
from uuid import UUID

import re2
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from anton.core.dispatch import OutboundMessage
from cowork.handlers._turn_history import reject_unreplayable_tool_rows
from cowork.build_info import build_trace_metadata
from cowork.channels.registry import PluginRegistry, get_registry
from cowork.channels.sharing import shared_state
from cowork.common.logger import log_context
from cowork.common.paths import O_NOFOLLOW, O_NONBLOCK, dir_open, is_child_name, opened_subdir_nofollow, read_bounded
from cowork.db.scoped import (
    LOCAL_SCOPE,
    SYSTEM_SCOPE,
    ScopedSession,
    TenantScope,
    scope_for_background_context,
    scope_for_org,
    unsafe_unscoped_session,
)
from cowork.db.units import FilesystemBusy, conversation_writes, run_db, run_efs, run_efs_unit
from cowork.handlers.responses import stage_remote_workspace
from cowork.harnesses.base import ChannelContext, HarnessProvider, get_harness
from cowork.models.channel import SENDER_ALLOWED, SENDER_BLOCKED, ChannelBinding, ChannelSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message as DBMessage
from cowork.models.project import Project
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.common.settings.user_settings import UserSettings, use_settings_scope, use_turn_settings
from cowork.services.artifact_roots import project_artifacts_base
from cowork.services.channel_bindings import ChannelBindingService
from cowork.services.channel_pairing import PAIRED_NOTICE, ChannelPairingService, code_in
from cowork.services.channel_senders import ChannelSenderService
from cowork.services.channels import ChannelConfigService, resolve_installation_by_external_account
from cowork.services.conversations import ConversationService
from cowork.services.files import store_file, upload_basename
from cowork.services.settings import unit_setting_service
from cowork.services.skills import SkillService
from cowork.streaming.answer_text import accumulate_answer_text
from cowork.turnqueue.remote_turn import RemoteTurnFailed, remote_turn_events

log = logging.getLogger(__name__)

# Channel turns run the server-configured channels_harness (default anton),
# pinned per conversation via per-message harness identity. The UI harness
# hotswitch (UserSettings.harness) never applies to channels.
DEFAULT_CHANNEL_HARNESS = "anton"
_DEFAULT_THREAD_KEY = "__default__"


def _fits(conversation: Conversation, shared: bool | None) -> bool:
    """Whether a message from a shared (or DM) chat may be added to
    `conversation` in org mode: one the runtime made, marked to match. An
    unknown `shared` (None) fits any such conversation: it never creates or
    changes a mark, so it cannot move DM text into a readable conversation."""
    if conversation.created_by is not None:
        return False
    return shared is None or (conversation.channel_type is not None) == shared


def _served_on_route(scoped: ScopedSession, binding: ChannelBinding, conversation_id) -> bool:
    """Whether the runtime itself last served `conversation_id` on `binding`
    (its session row names it), as opposed to the route having been pointed
    at it."""
    return scoped.exec(
        scoped.select(ChannelSession).where(
            ChannelSession.binding_id == binding.id,
            ChannelSession.anton_session_id == str(conversation_id),
        )
    ).first() is not None


def _reusable_in_org(scoped: ScopedSession, binding: ChannelBinding, conversation: Conversation, shared: bool | None) -> bool:
    """Org mode's reuse rule: the runtime's own conversation, marked to match,
    on this route only. An unknown message joins a marked conversation only
    when this route was the one served into it, so a group DM route pointed at
    a marked conversation (by an older server mid-rollout) starts afresh."""
    if not _fits(conversation, shared) or _shared_by_another_route(scoped, binding, conversation.id):
        return False
    if shared is None and conversation.channel_type is not None:
        return _served_on_route(scoped, binding, conversation.id)
    return True


def _shared_by_another_route(scoped: ScopedSession, binding: ChannelBinding, conversation_id) -> bool:
    """Whether a route other than `binding` points at `conversation_id`; such a
    conversation (a legacy pin) would mix two chats' messages."""
    return scoped.exec(
        scoped.select(ChannelBinding).where(
            ChannelBinding.anton_conversation_id == conversation_id,
            ChannelBinding.id != binding.id,
        )
    ).first() is not None


def turn_used_tools(events: list[dict]) -> bool:
    """Tool/scratchpad activity rides on stream events as ``tool_use_id``."""
    return any(isinstance(event, dict) and "tool_use_id" in event for event in events)


def is_new_command(text: str, *, is_mention: bool | None = None) -> bool:
    """True for a bare /new message; a /new@bot suffix counts unless the platform says the mention isn't us."""
    tokens = [
        t for t in (text or "").split()
        if not (t.startswith("@") or (t.startswith("<@") and t.endswith(">")))
    ]
    if len(tokens) != 1:
        return False
    cmd = tokens[0].lower()
    if cmd == "/new":
        return True
    return cmd.startswith("/new@") and len(cmd) > len("/new@") and is_mention is not False


# Platform typing indicators expire after a few seconds, so refresh while
# the turn runs. Module-level so tests can shrink it.
TYPING_REFRESH_S = 4.0

MAX_TURN_ATTACHMENTS = 3
#: The largest file a turn sends through an adapter that states no upload cap
#: of its own. Each first-party bridge states its platform's cap as
#: ``max_attachment_bytes``, and a turn never reads a file past it.
MAX_TURN_ATTACHMENT_BYTES = 100 * 1024 * 1024

#: What a sender hears when the pool or the file store refuses their message
#: before the agent answers it.
BUSY_NOTICE = "I'm overloaded right now and couldn't start on that. Please send it again in a minute."

# An end-of-turn save tries this many times while the pool stays full, the
# n-th retry after n times the backoff, before the conversation goes without it.
_SAVE_TRIES = 3
_SAVE_BACKOFF_S = 1.0

# re2 logs pattern-compile/search errors to the process's raw stderr by
# default (before absl::InitializeLog() — bypasses Python logging entirely),
# which would misrepresent an org-controlled trigger_pattern mistake as a
# server error in production logs.
_QUIET_REGEX_OPTIONS = re2.Options()
_QUIET_REGEX_OPTIONS.log_errors = False


@dataclass(frozen=True)
class TurnFile:
    """One file a turn produced, read for an adapter's send_attachment hook."""

    filename: str
    data: bytes


class Admission(Enum):
    """What the sender gate decided for one inbound message."""

    ADMIT = "admit"
    HOLD = "hold"
    PAIRED = "paired"


@dataclass(frozen=True)
class TurnOutcome:
    """What one channel turn produced."""

    reply: str
    used_tools: bool
    # The "path" of each artifact card the turn emitted
    # (response.artifact_created): the artifact's primary file, or its folder
    # when it has none. The card itself has no typed model; only its path is
    # read here.
    artifact_paths: tuple[str, ...]


class _TurnSetup(NamedTuple):
    """What a channel turn's first unit reads: the harness that runs the turn,
    and the settings every later read in the turn is served from."""

    harness_id: str
    settings: UserSettings


def read_turn_files(*, project_path: str, paths: Sequence[str], max_bytes: int) -> list[TurnFile]:
    """The files to send for a turn's artifact paths, at most
    MAX_TURN_ATTACHMENTS: each path that lies under the project's artifacts
    base and is a regular file of at most ``max_bytes``, the adapter's own
    upload cap. A file past the cap is skipped on its size alone, before any
    of it is read.

    The agent writes the artifact folders, so every folder below the project
    and the file itself are opened without following a link: a link it
    planted fails to open instead of sending a file from outside the folder.
    A path whose file can't be read is skipped.
    """
    project = Path(project_path)
    base = project_artifacts_base(project_path).relative_to(project).parts
    files: list[TurnFile] = []
    seen: set[str] = set()
    for path in paths:
        if len(files) == MAX_TURN_ATTACHMENTS:
            break
        if path in seen:
            continue
        seen.add(path)
        try:
            parts = Path(path).relative_to(project).parts
        except ValueError:
            continue
        # Below the base's folders: one artifact folder, then the file.
        if parts[: len(base)] != base or len(parts) < len(base) + 2 or not all(map(is_child_name, parts)):
            continue
        try:
            with opened_subdir_nofollow(project, *parts[:-1]) as folder:
                fd = dir_open(folder, parts[-1], os.O_RDONLY | O_NONBLOCK | O_NOFOLLOW)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                        continue
                    data = b"".join(read_bounded(fd, info.st_size))
                finally:
                    os.close(fd)
        except OSError:
            log.warning("channel turn: could not read an artifact's file to send", exc_info=True)
            continue
        files.append(TurnFile(filename=parts[-1], data=data))
    return files


async def typing_loop(adapter: Any, address: Any) -> None:
    while True:
        try:
            await adapter.set_typing(address=address)
        except Exception:
            log.debug("set_typing failed; continuing without indicator")
        await asyncio.sleep(TYPING_REFRESH_S)


def conversation_link(conversation_id: Any) -> str | None:
    template = (get_app_settings().conversation_link_template or "").strip()
    if not template:
        return None
    try:
        return template.format(conversation_id=conversation_id)
    except (KeyError, IndexError, ValueError):
        log.warning("invalid conversation_link_template; skipping link")
        return None


class _KeyedLocks:
    """Per-key async locks with refcounted cleanup.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._refcounts: dict[str, int] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            self._refcounts[key] = self._refcounts.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                self._refcounts[key] -= 1
                if self._refcounts[key] <= 0:
                    self._refcounts.pop(key, None)
                    self._locks.pop(key, None)


@dataclass(frozen=True)
class _Claim:
    """The installation that claims a platform account: its org, or None for
    the local installation."""

    org_id: str | None


def _claiming_installation(session: ScopedSession, *, channel_type: str, routing_key: str) -> _Claim | None:
    """Which installation claims ``routing_key``, as a unit. The lookup runs
    before any org scope exists, so it reads the raw session
    (resolve_installation_by_external_account)."""
    install = resolve_installation_by_external_account(
        unsafe_unscoped_session(session), channel_type, routing_key
    )
    return None if install is None else _Claim(org_id=install.org_id)


class LiveAdapterRegistry:
    """Process-wide cache of live channel adapters, keyed by (channel_type,
    org_id). org_id=None is the local/desktop installation — today's single-
    instance-per-channel-type behavior verbatim; an org lookup never sees it."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self._registry = registry if registry is not None else get_registry()
        self._cache: dict[tuple[str, str | None], Any] = {}

    def get(self, channel_type: str, org_id: str | None = None) -> Any | None:
        """Live adapter for a channel/org, or None if not configured/active."""
        return self._cache.get((channel_type, org_id))

    async def get_or_refresh(self, channel_type: str, org_id: str | None) -> Any | None:
        """Cache hit → return it, no database read. Miss (first webhook for
        an org this replica hasn't loaded yet) → refresh, then return the
        result either way. Used by the webhook path, which has no prior
        chance to have called refresh for an org it just resolved."""
        cached = self.get(channel_type, org_id)
        if cached is not None:
            return cached
        await self.refresh(channel_type, org_id)
        return self.get(channel_type, org_id)

    async def resolve_org_bridge(
        self, channel_type: str, routing_key: str
    ) -> tuple[Any, str | None] | None:
        """What the webhook path's org_resolver actually calls: unverified
        routing key -> which installation claims it -> that installation's own
        bridge. None means nothing claims this key."""
        claim = await run_db(
            partial(_claiming_installation, channel_type=channel_type, routing_key=routing_key),
            scope=SYSTEM_SCOPE,
        )
        if claim is None:
            return None
        bridge = await self.get_or_refresh(channel_type, claim.org_id)
        return (bridge, claim.org_id) if bridge is not None else None

    async def refresh(self, channel_type: str, org_id: str | None = None) -> bool:

        plugin = self._registry.get(channel_type)
        if plugin is None:
            self._cache.pop((channel_type, org_id), None)
            return False
        # LOCAL_SCOPE for local mode, org scope otherwise.
        creds = await run_db(
            lambda session: ChannelConfigService(session, registry=self._registry).load_credentials(
                channel_type
            ),
            scope=scope_for_org(org_id),
        )
        try:
            adapter = await plugin.factory(creds)
        except Exception:
            log.exception("failed building live adapter for channel %s (org=%s)", channel_type, org_id)
            adapter = None
        if adapter is None:
            self._cache.pop((channel_type, org_id), None)
            return False
        self._cache[(channel_type, org_id)] = adapter
        return True

    async def refresh_all(self) -> list[str]:
        """Local-mode boot bootstrap: one adapter per plugin, org_id=None. Org
        installations refresh lazily on first webhook — see the resolver in
        webhooks.py — since channels aren't reachable in org mode yet anyway."""
        active: list[str] = []
        for plugin in self._registry.all():
            if await self.refresh(plugin.channel_type):
                active.append(plugin.channel_type)
        return active

    async def remove(self, channel_type: str, org_id: str | None = None) -> None:

        adapter = self._cache.pop((channel_type, org_id), None)
        if adapter is not None:
            try:
                await adapter.shutdown()
            except Exception:
                log.exception("error shutting down channel adapter %s", channel_type)

    async def shutdown(self) -> None:
        for adapter in list(self._cache.values()):
            try:
                await adapter.shutdown()
            except Exception:
                log.exception("error shutting down channel adapter")
        self._cache.clear()


class AntonChannelRuntime:
    """Inbound sink: resolve binding → conversation → run Anton → deliver."""

    def __init__(
        self,
        adapters: LiveAdapterRegistry,
        *,
        default_project_id: UUID | None = None,
    ) -> None:
        self._adapters = adapters
        # None = resolve per scope. A fixed id only resolves for the seeded
        # desktop row; each org has its own, so pinning one here would 404 the
        # day channels are enabled in cloud (they are 403-gated today).
        self._default_project_id = default_project_id
        self._locks = _KeyedLocks()

    def _resolve_default_project_id(self, scoped: ScopedSession) -> UUID | None:
        if self._default_project_id is not None:
            return self._default_project_id
        from cowork.services.projects import ProjectService
        return ProjectService(scoped).default_project_id()

    @staticmethod
    def _lock_key(channel_type: str, event: Any) -> str:
        thread_key = event.address.thread_id or _DEFAULT_THREAD_KEY
        return f"{channel_type}:{event.address.platform_id}:{thread_key}"

    async def handle(self, channel_type: str, event: Any, org_id: str | None = None) -> None:
        log.info(
            "channel %s: runtime received inbound from %s thread=%s",
            channel_type, event.address.platform_id, event.address.thread_id,
        )
        async with self._locks.acquire(self._lock_key(channel_type, event)):
            await self._handle_locked(channel_type, event, org_id)

    async def _handle_locked(self, channel_type: str, event: Any, org_id: str | None) -> None:
        # One scope per turn. org_id comes from the webhook's own org
        # resolution (resolve_bridge) — None means local mode, or (in org
        # mode) a genuinely unresolved org, which still fails closed here.
        # An org_id alone never turns org mode on in local tenancy.
        #
        # Every read and write below is a unit of its own, so the turn holds
        # no pooled connection while the agent runs and the event loop never
        # waits for one.
        #
        # A unit the pool or the file store refuses before the agent answers
        # ends the turn with that refusal, so the event records `failed`, and
        # the sender hears BUSY_NOTICE rather than nothing: the platform got
        # its acknowledgement and never sends the message again. The saves
        # after the answer retry and never raise a busy refusal (_save_turn_row).
        scope = scope_for_org(org_id) if org_id else scope_for_background_context()
        # The sender comes first: nothing below, not even a busy notice, may
        # answer someone the owner has not allowed.
        try:
            admission = await run_db(partial(self._admit_sender, channel_type=channel_type, event=event), scope=scope)
        except PoolTimeoutError:
            log.warning("channel %s: no database connection to check the sender; the message was dropped", channel_type)
            raise
        if admission is Admission.PAIRED:
            log.info("channel %s: a sender paired with a one-time code", channel_type)
            await self._deliver(channel_type, event, PAIRED_NOTICE, org_id)
            return
        if admission is Admission.HOLD:
            log.info("channel %s: held a message from a sender the owner has not allowed", channel_type)
            return
        try:
            # A known thread's binding is a row. Only a new one may start in
            # the general project, whose folder is checked, which is file
            # store work.
            binding = await run_db(partial(self._find_binding, channel_type=channel_type, event=event), scope=scope)
            if binding is None:
                binding = await run_efs_unit(
                    partial(self._resolve_or_create_binding, channel_type=channel_type, event=event), scope=scope
                )
        except (PoolTimeoutError, FilesystemBusy):
            # The trigger rule is unknown here: answer only a message addressed to the bot.
            if event.message.is_mention or not event.message.is_group:
                await self._send_busy_notice(channel_type=channel_type, event=event, org_id=org_id)
            raise
        log.info(
            "channel %s: binding %s → project %s (trigger=%s)",
            channel_type, binding.id, binding.anton_project_id, binding.trigger_rule,
        )
        if not await self._should_respond(binding, event):
            log.info("channel %s: trigger rule %r skipped a message", channel_type, binding.trigger_rule)
            return
        if is_new_command(self._event_text(event), is_mention=event.message.is_mention):
            try:
                await self._start_fresh(scope, channel_type, binding, event, org_id)
            except (PoolTimeoutError, FilesystemBusy):
                await self._send_busy_notice(channel_type=channel_type, event=event, org_id=org_id)
                raise
            return
        # Optional hook: adapters with set_typing show a typing indicator
        # for the duration of the turn; others are untouched.
        adapter = self._adapters.get(channel_type, org_id)
        typing = None
        if adapter is not None and callable(getattr(adapter, "set_typing", None)):
            typing = asyncio.create_task(typing_loop(adapter, event.address))
        try:
            conversation = await run_db(
                partial(self._open_conversation, binding=binding, event=event), scope=scope
            )
            channel_context = ChannelContext(
                channel_type=channel_type,
                is_group=bool(event.message.is_group),
                display_name=binding.display_name,
                instructions=binding.instructions,
            )
            # Nested get_user_settings() reads (model/provider) must resolve
            # against this org, not fall back to local/global.
            with use_settings_scope(scope):
                outcome = await self._run_anton(
                    scope, conversation, event, adapter, channel_context=channel_context
                )
        except (PoolTimeoutError, FilesystemBusy):
            await self._send_busy_notice(channel_type=channel_type, event=event, org_id=org_id)
            raise
        finally:
            if typing is not None:
                typing.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing
        reply = outcome.reply
        log.info(
            "channel %s: turn complete (reply=%d chars, used_tools=%s)",
            channel_type, len(reply or ""), outcome.used_tools,
        )
        # Outside the refusal's try above: a refusal after the reply must not
        # send the notice as a second message.
        if reply and reply.strip():
            # The link is a channel affordance only; the stored assistant
            # message stays canonical (UI users are already in the conversation).
            outbound = reply
            if outcome.used_tools:
                link = conversation_link(conversation.id)
                if link:
                    outbound = f"{reply}\n\n{link}"
            await self._deliver(channel_type, event, outbound, org_id)
        if outcome.artifact_paths:
            await self.send_turn_artifacts(
                adapter=adapter, event=event, conversation=conversation, paths=outcome.artifact_paths, scope=scope,
            )

    async def _send_busy_notice(self, *, channel_type: str, event: Any, org_id: str | None) -> None:
        """Tell the sender a busy server refused their message. A notice that
        fails to send is logged, and the refusal still propagates."""
        try:
            await self._deliver(channel_type, event, BUSY_NOTICE, org_id)
        except Exception:
            log.warning("channel turn: could not deliver the busy notice", exc_info=True)

    async def _start_fresh(
        self, scope: TenantScope, channel_type: str, binding: ChannelBinding, event: Any,
        org_id: str | None,
    ) -> None:
        """Handle /new: detach the pinned conversation and confirm deterministically instead of running a turn."""
        name = await run_db(partial(self._detach_conversation, binding=binding), scope=scope)
        log.info("channel %s: /new detached conversation for binding %s", channel_type, binding.id)
        await self._deliver(
            channel_type, event,
            f'Starting fresh — your next message begins a new conversation in the "{name}" project.',
            org_id,
        )

    def _detach_conversation(self, scoped: ScopedSession, *, binding: ChannelBinding) -> str:
        """Unpin the binding's conversation, as a unit, and name the project
        the next message starts in."""
        ChannelBindingService(scoped).detach_conversation(binding)
        project = scoped.get(Project, binding.anton_project_id or self._resolve_default_project_id(scoped))
        return project.name if project else "general"

    @staticmethod
    def _admit_sender(scoped: ScopedSession, *, channel_type: str, event: Any) -> Admission:
        """Whether the event's sender may run a turn in its chat, as a unit.

        A message with no sender id is refused. A sender neither allowed nor
        blocked whose message is a valid pairing code is allowed and the
        message runs no turn. Otherwise a DM, or a group message that mentions
        the bot, records a pending request for the owner; other group chatter
        records nothing.
        """
        message = event.message
        sender_id = message.sender_id
        if not sender_id:
            return Admission.HOLD
        service = ChannelSenderService(scoped)
        group_id = event.address.platform_id
        status = service.status_of(channel_type, group_id, sender_id)
        if status == SENDER_ALLOWED:
            return Admission.ADMIT
        if status == SENDER_BLOCKED:
            return Admission.HOLD
        code = code_in(str(message.content))
        if code is not None and ChannelPairingService(scoped).redeem(
            channel_type=channel_type,
            code=code,
            group_id=group_id,
            sender_id=sender_id,
            sender_name=message.sender_name,
            is_group=bool(message.is_group),
        ):
            return Admission.PAIRED
        if message.is_group and not message.is_mention:
            return Admission.HOLD
        service.record_request(
            channel_type=channel_type,
            group_id=group_id,
            sender_id=sender_id,
            sender_name=message.sender_name,
            is_group=bool(message.is_group),
        )
        return Admission.HOLD

    @staticmethod
    def _find_binding(scoped: ScopedSession, *, channel_type: str, event: Any) -> ChannelBinding | None:
        """The binding of the event's thread, when it has one."""
        return scoped.exec(
            scoped.select(ChannelBinding).where(
                ChannelBinding.channel_type == channel_type,
                ChannelBinding.external_group_id == event.address.platform_id,
                ChannelBinding.external_thread_key == (event.address.thread_id or _DEFAULT_THREAD_KEY),
            )
        ).first()

    def _resolve_or_create_binding(self, scoped: ScopedSession, channel_type: str, event: Any) -> ChannelBinding:
        binding = self._find_binding(scoped, channel_type=channel_type, event=event)
        if binding is not None:
            return binding
        thread_id = event.address.thread_id
        binding = ChannelBinding(
            channel_type=channel_type,
            external_group_id=event.address.platform_id,
            external_thread_id=thread_id,
            external_thread_key=thread_id or _DEFAULT_THREAD_KEY,
            anton_project_id=self._resolve_default_project_id(scoped),
            trigger_rule="mention_only" if event.message.is_group else "always",
        )
        scoped.add(binding)
        scoped.commit()
        scoped.refresh(binding)
        return binding

    @staticmethod
    async def _should_respond(binding: ChannelBinding, event: Any) -> bool:
        rule = binding.trigger_rule
        if rule == "always":
            return True
        if rule == "mention_only":
            return bool(event.message.is_mention)
        if rule == "regex":
            pattern = binding.trigger_pattern
            if not pattern:
                return False
            content = str(event.message.content)
            # `trigger_pattern` is org-controlled (ChannelBindingService only
            # checks it parses, not that it's cheap to run) and is matched
            # against every inbound message, so a backtracking engine here
            # would be a ReDoS. re2 guarantees linear-time matching — no
            # pattern can make this hang — so no timeout is needed; still
            # off the event loop so a very large message can't stall other
            # channels/requests while it matches.
            try:
                return await asyncio.to_thread(
                    lambda: re2.search(pattern, content, options=_QUIET_REGEX_OPTIONS) is not None
                )
            except re2.error:
                return False
        return True

    def _ensure_conversation(
        self, scoped: ScopedSession, binding: ChannelBinding, *, shared: bool | None
    ) -> Conversation:
        """The route's conversation, created on its first message if it has none.

        Only a conversation from a shared channel (`shared` True) is marked with
        its channel type, which is what makes it readable to org admins; a
        direct message to the bot stays unmarked and so readable by nobody in
        the app, and so does a first message the adapter could not place (None).
        In org mode an existing conversation is reused only if this runtime
        could have made it for this message (no owner, marked to match) and no
        other route points at it. Anything else (a legacy pin to someone's chat,
        a DM route pointed at a shared conversation, two routes on one
        conversation) gets a fresh conversation instead.
        """
        if binding.anton_conversation_id is not None:
            existing = scoped.get(Conversation, binding.anton_conversation_id)
            if existing is not None and (
                not scoped.scope.org_mode or _reusable_in_org(scoped, binding, existing, shared)
            ):
                return existing
        topic = f"{binding.channel_type}: {binding.display_name or binding.external_group_id}"[:80]
        conversation = ConversationService(scoped).create_conversation(
            topic=topic,
            project_id=binding.anton_project_id or self._resolve_default_project_id(scoped),
            channel_type=binding.channel_type if shared is True else None,
        )
        binding.anton_conversation_id = conversation.id
        scoped.add(binding)
        scoped.commit()
        return conversation

    def _open_conversation(
        self, scoped: ScopedSession, *, binding: ChannelBinding, event: Any
    ) -> Conversation:
        """The turn's conversation, as a unit: found or created for the
        binding, its channel session touched, and its project loaded, so the
        harness can read both after the unit's session closes."""
        conversation = self._ensure_conversation(scoped, binding, shared=shared_state(event))
        self._touch_channel_session(scoped, binding, conversation, event)
        _ = conversation.project
        return conversation

    @staticmethod
    def _touch_channel_session(
        scoped: ScopedSession, binding: ChannelBinding, conversation: Conversation, event: Any
    ) -> None:
        # ChannelSession is a child (no org_id): anchored by its binding.
        key = event.address.thread_id or _DEFAULT_THREAD_KEY
        row = scoped.exec(
            scoped.select(ChannelSession).where(
                ChannelSession.binding_id == binding.id,
                ChannelSession.external_session_key == key,
            )
        ).first()
        now = datetime.now(timezone.utc)
        if row is None:
            row = ChannelSession(
                binding_id=binding.id,
                external_session_key=key,
                anton_session_id=str(conversation.id),
                last_message_at=now,
            )
        else:
            # Always the conversation just served: _ensure_conversation may have
            # replaced the route's, and its reuse rule reads this row.
            row.anton_session_id = str(conversation.id)
            row.last_message_at = now
        scoped.add(row)
        scoped.commit()

    def _read_turn_setup(self, scoped: ScopedSession, *, conversation: Conversation) -> _TurnSetup:
        """The turn's settings and its harness, read as one unit.

        The harness is the one pinned to this conversation (whatever first
        served it), else the configured channel agent. That is the persisted
        ``channels_harness`` setting (UI-selectable, env-seeded), never the
        desktop UI harness."""
        settings = unit_setting_service(scoped).load()
        # Messages are children (no org_id): anchored on the caller's conversation.
        pinned = scoped.exec(
            scoped.select(DBMessage.harness).where(
                DBMessage.conversation_id == conversation.id,
                DBMessage.role == "assistant",
                DBMessage.harness != None,  # noqa: E711
            ).limit(1)
        ).first()
        harness_id = pinned or (settings.channels_harness or "").strip() or DEFAULT_CHANNEL_HARNESS
        return _TurnSetup(harness_id=harness_id, settings=settings)

    async def _turn_stream(
        self, harness: HarnessProvider, harness_id: str, scope: TenantScope,
        conversation: Conversation, blocks: list[dict], text: str,
        channel_context: ChannelContext | None, turn_rows: list,
    ):
        """The one place a channel turn picks in-process vs remote-worker
        execution; the rest of _run_anton is identical either way."""
        if not TurnQueueSettings().is_remote:
            return harness.stream_response(
                conversation=conversation, input=blocks, channel_context=channel_context,
                trace_metadata=build_trace_metadata(),
            )
        await stage_remote_workspace(conv_id=conversation.id, scope=scope)
        # Known gap: channel_context (group/DM framing, per-channel instructions) has no
        # remote job field yet, so it's dropped here — unused on this path in org mode.
        return remote_turn_events(
            scope=scope,
            conv_id=conversation.id,
            org_id=scope.org_id,
            user_id=None,  # channel turns are org-scoped, not per-member
            input_text=text,
            model=None,  # deployment default, same as browser turns with no override
            turn_rows=turn_rows,
        )

    async def _run_anton(
        self, scope: TenantScope, conversation: Conversation, event: Any, adapter: Any = None,
        *, channel_context: ChannelContext | None = None,
    ) -> TurnOutcome:
        """Run one channel turn; returns its reply, whether tools ran, and the
        paths of the artifacts it created or touched.

        The turn's first unit loads its settings, and the rest of the turn
        runs with them bound (use_turn_settings) on the event loop, as a
        browser turn's handle() binds its own. Every later settings read, here,
        in the worker threads of the turn's units and in the remote turn's
        stream (its seed read and its memory save among them), is served from
        that snapshot, so no unit asks the pool for a second connection while
        it holds one. The binding sits here rather than in remote_turn_events:
        a context variable an async generator sets across its yields is reset
        in the wrong context when the generator is finalized from another
        task, which raises."""
        setup = await run_db(partial(self._read_turn_setup, conversation=conversation), scope=scope)
        with use_turn_settings(scope, setup.settings):
            harness_id = setup.harness_id
            try:
                harness = get_harness(harness_id)
            except ValueError:
                log.warning("harness %r is not registered; falling back to %s", harness_id, DEFAULT_CHANNEL_HARNESS)
                harness_id = DEFAULT_CHANNEL_HARNESS
                harness = get_harness(harness_id)

            text = self._event_text(event)
            # In a group, several people share one conversation — prefix each
            # message with the sender's name so the model (and the stored history)
            # can tell who said what. Applied after trigger gating, so regex rules
            # keep matching the raw text.
            sender_name = getattr(event.message, "sender_name", None)
            if text and event.message.is_group and sender_name:
                text = f"{sender_name}: {text}"
            blocks = await self.build_input_blocks(scope, adapter, event, text)

            names = [a.filename for a in (event.message.attachments or [])]
            content = text or (f"[attachments: {', '.join(names)}]" if names else "")
            # Send time captured before the turn
            sent_at = datetime.now(timezone.utc)

            collected: list[str] = []
            events: list[dict] = []
            turn_rows: list[dict] = []
            artifact_paths: list[str] = []

            def event_sink(event_type: str, data: dict) -> None:
                # Tool block-rows are for LLM-history persistence, not UI replay —
                # keep them out of the events log (mirrors handlers/responses.py).
                if event_type == "response.turn_history":
                    # Id-checked even in-process; see handlers/responses.py (ENG-2420).
                    turn_rows[:] = reject_unreplayable_tool_rows(data.get("rows") or [])
                    return
                if event_type == "response.artifact_created":
                    card = data.get("artifact")
                    if isinstance(card, dict) and isinstance(card.get("path"), str):
                        artifact_paths.append(card["path"])
                events.append(data)
                accumulate_answer_text(collected, event_type, data)

            stream = await self._turn_stream(
                harness, harness_id, scope, conversation, blocks, text, channel_context, turn_rows,
            )
            remote_failure: RemoteTurnFailed | None = None
            try:
                async for _chunk in harness.formatter(stream, harness_id, event_sink):
                    pass
            except RemoteTurnFailed as exc:
                remote_failure = exc
            finally:
                # Persist the user message only after the harness has read this
                # turn's history (it reads via a fresh query). Persisting earlier
                # would replay the message into this turn AND resend it as the live
                # input. In `finally` so a crashed turn still records the inbound
                # message, matching the pre-history-replay behaviour.
                await self._save_turn_row(
                    conversation_id=conversation.id,
                    save=lambda session: ConversationService(session).save_user_message(
                        conversation.id, content, created_at=sent_at
                    ),
                    scope=scope,
                    what="question",
                )

            reply = remote_failure.message if remote_failure is not None else "".join(collected)
            await self._save_turn_row(
                conversation_id=conversation.id,
                save=lambda session: ConversationService(session).save_assistant_turn(
                    conversation.id, reply, events, harness=harness_id, tool_rows=turn_rows,
                ),
                scope=scope,
                what="answer",
            )
        return TurnOutcome(reply=reply, used_tools=turn_used_tools(events), artifact_paths=tuple(artifact_paths))

    @staticmethod
    async def _save_turn_row(
        *, conversation_id: UUID, save: Callable[[ScopedSession], object], scope: TenantScope, what: str,
    ) -> None:
        """One end-of-turn save, in a unit. A busy pool is retried; a save
        still refused after _SAVE_TRIES is logged, and the turn goes on, so
        the reply the agent produced is delivered anyway. No HTTP caller waits
        on a channel turn, so the retries need not end within POOL_TIMEOUT.

        Each try holds the conversation's write lock, as every message writer
        in this process does: a turn sent from the UI into this conversation
        saves from a worker thread, and two writers numbering rows at once
        would give them the same seq. Nothing holds it during the backoff."""
        for attempt in range(1, _SAVE_TRIES + 1):
            try:
                async with conversation_writes(conversation_id):
                    await run_db(save, scope=scope)
                return
            except PoolTimeoutError:
                if attempt == _SAVE_TRIES:
                    log.warning(
                        "channel turn: no database connection freed to save a message; the conversation lacks it",
                        extra={
                            **log_context(conversation_id=conversation_id),
                            "step": f"save_{what}",
                            "attempts": _SAVE_TRIES,
                        },
                    )
                    return
                await asyncio.sleep(_SAVE_BACKOFF_S * attempt)

    @staticmethod
    def _event_text(event: Any) -> str:
        content = event.message.content
        return content if isinstance(content, str) else str(content)

    async def build_input_blocks(self, scope: TenantScope, adapter: Any, event: Any, text: str) -> list[dict]:
        """Harness input from the inbound event: stored media become image/file
        blocks (same shapes the responses handler builds), text rides last.

        Known gap: these blocks aren't staged for the remote worker in org mode
        (only browser-uploaded attachments are), so channel media is silently unavailable to the model there."""
        blocks: list[dict] = []
        fetch = getattr(adapter, "fetch_attachment", None) if adapter is not None else None
        for attachment in (event.message.attachments or []):
            data = attachment.data
            if data is None and callable(fetch):
                data = await fetch(attachment)
            if not data:
                continue
            # The filename comes from an external platform, so keep only its basename.
            stored = await store_file(
                scope=scope,
                filename=upload_basename(name=attachment.filename) or "file",
                content_type=attachment.mime_type,
                data=data,
                purpose="channel",
            )
            if (attachment.mime_type or "").startswith("image/"):
                blocks.append({"type": "image", "source": {
                    "type": "base64",
                    "media_type": attachment.mime_type,
                    "data": base64.standard_b64encode(data).decode("ascii"),
                }})
            else:
                blocks.append({"type": "file", "path": stored.path, "filename": stored.filename})
        if text or not blocks:
            blocks.append({"type": "text", "text": text})
        return blocks

    async def send_turn_artifacts(
        self, *, adapter: Any, event: Any, conversation: Conversation, paths: Sequence[str], scope: TenantScope,
    ) -> None:
        """Send the files of the artifacts the turn created or touched, the
        paths its own cards name, through the optional send_attachment hook:
        never another conversation's, though the project's artifacts base is
        shared. Best-effort per file; channels without the hook are untouched.
        ``conversation`` comes with its project loaded (_open_conversation).

        A file past the adapter's upload cap (its ``max_attachment_bytes``,
        else MAX_TURN_ATTACHMENT_BYTES) is skipped before it is read, since
        the platform would refuse it anyway. The files are read in one call
        on the file store's threads. The reply is already delivered, so a
        file store that frees no slot in time sends no files, with a WARNING,
        and the turn still ends well."""
        sender = getattr(adapter, "send_attachment", None) if adapter is not None else None
        if not callable(sender):
            return
        project = conversation.project
        if project is None:
            return
        max_bytes = getattr(adapter, "max_attachment_bytes", MAX_TURN_ATTACHMENT_BYTES)
        try:
            files = await run_efs(
                partial(read_turn_files, project_path=project.path, paths=paths, max_bytes=max_bytes), scope=scope,
            )
        except FilesystemBusy:
            log.warning(
                "channel turn: no file store slot freed in time to send the turn's files",
                extra=log_context(conversation_id=conversation.id),
            )
            return
        for file in files:
            try:
                await sender(address=event.address, data=file.data, filename=file.filename)
            except Exception:
                log.warning("channel %s: failed sending artifact %s", event.address.channel_type, file.filename)

    async def _deliver(self, channel_type: str, event: Any, reply: str, org_id: str | None = None) -> None:
        adapter = self._adapters.get(channel_type, org_id)
        if adapter is None:
            log.warning("channel %s: no live adapter; reply not delivered", channel_type)
            return
        await adapter.deliver(OutboundMessage(address=event.address, text=reply))
        log.info("channel %s: delivered reply to %s", channel_type, event.address.platform_id)
