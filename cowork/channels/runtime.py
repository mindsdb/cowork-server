from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from anton.core.dispatch import OutboundMessage
from cowork.build_info import build_trace_metadata
from cowork.channels.registry import PluginRegistry, get_registry
from cowork.db.scoped import LOCAL_SCOPE, SYSTEM_SCOPE, ScopedSession, TenantScope, scope_for_background_context
from cowork.db.session import get_open_session
from cowork.handlers.responses import ResponsesHandler
from cowork.harnesses.base import ChannelContext, HarnessProvider, get_harness
from cowork.models.channel import ChannelBinding, ChannelSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message as DBMessage
from cowork.models.project import Project
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.common.settings.user_settings import get_user_settings, use_settings_scope
from cowork.services.artifact_roots import conversation_artifacts_base
from cowork.services.artifacts import ProjectArtifacts, list_artifacts
from cowork.services.channel_bindings import ChannelBindingService
from cowork.services.channels import ChannelConfigService, resolve_installation_by_external_account
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService
from cowork.services.skills import SkillService
from cowork.turnqueue.remote_turn import RemoteTurnFailed, remote_turn_events

log = logging.getLogger(__name__)

# Channel turns run the server-configured channels_harness (default anton),
# pinned per conversation via per-message harness identity. The UI harness
# hotswitch (UserSettings.harness) never applies to channels.
DEFAULT_CHANNEL_HARNESS = "anton"
_DEFAULT_THREAD_KEY = "__default__"


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


def artifacts_since(project_path: str, conversation_id, since: float) -> list[tuple[str, str]]:
    """(path, filename) of artifact primaries created/updated after ``since``
    in this project. Time-window based: concurrent turns in the same project
    could cross-attribute — acceptable for the single-operator v1."""
    out: list[tuple[str, str]] = []
    # This runs for one known project, so the root is built directly rather than
    # resolved — the channel already holds the project it is answering for.
    source = ProjectArtifacts(
        base=conversation_artifacts_base(project_path, conversation_id),
        project_id=None,
        project_name=Path(project_path).name,
    )
    for card in list_artifacts([source]):
        folder = Path(card.get("folder") or "")
        try:
            if (folder / "metadata.json").stat().st_mtime < since:
                continue
        except OSError:
            continue
        primary = Path(card.get("path") or "")
        if primary.is_file():
            out.append((str(primary), primary.name))
    return out


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

    async def get_or_refresh(
        self, channel_type: str, org_id: str | None, *, session: ScopedSession | None = None
    ) -> Any | None:
        """Cache hit → return it, no session touched. Miss (first webhook for
        an org this replica hasn't loaded yet) → refresh, then return the
        result either way. Used by the webhook path, which has no prior
        chance to have called refresh for an org it just resolved."""
        cached = self.get(channel_type, org_id)
        if cached is not None:
            return cached
        await self.refresh(channel_type, org_id, session=session)
        return self.get(channel_type, org_id)

    async def resolve_org_bridge(
        self, channel_type: str, routing_key: str, *, session: Any | None = None
    ) -> tuple[Any, str | None] | None:
        """What the webhook path's org_resolver actually calls: unverified
        routing key -> which installation claims it -> that installation's own
        bridge. None means nothing claims this key. session is a raw Session,
        for tests — production always opens its own."""
        own_session = session is None
        raw = session or get_open_session()
        try:
            install = resolve_installation_by_external_account(raw, channel_type, routing_key)
            if install is None:
                return None
            scope = TenantScope(org_mode=True, org_id=install.org_id) if install.org_id else LOCAL_SCOPE
            bridge = await self.get_or_refresh(
                channel_type, install.org_id, session=ScopedSession(raw, scope)
            )
            return (bridge, install.org_id) if bridge is not None else None
        finally:
            if own_session:
                raw.close()

    async def refresh(
        self, channel_type: str, org_id: str | None = None, *, session: ScopedSession | None = None
    ) -> bool:

        plugin = self._registry.get(channel_type)
        if plugin is None:
            self._cache.pop((channel_type, org_id), None)
            return False
        own_session = session is None
        # No caller-supplied session: SYSTEM_SCOPE for local mode's one
        # installation, a real org scope otherwise — never the other way.
        scope = SYSTEM_SCOPE if org_id is None else TenantScope(org_mode=True, org_id=org_id)
        s = session or ScopedSession(get_open_session(), scope)
        try:
            creds = ChannelConfigService(s, registry=self._registry).load_credentials(channel_type)
        finally:
            if own_session:
                s.close()
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
        session = get_open_session()
        try:
            # One scope per turn. org_id comes from the webhook's own org
            # resolution (resolve_bridge) — None means local mode, or (in org
            # mode) a genuinely unresolved org, which still fails closed below.
            scope = TenantScope(org_mode=True, org_id=org_id) if org_id else scope_for_background_context()
            scoped = ScopedSession(session, scope)
            binding = self._resolve_or_create_binding(scoped, channel_type, event)
            log.info(
                "channel %s: binding %s → project %s (trigger=%s)",
                channel_type, binding.id, binding.anton_project_id, binding.trigger_rule,
            )
            if not self._should_respond(binding, event):
                log.info("channel %s: trigger rule %r skipped a message", channel_type, binding.trigger_rule)
                return
            if is_new_command(self._event_text(event), is_mention=event.message.is_mention):
                await self._start_fresh(scoped, channel_type, binding, event, org_id)
                return
            # Optional hook: adapters with set_typing show a typing indicator
            # for the duration of the turn; others are untouched.
            adapter = self._adapters.get(channel_type, org_id)
            typing = None
            if adapter is not None and callable(getattr(adapter, "set_typing", None)):
                typing = asyncio.create_task(typing_loop(adapter, event.address))
            # 1s slack for filesystem timestamp granularity.
            turn_started = time.time() - 1
            try:
                conversation = self._ensure_conversation(scoped, binding)
                self._touch_channel_session(scoped, binding, conversation, event)
                channel_context = ChannelContext(
                    channel_type=channel_type,
                    is_group=bool(event.message.is_group),
                    display_name=binding.display_name,
                    instructions=binding.instructions,
                )
                # Nested get_user_settings() reads (model/provider, channels_harness)
                # must resolve against this org, not fall back to local/global.
                with use_settings_scope(scope):
                    reply, used_tools = await self._run_anton(
                        scoped, conversation, event, adapter, channel_context=channel_context
                    )
            finally:
                if typing is not None:
                    typing.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await typing
            log.info(
                "channel %s: turn complete (reply=%d chars, used_tools=%s)",
                channel_type, len(reply or ""), used_tools,
            )
            if reply and reply.strip():
                # The link is a channel affordance only; the stored assistant
                # message stays canonical (UI users are already in the conversation).
                outbound = reply
                if used_tools:
                    link = conversation_link(conversation.id)
                    if link:
                        outbound = f"{reply}\n\n{link}"
                await self._deliver(channel_type, event, outbound, org_id)
            if used_tools:
                await self.send_turn_artifacts(adapter, event, conversation, turn_started)
        finally:
            session.close()


    async def _start_fresh(
        self, scoped: ScopedSession, channel_type: str, binding: ChannelBinding, event: Any,
        org_id: str | None,
    ) -> None:
        """Handle /new: detach the pinned conversation and confirm deterministically instead of running a turn."""
        ChannelBindingService(scoped).detach_conversation(binding)
        project = scoped.get(Project, binding.anton_project_id or self._resolve_default_project_id(scoped))
        name = project.name if project else "general"
        log.info("channel %s: /new detached conversation for binding %s", channel_type, binding.id)
        await self._deliver(
            channel_type, event,
            f'Starting fresh — your next message begins a new conversation in the "{name}" project.',
            org_id,
        )

    def _resolve_or_create_binding(self, scoped: ScopedSession, channel_type: str, event: Any) -> ChannelBinding:
        group_id = event.address.platform_id
        thread_id = event.address.thread_id
        thread_key = thread_id or _DEFAULT_THREAD_KEY
        binding = scoped.exec(
            scoped.select(ChannelBinding).where(
                ChannelBinding.channel_type == channel_type,
                ChannelBinding.external_group_id == group_id,
                ChannelBinding.external_thread_key == thread_key,
            )
        ).first()
        if binding is not None:
            return binding
        binding = ChannelBinding(
            channel_type=channel_type,
            external_group_id=group_id,
            external_thread_id=thread_id,
            external_thread_key=thread_key,
            anton_project_id=self._resolve_default_project_id(scoped),
            trigger_rule="mention_only" if event.message.is_group else "always",
        )
        scoped.add(binding)
        scoped.commit()
        scoped.refresh(binding)
        return binding

    @staticmethod
    def _should_respond(binding: ChannelBinding, event: Any) -> bool:
        rule = binding.trigger_rule
        if rule == "always":
            return True
        if rule == "mention_only":
            return bool(event.message.is_mention)
        if rule == "regex":
            pattern = binding.trigger_pattern
            if not pattern:
                return False
            try:
                return re.search(pattern, str(event.message.content)) is not None
            except re.error:
                return False
        return True

    def _ensure_conversation(self, scoped: ScopedSession, binding: ChannelBinding) -> Conversation:
        if binding.anton_conversation_id is not None:
            existing = scoped.get(Conversation, binding.anton_conversation_id)
            if existing is not None:
                return existing
        topic = f"{binding.channel_type}: {binding.display_name or binding.external_group_id}"[:80]
        conversation = ConversationService(scoped).create_conversation(
            topic=topic,
            project_id=binding.anton_project_id or self._resolve_default_project_id(scoped),
        )
        binding.anton_conversation_id = conversation.id
        scoped.add(binding)
        scoped.commit()
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
            row.last_message_at = now
        scoped.add(row)
        scoped.commit()

    def resolve_turn_harness(self, scoped: ScopedSession, conversation: Conversation) -> str:
        """Pinned harness for this conversation (whatever first served it), else
        the configured channel agent. This is the persisted ``channels_harness``
        setting (UI-selectable, env-seeded) — never the desktop UI harness."""
        # Messages are children (no org_id): anchored on the caller's conversation.
        pinned = scoped.exec(
            scoped.select(DBMessage.harness).where(
                DBMessage.conversation_id == conversation.id,
                DBMessage.role == "assistant",
                DBMessage.harness != None,  # noqa: E711
            ).limit(1)
        ).first()
        if pinned:
            return pinned
        return (get_user_settings().channels_harness or "").strip() or DEFAULT_CHANNEL_HARNESS

    async def _turn_stream(
        self, harness: HarnessProvider, harness_id: str, scoped: ScopedSession,
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
        await asyncio.to_thread(
            ResponsesHandler._stage_remote_workspace_files, scoped, conversation.id,
        )
        # Known gap: channel_context (group/DM framing, per-channel instructions) has no
        # remote job field yet, so it's dropped here — unused on this path in org mode.
        return remote_turn_events(
            session=scoped,
            conv_id=conversation.id,
            org_id=scoped.scope.org_id,
            user_id=None,  # channel turns are org-scoped, not per-member
            input_text=text,
            model=None,  # deployment default, same as browser turns with no override
            turn_rows=turn_rows,
        )

    async def _run_anton(
        self, scoped: ScopedSession, conversation: Conversation, event: Any, adapter: Any = None,
        *, channel_context: ChannelContext | None = None,
    ) -> tuple[str, bool]:
        """Run one channel turn; returns the reply text and whether tools ran."""
        harness_id = self.resolve_turn_harness(scoped, conversation)
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
        blocks = await self.build_input_blocks(scoped, adapter, event, text)

        _ = conversation.messages
        names = [a.filename for a in (event.message.attachments or [])]
        content = text or (f"[attachments: {', '.join(names)}]" if names else "")
        # Send time captured before the turn
        sent_at = datetime.now(timezone.utc)

        collected: list[str] = []
        events: list[dict] = []
        turn_rows: list[dict] = []

        def event_sink(event_type: str, data: dict) -> None:
            # Tool block-rows are for LLM-history persistence, not UI replay —
            # keep them out of the events log (mirrors handlers/responses.py).
            if event_type == "response.turn_history":
                turn_rows[:] = data.get("rows") or []
                return
            events.append(data)
            if event_type == "response.output_text.delta":
                collected.append(data.get("delta", ""))

        stream = await self._turn_stream(
            harness, harness_id, scoped, conversation, blocks, text, channel_context, turn_rows,
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
            ConversationService(scoped).save_user_message(
                conversation.id, content, created_at=sent_at
            )

        reply = remote_failure.message if remote_failure is not None else "".join(collected)
        ConversationService(scoped).save_assistant_turn(
            conversation.id, reply, events, harness=harness_id, tool_rows=turn_rows,
        )
        return reply, turn_used_tools(events)

    @staticmethod
    def _event_text(event: Any) -> str:
        content = event.message.content
        return content if isinstance(content, str) else str(content)

    async def build_input_blocks(self, scoped: ScopedSession, adapter: Any, event: Any, text: str) -> list[dict]:
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
            stored = FileService(scoped).create_file_from_bytes(
                filename=attachment.filename,
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

    async def send_turn_artifacts(self, adapter: Any, event: Any, conversation: Conversation, since: float) -> None:
        """Send files the turn produced through the optional send_attachment
        hook. Best-effort per file; channels without the hook are untouched."""
        sender = getattr(adapter, "send_attachment", None) if adapter is not None else None
        if not callable(sender):
            return
        project = conversation.project
        if project is None:
            return
        for path, filename in artifacts_since(project.path, conversation.id, since)[:MAX_TURN_ATTACHMENTS]:
            try:
                await sender(address=event.address, path=path, filename=filename)
            except Exception:
                log.warning("channel %s: failed sending artifact %s", event.address.channel_type, filename)

    async def _deliver(self, channel_type: str, event: Any, reply: str, org_id: str | None = None) -> None:
        adapter = self._adapters.get(channel_type, org_id)
        if adapter is None:
            log.warning("channel %s: no live adapter; reply not delivered", channel_type)
            return
        await adapter.deliver(OutboundMessage(address=event.address, text=reply))
        log.info("channel %s: delivered reply to %s", channel_type, event.address.platform_id)
