"""Quick edits from the artifact viewer's chat bar (ENG-4227): the server side
that is not HTTP.

The viewer's bar sends a change. Jev routes it (anton's ``route_request``):
a change that needs only the artifact's own files runs anton's artifact editor
right here (``run_quick_edit``), with no chat turn; anything else goes to the
artifact's chat as a normal agent turn, which the app sends itself.

This module holds what the route needs around that:

* ``anton_quick_edit()``: the installed anton's router and quick editor, or
  None. Imported lazily and feature-detected, so the server keeps working with
  an anton that has neither; the route then answers 501 and the capability
  ``canQuickEdit`` is false.
* ``quick_edit_available()``: the setting and the anton support together.
* ``QuickEditLocks``: one quick edit per artifact at a time in this process.
* ``resolve_backend()``: the editor's LLM and the router's Jev endpoint for
  the caller, from the same settings and keys a turn of theirs would use.
* ``record_quick_edit_in_conversation()`` / ``record_undo_in_conversation()``:
  the artifact's conversation keeps ONE message per run of quick edits, so
  the agent's later turns know what changed and the chat shows one card.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import ModuleType
from typing import Any
from uuid import UUID, uuid4

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope

logger = logging.getLogger(__name__)

__all__ = [
    "AntonQuickEdit",
    "QUICK_EDIT_HARNESS",
    "QuickEditBackend",
    "QuickEditLocks",
    "anton_quick_edit",
    "Decisions",
    "decisions_for",
    "decisions_for_scope",
    "direct_edit_available",
    "quick_edit_available",
    "quick_edit_locks",
    "QUICK_EDIT_GROUP_EVENT",
    "group_text",
    "quick_edit_entry",
    "record_quick_edit_in_conversation",
    "record_undo_in_conversation",
    "resolve_backend",
]

#: The ``harness`` the conversation note's messages are saved under, as a
#: direct answer's are saved under ``cowork-direct``.
QUICK_EDIT_HARNESS = "cowork-quick-edit"


@dataclass(frozen=True)
class AntonQuickEdit:
    """The installed anton's quick-edit API (anton ``artifact_editor``)."""

    route_request: Callable[..., Any]
    run_quick_edit: Callable[..., Any]
    # anton.core.artifact_editor.checkers, for the editor's checker chain.
    checkers: ModuleType | None
    # anton.core.artifact_editor.workspace.ArtifactWorkspace, for the file
    # list the router reads. None: the router gets no file list.
    workspace_cls: type | None
    # router.describe_artifact(folder) -> {"name", "description", "data"},
    # optional context for the router; None when this anton has none.
    describe_artifact: Callable[..., Any] | None = None
    # The optional keywords this anton's route_request takes beyond the
    # contract's (name, description, data, history).
    route_extras: frozenset[str] = frozenset()
    # router.decisions_available(settings) / decisions_config(settings):
    # whether a decisions model serves the person's configured provider, and
    # which. None when this anton has neither: decisions are then unavailable.
    decisions_available: Callable[..., Any] | None = None
    decisions_config: Callable[..., Any] | None = None


def anton_quick_edit() -> AntonQuickEdit | None:
    """The installed anton's router and quick editor, or None when it has
    neither. Never raises: an anton that fails to import them is one without."""
    try:
        # The dotted import, not `from package import quick`: it consults
        # sys.modules, so a module that is absent (or hidden) reads as absent
        # even after the package was imported.
        import anton.core.artifact_editor.quick
        import anton.core.artifact_editor.router

        quick = sys.modules["anton.core.artifact_editor.quick"]
        router = sys.modules["anton.core.artifact_editor.router"]
        if quick is None or router is None:
            return None
    except Exception:
        return None
    route_request = getattr(router, "route_request", None)
    run_quick_edit = getattr(quick, "run_quick_edit", None)
    if not callable(route_request) or not callable(run_quick_edit):
        return None
    try:
        from anton.core.artifact_editor import checkers
    except Exception:
        checkers = None
    try:
        from anton.core.artifact_editor.workspace import ArtifactWorkspace
    except Exception:
        ArtifactWorkspace = None
    describe = getattr(router, "describe_artifact", None)
    try:
        import inspect

        params = set(inspect.signature(route_request).parameters)
    except (TypeError, ValueError):
        params = set()
    return AntonQuickEdit(
        route_request=route_request,
        run_quick_edit=run_quick_edit,
        checkers=checkers,
        workspace_cls=ArtifactWorkspace,
        describe_artifact=describe if callable(describe) else None,
        route_extras=frozenset(params & ROUTE_EXTRAS),
        decisions_available=_callable(router, "decisions_available"),
        decisions_config=_callable(router, "decisions_config"),
    )


#: route_request's optional context, passed only when this anton takes it.
ROUTE_EXTRAS = frozenset({"name", "description", "data", "outline", "history", "recent_changes", "chat"})
#: From describe_artifact(folder).
DESCRIBED_EXTRAS = ("name", "description", "data", "outline")
#: The artifact's latest revision summaries the router reads.
MAX_RECENT_CHANGES = 6
#: The conversation's latest text messages the router reads, and their size.
MAX_CHAT = 4
MAX_CHAT_TEXT = 300


def _message_text(content: object) -> str:
    """A saved message's visible text: a string, or its text blocks. Tool
    blocks give nothing, so a tool row reads as empty."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text")
            and isinstance(b.get("text"), str)
        )
    return ""


def router_chat(session: ScopedSession, *, conversation_id: UUID) -> list[dict]:
    """The conversation's last MAX_CHAT text messages, oldest first, as
    ``[{role, text}]`` clipped to MAX_CHAT_TEXT: what the person and the
    agent said just before the bar message. Tool rows and quick-edit group
    records are skipped. Empty for a conversation the caller does not own."""
    from cowork.services.conversations import ConversationService

    try:
        rows = ConversationService(session).get_recent_messages(
            conversation_id, roles=("user", "assistant"), limit=MAX_CHAT * 6,
        )
    except ValueError:
        return []
    chat: list[dict] = []
    for message in rows:
        if message.harness == QUICK_EDIT_HARNESS:
            continue
        text = " ".join(_message_text(message.content).split())
        if text:
            role = getattr(message.role, "value", message.role)
            chat.append({"role": str(role), "text": text[:MAX_CHAT_TEXT]})
    return chat[-MAX_CHAT:]


def _callable(module: object, name: str) -> Callable[..., Any] | None:
    value = getattr(module, name, None)
    return value if callable(value) else None


def quick_edit_available() -> bool:
    """Whether this server can run a quick edit at all: the setting is on and
    the installed anton has the quick editor."""
    return bool(get_app_settings().quick_edit_enabled) and anton_quick_edit() is not None


# ── one quick edit per artifact ──────────────────────────────────────────

#: A lock older than this is presumed left behind (a response whose body was
#: never iterated never runs the release). Well past the edit's own timeout.
LOCK_STALE_AFTER_S = 15 * 60


@dataclass
class QuickEditLocks:
    """The artifacts a quick edit is running on in this process.

    In process, like the preview check broker the edit asks through: the edit
    and the stream that carries its checks live on the request that started
    it. An agent repair, which another process may hold, is refused through
    the revision journal instead (``begin_quick_edit``).
    """

    _held: dict[str, float] = field(default_factory=dict)
    clock: Callable[[], float] = time.monotonic

    def acquire(self, key: str) -> bool:
        now = self.clock()
        since = self._held.get(key)
        if since is not None and now - since < LOCK_STALE_AFTER_S:
            return False
        self._held[key] = now
        return True

    def release(self, key: str) -> None:
        self._held.pop(key, None)

    def held(self, key: str) -> bool:
        since = self._held.get(key)
        return since is not None and self.clock() - since < LOCK_STALE_AFTER_S

    def reset(self) -> None:
        self._held.clear()


quick_edit_locks = QuickEditLocks()


# ── is there a decisions model for this person? ──────────────────────────


@dataclass(frozen=True)
class Decisions:
    """Whether a decisions model (Jev) can route this person's bar messages:
    anton's ``decisions_config`` answer, ``(base_url, api_key, model)``."""

    available: bool
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)


NO_DECISIONS = Decisions(available=False)

#: Stands in for the MindsHub key a hosted org's edit mints only once it runs
#: (it keeps no stored key), so anton can say whether a model would serve it.
_MINTED_KEY = "minted-for-this-edit"


def _hosted_minted(user) -> bool:
    """Whether this person's MindsHub calls run on a key minted per call:
    a hosted org on MindsHub with no stored key, as remote turns are."""
    from cowork.common.settings.app_settings import TurnQueueSettings
    from cowork.common.settings.user_settings import Provider, provider_api_key_str

    return (
        user.resolved_planning_provider is Provider.MINDS_CLOUD
        and not provider_api_key_str(user, Provider.MINDS_CLOUD)
        and TurnQueueSettings().is_remote
    )


def decisions_for(anton: AntonQuickEdit | None, user) -> Decisions:
    """anton's answer for *user* (their ``UserSettings``), read from
    ``AntonSettings`` built the way their turns build them, so a person on
    their own OpenAI or Anthropic keys gets "unavailable". Unavailable too
    for an anton without ``decisions_available`` / ``decisions_config``, and
    for an organization that turned Jev off (``jev_decisions_enabled``: it
    must not receive their text). Never raises."""
    if anton is None or anton.decisions_available is None or anton.decisions_config is None:
        return NO_DECISIONS
    try:
        if getattr(user, "jev_decisions_enabled", True) is False:
            return NO_DECISIONS
        from cowork.harnesses.anton_harness.harness import anton_settings_for_user

        from cowork.common.settings.user_settings import Provider, provider_api_key_str

        settings = anton_settings_for_user(user)
        # Only what the person configured decides. AntonSettings also reads
        # the process's ANTON_* environment and ~/.anton/.env, and derives an
        # OpenAI base URL from the MindsHub URL; on a machine that once used
        # MindsHub that made a person on their own Anthropic key look like a
        # MindsHub one. A turn overlays only non-empty values; this gate
        # clears the two that decide "MindsHub" when the person has none.
        if hasattr(settings, "openai_base_url"):
            settings.openai_base_url = getattr(user, "openai_base_url", None) or ""
        if hasattr(settings, "minds_api_key"):
            settings.minds_api_key = provider_api_key_str(user, Provider.MINDS_CLOUD) or ""
            if _hosted_minted(user):
                settings.minds_api_key = _MINTED_KEY
        if not anton.decisions_available(settings):
            return NO_DECISIONS
        config = anton.decisions_config(settings)
    except Exception:
        logger.warning("Could not tell whether a decisions model serves this person", exc_info=True)
        return NO_DECISIONS
    if not isinstance(config, (tuple, list)) or len(config) != 3:
        return NO_DECISIONS
    base_url, api_key, model = (str(v) if v else None for v in config)
    if not model:
        return NO_DECISIONS
    if api_key == _MINTED_KEY:
        # The edit's minted key and its base replace these (resolve_backend).
        base_url, api_key = None, None
    return Decisions(available=True, model=model, base_url=base_url, api_key=api_key)


async def load_user_settings(scope: TenantScope):
    """The caller's ``UserSettings``, read off the event loop."""
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.db.units import run_on_unit_threads

    return await run_on_unit_threads(partial(get_user_settings, scope))


async def decisions_for_scope(anton: AntonQuickEdit | None, scope: TenantScope) -> tuple[Any, Decisions]:
    """The caller's settings and whether a decisions model serves them."""
    if anton is None or anton.decisions_available is None:
        return None, NO_DECISIONS
    user = await load_user_settings(scope)
    return user, decisions_for(anton, user)


async def direct_edit_available(scope: TenantScope) -> bool:
    """``quickEditDirect``: whether the bar can edit directly for the caller,
    not only hand the message to the chat. Never raises."""
    try:
        _user, decisions = await decisions_for_scope(anton_quick_edit(), scope)
    except Exception:
        logger.warning("Could not read whether quick edits can run directly", exc_info=True)
        return False
    return decisions.available


# ── the editor's LLM and the router's endpoint ───────────────────────────


@dataclass
class QuickEditBackend:
    """What one quick edit calls: the editor's LLM, and where to route."""

    llm: Any
    # Jev's decisions endpoint, as anton's route_request takes it: the base
    # URL ``/decisions`` is appended to, and its key. None: no router here,
    # so every message goes to the agent unless the app forces the edit.
    route_base_url: str | None = None
    route_api_key: str | None = field(default=None, repr=False)
    route_model: str | None = None
    # The correlation id of a turn key minted for this edit, revoked by
    # ``close``. None when the edit runs on stored keys.
    minted_correlation_id: str | None = None

    async def close(self) -> None:
        if self.minted_correlation_id:
            from cowork.turnqueue.producer import revoke_turn_key_quietly

            await revoke_turn_key_quietly(self.minted_correlation_id)
            self.minted_correlation_id = None


async def resolve_backend(
    scope: TenantScope,
    user=None,
    decisions: Decisions | None = None,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> QuickEditBackend:
    """The editor's LLM and the router's endpoint, for the caller.

    The caller's own settings, as their turns read them; the router's
    endpoint and model are anton's ``decisions_config``. Where the hosted
    server keeps no MindsHub key (remote turns mint one per turn), a short
    turn key is minted here the same way, routes too, and is revoked by
    ``close()``. Raises when no LLM can be built (no provider key): the route
    reports it as an error event.

    ``model`` / ``reasoning_effort`` are the bar's pick, already checked
    against the composer's catalog (``artifact_quick_edit._model_choice``),
    applied the way a turn applies the composer's: ``build_llm_client``'s
    ``model_override`` and ``effort_override``. None keeps the person's own
    settings. Routing never uses them: Jev is ``decisions_config``'s.
    """
    from cowork.common.settings.app_settings import TurnQueueSettings
    from cowork.services.providers import build_llm_client

    settings = user if user is not None else await load_user_settings(scope)
    decisions = decisions or NO_DECISIONS
    if _hosted_minted(settings):
        return await _minted_backend(
            scope, settings, TurnQueueSettings(), decisions, model=model, reasoning_effort=reasoning_effort,
        )
    return QuickEditBackend(
        llm=build_llm_client(effort_override=reasoning_effort, model_override=model, settings=settings),
        route_base_url=decisions.base_url,
        route_api_key=decisions.api_key,
        route_model=decisions.model,
    )


async def _minted_backend(
    scope: TenantScope, settings, queue_settings, decisions: Decisions,
    *, model: str | None = None, reasoning_effort: str | None = None,
) -> QuickEditBackend:
    """A MindsHub LLM on a turn key minted for this edit (hosted orgs keep no
    stored key), as the routing gate mints one (``_router_binding``)."""
    from anton.core.llm.client import LLMClient
    from anton.core.llm.openai import OpenAIProvider

    from cowork.common.settings.app_settings import MINDS_FREE_MODEL
    from cowork.turnqueue.producer import mint_llm_block

    corr = f"quick-edit-{uuid4()}"
    block = await mint_llm_block(
        org_id=scope.org_id,
        user_id=scope.user_id,
        correlation_id=corr,
        settings=queue_settings,
        workspace_id=getattr(settings, "hub_workspace_id", "") or None,
    )
    backend = QuickEditBackend(llm=None, minted_correlation_id=corr)
    try:
        provider = OpenAIProvider(
            api_key=block["api_key"],
            base_url=block["base_url"],
            flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        )
        planning_model = model or settings.resolved_planning_model or MINDS_FREE_MODEL
        backend.llm = LLMClient(
            planning_provider=provider,
            planning_model=planning_model,
            coding_provider=provider,
            coding_model=model or block.get("coding_model") or planning_model,
        )
        if decisions.available:
            backend.route_base_url = block["base_url"]
            backend.route_api_key = block["api_key"]
            backend.route_model = decisions.model
        return backend
    except BaseException:
        await backend.close()
        raise


# ── the conversation's record of quick edits ──────────────────────────────
#
# Consecutive quick edits on one artifact in one conversation, with no other
# message between them, are ONE assistant message (harness
# ``cowork-quick-edit``): a "group". Its content is compact model-facing
# text, rebuilt on every change, so the agent's later turns (which replay
# saved messages) know what changed. Its structured data rides as one event
# on the message, ``{"type": QUICK_EDIT_GROUP_EVENT, "kind":
# "quick_edit_group", ...}``: the messages API already returns each
# message's events, and an event without ``thought_role`` is never replayed
# to the model. No new column.

QUICK_EDIT_GROUP_EVENT = "response.quick_edit_group"
QUICK_EDIT_GROUP_KIND = "quick_edit_group"
#: A group past this many edits is closed and the next edit starts another,
#: so one message and its text stay small.
MAX_GROUP_EDITS = 50
_MAX_REQUEST = 4000
_MAX_TEXT_REQUEST = 200
_MAX_SUMMARY = 500
_MAX_TEXT_FILES = 6


def _clip(text: object, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def quick_edit_entry(
    *, edit_id: str | None, request: str, summary: str, changed: list[str],
    revision_id: str | None, model: str | None, at: str | None = None,
) -> dict:
    """One edit as the group lists it."""
    from datetime import datetime, timezone

    return {
        "editId": edit_id,
        "request": str(request or "")[:_MAX_REQUEST],
        "summary": _clip(summary, _MAX_SUMMARY),
        "changed": [str(p) for p in changed][:200],
        "revisionId": revision_id,
        "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "undone": False,
    }


def group_text(group: dict) -> str:
    """The model-facing text of a group: what was asked, what changed."""
    lines = [f"Changes made directly in the viewer of artifact `{group.get('artifactSlug') or group.get('artifactId')}`:"]
    for number, edit in enumerate(group.get("edits") or [], start=1):
        files = list(edit.get("changed") or [])
        shown = ", ".join(files[:_MAX_TEXT_FILES]) + (f" +{len(files) - _MAX_TEXT_FILES} more" if len(files) > _MAX_TEXT_FILES else "")
        where = f" ({shown})" if shown else ""
        undone = " (undone)" if edit.get("undone") else ""
        lines.append(
            f'{number}. Asked: "{_clip(edit.get("request"), _MAX_TEXT_REQUEST)}" → {_clip(edit.get("summary"), _MAX_SUMMARY)}{where}{undone}'
        )
    lines.append("These are draft changes; nothing was published.")
    return "\n".join(lines)


def _group_event(message, session):
    """The group event of *message*, or None."""
    from cowork.models.message_event import MessageEvent

    for event in session.exec(
        session.select(MessageEvent).where(MessageEvent.message_id == message.id)
    ).all():
        data = event.event_data
        if isinstance(data, dict) and data.get("type") == QUICK_EDIT_GROUP_EVENT:
            return event
    return None


def _turn_in_flight(session, conversation_id: UUID) -> bool:
    from cowork.models.message import Message

    return session.exec(
        session.select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.pending == True)  # noqa: E712 — SQL boolean column
        .limit(1)
    ).first() is not None


def _save_group_edit(
    session: ScopedSession, *, conversation_id: UUID, artifact_id: str, artifact_slug: str,
    artifact_title: str, edit: dict,
) -> bool:
    """Add *edit* to the conversation's quick-edit group for the artifact:
    the latest message when it is that group, else a new one. False (nothing
    written) while a turn runs in the conversation: its question is saved
    pending, and a message now would land between it and its answer."""
    from cowork.services.conversations import ConversationService

    service = ConversationService(session)
    # Raises ValueError for a conversation the caller does not own.
    service.get_conversation(conversation_id)
    if _turn_in_flight(session, conversation_id):
        return False
    latest = session.exec(
        service._ordered_messages(conversation_id, include_pending=True, newest_first=True).limit(1)
    ).first()
    event = None
    if latest is not None and latest.role == "assistant" and latest.harness == QUICK_EDIT_HARNESS:
        event = _group_event(latest, session)
        data = event.event_data if event is not None else None
        if not (
            isinstance(data, dict)
            and data.get("artifactId") == artifact_id
            and len(data.get("edits") or []) < MAX_GROUP_EDITS
        ):
            event = None
    if event is not None:
        group = dict(event.event_data)
        group["edits"] = [*group.get("edits", []), edit]
        group["artifactSlug"], group["artifactTitle"] = artifact_slug, artifact_title
        # Reassigned, not mutated: the JSON columns see a new value.
        event.event_data = group
        latest.content = group_text(group)
        session.add(event)
        session.add(latest)
        session.commit()
        return True
    group = {
        "type": QUICK_EDIT_GROUP_EVENT,
        "kind": QUICK_EDIT_GROUP_KIND,
        "artifactId": artifact_id,
        "artifactSlug": artifact_slug,
        "artifactTitle": artifact_title,
        "edits": [edit],
    }
    service.save_assistant_turn(conversation_id, group_text(group), [group], harness=QUICK_EDIT_HARNESS)
    return True


#: How many of a conversation's newest quick-edit groups an undo looks in.
_UNDO_LOOKBACK = 50


def _mark_undone(session: ScopedSession, *, conversation_id: UUID, edit_id: str) -> bool:
    """Mark *edit_id* undone in the group that lists it, and rebuild its
    text. False when no recent group lists it."""
    from cowork.models.message import Message
    from cowork.services.conversations import ConversationService

    service = ConversationService(session)
    service.get_conversation(conversation_id)
    groups = session.exec(
        service._ordered_messages(conversation_id, include_pending=True, newest_first=True)
        .where(Message.harness == QUICK_EDIT_HARNESS)
        .limit(_UNDO_LOOKBACK)
    ).all()
    for message in groups:
        event = _group_event(message, session)
        if event is None or not isinstance(event.event_data, dict):
            continue
        edits = event.event_data.get("edits") or []
        if not any(isinstance(e, dict) and e.get("editId") == edit_id for e in edits):
            continue
        group = dict(event.event_data)
        group["edits"] = [
            {**e, "undone": True} if isinstance(e, dict) and e.get("editId") == edit_id else e for e in edits
        ]
        event.event_data = group
        message.content = group_text(group)
        session.add(event)
        session.add(message)
        session.commit()
        return True
    return False


async def _conversation_write(conversation_id: UUID, scope: TenantScope, fn) -> bool:
    from cowork.db.units import conversation_writes, run_db

    try:
        async with conversation_writes(conversation_id):
            return await run_db(fn, scope=scope)
    except Exception:
        logger.warning("Could not update the quick edits of conversation %s", conversation_id, exc_info=True)
        return False


async def record_quick_edit_in_conversation(
    scope: TenantScope, *, conversation_id: UUID, artifact_id: str, artifact_slug: str,
    artifact_title: str, edit: dict,
) -> bool:
    """Add an applied quick edit to the artifact's conversation (see the
    section comment). Under the conversation's write lock, as a direct
    answer is saved. Skipped (False) while a turn runs there, for a
    conversation the caller does not own, and on any failure: the edit is
    saved either way."""
    return await _conversation_write(conversation_id, scope, partial(
        _save_group_edit, conversation_id=conversation_id, artifact_id=artifact_id,
        artifact_slug=artifact_slug, artifact_title=artifact_title, edit=edit,
    ))


async def record_undo_in_conversation(scope: TenantScope, *, conversation_id: UUID, edit_id: str) -> bool:
    """Mark an undone quick edit in its conversation's group."""
    return await _conversation_write(conversation_id, scope, partial(
        _mark_undone, conversation_id=conversation_id, edit_id=edit_id,
    ))
