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
* ``record_conversation_note()``: what the artifact's conversation keeps of
  an applied edit, so the agent's later turns know about it.
"""

from __future__ import annotations

import logging
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
    "conversation_note_text",
    "decisions_for",
    "decisions_for_scope",
    "direct_edit_available",
    "quick_edit_available",
    "quick_edit_locks",
    "record_conversation_note",
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
        from anton.core.artifact_editor import quick, router
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
        route_extras=frozenset(params & {"name", "description", "data", "history"}),
        decisions_available=_callable(router, "decisions_available"),
        decisions_config=_callable(router, "decisions_config"),
    )


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
    """Whether a decisions model (Jev) can route this person's bar messages,
    and which model anton names for it."""

    available: bool
    model: str | None = None


NO_DECISIONS = Decisions(available=False)


def _config_model(config: object) -> str | None:
    if isinstance(config, str):
        return config or None
    if isinstance(config, dict):
        model = config.get("model")
    else:
        model = getattr(config, "model", None)
    return model if isinstance(model, str) and model else None


def decisions_for(anton: AntonQuickEdit | None, user) -> Decisions:
    """anton's answer for *user* (their ``UserSettings``), read from
    ``AntonSettings`` built the way their turns build them, so a person on
    their own OpenAI or Anthropic keys gets "unavailable". An anton without
    ``decisions_available`` is unavailable too. Never raises."""
    if anton is None or anton.decisions_available is None:
        return NO_DECISIONS
    try:
        from cowork.harnesses.anton_harness.harness import anton_settings_for_user

        settings = anton_settings_for_user(user)
        if not anton.decisions_available(settings):
            return NO_DECISIONS
        config = anton.decisions_config(settings) if anton.decisions_config is not None else None
    except Exception:
        logger.warning("Could not tell whether a decisions model serves this person", exc_info=True)
        return NO_DECISIONS
    if config is None and anton.decisions_config is not None:
        return NO_DECISIONS
    return Decisions(available=True, model=_config_model(config))


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


_DECISIONS_SUFFIX = "/decisions"


def _route_target(settings, *, minds_base_url: str | None, minds_key: str | None) -> tuple[str | None, str | None, str | None]:
    """(base_url, api_key, model) of the decision endpoint, or Nones.

    The turn gate's own choice (``jev_route.gate_mode``): the organization's
    own decision model when configured, else Jev on MindsHub for a MindsHub
    caller who did not opt out. An own endpoint is used only where it is a
    ``.../decisions`` endpoint (route_request appends the path) and never
    from the hosted server, which dials an organization's own URL only
    through the turn gate's pinned transport.
    """
    from cowork.handlers import jev_route

    mode = jev_route.gate_mode(settings)
    if mode == "own":
        target = jev_route.own_decision_target(settings)
        if target is None or get_app_settings().tenancy_mode == "org":
            return None, None, None
        url = target.url.rstrip("/")
        if not url.endswith(_DECISIONS_SUFFIX):
            return None, None, None
        return url[: -len(_DECISIONS_SUFFIX)], target.api_key, settings.decision_model or None
    if mode == "jev" and minds_base_url and minds_key:
        # anton's route_request defaults to the Jev model the router was
        # evaluated with.
        return minds_base_url, minds_key, None
    return None, None, None


async def resolve_backend(scope: TenantScope, user=None) -> QuickEditBackend:
    """The editor's LLM and the router's endpoint, for the caller.

    The caller's own settings, as their turns read them. Where the hosted
    server keeps no MindsHub key (remote turns mint one per turn), a short
    turn key is minted here the same way and revoked by ``close()``.
    Raises when no LLM can be built (no provider key): the route reports it
    as an error event.
    """
    from cowork.common.settings.app_settings import TurnQueueSettings
    from cowork.common.settings.user_settings import Provider, provider_api_key_str
    from cowork.services.providers import build_llm_client, minds_chat_base_url

    settings = user if user is not None else await load_user_settings(scope)
    queue_settings = TurnQueueSettings()
    minds_key = provider_api_key_str(settings, Provider.MINDS_CLOUD)
    on_minds = settings.resolved_planning_provider is Provider.MINDS_CLOUD
    if on_minds and not minds_key and queue_settings.is_remote:
        return await _minted_backend(scope, settings, queue_settings)
    llm = build_llm_client(settings=settings)
    base, key, model = _route_target(
        settings,
        minds_base_url=minds_chat_base_url(settings.minds_url) if minds_key else None,
        minds_key=minds_key or None,
    )
    return QuickEditBackend(llm=llm, route_base_url=base, route_api_key=key, route_model=model)


async def _minted_backend(scope: TenantScope, settings, queue_settings) -> QuickEditBackend:
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
        )
        planning_model = settings.resolved_planning_model or MINDS_FREE_MODEL
        backend.llm = LLMClient(
            planning_provider=provider,
            planning_model=planning_model,
            coding_provider=provider,
            coding_model=block.get("coding_model") or planning_model,
        )
        base, key, model = _route_target(settings, minds_base_url=block["base_url"], minds_key=block["api_key"])
        backend.route_base_url, backend.route_api_key, backend.route_model = base, key, model
        return backend
    except BaseException:
        await backend.close()
        raise


# ── the conversation note ────────────────────────────────────────────────

_MAX_NOTE_MESSAGE = 4000
_MAX_NOTE_SUMMARY = 500
_MAX_NOTE_FILES = 10


def conversation_note_text(*, summary: str, changed: list[str], revision: dict | None) -> str:
    """The assistant side of the note: what changed, where, and how to undo."""
    lines = [f"I edited this artifact directly from its viewer (quick edit): {' '.join(summary.split())[:_MAX_NOTE_SUMMARY]}"]
    if changed:
        shown = ", ".join(changed[:_MAX_NOTE_FILES])
        more = f" and {len(changed) - _MAX_NOTE_FILES} more" if len(changed) > _MAX_NOTE_FILES else ""
        lines.append(f"Files changed: {shown}{more}.")
    if revision and revision.get("number"):
        lines.append(f"Saved as draft revision {revision['number']}; not published.")
    return "\n".join(lines)


def _save_note(session: ScopedSession, *, conversation_id: UUID, message: str, text: str) -> bool:
    from cowork.models.message import Message
    from cowork.services.conversations import ConversationService

    service = ConversationService(session)
    # Raises ValueError for a conversation the caller does not own.
    service.get_conversation(conversation_id)
    in_flight = session.exec(
        session.select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.pending == True)  # noqa: E712 — SQL boolean column
        .limit(1)
    ).first()
    if in_flight is not None:
        # A turn is running (its question is saved pending until it ends).
        # A note written now would land between its question and its answer.
        return False
    service.save_user_message(conversation_id, message[:_MAX_NOTE_MESSAGE])
    service.save_assistant_turn(
        conversation_id,
        text,
        [
            {"type": "response.output_text.delta", "delta": text},
            {"type": "response.completed"},
        ],
        harness=QUICK_EDIT_HARNESS,
    )
    return True


async def record_conversation_note(
    scope: TenantScope, *, conversation_id: UUID, message: str, text: str,
) -> bool:
    """Save an applied quick edit in the artifact's conversation, as the turn
    it replaced: the person's message and a short answer saying what changed.

    The same mechanism a direct answer uses (``_handle_direct_response``):
    later agent turns replay the conversation's saved messages, so the agent
    sees the edit, and the person sees it in the chat. Under the
    conversation's write lock. Skipped (False) while a turn runs in that
    conversation, for a conversation the caller does not own, and on any
    failure: the edit is saved either way.
    """
    from cowork.db.units import conversation_writes, run_db

    try:
        async with conversation_writes(conversation_id):
            return await run_db(
                partial(_save_note, conversation_id=conversation_id, message=message, text=text),
                scope=scope,
            )
    except Exception:
        logger.warning("Could not record the quick edit in conversation %s", conversation_id, exc_info=True)
        return False
