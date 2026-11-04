"""Chimera on a Cowork turn: whether it is on, what it runs, and who may offer it.

Chimera gives the planning model anton's `delegate` tool, which hands bounded,
checkable work to a cheaper sidekick model and returns a report the planning
model reviews. Two things turn it on for a turn:

- the composer's pseudo-model ``chimera`` (CHIMERA_MODEL_ID): the conversation
  stores ``chimera`` as its model, and each of its turns plans on
  ``chimera_main_model`` with ``chimera_sidekick_model`` as the sidekick. Both
  are MindsHub aliases, so the pick means something only while planning runs on
  MindsHub; anywhere else the turn runs the account's own models.
- ``chimera_enabled``: every turn, on the configured planning model, with the
  sidekick settings (``sidekick_provider`` / ``sidekick_model``). This is the
  path the evals use.

Either way Chimera is ignored on openai-compatible in v1, and it needs an anton
that ships it (``anton_supports_chimera``); without one the turn runs as before.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import TYPE_CHECKING, NamedTuple

from cowork.common.settings.app_settings import CHIMERA_MODEL_ID

if TYPE_CHECKING:
    from cowork.common.settings.user_settings import UserSettings

logger = logging.getLogger(__name__)

# What the planning role asks the MindsHub cache to keep its prefix for on a
# Chimera turn, unless the operator set ANTON_MAIN_CACHE_TTL: the main agent
# idles through whole delegations, longer than the gateway's five minutes.
CHIMERA_MAIN_CACHE_TTL = "1h"


class ChimeraTurn(NamedTuple):
    """How one turn runs, as far as Chimera decides it."""

    on: bool
    # The planning model the turn runs, as build_llm_client's model_override:
    # the request's own pick, the pseudo-model's main model, or None for the
    # account's planning model. Never CHIMERA_MODEL_ID.
    model: str | None
    # The pseudo-model's sidekick, a model from the planning provider's
    # catalogue; None runs the sidekick settings.
    sidekick_model: str | None = None


def chimera_turn(model: str | None, settings: UserSettings) -> ChimeraTurn:
    """Resolve the request's model pick against Chimera. Pure: settings only."""
    from cowork.common.settings.user_settings import Provider

    if model == CHIMERA_MODEL_ID:
        if settings.resolved_planning_provider is Provider.MINDS_CLOUD:
            return ChimeraTurn(True, settings.chimera_main_model, settings.chimera_sidekick_model)
        # Offered only on MindsHub (chimera_available). A conversation that
        # kept the pick across a provider switch runs the account's models
        # rather than sending a MindsHub alias to another provider.
        model = None
    # The common case reads nothing else: Chimera off resolves no provider.
    if not getattr(settings, "chimera_enabled", False):
        return ChimeraTurn(False, model)
    on = Provider.OPENAI_COMPATIBLE not in (
        settings.resolved_planning_provider,
        settings.resolved_sidekick_provider,
    )
    return ChimeraTurn(on, model)


def remote_turn_model(model: str | None, settings: UserSettings) -> str | None:
    """The model a hosted turn's pod runs for the request's pick.

    The pod builds its own anton session and has no Chimera wiring yet, so the
    pseudo-model runs its main model alone there instead of reaching the
    gateway as a model id nothing serves. Any other pick passes through.
    """
    if model != CHIMERA_MODEL_ID:
        return model
    return chimera_turn(model, settings).model


def _chimera_config_class():
    """anton's ChimeraConfig, or None when the installed anton has no Chimera."""
    try:
        from anton.core.chimera.config import ChimeraConfig
    except Exception:
        return None
    return ChimeraConfig


def anton_supports_chimera() -> bool:
    """Whether the installed anton can run a Chimera turn this server builds.

    All three pieces ship together in anton, but each is checked: a session
    config without ``chimera``, or a client without a sidekick role, would
    raise on every turn if passed one.
    """
    if _chimera_config_class() is None:
        return False
    try:
        from anton.core.llm.client import LLMClient
        from anton.core.session import ChatSessionConfig

        client_params = inspect.signature(LLMClient.__init__).parameters
        session_fields = {field.name for field in dataclasses.fields(ChatSessionConfig)}
    except Exception:
        return False
    return "sidekick_provider" in client_params and {"chimera", "sidekick_state"} <= session_fields


def chimera_available(settings: UserSettings) -> bool:
    """Whether the composer may offer the Chimera pseudo-model (``chimera_available``).

    True only where a pick would run Chimera: planning on MindsHub, turns built
    in this process, and an anton that ships it. Hosted turns run on a pod that
    has no Chimera wiring yet, and openai-compatible operator deployments have
    no MindsHub models to pair, so neither is offered it.
    """
    from cowork.common.chat_session import in_process_agent_allowed
    from cowork.common.settings.app_settings import TurnQueueSettings
    from cowork.common.settings.user_settings import Provider

    try:
        if settings.resolved_planning_provider is not Provider.MINDS_CLOUD:
            return False
        if TurnQueueSettings().is_remote or not in_process_agent_allowed():
            return False
    except Exception:
        logger.debug("could not resolve Chimera availability", exc_info=True)
        return False
    return anton_supports_chimera()


def build_chimera_config(anton_settings=None):
    """An enabled ChimeraConfig for a turn, or None when anton has none.

    The loop's limits come from anton's own settings when it declares them
    (ANTON_CHIMERA_MAX_ROUNDS, ANTON_CHIMERA_JEV_EVERY, ANTON_CHIMERA_JEV_THRESHOLD,
    which the operator or an eval sets in the server's environment); cowork's
    settings say only whether Chimera is on.
    """
    config_class = _chimera_config_class()
    if config_class is None:
        return None
    kwargs: dict = {"enabled": True}
    for field, attr in (
        ("max_rounds", "chimera_max_rounds"),
        ("jev_every", "chimera_jev_every"),
        ("jev_threshold", "chimera_jev_threshold"),
    ):
        value = getattr(anton_settings, attr, None)
        if value is not None:
            kwargs[field] = value
    try:
        return config_class(**kwargs)
    except Exception:
        logger.warning("Could not build anton's ChimeraConfig; this turn runs without Chimera", exc_info=True)
        return None


def planning_cache_ttl(anton_settings, *, chimera: bool) -> str | None:
    """The planning role's prompt-cache lifetime: ANTON_MAIN_CACHE_TTL when the
    operator set it, else an hour on a Chimera turn and the gateway's own
    default otherwise."""
    configured = getattr(anton_settings, "main_cache_ttl", None)
    if isinstance(configured, str) and configured:
        return configured
    return CHIMERA_MAIN_CACHE_TTL if chimera else None
