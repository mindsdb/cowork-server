"""The two-button confirmation card a tool shows before it acts.

The same card as the write-approval gate's: no typed answer, no default, and
outside the model's question budget, because the tool asks it, not the model.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from cowork.common.logger import get_logger

logger = get_logger(__name__)

Decision = Literal["confirmed", "cancelled", "timeout", "unavailable", "error"]


async def confirm_card(
    session: Any,
    *,
    key: str,
    prompt: str,
    confirm: tuple[str, str, str],
    cancel_detail: str,
    confirm_style: str | None = None,
) -> Decision:
    """Show the card and return the person's decision; only "confirmed" lets
    the tool act. ``key`` prefixes the card's id, ``confirm`` is the confirm
    button's (value, label, detail)."""
    value, label, detail = confirm
    try:
        from anton.core.interaction.elicit import AskOption, AskRequest, elicit

        style = {"style": confirm_style} if confirm_style else {}
        request = AskRequest(
            prompt=prompt,
            options=(
                AskOption(value=value, label=label, detail=detail, **style),
                AskOption(value="cancel", label="Cancel", detail=cancel_detail),
            ),
            allow_custom=False,
            timeout_s=getattr(getattr(session, "elicitor", None), "timeout_s", None),
        )
        answer = await elicit(session, f"{key}:{uuid.uuid4().hex}", request, counted=False)
    except Exception as exc:
        logger.warning("%s confirmation failed (%s)", key, type(exc).__name__)
        return "error"
    if answer.status == "answered" and answer.values == (value,):
        return "confirmed"
    if answer.status in ("answered", "cancelled"):
        return "cancelled"
    if answer.status in ("timeout", "unavailable"):
        return answer.status
    return "error"
