"""Onboarding seed (O1): the first-run cascade, created deterministically.

Never LLM-driven — first impressions must not be flaky. The cascade:

  1. Empty world (no pinned apps) → card 1 "Pin the tools you work in"
     (kind action, tool onboarding_pin_apps — resolving it just notes the
     picker opened; the user's actual PIN is the cascade trigger).
  2. Card 1 resolved AND no digest schedule → card 3 "Want this every
     morning at 9:00?" (tool schedule_digest — its executor writes the real
     schedules row via services/digest.create_digest_schedule).

Eligibility derives from world state on every call, so it is idempotent,
skips cleanly for established users, and re-derives if a user pins an app
next week. Both cards ride the versioned action descriptor — no new kind.
"""

from __future__ import annotations

from typing import Any, Callable

from sqlmodel import Session, select

from cowork.models.approval import Approval
from cowork.models.schedule import Schedule
from cowork.schemas.approvals import ActionDescriptorV1
from cowork.services.approvals import ApprovalService, register_executor
from cowork.services.digest import DIGEST_TITLE, create_digest_schedule

PIN_CARD_TOOL = "onboarding_pin_apps"
DIGEST_CARD_TOOL = "schedule_digest"

PIN_CARD_SUMMARY = "Pin the tools you work in"
DIGEST_CARD_SUMMARY = "Want this every morning at 9:00?"


# ---------------------------------------------------------------------------
# Executors (registered at import — create() validates executability)
# ---------------------------------------------------------------------------

def _exec_pin_card(session: Session, args: dict, raw_token: str) -> dict:
    from cowork.services.approvals import ApprovalService

    ApprovalService(session).consume_token(raw_token, tool=PIN_CARD_TOOL, args=args)
    # The pin itself is the user's gesture in the app picker — this card's
    # job was to open it. The apps-registry write is the cascade trigger.
    return {"executed": True, "noted": "app picker opened"}


def _exec_digest_card(session: Session, args: dict, raw_token: str) -> dict:
    from cowork.services.approvals import ApprovalService

    ApprovalService(session).consume_token(raw_token, tool=DIGEST_CARD_TOOL, args=args)
    hour = args.get("hour", 9)
    timezone_name = args.get("timezone", "UTC")
    schedule = create_digest_schedule(
        session,
        hour=hour if isinstance(hour, int) else 9,
        timezone_name=str(timezone_name),
    )
    return {
        "executed": True,
        "schedule_id": str(schedule.id),
        "next_run_at": schedule.next_run_at.isoformat(),
        "summary": "See you at 9:00.",
    }


register_executor(PIN_CARD_TOOL, _exec_pin_card)
register_executor(DIGEST_CARD_TOOL, _exec_digest_card)


# ---------------------------------------------------------------------------
# Seed logic
# ---------------------------------------------------------------------------

def _cards(session: Session, tool: str) -> list[Approval]:
    rows = session.exec(select(Approval)).all()
    return [a for a in rows if (a.action_descriptor or {}).get("tool") == tool]


def _digest_schedule_exists(session: Session) -> bool:
    return session.exec(select(Schedule).where(Schedule.title == DIGEST_TITLE)).first() is not None


def ensure_onboarding_cards(
    session: Session,
    *,
    pinned_apps: list[dict[str, Any]],
    digest_hour: int = 9,
    digest_timezone: str = "UTC",
) -> dict[str, Any]:
    """Idempotent cascade seeding. Returns what exists/was created for callers
    (and tests): {pin_card, digest_card} as ids or None."""
    service = ApprovalService(session)
    result: dict[str, Any] = {"pin_card": None, "digest_card": None}

    # Card 1 — only an empty world earns it, and only ever once (any status).
    pin_cards = _cards(session, PIN_CARD_TOOL)
    if not pinned_apps and not pin_cards:
        conversation_id = _onboarding_conversation(session)
        card = service.create(
            conversation_id=conversation_id,
            descriptor=ActionDescriptorV1(tool=PIN_CARD_TOOL, args={}, summary=PIN_CARD_SUMMARY),
        )
        pin_cards = [card]
    result["pin_card"] = str(pin_cards[0].id) if pin_cards else None

    # Card 3 — the content engine. Established users (apps already pinned, no
    # digest schedule) skip card 1 entirely and go straight here: one click
    # and tomorrow morning has its first real work on the board. Without this
    # branch the cascade was unreachable for exactly the users who'd benefit.
    card1_done = pin_cards and pin_cards[0].status in ("approved", "edited", "skipped")
    digest_cards = _cards(session, DIGEST_CARD_TOOL)
    if (card1_done or pinned_apps) and not digest_cards and not _digest_schedule_exists(session):
        card3 = service.create(
            conversation_id=pin_cards[0].conversation_id if pin_cards else _onboarding_conversation(session),
            descriptor=ActionDescriptorV1(
                tool=DIGEST_CARD_TOOL,
                args={"hour": digest_hour, "timezone": digest_timezone},
                summary=DIGEST_CARD_SUMMARY,
            ),
        )
        digest_cards = [card3]
    result["digest_card"] = str(digest_cards[0].id) if digest_cards else None
    return result


def _onboarding_conversation(session: Session) -> Any:
    """The onboarding cards' home conversation — one per account, found by
    topic; created on first use. (Cards need a conversation_id; the first-run
    board reads them through it.)"""
    from cowork.models.conversation import Conversation
    from cowork.services.projects import GENERAL_PROJECT_ID

    topic = "Getting started"
    conv = session.exec(select(Conversation).where(Conversation.topic == topic)).first()
    if conv is None:
        conv = Conversation(project_id=GENERAL_PROJECT_ID, topic=topic)
        session.add(conv)
        session.commit()
        session.refresh(conv)
    return conv.id
