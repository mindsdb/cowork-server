"""Onboarding endpoint — idempotent cascade seeding.

The client calls POST /ensure on first-run (and it may call it again any
time — eligibility re-derives from world state). The bridge is the source
of truth for pinned apps; a DOWN bridge means "unknown", never "empty" —
we seed nothing rather than hand an established user a beginner card.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.services.onboarding import ensure_onboarding_cards

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.post("/ensure")
async def ensure_onboarding(session: SessionDep) -> dict:
    from cowork.harnesses.anton_harness.browser_tools import _bridge_call

    try:
        state = await _bridge_call("GET", "/state", timeout=5.0)
        apps = (state or {}).get("apps") or []
    except Exception:
        # Unknown ≠ empty: never seed from an unreadable registry.
        return {"seeded": False, "reason": "bridge unavailable"}

    result = ensure_onboarding_cards(session, pinned_apps=apps)
    return {"seeded": True, **result}
