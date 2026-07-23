"""Approval metrics (M4): the reliability claim, measured.

The board's headline — "N things need you, Anton has the rest" — is a
reliability claim. These numbers keep it honest:

  - autonomy ratio: shipped (approved+edited) : needs-you (pending + skipped)
  - edit/skip rate: how often the human changes or refuses (approval quality)
  - time-to-resolve: how long proposals sit waiting (median seconds)
  - injection tripwire hits: measured attack rate on the content channel
  - gate quality: parked proposals + rejected tokens per tool (the signal
    that decides whether tool-surface consolidation is ever a unit)

The first three come from the approvals table; the counters are in-process
(gate/tripwire modules own them; M4 only reads).
"""

from __future__ import annotations

from statistics import median
from typing import Any

from sqlmodel import Session, select

from cowork.models.approval import Approval


def _seconds(a, b) -> float:
    return max(0.0, (b - a).total_seconds())


def approval_metrics(session: Session) -> dict[str, Any]:
    rows = session.exec(select(Approval)).all()

    def count(*statuses: str) -> int:
        return sum(1 for a in rows if a.status in statuses)

    shipped = count("approved", "edited")
    needs_you = count("pending", "skipped")
    approved, edited, skipped = count("approved"), count("edited"), count("skipped")

    decisions = approved + edited + skipped
    edit_rate = (edited / decisions) if decisions else 0.0
    skip_rate = (skipped / decisions) if decisions else 0.0

    resolve_times = [
        _seconds(a.created_at, a.resolved_at)
        for a in rows
        if a.resolved_at is not None and a.created_at is not None
    ]
    median_ttr = median(resolve_times) if resolve_times else None

    # In-process counters — owned by the gate/tripwire, read here.
    from cowork.harnesses.anton_harness.browser_tools import GATE_HITS, TRIPWIRE_HITS

    return {
        "shipped": shipped,
        "needsYou": needs_you,
        "autonomyRatio": round(shipped / needs_you, 3) if needs_you else None,
        "editRate": round(edit_rate, 3),
        "skipRate": round(skip_rate, 3),
        "medianTimeToResolveSeconds": median_ttr,
        "injectionTripwireHits": dict(TRIPWIRE_HITS),
        "gateQuality": {tool: dict(hits) for tool, hits in GATE_HITS.items()},
    }
