"""Browser Control services.

`BROWSER_CONNECT_FLOW_STEPS` is the single source of truth for the in-app
connect flow wording (Task A1) — interpolated into both the no-session
verdict detail (client.py) and the LLM tool prompt (anton_harness/tools.py)
so a future rename of the flow changes exactly one place.
"""
from __future__ import annotations

BROWSER_CONNECT_FLOW_STEPS = (
    "Connect Apps and Data → Connect → Browser Control → "
    "pick a Chrome tab and approve it"
)
