"""Coworker registry — every registered harness's descriptor + schema.

The frontend's coworker picker and Settings panel are pure consumers of
this endpoint: they never special-case a coworker by id, only render
from `category`/`priority`/`tags`/`configurationSchema`.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from cowork.harnesses.base import get_harness, list_descriptors

router = APIRouter()


@router.get("/")
def get_harnesses():
    return list_descriptors()


@router.get("/{harness_id}/status")
async def get_harness_status(harness_id: str):
    """Install/login status for the Settings "CLI Agents" panel. Runs the
    CLI's own status check (a subprocess call for CLI coworkers), so it's
    a separate lazy-loaded endpoint from the fast descriptor list above —
    the frontend calls this per-card, not on every Settings render."""
    try:
        harness = get_harness(harness_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    check = getattr(harness, "check_status", None)
    if check is None:
        # Not a CLI coworker (e.g. anton/hermes) — nothing to check.
        return {"installed": True, "path": None, "loggedIn": True, "detail": "Built-in agent — no external CLI required."}
    return await asyncio.to_thread(check)
