"""An invalid turn-queue setting stops the server at boot, not on every turn."""
from __future__ import annotations

import pytest
from fastapi import FastAPI

from cowork.server import lifespan


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["15", "2s", "0"])
async def test_an_invalid_gate_budget_fails_the_boot_without_echoing_it(monkeypatch, value):
    monkeypatch.setenv("COWORK_TURN_GATE_TOTAL_TIMEOUT_SECONDS", value)
    with pytest.raises(RuntimeError) as err:
        async with lifespan(FastAPI()):
            pass
    assert str(err.value) == "invalid turn-queue settings: gate_total_timeout_seconds"
    assert err.value.__suppress_context__, "the pydantic error would echo the value"
