"""A turn with "browser": "off" runs without the agent's browser.

The composer's Browser switch turns the browser off for one task. The handler
runs the turn on a copy of its settings snapshot with browser_enabled off, so
the harness's _browser_config and the producer's turn_block both see it off:
no browser tool for anton, and on the web no scope call that would start the
person's browser.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

import cowork.common.settings.user_settings as user_settings
import cowork.handlers.responses as responses_mod
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.schemas.responses import ResponsesRequest
from cowork.server import create_app
from cowork.streaming import registry

from _fakes import PausedHarness


class _SeesBrowser(PausedHarness):
    """Records browser_enabled as each turn's harness reads it, and answers at once."""

    def __init__(self) -> None:
        super().__init__()
        self.release.set()
        self.seen: list[bool] = []

    def stream_response(self, **kwargs):
        self.seen.append(user_settings.get_user_settings().browser_enabled)
        return super().stream_response(**kwargs)


@pytest.fixture(autouse=True)
def _forget_turns():
    yield
    registry.reset()


async def _ask(monkeypatch, harness: _SeesBrowser, **body) -> httpx.Response:
    async def decide(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: harness)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await asyncio.wait_for(
            client.post(
                "/api/v1/responses/",
                json={"input": "hi", "stream": True, "conversation": str(uuid4()), **body},
            ),
            timeout=10,
        )


async def test_a_turn_with_the_browser_off_runs_without_it_and_the_next_follows_the_setting(monkeypatch):
    """The copy is the one turn's: a turn without the key, and one that says
    "on", read the user's own setting again, which is off by default (beta)."""
    harness = _SeesBrowser()
    for body in ({"browser": "off"}, {}, {"browser": "on"}):
        answered = await _ask(monkeypatch, harness, **body)
        assert answered.status_code == 200 and "response.completed" in answered.text, answered.text
    # Off by default: a turn that doesn't turn it off still runs without it.
    assert harness.seen == [False, False, False]


def test_the_browser_is_off_by_default():
    assert user_settings.UserSettings.model_fields["browser_enabled"].default is False


@pytest.mark.parametrize("value", ["OFF", "none", "", False, None])
def test_browser_takes_only_on_or_off(value):
    assert ResponsesRequest(input="hi").browser == "on"
    with pytest.raises(ValidationError):
        ResponsesRequest(input="hi", browser=value)
