"""The current clock reaches both text and attachment turns, including resumes."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.session import _stamp_user_content

from cowork.harnesses.anton_harness import harness
from tests.test_harness_connector_usage_notes import _session_config


NOW = datetime(2026, 11, 12, 20, 17, tzinfo=timezone.utc)


@pytest.fixture
def fixed_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return NOW

    monkeypatch.setattr(harness, "datetime", Clock)


@pytest.mark.parametrize("attachment", [
    {"type": "image", "image_url": {"url": "https://example.com/dashboard.png"}},
    {"type": "file", "filename": "dashboard.csv", "path": "/tmp/dashboard.csv"},
])
@pytest.mark.parametrize("with_text", [True, False])
def test_resumed_attachment_turn_has_its_own_utc_clock(fixed_clock, attachment, with_text):
    blocks = ([{"type": "text", "text": "Refresh this dashboard every hour"}] if with_text else [])
    blocks.append(attachment)
    original = deepcopy(blocks)
    history = [{"role": "user", "content": "[2026-11-11 20:17 UTC] Make a dashboard"}]

    # Exercise the installed Anton boundary too: block lists receive no stamp
    # there, so the harness must supply this turn's clock before handing off.
    incoming = _stamp_user_content(harness.AntonHarness._to_anton_input(blocks), NOW)
    history.append({"role": "user", "content": incoming})

    assert history[-1]["content"][0] == {"type": "text", "text": "[2026-11-12 20:17 UTC]"}
    assert history[0]["content"] == "[2026-11-11 20:17 UTC] Make a dashboard"
    assert blocks == original
    if with_text:
        assert incoming[1] == original[0]
    if attachment["type"] == "image":
        assert incoming[-1] == attachment
    else:
        assert incoming[-1] == {
            "type": "text", "text": "[Attached file 'dashboard.csv' is at: /tmp/dashboard.csv]",
        }


def test_plain_text_is_stamped_once_by_anton(fixed_clock):
    incoming = harness.AntonHarness._to_anton_input([{"type": "text", "text": "Refresh hourly"}])
    assert incoming == "Refresh hourly"
    assert _stamp_user_content(incoming, NOW) == "[2026-11-12 20:17] Refresh hourly"


def test_multiple_text_blocks_get_a_clock_without_losing_content(fixed_clock):
    blocks = [{"type": "text", "text": "Refresh"}, {"type": "text", "text": "hourly"}]
    incoming = harness.AntonHarness._to_anton_input(blocks)
    assert incoming[0] == {"type": "text", "text": "[2026-11-12 20:17 UTC]"}
    assert incoming[1:] == blocks
    assert harness.AntonHarness._to_anton_input([]) == []


async def test_utc_guidance_reaches_the_actual_session_prompt(monkeypatch, tmp_path):
    async with _session_config(monkeypatch, LocalDataVault(tmp_path / "vault")) as config:
        suffix = config.system_prompt_context.suffix
        assert "Message send-time prefixes are UTC" in suffix
        assert "[YYYY-MM-DD HH:MM] without a timezone label" in suffix
        assert "convert that instant to the user's timezone" in suffix
        assert "Never attach a local UTC offset to the unchanged UTC clock reading" in suffix
