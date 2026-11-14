"""A cloud turn's connector templates as cards: one Anton named after a built-in
connector takes its own id, and one id is shown once."""
from __future__ import annotations

import json
from pathlib import Path

from cowork.services.connectors import custom_connectors
from cowork.services.connectors.custom_connectors import turn_connector_cards

# A template as anton's real tool produces it (recorded fixture).
_ENTRY = json.loads((Path(__file__).parent / "fixtures" / "turn_connector_reply.json").read_text())["data"]["entries"][0]


def _ids(cards):
    return [card["connector_id"] for card in cards]


def test_a_template_named_after_a_built_in_takes_its_own_id():
    cards, notices = turn_connector_cards([{**_ENTRY, "connector_id": "hasdata", "label": "HasData"}])

    assert _ids(cards) == ["hasdata_custom"]
    assert cards[0]["label"] == "HasData"
    assert notices == []


def test_two_templates_that_become_one_id_show_once():
    cards, notices = turn_connector_cards([
        {**_ENTRY, "connector_id": "hasdata"}, {**_ENTRY, "connector_id": "hasdata_custom"},
    ])

    assert _ids(cards) == ["hasdata_custom"]
    assert notices == ["A second connector template for 'hasdata_custom' was not shown."]


def test_a_renamed_id_that_is_also_built_in_is_refused(monkeypatch):
    builtin = {"hasdata", "hasdata_custom"}
    monkeypatch.setattr(custom_connectors, "is_builtin_connector_id", lambda connector_id: connector_id in builtin)

    cards, notices = turn_connector_cards([{**_ENTRY, "connector_id": "hasdata"}])

    assert cards == []
    assert notices == ["A connector template was not shown: 'hasdata_custom' is a built-in connector."]


def test_a_renamed_template_without_a_check_is_named_as_anton_named_it():
    entry = {k: v for k, v in _ENTRY.items() if k != "check"}

    _, notices = turn_connector_cards([{**entry, "connector_id": "hasdata"}])

    assert notices == ["A connector template was not shown: 'hasdata' has no connection check, so it cannot be connected."]


def test_a_template_without_a_check_is_a_notice_not_a_card():
    """An older pod's template with no check could never be connected."""
    entry = {k: v for k, v in _ENTRY.items() if k != "check"}

    cards, notices = turn_connector_cards([entry, _ENTRY])

    assert _ids(cards) == [_ENTRY["connector_id"]]
    assert notices == [
        f"A connector template was not shown: {_ENTRY['connector_id']!r} has no connection check, so it cannot be connected."
    ]
