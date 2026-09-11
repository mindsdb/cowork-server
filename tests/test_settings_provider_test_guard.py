"""ENG-2094: a STORED provider key never goes to a host the caller named.

``POST /api/v1/settings/test-providers`` treats an ``apiKey`` of ``""`` or
``"***"`` as "use the stored one", which is how the Settings UI re-tests a
provider it only ever received masked. Two provider types take their ping
target out of the same request body — ``openai-compatible`` reads ``baseUrl``
and ``minds-cloud`` reads ``mindsUrl`` — so before this guard a caller who
never knew the key could still choose where it was sent, and ``settings`` rows
are deployment-global (``_TENANCY_DEFERRED_TABLES``, cowork/db/scoped.py) so
the key is the whole deployment's.

These drive the endpoint function directly: the substitution happens before
``ping_providers``, and stubbing that out is what keeps the test from making a
real outbound request.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import settings as settings_endpoint
from cowork.common.settings.user_settings import UserSettings
from cowork.db.scoped import LOCAL_SCOPE

STORED_KEY = "sk-stored-do-not-leak"
STORED_MINDS_URL = "https://mdb.ai"
ATTACKER_URL = "https://attacker.example"


@pytest.fixture()
def pinged(monkeypatch):
    """Capture what would have been sent, instead of sending it."""
    seen: list[dict] = []

    async def _fake_ping(providers):
        seen.extend(providers)
        return ({}, {})

    monkeypatch.setattr(settings_endpoint, "ping_providers", _fake_ping)
    return seen


@pytest.fixture()
def stored(monkeypatch):
    """A deployment with a stored MindsHub key and its own saved URLs."""
    settings = UserSettings(
        minds_api_key=STORED_KEY,
        minds_url=STORED_MINDS_URL,
        openai_base_url="https://oc.internal",
    )

    class _Service:
        def __init__(self, *a, **kw):
            pass

        def load(self):
            return settings

    monkeypatch.setattr(settings_endpoint, "SettingService", _Service)
    monkeypatch.setattr(settings_endpoint, "resolve_stored_key", lambda s, ptype: STORED_KEY)
    return settings


def _test_providers(body):
    return asyncio.run(
        settings_endpoint.test_providers(
            session=None, scope=LOCAL_SCOPE, body=settings_endpoint._TestProvidersBody(providers=body)
        )
    )


def test_masked_key_against_a_foreign_minds_url_is_refused(stored, pinged):
    with pytest.raises(HTTPException) as exc:
        _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": ATTACKER_URL}])

    assert exc.value.status_code == 400
    assert "mindsUrl" in exc.value.detail
    assert pinged == [], "refused before anything was sent"


def test_empty_key_against_a_foreign_base_url_is_refused(stored, pinged):
    # "" is the other spelling of "use the stored one", and it is the one the
    # server's own no-body branch emits, so it has to be covered too.
    with pytest.raises(HTTPException) as exc:
        _test_providers([{"type": "openai-compatible", "apiKey": "", "baseUrl": ATTACKER_URL}])

    assert exc.value.status_code == 400
    assert pinged == []


def test_masked_key_against_the_stored_url_still_works(stored, pinged):
    # The Settings UI echoes back the stored URL with a masked key on every
    # mount-time verify; refusing that would break the Test button.
    _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": STORED_MINDS_URL}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


def test_a_caller_supplied_key_may_go_anywhere(stored, pinged):
    # Nothing stored is at risk when the caller brought their own credential.
    _test_providers(
        [{"type": "minds-cloud", "apiKey": "sk-callers-own", "mindsUrl": ATTACKER_URL}]
    )

    assert [p["mindsUrl"] for p in pinged] == [ATTACKER_URL]
    assert [p["apiKey"] for p in pinged] == ["sk-callers-own"]


def test_providers_with_a_hardcoded_host_are_unaffected(stored, pinged):
    # anthropic/openai/gemini ping a fixed vendor URL, so there is no body
    # field to aim and the guard must not get in the way.
    _test_providers([{"type": "anthropic", "apiKey": "***"}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


def test_a_provider_card_url_counts_as_stored(monkeypatch, pinged):
    # providers_json is what the Settings UI actually edits, and a deployment
    # can hold several cards, so "stored" cannot mean the scalar field alone.
    settings = UserSettings(
        minds_api_key=STORED_KEY,
        providers_json='[{"type": "openai-compatible", "baseUrl": "https://card.internal"}]',
    )

    class _Service:
        def __init__(self, *a, **kw):
            pass

        def load(self):
            return settings

    monkeypatch.setattr(settings_endpoint, "SettingService", _Service)
    monkeypatch.setattr(settings_endpoint, "resolve_stored_key", lambda s, ptype: STORED_KEY)

    _test_providers(
        [{"type": "openai-compatible", "apiKey": "***", "baseUrl": "https://card.internal/"}]
    )

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]
