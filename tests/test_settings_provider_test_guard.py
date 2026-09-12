"""ENG-2094: a STORED provider key never goes to a host the caller named.

``POST /api/v1/settings/test-providers`` treats an ``apiKey`` of ``""`` or
``"***"`` as "use the stored one", which is how the Settings UI re-tests a
provider it only ever received masked. Two provider types take their ping
target out of the same request body — ``openai-compatible`` reads ``baseUrl``
and ``minds-cloud`` reads ``mindsUrl`` — so before this guard a caller who
never knew the key could still choose where it was sent, and ``settings`` rows
are deployment-global (``_TENANCY_DEFERRED_TABLES``, cowork/db/scoped.py) so
the key is the whole deployment's.

The guard compares ORIGINS, not whole URL strings, and refuses one provider
rather than the request. Both choices are load-bearing and have their own
tests below: the server stores ``minds_url`` with a ``/v1`` path while the
Settings UI sends the same host without it, and both callers send every
configured provider in one request, so raising would blank the other
providers' status dots.

These drive the endpoint function directly: the substitution happens before
``ping_providers``, and stubbing that out is what keeps the test from making a
real outbound request.
"""
from __future__ import annotations

import asyncio

import pytest

from cowork.api.v1.endpoints import settings as settings_endpoint
from cowork.common.settings.app_settings import default_minds_api_host
from cowork.common.settings.user_settings import UserSettings
from cowork.db.scoped import LOCAL_SCOPE

STORED_KEY = "sk-stored-do-not-leak"
STORED_MINDS_URL = "https://mdb.ai"
ATTACKER_URL = "https://attacker.example"
REFUSAL = "this deployment has not saved"


@pytest.fixture()
def pinged(monkeypatch):
    """Capture what would have been sent, instead of sending it."""
    seen: list[dict] = []

    async def _fake_ping(providers):
        seen.extend(providers)
        return ({p.get("type"): "ok" for p in providers}, {p.get("type"): "connected" for p in providers})

    monkeypatch.setattr(settings_endpoint, "ping_providers", _fake_ping)
    return seen


def _deployment(monkeypatch, **fields):
    """Stand up a deployment whose stored settings are exactly ``fields``."""
    settings = UserSettings(**fields)

    class _Service:
        def __init__(self, *a, **kw):
            pass

        def load(self):
            return settings

    monkeypatch.setattr(settings_endpoint, "SettingService", _Service)
    monkeypatch.setattr(settings_endpoint, "resolve_stored_key", lambda s, ptype: STORED_KEY)
    return settings


@pytest.fixture()
def stored(monkeypatch):
    """A deployment with a stored MindsHub key and its own saved URLs."""
    return _deployment(
        monkeypatch,
        minds_api_key=STORED_KEY,
        minds_url=STORED_MINDS_URL,
        openai_base_url="https://oc.internal",
    )


def _test_providers(body):
    return asyncio.run(
        settings_endpoint.test_providers(
            session=None, scope=LOCAL_SCOPE, body=settings_endpoint._TestProvidersBody(providers=body)
        )
    )


def test_masked_key_against_a_foreign_minds_url_is_refused(stored, pinged):
    result = _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": ATTACKER_URL}])

    assert result["providerStatus"]["minds-cloud"] == "fail"
    assert REFUSAL in result["providerStatusDetails"]["minds-cloud"]
    assert "mindsUrl" in result["providerStatusDetails"]["minds-cloud"]
    assert pinged == [], "refused before anything was sent"


def test_empty_key_against_a_foreign_base_url_is_refused(stored, pinged):
    # "" is the other spelling of "use the stored one", and it is the one the
    # server's own no-body branch emits, so it has to be covered too.
    result = _test_providers([{"type": "openai-compatible", "apiKey": "", "baseUrl": ATTACKER_URL}])

    assert result["providerStatus"]["openai-compatible"] == "fail"
    assert REFUSAL in result["providerStatusDetails"]["openai-compatible"]
    assert pinged == []


def test_masked_key_against_the_stored_url_still_works(stored, pinged):
    # The Settings UI echoes back the stored URL with a masked key on every
    # mount-time verify; refusing that would break the Test button.
    _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": STORED_MINDS_URL}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


def test_the_path_the_settings_ui_actually_sends_is_not_refused(monkeypatch, pinged):
    """The renderer strips /v1; an unwritten minds_url row still has it.

    The renderer builds its MindsHub card with ``.replace(/\\/v1$/, '')``
    (cowork src/renderer/cowork/lib/settingsTransform.js) and a masked key,
    then the mount-time verify posts it. Every real writer stores the bare
    origin, so the two usually agree, but a deployment whose ``minds_url``
    row was never written falls back to ``default_minds_url``'s ``<host>/v1``
    and comparing whole URL strings refused its own Settings screen.
    """
    _deployment(monkeypatch, minds_api_key=STORED_KEY, providers_json="[]")

    _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": default_minds_api_host()}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


def test_a_stored_url_matches_the_same_host_without_its_path(monkeypatch, pinged):
    """The stored value and the body differ only by the path.

    This is what comparing whole URL strings got wrong, independently of the
    MindsHub default: an operator who stores a private endpoint with a path
    gets its own host refused. ``_origin`` is the line under test here, so
    nothing in this deployment carries the bare host on its own.
    """
    _deployment(monkeypatch, minds_api_key=STORED_KEY, minds_url="https://minds.internal/v1")

    _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": "https://minds.internal"}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


def test_a_stored_card_url_matches_the_same_host_without_its_path(monkeypatch, pinged):
    # Same comparison, on the openai-compatible side, where no vendor default
    # is unioned in at all.
    _deployment(
        monkeypatch,
        minds_api_key=STORED_KEY,
        providers_json='[{"type": "openai-compatible", "baseUrl": "https://oc.internal/v1"}]',
    )

    _test_providers([{"type": "openai-compatible", "apiKey": "***", "baseUrl": "https://oc.internal"}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


def test_the_vendor_default_is_allowed_when_nothing_is_stored(monkeypatch, pinged):
    # A minds-cloud ping that omits mindsUrl goes to default_minds_api_host(),
    # so naming that host is a destination the caller already reaches.
    _deployment(monkeypatch, minds_api_key=STORED_KEY, minds_url="", providers_json="[]")

    _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": default_minds_api_host()}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


def test_one_refused_provider_does_not_blank_the_others(stored, pinged):
    """Both callers send every configured provider in one request.

    ``ping_provider`` already degrades per provider, so a refusal has to join
    its results rather than end the request. Raising instead blanked the
    anthropic, openai and gemini dots over one bad URL.
    """
    result = _test_providers(
        [
            {"type": "minds-cloud", "apiKey": "***", "mindsUrl": ATTACKER_URL},
            {"type": "anthropic", "apiKey": "***"},
        ]
    )

    assert result["providerStatus"] == {"minds-cloud": "fail", "anthropic": "ok"}
    assert [p["type"] for p in pinged] == ["anthropic"]


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
    _deployment(
        monkeypatch,
        minds_api_key=STORED_KEY,
        providers_json='[{"type": "openai-compatible", "baseUrl": "https://card.internal"}]',
    )

    _test_providers(
        [{"type": "openai-compatible", "apiKey": "***", "baseUrl": "https://card.internal/"}]
    )

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]


@pytest.mark.parametrize(
    "supplied",
    [
        "https://mdb.ai@attacker.example",   # userinfo: the real host is the attacker's
        "https://mdb.ai.attacker.example",   # suffix: a substring test would pass this
        "http://mdb.ai",                     # scheme downgrade
        "https://mdb.ai:8443",               # a different port is a different listener
    ],
)
def test_a_host_that_only_looks_like_the_stored_one_is_refused(stored, pinged, supplied):
    result = _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": supplied}])

    assert result["providerStatus"]["minds-cloud"] == "fail"
    assert pinged == []


def test_a_url_naming_no_host_is_refused(stored, pinged):
    # Present but unparseable leaves nothing to compare, so it cannot be
    # waved through the way an absent field is.
    result = _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": "not-a-url"}])

    assert result["providerStatus"]["minds-cloud"] == "fail"
    assert pinged == []


def test_a_non_string_url_is_refused_rather_than_raising(stored, pinged):
    # The body is `list[dict[str, Any]]`, so pydantic passes a list straight
    # through. Raising here would 500 the whole request.
    result = _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": [ATTACKER_URL]}])

    assert result["providerStatus"]["minds-cloud"] == "fail"
    assert pinged == []


def test_a_non_string_provider_type_still_answers(stored, pinged):
    # Unknown types already came back as "unknown provider type" from
    # ping_provider; the guard must not turn that into a 500.
    result = _test_providers([{"type": 123, "apiKey": ""}])

    assert result["providerStatus"] == {123: "ok"}


def test_a_card_with_a_null_type_does_not_crash(monkeypatch, pinged):
    # providers_json is a plain string setting with no shape validation on
    # write, so a null type is storable and `.get("type", "")` hands back the
    # None rather than the default.
    _deployment(
        monkeypatch,
        minds_api_key=STORED_KEY,
        minds_url=STORED_MINDS_URL,
        providers_json='[{"type": null, "baseUrl": "https://card.internal"}]',
    )

    _test_providers([{"type": "minds-cloud", "apiKey": "***", "mindsUrl": STORED_MINDS_URL}])

    assert [p["apiKey"] for p in pinged] == [STORED_KEY]
