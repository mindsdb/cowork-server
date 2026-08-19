"""Per-role model defaults delivered by MindsHub's catalog rather than compiled in.

Which model a new user's planning, coding and router roles start on is declared
in MindsHub's model policy, travels on ``/v1/models`` as a per-row ``default_for``
list, and is cached into the ``minds_role_defaults`` setting by the
recommended-models endpoint. Resolution then reads the cached map with no network
call in the turn path, exactly as it already reads the availability map.

The compiled table in ``app_settings`` is demoted, not retired. It is what
resolves before any fetch has been persisted, which is every fresh install on its
first message, and what resolves when the catalog cannot be reached. Both of
those are pinned here, because a broken remote layer would otherwise leave the
existing tests green.
"""
import asyncio
import json

from pydantic import SecretStr

from cowork.common.settings.app_settings import MODEL_ROLE_DEFAULTS
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.scoped import LOCAL_SCOPE
from _fakes import FakeRequest

# What the catalog declares today: every role starts on the one alias the monthly
# included allowance covers.
COMPILED = MODEL_ROLE_DEFAULTS["minds_cloud"]

# A catalog that moves the roles somewhere the compiled table does not name, so a
# passing assertion cannot be the compiled value by coincidence.
MOVED = json.dumps({"planning": "sonnet", "coding": "haiku", "router": "kimi"})
EVERYTHING_ENABLED = json.dumps(
    {"mindshub_air": True, "sonnet": True, "haiku": True, "kimi": True}
)


def _minds(**kw) -> UserSettings:
    return UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb_test"),
        **kw,
    )


# ── Resolution ────────────────────────────────────────────────────────

def test_the_declared_default_wins_over_the_compiled_table():
    """The whole point: the default moves by config, with no client release."""
    s = _minds(minds_role_defaults=MOVED, minds_model_enabled=EVERYTHING_ENABLED)

    assert s.planning_model == "sonnet"
    assert s.coding_model == "haiku"
    assert s.router_model == "kimi"
    assert s.resolved_planning_model == "sonnet"
    assert s.resolved_coding_model == "haiku"
    assert s.resolved_router_model == "kimi"


def test_nothing_persisted_resolves_to_the_compiled_table():
    """The genuinely-fresh-install case, and why the constants cannot be retired.

    Resolution is synchronous and offline by design, so a first message is sent
    before anything has ever fetched ``/v1/models``. Without this the remote layer
    could be broken end to end and every other test here would still pass.
    """
    s = _minds()

    assert s.planning_model == COMPILED["planning"]
    assert s.coding_model == COMPILED["coding"]
    assert s.router_model == COMPILED["router"]


def test_an_unreachable_catalog_resolves_to_the_compiled_table():
    """Same answer as a fresh install, reached a different way.

    A gateway that is down never refreshes the cached map, so the stored value
    stays whatever it was: empty on an install that has never reached one.
    """
    s = _minds(minds_role_defaults="{}", minds_model_enabled="{}")

    assert s.planning_model == COMPILED["planning"]
    assert s.resolved_router_model == COMPILED["router"]


def test_a_partially_declared_catalog_fills_only_what_it_names():
    """A role the catalog says nothing about keeps the compiled answer.

    The gateway-side gate requires all three, so this is the deploy-order case: a
    client meeting a catalog that declares some roles and not others must not lose
    the roles it was not told about.
    """
    s = _minds(
        minds_role_defaults=json.dumps({"planning": "sonnet"}),
        minds_model_enabled=EVERYTHING_ENABLED,
    )

    assert s.planning_model == "sonnet"
    assert s.coding_model == COMPILED["coding"]
    assert s.router_model == COMPILED["router"]


def test_an_explicit_pick_is_never_overwritten_by_a_moved_default():
    """The load-bearing negative case.

    A default fills an unset value and nothing else. A user who chose a model
    keeps it however the declared default later moves, which is what makes moving
    one a safe edit.
    """
    s = _minds(
        planning_model="opus",
        coding_model="opus",
        router_model="opus",
        minds_role_defaults=MOVED,
        minds_model_enabled=EVERYTHING_ENABLED,
    )

    assert s.planning_model == "opus"
    assert s.coding_model == "opus"
    assert s.router_model == "opus"
    assert s.resolved_planning_model == "opus"


def test_direct_providers_ignore_the_catalog_entirely():
    """MindsHub has nothing to say about a model it does not serve.

    The BYOK defaults are Anthropic/OpenAI/Gemini ids that are not in our catalog,
    so a declared default must not leak across and hand an Anthropic account a
    MindsHub alias.
    """
    s = UserSettings(
        planning_provider=Provider.ANTHROPIC,
        coding_provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("sk-ant-test"),
        minds_role_defaults=MOVED,
    )

    assert s.planning_model == MODEL_ROLE_DEFAULTS["anthropic"]["planning"]
    assert s.coding_model == MODEL_ROLE_DEFAULTS["anthropic"]["coding"]


def test_a_declared_default_still_loses_to_the_availability_map():
    """Declaring a default is not granting access to it.

    An alias the wallet cannot pay for is ``enabled: false``, and resolving onto
    it would deny every turn. The availability fallback runs on the declared value
    exactly as it ran on the compiled one.
    """
    s = _minds(
        minds_role_defaults=MOVED,
        minds_model_enabled=json.dumps({"mindshub_air": True, "sonnet": False}),
    )

    assert s.planning_model == "mindshub_air"


# ── Parsing the stored map ────────────────────────────────────────────

def test_a_corrupt_stored_map_falls_back_rather_than_raising():
    for stored in ("not json", "[]", "null", ""):
        s = _minds(minds_role_defaults=stored)
        assert s.planning_model == COMPILED["planning"], stored


def test_only_real_role_names_and_real_model_ids_are_read():
    """A value that is not a usable model id is worse than no value at all.

    The role is looked up by name and its value is handed to the gateway as a
    model id, so a number, a null or a blank would resolve the role onto something
    that cannot exist. The compiled fallback is strictly better than that.
    """
    s = _minds(
        minds_role_defaults=json.dumps(
            {"planning": "", "coding": None, "router": 7, "planing": "sonnet"}
        )
    )

    assert s.planning_model == COMPILED["planning"]
    assert s.coding_model == COMPILED["coding"]
    assert s.router_model == COMPILED["router"]


def test_the_stored_map_is_parsed_once_per_instance():
    s = _minds(minds_role_defaults=MOVED)

    first = s._minds_role_default_map()
    assert s._minds_role_default_map() is first


# ── Parsing the catalog listing ───────────────────────────────────────

def _rows(*rows):
    return {"data": list(rows)}


def _row(model_id, **extra):
    return {"id": model_id, "object": "model", **extra}


def _parse(monkeypatch, payload):
    """Run one /v1/models fetch against a stubbed transport."""
    from cowork.services import providers

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return payload

        @staticmethod
        def raise_for_status():
            return None

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **kw):
            return _Response()

    monkeypatch.setattr(providers.httpx, "AsyncClient", _Client)
    providers._minds_models_cache.clear()
    return asyncio.run(providers.fetch_minds_models("https://api.mindshub.ai", "mdb_test"))


def test_default_for_is_inverted_into_a_role_map(monkeypatch):
    listing = _parse(
        monkeypatch,
        _rows(
            _row("mindshub_air", default_for=["planning", "coding", "router"]),
            _row("sonnet", default_for=[]),
        ),
    )

    assert listing.role_defaults == {
        "planning": "mindshub_air",
        "coding": "mindshub_air",
        "router": "mindshub_air",
    }


def test_a_gateway_that_declares_nothing_yields_an_empty_map(monkeypatch):
    """Every gateway predating the field, and every plain OpenAI-compatible one.

    Empty is what leaves the compiled table in charge, so this is the deploy-order
    guarantee: a new client against an old gateway behaves exactly as before.
    """
    listing = _parse(monkeypatch, _rows(_row("mindshub_air"), _row("sonnet")))

    assert listing.role_defaults == {}


def test_a_role_we_do_not_serve_is_dropped(monkeypatch):
    """A console typo must not become a stored key nothing ever reads."""
    listing = _parse(
        monkeypatch,
        _rows(_row("mindshub_air", default_for=["planing", "coding", 7, None])),
    )

    assert listing.role_defaults == {"coding": "mindshub_air"}


def test_an_embedding_row_cannot_claim_a_role(monkeypatch):
    """Embeddings are filtered before anything is recorded, and must stay so.

    An embeddings model in a chat role errors every turn, so a catalog that
    mistakenly declared one has to lose the claim with the row.
    """
    listing = _parse(
        monkeypatch,
        _rows(
            _row("embed-small", embedding=True, default_for=["planning"]),
            _row("mindshub_air", default_for=["coding"]),
        ),
    )

    assert listing.role_defaults == {"coding": "mindshub_air"}


# ── Caching it, and telling the picker ────────────────────────────────

# The keys every endpoint test writes, so each one tears down exactly what it set
# and the next test starts from a genuinely fresh install.
_TOUCHED = ("minds_api_key", "minds_model_enabled", "minds_role_defaults")


def _stub_listing(monkeypatch, *, ids, enabled=None, role_defaults=None):
    from cowork.api.v1.endpoints import settings as endpoint
    from cowork.services.providers import MindsModelListing

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return MindsModelListing(ids, {}, enabled or {}, {}, {}, {}, role_defaults or {})

    monkeypatch.setattr(endpoint, "fetch_minds_models", fake_fetch)


class _Endpoint:
    """One open session with MindsHub configured, cleaned up on exit."""

    def __enter__(self):
        from cowork.db.session import get_open_session
        from cowork.services.settings import SettingService

        self.session = get_open_session()
        SettingService(self.session).upsert_setting("minds_api_key", "mdb_test")
        return self

    def call(self):
        from cowork.api.v1.endpoints.settings import recommended_models

        return asyncio.run(recommended_models(FakeRequest(), self.session, LOCAL_SCOPE))

    def stored(self, key):
        from cowork.services.settings import SettingService

        return getattr(SettingService(self.session).load(), key)

    def __exit__(self, *exc):
        from cowork.services.settings import SettingService

        service = SettingService(self.session)
        for key in _TOUCHED:
            try:
                service.delete_setting(key)
            except ValueError:
                pass
        self.session.close()
        return False


def test_the_endpoint_caches_the_declared_defaults(monkeypatch):
    _stub_listing(
        monkeypatch,
        ids=["mindshub_air", "sonnet"],
        enabled={"mindshub_air": True, "sonnet": True},
        role_defaults={"planning": "sonnet", "coding": "sonnet", "router": "sonnet"},
    )

    with _Endpoint() as endpoint:
        endpoint.call()

        assert json.loads(endpoint.stored("minds_role_defaults")) == {
            "planning": "sonnet",
            "coding": "sonnet",
            "router": "sonnet",
        }


def test_defaults_are_cached_even_when_the_gateway_sends_no_enabled_flags(monkeypatch):
    """The nesting this deliberately avoids.

    The availability write is gated on a non-empty ``enabled`` map. A gateway that
    lists ids without flags produces an empty one while still publishing perfectly
    good role defaults, so folding this write into that block would drop them for
    exactly that gateway.
    """
    _stub_listing(
        monkeypatch,
        ids=["mindshub_air"],
        enabled={},
        role_defaults={"planning": "mindshub_air"},
    )

    with _Endpoint() as endpoint:
        endpoint.call()

        assert json.loads(endpoint.stored("minds_role_defaults")) == {"planning": "mindshub_air"}


def test_an_empty_declaration_never_wipes_a_good_cache(monkeypatch):
    """Writing {} would drop every role back to a value only a release can change."""
    with _Endpoint() as endpoint:
        _stub_listing(monkeypatch, ids=["mindshub_air"], role_defaults={"planning": "sonnet"})
        endpoint.call()

        _stub_listing(monkeypatch, ids=["mindshub_air"], role_defaults={})
        endpoint.call()

        assert json.loads(endpoint.stored("minds_role_defaults")) == {"planning": "sonnet"}


def test_the_picker_is_told_what_the_server_will_resolve(monkeypatch):
    """The picker and resolution must not disagree, asserted rather than assumed.

    The picker reads ``recommendedPair`` to show which model a role starts on, and
    writes that value back as an explicit pin when the user saves. Left on the
    static table it would show the compiled model while turns ran the declared
    one, and saving would then pin the wrong one.
    """
    declared = {"planning": "sonnet", "coding": "haiku", "router": "kimi"}
    _stub_listing(
        monkeypatch,
        ids=["mindshub_air", "sonnet", "haiku", "kimi"],
        enabled={"mindshub_air": True, "sonnet": True, "haiku": True, "kimi": True},
        role_defaults=declared,
    )

    with _Endpoint() as endpoint:
        payload = endpoint.call()

    assert payload["recommendedPair"]["minds-cloud"] == ["sonnet", "haiku", "kimi"]

    resolved = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb_test"),
        minds_role_defaults=json.dumps(declared),
        minds_model_enabled=EVERYTHING_ENABLED,
    )
    assert payload["recommendedPair"]["minds-cloud"] == [
        resolved.resolved_planning_model,
        resolved.resolved_coding_model,
        resolved.resolved_router_model,
    ]


def test_a_role_the_catalog_omits_keeps_the_compiled_slot_in_the_pair(monkeypatch):
    """The pair stays a full triple, so an older client indexing it still works."""
    _stub_listing(
        monkeypatch,
        ids=["mindshub_air", "sonnet"],
        role_defaults={"planning": "sonnet"},
    )

    with _Endpoint() as endpoint:
        payload = endpoint.call()

    assert payload["recommendedPair"]["minds-cloud"] == [
        "sonnet",
        COMPILED["coding"],
        COMPILED["router"],
    ]
