"""Tier-aware model defaults (ENG-597).

MindsHub gates models per plan tier: a wallet that cannot pay gets the paid
aliases as ``enabled: false`` from ``/v1/models``. Every minds-cloud role now
defaults to the free model, so the canonical default is affordable by
construction; the availability map matters for the cases where it is not,
namely a stored pin and a default that the catalog has stopped serving. These
tests pin the fix:

- ``UserSettings`` resolves its planning/coding defaults against the cached
  availability map (``minds_model_enabled``), falling back to the first
  enabled model — and ONLY when the user hasn't explicitly picked a model.
- The readiness resolver's provider-switch branch (``_resolved_model``) is
  tier-aware too, so switching a keyless account onto minds-cloud never lands
  on a locked model.
- The recommended-models endpoint caches the live map (and never wipes a
  previously-good cache on a failed fetch).

The map preserves /v1/models order; the gateway lists the tier's baseline
model first, so "first enabled" is the intended fallback.
"""
import asyncio
import json

from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.scoped import LOCAL_SCOPE
from _fakes import FakeRequest

# The gateway's free-tier registry shape: whole catalog listed, paid models
# disabled, the baseline model first and enabled.
FREE_MAP = json.dumps(
    {"mindshub_air": True, "kimi": False, "gpt-codex": False, "haiku": False, "sonnet": False}
)
PAID_MAP = json.dumps(
    {"mindshub_air": True, "kimi": True, "gpt-codex": True, "haiku": True, "sonnet": True}
)


def _minds(**kw) -> UserSettings:
    return UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb_test"),
        **kw,
    )


# ── Default resolution (apply_model_defaults) ─────────────────────────

def test_free_tier_map_keeps_the_canonical_defaults():
    # The canonical default IS the free model, so nothing has to fall back.
    s = _minds(minds_model_enabled=FREE_MAP)
    assert s.planning_model == "mindshub_air"
    assert s.coding_model == "mindshub_air"
    assert s.router_model == "mindshub_air"


def test_paid_tier_gets_the_same_defaults_as_everyone_else():
    # A funded wallet does not change what an unset model resolves to. Paying
    # for a better model is an explicit pick in the picker, not a default.
    s = _minds(minds_model_enabled=PAID_MAP)
    assert s.planning_model == "mindshub_air"
    assert s.coding_model == "mindshub_air"
    assert s.router_model == "mindshub_air"


def test_absent_map_keeps_canonical_defaults():
    # No cached map (fresh install, fetch never ran). This is the state a
    # brand-new account is in on its first turn, and the reason the default
    # itself has to be affordable rather than corrected by the map.
    s = _minds()
    assert s.planning_model == "mindshub_air"
    assert s.coding_model == "mindshub_air"


def test_explicit_model_choice_is_never_rewritten():
    # A user-picked model stays put even when locked — that case is the
    # error-card lane (ENG-598), not a silent switch.
    s = _minds(minds_model_enabled=FREE_MAP, planning_model="sonnet")
    assert s.planning_model == "sonnet"


def test_all_disabled_map_keeps_canonical_default():
    # Degenerate metadata (nothing enabled) must not invent a model.
    s = _minds(minds_model_enabled=json.dumps({"kimi": False, "gpt-codex": False}))
    assert s.planning_model == "mindshub_air"


def test_default_missing_from_nonempty_map_falls_back_to_first_enabled():
    # A non-empty map always carries every alias the catalog currently
    # serves, so missing means gone (renamed/retired), same as disabled.
    # The map here deliberately omits the canonical default, which is the only
    # way to reach this branch now that the default is the free model.
    s = _minds(minds_model_enabled=json.dumps({"sonnet": True}))
    assert s.planning_model == "sonnet"


def test_invalid_map_json_degrades_to_canonical_default():
    s = _minds(minds_model_enabled="not json")
    assert s.planning_model == "mindshub_air"


def test_map_order_decides_the_fallback():
    # First enabled entry in map order wins (mirrors /v1/models ordering).
    # The canonical default is disabled here, which is the drained-wallet
    # state: even the free model stops being callable, so ordering decides.
    s = _minds(
        minds_model_enabled=json.dumps(
            {"mindshub_air": False, "sonnet": True, "kimi": True}
        )
    )
    assert s.planning_model == "sonnet"


def test_direct_providers_ignore_the_minds_map():
    # The map is minds-cloud-only; a BYOK anthropic default is untouched even
    # with a cached map lying around.
    s = UserSettings(
        planning_provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("sk-ant-test"),
        minds_model_enabled=json.dumps({"claude-sonnet-4-6": False}),
    )
    assert s.planning_model == "claude-sonnet-4-6"


# ── Provider-switch resolution (_resolved_model) ──────────────────────

def test_provider_switch_onto_minds_is_tier_aware():
    # planning_provider=anthropic with no anthropic key + a minds key →
    # resolver switches to minds-cloud; the model it hands out must respect
    # the tier map (not the locked canonical default).
    s = UserSettings(
        planning_provider=Provider.ANTHROPIC,
        minds_api_key=SecretStr("mdb_test"),
        minds_model_enabled=FREE_MAP,
    )
    assert s.resolved_planning_provider == Provider.MINDS_CLOUD
    assert s.resolved_planning_model == "mindshub_air"
    assert s.resolved_coding_model == "mindshub_air"


def test_provider_switch_onto_minds_paid_keeps_canonical():
    # A funded wallet resolves to the same canonical default as an empty one:
    # the switch picks the default, and the default is the free model.
    s = UserSettings(
        planning_provider=Provider.ANTHROPIC,
        minds_api_key=SecretStr("mdb_test"),
        minds_model_enabled=PAID_MAP,
    )
    assert s.resolved_planning_model == "mindshub_air"


# ── Wallet-aware resolution, all three roles (ENG-1632, ENG-1632 follow-up) ──
#
# Originally aux-roles-only (coding = completion verifier + scratchpad,
# router = gating + summarization): they're invisible in default mode, so a
# wallet-locked pin there 402s on every turn with no way for the user to see
# or fix it. Resolution — never the stored row — falls back to a
# strictly-enabled model. Planning was later given the same fallback: a
# MindsHub sign-in rewrites planning_provider without touching the paired
# planning_model, so a pre-sign-in BYOK pick survives and 404s every turn,
# never reaching the picker's ENG-1248 "Needs credits" lane at all (that
# lane is for a real, currently-unaffordable MindsHub model — a separate,
# still-visible per-model tag, unaffected by this fallback).

# Whole catalog listed, everything paid locked — the ENG-1632 cohort's map.
LOCKED_MAP = json.dumps(
    {"mindshub_air": True, "sonnet": False, "opus": False, "haiku": False, "kimi": False}
)


def _pinned(**kw) -> UserSettings:
    return _minds(router_provider=Provider.MINDS_CLOUD, **kw)


def test_locked_coding_pin_resolves_to_first_enabled():
    s = _pinned(minds_model_enabled=LOCKED_MAP, coding_model="haiku")
    assert s.coding_model == "haiku"  # the stored row is never rewritten
    assert s.resolved_coding_model == "mindshub_air"


def test_locked_router_pin_resolves_to_first_enabled():
    s = _pinned(minds_model_enabled=LOCKED_MAP, router_model="kimi")
    assert s.router_model == "kimi"
    assert s.resolved_router_model == "mindshub_air"


def test_locked_planning_pin_resolves_to_first_enabled():
    # ENG-1632 follow-up: planning now gets the same self-healing fallback
    # as coding/router. MindsHub SSO sign-in rewrites planning_provider to
    # minds_cloud without touching the paired planning_model, so a stale
    # pick from a prior provider would otherwise 404 the gateway on every
    # single turn — never reaching the picker's "Needs credits" lane at all.
    s = _pinned(minds_model_enabled=LOCKED_MAP, planning_model="sonnet")
    assert s.planning_model == "sonnet"  # the stored row is never rewritten
    assert s.resolved_planning_model == "mindshub_air"


def test_stale_byok_planning_pin_survives_a_provider_switch_onto_minds():
    # The exact reported bug: signed out with e.g. Anthropic configured for
    # planning (a real, non-MindsHub id), then MindsHub sign-in flips
    # planning_provider to minds_cloud without touching planning_model. The
    # foreign id can never appear in the MindsHub enabled map (it isn't a
    # MindsHub model), so it must fall back rather than 404 every turn.
    s = _pinned(
        minds_model_enabled=json.dumps({"mindshub_air": True}),
        planning_model="claude-opus-4-8",
    )
    assert s.planning_model == "claude-opus-4-8"
    assert s.resolved_planning_model == "mindshub_air"


def test_latest_prefixed_pin_is_probed_bare():
    # /v1/models ids are always bare; login-era pins carry "latest:". The map
    # probe must strip it or those pins silently escape the fallback.
    s = _pinned(minds_model_enabled=LOCKED_MAP, coding_model="latest:haiku")
    assert s.resolved_coding_model == "mindshub_air"


def test_funded_pin_is_kept():
    # The map is written from the full catalogue, so a funded wallet lists the
    # pinned ids as enabled — all three roles stay put.
    funded = json.dumps(
        {"mindshub_air": True, "sonnet": True, "haiku": True, "kimi": True}
    )
    s = _pinned(
        minds_model_enabled=funded, planning_model="sonnet", coding_model="haiku", router_model="kimi",
    )
    assert s.resolved_planning_model == "sonnet"
    assert s.resolved_coding_model == "haiku"
    assert s.resolved_router_model == "kimi"


def test_fully_drained_map_keeps_the_pin():
    # Nothing enabled (drained wallet AND spent allowance) → no fallback
    # exists; keep the stored value and let anton's verifier deny quietly.
    all_off = json.dumps(
        {"mindshub_air": False, "sonnet": False, "haiku": False, "kimi": False}
    )
    s = _pinned(minds_model_enabled=all_off, coding_model="haiku")
    assert s.resolved_coding_model == "haiku"


def test_absent_map_keeps_the_pin():
    s = _pinned(coding_model="haiku")
    assert s.resolved_coding_model == "haiku"


def test_pin_absent_from_nonempty_map_is_treated_as_not_served():
    # The opposite of _enabled_aware_default's absent-means-available rule,
    # deliberately: that rule probes OUR canonical default (absence = older
    # gateway), this one probes a USER-STORED id that can be anything. The map
    # is written from the full /v1/models catalogue, so an id it doesn't carry
    # is not served and 404s every aux call — the drpconcepcion cohort stored
    # a Gemini id against minds-cloud, which the map can never mark false.
    s = _pinned(
        minds_model_enabled=json.dumps({"mindshub_air": True}),
        coding_model="gemini-3.6-flash",
    )
    assert s.resolved_coding_model == "mindshub_air"


def test_byok_aux_pin_ignores_the_minds_map():
    s = UserSettings(
        planning_provider=Provider.ANTHROPIC,
        coding_provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("sk-ant-test"),
        minds_model_enabled=json.dumps({"claude-haiku-4-5-20251001": False}),
        coding_model="claude-haiku-4-5-20251001",
        planning_model="claude-haiku-4-5-20251001",
    )
    # wallet_aware only ever probes when the RESOLVED provider is
    # minds-cloud — a real BYOK provider's pick is never second-guessed
    # against MindsHub's map, planning included.
    assert s.resolved_coding_model == "claude-haiku-4-5-20251001"
    assert s.resolved_planning_model == "claude-haiku-4-5-20251001"


# ── Endpoint cache write (recommended-models) ─────────────────────────

def _delete_settings(session, *keys: str) -> None:
    from cowork.services.settings import SettingService

    service = SettingService(session)
    for key in keys:
        try:
            service.delete_setting(key)
        except ValueError:
            pass


def _set_settings(session, **values: str) -> None:
    from cowork.services.settings import SettingService

    service = SettingService(session)
    for key, value in values.items():
        service.upsert_setting(key, value)


def test_recommended_models_caches_enabled_map(monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        from cowork.services.providers import MindsModelListing

        return MindsModelListing(
            ["mindshub_air", "sonnet"], {}, {"mindshub_air": True, "sonnet": False}, {}, {}, {}, {}
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)
    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_test")
        asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))
        cached = SettingService(session).load().minds_model_enabled
        assert json.loads(cached) == {"mindshub_air": True, "sonnet": False}
    finally:
        _delete_settings(session, "minds_api_key", "minds_model_enabled")
        session.close()


def test_recommended_models_failed_fetch_preserves_cache(monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        from cowork.services.providers import _empty_listing

        return _empty_listing()  # fetch failed

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)
    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_test", minds_model_enabled=FREE_MAP)
        asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))
        cached = SettingService(session).load().minds_model_enabled
        assert json.loads(cached) == json.loads(FREE_MAP)  # untouched
    finally:
        _delete_settings(session, "minds_api_key", "minds_model_enabled")
        session.close()


def test_enabled_map_accepts_only_real_bools():
    # bool("false") is True, so a stringy value must be dropped rather than
    # misread as enabled. A dropped entry is absent, which the consumers treat
    # as "available" — the map's own convention, so this can't over-lock.
    s = _minds(minds_model_enabled=json.dumps({"mindshub_air": True, "opus": "false", "gpt": 1}))
    assert s._minds_enabled_map() == {"mindshub_air": True}
