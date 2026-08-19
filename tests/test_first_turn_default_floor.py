"""First-turn model default must be affordable by construction (ENG-748).

A brand-new org can send its first message *before* the renderer has ever
called ``/settings/recommended-models``, so the availability map
(``minds_model_enabled``) is still empty at model-resolution time. ENG-748
tracked the resulting gap: with an empty map the resolver handed out the paid
canonical default (``sonnet``) and the empty free-tier wallet 403'd on turn one.

The fix is a FLOOR, not a cache-warm. ``role_defaults`` (app_settings) makes the
org-mode minds-cloud default the free-allowance model (``mindshub_air``), which
the free monthly allowance always covers — so a cold boot is correct regardless
of whether the availability map has been fetched yet. cowork-server#341 (warm
the map on the turn path) was closed as superseded: it added yet another refresh
trigger for a cache that no longer gates correctness, reintroducing the
multiple-writers-of-availability shape these defaults were meant to retire.

These tests pin the floor so a future refactor can't quietly bring back the
paid-default-on-first-turn 403 — the empty-map path is what a real first turn
hits, not the funded happy path. In org mode the free model is the default for
*every* tenant (funded or not): premium is opt-in via an explicit pick, which is
honored when the map says it's enabled. Desktop (local mode) is deliberately
unaffected: it keeps the premium canonical defaults, gated on a real stored key.

Companion coverage: ``test_enabled_aware_model_defaults`` (local-mode map
resolution + endpoint cache writes) and ``test_org_mode_readiness`` (org-mode
readiness with nothing stored).
"""
import json

import pytest

from cowork.common.settings import user_settings as us
from cowork.common.settings.user_settings import Provider, UserSettings


# UserSettings reads bare env vars and the .env chain, so a developer's exported
# key (or a stale .env line) would otherwise steer the readiness resolver and
# make these pass locally / fail in CI. Clear them, then drive tenancy_mode via
# the real settings + cache clear so every module's get_app_settings() agrees
# (user_settings, app_settings.role_defaults, providers).
_KEY_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "MINDS_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "PLANNING_PROVIDER",
    "CODING_PROVIDER",
)


def _mode(monkeypatch, mode: str) -> None:
    for name in _KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    us.get_app_settings.cache_clear()


def _settings(**kw) -> UserSettings:
    # No .env chain: the file would reintroduce the keys _mode just cleared.
    return UserSettings(_env_file=None, **kw)


@pytest.fixture
def org_mode(monkeypatch):
    _mode(monkeypatch, "org")
    yield
    us.get_app_settings.cache_clear()


@pytest.fixture
def local_mode(monkeypatch):
    _mode(monkeypatch, "local")
    yield
    us.get_app_settings.cache_clear()


# A funded org's map: the whole catalog listed, paid models enabled.
PAID_MAP = json.dumps({"mindshub_air": True, "sonnet": True, "haiku": True, "kimi": True})


# ── The regression: cold-boot empty map must floor to the free model ──

def test_org_empty_map_floors_every_role_to_the_free_model(org_mode):
    # The exact first-turn state: the field default "{}" — no /recommended-models
    # fetch has landed yet. Every role must resolve to the free-allowance model,
    # never the paid canonical, or turn one 403s.
    s = _settings(minds_model_enabled="{}")
    assert s.resolved_planning_model == "mindshub_air"
    assert s.resolved_coding_model == "mindshub_air"
    assert s.resolved_router_model == "mindshub_air"


def test_org_free_map_floors_planning_to_the_free_model(org_mode):
    # The realistic free-tier gateway shape: whole catalog listed, paid models
    # disabled, the free model enabled. The default still floors to the free
    # model — matching the empty-map case above, so warming the map buys no
    # correctness on the planning default.
    free_map = json.dumps(
        {"mindshub_air": True, "sonnet": False, "haiku": False, "kimi": False}
    )
    s = _settings(minds_model_enabled=free_map)
    assert s.resolved_planning_model == "mindshub_air"


# ── The floor is a default, not a cap ─────────────────────────────────

def test_org_explicit_paid_pick_is_honored_when_enabled(org_mode):
    # role_defaults floors the *default*; it does not cap choice. A funded org
    # that explicitly picks a premium model (map says enabled) keeps it.
    s = _settings(minds_model_enabled=PAID_MAP, planning_model="sonnet")
    assert s.resolved_planning_model == "sonnet"


# ── The floor is org-only ─────────────────────────────────────────────

def test_local_mode_keeps_paid_canonical_on_empty_map(local_mode):
    # Desktop resolves against a real stored key and keeps the premium
    # canonical defaults; the free-first floor is an org-mode override only.
    s = _settings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        router_provider=Provider.MINDS_CLOUD,
        minds_api_key="mdb_test",
        minds_model_enabled="{}",
    )
    assert s.resolved_planning_model == "sonnet"
    assert s.resolved_coding_model == "haiku"
    assert s.resolved_router_model == "kimi"
