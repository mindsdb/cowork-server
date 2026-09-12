"""The usage read: free monthly tokens, wallet balance, auto top up.

Outbound HTTP is stubbed at the shared `get_auth_json`, same as the workspace tests,
so each case seeds the auth bodies and asserts the one view the desktop gets.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import hub_usage as ep
from cowork.api.v1.router import api_router
from cowork.db.scoped import TenantScope
from cowork.principal import HEADER_HUB_CREDENTIAL, Principal, get_principal
from cowork.services import hub_usage as svc

PATH = "/api/v1/hub/usage/"
PRINCIPAL = Principal(user_id="user-a", org_id="org-a")


def _client(principal: Principal | None) -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_principal] = lambda: principal
    return TestClient(app)

ENTITLEMENTS = {
    "included_percent_remaining": 12.4,
    "next_refresh_at": "2026-09-11T00:00:00Z",
    "is_billing_owner": True,
    "feature_gates": {},
}

WALLET = {
    "balance_usd": "8.42",
    "can_consume": True,
    "has_topped_up": True,
    "auto_recharge": {
        "enabled": True,
        "threshold_usd": "5.00",
        "recharge_to_usd": "20.00",
        "status": "ok",
        "last_charge_failed": False,
        "pending_action": False,
        "cap_reached": False,
    },
    "payment_method": {"brand": "visa", "last4": "4242"},
    "low_balance_alert": "low",
    "alert_dismissed": False,
}


SUMMARY = {
    "scope": "organization",
    "group_by": "model",
    "results": [],
    "totals": {"usage": {}, "cost": {"total_usd": "0.02", "unattributed_usd": "0"}},
    "range": {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"},
    "meta": {"generated_at": "2026-08-28T00:00:00Z", "cost_source": "metronome_invoice"},
}


class FakeRequest:
    def __init__(self, bearer: str = "jwt-abc") -> None:
        self.headers = {HEADER_HUB_CREDENTIAL: f"Bearer {bearer}"} if bearer else {}


@pytest.fixture(autouse=True)
def _clean_cache():
    svc.reset_cache_for_tests()
    yield
    svc.reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def calls(monkeypatch):
    asked: list[str] = []
    answers: dict[str, object] = {}

    async def _fake(path: str, bearer_token: str):
        asked.append(path)
        return answers.get(path)

    monkeypatch.setattr(svc, "get_auth_json", _fake)
    return type("Calls", (), {"asked": asked, "answers": answers})()


def _scope() -> TenantScope:
    return TenantScope(org_mode=True, org_id="org-a", user_id="user-a")


def _fetch(bearer: str = "jwt-abc", user_id: str = "user-a", org_id: str = "org-a"):
    return asyncio.run(svc.fetch_hub_usage(bearer_token=bearer, org_id=org_id, user_id=user_id))


def test_no_bearer_is_unreachable_and_asks_nothing(calls):
    view = _fetch(bearer="")
    assert view.reachable is False
    assert calls.asked == []


def test_both_reads_land_in_one_view(calls):
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS
    calls.answers[svc.WALLET_PATH] = WALLET

    view = _fetch()

    assert view.reachable is True
    assert view.is_billing_owner is True
    assert view.free_tokens.percent_remaining == 12.4
    assert view.free_tokens.remaining == 12.4
    assert view.free_tokens.limit == 100
    assert view.free_tokens.resets_at == "2026-09-11T00:00:00Z"
    assert view.balance.usd == 8.42
    assert view.balance.alert == "low"
    assert view.balance.has_topped_up is True
    assert view.auto_top_up.enabled is True
    assert view.auto_top_up.threshold_usd == 5.0
    assert view.auto_top_up.recharge_to_usd == 20.0
    assert view.auto_top_up.status == "ok"
    assert sorted(calls.asked) == sorted([svc.ENTITLEMENTS_PATH, svc.WALLET_PATH, svc.USAGE_SUMMARY_PATH])


def test_wire_shape_is_camel_case(calls):
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS
    calls.answers[svc.WALLET_PATH] = WALLET

    body = _fetch().model_dump(by_alias=True)

    assert body["freeTokens"]["resetsAt"] == "2026-09-11T00:00:00Z"
    assert body["autoTopUp"]["rechargeToUsd"] == 20.0
    assert body["isBillingOwner"] is True


def test_a_wallet_the_caller_cannot_see_still_leaves_free_tokens(calls):
    """Starter-tier orgs get 4xx on /wallet/; the allowance must still render."""
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS

    view = _fetch()

    assert view.reachable is True
    assert view.free_tokens.percent_remaining == 12.4
    assert view.balance is None
    assert view.auto_top_up is None


def test_both_reads_failing_is_unreachable(calls):
    view = _fetch()
    assert view.reachable is False
    assert view.free_tokens is None
    assert view.balance is None


def test_auto_top_up_status_falls_back_to_the_flags(calls):
    """Auth versions without `status` still report a failed charge."""
    wallet = {**WALLET, "auto_recharge": {**WALLET["auto_recharge"], "status": None, "last_charge_failed": True}}
    calls.answers[svc.WALLET_PATH] = wallet

    assert _fetch().auto_top_up.status == "payment_failed"


def test_the_proportion_is_reported_out_of_one_hundred(calls):
    """Auth publishes no allowance size, so the compatibility triple is a ratio.

    A desktop build that predates ``percentRemaining`` divides ``remaining`` by
    ``limit``; out of 100 that lands on the same figure the percentage carries.
    """
    calls.answers[svc.ENTITLEMENTS_PATH] = {"included_percent_remaining": 70.0}

    tokens = _fetch().free_tokens

    assert tokens.percent_remaining == 70.0
    assert tokens.limit == 100
    assert tokens.used == 30
    assert tokens.remaining == 70


def test_a_fractional_percentage_is_not_rounded_to_exhausted(calls):
    """0.4% of the allowance is still a usable turn for a cache-heavy caller.

    Rounding it to zero would draw the exhausted state for an account that can
    still work, which is the failure the percentage was introduced to prevent.
    """
    calls.answers[svc.ENTITLEMENTS_PATH] = {"included_percent_remaining": 0.4}

    tokens = _fetch().free_tokens

    assert tokens.remaining == 0.4
    assert tokens.remaining > 0


def test_the_deprecated_counters_are_ignored(calls):
    """The old object is no longer read, whatever it happens to carry.

    It used to hold the allowance size. Dividing by a withdrawn limit reads as
    0% used and draws a full untouched bar for an account with nothing left.
    """
    calls.answers[svc.ENTITLEMENTS_PATH] = {
        "included_percent_remaining": 25.0,
        "included_tokens": {"limit": 6_500_000, "used": 1_000_000, "remaining": 5_500_000},
    }

    tokens = _fetch().free_tokens

    assert tokens.percent_remaining == 25.0
    assert tokens.limit == 100


def test_a_null_percentage_is_unlimited_not_no_grant(calls):
    # Auth's wire shape for an unlimited allowance: the percentage is null,
    # because there is nothing to count down.
    calls.answers[svc.ENTITLEMENTS_PATH] = {"included_percent_remaining": None}
    tokens = _fetch().free_tokens
    assert tokens.percent_remaining is None
    assert tokens.limit == -1
    assert tokens.remaining == -1


def test_a_server_without_the_percentage_reports_no_allowance(calls):
    """Nothing honest is left to report, so report nothing rather than guess.

    The absolute this server would have sent has been withdrawn. Guessing
    "uncapped" would tell a caller with an empty wallet that Air can carry the
    task while every turn fails.
    """
    calls.answers[svc.ENTITLEMENTS_PATH] = {"included_tokens": {"limit": 100, "used": 30}}

    assert _fetch().free_tokens is None


def test_an_ineligible_org_stays_no_grant(calls):
    """An org auth reports as not free-grant-eligible: limit 0, and 0 it stays.

    The one state the percentage cannot express on its own, because an
    exhausted allowance and an absent one both read 0%. Were it ever to read as
    uncapped, a caller with an empty wallet would be told MindsHub Air can carry
    the task and every turn would fail instead.
    """
    calls.answers[svc.ENTITLEMENTS_PATH] = {
        "included_percent_remaining": 0.0,
        "free_grant_eligible": False,
    }

    tokens = _fetch().free_tokens

    assert tokens.limit == 0
    assert tokens.remaining == 0


def test_the_cache_is_per_caller_not_per_org(calls):
    """Two people in one org must not see each other's allowance or owner flag."""
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS
    calls.answers[svc.WALLET_PATH] = WALLET

    _fetch(user_id="user-a")
    calls.answers[svc.ENTITLEMENTS_PATH] = {**ENTITLEMENTS, "is_billing_owner": False}
    view_b = _fetch(user_id="user-b")

    assert view_b.is_billing_owner is False
    assert calls.asked.count(svc.ENTITLEMENTS_PATH) == 2


def test_a_desktop_account_switch_does_not_reuse_the_previous_usage(calls):
    """The key has to include the CREDENTIAL, not just the identity.

    Outside org mode `scope_from_principal` returns LOCAL_SCOPE, so `org_id` and
    `user_id` are both empty for every request on a desktop install, and an
    identity-only key collapses to one shared entry. Sign out, sign in as another
    MindsHub account, and the first account's balance and owner flag would be
    served for the rest of the TTL.
    """
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS
    calls.answers[svc.WALLET_PATH] = WALLET
    first = _fetch(bearer="jwt-first-account", org_id="", user_id="")
    assert first.balance.usd == 8.42
    assert first.is_billing_owner is True

    # Same process, same (empty) scope, a different MindsHub session.
    calls.answers[svc.ENTITLEMENTS_PATH] = {**ENTITLEMENTS, "is_billing_owner": False}
    calls.answers[svc.WALLET_PATH] = {**WALLET, "balance_usd": "0.01"}
    second = _fetch(bearer="jwt-second-account", org_id="", user_id="")

    assert second.balance.usd == 0.01
    assert second.is_billing_owner is False
    assert calls.asked.count(svc.ENTITLEMENTS_PATH) == 2


def test_expired_entries_are_swept_rather_than_held_for_the_process_lifetime(calls):
    """Nothing re-reads a departed caller's key, so nothing would ever drop it."""
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS
    _fetch(bearer="jwt-someone-who-leaves", org_id="", user_id="")
    assert len(svc._cache) == 1

    stale = {k: (stamped - (svc._MAX_TTL_S + 1), v) for k, (stamped, v) in svc._cache.items()}
    svc._cache.clear()
    svc._cache.update(stale)

    _fetch(bearer="jwt-somebody-else", org_id="", user_id="")

    assert len(svc._cache) == 1, "the departed caller's entry was never dropped"


def test_an_uncapped_grant_reports_no_countdown(calls):
    calls.answers[svc.ENTITLEMENTS_PATH] = {"included_percent_remaining": None}
    assert _fetch().free_tokens.remaining == -1


def test_credit_spend_needs_a_known_cost_source(calls):
    """Only the invoice source is a real number; an absent field (older auth) is accepted."""
    calls.answers[svc.WALLET_PATH] = WALLET
    calls.answers[svc.USAGE_SUMMARY_PATH] = {**SUMMARY, "meta": {"cost_source": "estimate"}}
    assert _fetch().credit_spend is None

    svc.reset_cache_for_tests()
    calls.answers[svc.USAGE_SUMMARY_PATH] = {**SUMMARY, "meta": {}}
    assert _fetch().credit_spend.usd == 0.02


def test_a_successful_read_is_cached(calls):
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS
    calls.answers[svc.WALLET_PATH] = WALLET

    _fetch()
    _fetch()

    assert len(calls.asked) == 3


def test_credit_spend_comes_from_the_usage_summary(calls):
    calls.answers[svc.WALLET_PATH] = WALLET
    calls.answers[svc.USAGE_SUMMARY_PATH] = SUMMARY

    spend = _fetch().credit_spend

    assert spend.usd == 0.02
    assert spend.period_start == "2026-08-01T00:00:00Z"
    assert spend.period_end == "2026-09-01T00:00:00Z"


def test_an_unknown_cost_is_not_reported_as_zero(calls):
    calls.answers[svc.WALLET_PATH] = WALLET
    calls.answers[svc.USAGE_SUMMARY_PATH] = {**SUMMARY, "meta": {"cost_source": "unavailable"}}

    assert _fetch().credit_spend is None


def test_credit_spend_falls_back_to_the_wallet_block(calls):
    calls.answers[svc.WALLET_PATH] = {
        **WALLET,
        "credit_spend": {"amount_usd": "1.50", "period_start": "2026-08-01", "period_end": "2026-09-01"},
    }

    spend = _fetch().credit_spend

    assert spend.usd == 1.5
    assert spend.period_start == "2026-08-01"


def test_the_route_answers_the_same_view_the_service_does(calls):
    calls.answers[svc.ENTITLEMENTS_PATH] = ENTITLEMENTS
    calls.answers[svc.WALLET_PATH] = WALLET

    view = asyncio.run(ep.get_hub_usage(FakeRequest(), _scope()))

    assert view.reachable is True
    assert view.balance.usd == 8.42


# ── permission wiring (ENG-2094): the bare-function tests above never touch
# FastAPI's dependency graph, so they can't prove AuthenticatedInOrgMode is
# actually declared on the route — these go through the real router instead.


def test_route_requires_identity_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")

    resp = _client(principal=None).get(PATH)

    assert resp.status_code == 401


def test_route_allows_an_authenticated_member_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")

    async def _fake(path, bearer_token):
        return {svc.ENTITLEMENTS_PATH: ENTITLEMENTS, svc.WALLET_PATH: WALLET}.get(path)

    monkeypatch.setattr(svc, "get_auth_json", _fake)

    resp = _client(principal=PRINCIPAL).get(
        PATH, headers={HEADER_HUB_CREDENTIAL: "Bearer jwt-abc"}
    )

    assert resp.status_code == 200
    assert resp.json()["balance"]["usd"] == 8.42


def test_route_is_unchanged_in_local_mode_with_no_principal(monkeypatch):
    # tenancy_mode defaults to "local" — no COWORK_TENANCY_MODE set.
    async def _fake(path, bearer_token):
        return {svc.ENTITLEMENTS_PATH: ENTITLEMENTS, svc.WALLET_PATH: WALLET}.get(path)

    monkeypatch.setattr(svc, "get_auth_json", _fake)

    resp = _client(principal=None).get(
        PATH, headers={HEADER_HUB_CREDENTIAL: "Bearer jwt-abc"}
    )

    assert resp.status_code == 200
    assert resp.json()["balance"]["usd"] == 8.42
