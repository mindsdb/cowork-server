"""The usage read: free monthly tokens, wallet balance, auto top up.

Outbound HTTP is stubbed at the shared `_get_json`, same as the workspace tests,
so each case seeds the auth bodies and asserts the one view the desktop gets.
"""
from __future__ import annotations

import asyncio

import pytest

from cowork.api.v1.endpoints import hub_usage as ep
from cowork.db.scoped import TenantScope
from cowork.principal import HEADER_HUB_CREDENTIAL
from cowork.services import hub_usage as svc

ENTITLEMENTS = {
    "included_tokens": {"limit": 5_000_000, "used": 4_380_000, "remaining": 620_000},
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


@pytest.fixture
def calls(monkeypatch):
    asked: list[str] = []
    answers: dict[str, object] = {}

    async def _fake(path: str, bearer_token: str):
        asked.append(path)
        return answers.get(path)

    monkeypatch.setattr(svc, "_get_json", _fake)
    return type("Calls", (), {"asked": asked, "answers": answers})()


def _scope() -> TenantScope:
    return TenantScope(org_mode=True, org_id="org-a", user_id="user-a")


def _fetch(bearer: str = "jwt-abc"):
    return asyncio.run(svc.fetch_hub_usage(bearer_token=bearer, org_id="org-a"))


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
    assert view.free_tokens.remaining == 620_000
    assert view.free_tokens.limit == 5_000_000
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
    assert view.free_tokens.remaining == 620_000
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


def test_remaining_is_derived_when_auth_omits_it(calls):
    calls.answers[svc.ENTITLEMENTS_PATH] = {"included_tokens": {"limit": 100, "used": 30}}
    assert _fetch().free_tokens.remaining == 70


def test_unlimited_grant_passes_through_as_minus_one(calls):
    calls.answers[svc.ENTITLEMENTS_PATH] = {"included_tokens": {"limit": -1, "used": 30, "remaining": -1}}
    assert _fetch().free_tokens.limit == -1


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
