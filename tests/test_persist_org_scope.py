"""An organization's credential is never saved to this server's vault.

In org tenancy every connection lives in auth's data vault, and no org turn
reads this server's. ``persist_connection`` refuses an org scope, with or
without an org id, and any save under org tenancy, before the vault is
touched. Each caller that can run in an org deployment answers with
the fixed refusal and saves nothing. A personal scope (self-hosted) and the
desktop's local tenancy still save.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.api.v1.endpoints.connectors import submissions as submissions_ep
from cowork.common.settings.app_settings import OAuthSettings, get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.handlers import probe as probe_handler
from cowork.handlers.probe import ProbeHandler
from cowork.services.connectors.oauth.google import OAuthService
from cowork.services.connectors.persist import (
    ORG_LOCAL_SAVE_REFUSAL,
    OrgLocalSaveRefused,
    own_vault_root,
    persist_connection,
)
from cowork.services.connectors.probe import ProbeOutcome

ORG = TenantScope(org_mode=True, org_id="33333333-3333-4333-8333-333333333333", user_id="u-1")
ORG_WITHOUT_ID = TenantScope(org_mode=True)
PERSON = TenantScope(personal=True, org_id="partition-1", user_id="u-1")
TOKEN = "org-token-do-not-keep-9d1e"


class _UntouchableVault:
    """A vault any use of which fails the test."""

    def __getattr__(self, name: str):
        raise AssertionError(f"the refused save touched the vault ({name})")


@pytest.fixture
def org_tenancy(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "home" / "data-vault"))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


@pytest.fixture
def local_tenancy(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "home" / "data-vault"))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


def _record_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


@pytest.mark.parametrize("scope", [ORG, ORG_WITHOUT_ID], ids=["org scope", "org scope with no org id"])
def test_an_org_scope_is_refused_before_the_vault_is_touched(org_tenancy, scope):
    with pytest.raises(OrgLocalSaveRefused) as refused:
        persist_connection("linear", "personal-api-key", "", {"api_key": TOKEN}, vault=_UntouchableVault(), scope=scope)

    assert str(refused.value) == ORG_LOCAL_SAVE_REFUSAL
    assert TOKEN not in str(refused.value)


def test_a_save_with_no_scope_is_refused_under_org_tenancy(org_tenancy):
    with pytest.raises(OrgLocalSaveRefused):
        persist_connection("linear", "personal-api-key", "", {"api_key": TOKEN}, vault=_UntouchableVault())


@pytest.mark.parametrize("scope", [LOCAL_SCOPE, PERSON], ids=["operator scope", "personal scope"])
def test_any_scope_is_refused_under_org_tenancy(org_tenancy, scope):
    """No code path on an org deployment saves to this server's vault, whatever scope it holds."""
    with pytest.raises(OrgLocalSaveRefused):
        persist_connection("linear", "personal-api-key", "", {"api_key": TOKEN}, vault=_UntouchableVault(), scope=scope)


def test_a_save_with_no_scope_still_writes_on_the_desktop(local_tenancy):
    vault = LocalDataVault(local_tenancy / "vault")

    slug = persist_connection("linear", "personal-api-key", "", {"api_key": TOKEN}, vault=vault)

    assert vault.read_record("linear", slug)["fields"]["api_key"] == TOKEN


def test_a_personal_scope_still_writes(local_tenancy):
    """Self-hosted keeps its per-person local vault."""
    slug = persist_connection("linear", "personal-api-key", "", {"api_key": TOKEN}, scope=PERSON)

    assert slug
    assert _record_files(own_vault_root(PERSON))


@pytest.fixture
def accepting_probe(monkeypatch):
    """A probe that accepts, so the save decides the outcome."""

    class AcceptingProbe:
        def __init__(self, **kwargs) -> None:
            pass

        async def run(self):
            yield "verdict", ProbeOutcome(status="success", summary="Connection works.")

    monkeypatch.setattr(probe_handler, "CredentialProbe", AcceptingProbe)
    monkeypatch.setattr(ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object()))
    monkeypatch.setattr("anton.workspace.Workspace", lambda path: SimpleNamespace(path=path))


_GRANT_METHOD = {
    "id": "oauth", "label": "Sign in", "submit_action": "oauth_launch",
    "oauth": {"auth_url": "https://example.com/auth", "token_url": "https://example.com/token"},
}
_GRANT_FORM = {"form_id": "fm_orgsave02", "engine": "acme_crm", "title": "Acme CRM", "methods": [_GRANT_METHOD]}


async def test_an_oauth_grant_the_probe_would_save_first_is_refused_for_an_org(
    org_tenancy, accepting_probe, monkeypatch,
):
    """The probe names the fixed refusal and never a submitted value. The
    save after a passing probe is not driven here: under org tenancy the
    probe's own file work trips the blocking guard before it, and the
    submission route refuses every org submission that would reach it."""
    monkeypatch.setattr(probe_handler, "vault_for_scope", lambda scope=None: _UntouchableVault())
    form_spec, values, method = _GRANT_FORM, {"access_token": TOKEN, "refresh_token": "r-1"}, "oauth"
    submission_id = submissions_ep.store.stage(
        form_id=form_spec["form_id"], connector_id="acme_crm", conversation_id=None,
        values=values, form_spec=form_spec,
    )

    probed = "".join(
        [event async for event in ProbeHandler(scope=ORG).run(submission_id, "acme_crm", method, "acme", None)]
    )

    assert ORG_LOCAL_SAVE_REFUSAL in probed
    assert "Saved as" not in probed
    assert TOKEN not in probed
    assert _record_files(own_vault_root(ORG)) == []


class _FakeOAuthStore:
    def __init__(self, pending: dict[str, dict[str, Any]]) -> None:
        self.pending = pending
        self.outcomes: dict[str, dict[str, Any]] = {}

    def get_pending(self, service: str) -> dict[str, Any] | None:
        return self.pending.get(service)

    def clear_pending(self, service: str, *, error: str = "") -> None:
        self.pending.pop(service, None)

    def set_outcome(self, state: str, outcome: dict[str, Any]) -> None:
        self.outcomes[state] = outcome

    def get_outcome(self, state: str) -> dict[str, Any] | None:
        return self.outcomes.get(state)


def test_a_google_callback_for_an_org_saves_nothing(org_tenancy, monkeypatch):
    """With a pending start, a token exchange and userinfo faked, the callback
    reaches the real save, which refuses; the outcome carries the fixed text."""
    store = _FakeOAuthStore({"gmail": {
        "state": "s-1", "clientId": "gmail-client-id", "clientSecret": "gmail-client-secret",
        "redirectUri": "http://127.0.0.1/cb", "verifier": "v", "startedAt": "",
    }})
    svc = OAuthService()
    monkeypatch.setattr(svc, "_store", lambda settings: store)
    monkeypatch.setattr(
        svc, "_exchange_code",
        lambda **kwargs: {"access_token": TOKEN, "refresh_token": "rt-1", "expires_in": 3600},
    )
    monkeypatch.setattr(
        "cowork.services.connectors.oauth.google._USERINFO_FETCHERS",
        {"gmail": lambda access_token: {"email": "user@example.com"}},
    )

    html = svc.callback("gmail", code="auth-code", state="s-1", error="", settings=OAuthSettings(_env_file=None), scope=ORG)

    assert "connection failed" in html.lower()
    assert store.outcomes["s-1"]["status"] == "error"
    assert ORG_LOCAL_SAVE_REFUSAL in store.outcomes["s-1"]["error"]
    assert TOKEN not in html
    assert _record_files(own_vault_root(ORG)) == []
