"""The MindsHub workspace selector: the gate, the listing, and the stored pick.

The org cases call the handlers directly with an explicitly built TenantScope.
`cowork.server.app` wires the principal middleware only when the process started
in org mode, so a TestClient request would silently run under LOCAL_SCOPE and
prove nothing about per-user isolation.

Outbound HTTP is stubbed at `_get_json`, which is also what lets a test assert
the harder half of the flag contract: with the gate off, the workspace path is
never requested at all.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from cowork.api.v1.endpoints import hub_workspaces as ep
from cowork.common.settings.app_settings import (
    default_minds_auth_host,
    get_app_settings,
)
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_engine
from cowork.principal import HEADER_HUB_CREDENTIAL, caller_bearer, hub_credential
from cowork.schemas.hub_workspaces import HubWorkspaceActivateRequest
from cowork.services import hub_workspaces as svc
from cowork.services.settings import SettingService

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_A2 = "a2a2a2a2-a2a2-a2a2-a2a2-a2a2a2a2a2a2"

WS_DEFAULT = "d0000000-0000-0000-0000-000000000001"
WS_CLIENT_A = "c0000000-0000-0000-0000-00000000000a"
WS_ARCHIVED = "a0000000-0000-0000-0000-00000000000f"

ENTITLEMENTS = "/entitlements/me/"
WORKSPACES = "/organizations/current/workspaces/"


def _rows(*, archived: bool = False) -> dict:
    rows = [
        {
            "id": WS_DEFAULT,
            "slug": "default",
            "display_name": "Default",
            "is_default": True,
            "archived_at": None,
            "role": "member",
        },
        {
            "id": WS_CLIENT_A,
            "slug": "client-a",
            "display_name": "Client A",
            "is_default": False,
            "archived_at": None,
            "role": "manager",
        },
    ]
    if archived:
        rows.append(
            {
                "id": WS_ARCHIVED,
                "slug": "old",
                "display_name": "Old client",
                "is_default": False,
                "archived_at": "2026-08-01T00:00:00Z",
                "role": "manager",
            }
        )
    return {"results": rows}


class FakeRequest:
    """Just enough Request for `hub_credential`.

    Sends the credential the way the desktop shell has to: its own header, not
    Authorization, which Electron overwrites with the loopback token.
    """

    def __init__(self, bearer: str = "jwt-abc") -> None:
        self.headers = (
            {HEADER_HUB_CREDENTIAL: f"Bearer {bearer}"} if bearer else {}
        )


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture(autouse=True)
def _clean_state():
    """Drop the caches and every stored pick between tests.

    The settings table is shared across this file's tests, so a pick stored by
    one leaks into the next. Without this the three "the switch was refused"
    tests pass on a row an earlier test wrote, which is the exact shape of a test
    that would keep passing after the refusal stopped working.
    """
    svc.reset_caches_for_tests()
    yield
    svc.reset_caches_for_tests()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        for scope in (
            LOCAL_SCOPE,
            _scope(ORG_A, USER_A),
            _scope(ORG_A, USER_A2),
            _scope(ORG_B, USER_A),
        ):
            SettingService(s, scope).delete_setting(ep.SETTING_KEY)
        # The host-derivation test seeds `minds_url` with an attacker host to
        # prove nothing reads it. One session-scoped SQLite file backs the whole
        # run, so leaving it there hands every later test a tenant-settable value
        # pointing off-site.
        SettingService(s, LOCAL_SCOPE).delete_setting("minds_url")


@pytest.fixture
def calls(monkeypatch):
    """Stub `_get_json` and record every path asked for.

    Answers are seeded per test by mutating `answers`; a path with no entry
    answers None, which is the unreachable case.
    """
    asked: list[str] = []
    answers: dict[str, object] = {}

    async def _fake(path: str, bearer_token: str):
        asked.append(path)
        return answers.get(path)

    monkeypatch.setattr(svc, "_get_json", _fake)
    return type("Calls", (), {"asked": asked, "answers": answers})()


def _gate(calls, on: bool) -> None:
    calls.answers[ENTITLEMENTS] = {"feature_gates": {"authorization_ui": on}}


def _scope(org_id: str = ORG_A, user_id: str = USER_A) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org_id, user_id=user_id)


def _view(session, scope, bearer: str = "jwt-abc"):
    return asyncio.run(ep._build_view(FakeRequest(bearer), session, scope))


# ── The flag ─────────────────────────────────────────────────────────


def test_gate_off_reports_disabled_and_never_asks_for_workspaces(session, calls):
    """The half of the flag contract that a rendering test cannot see."""
    _gate(calls, False)

    view = _view(session, _scope())

    assert view.enabled is False
    assert view.workspaces == []
    assert calls.asked == [ENTITLEMENTS]
    assert WORKSPACES not in calls.asked


def test_absent_gates_field_reads_as_off(session, calls):
    """A version of auth that predates the field is indistinguishable from off.

    This is what lets the client ship before auth's half lands.
    """
    calls.answers[ENTITLEMENTS] = {"org_role": "member", "permissions": {}}

    view = _view(session, _scope())

    assert view.enabled is False
    assert calls.asked == [ENTITLEMENTS]


def test_unreachable_auth_reads_as_off(session, calls):
    """No answer is not the same as yes."""
    view = _view(session, _scope())

    assert view.enabled is False
    assert calls.asked == [ENTITLEMENTS]


def test_no_bearer_reads_as_off_without_asking(session, calls):
    view = _view(session, _scope(), bearer="")

    assert view.enabled is False
    assert calls.asked == []


def test_force_on_override_needs_no_bearer_and_cannot_turn_the_gate_off(
    session, calls, monkeypatch
):
    """The development override is ON only, so it cannot escape the kill switch."""
    monkeypatch.setenv("COWORK_HUB_WORKSPACES_FORCE_ON", "true")
    get_app_settings.cache_clear()
    try:
        _gate(calls, False)
        calls.answers[WORKSPACES] = _rows()

        view = _view(session, _scope(), bearer="")

        assert view.enabled is True
        # The gate was never consulted: the override short-circuits ahead of it.
        assert ENTITLEMENTS not in calls.asked
    finally:
        get_app_settings.cache_clear()


# ── The listing ──────────────────────────────────────────────────────


def test_listing_rows_and_default_active_when_nothing_stored(session, calls):
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()

    view = _view(session, _scope())

    assert view.enabled is True
    assert view.reachable is True
    assert [w.id for w in view.workspaces] == [WS_DEFAULT, WS_CLIENT_A]
    assert view.active_workspace_id == WS_DEFAULT
    assert view.workspaces[1].display_name == "Client A"
    assert view.workspaces[1].role == "manager"


def test_unreachable_listing_is_not_a_single_workspace_org(session, calls):
    """An empty list with `reachable=False` must stay distinguishable.

    Collapsing the two is how a caller with three workspaces silently loses two
    during an auth outage.
    """
    _gate(calls, True)

    view = _view(session, _scope())

    assert view.enabled is True
    assert view.reachable is False
    assert view.workspaces == []


def test_archived_workspace_is_dropped_unless_it_is_the_active_one(session, calls):
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows(archived=True)

    view = _view(session, _scope())
    assert WS_ARCHIVED not in [w.id for w in view.workspaces]

    # Stored pick on the archived one: the row under the check must not vanish.
    SettingService(session, _scope()).upsert_setting(ep.SETTING_KEY, WS_ARCHIVED)
    svc.reset_caches_for_tests()
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows(archived=True)

    view = _view(session, _scope())
    assert view.active_workspace_id == WS_ARCHIVED
    assert WS_ARCHIVED in [w.id for w in view.workspaces]


def test_a_row_with_no_id_is_dropped_without_losing_the_listing(session, calls):
    _gate(calls, True)
    calls.answers[WORKSPACES] = {"results": [{"display_name": "nameless"}, *_rows()["results"]]}

    view = _view(session, _scope())

    assert [w.id for w in view.workspaces] == [WS_DEFAULT, WS_CLIENT_A]


def test_a_bare_list_body_is_accepted(session, calls):
    """A pagination change upstream degrades to a full list, not an empty menu."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()["results"]

    view = _view(session, _scope())

    assert [w.id for w in view.workspaces] == [WS_DEFAULT, WS_CLIENT_A]


def test_stored_pick_naming_nothing_live_falls_back_to_default(session, calls):
    """A revoked grant or an id from another org must not leave no row checked."""
    SettingService(session, _scope()).upsert_setting(
        ep.SETTING_KEY, "f0000000-0000-0000-0000-00000000dead"
    )
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()

    view = _view(session, _scope())

    assert view.active_workspace_id == WS_DEFAULT


# ── Switching ────────────────────────────────────────────────────────


def _activate(session, scope, workspace_id: str, bearer: str = "jwt-abc"):
    return asyncio.run(
        ep.set_active_hub_workspace(
            HubWorkspaceActivateRequest(workspace_id=workspace_id),
            FakeRequest(bearer),
            session,
            scope,
        )
    )


def test_switching_stores_the_pick_and_reports_it(session, calls):
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()

    view = _activate(session, _scope(), WS_CLIENT_A)

    assert view.active_workspace_id == WS_CLIENT_A
    assert SettingService(session, _scope()).load().hub_workspace_id == WS_CLIENT_A


def test_switching_to_a_workspace_not_in_the_listing_is_refused(session, calls):
    """Auth decides who may see which workspace; the listing is that answer."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()

    with pytest.raises(HTTPException) as caught:
        _activate(session, _scope(), "b0000000-0000-0000-0000-0000000000bb")

    assert caught.value.status_code == 403
    assert SettingService(session, _scope()).load().hub_workspace_id == ""


def test_switching_refuses_rather_than_storing_an_unverifiable_id(session, calls):
    _gate(calls, True)

    with pytest.raises(HTTPException) as caught:
        _activate(session, _scope(), WS_CLIENT_A)

    assert caught.value.status_code == 503
    assert SettingService(session, _scope()).load().hub_workspace_id == ""


def test_switching_with_the_gate_off_is_not_a_route(session, calls):
    _gate(calls, False)

    with pytest.raises(HTTPException) as caught:
        _activate(session, _scope(), WS_CLIENT_A)

    assert caught.value.status_code == 404
    assert SettingService(session, _scope()).load().hub_workspace_id == ""


# ── One caller's answer is not another's ─────────────────────────────


def _rows_for(*ids) -> dict:
    """A listing carrying only the named workspaces, as auth would answer it."""
    by_id = {row["id"]: row for row in _rows(archived=True)["results"]}
    return {"results": [by_id[i] for i in ids]}


def test_one_members_listing_is_not_served_to_another(session, calls):
    """auth answers this per caller, so the cache has to key on the caller.

    `visible_workspaces` gives an owner or admin every workspace in the
    organization and a plain member only the ones they hold a grant on. Keyed on
    the organization alone, the first caller through warms an entry that every
    other member of that org then reads, and this process serves all of them:
    org mode, two replicas, one module-level dict.
    """
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT, WS_CLIENT_A)
    admin = _view(session, _scope(user_id=USER_A))
    assert [w.id for w in admin.workspaces] == [WS_DEFAULT, WS_CLIENT_A]

    # Same org, same moment, a member auth answers differently for.
    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT)
    member = _view(session, _scope(user_id=USER_A2))

    assert [w.id for w in member.workspaces] == [WS_DEFAULT]


def test_a_member_cannot_switch_into_a_workspace_from_anothers_cached_listing(session, calls):
    """The grant check reads the same cache the menu does, so a shared entry is
    not only a leak: it is the switch's authorization, and it would pass."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT, WS_CLIENT_A)
    _view(session, _scope(user_id=USER_A))

    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT)
    with pytest.raises(HTTPException) as caught:
        _activate(session, _scope(user_id=USER_A2), WS_CLIENT_A)

    assert caught.value.status_code == 403
    assert SettingService(session, _scope(user_id=USER_A2)).load().hub_workspace_id == ""


def test_the_gate_verdict_is_not_shared_across_callers(session, calls):
    """`authorization_ui` declares `idType: userID`, so auth evaluates it for
    whoever presents the bearer. A rule below 100%, or one per-user override, and
    an org-keyed verdict hands one person's answer to everyone beside them."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()
    assert _view(session, _scope(user_id=USER_A)).enabled is True

    calls.answers[ENTITLEMENTS] = {"feature_gates": {"authorization_ui": False}}

    assert _view(session, _scope(user_id=USER_A2)).enabled is False


def test_switching_into_an_archived_workspace_is_refused(session, calls):
    """The writable set is the set the menu offered, and the menu drops archived
    rows. Accepting one stores a pick no client could have made."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows(archived=True)

    with pytest.raises(HTTPException) as caught:
        _activate(session, _scope(), WS_ARCHIVED)

    assert caught.value.status_code == 409
    assert SettingService(session, _scope()).load().hub_workspace_id == ""


def test_a_desktop_account_switch_does_not_reuse_the_previous_listing(session, calls):
    """The per-caller key has to include the CREDENTIAL, not just the identity.

    Outside org mode `scope_from_principal` returns LOCAL_SCOPE, so `user_id` is
    None for every request on a desktop install and an identity-only key collapses
    to one shared entry. Sign out, sign in as someone else, and the previous
    account's workspaces would be served for the rest of the TTL, with the grant
    check on `PUT /active` reading them.
    """
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT, WS_CLIENT_A)
    first = _view(session, LOCAL_SCOPE, bearer="jwt-first-account")
    assert [w.id for w in first.workspaces] == [WS_DEFAULT, WS_CLIENT_A]

    # Same process, same (empty) scope, a different MindsHub session.
    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT)
    second = _view(session, LOCAL_SCOPE, bearer="jwt-second-account")

    assert [w.id for w in second.workspaces] == [WS_DEFAULT]


def test_a_desktop_account_switch_cannot_switch_on_the_previous_listing(session, calls):
    """Same key, but on the path where it authorizes rather than renders."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT, WS_CLIENT_A)
    _view(session, LOCAL_SCOPE, bearer="jwt-first-account")

    calls.answers[WORKSPACES] = _rows_for(WS_DEFAULT)
    with pytest.raises(HTTPException) as caught:
        _activate(session, LOCAL_SCOPE, WS_CLIENT_A, bearer="jwt-second-account")

    assert caught.value.status_code == 403


def test_the_archived_refusal_survives_a_write_through_the_settings_route(session, calls):
    """The 409 must not read a value any caller can write.

    `hub_workspace_id` is an untagged UserSettings field, so
    `PUT /api/v1/settings/hub_workspace_id` stores it with no gate and no listing
    check. `selectable` keeps the ACTIVE row even when archived, so a refusal
    phrased as "is this in the set the menu offered" would have been talked into
    accepting an archived workspace by one call to that route.
    """
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows(archived=True)
    # Exactly what the unguarded second writer does.
    SettingService(session, _scope()).upsert_setting(ep.SETTING_KEY, WS_ARCHIVED)

    with pytest.raises(HTTPException) as caught:
        _activate(session, _scope(), WS_ARCHIVED)

    assert caught.value.status_code == 409


def test_a_gate_off_verdict_is_cached_as_long_as_a_gate_on_one(session, calls):
    """A verdict auth actually returned is an answer, whichever way it went.

    The TTL used to be picked from the verdict, so a definite "off" got the 15s
    failure budget. That is the state this ships in, and per-caller keys made it
    one extra entitlements read per person rather than per organization.
    """
    _gate(calls, False)
    assert _view(session, _scope()).enabled is False
    asked_once = calls.asked.count(ENTITLEMENTS)

    # Age the entry past _TTL_FAIL but leave it well inside _TTL_OK.
    key = next(iter(svc._gate_cache))
    stamped, value, answered = svc._gate_cache[key]
    assert answered is True
    svc._gate_cache[key] = (stamped - (svc._TTL_FAIL + 1), value, answered)

    assert _view(session, _scope()).enabled is False
    assert calls.asked.count(ENTITLEMENTS) == asked_once, (
        "an answered gate-off verdict was re-fetched inside its own TTL"
    )


def test_an_unanswered_gate_read_keeps_the_short_ttl(session, calls):
    """The other half: no answer is still cached, but only briefly."""
    calls.answers.pop(ENTITLEMENTS, None)  # unreachable
    assert _view(session, _scope()).enabled is False
    asked_once = calls.asked.count(ENTITLEMENTS)

    key = next(iter(svc._gate_cache))
    stamped, value, answered = svc._gate_cache[key]
    assert answered is False
    svc._gate_cache[key] = (stamped - (svc._TTL_FAIL + 1), value, answered)

    _view(session, _scope())

    assert calls.asked.count(ENTITLEMENTS) == asked_once + 1


def test_expired_entries_are_swept_rather_than_held_for_the_process_lifetime(session, calls):
    """Nothing re-reads a departed caller's key, so nothing would ever drop it."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()
    _view(session, LOCAL_SCOPE, bearer="jwt-someone-who-leaves")
    assert len(svc._listing_cache) == 1

    stale = {
        k: (stamped - (svc._MAX_TTL_S + 1), v)
        for k, (stamped, v) in svc._listing_cache.items()
    }
    svc._listing_cache.clear()
    svc._listing_cache.update(stale)

    _view(session, LOCAL_SCOPE, bearer="jwt-somebody-else")

    assert len(svc._listing_cache) == 1, "the departed caller's entry was never dropped"


# ── Where the pick is stored ─────────────────────────────────────────


def test_two_people_in_one_org_hold_separate_picks(session, calls):
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()
    _activate(session, _scope(user_id=USER_A), WS_CLIENT_A)

    assert SettingService(session, _scope(user_id=USER_A)).load().hub_workspace_id == WS_CLIENT_A
    assert SettingService(session, _scope(user_id=USER_A2)).load().hub_workspace_id == ""


def test_one_person_in_two_orgs_holds_separate_picks(session, calls):
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()
    _activate(session, _scope(org_id=ORG_A), WS_CLIENT_A)

    assert SettingService(session, _scope(org_id=ORG_A)).load().hub_workspace_id == WS_CLIENT_A
    assert SettingService(session, _scope(org_id=ORG_B)).load().hub_workspace_id == ""


def test_the_settings_writer_cannot_grant_a_workspace_it_only_stores_a_string(
    session, calls
):
    """There is a SECOND writer for this key, and it has no grant check.

    `hub_workspace_id` is a declared `UserSettings` field, so
    `PUT /api/v1/settings/hub_workspace_id` stores it like any other setting and
    never consults the listing. That is the writer-nobody-enumerated shape the
    model-value inventory in `cowork/services/providers.py` warns about, so it is
    worth pinning rather than assuming.

    It grants nothing, and this is why: resolution filters the stored id against
    the listing auth returned, so an id the caller has no grant on resolves to
    the default workspace and the menu renders exactly the same rows. The stored
    value decides which row carries a check, never which rows exist.
    """
    SettingService(session, _scope()).upsert_setting(
        ep.SETTING_KEY, "b0000000-0000-0000-0000-0000000000bb"
    )
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()

    view = _view(session, _scope())

    assert view.active_workspace_id == WS_DEFAULT
    assert [w.id for w in view.workspaces] == [WS_DEFAULT, WS_CLIENT_A]


def test_the_pick_is_a_personal_setting_not_org_configuration(session):
    """An untagged field writes per-user. Tagged ORG, one admin's pick would
    become everyone's."""
    from cowork.common.settings.user_settings import setting_is_org_scoped

    assert setting_is_org_scoped(ep.SETTING_KEY) is False


def test_a_desktop_install_stores_the_pick_in_the_global_row(session):
    """Local mode carries no org and no user, so there is no per-user row to
    write. One install, one person, one row."""
    SettingService(session, LOCAL_SCOPE).upsert_setting(ep.SETTING_KEY, WS_CLIENT_A)

    assert SettingService(session, LOCAL_SCOPE).load().hub_workspace_id == WS_CLIENT_A


# ── What this must not touch ─────────────────────────────────────────


def test_nothing_on_the_turn_path_reads_the_stored_workspace():
    """The selector changes what the client shows, not what a turn is billed to.

    Both turn credentials are workspace-blind: a desktop turn presents a
    long-lived key bound to a user and an organization, and a cloud turn presents
    a minted key whose request body has no workspace field. Attributing usage to a
    workspace is separate work that has not shipped, so this asserts the boundary
    rather than trusting it.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    targets = [
        repo / "cowork" / "turnqueue",
        repo / "cowork" / "handlers",
        repo / "cowork" / "harnesses",
        repo / "cowork" / "services" / "providers.py",
    ]

    # Read in Python rather than shelling out to grep. The shell version asserted
    # on stdout alone, so a wrong working directory or a renamed path made grep
    # error, print nothing, and the guard pass having checked no files at all.
    missing = [str(t.relative_to(repo)) for t in targets if not t.exists()]
    assert not missing, f"this guard points at paths that no longer exist: {missing}"

    # Every file, not just `*.py`. The shell version this replaced was a plain
    # `grep -rln`, so it also read the skill markdown and prompt templates under
    # `cowork/harnesses/`, and the turn path can name a setting from one of those
    # as easily as from code.
    def _walk(target):
        if target.is_file():
            return [target]
        return sorted(f for f in target.rglob("*") if f.is_file())

    files = [f for t in targets for f in _walk(t)]
    assert files, "the guard matched no files, so it proved nothing"
    hits = [
        str(f.relative_to(repo))
        for f in files
        if "hub_workspace_id" in f.read_text(encoding="utf-8", errors="ignore")
    ]

    assert hits == [], f"the turn path reads the stored workspace: {hits}"


def test_the_read_goes_to_the_operator_auth_host_with_the_callers_own_bearer(
    monkeypatch, session
):
    """`minds_url` and the stored provider key are both tenant-settable, so an org
    admin who could steer this read would harvest members' bearers.

    Asserted on the URL and header actually built, rather than by grepping the
    module: this is the only test that exercises `_get_json`, since every other
    test in this file stubs it.
    """
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.setenv("POD_NAMESPACE", "")
    # A tenant-settable value an org admin controls. Nothing here may read it.
    SettingService(session, LOCAL_SCOPE).upsert_setting(
        "minds_url", "https://attacker.test/v1"
    )
    seen: list[tuple[str, dict]] = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return _rows()

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, headers=None):
            seen.append((url, headers or {}))
            return FakeResponse()

    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeClient)

    listing = asyncio.run(
        svc.fetch_hub_workspaces(bearer_token="jwt-abc", org_id=ORG_A, user_id=USER_A)
    )

    assert listing.reachable is True
    url, headers = seen[0]
    assert url == "https://auth.staging.mindshub.ai/v1/organizations/current/workspaces/"
    assert headers["Authorization"] == "Bearer jwt-abc"
    assert "attacker.test" not in url


def test_the_route_answers_the_same_view_its_builder_does(session, calls):
    """Covers the route function itself, not just the builder underneath it."""
    _gate(calls, True)
    calls.answers[WORKSPACES] = _rows()

    view = asyncio.run(
        ep.get_hub_workspaces(FakeRequest(), session, _scope())
    )

    assert view.enabled is True
    assert view.active_workspace_id == WS_DEFAULT


@pytest.mark.parametrize(
    "failure",
    [
        "raises",
        "times-out",
        "non-json",
    ],
)
def test_every_transport_failure_reads_as_unreachable(monkeypatch, failure):
    """The fail-closed guarantee, one case per way a call can go wrong.

    A menu open must never raise into the UI and must never run past the budget,
    so each of these returns "we could not ask" rather than propagating.

    The timeout case SLEEPS and answers cleanly, and the assertion is on ELAPSED
    TIME, which is the only thing that can tell the ceiling apart from the
    blanket `except`. Raising `asyncio.TimeoutError` from the fake proved
    nothing: it lands in that handler with or without the `asyncio.wait_for`
    around the fetch, so the budget could be deleted and this stayed green.
    """

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            if failure == "non-json":
                raise ValueError("not json")
            return _rows()

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, headers=None):
            if failure == "raises":
                raise RuntimeError("connection reset")
            if failure == "times-out":
                # Far past the patched budget, and it answers successfully if it
                # is ever allowed to finish, so only the ceiling can fail it.
                await asyncio.sleep(30)
            return FakeResponse()

    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(svc, "_TIMEOUT_S", 0.05)

    started = time.monotonic()
    listing = asyncio.run(
        svc.fetch_hub_workspaces(bearer_token="jwt-abc", org_id=ORG_A, user_id=USER_A)
    )
    enabled = asyncio.run(
        svc.authorization_ui_enabled(bearer_token="jwt-abc", org_id=ORG_A, user_id=USER_A)
    )
    elapsed = time.monotonic() - started

    assert listing.reachable is False
    assert listing.workspaces == []
    assert enabled is False
    if failure == "times-out":
        assert elapsed < 2.0, f"the total-time ceiling did not fire: two calls took {elapsed:.1f}s"


@pytest.mark.parametrize(
    "body",
    [
        {"results": "not-a-list"},
        {"results": None},
        "a bare string",
        42,
    ],
)
def test_a_body_that_is_not_a_list_of_rows_reads_as_unreachable(body):
    assert svc._parse_listing(body).reachable is False


def test_a_row_that_fails_validation_drops_only_that_row():
    """Losing one workspace from the menu beats losing all of them."""
    listing = svc._parse_listing(
        {
            "results": [
                {"id": WS_DEFAULT, "display_name": "Default", "is_default": "yes-please"},
                {"id": WS_CLIENT_A, "display_name": "Client A"},
            ]
        }
    )

    assert [w.id for w in listing.workspaces] == [WS_CLIENT_A]
    assert listing.reachable is True


def test_resolve_active_with_no_workspaces_is_nothing_rather_than_a_crash():
    assert svc.resolve_active([], "") is None
    assert svc.resolve_active([], WS_CLIENT_A) is None


def test_resolve_active_falls_through_to_the_first_row_with_no_default():
    """An organization whose default workspace the caller has no grant on."""
    rows = [
        svc.HubWorkspace(id=WS_CLIENT_A, display_name="Client A"),
        svc.HubWorkspace(id=WS_ARCHIVED, display_name="Old"),
    ]

    assert svc.resolve_active(rows, "").id == WS_CLIENT_A


def test_an_http_refusal_is_not_mistaken_for_an_empty_organization(monkeypatch):
    """A 403 and "you have no workspaces" must not render the same."""

    class FakeResponse:
        status_code = 403

        @staticmethod
        def json():
            return {"detail": "nope"}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeClient)

    listing = asyncio.run(
        svc.fetch_hub_workspaces(bearer_token="jwt-abc", org_id=ORG_A, user_id=USER_A)
    )

    assert listing.reachable is False
    assert listing.workspaces == []


# ── Host derivation and the bearer ───────────────────────────────────


@pytest.mark.parametrize(
    "env,namespace,expected",
    [
        # A desktop install sets no ENV, which is production.
        ("", "", "https://auth.mindshub.ai"),
        ("staging", "", "https://auth.staging.mindshub.ai"),
        ("dev", "", "https://auth.dev.mindshub.ai"),
        # A per-PR env runs ENV=development and has its OWN auth database, so
        # the namespace decides. Matches `auth-<env>.dev.mindshub.ai` in the
        # argocd-envs chart; the ENV slug alone would send these reads to dev's
        # auth, which answers 401 for a caller it has never seen.
        ("development", "pr-42", "https://auth-pr-42.dev.mindshub.ai"),
        ("dev", "pr-1829", "https://auth-pr-1829.dev.mindshub.ai"),
    ],
)
def test_auth_host_derivation(monkeypatch, env, namespace, expected):
    monkeypatch.setenv("ENV", env)
    monkeypatch.setenv("POD_NAMESPACE", namespace)

    assert default_minds_auth_host() == expected


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("BEARER  abc  ", "abc"),
        ("Basic abc", ""),
        ("abc", ""),
        ("", ""),
    ],
)
def test_caller_bearer_parsing(header, expected):
    request = type("R", (), {"headers": {"Authorization": header}})()

    assert caller_bearer(request) == expected


def test_caller_bearer_with_no_request_is_empty():
    assert caller_bearer(None) == ""


# ── Which header carries the credential ──────────────────────────────


def test_the_hub_header_wins_over_the_loopback_authorization():
    """The desktop case, and the whole reason this header exists.

    Electron's main process overwrites Authorization on every loopback request
    with the server's own token, so the caller's JWT can only arrive under its
    own name. Reading Authorization here would forward the loopback token to
    auth, which rejects it, and the menu would be empty on every desktop install.
    """
    request = type(
        "R",
        (),
        {
            "headers": {
                "Authorization": "Bearer loopback-token-not-the-users",
                HEADER_HUB_CREDENTIAL: "Bearer real-user-jwt",
            }
        },
    )()

    assert hub_credential(request) == "real-user-jwt"
    # The other helper is untouched, so the model-catalog fetch cannot be steered
    # by a client setting the hub header.
    assert caller_bearer(request) == "loopback-token-not-the-users"


def test_the_web_shell_falls_back_to_authorization(monkeypatch):
    """No hook there, so the ingress-forwarded Authorization is the JWT."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        request = type("R", (), {"headers": {"Authorization": "Bearer web-jwt"}})()

        assert hub_credential(request) == "web-jwt"
    finally:
        get_app_settings.cache_clear()


def test_a_desktop_install_never_forwards_the_servers_own_token(monkeypatch):
    """The fallback is org mode only, and this is why.

    On a desktop install the Electron main process assigns THIS server's bearer
    to `Authorization` on every loopback request, so a caller with no MindsHub
    session leaves that header holding our own credential. Falling back to it
    puts that credential on the wire to auth. It gets refused, and it has still
    left the machine.
    """
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        request = type("R", (), {"headers": {"Authorization": "Bearer the-loopback-token"}})()

        assert hub_credential(request) == ""
    finally:
        get_app_settings.cache_clear()


@pytest.mark.parametrize("value", ["", "Basic abc", "real-user-jwt"])
def test_a_malformed_hub_header_falls_back_rather_than_forwarding_junk(value, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        request = type(
            "R",
            (),
            {"headers": {HEADER_HUB_CREDENTIAL: value, "Authorization": "Bearer fallback"}},
        )()

        assert hub_credential(request) == "fallback"
    finally:
        get_app_settings.cache_clear()


def test_hub_credential_with_no_request_is_empty():
    assert hub_credential(None) == ""
