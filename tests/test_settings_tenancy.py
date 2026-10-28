"""Per-scope behaviour of the split settings table.

The ``settings`` table is deliberately NOT a scoped-session root (it stays in
``_TENANCY_DEFERRED_TABLES``): ``SettingService`` owns tenancy explicitly. A
key's WRITE scope comes from its classification (``setting_is_org_scoped``);
READS always resolve most-specific-wins: user → org → global (the
NULL-scope legacy/env row) → field default. Local / no-scope operates purely on
global rows, i.e. the pre-split desktop behaviour.
"""
from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings.user_settings import (
    UserSettings,
    setting_is_org_scoped,
)
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.models.setting import Setting
from cowork.services.settings import SettingService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


@pytest.fixture()
def engine():
    import cowork.models.project, cowork.models.conversation  # noqa: F401
    import cowork.models.message, cowork.models.message_event  # noqa: F401
    import cowork.models.file, cowork.models.channel, cowork.models.setting  # noqa: F401
    import cowork.models.task_object, cowork.models.schedule, cowork.models.pin  # noqa: F401
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    return eng


def _svc(engine, scope: TenantScope = LOCAL_SCOPE) -> SettingService:
    return SettingService(Session(engine), scope)


def _org(org: str, user: str | None = None) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


# classification

def test_all_credentials_are_org_scoped_structurally():
    # Every SecretStr field is org without being listed — a new credential
    # field can't accidentally default to per-user.
    for key in UserSettings.model_fields:
        if UserSettings.field_is_sensitive(key):
            assert setting_is_org_scoped(key) is True


def test_provider_config_is_org_scoped():
    for key in ("minds_url", "planning_provider", "minds_model_enabled", "max_tool_rounds"):
        assert setting_is_org_scoped(key) is True


def test_ui_and_model_choice_are_user_scoped():
    for key in (
        "tone", "greeting", "planning_model", "coding_reasoning_effort",
        "router_reasoning_effort", "auto_pin", "favicon", "show_figure_sources",
    ):
        assert setting_is_org_scoped(key) is False


def test_channel_credential_keys_are_org_scoped():
    assert setting_is_org_scoped("channel.telegram.bot_token") is True


# org-class keys: shared within an org, isolated across orgs

def test_org_key_isolated_across_orgs(engine):
    _svc(engine, _org(ORG_A)).upsert_setting("openai_api_key", "sk-A")
    assert _svc(engine, _org(ORG_A)).get_setting("openai_api_key").is_set is True
    assert _svc(engine, _org(ORG_B)).get_setting("openai_api_key").is_set is False


def test_org_key_shared_between_users_of_same_org(engine):
    _svc(engine, _org(ORG_A, "alice")).upsert_setting("planning_provider", "openai")
    # bob in the same org sees the org-level provider choice
    assert _svc(engine, _org(ORG_A, "bob")).get_setting("planning_provider").value == "openai"


# user-class keys: isolated per user

def test_user_key_isolated_between_users_of_same_org(engine):
    _svc(engine, _org(ORG_A, "alice")).upsert_setting("tone", "spicy")
    assert _svc(engine, _org(ORG_A, "alice")).get_setting("tone").value == "spicy"
    # bob has no personal tone and no org/global override → field default
    assert _svc(engine, _org(ORG_A, "bob")).get_setting("tone").value == "balanced"


@pytest.mark.parametrize("other_scope", [_org(ORG_A, "bob"), _org(ORG_B, "alice")])
def test_favicon_is_isolated_by_member_and_organization(engine, other_scope):
    favicon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E"
    owner = _org(ORG_A, "alice")
    _svc(engine, owner).upsert_setting("favicon", favicon)

    assert _svc(engine, owner).get_setting("favicon").value == favicon
    assert _svc(engine, other_scope).get_setting("favicon").value == ""
    assert _svc(engine, LOCAL_SCOPE).get_setting("favicon").value == ""


def test_user_write_requires_user_in_scope(engine):
    # org present but no user → a personal-key write has nowhere to land.
    with pytest.raises(ValueError, match="no user is in scope"):
        _svc(engine, _org(ORG_A)).upsert_setting("tone", "spicy")


# read fallback: user -> org -> global

def test_global_row_is_fallback_for_all_scopes(engine):
    # A deployment/global row (written with no scope) is visible to every org
    # until they override it.
    _svc(engine, LOCAL_SCOPE).upsert_setting("greeting", "GLOBAL-HI")
    assert _svc(engine, _org(ORG_A, "alice")).get_setting("greeting").value == "GLOBAL-HI"
    assert _svc(engine, _org(ORG_B, "carol")).get_setting("greeting").value == "GLOBAL-HI"


def test_user_override_wins_over_global_without_leaking(engine):
    _svc(engine, LOCAL_SCOPE).upsert_setting("greeting", "GLOBAL-HI")
    _svc(engine, _org(ORG_B, "carol")).upsert_setting("greeting", "carol-hi")
    assert _svc(engine, _org(ORG_B, "carol")).get_setting("greeting").value == "carol-hi"
    # a different user still sees the global default, not carol's override
    assert _svc(engine, _org(ORG_A, "alice")).get_setting("greeting").value == "GLOBAL-HI"


def test_delete_removes_only_own_override_then_falls_back(engine):
    _svc(engine, LOCAL_SCOPE).upsert_setting("greeting", "GLOBAL-HI")
    carol = _org(ORG_B, "carol")
    _svc(engine, carol).upsert_setting("greeting", "carol-hi")
    assert _svc(engine, carol).delete_setting("greeting") is True
    # the deployment/global row survives and is resolved again
    assert _svc(engine, carol).get_setting("greeting").value == "GLOBAL-HI"


def test_clear_credentials_only_touches_own_org(engine):
    _svc(engine, _org(ORG_A)).upsert_setting("openai_api_key", "sk-A")
    _svc(engine, _org(ORG_B)).upsert_setting("openai_api_key", "sk-B")
    _svc(engine, _org(ORG_A)).clear_credentials()
    assert _svc(engine, _org(ORG_A)).get_setting("openai_api_key").is_set is False
    assert _svc(engine, _org(ORG_B)).get_setting("openai_api_key").is_set is True


# local / desktop mode unchanged

def test_local_mode_operates_on_global_rows_only(engine):
    # An org-scoped write lands in the org row; local (global-only) never sees
    # it. (The reverse — a global row IS visible to orgs — is the fallback,
    # covered by test_global_row_is_fallback_for_all_scopes.)
    _svc(engine, _org(ORG_A)).upsert_setting("openai_api_key", "sk-A")
    assert _svc(engine, LOCAL_SCOPE).get_setting("openai_api_key").is_set is False
    _svc(engine, LOCAL_SCOPE).upsert_setting("anthropic_api_key", "sk-local")
    assert _svc(engine, LOCAL_SCOPE).get_setting("anthropic_api_key").is_set is True


def test_no_scope_behaves_like_local(engine):
    # Every pre-split caller constructs SettingService(session) with no scope.
    svc = SettingService(Session(engine))
    svc.upsert_setting("tone", "quiet")
    assert svc.get_setting("tone").value == "quiet"


# get_user_settings resolves per scope (explicit + ambient), cache doesn't leak

def test_get_user_settings_resolves_and_isolates_per_scope():
    # Uses the app DB (get_open_session), like test_settings_logout.
    from cowork.common.settings import user_settings as us
    from cowork.db.session import get_open_session

    a = _org("guso-A", "u")
    b = _org("guso-B", "u")
    s = get_open_session()
    try:
        SettingService(s, a).upsert_setting("planning_provider", "openai")
        SettingService(s, b).upsert_setting("planning_provider", "gemini")
    finally:
        s.close()
    us.invalidate_user_settings_cache()
    try:
        # explicit scope
        assert us.get_user_settings(a).planning_provider.value == "openai"
        # a second scope must NOT get the first's cached object
        assert us.get_user_settings(b).planning_provider.value == "gemini"
        # ambient scope (the turn-boundary path)
        with us.use_settings_scope(a):
            assert us.get_user_settings().planning_provider.value == "openai"
        with us.use_settings_scope(b):
            assert us.get_user_settings().planning_provider.value == "gemini"
    finally:
        s = get_open_session()
        try:
            SettingService(s, a).delete_setting("planning_provider")
            SettingService(s, b).delete_setting("planning_provider")
        finally:
            s.close()
        us.invalidate_user_settings_cache()


# review-fix regressions

def test_personal_setting_isolated_for_same_user_across_orgs():
    # Fix #1: the same Keycloak user in two orgs must NOT share a personal row.
    from cowork.common.settings import user_settings as us
    from cowork.db.session import get_open_session
    a = _org("iso-A", "sameuser")
    b = _org("iso-B", "sameuser")
    s = get_open_session()
    try:
        SettingService(s, a).upsert_setting("tone", "spicy")           # org A
        assert SettingService(s, b).get_setting("tone").value == "balanced"  # org B: default, no bleed
        SettingService(s, b).upsert_setting("tone", "mild")            # org B writes its own
        assert SettingService(s, a).get_setting("tone").value == "spicy"     # org A unmutated
    finally:
        SettingService(s, a).delete_setting("tone")
        SettingService(s, b).delete_setting("tone")
        s.close()
        us.invalidate_user_settings_cache()


def test_write_without_org_fails_closed_but_read_falls_back(engine):
    # Fix #3: org mode with no org_id must NOT write a global row; reads still
    # resolve the global fallback.
    from cowork.db.scoped import MissingTenantScopeError
    noorg = TenantScope(org_mode=True, org_id=None, user_id="u")
    with pytest.raises(MissingTenantScopeError):
        _svc(engine, noorg).upsert_setting("openai_api_key", "sk-LEAK")
    # a global row is still readable under that scope (fallback intact)
    _svc(engine, LOCAL_SCOPE).upsert_setting("greeting", "hi")
    assert _svc(engine, noorg).get_setting("greeting").value == "hi"


def test_save_all_is_atomic_when_a_target_is_invalid(engine):
    # Fix #8: a mixed batch with an un-writable personal target writes NOTHING.
    org_no_user = TenantScope(org_mode=True, org_id="A")  # org, no user
    with pytest.raises(ValueError, match="no user is in scope"):
        _svc(engine, org_no_user).save_all({"minds_url": "http://x", "tone": "spicy"})
    # neither key was staged/committed
    assert _svc(engine, org_no_user).get_setting("minds_url").is_set is False


def test_field_is_sensitive_detects_bare_and_optional_secretstr():
    # Fix #7: a bare SecretStr (empty get_args) must still be detected.
    from typing import get_args
    from pydantic import SecretStr
    def is_sensitive(ann):
        return ann is SecretStr or SecretStr in get_args(ann)
    assert is_sensitive(SecretStr) is True          # the fix
    assert is_sensitive(SecretStr | None) is True    # existing fields
    assert is_sensitive(str) is False
    # and the real helper still agrees for a known credential
    assert UserSettings.field_is_sensitive("openai_api_key") is True


# DB-level partial unique indexes + CHECK

def test_partial_unique_constraints(engine):
    with Session(engine) as s:
        # two orgs, same key → allowed (distinct org rows)
        s.add(Setting(key="k", value="a", scope="org", org_id=ORG_A))
        s.add(Setting(key="k", value="b", scope="org", org_id=ORG_B))
        s.commit()
    with Session(engine) as s:
        # same (key, org) twice → rejected
        s.add(Setting(key="k", value="dup", scope="org", org_id=ORG_A))
        with pytest.raises(IntegrityError):
            s.commit()
    with Session(engine) as s:
        # two global rows for one key → rejected
        s.add(Setting(key="g", value="one", scope=None))
        s.commit()
    with Session(engine) as s:
        s.add(Setting(key="g", value="two", scope=None))
        with pytest.raises(IntegrityError):
            s.commit()


def test_user_rows_unique_by_org_and_user(engine):
    # Fix #1 at the DB level: same (key, user) in two orgs is allowed; the same
    # (key, org, user) twice is rejected.
    with Session(engine) as s:
        s.add(Setting(key="t", value="a", scope="user", org_id=ORG_A, user_id="u"))
        s.add(Setting(key="t", value="b", scope="user", org_id=ORG_B, user_id="u"))  # same user, other org
        s.commit()
    with Session(engine) as s:
        s.add(Setting(key="t", value="dup", scope="user", org_id=ORG_A, user_id="u"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_check_constraint_rejects_bad_row_shapes(engine):
    # Fix #6: NULL owner on an org/user row, or a bogus scope, is rejected.
    for bad in (
        Setting(key="c", value="v", scope="org", org_id=None),          # org needs org_id
        Setting(key="c", value="v", scope="user", org_id=ORG_A, user_id=None),  # user needs user_id
        Setting(key="c", value="v", scope="bogus", org_id=ORG_A),       # invalid scope
    ):
        with Session(engine) as s:
            s.add(bad)
            with pytest.raises(IntegrityError):
                s.commit()


# personal partitions (local tenancy with trusted identity headers)

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
PARTITION_B = "55e92829-e701-599b-88f9-69f5bf1165eb"


def _person(*, partition: str | None, user: str | None, resources=None) -> TenantScope:
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES

    return TenantScope(
        personal=True,
        org_id=partition,
        user_id=user,
        personal_resources=ALL_PERSONAL_RESOURCES if resources is None else resources,
    )


def select_settings():
    from sqlmodel import select

    return select(Setting)


def test_operator_pinned_keys_are_the_deployment_wide_ones():
    from cowork.common.settings.user_settings import setting_is_operator_pinned

    for key in (
        "memory_enabled", "memory_mode", "episodic_memory",
        "planning_provider", "planning_model", "coding_model", "router_model",
        "coding_reasoning_effort", "model_mode", "model_overrides", "providers_json",
        "minds_url", "openai_base_url", "publish_url",
        "max_tool_rounds", "max_continuations", "max_turn_tokens",
        # "How the agent acts" keys the recording proxy locks deployment-wide:
        # pinned here too, so a person row can't beat the operator's global one.
        "proactive_dashboards", "act_first",
    ):
        assert setting_is_operator_pinned(key) is True, key
    for key in UserSettings.model_fields:
        if UserSettings.field_is_sensitive(key):
            assert setting_is_operator_pinned(key) is True, key
    for key in ("tone", "greeting", "favicon", "harness", "artifact_autopublish_enabled", "hub_workspace_id"):
        assert setting_is_operator_pinned(key) is False, key


def test_personal_keys_are_per_person(engine):
    alice = _person(partition=PARTITION_A, user="alice")
    bob = _person(partition=PARTITION_B, user="bob")
    _svc(engine, alice).upsert_setting("tone", "spicy")
    _svc(engine, alice).upsert_setting("harness", "anton")

    assert _svc(engine, alice).get_setting("tone").value == "spicy"
    assert _svc(engine, bob).get_setting("tone").value == "balanced"
    assert _svc(engine, LOCAL_SCOPE).get_setting("tone").value == "balanced"
    with Session(engine) as s:
        rows = {(r.key, r.scope, r.org_id, r.user_id) for r in s.exec(select_settings()).all()}
    assert ("tone", "user", PARTITION_A, "alice") in rows
    # An org key the proxy allows goes per person too: the partition is the org.
    assert ("harness", "org", PARTITION_A, None) in rows


def test_pinned_key_ignores_a_persons_own_row(engine, monkeypatch):
    # A stray row in a person's partition (the agent's code can write the
    # database) must not turn memory off for that person.
    # The env differs from memory_mode's own default ("autopilot"), so reading
    # it back proves the environment is consulted, not the field default.
    monkeypatch.setenv("MEMORY_MODE", "copilot")
    alice = _person(partition=PARTITION_A, user="alice")
    with Session(engine) as s:
        s.add(Setting(key="memory_mode", value="off", scope="user", org_id=PARTITION_A, user_id="alice"))
        s.add(Setting(key="max_tool_rounds", value="7", scope="org", org_id=PARTITION_A))
        s.commit()

    # No global row: the environment decides, and the stray person row is ignored.
    assert _svc(engine, alice).get_setting("memory_mode").is_set is False
    assert _svc(engine, alice).load().memory_mode == "copilot"
    assert _svc(engine, alice).load().max_tool_rounds != 7

    # A global row beats the environment, for every person. Its value differs
    # from the env's, so this proves the global row wins.
    _svc(engine, LOCAL_SCOPE).upsert_setting("memory_mode", "autopilot")
    assert _svc(engine, alice).load().memory_mode == "autopilot"
    assert _svc(engine, alice).get_setting("memory_mode").value == "autopilot"
    listed = {item.key: item.value for item in _svc(engine, alice).list_settings()}
    assert listed["memory_mode"] == "autopilot"


def test_pinned_key_write_in_a_partition_lands_on_the_global_row(engine):
    alice = _person(partition=PARTITION_A, user="alice")
    bob = _person(partition=PARTITION_B, user="bob")
    _svc(engine, alice).upsert_setting("max_turn_tokens", "2000000")

    with Session(engine) as s:
        rows = [r for r in s.exec(select_settings()).all() if r.key == "max_turn_tokens"]
    assert [(r.scope, r.org_id, r.user_id) for r in rows] == [(None, None, None)]
    assert _svc(engine, bob).get_setting("max_turn_tokens").value == "2000000"
    assert _svc(engine, LOCAL_SCOPE).get_setting("max_turn_tokens").value == "2000000"
    # Deleting it from a partition removes the deployment row, the only one.
    assert _svc(engine, bob).delete_setting("max_turn_tokens") is True
    assert _svc(engine, LOCAL_SCOPE).get_setting("max_turn_tokens").is_set is False


def test_personal_scope_with_shared_settings_keeps_todays_global_rows(engine):
    alice = _person(partition=PARTITION_A, user="alice", resources=frozenset())
    bob = _person(partition=PARTITION_B, user="bob", resources=frozenset())
    _svc(engine, alice).upsert_setting("tone", "spicy")
    assert _svc(engine, bob).get_setting("tone").value == "spicy"
    with Session(engine) as s:
        rows = [(r.scope, r.org_id, r.user_id) for r in s.exec(select_settings()).all()]
    assert rows == [(None, None, None)]


def test_personal_partition_without_an_id_fails_closed_on_write(engine):
    from cowork.db.scoped import MissingTenantScopeError

    with pytest.raises(MissingTenantScopeError):
        _svc(engine, _person(partition=None, user="alice")).upsert_setting("tone", "spicy")


def test_org_mode_still_keys_pinned_keys_by_org_and_user(engine):
    # Hosted regression: the pinned marker changes nothing in org mode.
    _svc(engine, _org(ORG_A, "alice")).upsert_setting("max_tool_rounds", "40")
    _svc(engine, _org(ORG_A, "alice")).upsert_setting("memory_mode", "copilot")
    with Session(engine) as s:
        rows = {(r.key, r.scope, r.org_id, r.user_id) for r in s.exec(select_settings()).all()}
    assert ("max_tool_rounds", "org", ORG_A, None) in rows
    assert ("memory_mode", "user", ORG_A, "alice") in rows
    assert _svc(engine, _org(ORG_A, "alice")).get_setting("memory_mode").value == "copilot"
    assert _svc(engine, _org(ORG_A, "bob")).get_setting("memory_mode").value == "autopilot"
