"""COWORK_ARTIFACT_WEBSITES: off by default, reported by /health in MindsHub
mode only, and refused at start on an anton without the website mode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cowork import code_sandbox
from cowork.common.settings.app_settings import AppSettings
from cowork.common.settings.user_settings import Provider


def test_websites_are_off_by_default(monkeypatch):
    monkeypatch.delenv("COWORK_ARTIFACT_WEBSITES", raising=False)
    assert AppSettings().artifact_websites_enabled() is False


@pytest.mark.parametrize(
    ("value", "enabled"),
    [("on", True), ("ON", True), (" on ", True), ("off", False), ("", False)],
)
def test_the_setting_reads_on_and_off(monkeypatch, value, enabled):
    monkeypatch.setenv("COWORK_ARTIFACT_WEBSITES", value)
    assert AppSettings().artifact_websites_enabled() is enabled


def test_any_other_value_fails_settings_load(monkeypatch):
    monkeypatch.setenv("COWORK_ARTIFACT_WEBSITES", "yes")
    with pytest.raises(ValidationError):
        AppSettings()


def _health(*, tenancy_mode: str, websites: bool) -> dict:
    from cowork.api.v1.endpoints import health

    settings = SimpleNamespace(
        owner="", tenancy_mode=tenancy_mode, draft_links_enabled=lambda: False,
        artifact_websites_offered=lambda: tenancy_mode == "org" and websites,
    )
    user = SimpleNamespace(
        config_status={},
        resolved_planning_provider=Provider.OPENAI,
        resolved_coding_provider=Provider.OPENAI,
        resolved_router_provider=Provider.OPENAI,
    )
    with (
        patch.object(health, "get_app_settings", return_value=settings),
        patch.object(health, "get_user_settings", return_value=user),
    ):
        return health.health()


@pytest.mark.parametrize(
    ("tenancy_mode", "websites", "reported"),
    [("org", True, True), ("org", False, False), ("local", True, False)],
)
def test_health_reports_websites_in_mindshub_mode_with_the_setting_on(tenancy_mode, websites, reported):
    assert _health(tenancy_mode=tenancy_mode, websites=websites)["artifact_websites"] is reported


def test_websites_on_need_an_anton_with_the_website_mode(monkeypatch):
    import anton.publish_access as publish_access

    monkeypatch.delattr(publish_access, "WEBSITE_MODE", raising=False)
    with pytest.raises(RuntimeError) as refused:
        code_sandbox._check_artifact_websites_startup(SimpleNamespace(artifact_websites="on"))
    assert "COWORK_ARTIFACT_WEBSITES=on needs an anton with the website mode" in str(refused.value)


def test_websites_off_start_on_any_anton(monkeypatch):
    import anton.publish_access as publish_access

    monkeypatch.delattr(publish_access, "WEBSITE_MODE", raising=False)
    code_sandbox._check_artifact_websites_startup(SimpleNamespace(artifact_websites="off"))


def test_websites_on_start_on_the_pinned_anton():
    code_sandbox._check_artifact_websites_startup(SimpleNamespace(artifact_websites="on"))


def test_check_startup_runs_the_websites_check(monkeypatch):
    seen = []
    monkeypatch.setattr(code_sandbox, "_check_live_apps_startup", lambda settings: None)
    monkeypatch.setattr(code_sandbox, "_check_artifact_websites_startup", seen.append)
    settings = SimpleNamespace(trust_identity_headers=False, code_sandbox="off", artifact_websites="on")

    code_sandbox.check_startup(settings, remote_turns=False)

    assert seen == [settings]
