"""/health says whether the agent works on docs on this deployment: always on
the desktop; in org mode only with remote docs on and ask_user on, since doc
tools answer the pod on the stdin only an interactive turn keeps open."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import Provider


def _user_settings():
    return SimpleNamespace(
        config_status={},
        resolved_planning_provider=Provider.OPENAI,
        resolved_coding_provider=Provider.OPENAI,
        resolved_router_provider=Provider.OPENAI,
    )


def _app_settings(**overrides):
    fields = {"owner": "", "tenancy_mode": "local", "draft_links_enabled": lambda: False,
              "artifact_websites_offered": lambda: False, "remote_docs": False, "ask_user_enabled": True}
    fields.update(overrides)
    fields["remote_docs_available"] = lambda: bool(fields["remote_docs"] and fields["ask_user_enabled"])
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("tenancy,remote_docs,ask_user,expected", [
    ("local", False, False, True),
    ("local", True, True, True),
    ("org", False, True, False),
    ("org", True, False, False),
    ("org", True, True, True),
])
def test_docs_enabled(tenancy, remote_docs, ask_user, expected):
    from cowork.api.v1.endpoints import health

    settings = _app_settings(tenancy_mode=tenancy, remote_docs=remote_docs, ask_user_enabled=ask_user)
    with (
        patch.object(health, "get_app_settings", return_value=settings),
        patch.object(health, "get_user_settings", return_value=_user_settings()),
    ):
        body = health.health()
    assert body["docs_enabled"] is expected


def test_remote_docs_is_off_unless_cowork_remote_docs_turns_it_on(monkeypatch):
    monkeypatch.delenv("COWORK_REMOTE_DOCS", raising=False)
    get_app_settings.cache_clear()
    try:
        assert get_app_settings().remote_docs is False
        monkeypatch.setenv("COWORK_REMOTE_DOCS", "true")
        get_app_settings.cache_clear()
        assert get_app_settings().remote_docs is True
    finally:
        get_app_settings.cache_clear()
