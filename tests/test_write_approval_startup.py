"""ANTON_WRITE_APPROVAL=true holds an agent cell that changes connected data or
existing files until the person approves it. anton reads the setting itself,
from this process's environment, so the server's part is to refuse to start
where the setting would hold nothing: an anton without it, no way to ask, or
turns on remote workers. Each would leave the deployment saying changes wait
for approval while every cell ran unheld, or was refused with no card.
"""
from __future__ import annotations

import pytest
from pydantic_settings import BaseSettings

import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.harnesses.anton_harness.settings import AntonHarnessSettings


@pytest.fixture
def approval_on(monkeypatch):
    monkeypatch.setenv("ANTON_WRITE_APPROVAL", "true")
    get_app_settings.cache_clear()
    yield monkeypatch
    get_app_settings.cache_clear()


class _AntonWithoutTheSetting(BaseSettings):
    model_config = {"env_prefix": "ANTON_", "extra": "ignore"}

    max_tool_rounds: int = 25


class _AntonWithTheSetting(_AntonWithoutTheSetting):
    write_approval: bool = False


def _refusal() -> str:
    get_app_settings.cache_clear()
    with pytest.raises(RuntimeError) as refused:
        cowork_server.create_app()
    return str(refused.value)


def test_the_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ANTON_WRITE_APPROVAL", raising=False)
    assert AntonHarnessSettings(_env_file=None).write_approval is False


def test_the_setting_reads_anton_write_approval(approval_on):
    assert AntonHarnessSettings(_env_file=None).write_approval is True


def test_an_anton_without_the_setting_is_refused(approval_on):
    import anton.core.settings

    approval_on.setattr(anton.core.settings, "CoreSettings", _AntonWithoutTheSetting)
    message = _refusal()
    assert message.startswith("ANTON_WRITE_APPROVAL=true needs an anton that holds writes")
    assert "run every cell without asking" in message


def test_an_anton_with_the_setting_starts(approval_on):
    import anton.core.settings

    approval_on.setattr(anton.core.settings, "CoreSettings", _AntonWithTheSetting)
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_questions_off_is_refused(approval_on):
    import anton.core.settings

    approval_on.setattr(anton.core.settings, "CoreSettings", _AntonWithTheSetting)
    approval_on.setenv("COWORK_ASK_USER_ENABLED", "false")
    assert "needs COWORK_ASK_USER_ENABLED" in _refusal()


def test_remote_turns_are_refused(approval_on):
    import anton.core.settings

    approval_on.setattr(anton.core.settings, "CoreSettings", _AntonWithTheSetting)
    approval_on.setenv("COWORK_TURN_BACKEND", "remote")
    approval_on.setenv("COWORK_TURN_QUEUE_URL", "redis://localhost:6379/0")
    message = _refusal()
    assert message.startswith("ANTON_WRITE_APPROVAL=true is not supported with COWORK_TURN_BACKEND=remote")


def test_off_checks_nothing(monkeypatch):
    """The desktop and hosted Cowork never set it, and start as before."""
    import anton.core.settings

    monkeypatch.delenv("ANTON_WRITE_APPROVAL", raising=False)
    monkeypatch.setattr(anton.core.settings, "CoreSettings", _AntonWithoutTheSetting)
    get_app_settings.cache_clear()
    cowork_server.create_app()
