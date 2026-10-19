"""ANTON_WRITE_APPROVAL=true holds an agent cell that changes connected data or
existing files until the person approves it. anton reads the setting itself,
through its own AntonSettings, so the server's part is to refuse to start
where the setting would hold nothing: an anton without it, an anton that
doesn't read the value this server read, no way to ask, or turns on remote
workers. Each would leave the deployment saying changes wait for approval
while every cell ran unheld, or was refused with no card.
"""
from __future__ import annotations

import pytest
from pydantic_settings import BaseSettings

import cowork.server as cowork_server
from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import get_app_settings
from cowork.harnesses.anton_harness.settings import AntonHarnessSettings


@pytest.fixture
def approval_on(monkeypatch):
    monkeypatch.setenv("ANTON_WRITE_APPROVAL", "true")
    get_app_settings.cache_clear()
    yield monkeypatch
    get_app_settings.cache_clear()


@pytest.fixture
def cowork_home_env():
    """`<COWORK_HOME>/.env`, put back afterwards: COWORK_HOME is session-scoped
    (see conftest), so a line left behind would follow every later test.
    """
    path = cowork_home() / ".env"
    original = path.read_text(encoding="utf-8") if path.exists() else None
    try:
        yield path
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original, encoding="utf-8")
        get_app_settings.cache_clear()


class _AntonWithoutTheSetting(BaseSettings):
    """Stands in for anton's AntonSettings. Reads the process environment and
    no .env file."""

    model_config = {"env_prefix": "ANTON_", "extra": "ignore"}

    max_tool_rounds: int = 25


class _AntonWithTheSetting(_AntonWithoutTheSetting):
    write_approval: bool = False


def _use_anton(*, monkeypatch: pytest.MonkeyPatch, anton_settings: type[BaseSettings]) -> None:
    import anton.config.settings

    monkeypatch.setattr(anton.config.settings, "AntonSettings", anton_settings)


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
    _use_anton(monkeypatch=approval_on, anton_settings=_AntonWithoutTheSetting)
    message = _refusal()
    assert message.startswith("ANTON_WRITE_APPROVAL=true needs an anton that holds writes")
    assert "run every cell without asking" in message
    assert "leave ANTON_WRITE_APPROVAL off until then" in message


def test_an_anton_with_the_setting_starts(approval_on):
    _use_anton(monkeypatch=approval_on, anton_settings=_AntonWithTheSetting)
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_a_value_anton_does_not_read_is_refused(monkeypatch, tmp_path, cowork_home_env):
    """The value sits only in <COWORK_HOME>/.env, which this server reads and
    anton's AntonSettings doesn't once COWORK_HOME is set to a folder other
    than ~/.cowork, so every session would get the setting off."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTON_WRITE_APPROVAL", raising=False)
    original = cowork_home_env.read_text(encoding="utf-8") if cowork_home_env.exists() else ""
    cowork_home_env.write_text(f"{original}\nANTON_WRITE_APPROVAL=true\n", encoding="utf-8")
    _use_anton(monkeypatch=monkeypatch, anton_settings=_AntonWithTheSetting)

    assert AntonHarnessSettings().write_approval is True
    message = _refusal()
    assert message.startswith("ANTON_WRITE_APPROVAL=true is set where anton doesn't read it")
    assert "in the server's process environment" in message


def test_questions_off_is_refused(approval_on):
    _use_anton(monkeypatch=approval_on, anton_settings=_AntonWithTheSetting)
    approval_on.setenv("COWORK_ASK_USER_ENABLED", "false")
    assert "needs COWORK_ASK_USER_ENABLED" in _refusal()


def test_remote_turns_are_refused(approval_on):
    _use_anton(monkeypatch=approval_on, anton_settings=_AntonWithTheSetting)
    approval_on.setenv("COWORK_TURN_BACKEND", "remote")
    message = _refusal()
    assert message.startswith("ANTON_WRITE_APPROVAL=true is not supported with COWORK_TURN_BACKEND=remote")


def test_off_checks_nothing(monkeypatch):
    """The desktop and hosted Cowork never set it, and start as before."""
    monkeypatch.delenv("ANTON_WRITE_APPROVAL", raising=False)
    _use_anton(monkeypatch=monkeypatch, anton_settings=_AntonWithoutTheSetting)
    get_app_settings.cache_clear()
    cowork_server.create_app()
