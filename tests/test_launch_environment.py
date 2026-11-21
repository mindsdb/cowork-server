"""A local launch moves the server's own COWORK_* and MINDSHUB_* variables out
of its environment (cowork.common.launch_environment), so git, project
commands, terminals, the Codex app-server and anton's cells inherit none of
them, while the server keeps reading the values it started with.

Each runtime reader of those variables is pinned below against a scrubbed
environment: the live variable is gone and the captured value is all there is.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import MappingProxyType

import pytest

from cowork.common import launch_environment
from cowork.common.launch_environment import launch_value, scrub_launch_environment


@pytest.fixture
def scrubbed(monkeypatch):
    """Stand in for a launch that captured ``values`` and removed them from
    os.environ. Returns a setter."""

    def scrub(**values: str) -> None:
        for name in values:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(launch_environment, "_captured", MappingProxyType(dict(values)))

    monkeypatch.setattr(launch_environment, "_captured", MappingProxyType({}))
    return scrub


class TestLaunchValue:
    def test_the_live_environment_wins(self, scrubbed, monkeypatch):
        scrubbed(COWORK_PROBE="captured")
        monkeypatch.setenv("COWORK_PROBE", "live")
        assert launch_value("COWORK_PROBE") == "live"

    def test_the_captured_value_answers_once_the_variable_is_gone(self, scrubbed):
        scrubbed(COWORK_PROBE="captured")
        assert launch_value("COWORK_PROBE") == "captured"

    def test_nothing_answers_none(self, scrubbed):
        scrubbed()
        assert launch_value("COWORK_PROBE") is None


def test_the_scrub_moves_only_its_prefixes(monkeypatch):
    fake_environment = {
        "COWORK_HOME": "/srv/cowork",
        "MINDSHUB_CODEX_API_KEY": "fake0000",
        "ANTON_MINDS_API_KEY": "fake1111",
        "ANTHROPIC_API_KEY": "fake2222",
        "PATH": "/usr/bin",
        "XCOWORK_HOME": "kept",
    }
    monkeypatch.setattr(os, "environ", fake_environment)
    monkeypatch.setattr(launch_environment, "_captured", MappingProxyType({}))

    scrub_launch_environment()

    assert fake_environment == {
        "ANTON_MINDS_API_KEY": "fake1111",
        "ANTHROPIC_API_KEY": "fake2222",
        "PATH": "/usr/bin",
        "XCOWORK_HOME": "kept",
    }
    assert dict(launch_environment.captured_environment()) == {
        "COWORK_HOME": "/srv/cowork",
        "MINDSHUB_CODEX_API_KEY": "fake0000",
    }


class TestRuntimeReaders:
    """Each reader that runs after the scrub, given only the captured value."""

    def test_cowork_home(self, scrubbed, tmp_path):
        from cowork.common.paths import cowork_home

        scrubbed(COWORK_HOME=str(tmp_path / "account"))
        assert cowork_home() == tmp_path / "account"

    def test_settings_built_after_the_scrub(self, scrubbed, tmp_path):
        from cowork.common.settings.app_settings import (
            CodingSettings,
            ConnectorSettings,
            OAuthSettings,
            TurnQueueSettings,
        )

        scrubbed(
            COWORK_CODING_DIR=str(tmp_path / "org" / "coding"),
            COWORK_VAULT_DIR=str(tmp_path / "org" / "data-vault"),
            COWORK_SERVER_ORIGIN="http://127.0.0.1:41234",
            COWORK_TURN_JEV_ROUTE_MODEL="jev-probe",
        )
        assert CodingSettings().root_dir == str(tmp_path / "org" / "coding")
        assert ConnectorSettings().vault_dir == str(tmp_path / "org" / "data-vault")
        assert OAuthSettings().server_origin == "http://127.0.0.1:41234"
        # An env_prefix class reads the captured values under its prefix.
        assert TurnQueueSettings().jev_route_model == "jev-probe"

    def test_settings_rank_the_live_environment_first(self, scrubbed, monkeypatch, tmp_path):
        from cowork.common.settings.app_settings import CodingSettings

        scrubbed(COWORK_CODING_DIR=str(tmp_path / "captured"))
        monkeypatch.setenv("COWORK_CODING_DIR", str(tmp_path / "live"))
        assert CodingSettings().root_dir == str(tmp_path / "live")

    def test_the_turn_idle_bound(self, scrubbed):
        from cowork.streaming.registry import _idle_bound_seconds

        scrubbed(COWORK_MAX_TURN_IDLE_SECONDS="1234")
        assert _idle_bound_seconds() == 1234

    def test_the_redis_url(self, scrubbed):
        from cowork.turnqueue import redis_client

        scrubbed(COWORK_TURN_REDIS_URL="redis://redis.example.test:6380/2")
        assert redis_client._url() == "redis://redis.example.test:6380/2"

    def test_the_codex_compaction_limit(self, scrubbed):
        from cowork.coding.engines import codex_config

        scrubbed(MINDSHUB_CODE_AUTO_COMPACT_TOKEN_LIMIT="48000")
        assert codex_config.auto_compact_token_limit() == 48000

    def test_the_connector_test_allowlist(self, scrubbed):
        from cowork.services.connectors import connector_test_run

        scrubbed(COWORK_CONNECTOR_TEST_PRIVATE_ALLOWLIST="db.internal,10.1.0.0/16")
        connector_test_run.private_allowlist.cache_clear()
        try:
            hosts, networks = connector_test_run.private_allowlist()
        finally:
            connector_test_run.private_allowlist.cache_clear()
        assert "db.internal" in hosts
        assert [str(network) for network in networks] == ["10.1.0.0/16"]

    def test_the_developer_oauth_file_pointer(self, scrubbed, monkeypatch, tmp_path):
        from pathlib import Path

        from cowork.common.settings import app_settings

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        scrubbed(COWORK_DEV_OAUTH_ENV_FILE=str(tmp_path / ".cowork-dev" / ".env"))
        assert app_settings._dev_oauth_env_path() == tmp_path / ".cowork-dev" / ".env"


_LAUNCH = """
import json, os, subprocess, sys
import uvicorn
uvicorn.run = lambda *a, **k: None
from cowork import cli
cli.main()
from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import CodingSettings
child = subprocess.run(
    [sys.executable, "-c", "import json, os; print(json.dumps(sorted(os.environ)))"],
    capture_output=True, text=True, check=True,
)
child_names = json.loads(child.stdout)
print("RESULT" + json.dumps({
    "own": sorted(name for name in os.environ if name.startswith(("COWORK_", "MINDSHUB_"))),
    "child": [name for name in child_names if name.startswith(("COWORK_", "MINDSHUB_"))],
    "anton_own": os.environ.get("ANTON_PROBE_MARKER"),
    "anton_child": "ANTON_PROBE_MARKER" in child_names,
    "home": str(cowork_home()),
    "coding_dir": CodingSettings().root_dir,
}))
"""


def test_a_local_launch_leaves_no_cowork_or_mindshub_variable_to_its_children(tmp_path):
    """cowork.cli.main with uvicorn stubbed out: the server's own environment,
    and a process it starts afterwards, hold none of its variables, anton's
    stay, and the server still reads the values it started with."""
    env = dict(os.environ)
    env.update({
        "COWORK_HOME": str(tmp_path / "home"),
        "COWORK_CODING_DIR": str(tmp_path / "org" / "coding"),
        "MINDSHUB_PROBE_MARKER": "fake-mindshub",
        "ANTON_PROBE_MARKER": "fake-anton",
    })
    env.pop("COWORK_APP_SECRET_STDIN", None)
    result = subprocess.run(
        [sys.executable, "-c", _LAUNCH],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT"))
    seen = json.loads(line[len("RESULT"):])

    assert seen["own"] == []
    assert seen["child"] == []
    assert seen["anton_own"] == "fake-anton"
    assert seen["anton_child"] is True
    assert seen["home"] == str(tmp_path / "home")
    assert seen["coding_dir"] == str(tmp_path / "org" / "coding")

