"""The desktop app's launch hand-off (cowork.app_secret, COWORK_APP_SECRET_STDIN=2).

The app writes one stdin line carrying the app secret, the window's bearer
token and the window proof. A server started with value 2 reads it whatever
its settings say about tenancy, takes its bearer token from it alone, and
refuses to start when the line is missing, late, malformed or incomplete.
Value 1 and no value behave as before (tests/test_app_held_mcp_secrets.py).

Every secret here is a fixed fake hex value.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork import app_secret
from cowork.api.v1.router import api_router
from cowork.auth_middleware import _read_token
from cowork.common import launch_environment
from cowork.common.settings.app_settings import get_app_settings

SECRET = "ab" * 32
TOKEN = "cd" * 32
PROOF = "ef" * 32
OLD_ENV_TOKEN = "0123456789abcdef" * 2
PROCESS_ENV_TOKEN = "89" * 32
GOOD = {"app_secret": SECRET, "auth_token": TOKEN, "window_proof": PROOF}


def _line(payload) -> str:
    return json.dumps(payload) + "\n"


@pytest.fixture
def nothing_held(monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", None)
    monkeypatch.setattr(app_secret, "_handoff", None)


@pytest.fixture
def handoff_held(monkeypatch):
    handoff = app_secret.LaunchHandoff.model_validate(GOOD)
    monkeypatch.setattr(app_secret, "_handoff", handoff)
    monkeypatch.setattr(app_secret, "_secret", handoff.app_secret)
    return handoff


class TestContract:
    def test_the_names_the_desktop_mirrors(self):
        assert app_secret.STDIN_FLAG == "COWORK_APP_SECRET_STDIN"
        assert app_secret.WINDOW_PROOF_HEADER == "X-Cowork-Window-Proof"
        assert app_secret.SECRET_PROTOCOL == 1
        assert app_secret.HANDOFF_PROTOCOL == 2
        # The desktop finds the server's refusal line in its output by this
        # prefix, and shows the line.
        assert app_secret.SERVER_REFUSAL_PREFIX == "cowork-server: "

    def test_the_hand_off_keys_the_desktop_writes(self):
        """The desktop builds the hand-off line with these keys, in
        launchStdinLine (src/shared/server-launch.ts) and in
        scripts/start-server.mjs. A key renamed on one side stops every
        desktop launch with "has no <key>"."""
        assert set(app_secret.LaunchHandoff.model_fields) == {"app_secret", "auth_token", "window_proof"}


class TestRead:
    def test_a_good_line_is_held(self, nothing_held):
        handoff = app_secret.read_handoff_from_stdin(stream=io.StringIO(_line(GOOD)))

        assert app_secret.held_handoff() == handoff
        assert (handoff.app_secret, handoff.auth_token, handoff.window_proof) == (SECRET, TOKEN, PROOF)
        # The hand-off's app secret is the one every app-secret route checks.
        assert app_secret.get_app_secret() == SECRET
        assert app_secret.held_protocol() == app_secret.HANDOFF_PROTOCOL
        assert not {SECRET, TOKEN, PROOF} & set(os.environ.values())

    def test_its_repr_names_no_secret(self, nothing_held):
        handoff = app_secret.read_handoff_from_stdin(stream=io.StringIO(_line(GOOD)))
        assert not any(value in repr(handoff) for value in (SECRET, TOKEN, PROOF))

    def test_an_extra_key_is_ignored(self, nothing_held):
        line = _line({**GOOD, "hooks_token": "12" * 32, "future": {"nested": [1, 2]}})
        assert app_secret.read_handoff_from_stdin(stream=io.StringIO(line)).auth_token == TOKEN

    @pytest.mark.parametrize("key", ["app_secret", "auth_token", "window_proof"])
    def test_a_missing_key_is_named(self, nothing_held, key):
        payload = {name: value for name, value in GOOD.items() if name != key}
        with pytest.raises(app_secret.LaunchHandoffError) as refused:
            app_secret.read_handoff_from_stdin(stream=io.StringIO(_line(payload)))
        assert str(refused.value) == f"the launch hand-off on stdin has no {key}"
        assert app_secret.held_handoff() is None
        assert app_secret.get_app_secret() is None

    @pytest.mark.parametrize(
        "value",
        ["", "ab" * 15, "ab" * 65, "AB" * 32, "zz" * 32, 1234, None, ["ab" * 32]],
        ids=["empty", "short", "long", "uppercase", "not-hex", "number", "null", "list"],
    )
    @pytest.mark.parametrize("key", ["app_secret", "auth_token", "window_proof"])
    def test_an_invalid_value_is_named_without_its_value(self, nothing_held, key, value):
        with pytest.raises(app_secret.LaunchHandoffError) as refused:
            app_secret.read_handoff_from_stdin(stream=io.StringIO(_line({**GOOD, key: value})))
        assert str(refused.value) == f"the launch hand-off on stdin has an invalid {key}"
        assert app_secret.held_handoff() is None
        assert app_secret.get_app_secret() is None

    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("{not json\n", "the launch hand-off on stdin is not valid JSON"),
            ('{"app_secret": "' + SECRET + '"\n', "the launch hand-off on stdin is not valid JSON"),
            ("[1]\n", "the launch hand-off on stdin is not a JSON object"),
            ("null\n", "the launch hand-off on stdin is not a JSON object"),
            ("", "the launch hand-off on stdin is empty"),
            ("\n", "the launch hand-off on stdin is empty"),
        ],
        ids=["malformed", "truncated", "array", "null", "eof", "blank"],
    )
    def test_a_malformed_or_empty_line_is_refused(self, nothing_held, line, message):
        with pytest.raises(app_secret.LaunchHandoffError) as refused:
            app_secret.read_handoff_from_stdin(stream=io.StringIO(line))
        assert str(refused.value) == message
        assert SECRET not in str(refused.value)
        assert app_secret.held_handoff() is None

    @pytest.mark.parametrize("errors", ["strict", "surrogateescape"])
    def test_bytes_that_are_not_utf8_are_named(self, nothing_held, errors):
        """stdin decodes strictly under a UTF-8 locale and with surrogateescape
        under the C locale; either way the refusal names the encoding."""
        raw = ('{"app_secret": "' + SECRET + '", "note": "').encode() + b"\xff\xfe" + b'"}\n'
        stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors=errors)
        with pytest.raises(app_secret.LaunchHandoffError) as refused:
            app_secret.read_handoff_from_stdin(stream=stream)
        assert str(refused.value) == "the launch hand-off on stdin is not valid UTF-8"
        assert app_secret.held_handoff() is None

    def test_no_stdin_is_named_at_once(self, nothing_held, monkeypatch):
        """A parent that starts the server with fd 0 closed leaves sys.stdin
        None: the refusal says so without waiting out the timeout."""
        monkeypatch.setattr(sys, "stdin", None)
        with pytest.raises(app_secret.LaunchHandoffError) as refused:
            app_secret.read_handoff_from_stdin(timeout=30)
        assert str(refused.value) == "no launch hand-off can arrive: the server has no stdin"
        assert app_secret.held_handoff() is None
        # The app secret alone is optional: without stdin the server runs in legacy mode.
        assert app_secret.read_from_stdin(timeout=30) is False
        assert app_secret.get_app_secret() is None

    @pytest.mark.parametrize(
        "line",
        [
            "a" * app_secret._MAX_LINE,
            _line({**GOOD, "padding": "b" * app_secret._MAX_LINE}),
        ],
        ids=["bound-without-newline", "past-the-bound"],
    )
    def test_a_line_at_the_bound_is_named_too_long(self, nothing_held, line):
        with pytest.raises(app_secret.LaunchHandoffError) as refused:
            app_secret.read_handoff_from_stdin(stream=io.StringIO(line))
        assert str(refused.value) == f"the launch hand-off on stdin is {app_secret._MAX_LINE} characters or longer"
        assert app_secret.held_handoff() is None

    def test_a_line_just_under_the_bound_is_read(self, nothing_held):
        padded = _line({**GOOD, "padding": ""})
        line = _line({**GOOD, "padding": "b" * (app_secret._MAX_LINE - len(padded))})
        assert len(line) == app_secret._MAX_LINE
        assert app_secret.read_handoff_from_stdin(stream=io.StringIO(line)).auth_token == TOKEN

    def test_a_parent_that_never_writes_is_refused_after_the_timeout(self, nothing_held):
        read_end, write_end = os.pipe()
        never_written = os.fdopen(read_end)
        try:
            with pytest.raises(app_secret.LaunchHandoffError) as refused:
                app_secret.read_handoff_from_stdin(stream=never_written, timeout=0.2)
        finally:
            # EOF ends the reader thread still blocked on the pipe.
            os.close(write_end)
            never_written.close()
        assert str(refused.value) == "no launch hand-off arrived on stdin within 0.2 s"
        assert app_secret.held_handoff() is None


class TestWindowProof:
    def test_only_the_held_proof_matches(self, handoff_held):
        assert app_secret.window_proof_matches(given=PROOF)
        assert not app_secret.window_proof_matches(given="12" * 32)
        assert not app_secret.window_proof_matches(given=PROOF[:-2])
        assert not app_secret.window_proof_matches(given="")
        assert not app_secret.window_proof_matches(given=None)
        # Neither of the hand-off's other secrets stands in for the proof.
        assert not app_secret.window_proof_matches(given=SECRET)
        assert not app_secret.window_proof_matches(given=TOKEN)

    def test_nothing_matches_without_a_hand_off(self, monkeypatch):
        monkeypatch.setattr(app_secret, "_handoff", None)
        monkeypatch.setattr(app_secret, "_secret", SECRET)
        assert not app_secret.window_proof_matches(given=PROOF)


def _health_client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


class TestHealth:
    def test_reports_protocol_2_while_the_hand_off_is_held(self, sharepoint_stores, handoff_held):
        body = _health_client().get("/api/v1/health/").json()
        assert body["app_secret_protocol"] == app_secret.HANDOFF_PROTOCOL
        assert not any(value in json.dumps(body) for value in (SECRET, TOKEN, PROOF))

    def test_reports_protocol_1_for_the_app_secret_alone(self, sharepoint_stores, monkeypatch):
        monkeypatch.setattr(app_secret, "_handoff", None)
        monkeypatch.setattr(app_secret, "_secret", SECRET)
        assert _health_client().get("/api/v1/health/").json()["app_secret_protocol"] == 1

    def test_omits_the_field_without_either(self, sharepoint_stores, nothing_held):
        assert "app_secret_protocol" not in _health_client().get("/api/v1/health/").json()


@pytest.fixture
def local_home(monkeypatch, tmp_path):
    """A desktop-like local start whose cowork-home .env still holds the token
    an earlier launch mirrored there."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"KEEP_ME=1\nCOWORK_AUTH_TOKEN={OLD_ENV_TOKEN}\n", encoding="utf-8")
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    monkeypatch.delenv("COWORK_AUTH_TOKEN", raising=False)
    get_app_settings.cache_clear()
    yield home
    get_app_settings.cache_clear()


class TestCreateApp:
    @pytest.mark.parametrize("require_auth", ["true", "false"])
    def test_a_held_hand_off_is_the_only_token(self, local_home, handoff_held, monkeypatch, require_auth):
        """With the hand-off the bearer check is on whatever the settings say,
        and nothing generates, mirrors or reads the .env token. A
        COWORK_AUTH_TOKEN in the process environment, live or captured at
        launch, is no token either."""
        from cowork import server

        monkeypatch.setenv("COWORK_REQUIRE_AUTH", require_auth)
        monkeypatch.setenv("COWORK_AUTH_TOKEN", PROCESS_ENV_TOKEN)
        monkeypatch.setattr(
            launch_environment, "_captured", MappingProxyType({"COWORK_AUTH_TOKEN": PROCESS_ENV_TOKEN})
        )
        get_app_settings.cache_clear()
        removed: list[Path] = []
        real_remove = server.remove_auth_token

        def remove(env_path):
            removed.append(env_path)
            return real_remove(env_path)

        def never(*args, **kwargs):
            raise AssertionError("a held hand-off must not generate or mirror a token")

        monkeypatch.setattr(server, "remove_auth_token", remove)
        monkeypatch.setattr(server, "ensure_auth_token", never)
        monkeypatch.setattr(server, "sync_auth_token", never)

        # No `with`: skipping the lifespan skips boot migrations, which would
        # collide with the schema conftest already created.
        client = TestClient(server.create_app())
        try:
            assert client.get("/api/v1/conversations/").status_code == 401
            old = client.get("/api/v1/conversations/", headers={"Authorization": f"Bearer {OLD_ENV_TOKEN}"})
            assert old.status_code == 401
            from_process = client.get(
                "/api/v1/conversations/", headers={"Authorization": f"Bearer {PROCESS_ENV_TOKEN}"}
            )
            assert from_process.status_code == 401
            authed = client.get("/api/v1/conversations/", headers={"Authorization": f"Bearer {TOKEN}"})
            assert authed.status_code == 200
        finally:
            client.close()

        assert removed == [local_home / ".env"]
        text = (local_home / ".env").read_text(encoding="utf-8")
        assert "COWORK_AUTH_TOKEN" not in text
        assert "KEEP_ME=1" in text
        assert TOKEN not in text

    def test_without_a_hand_off_the_env_token_is_mirrored_as_before(self, local_home, nothing_held, monkeypatch):
        from cowork import server

        monkeypatch.setenv("COWORK_REQUIRE_AUTH", "true")
        get_app_settings.cache_clear()
        calls: list[str] = []
        real_ensure, real_sync = server.ensure_auth_token, server.sync_auth_token
        monkeypatch.setattr(server, "ensure_auth_token", lambda path: calls.append("ensure") or real_ensure(path))
        monkeypatch.setattr(
            server, "sync_auth_token", lambda path, token: calls.append("sync") or real_sync(path, token)
        )

        client = TestClient(server.create_app())
        try:
            authed = client.get("/api/v1/conversations/", headers={"Authorization": f"Bearer {OLD_ENV_TOKEN}"})
            assert authed.status_code == 200
        finally:
            client.close()
        # ensure_auth_token finds the earlier copy, so nothing new is generated.
        assert calls == ["ensure", "sync"]
        assert _read_token(local_home / ".env") == OLD_ENV_TOKEN

    def test_org_tenancy_with_a_hand_off_refuses_to_start(self, local_home, handoff_held, monkeypatch):
        from cowork import server

        monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
        monkeypatch.setenv("COWORK_REQUIRE_AUTH", "false")
        get_app_settings.cache_clear()
        with pytest.raises(RuntimeError, match="nor is the desktop app's launch hand-off"):
            server.create_app()


_CLI = """
import functools, json, os, sys
import uvicorn
uvicorn.run = lambda *a, **k: print("UVICORN_STARTED")
from cowork import app_secret, cli
app_secret.read_handoff_from_stdin = functools.partial(app_secret.read_handoff_from_stdin, timeout=1.0)
cli.main()
handoff = app_secret.held_handoff()
print("HELD" + json.dumps({
    "protocol": app_secret.held_protocol(),
    "auth_token": handoff.auth_token if handoff else None,
    "app_secret": app_secret.get_app_secret(),
}))
"""


_ROOT = Path(__file__).resolve().parents[1]


def _cli_env(tmp_path: Path, flag: str | None) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name != app_secret.STDIN_FLAG}
    # The child runs in tmp_path, so without this it imports whatever cowork
    # the interpreter finds, which in a shared virtualenv is another checkout.
    env["PYTHONPATH"] = str(_ROOT)
    env["COWORK_HOME"] = str(tmp_path / "home")
    if flag is not None:
        env[app_secret.STDIN_FLAG] = flag
    return env


def _held(stdout: str) -> dict:
    line = next(line for line in stdout.splitlines() if line.startswith("HELD"))
    return json.loads(line[len("HELD"):])


class TestCli:
    """cowork.cli.main in a child process with uvicorn stubbed out, so the
    real startup order (flag, hand-off, .env, settings, scrub) is what runs."""

    def _run(self, tmp_path: Path, flag: str | None, stdin: str) -> subprocess.CompletedProcess[str]:
        (tmp_path / "home").mkdir(exist_ok=True)
        return subprocess.run(
            [sys.executable, "-c", _CLI],
            input=stdin, capture_output=True, text=True,
            env=_cli_env(tmp_path, flag), cwd=tmp_path, timeout=60, check=False,
        )

    def test_value_2_holds_the_hand_off(self, tmp_path):
        result = self._run(tmp_path, "2", _line(GOOD))
        assert result.returncode == 0, result.stderr
        assert "UVICORN_STARTED" in result.stdout
        assert _held(result.stdout) == {"protocol": 2, "auth_token": TOKEN, "app_secret": SECRET}

    def test_value_2_reads_stdin_even_when_the_env_file_names_org_tenancy(self, tmp_path):
        (tmp_path / "home").mkdir()
        (tmp_path / "home" / ".env").write_text("COWORK_TENANCY_MODE=org\n", encoding="utf-8")
        result = self._run(tmp_path, "2", _line(GOOD))
        assert result.returncode == 0, result.stderr
        assert _held(result.stdout)["protocol"] == 2

    @pytest.mark.parametrize(
        ("stdin", "message"),
        [
            (_line({"app_secret": SECRET, "auth_token": TOKEN}), "the launch hand-off on stdin has no window_proof"),
            (_line({**GOOD, "auth_token": "XYZ"}), "the launch hand-off on stdin has an invalid auth_token"),
            ("{not json\n", "the launch hand-off on stdin is not valid JSON"),
            ("", "the launch hand-off on stdin is empty"),
        ],
        ids=["missing-key", "invalid-value", "malformed", "empty"],
    )
    def test_value_2_exits_with_one_line_naming_the_problem(self, tmp_path, stdin, message):
        # A token an earlier launch mirrored into .env is no fallback.
        (tmp_path / "home").mkdir()
        (tmp_path / "home" / ".env").write_text(f"COWORK_AUTH_TOKEN={OLD_ENV_TOKEN}\n", encoding="utf-8")
        result = self._run(tmp_path, "2", stdin)
        assert result.returncode != 0
        assert "UVICORN_STARTED" not in result.stdout
        assert result.stderr.strip().splitlines() == [f"{app_secret.SERVER_REFUSAL_PREFIX}{message}"]

    def test_value_2_exits_when_the_line_never_arrives(self, tmp_path):
        (tmp_path / "home").mkdir()
        process = subprocess.Popen(
            [sys.executable, "-c", _CLI],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=_cli_env(tmp_path, "2"), cwd=tmp_path,
        )
        try:
            returncode = process.wait(timeout=60)
        finally:
            process.stdin.close()
        stdout, stderr = process.stdout.read(), process.stderr.read()
        process.stdout.close()
        process.stderr.close()
        assert returncode != 0
        assert "UVICORN_STARTED" not in stdout
        assert stderr.strip().splitlines() == [
            f"{app_secret.SERVER_REFUSAL_PREFIX}no launch hand-off arrived on stdin within 1 s"
        ]

    @pytest.mark.skipif(sys.platform == "win32", reason="closes fd 0 in the child, which Windows can't do")
    def test_value_2_exits_with_one_line_when_stdin_is_closed(self, tmp_path):
        (tmp_path / "home").mkdir()
        result = subprocess.run(
            [sys.executable, "-c", _CLI],
            capture_output=True, text=True, env=_cli_env(tmp_path, "2"), cwd=tmp_path, timeout=60,
            check=False, preexec_fn=lambda: os.close(0),
        )
        assert result.returncode != 0
        assert "UVICORN_STARTED" not in result.stdout
        assert result.stderr.strip().splitlines() == [
            f"{app_secret.SERVER_REFUSAL_PREFIX}no launch hand-off can arrive: the server has no stdin"
        ]

    def test_value_1_holds_the_app_secret_alone(self, tmp_path):
        result = self._run(tmp_path, "1", _line(GOOD))
        assert result.returncode == 0, result.stderr
        assert _held(result.stdout) == {"protocol": 1, "auth_token": None, "app_secret": SECRET}

    def test_no_value_reads_nothing(self, tmp_path):
        result = self._run(tmp_path, None, _line(GOOD))
        assert result.returncode == 0, result.stderr
        assert _held(result.stdout) == {"protocol": None, "auth_token": None, "app_secret": None}
