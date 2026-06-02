"""Tests for the self-update mechanism in cowork.updater."""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cowork.updater import (
    _current_version,
    _do_update_check,
    _find_uv,
    _latest_pypi_version,
    _parse_version_tuple,
    maybe_self_update,
    _LOOP_GUARD_VAR,
    _DISABLE_VAR,
    _PACKAGE_NAME,
)


# ---------------------------------------------------------------------------
# _parse_version_tuple
# ---------------------------------------------------------------------------

class TestParseVersionTuple:
    def test_simple(self):
        assert _parse_version_tuple("0.1.2") == (0, 1, 2)

    def test_two_part(self):
        assert _parse_version_tuple("1.0") == (1, 0)

    def test_prerelease_stripped(self):
        assert _parse_version_tuple("1.2.3a1") == (1, 2, 3)
        assert _parse_version_tuple("1.2.3rc2") == (1, 2, 3)

    def test_post_release_stripped(self):
        assert _parse_version_tuple("1.2.3.post1") == (1, 2, 3)

    def test_comparison_logic(self):
        assert _parse_version_tuple("0.1.3") > _parse_version_tuple("0.1.2")
        assert _parse_version_tuple("0.2.0") > _parse_version_tuple("0.1.9")
        assert _parse_version_tuple("1.0.0") > _parse_version_tuple("0.99.99")
        assert _parse_version_tuple("0.1.2") == _parse_version_tuple("0.1.2")


# ---------------------------------------------------------------------------
# _latest_pypi_version
# ---------------------------------------------------------------------------

class TestLatestPyPIVersion:
    def test_success(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(
            {"info": {"version": "0.2.0"}}
        ).encode()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("cowork.updater.urlopen", return_value=fake_resp):
            assert _latest_pypi_version() == "0.2.0"

    def test_network_error_returns_none(self):
        from urllib.error import URLError

        with patch("cowork.updater.urlopen", side_effect=URLError("no network")):
            assert _latest_pypi_version() is None

    def test_timeout_returns_none(self):
        with patch("cowork.updater.urlopen", side_effect=OSError("timed out")):
            assert _latest_pypi_version() is None

    def test_malformed_json_returns_none(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = b"not json"
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("cowork.updater.urlopen", return_value=fake_resp):
            assert _latest_pypi_version() is None

    def test_missing_key_returns_none(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({"other": "data"}).encode()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("cowork.updater.urlopen", return_value=fake_resp):
            assert _latest_pypi_version() is None


# ---------------------------------------------------------------------------
# _find_uv
# ---------------------------------------------------------------------------

class TestFindUv:
    def test_found_on_path(self):
        with patch("shutil.which", return_value="/usr/bin/uv"):
            assert _find_uv() == "/usr/bin/uv"

    def test_fallback_locations(self):
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.access", return_value=True):
            result = _find_uv()
            assert result is not None
            assert "uv" in result

    def test_not_found(self):
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.is_file", return_value=False):
            assert _find_uv() is None


# ---------------------------------------------------------------------------
# _do_update_check (integration of the pieces)
# ---------------------------------------------------------------------------

class TestDoUpdateCheck:
    def test_loop_guard_prevents_recheck(self):
        """After a re-exec, the env var prevents an infinite loop."""
        with patch.dict(os.environ, {_LOOP_GUARD_VAR: "1"}):
            # Should return immediately without hitting PyPI
            with patch("cowork.updater._latest_pypi_version") as mock_pypi:
                _do_update_check()
                mock_pypi.assert_not_called()

    def test_disable_env_var(self):
        """COWORK_SERVER_DISABLE_AUTOUPDATE=1 skips the check."""
        with patch.dict(os.environ, {_DISABLE_VAR: "1"}, clear=False):
            with patch("cowork.updater._latest_pypi_version") as mock_pypi:
                _do_update_check()
                mock_pypi.assert_not_called()

    def test_disable_env_var_true(self):
        with patch.dict(os.environ, {_DISABLE_VAR: "true"}, clear=False):
            with patch("cowork.updater._latest_pypi_version") as mock_pypi:
                _do_update_check()
                mock_pypi.assert_not_called()

    def test_pypi_unreachable_continues(self):
        """If PyPI is down, server boots on current version."""
        env = {k: v for k, v in os.environ.items()
               if k not in (_LOOP_GUARD_VAR, _DISABLE_VAR)}
        with patch.dict(os.environ, env, clear=True), \
             patch("cowork.updater._current_version", return_value="0.1.2"), \
             patch("cowork.updater._latest_pypi_version", return_value=None), \
             patch("cowork.updater._find_uv") as mock_uv:
            _do_update_check()
            mock_uv.assert_not_called()  # should not attempt upgrade

    def test_already_up_to_date(self):
        """No upgrade when current == latest."""
        env = {k: v for k, v in os.environ.items()
               if k not in (_LOOP_GUARD_VAR, _DISABLE_VAR)}
        with patch.dict(os.environ, env, clear=True), \
             patch("cowork.updater._current_version", return_value="0.1.2"), \
             patch("cowork.updater._latest_pypi_version", return_value="0.1.2"), \
             patch("cowork.updater._find_uv") as mock_uv:
            _do_update_check()
            mock_uv.assert_not_called()

    def test_newer_version_no_uv(self):
        """If uv is missing, log warning but don't crash."""
        env = {k: v for k, v in os.environ.items()
               if k not in (_LOOP_GUARD_VAR, _DISABLE_VAR)}
        with patch.dict(os.environ, env, clear=True), \
             patch("cowork.updater._current_version", return_value="0.1.2"), \
             patch("cowork.updater._latest_pypi_version", return_value="0.2.0"), \
             patch("cowork.updater._find_uv", return_value=None), \
             patch("subprocess.run") as mock_run:
            _do_update_check()
            mock_run.assert_not_called()

    def test_uv_upgrade_failure_continues(self):
        """If `uv tool install --upgrade` fails, server still boots."""
        env = {k: v for k, v in os.environ.items()
               if k not in (_LOOP_GUARD_VAR, _DISABLE_VAR)}
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="some error"
        )
        with patch.dict(os.environ, env, clear=True), \
             patch("cowork.updater._current_version", return_value="0.1.2"), \
             patch("cowork.updater._latest_pypi_version", return_value="0.2.0"), \
             patch("cowork.updater._find_uv", return_value="/usr/bin/uv"), \
             patch("subprocess.run", return_value=failed), \
             patch("os.execv") as mock_execv:
            _do_update_check()
            mock_execv.assert_not_called()  # should NOT re-exec on failure

    def test_successful_upgrade_reexecs(self):
        """Happy path: newer version found, uv succeeds, re-exec happens."""
        env = {k: v for k, v in os.environ.items()
               if k not in (_LOOP_GUARD_VAR, _DISABLE_VAR)}
        success = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        with patch.dict(os.environ, env, clear=True), \
             patch("cowork.updater._current_version", return_value="0.1.2"), \
             patch("cowork.updater._latest_pypi_version", return_value="0.2.0"), \
             patch("cowork.updater._find_uv", return_value="/usr/bin/uv"), \
             patch("subprocess.run", return_value=success) as mock_run, \
             patch("os.execv") as mock_execv:
            _do_update_check()
            # Verify uv was called correctly
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args == ["/usr/bin/uv", "tool", "install", "--upgrade", _PACKAGE_NAME]
            # Verify re-exec
            mock_execv.assert_called_once()
            # Verify loop guard was set
            assert os.environ.get(_LOOP_GUARD_VAR) == "1"

    def test_subprocess_timeout_does_not_crash(self):
        """If uv hangs and times out, server still boots."""
        env = {k: v for k, v in os.environ.items()
               if k not in (_LOOP_GUARD_VAR, _DISABLE_VAR)}
        with patch.dict(os.environ, env, clear=True), \
             patch("cowork.updater._current_version", return_value="0.1.2"), \
             patch("cowork.updater._latest_pypi_version", return_value="0.2.0"), \
             patch("cowork.updater._find_uv", return_value="/usr/bin/uv"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("uv", 120)):
            # Should be caught by maybe_self_update's outer try/except
            maybe_self_update()  # must not raise


# ---------------------------------------------------------------------------
# maybe_self_update (top-level wrapper)
# ---------------------------------------------------------------------------

class TestMaybeSelfUpdate:
    def test_swallows_all_exceptions(self):
        """The outer wrapper must never let an exception escape."""
        with patch("cowork.updater._do_update_check", side_effect=RuntimeError("boom")):
            maybe_self_update()  # must not raise

    def test_swallows_keyboard_interrupt(self):
        """Even KeyboardInterrupt during update check shouldn't propagate
        (it's caught by the broad except Exception — but KeyboardInterrupt
        is BaseException, so let's verify behavior)."""
        with patch("cowork.updater._do_update_check", side_effect=KeyboardInterrupt):
            # KeyboardInterrupt is BaseException, not Exception, so it WILL propagate.
            # This documents the current behavior — if this is undesirable, the
            # code should catch BaseException instead.
            with pytest.raises(KeyboardInterrupt):
                maybe_self_update()
