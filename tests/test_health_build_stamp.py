"""Tests for the /health build-stamp integration.

Verifies that config_status includes the build object from
~/.cowork/server-build-stamp.json when present, and null when absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import cowork.common.settings.user_settings as us
from cowork.common.settings.user_settings import (
    _BUILD_STAMP_PATH,
    _read_build_stamp,
    get_user_settings,
)


def test_read_build_stamp_returns_none_when_missing():
    assert _read_build_stamp() is None


def test_read_build_stamp_returns_dict_when_present():
    stamp_dir = _BUILD_STAMP_PATH.parent
    original_stamp = None
    if _BUILD_STAMP_PATH.is_file():
        original_stamp = _BUILD_STAMP_PATH.read_text()

    try:
        stamp_dir.mkdir(parents=True, exist_ok=True)
        stamp = {"hash": "abc1234", "installed_at": "2026-07-07T00:00:00Z", "channel": "dev"}
        _BUILD_STAMP_PATH.write_text(json.dumps(stamp))
        result = _read_build_stamp()
        assert result == stamp
    finally:
        if original_stamp is not None:
            _BUILD_STAMP_PATH.write_text(original_stamp)
        elif _BUILD_STAMP_PATH.is_file():
            _BUILD_STAMP_PATH.unlink()


def test_config_status_includes_build_when_stamp_exists():
    stamp = {"hash": "deadbeef", "installed_at": "2026-01-01T00:00:00Z", "channel": "test"}
    stamp_dir = _BUILD_STAMP_PATH.parent
    original_stamp = None
    if _BUILD_STAMP_PATH.is_file():
        original_stamp = _BUILD_STAMP_PATH.read_text()

    try:
        stamp_dir.mkdir(parents=True, exist_ok=True)
        _BUILD_STAMP_PATH.write_text(json.dumps(stamp))
        settings = get_user_settings()
        status = settings.config_status
        assert "build" in status
        assert status["build"] == stamp
    finally:
        if original_stamp is not None:
            _BUILD_STAMP_PATH.write_text(original_stamp)
        elif _BUILD_STAMP_PATH.is_file():
            _BUILD_STAMP_PATH.unlink()
        us._config_status_cache = None


def test_config_status_build_is_null_when_stamp_missing():
    original_existed = _BUILD_STAMP_PATH.is_file()
    original_content = None
    if original_existed:
        original_content = _BUILD_STAMP_PATH.read_text()
        _BUILD_STAMP_PATH.unlink()

    us._config_status_cache = None

    try:
        settings = get_user_settings()
        status = settings.config_status
        assert "build" in status
        assert status["build"] is None
    finally:
        if original_content is not None:
            _BUILD_STAMP_PATH.write_text(original_content)
