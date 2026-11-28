"""Opt-in fixtures imported explicitly by artifact test modules."""
from __future__ import annotations

import pytest

from cowork.common.settings.app_settings import get_app_settings


@pytest.fixture
def local_mode(monkeypatch):
    """Use local tenancy for one test and clear its cached settings afterward."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()
