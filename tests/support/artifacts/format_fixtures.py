"""Local format-test tenancy and trusted-header cleanup."""
from __future__ import annotations

import pytest

from cowork.common.settings.app_settings import get_app_settings


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()
