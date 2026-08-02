from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anton.core.datasources.data_vault import LocalDataVault

from cowork.harnesses.anton_harness.tools import _cowork_label_connection


class TestLabelConnectionToolReportsActualValue:
    @pytest.mark.asyncio
    async def test_reports_suffixed_label(self, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("gmail", "acct1", {"email": "a@b.com", "_user_label": "Support"})
        vault.save("gmail", "acct2", {"email": "c@d.com"})
        monkeypatch.setattr("cowork.services.connectors.persist._default_vault", lambda: vault)
        result = await _cowork_label_connection(
            MagicMock(), {"engine": "gmail", "name": "acct2", "label": "Support"}
        )
        assert "Support 2" in result

    @pytest.mark.asyncio
    async def test_reports_not_found(self, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        monkeypatch.setattr("cowork.services.connectors.persist._default_vault", lambda: vault)
        result = await _cowork_label_connection(
            MagicMock(), {"engine": "gmail", "name": "missing", "label": "Support"}
        )
        assert "no connection" in result.lower()
