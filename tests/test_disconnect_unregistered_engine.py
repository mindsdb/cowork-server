"""A connection whose engine is not in the registry can still be disconnected.

Records saved before connector ids were validated carry a synthesized
``fm_<hex>`` engine. They cannot be repaired, so removing and reconnecting
them must work.
"""

from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors.connections import delete_connection
from cowork.db.scoped import LOCAL_SCOPE


@pytest.fixture
def vault_dir(tmp_path, monkeypatch):
    path = Path(tmp_path) / "vault"

    def settings():
        return type("S", (), {"vault_dir": str(path)})()

    monkeypatch.setattr("cowork.services.connectors.connections.ConnectorSettings", settings)
    return path


@pytest.mark.asyncio
async def test_fm_engine_record_is_deleted(vault_dir):
    vault = LocalDataVault(vault_dir)
    vault.save(
        "fm_ec163d25cf", "fm_ec163d25cf-2cf3a6",
        {"client_id": "86nwdt9sl34cuy", "client_secret": "s"}, secure_keys=["client_secret"],
    )

    await delete_connection("fm_ec163d25cf", "fm_ec163d25cf-2cf3a6", LOCAL_SCOPE, request=None)

    assert vault.list_connections() == []


@pytest.mark.asyncio
async def test_deleting_a_missing_unregistered_connection_is_a_404(vault_dir):
    with pytest.raises(HTTPException) as exc:
        await delete_connection("fm_ec163d25cf", "nope", LOCAL_SCOPE, request=None)
    assert exc.value.status_code == 404
