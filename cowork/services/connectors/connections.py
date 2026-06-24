from __future__ import annotations

from typing import Any

from cowork.common.settings.app_settings import ConnectorSettings
from cowork.schemas.connectors import ConnectionDetailResponse, ConnectionSummaryResponse
from cowork.services.connectors import health as health_mod
from cowork.services.connectors.specs._registry import registry

# TODO: A harness-agnostic sentinel value would be better here.
_SENTINEL = "ANTON_VAULT_KEEP"


class ConnectionsService:
    def _vault(self):
        from pathlib import Path
        from cowork.services.connectors.encrypted_vault import build_vault
        return build_vault(Path(ConnectorSettings().vault_dir))

    def list(self) -> list[ConnectionSummaryResponse]:
        vault = self._vault()
        result = []
        for item in vault.list_connections():
            engine = item.get("engine", "")
            name = item.get("name", "")
            spec = registry.get_connector(engine)
            # Read the full record once so health can see both the credential
            # fields (token expiry) and the stamped last-test metadata.
            record = vault.read_record(engine, name) if hasattr(vault, "read_record") else None
            fields = (record or {}).get("fields") or {}
            last_test_result = (record or {}).get("last_test_result")
            status = health_mod.compute_health(fields, last_test_result=last_test_result)
            result.append(ConnectionSummaryResponse(
                engine=engine,
                name=name,
                created_at=item.get("created_at"),
                updated_at=(record or {}).get("updated_at"),
                label=spec.label if spec else None,
                logo=spec.logo if spec else None,
                logo_color=spec.logo_color if spec else None,
                health=status.status,
                health_detail=status.detail,
                reconnectable=status.reconnectable,
                last_tested_at=(record or {}).get("last_tested_at"),
                last_test_result=last_test_result,
                expires_at=status.expires_at,
                encrypted=True,
            ))
        return result

    def get(self, engine: str, name: str) -> ConnectionDetailResponse | None:
        vault = self._vault()
        if hasattr(vault, "read_record"):
            record = vault.read_record(engine, name)
        else:
            raw = vault.load(engine, name)
            record = {"engine": engine, "name": name, "fields": raw} if raw is not None else None

        if record is None:
            return None

        raw_fields: dict = dict(record.get("fields") or {})
        last_test_result = record.get("last_test_result")
        # Compute health from the *real* (unmasked) fields before we mask
        # secrets for the response.
        status = health_mod.compute_health(raw_fields, last_test_result=last_test_result)

        fields = dict(raw_fields)
        for key in record.get("secure_keys") or []:
            if key in fields:
                fields[key] = _SENTINEL

        return ConnectionDetailResponse(
            engine=record.get("engine", engine),
            name=record.get("name", name),
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
            connector_id=fields.pop("_connector_id", None),
            method=fields.pop("_method", None),
            fields=fields,
            health=status.status,
            health_detail=status.detail,
            reconnectable=status.reconnectable,
            last_tested_at=record.get("last_tested_at"),
            last_test_result=last_test_result,
            last_test_error=record.get("last_test_error"),
            expires_at=status.expires_at,
            encrypted=True,
        )

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        """Raw vault record (decrypted fields + plaintext metadata), or None."""
        vault = self._vault()
        if hasattr(vault, "read_record"):
            return vault.read_record(engine, name)
        raw = vault.load(engine, name)
        return {"engine": engine, "name": name, "fields": raw} if raw is not None else None

    def record_test_result(
        self, engine: str, name: str, *, result: str, error: str | None = None
    ) -> bool:
        """Persist a "Test connection" outcome onto the saved record."""
        vault = self._vault()
        if not hasattr(vault, "record_test_result"):
            return False
        return vault.record_test_result(engine, name, result=result, error=error)

    def delete(self, engine: str, name: str) -> bool:
        return self._vault().delete(engine, name)


service = ConnectionsService()
