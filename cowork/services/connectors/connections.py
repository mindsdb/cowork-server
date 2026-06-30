from __future__ import annotations

from anton.core.datasources.data_vault import is_secret_key

from cowork.common.settings.app_settings import ConnectorSettings
from cowork.schemas.connectors import ConnectionDetailResponse, ConnectionSummaryResponse
from cowork.services.connectors.identity import (
    VAULT_KEEP_SENTINEL as _SENTINEL,
    connection_display_name,
)
from cowork.services.connectors.specs._registry import registry


class ConnectionsService:
    def _vault(self):
        from pathlib import Path
        from anton.core.datasources.data_vault import LocalDataVault
        return LocalDataVault(Path(ConnectorSettings().vault_dir))

    def list(self) -> list[ConnectionSummaryResponse]:
        vault = self._vault()
        result = []
        for item in vault.list_connections():
            engine = item.get("engine", "")
            name = item.get("name", "")
            spec = registry.get_connector(engine)
            # Load the record's fields to derive a human display name (label or
            # identity) so the card shows e.g. "Support" / "user@gmail.com"
            # instead of the opaque slug.
            record = vault.read_record(engine, name) if hasattr(vault, "read_record") else None
            fields = (record or {}).get("fields") if record else (vault.load(engine, name) or {})
            result.append(ConnectionSummaryResponse(
                engine=engine,
                name=name,
                display_name=connection_display_name(fields or {}),
                created_at=item.get("created_at"),
                label=spec.label if spec else None,
                logo=spec.logo if spec else None,
                logo_color=spec.logo_color if spec else None,
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

        fields: dict = dict(record.get("fields") or {})
        # Mask secrets. Prefer the record's explicit secure_keys; fall back to the
        # name heuristic so records saved before secure_keys was persisted don't
        # leak their secret values through this endpoint.
        secure_keys = record.get("secure_keys")
        for key in list(fields):
            if not key.startswith("_") and is_secret_key(key, secure_keys):
                fields[key] = _SENTINEL

        display_name = connection_display_name(fields)
        fields.pop("_label", None)  # surfaced via display_name, not as a field row
        return ConnectionDetailResponse(
            engine=record.get("engine", engine),
            name=record.get("name", name),
            display_name=display_name,
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
            connector_id=fields.pop("_connector_id", None),
            method=fields.pop("_method", None),
            fields=fields,
        )

    def delete(self, engine: str, name: str) -> bool:
        return self._vault().delete(engine, name)


service = ConnectionsService()
