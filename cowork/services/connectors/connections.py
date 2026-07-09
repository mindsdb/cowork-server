from __future__ import annotations

import json

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
                status=(fields or {}).get("status"),
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
        masked_keys: list[str] = []
        for key in list(fields):
            if not key.startswith("_") and is_secret_key(key, secure_keys):
                fields[key] = _SENTINEL
                masked_keys.append(key)

        display_name = connection_display_name(fields)
        # Echo the stored label back as the form's `label` field so the edit
        # form pre-fills "Name this connection" with the current value (the
        # field is named `label`; the record stores it as `_label`).
        stored_label = fields.pop("_label", None)
        if stored_label:
            fields["label"] = stored_label
        return ConnectionDetailResponse(
            engine=record.get("engine", engine),
            name=record.get("name", name),
            display_name=display_name,
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
            connector_id=fields.pop("_connector_id", None),
            method=fields.pop("_method", None),
            fields=fields,
            secure_keys=masked_keys,
        )

    def patch_token(self, engine: str, name: str, updates: dict) -> bool:
        """Partially update token fields on an existing vault entry.

        Only ``access_token``, ``expires_at``, and ``status`` are written;
        ``refresh_token`` is never stored in the vault — it lives in the OS keychain.
        Returns ``False`` if the entry does not exist.
        """
        vault = self._vault()
        if hasattr(vault, "read_record"):
            record = vault.read_record(engine, name)
        else:
            raw = vault.load(engine, name)
            record = {"engine": engine, "name": name, "fields": raw} if raw is not None else None
        if record is None:
            return False
        fields = dict(record.get("fields") or {})
        secure_keys = record.get("secure_keys")
        fields.update(updates)
        vault.save(engine, name, fields, secure_keys=secure_keys)
        return True

    def merge_picked_files(self, engine: str, name: str, files: list[dict]) -> list[dict] | None:
        """Merge newly Google-Picker-granted files into the connection's
        persisted `picked_files` list (deduped by id, new entries win on
        conflict), and store it back as a JSON string field.

        Storing it as a vault field (not a side table) means it flows
        through the existing `inject_env` namespacing for free — the
        agent sees it as DS_<ENGINE>_<NAME>__PICKED_FILES alongside the
        connection's other credentials, without any extra plumbing.

        Returns the merged list, or None if the connection doesn't exist.
        """
        vault = self._vault()
        if hasattr(vault, "read_record"):
            record = vault.read_record(engine, name)
        else:
            raw = vault.load(engine, name)
            record = {"engine": engine, "name": name, "fields": raw} if raw is not None else None
        if record is None:
            return None

        fields = dict(record.get("fields") or {})
        secure_keys = record.get("secure_keys")
        try:
            existing = json.loads(fields.get("picked_files") or "[]")
        except (json.JSONDecodeError, TypeError):
            existing = []

        by_id = {f["id"]: f for f in existing if isinstance(f, dict) and "id" in f}
        for f in files:
            by_id[f["id"]] = f
        merged = list(by_id.values())

        fields["picked_files"] = json.dumps(merged)
        vault.save(engine, name, fields, secure_keys=secure_keys)
        return merged

    def remove_picked_file(self, engine: str, name: str, file_id: str) -> list[dict] | None:
        """Drop one file from the connection's persisted `picked_files`
        list — the inverse of merge_picked_files. Only revokes our own
        bookkeeping of the grant (stops the agent from being told about
        the file); does not necessarily revoke Google's own server-side
        record of it.

        Returns the resulting list, or None if the connection doesn't exist.
        """
        vault = self._vault()
        if hasattr(vault, "read_record"):
            record = vault.read_record(engine, name)
        else:
            raw = vault.load(engine, name)
            record = {"engine": engine, "name": name, "fields": raw} if raw is not None else None
        if record is None:
            return None

        fields = dict(record.get("fields") or {})
        secure_keys = record.get("secure_keys")
        try:
            existing = json.loads(fields.get("picked_files") or "[]")
        except (json.JSONDecodeError, TypeError):
            existing = []

        remaining = [f for f in existing if not (isinstance(f, dict) and f.get("id") == file_id)]

        fields["picked_files"] = json.dumps(remaining)
        vault.save(engine, name, fields, secure_keys=secure_keys)
        return remaining

    def delete(self, engine: str, name: str) -> bool:
        return self._vault().delete(engine, name)


service = ConnectionsService()
