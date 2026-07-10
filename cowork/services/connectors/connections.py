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
from cowork.services.connectors.vault_lock import discard_lock, lock_for


class ConnectionsService:

    def _vault(self):
        from pathlib import Path
        from anton.core.datasources.data_vault import LocalDataVault
        return LocalDataVault(Path(ConnectorSettings().vault_dir))

    def _read_record(self, vault, engine: str, name: str) -> dict | None:
        """Full on-disk record via read_record() when the vault supports
        it, else a load()+wrap fallback for older vault shims that only
        expose the fields dict. Shared by every method below — the fallback
        shape must stay consistent or callers reading `secure_keys` off it
        would silently regress."""
        if hasattr(vault, "read_record"):
            return vault.read_record(engine, name)
        raw = vault.load(engine, name)
        return {"engine": engine, "name": name, "fields": raw} if raw is not None else None

    @staticmethod
    def _load_picked_files(fields: dict) -> list[dict]:
        try:
            return json.loads(fields.get("_picked_files") or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

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
            record = self._read_record(vault, engine, name)
            fields = (record or {}).get("fields") if record else {}
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
        record = self._read_record(vault, engine, name)
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
        with lock_for(engine, name):
            record = self._read_record(vault, engine, name)
            if record is None:
                return False
            fields = dict(record.get("fields") or {})
            secure_keys = record.get("secure_keys")
            fields.update(updates)
            vault.save(engine, name, fields, secure_keys=secure_keys)
            return True

    def merge_picked_files(self, engine: str, name: str, files: list[dict]) -> list[dict] | None:
        """Merge newly Google-Picker-granted files into the connection's
        persisted `_picked_files` list (deduped by id), and store it back
        as a JSON string field.

        Each file carries a `projects` list — the project(s) it was
        explicitly added to (empty when picked from connection-details,
        which has no project context). On conflict (same file id picked
        again, possibly for a different project), the two `projects`
        lists are UNIONed rather than one overwriting the other — a file
        already showing under Project A shouldn't disappear from it just
        because it was also just added to Project B. Every other field
        (name, mimeType, etc.) is refreshed from the newest pick.

        Storing it as a vault field (not a side table) means it flows
        through the existing `inject_env` namespacing for free, without
        any extra plumbing. The leading underscore matters: it's the
        vault's existing convention for internal bookkeeping fields
        (`_label`, `_connector_id`, `_method`) that the agent-visible
        "Connected Data Sources" credential listing skips — this field is
        picker metadata, not a credential, and must not be listed
        alongside real OAuth tokens where the agent might mistake it for
        one and try to use it as an auth parameter.

        Returns the merged list, or None if the connection doesn't exist.
        """
        vault = self._vault()
        with lock_for(engine, name):
            record = self._read_record(vault, engine, name)
            if record is None:
                return None

            fields = dict(record.get("fields") or {})
            secure_keys = record.get("secure_keys")
            existing = self._load_picked_files(fields)

            by_id = {f["id"]: f for f in existing if isinstance(f, dict) and "id" in f}
            for f in files:
                fid = f.get("id")
                if fid is None:
                    continue
                prior = by_id.get(fid)
                if prior:
                    prior_projects = prior.get("projects") or []
                    incoming_projects = f.get("projects") or []
                    merged_projects = list(dict.fromkeys([*prior_projects, *incoming_projects]))
                    by_id[fid] = {**prior, **f, "projects": merged_projects}
                else:
                    by_id[fid] = f
            merged = list(by_id.values())

            fields["_picked_files"] = json.dumps(merged)
            vault.save(engine, name, fields, secure_keys=secure_keys)
            return merged

    def remove_picked_file(self, engine: str, name: str, file_id: str, project: str) -> list[dict] | None:
        """Untag one file from `project` — the inverse of merge_picked_files'
        union. Only removes `project` from that file's `projects` list; the
        entry itself is never dropped from `_picked_files` here, even if
        `project` was its only tag (it just becomes untagged, same as a file
        picked directly from connection-details with no project context —
        still visible there, invisible under any project's Project files).
        Only revokes our own bookkeeping of the grant (stops the agent from
        being told about the file for this project); does not revoke
        Google's own server-side record of it.

        Returns the resulting list, or None if the connection doesn't exist.
        """
        vault = self._vault()
        with lock_for(engine, name):
            record = self._read_record(vault, engine, name)
            if record is None:
                return None

            fields = dict(record.get("fields") or {})
            secure_keys = record.get("secure_keys")
            existing = self._load_picked_files(fields)

            remaining = []
            for f in existing:
                if isinstance(f, dict) and f.get("id") == file_id:
                    f = {**f, "projects": [p for p in (f.get("projects") or []) if p != project]}
                remaining.append(f)

            fields["_picked_files"] = json.dumps(remaining)
            vault.save(engine, name, fields, secure_keys=secure_keys)
            return remaining

    def delete(self, engine: str, name: str) -> bool:
        deleted = self._vault().delete(engine, name)
        discard_lock(engine, name)
        return deleted


service = ConnectionsService()
