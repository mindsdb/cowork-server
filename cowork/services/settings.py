import json
import logging
from enum import Enum
from typing import Any

from cryptography.fernet import InvalidToken
from pydantic import SecretStr, ValidationError
from sqlmodel import Session, select

from cowork.common.encryption import decrypt, encrypt
from cowork.common.settings.runtime_credential import get_minds_credential
from cowork.common.settings.user_settings import (
    UserSettings,
    invalidate_user_settings_cache,
    setting_is_org_scoped,
)
from cowork.db.scoped import TenantScope
from cowork.models.setting import Setting
from cowork.schemas.settings import SettingResponse

logger = logging.getLogger(__name__)


def _mask_provider_keys(providers_json: str) -> str:
    """Return providers_json with each card's apiKey replaced by '***'.

    providers_json is non-sensitive (so GET /settings/ returns it verbatim),
    but each card embeds the raw provider key — the same secret that's masked
    in the sibling key fields. Mask it here so the list/get responses don't
    leak it (ENG-462). Fails closed: an unparseable value returns '[]' rather
    than risk echoing a raw key.
    """
    try:
        cards = json.loads(providers_json or "[]")
    except (ValueError, TypeError):
        return "[]"
    if isinstance(cards, list):
        for card in cards:
            if isinstance(card, dict) and card.get("apiKey"):
                card["apiKey"] = "***"
    return json.dumps(cards)


class SettingService:
    """Read/write settings with per-scope routing.

    A key's write scope comes from ``setting_is_org_scoped``; reads resolve
    user → org → global (NULL-scope legacy/env row) → field default. Scope is
    explicit here, not inherited from ScopedSession (``settings`` stays in
    ``_TENANCY_DEFERRED_TABLES``). No scope → operates on global rows only, the
    pre-split desktop behavior. Admin gating on org-key writes is an endpoint
    concern; this layer only routes.
    """

    def __init__(self, session: Session, scope: TenantScope | None = None) -> None:
        self.session = session
        self.scope = scope

    def _org_active(self) -> bool:
        return bool(self.scope and self.scope.org_mode and self.scope.org_id)

    def _global_row(self, key: str) -> Setting | None:
        return self.session.exec(
            select(Setting).where(Setting.key == key, Setting.scope.is_(None))
        ).first()

    def _scoped_row(self, key: str, scope: str, *, org_id: str | None = None, user_id: str | None = None) -> Setting | None:
        stmt = select(Setting).where(Setting.key == key, Setting.scope == scope)
        if org_id is not None:
            stmt = stmt.where(Setting.org_id == org_id)
        if user_id is not None:
            stmt = stmt.where(Setting.user_id == user_id)
        return self.session.exec(stmt).first()

    def _fetch_row(self, key: str) -> Setting | None:
        """Winning row for read: user → org → global. A user row is identified
        by (org_id, user_id) — the same person in another org must not match."""
        if not self._org_active():
            return self._global_row(key)
        if self.scope.user_id:
            row = self._scoped_row(key, "user", org_id=self.scope.org_id, user_id=self.scope.user_id)
            if row is not None:
                return row
        row = self._scoped_row(key, "org", org_id=self.scope.org_id)
        if row is not None:
            return row
        return self._global_row(key)

    def _own_row(self, key: str) -> Setting | None:
        """Row at the key's own write scope, no fallback — delete/clear act on
        this so a tenant removes its own override, never a deployment row."""
        if not self._org_active():
            return self._global_row(key)
        if not setting_is_org_scoped(key) and self.scope.user_id:
            return self._scoped_row(key, "user", org_id=self.scope.org_id, user_id=self.scope.user_id)
        return self._scoped_row(key, "org", org_id=self.scope.org_id)

    def _fetch_all_rows(self) -> list[Setting]:
        """One resolved row per key: global → org → user, most-specific wins."""
        if not self._org_active():
            return list(
                self.session.exec(select(Setting).where(Setting.scope.is_(None))).all()
            )
        by_key: dict[str, Setting] = {}
        for row in self.session.exec(select(Setting).where(Setting.scope.is_(None))).all():
            by_key[row.key] = row
        for row in self.session.exec(
            select(Setting).where(Setting.scope == "org", Setting.org_id == self.scope.org_id)
        ).all():
            by_key[row.key] = row
        if self.scope.user_id:
            for row in self.session.exec(
                select(Setting).where(
                    Setting.scope == "user",
                    Setting.org_id == self.scope.org_id,
                    Setting.user_id == self.scope.user_id,
                )
            ).all():
                by_key[row.key] = row
        return list(by_key.values())

    def _require_writable(self) -> None:
        """Mutations fail closed in org mode with no org in scope: a write would
        otherwise land in a global row visible to every tenant. Reads keep their
        global fallback — this guards writes/deletes only."""
        if self.scope and self.scope.org_mode and not self.scope.org_id:
            from cowork.db.scoped import MissingTenantScopeError
            raise MissingTenantScopeError("settings write requires an organization in scope")

    def _require_write_target(self, key: str) -> None:
        """A personal-key write needs a user in scope — checked up front so a
        batch resolves every target BEFORE staging any row (all-or-nothing)."""
        if self._org_active() and not setting_is_org_scoped(key) and not self.scope.user_id:
            raise ValueError(f"'{key}' is a personal setting but no user is in scope")

    def _new_row(self, key: str, store_val: str) -> Setting:
        """A fresh Setting stamped for the key's write scope."""
        if not self._org_active():
            return Setting(key=key, value=store_val)
        if setting_is_org_scoped(key):
            return Setting(key=key, value=store_val, scope="org", org_id=self.scope.org_id)
        if not self.scope.user_id:
            raise ValueError(f"'{key}' is a personal setting but no user is in scope")
        return Setting(
            key=key, value=store_val, scope="user",
            user_id=self.scope.user_id, org_id=self.scope.org_id,
        )

    def _write_row(self, key: str, store_val: str) -> None:
        row = self._own_row(key)
        if row is None:
            row = self._new_row(key, store_val)
        else:
            row.value = store_val
        self.session.add(row)

    @staticmethod
    def _validate_key(key: str) -> None:
        if key not in UserSettings.model_fields:
            raise ValueError(f"Unknown setting: '{key}'")

    @staticmethod
    def _raw_data(rows: list[Setting]) -> dict[str, str]:
        """Decrypted field → value map for ``rows``, before model validation.

        Split out of ``_load`` so ``load_pending`` can overlay in-flight values
        on the same footing as stored ones (both plaintext at this point).
        """
        data: dict[str, str] = {}
        for row in rows:
            if row.key not in UserSettings.model_fields:
                continue
            if UserSettings.field_is_sensitive(row.key):
                try:
                    decrypted = decrypt(row.value)
                except InvalidToken:
                    # Wrong master key → treat as unset, not a load-wide failure.
                    logger.warning(
                        "settings: %r could not be decrypted (master key mismatch); treating as unset",
                        row.key,
                    )
                    continue
                # An empty credential is no credential: a blank sensitive value
                # (e.g. a key cleared in the UI, which upserts "") reads as unset,
                # so the provider is honestly not-configured rather than present.
                if not decrypted:
                    continue
                data[row.key] = decrypted
            else:
                data[row.key] = row.value
        # The desktop app hands its MindsHub credential over at runtime instead
        # of storing it, so overlay it here — the one point every reader of
        # get_user_settings() goes through. It beats a stored row deliberately:
        # an install upgrading from a build that persisted its key still has
        # that row until the migration clears it, and a stale key must never
        # shadow the live credential. Returns None outside local mode.
        runtime_minds_key = get_minds_credential()
        if runtime_minds_key:
            data["minds_api_key"] = runtime_minds_key
        return data

    @staticmethod
    def _load(rows: list[Setting]) -> UserSettings:
        return UserSettings(**SettingService._raw_data(rows))

    @staticmethod
    def _is_set(key: str, settings: UserSettings, set_keys: set[str]) -> bool:
        # Sensitive fields count as set only when they carry a non-empty value
        # (a blank row reads as unset, matching _load); other fields are set
        # whenever a row exists.
        if UserSettings.field_is_sensitive(key):
            return getattr(settings, key) is not None
        return key in set_keys

    @staticmethod
    def _to_response(key: str, settings: UserSettings, is_set: bool) -> SettingResponse:
        field_info = UserSettings.model_fields[key]
        is_sensitive = UserSettings.field_is_sensitive(key)
        field_val = getattr(settings, key)

        value = None
        if not is_sensitive and field_val is not None:
            value = field_val.value if isinstance(field_val, Enum) else str(field_val)
            if key == "providers_json":
                value = _mask_provider_keys(value)

        return SettingResponse(
            key=key,
            label=field_info.title or key,
            description=field_info.description or "",
            is_sensitive=is_sensitive,
            is_set=is_set,
            value=value,
            options=UserSettings.field_options(key),
        )

    def load(self) -> UserSettings:
        return self._load(self._fetch_all_rows())

    def load_pending(self, updates: dict[str, Any]) -> UserSettings:
        """Settings as they WILL be once ``updates`` is written.

        A validator that reads the stored state answers a question about the
        PREVIOUS config, which is wrong whenever a request changes more than one
        related key at once — and the Settings form always does: it ships
        provider + credential + model in a single bulk PUT, and no UI path saves
        a provider without its model. Resolving against the pre-write DB there
        checked the new model against the OLD provider's catalog, which both
        rejected legitimate provider switches and skipped the check entirely on
        the switch INTO MindsHub (ENG-1358 review).

        Skips the write-diff sentinels (``None`` / ``***``) exactly as the
        writers do, and treats a blank credential as unset, matching ``_load``.
        Read-only: nothing here touches the session.
        """
        data = self._raw_data(self._fetch_all_rows())
        for key, value in (updates or {}).items():
            if key not in UserSettings.model_fields:
                continue
            if value is None or value == "***":
                continue
            if UserSettings.field_is_sensitive(key) and not value:
                data.pop(key, None)  # clearing a key = unset, not empty-string
                continue
            data[key] = value
        return UserSettings(**data)

    def list_settings(self) -> list[SettingResponse]:
        rows = self._fetch_all_rows()
        settings = self._load(rows)
        set_keys = {row.key for row in rows}
        return [self._to_response(key, settings, self._is_set(key, settings, set_keys)) for key in UserSettings.model_fields]

    def get_setting(self, key: str) -> SettingResponse:
        self._validate_key(key)
        row = self._fetch_row(key)
        settings = self._load([row] if row else [])
        set_keys = {row.key} if row is not None else set()
        return self._to_response(key, settings, self._is_set(key, settings, set_keys))

    def _encode_for_store(self, key: str, value: str) -> tuple[str, UserSettings]:
        """Validate ``value`` for ``key``; return (stored-string, model).

        The stored string is Fernet-encrypted for sensitive fields and the enum
        ``.value`` for enums. Raises ``ValueError`` for an unknown key or a
        value that fails the field's validation. One place for the encode rules
        that ``upsert_setting`` and ``save_all`` share.
        """
        self._validate_key(key)
        try:
            validated = UserSettings.model_validate({key: value})
        except ValidationError as e:
            raise ValueError(str(e))
        field_val = getattr(validated, key)
        if UserSettings.field_is_sensitive(key):
            raw = field_val.get_secret_value() if isinstance(field_val, SecretStr) else str(field_val)
            return encrypt(raw), validated
        if isinstance(field_val, Enum):
            return field_val.value, validated
        return (str(field_val) if field_val is not None else value), validated

    def upsert_setting(self, key: str, value: str) -> SettingResponse:
        self._require_writable()
        store_val, validated = self._encode_for_store(key, value)
        self._require_write_target(key)
        try:
            self._write_row(key, store_val)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        invalidate_user_settings_cache()
        return self._to_response(key, validated, True)

    def save_all(self, updates: dict[str, str]) -> list[str]:
        """Validate and upsert many settings in ONE transaction (all-or-nothing).

        Unlike ``bulk_upsert`` (best-effort, used by the .env sync) this raises
        on the first invalid key/value/target and writes NOTHING, so a
        Settings-form save can't half-apply. ``***`` (the unchanged-secret
        sentinel) and ``None`` are skipped, matching the client's write-diff.
        Returns the keys written.
        """
        encoded: dict[str, str] = {}
        for key, value in updates.items():
            if value is None or value == "***":
                continue
            store_val, _ = self._encode_for_store(key, value)  # raises → nothing staged
            encoded[key] = store_val
        # Resolve every write TARGET before staging any row, so a bad target
        # (e.g. a personal key with no user in scope) fails all-or-nothing
        # instead of leaving earlier rows staged for a caller to commit.
        self._require_writable()
        for key in encoded:
            self._require_write_target(key)
        try:
            for key, store_val in encoded.items():
                self._write_row(key, store_val)
            if encoded:
                self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        if encoded:
            invalidate_user_settings_cache()
        return list(encoded.keys())

    def bulk_upsert(self, updates: dict[str, str]) -> list[str]:
        """Upsert multiple settings in a single transaction (best-effort).

        Returns the keys actually written. Skips None, masked placeholders
        (``***``), invalid values, and un-writable targets. Rolls back the whole
        batch on an unexpected mutation error rather than leaving it staged.
        """
        self._require_writable()
        written: list[str] = []
        try:
            for key, value in updates.items():
                if value is None or value == "***":
                    continue
                self._validate_key(key)
                try:
                    store_val, _ = self._encode_for_store(key, value)
                    self._require_write_target(key)
                except ValueError:
                    continue
                self._write_row(key, store_val)
                written.append(key)
            if written:
                self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        if written:
            invalidate_user_settings_cache()
        return written

    def delete_setting(self, key: str) -> bool:
        self._validate_key(key)
        self._require_writable()
        self._require_write_target(key)
        row = self._own_row(key)
        if row is None:
            return False
        try:
            self.session.delete(row)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        invalidate_user_settings_cache()
        return True

    def clear_credentials(self) -> list[str]:
        """Delete all credential and provider-connectivity keys.

        Used by the logout flow to wipe API keys from the DB so
        ``config_ready`` returns ``False``.  Provider/model preferences
        are left intact so they survive a re-login cycle.
        """
        credential_keys = [
            field_name
            for field_name in UserSettings.model_fields
            if UserSettings.field_is_sensitive(field_name)
        ]
        # Also clear provider connectivity state and the UI provider
        # cards — stale entries from a previous account shouldn't bleed
        # into a fresh session.
        credential_keys += [
            "openai_base_url",
            "minds_url",
            "providers_json",
            "provider_status",
            "provider_status_details",
        ]
        self._require_writable()
        deleted: list[str] = []
        try:
            for key in credential_keys:
                row = self._own_row(key)
                if row is not None:
                    self.session.delete(row)
                    deleted.append(key)
            if deleted:
                self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        if deleted:
            invalidate_user_settings_cache()
        return deleted
