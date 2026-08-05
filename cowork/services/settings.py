import json
import logging
import threading
from enum import Enum

from cryptography.fernet import InvalidToken
from pydantic import SecretStr, ValidationError
from sqlmodel import Session, select

from cowork.common.encryption import decrypt, encrypt
from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.env_boundary import (
    atomic_write_env,
    db_to_env,
    env_reconcile_vars,
    merge_env_lines,
)
from cowork.common.settings.user_settings import (
    UserSettings,
    invalidate_user_settings_cache,
)
from cowork.models.setting import Setting
from cowork.schemas.settings import SettingResponse

logger = logging.getLogger(__name__)

# Serializes the .env export's read/merge/write so two concurrent settings writes
# can't lost-update the file — the last exporter re-reads the DB under the lock and
# installs the latest committed state (ENG-1127 review). In-process; the client's
# own .env writes go away in Phase B, making the server the sole writer.
_env_export_lock = threading.Lock()


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
    def __init__(self, session: Session) -> None:
        self.session = session

    def _fetch_row(self, key: str) -> Setting | None:
        return self.session.exec(select(Setting).where(Setting.key == key)).first()

    def _fetch_all_rows(self) -> list[Setting]:
        return list(self.session.exec(select(Setting)).all())

    @staticmethod
    def _validate_key(key: str) -> None:
        if key not in UserSettings.model_fields:
            raise ValueError(f"Unknown setting: '{key}'")

    @staticmethod
    def _load(rows: list[Setting]) -> UserSettings:
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
        return UserSettings(**data)

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

    def _write_row(self, key: str, store_val: str) -> None:
        row = self._fetch_row(key)
        if row is None:
            row = Setting(key=key, value=store_val)
        else:
            row.value = store_val
        self.session.add(row)

    def _after_write(self, *, export_env: bool = True) -> None:
        """Post-commit hook shared by the settings mutators.

        Invalidates the cache and (desktop install) mirrors the DB to the CLI's
        ``.env`` (ENG-1127); ``export_env=False`` skips the mirror for the seeding migration.
        """
        invalidate_user_settings_cache()
        if export_env:
            self._export_env_for_cli()

    def _export_env_for_cli(self) -> None:
        """Mirror the DB's aliased settings to the CLI's ``.env`` (best-effort).

        Local (desktop) tenancy only — a cloud pod must not spill decrypted secrets
        to disk (ENG-1127). Never raises — a stale ``.env`` must not fail a save.
        """
        try:
            if get_app_settings().tenancy_mode != "local":
                return
            # Serialize the whole read/merge/write. The DB read is INSIDE the lock,
            # so the last exporter to acquire it installs the latest committed
            # state — no lost update between concurrent settings writes.
            with _env_export_lock:
                rows = self._fetch_all_rows()
                settings = self._load(rows)
                managed = db_to_env(settings, {row.key for row in rows})
                path = cowork_home() / ".env"
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                content = merge_env_lines(existing, managed, env_reconcile_vars(settings))
                if content != existing:
                    atomic_write_env(path, content)
        except Exception as exc:  # noqa: BLE001 - export is best-effort
            logger.warning("settings: .env export for the CLI failed: %s", exc)

    def upsert_setting(self, key: str, value: str, *, export_env: bool = True) -> SettingResponse:
        store_val, validated = self._encode_for_store(key, value)
        self._write_row(key, store_val)
        self.session.commit()
        self._after_write(export_env=export_env)
        return self._to_response(key, validated, True)

    def save_all(self, updates: dict[str, str]) -> list[str]:
        """Validate and upsert many settings in ONE transaction (all-or-nothing).

        Unlike ``bulk_upsert`` (best-effort, used by the .env sync) this raises
        on the first invalid key/value and writes NOTHING, so a Settings-form
        save can't half-apply the way the per-key PUT loop could. ``***`` (the
        unchanged-secret sentinel) and ``None`` are skipped, matching the
        client's write-diff. Returns the keys written.
        """
        encoded: dict[str, str] = {}
        for key, value in updates.items():
            if value is None or value == "***":
                continue
            store_val, _ = self._encode_for_store(key, value)  # raises → nothing written
            encoded[key] = store_val
        for key, store_val in encoded.items():
            self._write_row(key, store_val)
        if encoded:
            self.session.commit()
            self._after_write()
        return list(encoded.keys())

    def bulk_upsert(self, updates: dict[str, str]) -> list[str]:
        """Upsert multiple settings in a single transaction.

        Returns the list of keys that were actually written.
        Skips None values and masked placeholders (``***``).
        """
        written: list[str] = []
        for key, value in updates.items():
            if value is None or value == "***":
                continue
            self._validate_key(key)
            try:
                validated = UserSettings.model_validate({key: value})
            except ValidationError:
                continue

            field_val = getattr(validated, key)
            if UserSettings.field_is_sensitive(key):
                raw = field_val.get_secret_value() if isinstance(field_val, SecretStr) else str(field_val)
                store_val = encrypt(raw)
            elif isinstance(field_val, Enum):
                store_val = field_val.value
            else:
                store_val = str(field_val) if field_val is not None else value

            row = self._fetch_row(key)
            if row is None:
                row = Setting(key=key, value=store_val)
            else:
                row.value = store_val
            self.session.add(row)
            written.append(key)

        if written:
            self.session.commit()
            invalidate_user_settings_cache()
        return written

    def delete_setting(self, key: str) -> bool:
        self._validate_key(key)
        row = self._fetch_row(key)
        if row is None:
            return False
        self.session.delete(row)
        self.session.commit()
        self._after_write()
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
        deleted: list[str] = []
        for key in credential_keys:
            row = self._fetch_row(key)
            if row is not None:
                self.session.delete(row)
                deleted.append(key)
        if deleted:
            self.session.commit()
            self._after_write()
        return deleted

