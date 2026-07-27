import json
import logging
from enum import Enum

from cryptography.fernet import InvalidToken
from pydantic import SecretStr, ValidationError
from sqlmodel import Session, select

from cowork.common.encryption import decrypt, encrypt
from cowork.common.settings.user_settings import (
    UserSettings,
    invalidate_user_settings_cache,
)
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

    def upsert_setting(self, key: str, value: str) -> SettingResponse:
        self._validate_key(key)
        try:
            validated = UserSettings.model_validate({key: value})
        except ValidationError as e:
            raise ValueError(str(e))

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
        self.session.commit()
        invalidate_user_settings_cache()

        return self._to_response(key, validated, True)

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
        deleted: list[str] = []
        for key in credential_keys:
            row = self._fetch_row(key)
            if row is not None:
                self.session.delete(row)
                deleted.append(key)
        if deleted:
            self.session.commit()
            invalidate_user_settings_cache()
        return deleted

