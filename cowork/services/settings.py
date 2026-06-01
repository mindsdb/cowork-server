from enum import Enum

from pydantic import SecretStr, ValidationError
from sqlmodel import Session, select

from cowork.common.encryption import decrypt, encrypt
from cowork.common.settings.user_settings import (
    UserSettings,
    invalidate_user_settings_cache,
)
from cowork.models.setting import Setting
from cowork.schemas.settings import SettingResponse


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
        data = {
            row.key: (decrypt(row.value) if UserSettings.field_is_sensitive(row.key) else row.value)
            for row in rows
            if row.key in UserSettings.model_fields
        }
        return UserSettings(**data)

    @staticmethod
    def _to_response(key: str, settings: UserSettings, is_set: bool) -> SettingResponse:
        field_info = UserSettings.model_fields[key]
        is_sensitive = UserSettings.field_is_sensitive(key)
        field_val = getattr(settings, key)

        value = None
        if not is_sensitive and field_val is not None:
            value = field_val.value if isinstance(field_val, Enum) else str(field_val)

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
        return [self._to_response(key, settings, key in set_keys) for key in UserSettings.model_fields]

    def get_setting(self, key: str) -> SettingResponse:
        self._validate_key(key)
        row = self._fetch_row(key)
        settings = self._load([row] if row else [])
        return self._to_response(key, settings, row is not None)

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

