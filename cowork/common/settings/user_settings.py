from cowork.schemas.settings import UserSettings

_cache: UserSettings | None = None


def get_user_settings() -> UserSettings:
    global _cache
    if _cache is None:
        _cache = _load_from_db()
    return _cache


def invalidate_user_settings_cache() -> None:
    global _cache
    _cache = None


def _load_from_db() -> UserSettings:
    from sqlmodel import select

    from cowork.db.session import get_open_session
    from cowork.models.setting import Setting
    from cowork.services.settings import SettingService

    session = get_open_session()
    try:
        rows = list(session.exec(select(Setting)).all())
        return SettingService._load(rows)
    finally:
        session.close()
