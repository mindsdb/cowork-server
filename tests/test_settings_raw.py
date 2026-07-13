from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _local_request():
    """Loopback stand-in for the Request arg the raw-settings endpoints now
    take — they 403 non-loopback callers (settings._require_local, ENG-457)."""
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))


def _delete_settings(session, *keys: str) -> None:
    from cowork.services.settings import SettingService

    service = SettingService(session)
    for key in keys:
        try:
            service.delete_setting(key)
        except ValueError:
            pass


def test_raw_settings_write_syncs_legacy_env_to_db(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import _RawSettingsBody, write_raw_settings
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    env_path = tmp_path / ".anton" / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "\n".join(
            [
                "ANTON_MINDS_API_KEY=existing-key",
                "ANTON_PLANNING_PROVIDER=openai-compatible",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_endpoint, "_ENV_PATH", env_path)

    session = get_open_session()
    try:
        _delete_settings(session, "minds_api_key", "planning_provider", "planning_model")

        response = write_raw_settings(_RawSettingsBody(content="ANTON_PLANNING_MODEL=_reason_"), session, _local_request())

        assert response == {"ok": True}
        assert settings_endpoint.read_raw_settings(_local_request())["ANTON_MINDS_API_KEY"] == "existing-key"

        loaded = SettingService(session).load()
        assert loaded.minds_api_key.get_secret_value() == "existing-key"
        assert loaded.planning_provider.value == "minds_cloud"
        assert loaded.planning_model == "_reason_"
    finally:
        _delete_settings(session, "minds_api_key", "planning_provider", "planning_model")
        session.close()


def test_raw_settings_write_rejects_invalid_db_values_before_env_write(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import _RawSettingsBody, write_raw_settings
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    env_path = tmp_path / ".anton" / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("ANTON_PLANNING_PROVIDER=anthropic\n", encoding="utf-8")
    monkeypatch.setattr(settings_endpoint, "_ENV_PATH", env_path)

    session = get_open_session()
    try:
        _delete_settings(session, "planning_provider", "planning_model")

        with pytest.raises(HTTPException) as exc:
            write_raw_settings(
                _RawSettingsBody(
                    content="\n".join(
                        [
                            "ANTON_PLANNING_PROVIDER=not-a-provider",
                            "ANTON_PLANNING_MODEL=_reason_",
                        ]
                    )
                ),
                session,
                _local_request(),
            )

        assert exc.value.status_code == 400
        assert env_path.read_text(encoding="utf-8") == "ANTON_PLANNING_PROVIDER=anthropic\n"

        service = SettingService(session)
        assert service._fetch_row("planning_provider") is None
        assert service._fetch_row("planning_model") is None
    finally:
        _delete_settings(session, "planning_provider", "planning_model")
        session.close()
