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


def test_raw_settings_write_syncs_credentials_but_not_models(tmp_path, monkeypatch):
    # ENG-739: /settings/raw syncs credentials + provider selection to the DB,
    # but NOT model keys — a model in .env is CLI-only and must never be pushed
    # to the DB, or a bulk sync (web token refresh, re-login) would re-pin a
    # user who fixed a locked-model 403 via the picker. The .env line is still
    # written to disk for the standalone CLI.
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
        raw = settings_endpoint.read_raw_settings(_local_request())
        assert raw["ANTON_MINDS_API_KEY"] == "existing-key"
        # The model line IS preserved in .env (CLI-only surface)…
        assert raw["ANTON_PLANNING_MODEL"] == "_reason_"

        # …but is NOT synced to the DB: the row stays unset (resolves to a
        # default), while credentials + provider are synced as before.
        service = SettingService(session)
        assert service._fetch_row("planning_model") is None
        loaded = service.load()
        assert loaded.minds_api_key.get_secret_value() == "existing-key"
        assert loaded.planning_provider.value == "minds_cloud"
        assert loaded.planning_model != "_reason_"
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
