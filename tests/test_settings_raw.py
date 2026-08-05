from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _local_request():
    """Loopback stand-in for the Request arg the raw-settings endpoints now
    take — they 403 non-loopback callers (guards.require_local, ENG-457)."""
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))


def _delete_settings(session, *keys: str) -> None:
    from cowork.services.settings import SettingService

    service = SettingService(session)
    for key in keys:
        try:
            service.delete_setting(key)
        except ValueError:
            pass


def test_raw_settings_write_syncs_only_incoming_not_the_whole_env(tmp_path, monkeypatch):
    # ENG-1127 review: /settings/raw must sync ONLY the recognised vars in THIS
    # request to the DB, never the whole merged .env. The server now mirrors
    # DB->.env, so the file can hold a preserved/translated cluster (a stale
    # minds-cloud line, a gemini role written as openai-compatible); re-syncing all
    # of it would overwrite the authoritative DB choice from the CLI's derived
    # file. Models are still never synced (ENG-739). The full .env is still written
    # to disk for the standalone CLI.
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import _RawSettingsBody, write_raw_settings
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    env_path = tmp_path / ".anton" / ".env"
    env_path.parent.mkdir(parents=True)
    # A STALE cluster line already on disk that the incoming request does NOT touch.
    env_path.write_text("ANTON_CODING_PROVIDER=minds-cloud\n", encoding="utf-8")
    monkeypatch.setattr(settings_endpoint, "_ENV_PATH", env_path)

    session = get_open_session()
    keys = ("minds_api_key", "planning_provider", "planning_model", "coding_provider")
    try:
        _delete_settings(session, *keys)

        response = write_raw_settings(
            _RawSettingsBody(content="\n".join([
                "ANTON_MINDS_API_KEY=new-key",
                "ANTON_PLANNING_PROVIDER=minds_cloud",
                "ANTON_PLANNING_MODEL=_reason_",
            ])),
            session, _local_request(),
        )
        assert response == {"ok": True}

        service = SettingService(session)
        loaded = service.load()
        # Incoming credential + provider synced to the DB.
        assert loaded.minds_api_key.get_secret_value() == "new-key"
        assert loaded.planning_provider.value == "minds_cloud"
        # Model in the request is NOT synced (ENG-739).
        assert service._fetch_row("planning_model") is None
        # The STALE on-disk coding_provider is NOT pulled into the DB (Finding 1).
        assert service._fetch_row("coding_provider") is None

        # The full .env is still written to disk for the CLI — merge preserves the
        # untouched stale line and the CLI-only model line.
        raw = settings_endpoint.read_raw_settings(_local_request())
        assert raw["ANTON_MINDS_API_KEY"] == "new-key"
        assert raw["ANTON_CODING_PROVIDER"] == "minds-cloud"
        assert raw["ANTON_PLANNING_MODEL"] == "_reason_"
    finally:
        _delete_settings(session, *keys)
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
