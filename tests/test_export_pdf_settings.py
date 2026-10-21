"""PDF export settings come from the process environment only, and a bad one stops the start."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import dotenv
import pytest
import yaml

from cowork import cli
from cowork.common.settings.app_settings import (
    EXPORT_PDF_ENGINE_VAR,
    EXPORT_RENDER_URL_VAR,
    AppSettings,
    _env_file_chain,
    get_export_pdf_settings,
    load_export_pdf_settings,
)
from cowork.server import create_app

CASES = yaml.safe_load((Path(__file__).parent / "fixtures" / "render_pdf_cases.yaml").read_text(encoding="utf-8"))

PDF_LINES = f"{EXPORT_PDF_ENGINE_VAR}=render\n{EXPORT_RENDER_URL_VAR}=http://127.0.0.1:9000\n"


@pytest.fixture(autouse=True)
def _no_pdf_settings(monkeypatch):
    # setenv first so monkeypatch removes whatever a test or a .env load adds.
    for name in (EXPORT_PDF_ENGINE_VAR, EXPORT_RENDER_URL_VAR):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    get_export_pdf_settings.cache_clear()
    yield
    get_export_pdf_settings.cache_clear()


def test_an_unset_engine_offers_no_pdf():
    settings = load_export_pdf_settings(environ={})

    assert settings.engine is None
    assert settings.render_url is None


def test_a_render_url_without_the_engine_offers_no_pdf():
    settings = load_export_pdf_settings(environ={EXPORT_RENDER_URL_VAR: "http://127.0.0.1:9000"})

    assert settings.engine is None
    assert settings.render_url is None


@pytest.mark.parametrize("url", ["http://127.0.0.1:9000", "https://render.internal/base/"])
def test_render_with_an_http_url_turns_pdf_on(url):
    settings = load_export_pdf_settings(environ={EXPORT_PDF_ENGINE_VAR: "render", EXPORT_RENDER_URL_VAR: url})

    assert settings.engine == "render"
    assert str(settings.render_url).rstrip("/") == url.rstrip("/")
    assert settings.timeout_seconds == 60.0


def test_an_engine_other_than_render_fails_settings_load():
    with pytest.raises(ValueError, match=f"{EXPORT_PDF_ENGINE_VAR} must be 'render' or unset"):
        load_export_pdf_settings(environ={EXPORT_PDF_ENGINE_VAR: "chrome"})


def test_render_without_a_url_refuses_startup(monkeypatch):
    monkeypatch.setenv(EXPORT_PDF_ENGINE_VAR, "render")

    with pytest.raises(ValueError, match=f"needs {EXPORT_RENDER_URL_VAR}"):
        create_app()


@pytest.mark.parametrize("url", CASES["refused_render_urls"])
def test_a_render_url_that_isnt_an_http_base_address_refuses_startup(monkeypatch, url):
    monkeypatch.setenv(EXPORT_PDF_ENGINE_VAR, "render")
    monkeypatch.setenv(EXPORT_RENDER_URL_VAR, url)

    with pytest.raises(ValueError, match=EXPORT_RENDER_URL_VAR):
        create_app()


def test_an_engine_other_than_render_refuses_startup(monkeypatch):
    monkeypatch.setenv(EXPORT_PDF_ENGINE_VAR, "chrome")

    with pytest.raises(ValueError, match=EXPORT_PDF_ENGINE_VAR):
        create_app()


def test_the_cli_refuses_to_start_before_serving(monkeypatch):
    served: list[str] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **_: served.append(app))
    monkeypatch.setenv(EXPORT_PDF_ENGINE_VAR, "render")
    monkeypatch.setenv(EXPORT_RENDER_URL_VAR, "ftp://render.internal/")

    with pytest.raises(ValueError, match=EXPORT_RENDER_URL_VAR):
        cli.main()
    assert served == []


def test_pdf_settings_only_in_cowork_home_dotenv_offer_no_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(PDF_LINES + "COWORK_SERVER_HOST=0.0.0.0\n", encoding="utf-8")

    # The other settings read this file, so it is live.
    assert AppSettings(_env_file=_env_file_chain()).host == "0.0.0.0"
    assert get_export_pdf_settings().engine is None


def test_pdf_settings_written_through_settings_raw_offer_no_pdf_after_a_restart(monkeypatch, tmp_path):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.db.session import get_open_session

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_endpoint, "_ENV_PATH", env_path)

    async def _no_warm(session, scope=None):
        return False

    monkeypatch.setattr(settings_endpoint, "warm_enabled_model_map", _no_warm)
    session = get_open_session()
    try:
        asyncio.run(
            settings_endpoint.write_raw_settings(settings_endpoint._RawSettingsBody(content=PDF_LINES), session)
        )
    finally:
        session.close()

    # On the desktop the server is installed under COWORK_HOME, so the CLI's
    # load_dotenv() finds <COWORK_HOME>/.env walking up from the package.
    real_load_dotenv = dotenv.load_dotenv
    monkeypatch.setattr(
        dotenv, "load_dotenv", lambda **kwargs: real_load_dotenv(dotenv_path=env_path, **kwargs)
    )
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **_: None)
    cli.main()

    # The .env reached the process environment, so reading it now would turn PDF on...
    assert os.environ[EXPORT_PDF_ENGINE_VAR] == "render"
    assert load_export_pdf_settings(environ=os.environ).engine == "render"
    # ...but the server read its PDF settings before that, and offers none.
    assert get_export_pdf_settings().engine is None
