"""PDF export settings come from the process environment only, and a bad one stops the start."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import dotenv
import pytest
import uvicorn
import yaml
from pydantic import BaseModel

from cowork import cli
from cowork.common.settings.app_settings import (
    EXPORT_PDF_ENGINE_VAR,
    EXPORT_RENDER_URL_VAR,
    AppSettings,
    _env_file_chain,
    get_export_pdf_settings,
    load_export_pdf_settings,
)
from cowork.schemas.artifact_export import ExportPdfSettings
from cowork.server import create_app


class RenderUrlCases(BaseModel, frozen=True):
    """The tables of render_pdf_cases.yaml this file reads."""

    refused_render_urls: list[str]
    accepted_render_urls: list[str]


CASES = RenderUrlCases.model_validate(
    yaml.safe_load((Path(__file__).parent / "fixtures" / "render_pdf_cases.yaml").read_text(encoding="utf-8"))
)

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


@pytest.mark.parametrize("url", CASES.accepted_render_urls)
def test_render_with_an_http_url_turns_pdf_on(url):
    settings = load_export_pdf_settings(environ={EXPORT_PDF_ENGINE_VAR: "render", EXPORT_RENDER_URL_VAR: url})

    assert settings == ExportPdfSettings(engine="render", render_url=url)
    assert str(settings.render_url).rstrip("/") == url.rstrip("/")
    assert settings.timeout_seconds == 60.0
    assert settings.max_pdf_bytes == 48 * 1024 * 1024
    assert settings.max_pages == 500


def test_an_engine_other_than_render_fails_settings_load():
    with pytest.raises(ValueError, match=f"{EXPORT_PDF_ENGINE_VAR} must be 'render' or unset"):
        load_export_pdf_settings(environ={EXPORT_PDF_ENGINE_VAR: "chrome"})


def test_render_without_a_url_refuses_startup(monkeypatch):
    monkeypatch.setenv(EXPORT_PDF_ENGINE_VAR, "render")

    with pytest.raises(ValueError, match=f"needs {EXPORT_RENDER_URL_VAR}"):
        create_app()


@pytest.mark.parametrize("url", CASES.refused_render_urls)
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


def test_a_refused_render_url_stays_out_of_the_error():
    """The URL can carry credentials, so the message names the setting instead."""
    with pytest.raises(ValueError) as refused:
        load_export_pdf_settings(
            environ={
                EXPORT_PDF_ENGINE_VAR: "render",
                EXPORT_RENDER_URL_VAR: "ftp://user:secret@render.internal/",
            }
        )
    assert EXPORT_RENDER_URL_VAR in str(refused.value)
    assert "secret" not in str(refused.value)
    assert "render.internal" not in str(refused.value)


def test_pdf_settings_only_in_cowork_home_dotenv_offer_no_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(PDF_LINES + "COWORK_SERVER_HOST=0.0.0.0\n", encoding="utf-8")

    # The other settings read this file, so it is live.
    assert AppSettings(_env_file=_env_file_chain()).host == "0.0.0.0"
    assert get_export_pdf_settings().engine is None


def test_pdf_settings_only_in_the_dotenv_chain_offer_no_pdf_in_a_fresh_process(tmp_path):
    # A fresh process sets COWORK_HOME before anything imports, so a reader of
    # the .env chain built at import would see <COWORK_HOME>/.env. The working
    # directory is the same folder, so the chain's relative .env is that file too.
    (tmp_path / ".env").write_text(PDF_LINES + "COWORK_SERVER_HOST=0.0.0.0\n", encoding="utf-8")
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in (EXPORT_PDF_ENGINE_VAR, EXPORT_RENDER_URL_VAR)
    }
    env["COWORK_HOME"] = str(tmp_path)
    script = (
        "from cowork.common.settings.app_settings import AppSettings, get_export_pdf_settings\n"
        "print(AppSettings().host)\n"
        "print(get_export_pdf_settings().engine)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    host, engine = result.stdout.splitlines()[-2:]
    # The other settings read the file in that process, so it is live there.
    assert host == "0.0.0.0"
    assert engine == "None"


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

    _cli_loads_dotenv(monkeypatch, env_path=env_path)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **_: None)
    cli.main()

    # The server read its PDF settings before the .env joined the environment...
    assert get_export_pdf_settings().engine is None
    # ...and the .env's PDF settings never stay there, so a process that reads
    # them afresh offers no PDF either.
    assert EXPORT_PDF_ENGINE_VAR not in os.environ
    assert EXPORT_RENDER_URL_VAR not in os.environ
    assert load_export_pdf_settings(environ=os.environ).engine is None


def test_a_dotenv_that_sets_web_concurrency_still_serves_in_one_process(monkeypatch, tmp_path):
    # uvicorn reads WEB_CONCURRENCY as it starts, after the .env joined the
    # environment, and each worker process it starts reads the PDF settings
    # afresh.
    env_path = tmp_path / ".env"
    env_path.write_text(PDF_LINES + "WEB_CONCURRENCY=2\n", encoding="utf-8")
    monkeypatch.setenv("WEB_CONCURRENCY", "")
    monkeypatch.delenv("WEB_CONCURRENCY")
    _cli_loads_dotenv(monkeypatch, env_path=env_path)
    configs: list[uvicorn.Config] = []
    # uvicorn.run builds this Config, and starts worker processes when its count is over 1.
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda app, **kwargs: configs.append(uvicorn.Config(app, **{**kwargs, "log_config": None})),
    )

    cli.main()

    assert os.environ["WEB_CONCURRENCY"] == "2"
    assert [config.workers for config in configs] == [1]


def test_pdf_settings_the_server_started_with_stay_in_the_environment(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(f"{EXPORT_RENDER_URL_VAR}=http://127.0.0.1:9999\n", encoding="utf-8")
    monkeypatch.setenv(EXPORT_PDF_ENGINE_VAR, "render")
    monkeypatch.setenv(EXPORT_RENDER_URL_VAR, "http://127.0.0.1:9000")
    _cli_loads_dotenv(monkeypatch, env_path=env_path)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **_: None)

    cli.main()

    assert (os.environ[EXPORT_PDF_ENGINE_VAR], os.environ[EXPORT_RENDER_URL_VAR]) == (
        "render",
        "http://127.0.0.1:9000",
    )
    assert str(get_export_pdf_settings().render_url) == "http://127.0.0.1:9000/"


def _cli_loads_dotenv(monkeypatch: pytest.MonkeyPatch, *, env_path: Path) -> None:
    """Make the CLI's load_dotenv() read ``env_path``.

    On the desktop the server is installed under COWORK_HOME, so the CLI's
    load_dotenv() finds <COWORK_HOME>/.env walking up from the package.
    """
    real_load_dotenv = dotenv.load_dotenv
    monkeypatch.setattr(
        dotenv, "load_dotenv", lambda **kwargs: real_load_dotenv(dotenv_path=env_path, **kwargs)
    )
