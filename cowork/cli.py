"""CLI entry points for cowork server and developer setup commands."""

import os
from pathlib import Path

import uvicorn

from cowork.common.settings.app_settings import (
    EXPORT_PDF_ENGINE_VAR,
    EXPORT_RENDER_URL_VAR,
    get_app_settings,
    get_export_pdf_settings,
)
from cowork.dev_setup import run_dev_setup

# Long enough for in-flight requests to finish, short enough that a supervisor
# never has to escalate to SIGKILL.
SHUTDOWN_GRACE_SECONDS = 5


def main() -> None:
    # Read the PDF export settings before the .env file below joins the
    # environment: they come from the environment the server started with.
    # On the desktop that file can be <COWORK_HOME>/.env, which
    # POST /settings/raw writes.
    get_export_pdf_settings()
    started_with = {name: os.environ.get(name) for name in (EXPORT_PDF_ENGINE_VAR, EXPORT_RENDER_URL_VAR)}
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    # Whatever reads the environment from here on, in this process or one it
    # starts, sees the PDF settings the server started with.
    for name, value in started_with.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    settings = get_app_settings()
    port = settings.port
    host = settings.host
    # A SIGTERM from the desktop, a logout or a system shutdown must end the
    # process even while a task's event stream is still open; without a bound
    # uvicorn waits for that connection forever and keeps the port.
    uvicorn.run(
        "cowork.server:app",
        host=host,
        port=port,
        reload=False,
        # Left unset, uvicorn takes a worker count from WEB_CONCURRENCY, which
        # the .env file can set, and each worker process reads the PDF
        # settings afresh.
        workers=1,
        log_config=str(Path(__file__).with_name("logging.json")),
        timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS,
    )


def dev_setup_main() -> None:
    """Run local dev setup (schema create + base seed data)."""
    run_dev_setup()


if __name__ == "__main__":
    main()
