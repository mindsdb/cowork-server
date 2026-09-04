"""CLI entry points for cowork server and developer setup commands."""

import uvicorn

from cowork.common.settings.app_settings import get_app_settings
from cowork.dev_setup import run_dev_setup

# Long enough for in-flight requests to finish, short enough that a supervisor
# never has to escalate to SIGKILL.
SHUTDOWN_GRACE_SECONDS = 5


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
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
        log_level="info",
        timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS,
    )


def dev_setup_main() -> None:
    """Run local dev setup (schema create + base seed data)."""
    run_dev_setup()


if __name__ == "__main__":
    main()
