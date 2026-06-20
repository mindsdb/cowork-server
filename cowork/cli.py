"""CLI entry points for cowork server and developer setup commands."""

import subprocess
import sys

import uvicorn

from cowork.common.settings.app_settings import get_app_settings
from cowork.dev_setup import run_dev_setup


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    settings = get_app_settings()
    port = settings.port
    host = settings.host
    uvicorn.run("cowork.server:app", host=host, port=port, reload=False, log_level="info")


def dev_setup_main() -> None:
    """Run local dev setup (schema create + base seed data)."""
    run_dev_setup()


def install_browsers_main() -> None:
    """Install the browser runtime used for artifact visual diffs."""
    raise SystemExit(subprocess.call([sys.executable, "-m", "playwright", "install", "chromium"]))


if __name__ == "__main__":
    main()
