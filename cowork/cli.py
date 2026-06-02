"""CLI entry points for cowork server and developer setup commands."""

import uvicorn

from cowork.common.settings.app_settings import get_app_settings
from cowork.dev_setup import run_dev_setup


def main() -> None:
    from cowork.updater import maybe_self_update
    maybe_self_update()

    settings = get_app_settings()
    port = settings.port
    host = settings.host
    uvicorn.run("cowork.server:app", host=host, port=port, reload=False, log_level="info")


def dev_setup_main() -> None:
    """Run local dev setup (schema create + base seed data)."""
    run_dev_setup()


if __name__ == "__main__":
    main()
