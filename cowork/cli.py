"""CLI entry points for cowork server and developer setup commands."""

import uvicorn

from cowork.common.settings.app_settings import get_app_settings
from cowork.dev_setup import link_local_siblings, run_dev_setup


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
    """Run local dev setup (schema create + base seed data + link local siblings)."""
    run_dev_setup()
    link_local_siblings()


def dev_link_main() -> None:
    """Overlay local sibling-repo checkouts (e.g. ../anton) as editable installs.

    Used by the cowork dev launchers before starting the server so a
    developer's local feature branches are picked up. See link_local_siblings.
    """
    link_local_siblings()


if __name__ == "__main__":
    main()
