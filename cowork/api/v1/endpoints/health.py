from importlib.metadata import version, PackageNotFoundError

from fastapi import APIRouter

from cowork.common.settings.user_settings import get_user_settings

router = APIRouter()


def _get_server_version() -> str:
    try:
        return version("cowork-server")
    except PackageNotFoundError:
        return "dev"


# Health endpoint — the Electron app and dev-web.mjs probe this
# to know when the server is ready before mounting the renderer.
# The cowork frontend also reads config_ready / config_error from
# this response to gate the home view input box.
@router.get("/", response_model=dict)
def health() -> dict:
    settings = get_user_settings()
    return {
        "status": "ok",
        "anton_available": True,
        "mode": "anton",
        "server_version": _get_server_version(),
        **settings.config_status,
    }
