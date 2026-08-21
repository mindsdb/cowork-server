from importlib.metadata import version, PackageNotFoundError

from fastapi import APIRouter

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import get_user_settings

router = APIRouter()


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _anton_install_id() -> str:
    """anton's own analytics install id, so the desktop app can join its
    anonymous per-turn cost events to an identified user (ENG-1689).

    The app cannot compute this itself, which is the whole reason it is served
    here. It is a truncated SHA-256 of the machine's MAC, produced inside
    anton's Python process and — on desktop — never written to disk. ENG-440's
    `installation-id.ts` says so explicitly, and mints its own unrelated random
    id instead: *"that fingerprint is MAC-derived on desktop and only writes a
    file in the Docker fallback, so there's nothing for the Electron process to
    read."*

    Re-deriving it in Node was rejected: `uuid.getnode()` picks a MAC by
    Python's own platform probe order, which `os.networkInterfaces()` does not
    reproduce on a multi-NIC machine. That failure mode is silent — the join
    key looks present and matches nothing — so the value is read from the one
    process that already has it.

    Returns "" rather than raising. This is the readiness probe the desktop app
    polls before mounting the renderer; it must not fail because an analytics
    identifier could not be resolved.
    """
    try:
        from anton.analytics import get_installation_id

        return get_installation_id() or ""
    except Exception:  # pragma: no cover - defensive
        return ""


# Health endpoint — the Electron app and dev-web.mjs probe this
# to know when the server is ready before mounting the renderer.
# The cowork frontend also reads config_ready / config_error from
# this response to gate the home view input box.
#
# `owner` echoes the per-install token the desktop app passed via
# COWORK_SERVER_OWNER. The app adopts an already-running server only when
# this matches its own token, so one OS user's app can't drive another
# user's sidecar on a shared loopback port (ENG-439). Empty when unset.
@router.get("/", response_model=dict)
def health() -> dict:
    settings = get_user_settings()
    return {
        "status": "ok",
        "anton_available": True,
        "mode": "anton",
        "server_version": _pkg_version("cowork-server"),
        "anton_version": _pkg_version("anton-agent"),
        "owner": get_app_settings().owner,
        # Provider config is org-owned (admin-only writes) in org mode, so the
        # client must not finalize onboarding by writing it. `config_ready` can't
        # express this: it says the deployment can run, not who may configure it.
        "org_mode": get_app_settings().tenancy_mode == "org",
        # anton's analytics install id, desktop only (ENG-1689). Empty in org
        # mode deliberately: there the id fingerprints the SERVER, identical for
        # every user of the deployment, so publishing it would answer no
        # question and would expose one host's fingerprint to all of them. It is
        # also useless for the join it exists for — web turns execute in a
        # scratchpad pod, so the machine that ran the turn is not this one.
        "aid": "" if get_app_settings().tenancy_mode == "org" else _anton_install_id(),
        **settings.config_status,
    }
