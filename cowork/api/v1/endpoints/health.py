from importlib.metadata import version, PackageNotFoundError

from fastapi import APIRouter

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import (
    Provider,
    UserSettings,
    get_user_settings,
)

router = APIRouter()


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# What anton returns when it cannot fingerprint the machine at all. Not a valid
# id: every such machine reports the same string, so it would join them together.
_AID_SENTINEL = "unknown"


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

    **The sentinel must not escape.** ``get_installation_id`` returns the
    literal string ``"unknown"`` when it cannot fingerprint the machine at all
    — a container with stripped networking whose fallback file is unwritable,
    or any exception from ``uuid.getnode()``. anton stamps that same
    ``"unknown"`` on its own events, so passing it through would produce a join
    key that *matches* across every unfingerprintable machine and silently
    merges them into one identity. That is ENG-713's over-merge outcome reached
    without an alias, and it is worse than no key at all because the value looks
    valid. Absent beats colliding.
    """
    try:
        from anton.analytics import get_installation_id

        aid = get_installation_id() or ""
        return "" if aid == _AID_SENTINEL else aid
    except Exception:  # pragma: no cover - defensive
        return ""


def _minds_runtime_credential_required(
    settings: UserSettings, *, org_mode: bool
) -> bool:
    """Whether a required desktop role can consume the runtime Minds JWT.

    Router-only MindsHub use stays out of the wake-up barrier because it is an
    optional fail-open probe. Org deployments mint per-turn credentials and do
    not consume the desktop's in-memory hand-over.
    """
    if org_mode:
        return False
    return Provider.MINDS_CLOUD in (
        getattr(settings, "resolved_planning_provider", None),
        getattr(settings, "resolved_coding_provider", None),
    )


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
    # Read once: `org_mode` and the `aid` gate are the same decision, and two
    # separate reads could in principle disagree within one response.
    _org_mode = get_app_settings().tenancy_mode == "org"
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
        "org_mode": _org_mode,
        "minds_runtime_credential_required": _minds_runtime_credential_required(
            settings, org_mode=_org_mode
        ),
        # anton's analytics install id, desktop only (ENG-1689). Empty in org
        # mode deliberately: there the id fingerprints the SERVER, identical for
        # every user of the deployment, so publishing it would answer no
        # question and would expose one host's fingerprint to all of them. It is
        # also useless for the join it exists for — web turns execute in a
        # scratchpad pod, so the machine that ran the turn is not this one.
        "aid": "" if _org_mode else _anton_install_id(),
        **settings.config_status,
    }
