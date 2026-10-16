"""Locks the agent's code to the asker's own space: COWORK_CODE_SANDBOX.

With ``landlock``, every process that runs agent code (a cell, the venv check,
a package install, the artifact verifier's checks) starts through anton's
launcher (``anton.core.backends.sandbox``). The launcher locks the process with
Landlock to the paths one ``CellSandbox`` names and hands it an allowlisted
environment. ``build_cell_sandbox`` builds that policy, the same way for the
three places cowork-server starts agent code: a turn
(AntonHarness._build_chat_session), the credential probe
(handlers/probe.py), and the artifact-backend pool
(services/scratchpad_runtime.py). With ``off`` it builds none, and every spawn
keeps today's arguments and environment.

What a cell reaches under the policy:

- read, write and execute: the conversation's project folder;
- read and write, no execute: the asker's own HOME and TMPDIR, ``home/`` and
  ``tmp/`` in the ``sandbox`` store of their partition, and
  ``<cowork-home>/shared-files``, which every person's agent uses;
- read only: the folders the caller names (the turn's skill folders and the
  conversation's attachment folders), plus the system files anton's policy
  always carries.

The launcher grants what a path resolves to, and agent code that ran before
the sandbox could plant a link anywhere on the volume. So a grant inside one
of the server's own roots (cowork-home, the shared root, and the projects,
files and skills roots) is opened from that root without following links
(``_granted_dir``): a link on the way refuses a required grant and leaves an
optional one out, instead of granting another person's folder.

``forbidden_roots`` names the folders no grant may equal or contain:
cowork-home, the shared root, its ``shared/`` folder, every partition, and the
managed folder. The policy model refuses such a grant when it is built, so a
builder bug fails the session instead of opening every person's space.

``check_startup`` runs in create_app. It refuses identity headers with the
sandbox off, and ``landlock`` where it can't hold. ``_probe_abi`` and
``_run_self_test`` are the one seam tests stub, since neither a macOS machine
nor every CI kernel has Landlock.
"""

from __future__ import annotations

import errno
import getpass
import inspect
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from cowork.common.logger import get_logger
from cowork.common.paths import cowork_home, opened_subdir_nofollow
from cowork.common.settings.app_settings import AppSettings, get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import TenantScope, scoped_storage_root

if TYPE_CHECKING:
    from anton.core.backends.sandbox import CellSandbox

logger = get_logger(__name__)

#: The anton launcher API version this server builds policies for
#: (anton.core.backends.sandbox.LAUNCHER_API_VERSION).
LAUNCHER_API_VERSION = 1

#: What a Code mode route answers under the sandbox.
NO_CODE_MODE_DETAIL = (
    "Code mode is off on this deployment because agent code runs in a sandbox "
    "(COWORK_CODE_SANDBOX=landlock)."
)


def code_sandbox_on() -> bool:
    """Whether agent code runs under Landlock on this deployment."""
    return get_app_settings().code_sandbox == "landlock"


def shared_files_dir() -> Path:
    """``<cowork-home>/shared-files``: the one folder every person's agent
    reads and writes. check_startup creates it."""
    return cowork_home() / "shared-files"


def backends_off_message() -> str:
    """anton's own refusal of app backends under the sandbox
    (anton.core.backends.sandbox.BACKENDS_OFF_MESSAGE), read when a caller
    refuses, so the preview, its proxy and the agent's launch_backend give
    one reason. check_startup makes sure the module exists under landlock."""
    from anton.core.backends.sandbox import BACKENDS_OFF_MESSAGE

    return BACKENDS_OFF_MESSAGE


def _sandbox_root(scope: TenantScope | None) -> Path:
    """The ``sandbox`` store of ``scope``: ``<shared_root>/<partition>/sandbox``
    for a person, ``<cowork-home>/sandbox`` for the operator view."""
    return scoped_storage_root(cowork_home() / "sandbox", scope, store="sandbox")


def _real(path: Path) -> Path:
    """``path`` absolute, normalized and with every symlink resolved: the form
    the policy model takes, and the folder the launcher grants. A setting
    such as COWORK_HOME may be relative or hold ``..``."""
    return Path(os.path.realpath(path))


class _Anchored(NamedTuple):
    """A path as one of the server's roots, resolved, and the names below it."""

    root: Path
    names: tuple[str, ...]


def _trusted_roots() -> list[Path]:
    """The folders the operator configures for the server's own state:
    cowork-home, the shared root, and the projects, files and skills roots.
    The outermost one holding a path is trusted as configured, a link
    included. Nothing below it is, a root configured inside it included,
    since agent code before the sandbox could write anywhere under it."""
    settings = get_app_settings()
    return [
        cowork_home(),
        Path(settings.storage.shared_root),
        Path(settings.project.root_dir),
        Path(settings.file.root_dir),
        Path(settings.skill.root_dir),
    ]


def _anchored(path: Path) -> _Anchored | None:
    """``path`` below the outermost trusted root that holds it, the one that
    leaves the most names to check, or None when no trusted root holds it (an
    adopted project folder, the packaged and managed skills). Each root is
    compared as written and as resolved, since a project's stored path is
    resolved (safe_join)."""
    lexical = Path(os.path.abspath(path))
    best: _Anchored | None = None
    for root in _trusted_roots():
        real = _real(root)
        for form in (Path(os.path.abspath(root)), real):
            try:
                names = lexical.relative_to(form).parts
            except ValueError:
                continue
            if best is None or len(names) > len(best.names):
                best = _Anchored(real, names)
    return best


def _granted_dir(path: Path, *, create: bool = False, missing_ok: bool = False) -> Path:
    """``path`` as the policy grants it: absolute, normalized, with every
    link resolved.

    Inside a trusted root (``_anchored``), every folder below the root is
    opened without following links (opened_subdir_nofollow), and made first
    with ``create``. The grant is the resolved root joined with those names,
    the folder that was opened. Outside every trusted root, the path is
    resolved. Raises OSError when a folder on the way is a link or not a
    folder (ELOOP or ENOTDIR), and FileNotFoundError when one is missing,
    unless ``missing_ok``: the launcher skips a missing optional grant.
    """
    anchored = _anchored(path)
    if anchored is None:
        return _real(path)
    if create:
        anchored.root.mkdir(parents=True, exist_ok=True)
    try:
        with opened_subdir_nofollow(anchored.root, *anchored.names, create=create):
            pass
    except FileNotFoundError:
        if not missing_ok:
            raise
    return anchored.root.joinpath(*anchored.names)


#: What opening a link, or a file, as a folder without following links raises:
#: ELOOP on Linux, ENOTDIR on macOS and for a file.
_LINK_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR})


def _required_dir(path: Path, *, create: bool = False) -> Path:
    """``_granted_dir`` for a folder the session can't run without. A link
    on the way refuses the session with an OSError that names the folder."""
    try:
        return _granted_dir(path, create=create)
    except OSError as error:
        if error.errno not in _LINK_ERRNOS:
            raise
        raise OSError(
            error.errno,
            f"The code sandbox grants no folder through a link, and a folder on the way to "
            f"{path} is a link or not a folder. Agent code that ran before the sandbox may "
            "have left it there: remove it, then try again.",
        ) from error


def _read_only_dirs(paths: Iterable[Path]) -> tuple[Path, ...]:
    """The read-only grants for ``paths``, each once. A folder behind a link,
    or that isn't a folder, is left out and logged: granting it would grant
    where the link points, maybe another person's folder."""
    granted: dict[Path, None] = {}
    for path in paths:
        try:
            granted[_granted_dir(path, missing_ok=True)] = None
        except OSError as error:
            logger.warning(
                "code sandbox: not granting %s, a folder on the way is a link or not a folder (%s)",
                path,
                error.strerror,
            )
    return tuple(granted)


def _private_dir(root: Path, *, name: str) -> Path:
    """``root/name``, created when missing, as the policy grants it. Opened
    from the shared root (cowork-home for the operator view) without following
    links, so a link at the partition, ``sandbox/`` or ``name`` refuses the
    session instead of handing the cell where it points."""
    return _required_dir(root / name, create=True)


def refuse_a_linked_project(project_dir: Path) -> None:
    """Raise OSError when a folder on the way to ``project_dir`` below the
    server's roots is a link or not a folder (``_required_dir``).

    The harness calls it under the sandbox before the server writes into the
    project: the server isn't locked, so its own writes would follow such a
    link into another person's space before the policy refused it.
    """
    _required_dir(project_dir)


def probe_workspaces_dir(scope: TenantScope | None) -> Path:
    """Where a credential probe with no conversation runs under the sandbox:
    ``probes/`` in the asker's sandbox store, beside their HOME and TMPDIR and
    never in the shared /tmp."""
    return _private_dir(_sandbox_root(scope), name="probes")


def _partition_roots(shared_root: Path) -> list[Path]:
    """Every person's partition folder: the children of the shared root named
    by a partition id (cowork.principal.personal_partition_id, a UUID)."""
    try:
        entries = list(shared_root.iterdir())
    except FileNotFoundError:
        return []
    roots: list[Path] = []
    for entry in entries:
        try:
            if str(UUID(entry.name)) == entry.name:
                roots.append(entry)
        except ValueError:
            continue
    return roots


def _forbidden_roots(settings: AppSettings) -> tuple[Path, ...]:
    shared_root = Path(settings.storage.shared_root)
    roots = [cowork_home(), shared_root, shared_root / "shared", *_partition_roots(shared_root)]
    if settings.managed_dir is not None:
        roots.append(settings.managed_dir)
    return tuple(dict.fromkeys(_real(root) for root in roots))


def build_cell_sandbox(
    *,
    scope: TenantScope | None,
    project_dir: Path,
    read_only_dirs: Iterable[Path] = (),
) -> CellSandbox | None:
    """The policy one session's agent code runs under, or None with the
    sandbox off.

    ``project_dir`` is the session's working folder: the conversation's
    project, or a probe's own folder. ``read_only_dirs`` are the extra folders
    its cells may read. HOME and TMPDIR are created here, in ``scope``'s
    sandbox store, before the first spawn needs them, and so is the shared
    folder. Raises ValueError when a grant equals or contains a forbidden
    root, OSError when a folder on the way to the project, HOME, TMPDIR or
    the shared folder is a link (``_required_dir``), and
    MissingTenantScopeError when a partitioned deployment passes no scope.
    """
    settings = get_app_settings()
    if settings.code_sandbox != "landlock":
        return None
    from anton.core.backends.sandbox import CellSandbox

    root = _sandbox_root(scope)
    return CellSandbox(
        project_dir=_required_dir(project_dir),
        home_dir=_private_dir(root, name="home"),
        tmp_dir=_private_dir(root, name="tmp"),
        shared_dir=_required_dir(shared_files_dir(), create=True),
        read_only_dirs=_read_only_dirs(read_only_dirs),
        forbidden_roots=_forbidden_roots(settings),
        user_name=getpass.getuser(),
    )


# ── startup ──────────────────────────────────────────────────────────────────


def _probe_abi() -> int:
    """The running kernel's Landlock ABI, 0 when Landlock is unavailable."""
    from anton.core.backends.sandbox import landlock_abi

    return landlock_abi()


def _run_self_test() -> None:
    """Run anton's launcher self-test in a child process: it must write inside
    a granted folder and be denied a file beside it. Raises SandboxUnavailable
    with the reason otherwise."""
    from anton.core.backends.sandbox import self_test

    scratch = cowork_home() / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    self_test(scratch_dir=scratch)


def _anton_sandbox_gap() -> str | None:
    """What the installed anton lacks to carry the policy, or None when it
    has all of it. A missing piece would drop the policy and run every cell
    unlocked, so create_app refuses instead."""
    try:
        from anton.core.backends import sandbox
    except ImportError:
        return "has no anton.core.backends.sandbox"
    version = getattr(sandbox, "LAUNCHER_API_VERSION", None)
    if version is None:
        return "has no LAUNCHER_API_VERSION in anton.core.backends.sandbox"
    if version != LAUNCHER_API_VERSION:
        return (
            f"has launcher API version {version}, and this server builds "
            f"policies for version {LAUNCHER_API_VERSION}"
        )
    from anton.core.backends.local import LocalScratchpadRuntime
    from anton.core.session import ChatSessionConfig
    from anton.workspace import Workspace

    for owner in (ChatSessionConfig, LocalScratchpadRuntime):
        if "cell_sandbox" not in inspect.signature(owner).parameters:
            return f"has a {owner.__name__} that takes no cell_sandbox"
    # The harness creates the workspace's .anton folder and files with it,
    # so a link agent code planted there is refused rather than followed.
    if "refuse_links" not in inspect.signature(Workspace.initialize).parameters:
        return "has a Workspace.initialize that takes no refuse_links"
    return None


def check_startup(settings: AppSettings, *, remote_turns: bool) -> None:
    """Refuse to start where the agent's code would run unlocked, and say
    what a shared project leaves shared.

    Raises RuntimeError, naming COWORK_CODE_SANDBOX and the fix: identity
    headers with the sandbox off; and with ``landlock``, org tenancy, remote
    turns, an anton without the launcher, a kernel below the Landlock floor,
    or a self-test that can't confirm a denied path is denied. Then creates
    the shared folder.
    """
    if settings.trust_identity_headers and settings.code_sandbox == "off":
        raise RuntimeError(
            "COWORK_TRUST_IDENTITY_HEADERS needs COWORK_CODE_SANDBOX=landlock. With the "
            "sandbox off, each person's agent code runs with the server's own "
            "environment and files, so it can read and change every other person's "
            "space. Set COWORK_CODE_SANDBOX=landlock on a Linux host with Landlock "
            "(kernel 6.2 or later)."
        )
    if settings.code_sandbox != "landlock":
        return
    if settings.tenancy_mode == "org":
        raise RuntimeError(
            "COWORK_CODE_SANDBOX=landlock is not supported in org tenancy mode: org "
            "turns run on remote workers, never in this process. Leave "
            "COWORK_CODE_SANDBOX unset."
        )
    if remote_turns:
        raise RuntimeError(
            "COWORK_CODE_SANDBOX=landlock is not supported with COWORK_TURN_BACKEND=remote: "
            "remote workers start agent code without the launcher. Leave "
            "COWORK_TURN_BACKEND unset to run turns in this process."
        )
    gap = _anton_sandbox_gap()
    if gap is not None:
        raise RuntimeError(
            f"COWORK_CODE_SANDBOX=landlock needs an anton with the Landlock launcher. The "
            f"installed anton {gap}, so the policy would be dropped and agent code would "
            "run unlocked. Install an anton release that ships anton.core.backends.sandbox."
        )
    from anton.core.backends.sandbox import MIN_LANDLOCK_ABI

    abi = _probe_abi()
    if abi < MIN_LANDLOCK_ABI:
        raise RuntimeError(
            f"COWORK_CODE_SANDBOX=landlock needs Landlock ABI {MIN_LANDLOCK_ABI} or later, "
            f"and this kernel reports ABI {abi} (0 means Landlock is missing, off, or "
            "blocked by a seccomp filter). Run on Linux 6.2 or later with Landlock "
            "enabled (Debian 12 needs the bookworm-backports kernel, or Debian 13), "
            "under Docker Engine 20.10.18 or later."
        )
    try:
        _run_self_test()
    except Exception as error:
        raise RuntimeError(
            "COWORK_CODE_SANDBOX=landlock: the startup self-test couldn't confirm that "
            f"Landlock denies a path outside its grants ({error}). Check that the "
            "kernel's Landlock works inside this container and that the installed "
            "anton's launcher runs."
        ) from error
    try:
        _required_dir(shared_files_dir(), create=True)
    except OSError as error:
        raise RuntimeError(
            f"COWORK_CODE_SANDBOX=landlock can't make the shared folder: {error}"
        ) from error
    if (
        settings.trust_identity_headers
        and PersonalResource.PROJECTS not in settings.personal_resources
    ):
        logger.warning(
            "code sandbox: projects are left off COWORK_PERSONAL_RESOURCES, so every "
            "member of a shared project reads and writes its folder, including its "
            "venvs and saved cell state. One member's code can plant a file there "
            "that runs in the next member's scratchpad, with that member's "
            "connections. Add projects to COWORK_PERSONAL_RESOURCES to keep each "
            "person's agent code to their own projects."
        )
