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

- read, write and execute: the conversation's project folder, and the asker's
  own HOME and TMPDIR, ``home/`` and ``tmp/`` in the ``sandbox`` store of
  their partition;
- read and write, no execute: ``<cowork-home>/shared-files``, which every
  person's agent uses. No file there runs by its path, so nothing one person
  leaves there starts as a program in another's cell; an interpreter can
  still run a script it reads from there (``python script.py``);
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
managed folder. ``other_partitions`` names every partition but the asker's,
inside which the policy model refuses a writable grant. The model refuses
such a grant when it is built, so a builder bug fails the session instead of
opening every person's space.

The server itself isn't locked, and it reads and writes some of a person's
stores outside the sandbox: their data vault, the memory tiers a turn hands
anton (the deployment's shared ones where memory is shared), and the publish
store. ``refuse_planted_links`` refuses a link, or a hard-linked file, that
agent code before the sandbox left there, before the server follows it into
another person's space.

``live_apps_on`` asks anton whether live apps run full-stack apps in app
slots outside this server.

``check_startup`` runs in create_app. It refuses identity headers with the
sandbox off, live apps without what keeps each app to its launcher
(``_check_live_apps_startup``), and ``landlock`` where it can't hold. ``_probe_abi`` and
``_run_self_test`` are the one seam tests stub, since neither a macOS machine
nor every CI kernel has Landlock.
"""

from __future__ import annotations

import errno
import getpass
import inspect
import os
import stat
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import urlsplit
from uuid import UUID

from cowork.common.logger import get_logger
from cowork.common.paths import (
    PinnedDir,
    cowork_home,
    dir_lstat,
    dir_scandir,
    open_pinned_child,
    opened_subdir_nofollow,
    pinned_dir,
)
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


#: anton's live-apps switch, which the live-apps overlay sets on this service.
#: anton reads it (live_apps_enabled); check_startup reads it by name only to
#: refuse it on an anton that can't.
LIVE_APP_SLOTS_VARIABLE = "ANTON_LIVE_APP_SLOTS"

#: The shortest COWORK_LIVE_APPS_LAUNCH_TOKEN check_startup accepts.
MIN_LIVE_APPS_LAUNCH_TOKEN_LENGTH = 32

#: What this server uses of anton's live-app slots (anton.core.artifacts.live_app_slots).
_LIVE_APP_SLOTS_API = (
    "live_apps_enabled",
    "live_app_slot_addresses",
    "SlotStatus",
    "LiveAppLauncher",
    "LAUNCH_FILES",
    "LAUNCH_DIRS",
    "MAX_LAUNCH_FILES",
    "MAX_LAUNCH_FILE_BYTES",
    "MAX_LAUNCH_TOTAL_BYTES",
    "MAX_REPLY_BYTES",
)


def live_apps_on() -> bool:
    """Whether live apps run in app slots outside this server: anton's own
    reading of ANTON_LIVE_APP_SLOTS. An anton without the slots reads as off,
    and that means the variable is unset: check_startup refuses it on such an
    anton (_live_apps_gap)."""
    try:
        from anton.core.artifacts.live_app_slots import live_apps_enabled
    except ImportError:
        return False
    return live_apps_enabled()


def refuse_a_missing_policy(cell_sandbox: CellSandbox | None, *, site: str) -> None:
    """Raise RuntimeError when the sandbox is on and ``cell_sandbox`` is None.

    Called where anton's executor is built (build_chat_session, and the
    backend pool's _make_runtime and get_or_create), behind the static
    test that reads each construction (tests/test_no_subprocess_static.py):
    a construction that drops the policy would run the agent's code
    unlocked. ``site`` names what was being built.
    """
    if cell_sandbox is None and code_sandbox_on():
        raise RuntimeError(
            f"COWORK_CODE_SANDBOX=landlock, and {site} was built without a code-sandbox "
            "policy, so its agent code would run unlocked. Build the policy with "
            "cowork.code_sandbox.build_cell_sandbox and pass it as cell_sandbox."
        )


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


class _TrustedRoots:
    """Configured roots resolved for one policy build, never descendant grants."""

    def __init__(self) -> None:
        self.resolved = {root: _real(root) for root in dict.fromkeys(_trusted_roots())}
        self.forms = tuple(
            (Path(os.path.normcase(form)).parts, real)
            for root, real in self.resolved.items()
            for form in dict.fromkeys((Path(os.path.abspath(root)), real))
        )

    def real(self, path: Path) -> Path:
        """Reuse only configured roots; other paths still resolve where requested."""
        if path in self.resolved:
            return self.resolved[path]
        return _real(path)


def _anchored(path: Path, *, roots: _TrustedRoots | None = None) -> _Anchored | None:
    """``path`` below the outermost trusted root that holds it, the one that
    leaves the most names to check, or None when no trusted root holds it (an
    adopted project folder, the packaged and managed skills). Each root is
    compared as written and as resolved, since a project's stored path is
    resolved (safe_join)."""
    lexical = Path(os.path.abspath(path))
    parts = lexical.parts
    matched_parts = Path(os.path.normcase(lexical)).parts
    best: _Anchored | None = None
    for form, real in (roots if roots is not None else _TrustedRoots()).forms:
        if matched_parts[:len(form)] != form:
            continue
        names = parts[len(form):]
        if best is None or len(names) > len(best.names):
            best = _Anchored(real, names)
    return best


def _granted_dir(
    path: Path, *, create: bool = False, missing_ok: bool = False, roots: _TrustedRoots | None = None,
) -> Path:
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
    anchored = _anchored(path, roots=roots)
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


def _required_dir(path: Path, *, create: bool = False, roots: _TrustedRoots | None = None) -> Path:
    """``_granted_dir`` for a folder the session can't run without. A link
    on the way refuses the session with an OSError that names the folder."""
    try:
        return _granted_dir(path, create=create, roots=roots)
    except OSError as error:
        if error.errno not in _LINK_ERRNOS:
            raise
        raise OSError(
            error.errno,
            f"The code sandbox grants no folder through a link, and a folder on the way to "
            f"{path} is a link or not a folder. Agent code that ran before the sandbox may "
            "have left it there: remove it, then try again.",
        ) from error


def _read_only_dirs(paths: Iterable[Path], *, roots: _TrustedRoots | None = None) -> tuple[Path, ...]:
    """The read-only grants for ``paths``, each once. A folder behind a link,
    or that isn't a folder, is left out and logged: granting it would grant
    where the link points, maybe another person's folder."""
    granted: dict[Path, None] = {}
    for path in paths:
        try:
            granted[_granted_dir(path, missing_ok=True, roots=roots)] = None
        except OSError as error:
            logger.warning(
                "code sandbox: not granting %s, a folder on the way is a link or not a folder (%s)",
                path,
                error.strerror,
            )
    return tuple(granted)


def _private_dir(root: Path, *, name: str, roots: _TrustedRoots | None = None) -> Path:
    """``root/name``, created when missing, as the policy grants it. Opened
    from the shared root (cowork-home for the operator view) without following
    links, so a link at the partition, ``sandbox/`` or ``name`` refuses the
    session instead of handing the cell where it points."""
    return _required_dir(root / name, create=True, roots=roots)


def refuse_a_linked_project(project_dir: Path) -> None:
    """Raise OSError when a folder on the way to ``project_dir`` below the
    server's roots is a link or not a folder (``_required_dir``), and make
    the project folder, and any missing folder above it, without following
    links when it is gone, as a turn with the sandbox off makes it.

    The harness calls it under the sandbox before the server writes into the
    project: the server isn't locked, so its own writes would follow such a
    link into another person's space before the policy refused it.
    """
    _required_dir(project_dir, create=True)


def refuse_a_linked_folder(folder: Path) -> None:
    """Raise OSError when a folder on the way to ``folder`` below the server's
    roots is a link or not a folder (``_required_dir``), and FileNotFoundError
    when ``folder`` is gone. Unlike refuse_a_linked_project it makes nothing,
    so it fits a read of someone else's folder."""
    _required_dir(folder, create=False)


class PlantedLinkError(OSError):
    """A link, or a file with a second name (a hard link), in a store the
    server reads and writes outside the sandbox (refuse_planted_links)."""


def _planted(path: Path, *, what: str) -> PlantedLinkError:
    return PlantedLinkError(
        f"{path} is {what}, and the server reads and writes it outside the code "
        "sandbox, so it would follow it wherever it leads. Agent code that ran before "
        "the sandbox may have left it there: remove it, then try again."
    )


def _refuse_planted_entry(folder: PinnedDir, *, name: str, path: Path) -> None:
    """Refuse ``name`` in ``folder`` when it is a link, or a regular file with
    more than one name, and walk it when it is a folder. ``path`` is where it
    sits, for the error. Nothing is followed."""
    try:
        info = dir_lstat(folder, name)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise _planted(path, what="a link")
    if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
        raise _planted(path, what=f"a file with {info.st_nlink} names (a hard link)")
    if not stat.S_ISDIR(info.st_mode):
        return
    below = open_pinned_child(folder, name)
    try:
        with dir_scandir(below) as entries:
            children = [entry.name for entry in entries]
        for child in children:
            _refuse_planted_entry(below, name=child, path=path / child)
    finally:
        below.close()


def refuse_planted_links(path: Path, *, only: tuple[str, ...] = ()) -> None:
    """Under the sandbox, raise PlantedLinkError when ``path``, anything below
    it, or a folder on the way to it from its trusted root (``_anchored``) is
    a link, or when a regular file there has more than one name. Does nothing
    with the sandbox off. Outside every trusted root, ``path`` is trusted as
    configured and resolved, as _granted_dir has it, and what is below it is
    checked.

    With ``only``, ``path`` is a folder holding more than the store, and just
    those names in it are checked: the publish store's files, in a folder
    that may be cowork-home itself, or one the operator sets outside every
    trusted root. Then only the folder is trusted as configured, never a
    link at one of those names.

    For the stores of a person that the server reads and writes outside the
    sandbox: their data vault, the memory tiers a turn hands anton (the
    deployment's shared ones where memory is shared), and the publish store.
    No locked cell can plant either kind of link there (no grant carries the
    right to make one, and none reaches these stores), but agent code that
    ran before the sandbox could, and the server would follow it into
    another person's space. So this checks the stores as they are,
    not against a race. Neither the server nor anton makes a link in them:
    skill links sit in a project's ``skills/`` and a turn's scratch folder,
    and the memory layout's runtime links (MemoryLayout.ensure_layout) at a
    harness's own path, outside every store. So no link is exempt. A missing
    folder passes: the caller makes it below folders this checked.
    """
    if not code_sandbox_on():
        return
    anchored = _anchored(path)
    if anchored is None:
        resolved = _real(path)
        anchored = _Anchored(resolved, ()) if only else _Anchored(resolved.parent, (resolved.name,))
    # The folders opened on the way, and the names checked in the last one.
    # Without ``only`` that is ``path`` itself, or everything in its trusted
    # root when ``path`` is that root.
    if only:
        on_the_way, checked = anchored.names, only
    else:
        on_the_way, checked = anchored.names[:-1], anchored.names[-1:]
    here = anchored.root
    try:
        with ExitStack() as opened:
            folder = opened.enter_context(pinned_dir(here))
            for name in on_the_way:
                here = here / name
                info = dir_lstat(folder, name)
                if stat.S_ISLNK(info.st_mode):
                    raise _planted(here, what="a link")
                if not stat.S_ISDIR(info.st_mode):
                    return
                folder = open_pinned_child(folder, name)
                opened.callback(folder.close)
            if not checked:
                with dir_scandir(folder) as entries:
                    checked = tuple(entry.name for entry in entries)
            for name in checked:
                _refuse_planted_entry(folder, name=name, path=here / name)
    except FileNotFoundError:
        return


def probe_workspaces_dir(scope: TenantScope | None) -> Path:
    """Where a credential probe with no conversation runs under the sandbox:
    ``probes/`` in the asker's sandbox store, beside their HOME and TMPDIR and
    never in the shared /tmp."""
    return _private_dir(_sandbox_root(scope), name="probes")


class _Partition(NamedTuple):
    """One person's partition folder: its name in the shared root, and the
    folder it resolves to (``_real``)."""

    name: str
    real: Path


def _resolved_entry(entry: os.DirEntry[str], *, real_root: Path) -> Path:
    """What ``entry`` in the shared root resolves to, given the root resolved.
    os.path.realpath resolves the parent first and then follows the last name
    only when it is a link, so an entry that isn't one resolves to the
    resolved root joined with its name. A link, or an entry whose type can't
    be read, is resolved on its own."""
    try:
        linked = entry.is_symlink()
    except OSError:
        linked = True
    if linked:
        return _real(Path(entry.path))
    return real_root / entry.name


def _partition_roots(shared_root: Path, *, roots: _TrustedRoots | None = None) -> list[_Partition]:
    """Every person's partition folder: the children of the shared root named
    by a partition id (cowork.principal.personal_partition_id, a UUID), each
    resolved once. The shared root is resolved once for all of them, and only
    a partition that is a link is resolved on its own (``_resolved_entry``)."""
    real_root = roots.real(shared_root) if roots is not None else _real(shared_root)
    try:
        with os.scandir(real_root) as scanned:
            entries = list(scanned)
    except FileNotFoundError:
        return []
    partitions: list[_Partition] = []
    for entry in entries:
        try:
            if str(UUID(entry.name)) != entry.name:
                continue
        except ValueError:
            continue
        partitions.append(_Partition(entry.name, _resolved_entry(entry, real_root=real_root)))
    return partitions


def _forbidden_roots(
    settings: AppSettings, *, partitions: list[_Partition], roots: _TrustedRoots | None = None,
) -> tuple[Path, ...]:
    """The folders no grant may equal or contain, each resolved and named
    once: cowork-home, the shared root, its ``shared/`` folder, every
    partition (resolved by _partition_roots), and the managed folder."""
    shared_root = Path(settings.storage.shared_root)
    real = roots.real if roots is not None else _real
    forbidden = [real(root) for root in (cowork_home(), shared_root, shared_root / "shared")]
    forbidden += [partition.real for partition in partitions]
    if settings.managed_dir is not None:
        forbidden.append(real(settings.managed_dir))
    return tuple(dict.fromkeys(forbidden))


def _other_partitions(
    scope: TenantScope | None, *, partitions: list[_Partition]
) -> tuple[Path, ...]:
    """Every partition but the asker's own: all of them for the operator view.
    The asker's own is the one named by their partition id in the shared
    root, whatever it resolves to."""
    own = scope.org_id if scope is not None and scope.personal else None
    return tuple(
        dict.fromkeys(partition.real for partition in partitions if partition.name != own)
    )


def _cell_sandbox(
    *,
    project_dir: Path,
    home_dir: Path,
    tmp_dir: Path,
    shared_dir: Path,
    read_only_dirs: tuple[Path, ...],
    forbidden_roots: tuple[Path, ...],
    other_partitions: tuple[Path, ...],
) -> CellSandbox:
    """anton's policy model with these fields: the one construction of it, so
    the startup check (_policy_model_gap) builds exactly what a session does."""
    from anton.core.backends.sandbox import CellSandbox

    return CellSandbox(
        project_dir=project_dir,
        home_dir=home_dir,
        tmp_dir=tmp_dir,
        shared_dir=shared_dir,
        read_only_dirs=read_only_dirs,
        forbidden_roots=forbidden_roots,
        other_partitions=other_partitions,
        user_name=getpass.getuser(),
    )


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
    root, or when a writable grant sits inside another person's partition,
    OSError when a folder on the way to the project, HOME, TMPDIR or the
    shared folder is a link (``_required_dir``), and MissingTenantScopeError
    when a partitioned deployment passes no scope.
    """
    settings = get_app_settings()
    if settings.code_sandbox != "landlock":
        return None
    root = _sandbox_root(scope)
    roots = _TrustedRoots()
    partitions = _partition_roots(Path(settings.storage.shared_root), roots=roots)
    return _cell_sandbox(
        project_dir=_required_dir(project_dir, roots=roots),
        home_dir=_private_dir(root, name="home", roots=roots),
        tmp_dir=_private_dir(root, name="tmp", roots=roots),
        shared_dir=_required_dir(shared_files_dir(), create=True, roots=roots),
        read_only_dirs=_read_only_dirs(read_only_dirs, roots=roots),
        forbidden_roots=_forbidden_roots(settings, partitions=partitions, roots=roots),
        other_partitions=_other_partitions(scope, partitions=partitions),
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
    return _policy_model_gap()


def _policy_model_gap() -> str | None:
    """What anton's CellSandbox refuses of the fields build_cell_sandbox
    passes, or None when it takes them all: a throwaway policy built the same
    way (_cell_sandbox), over folders nothing opens. Without it every session
    would fail to build its policy rather than the server refusing to start."""
    check = Path("/cowork-code-sandbox-check")
    other = check / "another-person"
    try:
        _cell_sandbox(
            project_dir=check / "project",
            home_dir=check / "home",
            tmp_dir=check / "tmp",
            shared_dir=check / "shared-files",
            read_only_dirs=(check / "skills",),
            forbidden_roots=(check / "cowork-home", other),
            other_partitions=(other,),
        )
    except (TypeError, ValueError) as error:
        return f"has a CellSandbox that refuses the policy this server builds ({error})"
    return None


def _live_apps_gap() -> str | None:
    """What the installed anton lacks of the live-app slots this server hands
    its people's launches to, or None when it has all of it."""
    try:
        from anton.core.artifacts import live_app_slots
    except ImportError:
        return "has no anton.core.artifacts.live_app_slots"
    missing = [name for name in _LIVE_APP_SLOTS_API if not hasattr(live_app_slots, name)]
    if missing:
        return f"has a live_app_slots without {', '.join(missing)}"
    from anton.core.session import ChatSessionConfig

    if "live_app_host" not in inspect.signature(ChatSessionConfig).parameters:
        return "has a ChatSessionConfig that takes no live_app_host"
    return None


def _check_live_apps_startup(settings: AppSettings) -> None:
    """Raise RuntimeError, naming what to change, where ANTON_LIVE_APP_SLOTS
    is set and live apps can't keep each app to the person who launched it.

    It runs when either the variable is set or anton reads live apps as on,
    since every runtime decision (live_apps_on) asks anton, and the slot
    paths skip the bearer only behind these refusals."""
    if not (os.environ.get(LIVE_APP_SLOTS_VARIABLE, "").strip() or live_apps_on()):
        return
    if settings.code_sandbox != "landlock":
        raise RuntimeError(
            f"{LIVE_APP_SLOTS_VARIABLE} needs COWORK_CODE_SANDBOX=landlock. Live apps run "
            "outside this server only for an agent whose code is locked to its person's "
            "space; leave the sandbox on and add the live-apps overlay beside it."
        )
    if not settings.trust_identity_headers:
        raise RuntimeError(
            f"{LIVE_APP_SLOTS_VARIABLE} needs COWORK_TRUST_IDENTITY_HEADERS. Only the person "
            "who launched an app may open it, and without identity headers this server "
            "knows no person."
        )
    if PersonalResource.PROJECTS not in settings.personal_resources:
        raise RuntimeError(
            f"{LIVE_APP_SLOTS_VARIABLE} needs projects in COWORK_PERSONAL_RESOURCES. In a "
            "shared project every member's agent can rewrite an app's backend, which then "
            "runs with the connections of whoever launches it next."
        )
    gap = _live_apps_gap()
    if gap is not None:
        raise RuntimeError(
            f"{LIVE_APP_SLOTS_VARIABLE} needs an anton with live-app slots. The installed "
            f"anton {gap}. Install the anton this cowork-server pins."
        )
    from anton.core.artifacts.live_app_slots import live_app_slot_addresses

    try:
        # A blank entry fails to parse, so a set variable names a slot.
        live_app_slot_addresses()
    except ValueError as error:
        raise RuntimeError(f"{LIVE_APP_SLOTS_VARIABLE} doesn't parse: {error}") from error
    viewer_login_url = settings.live_apps_viewer_login_url
    if viewer_login_url != viewer_login_url.strip() or not _is_viewer_login_url(viewer_login_url):
        raise RuntimeError(
            f"{LIVE_APP_SLOTS_VARIABLE} needs COWORK_LIVE_APPS_VIEWER_LOGIN_URL, the viewer "
            "sign-in's http:// or https:// address that tells this server who opens an app, "
            "for example http://viewer-login:4180."
        )
    token = settings.live_apps_launch_token.get_secret_value()
    if (
        len(token) < MIN_LIVE_APPS_LAUNCH_TOKEN_LENGTH
        or token != token.strip()
        or not token.isascii()
        or not token.isprintable()
    ):
        raise RuntimeError(
            f"{LIVE_APP_SLOTS_VARIABLE} needs COWORK_LIVE_APPS_LAUNCH_TOKEN of at least "
            f"{MIN_LIVE_APPS_LAUNCH_TOKEN_LENGTH} printable ASCII characters with no space "
            "around them, the token every app slot checks before it takes a launch."
        )


def _is_viewer_login_url(value: str) -> bool:
    """Whether ``value`` is an http or https address with a host and no query
    or fragment, which the access check can join /oauth2/userinfo to."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname) and not parts.query and not parts.fragment


#: cowork-home's own files and folders the server reads and writes at fixed
#: names: its settings (.env), the encryption key, every answer stream, the
#: turns' scratch folders and the self-test's, and the shared registries.
_COWORK_HOME_ENTRIES = (".env", ".master_key", "streams", "tmp", "shared")


def _refuse_linked_cowork_home_entries() -> None:
    """Raise RuntimeError when one of _COWORK_HOME_ENTRIES is a link. The
    server isn't locked, so it would read and write wherever the link leads,
    such as every person's answer streams into the shared folder."""
    for name in _COWORK_HOME_ENTRIES:
        entry = cowork_home() / name
        if entry.is_symlink():
            raise RuntimeError(
                f"COWORK_CODE_SANDBOX=landlock: {entry} is a link, and the server reads "
                "and writes it outside the sandbox, so it would follow it wherever it "
                "leads. Agent code that ran before the sandbox may have left it there: "
                "remove the link and put the real file or folder back, then start again."
            )


def check_startup(settings: AppSettings, *, remote_turns: bool) -> None:
    """Refuse to start where the agent's code would run unlocked, and say
    what a shared project leaves shared.

    Raises RuntimeError, naming COWORK_CODE_SANDBOX and the fix: identity
    headers with the sandbox off; live apps without what keeps each app to
    its launcher (_check_live_apps_startup); and with ``landlock``, org tenancy, remote
    turns, private connections with shared projects, an anton without the launcher or whose policy model refuses this
    server's policy, a kernel below the Landlock floor, a link at one of
    cowork-home's own entries (_COWORK_HOME_ENTRIES), or a self-test that
    can't confirm a denied path is denied. Then creates the shared folder.
    """
    if settings.trust_identity_headers and settings.code_sandbox == "off":
        raise RuntimeError(
            "COWORK_TRUST_IDENTITY_HEADERS needs COWORK_CODE_SANDBOX=landlock. With the "
            "sandbox off, each person's agent code runs with the server's own "
            "environment and files, so it can read and change every other person's "
            "space. Set COWORK_CODE_SANDBOX=landlock on a Linux host with Landlock "
            "(kernel 6.2 or later)."
        )
    _check_live_apps_startup(settings)
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
    if (
        settings.trust_identity_headers
        and PersonalResource.CONNECTIONS in settings.personal_resources
        and PersonalResource.PROJECTS not in settings.personal_resources
    ):
        raise RuntimeError(
            "With COWORK_TRUST_IDENTITY_HEADERS, private connections require personal projects: "
            "shared project venvs and saved cell state can run one person's code with another "
            "person's connections. Add projects to COWORK_PERSONAL_RESOURCES before starting. "
            "Preserve any shared project files before changing their storage namespace."
        )
    gap = _anton_sandbox_gap()
    if gap is not None:
        raise RuntimeError(
            f"COWORK_CODE_SANDBOX=landlock needs an anton with the Landlock launcher. The "
            f"installed anton {gap}, so agent code would run unlocked or not at all. "
            "Install the anton this cowork-server pins (launcher API version "
            f"{LAUNCHER_API_VERSION})."
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
    _refuse_linked_cowork_home_entries()
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
