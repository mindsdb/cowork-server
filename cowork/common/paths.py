"""The filesystem root for all cowork state.

Every piece of cowork state — the SQLite database, uploaded files, projects,
skills, memory, streams, the connector vault, the master key, the ``.env`` —
lives under a single data root. Pointing that root elsewhere isolates an
entire install: preview/stable desktop builds set ``COWORK_HOME`` to
``~/.cowork-<kind>`` so their state never collides with a user's production
``~/.cowork`` (ENG-324). Production leaves it unset and gets ``~/.cowork``.

Every default path in the codebase MUST derive from :func:`cowork_home` (via
the settings classes or directly) — a path that hardcodes ``~/.cowork`` would
silently leak across builds and defeat the isolation.
"""

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".cowork"

_IS_WINDOWS = os.name == "nt"

# ``O_DIRECTORY`` / ``O_NOFOLLOW`` are POSIX-only and simply absent from ``os``
# on Windows (referencing ``os.O_DIRECTORY`` there raises AttributeError). They
# exist so a directory-relative open refuses to traverse a symlink an untrusted
# agent pod could plant on shared multi-tenant storage — the org-mode threat
# model (Linux, EFS mounted read-write into each pod). Windows only ever runs
# ``tenancy_mode="local"`` (a single-user desktop sidecar), where no such pod
# exists and nobody but the local user can plant a link, so degrading these to
# 0 there drops a defence that has nothing left to defend. See ``PinnedDir``.
O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def cowork_home() -> Path:
    """Root directory for all cowork state (default ``~/.cowork``).

    Overridable via the ``COWORK_HOME`` env var, which desktop preview/stable
    builds set to isolate their data. Read from the environment on each call so
    tests can monkeypatch it; the desktop app sets it before the server process
    starts, so it is stable for the lifetime of a real run.
    """
    raw = os.environ.get("COWORK_HOME")
    return Path(raw).expanduser() if raw else _DEFAULT_HOME


def pod_local_only(local_path: Path, name: str) -> Path:
    """Relocate *local_path* off shared storage in org mode; a no-op otherwise.

    *local_path* is whatever a caller already computes for local (desktop)
    mode, normally something under ``cowork_home()``. In local mode this
    function returns it untouched, so desktop behaviour is exactly what it
    was before this function existed.

    In org mode, ``cowork_home()`` is EFS (``COWORK_HOME=/mnt/cowork-shared``),
    shared byte-for-byte across every replica and every organization. Three
    callers write under it with no org_id segment to key on: the connector
    probe's plaintext credential env files, publish's state.json (which holds
    publish_history), and the anton harness's temporary data-vault directory.
    Anything they write under ``cowork_home()`` therefore lands at the shared
    namespace root, readable by any organization's request, and, unlike the
    old ephemeral container disk, survives a pod restart.
    ``scoped_storage_root`` cannot fix this: it pivots on an org_id, and none
    of these three carry one, nor should they, since none of them are
    organization data.

    Org mode substitutes ``<pod_scratch_dir>/<name>`` instead, ignoring
    *local_path* entirely. ``pod_scratch_dir`` (see AppSettings) defaults to
    the container's own ``tempfile.gettempdir()``, which this plan never
    mounts shared storage onto, so it is always one pod's own disk and dies
    with the container the same way ``cowork_home()`` did before COWORK_HOME
    pointed at EFS.
    """
    from cowork.common.settings.app_settings import get_app_settings

    settings = get_app_settings()
    if settings.tenancy_mode != "org":
        return local_path
    return Path(settings.pod_scratch_dir) / name


@dataclass(frozen=True)
class PinnedDir:
    """A directory pinned for symlink-safe operations on its direct children.

    POSIX: ``fd`` is a directory descriptor (opened ``O_NOFOLLOW`` where the
    directory itself is agent-reachable). Every child operation passes
    ``dir_fd=fd``, so the kernel resolves the name against the pinned inode with
    nothing left to swap between check and use — the shared-EFS cross-tenant
    defence described on ``opened_subdir_nofollow``.

    Windows: there is no ``dir_fd`` (and no ``O_NOFOLLOW``/``O_DIRECTORY``), and
    the threat cannot arise — Windows only runs ``tenancy_mode="local"``, a
    single-user desktop with no pod planting links on shared storage. So ``fd``
    is ``None`` and each operation joins the child onto ``path``.

    Callers MUST go through the ``dir_*`` helpers below rather than read ``fd``
    or ``path`` directly, so the one platform branch stays in one place. Every
    child *name* passed to a helper must already be a validated single path
    component (the call sites guarantee this); the helpers do not re-check it.
    """

    fd: int | None
    path: Path

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)


def dir_open(d: PinnedDir, name: str, flags: int, mode: int = 0o777) -> int:
    """``os.open`` a child of *d*, returning a file descriptor."""
    if d.fd is not None:
        return os.open(name, flags, mode, dir_fd=d.fd)
    return os.open(d.path / name, flags, mode)


def dir_lstat(d: PinnedDir, name: str) -> os.stat_result:
    if d.fd is not None:
        return os.lstat(name, dir_fd=d.fd)
    return os.lstat(d.path / name)


def dir_stat(
    d: PinnedDir, name: str, *, follow_symlinks: bool = True
) -> os.stat_result:
    if d.fd is not None:
        return os.stat(name, dir_fd=d.fd, follow_symlinks=follow_symlinks)
    return os.stat(d.path / name, follow_symlinks=follow_symlinks)


def dir_unlink(d: PinnedDir, name: str) -> None:
    if d.fd is not None:
        os.unlink(name, dir_fd=d.fd)
    else:
        os.unlink(d.path / name)


def dir_mkdir(d: PinnedDir, name: str) -> None:
    if d.fd is not None:
        os.mkdir(name, dir_fd=d.fd)
    else:
        os.mkdir(d.path / name)


def dir_rmtree(d: PinnedDir, name: str) -> None:
    if d.fd is not None:
        shutil.rmtree(name, dir_fd=d.fd)
    else:
        shutil.rmtree(d.path / name)


def dir_rename_into(d: PinnedDir, src: str | Path, name: str) -> None:
    """Rename the absolute path *src* to child *name* of *d*.

    Only the destination is pinned; *src* is passed as an absolute path (see
    ``ProjectService._rename_in_root``)."""
    if d.fd is not None:
        os.rename(str(src), name, dst_dir_fd=d.fd)
    else:
        os.rename(str(src), str(d.path / name))


def dir_scandir(d: PinnedDir) -> "Iterator[os.DirEntry[str]]":
    """Scan the direct entries of *d*. Entries expose ``.name``,
    ``.is_symlink()`` and ``.is_dir(follow_symlinks=False)`` on both platforms."""
    if d.fd is not None:
        return os.scandir(d.fd)
    return os.scandir(d.path)


def open_pinned_child(d: PinnedDir, name: str, *, nofollow: bool = True) -> PinnedDir:
    """Descend into direct child directory *name* of *d*, returning a new
    ``PinnedDir`` the caller must ``close()``.

    POSIX opens it (``O_NOFOLLOW`` by default, refusing a planted link) relative
    to *d*'s descriptor; Windows just joins the path."""
    if d.fd is None:
        return PinnedDir(None, d.path / name)
    flags = os.O_RDONLY | O_DIRECTORY | (O_NOFOLLOW if nofollow else 0)
    return PinnedDir(os.open(name, flags, dir_fd=d.fd), d.path / name)


@contextmanager
def pinned_dir(
    base: Path | str, *, create: bool = False, nofollow_base: bool = False
) -> Iterator[PinnedDir]:
    """Yield a :class:`PinnedDir` for the trusted directory *base*.

    POSIX opens *base* as a directory descriptor (``O_NOFOLLOW`` when
    *nofollow_base*, for a base an agent could itself have swapped for a link);
    Windows carries *base* as a path with no descriptor. With *create*, *base*
    is ``mkdir``'d (parents included) first.
    """
    base = Path(base)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    if _IS_WINDOWS:
        yield PinnedDir(None, base)
        return
    fd = os.open(base, os.O_RDONLY | O_DIRECTORY | (O_NOFOLLOW if nofollow_base else 0))
    d = PinnedDir(fd, base)
    try:
        yield d
    finally:
        d.close()


@contextmanager
def opened_subdir_nofollow(
    base: Path | str, *names: str, create: bool = False
) -> Iterator[PinnedDir]:
    """Yield a :class:`PinnedDir` for ``base/<names...>``, opening every
    component below *base* with ``O_NOFOLLOW`` so no symlink in the chain can
    redirect the caller out of *base*'s tree.

    *base* is trusted (a cowork-server-created directory, e.g. a project dir)
    and is opened normally. Each *name* is opened ``O_NOFOLLOW``, so a symlink
    planted at any level (the agent's pod mounts its own subtree read-write and
    can swap a component for a link into another org) raises ``OSError``
    (``ELOOP``) instead of being traversed. With ``create=True`` each missing
    level is ``mkdir``'d first; the open is still ``O_NOFOLLOW``, so a link
    already squatting the name is refused, not followed.

    The caller acts relative to the yielded handle via the ``dir_*`` helpers,
    which on POSIX pass ``dir_fd`` so the kernel resolves against the pinned
    inode with nothing left to swap between check and use. This is the same
    defence ``ProjectService`` applies to the projects root, and closes the
    ``safe_join`` gap where a symlinked *base* has already escaped before the
    containment check runs.

    Windows has no ``dir_fd``/``O_NOFOLLOW`` and no shared-tenant threat (local
    desktop only), so the handle carries ``base/<names...>`` as a plain path,
    ``mkdir``'d level-by-level when *create*. A link planted mid-chain there
    would be traversed, but only the single local user could plant one.
    """
    base = Path(base)
    if _IS_WINDOWS:
        target = base.joinpath(*names)
        if create:
            target.mkdir(parents=True, exist_ok=True)
        yield PinnedDir(None, target)
        return
    fd = os.open(base, os.O_RDONLY | O_DIRECTORY)
    try:
        for name in names:
            if create:
                try:
                    os.mkdir(name, dir_fd=fd)
                except FileExistsError:
                    pass
            nxt = os.open(name, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        yield PinnedDir(fd, base.joinpath(*names))
    finally:
        os.close(fd)


def safe_join(base: Path | str, *parts: str) -> Path:
    """Join user-controlled *parts* onto *base*, guaranteeing containment.

    Two checks, both required:

    1. Lexical. Normalizes the result and rejects anything landing outside
       *base* (a ``..`` segment, an absolute component resetting the join, a
       name carrying a separator). Comparison is on whole path components, so
       *base* and a sibling like ``<base>-other`` are correctly unrelated.

    2. Symbolic. Resolves symlinks and re-checks. On shared storage the agent
       writes into its own org's tree but cowork-server reads every org's, so a
       link like ``<org_A>/x -> ../../<org_B>`` is a cross-tenant read if we
       only check the string. ``os.path.realpath`` on a path that does not exist
       yet resolves the existing prefix and leaves the rest literal, so callers
       that join before creating still work.

    Returns the RESOLVED path, not the lexical one. Returning the lexical path
    meant the caller acted on a different path from the one that was checked:
    every component was still a live symlink at use time, so the containment
    proof applied to a string nobody subsequently used. This does not make the
    join atomic. An attacker who can replace a component between this call and
    the caller's open or mkdir still redirects it, and closing that needs
    O_NOFOLLOW at the use site. It does remove the check-one-path-use-another
    mismatch, which is the part that made the guard misleading.
    """
    base_norm = os.path.normpath(str(base))
    target = os.path.normpath(os.path.join(base_norm, *parts))
    if os.path.commonpath([base_norm, target]) != base_norm:
        raise ValueError(f"path {target!r} escapes base directory {base_norm!r}")

    base_real = os.path.realpath(base_norm)
    target_real = os.path.realpath(target)
    if os.path.commonpath([base_real, target_real]) != base_real:
        raise ValueError(
            f"path {target!r} resolves outside base directory {base_norm!r}"
        )

    return Path(target_real)


def safe_join_lexical(base: Path | str, *parts: str) -> Path:
    """Join user-controlled *parts* onto *base*, checking the string only.

    Same lexical check as :func:`safe_join` (normpath plus a whole-component
    ``commonpath`` comparison), but it stops there: it does NOT call
    ``os.path.realpath`` and does NOT reject a result that a symlink would
    resolve outside *base*. It gives no protection against reading through a
    symlink that escapes *base*; use :func:`safe_join` for that, and for
    every path this process is about to read.

    This exists for the one legitimate case where a resolved-outside-base
    result is correct, not a bug: computing the location of a symlink this
    process is about to create or remove, where the whole point is that the
    link's target lives outside its own directory. ``skill_links.py`` fans a
    canonical skill out to per-project ``skills/<slug>`` symlinks; each
    project's link legitimately resolves to the shared skill store, a sibling
    of the projects root, not a path under it. Passing that computation
    through :func:`safe_join` would resolve the existing symlink on every call
    after the first and raise, even though nothing is being read through it.
    """
    base_norm = os.path.normpath(str(base))
    target = os.path.normpath(os.path.join(base_norm, *parts))
    if os.path.commonpath([base_norm, target]) != base_norm:
        raise ValueError(f"path {target!r} escapes base directory {base_norm!r}")
    return Path(target)
