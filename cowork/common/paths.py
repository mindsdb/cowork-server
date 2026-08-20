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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".cowork"


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


@contextmanager
def opened_subdir_nofollow(
    base: Path | str, *names: str, create: bool = False
) -> Iterator[int]:
    """Yield a directory descriptor for ``base/<names...>``, opening every
    component below *base* with ``O_NOFOLLOW`` so no symlink in the chain can
    redirect the caller out of *base*'s tree.

    *base* is trusted (a cowork-server-created directory, e.g. a project dir)
    and is opened normally. Each *name* is opened ``O_NOFOLLOW``, so a symlink
    planted at any level (the agent's pod mounts its own subtree read-write and
    can swap a component for a link into another org) raises ``OSError``
    (``ELOOP``) instead of being traversed. With ``create=True`` each missing
    level is ``mkdir``'d first; the open is still ``O_NOFOLLOW``, so a link
    already squatting the name is refused, not followed.

    The caller acts relative to the yielded fd (``os.open(child, dir_fd=fd)``,
    ``shutil.rmtree(child, dir_fd=fd)``), which the kernel resolves against the
    pinned inode with nothing left to swap between check and use. This is the
    same defence ``ProjectService`` applies to the projects root, and closes
    the ``safe_join`` gap where a symlinked *base* has already escaped before
    the containment check runs. Caller must NOT close the yielded fd.
    """
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in names:
            if create:
                try:
                    os.mkdir(name, dir_fd=fd)
                except FileExistsError:
                    pass
            nxt = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        yield fd
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

    return Path(target)


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
