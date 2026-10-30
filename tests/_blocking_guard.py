"""A runtime guard for blocking work where it stops other requests.

cowork-server answers every request from one event loop. A stat, a listing or
an open under COWORK_HOME, an NFS hard mount on hosted pods, that runs on the
loop stops every request on the pod until the file server answers. The same
call on one of anyio's default threads takes a thread that every
authenticated request's sync dependencies need, and on a database unit's
thread it holds a pooled connection while it waits. A pool checkout on the
loop stops the loop for up to POOL_TIMEOUT when the pool is full. File store
work belongs in ``run_efs`` and database work in ``run_db`` (cowork/db/units.py).

The guard wraps ``os.stat``, ``os.lstat``, ``os.fstat``, ``os.open``,
``os.scandir``, ``os.listdir``, ``open`` and ``io.open``, and listens for
SQLAlchemy pool checkouts. It records a site:

* on a thread running an event loop: a pool checkout, and a file store call
  under COWORK_HOME or on a file descriptor. The site is the innermost
  coroutine on the stack, the one that stopped the loop, when it is
  cowork's.
* on a worker thread, anyio's or the loop's default executor's, that
  ``run_efs`` did not start: a file store call under COWORK_HOME or on a file
  descriptor. The site is the outermost cowork function on the thread's stack,
  the one handed to the thread, past ``run_db``'s own runner.

File store calls count only in org tenancy, where COWORK_HOME is the shared
file store. On a desktop it is a local disk, and its file work on the loop is
the desktop's own trade. A stat of ``<COWORK_HOME>/.env`` doesn't count
either: pydantic-settings fixes the dotenv chain when cowork is imported,
and this suite imports it in local tenancy, so an org-tenancy test stats a
file an org pod's chain leaves out (``_env_file_chain``).

Nothing is recorded for work a test arranged: a call whose innermost
coroutine is a test's, such as a stub that runs a unit inline, or a function
a test handed to a thread. A library's own thread, such as Starlette's
``FileResponse`` reading a file, names no cowork function to blame either.

Each site is checked against ``ALLOWLIST``, keyed by (path under the repo,
function, kind), each entry with its reason. A site not on the list fails the
test that reached it. An entry no test reached fails the session, but only on
a full-suite run, since what a runtime guard finds depends on which tests ran
(tests/conftest.py).
"""
from __future__ import annotations

import asyncio
import builtins
import io
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from inspect import CO_ASYNC_GENERATOR, CO_COROUTINE
from typing import NamedTuple

from sqlalchemy import event
from sqlalchemy.pool import Pool

import cowork
from cowork.db import units

LOOP_FILE_STORE = "file store on the loop"
THREAD_FILE_STORE = "file store off run_efs"
LOOP_POOL_CHECKOUT = "pool checkout on the loop"

_COWORK_DIR = os.path.dirname(os.path.abspath(cowork.__file__)) + os.sep
_REPO_DIR = os.path.dirname(_COWORK_DIR.rstrip(os.sep)) + os.sep
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
_UNITS_FILE = os.path.abspath(units.__file__)
_ASYNC_CODE = CO_COROUTINE | CO_ASYNC_GENERATOR
_WRAPPED_OS = ("stat", "lstat", "fstat", "open", "scandir", "listdir")


class Site(NamedTuple):
    path: str
    function: str
    kind: str


#: (path under the repo, function, kind, reason). One entry per site.
ALLOWLIST: tuple[tuple[str, str, str, str], ...] = (
    ('cowork/api/v1/endpoints/artifact_workspace.py', 'set_artifact_access', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/api/v1/endpoints/artifact_workspace.py', '_sync_live_artifact', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/api/v1/endpoints/artifacts.py', 'delete_artifact_for_request', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/handlers/responses.py', 'ResponsesHandler._run_turn', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/handlers/responses.py', 'ResponsesHandler._collect', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/harnesses/anton_harness/harness.py', 'AntonHarness._build_turn_session', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/services/artifact_autopublish.py', 'autopublish_project_artifacts', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/services/providers.py', 'ping_provider.<locals>._chat_probe', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/turnqueue/producer.py', 'stream_remote_replies', LOOP_POOL_CHECKOUT, "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    ('cowork/services/memory.py', 'MemoryService.update_memory', LOOP_POOL_CHECKOUT, 'MemoryService runs its sync body on the loop: the project row and the memory files, and the settings through get_user_settings'),
    ('cowork/services/memory.py', 'MemoryService.get_memory', LOOP_POOL_CHECKOUT, 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    ('cowork/services/memory.py', 'MemoryService.list_memory', LOOP_POOL_CHECKOUT, 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    ('cowork/services/memory.py', 'MemoryService.delete_memory', LOOP_POOL_CHECKOUT, 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    ('cowork/api/v1/endpoints/project_files.py', 'upload_project_files', LOOP_POOL_CHECKOUT, 'lists the projects for its pinned-directory inventory on the loop, then writes the uploads through it'),
    ('cowork/server.py', 'lifespan', LOOP_POOL_CHECKOUT, 'startup work before uvicorn binds the port, when no request waits on the loop'),
    ('cowork/api/v1/endpoints/coding.py', 'inference_proxy', LOOP_POOL_CHECKOUT, 'desktop only (LoopbackDesktopOnly on the router)'),
    ('cowork/api/v1/endpoints/publish.py', 'activate_version_endpoint', LOOP_POOL_CHECKOUT, 'desktop only (DesktopOnly on the publish router)'),
    ('cowork/api/v1/endpoints/publish.py', 'list_publishable_endpoint', LOOP_POOL_CHECKOUT, 'desktop only (DesktopOnly on the publish router)'),
    ('cowork/api/v1/endpoints/publish.py', 'list_versions_endpoint', LOOP_POOL_CHECKOUT, 'desktop only (DesktopOnly on the publish router)'),
    ('cowork/api/v1/endpoints/publish.py', 'unpublish_artifact', LOOP_POOL_CHECKOUT, 'desktop only (DesktopOnly on the publish router)'),
    ('cowork/api/v1/endpoints/settings.py', 'write_raw_settings', LOOP_POOL_CHECKOUT, 'desktop only (LoopbackDesktopOnly)'),
    ('cowork/services/providers.py', 'warm_enabled_model_map', LOOP_POOL_CHECKOUT, 'desktop only: the boot warm and POST /api/v1/settings/raw, which org tenancy skips and refuses'),
    ('cowork/services/memory.py', 'MemoryService.update_memory', LOOP_FILE_STORE, 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    ('cowork/services/memory.py', 'MemoryService.get_memory', LOOP_FILE_STORE, 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    ('cowork/services/memory.py', 'MemoryService.list_memory', LOOP_FILE_STORE, 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    ('cowork/services/memory.py', 'MemoryService.delete_memory', LOOP_FILE_STORE, 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    ('cowork/api/v1/endpoints/project_files.py', 'upload_project_files', LOOP_FILE_STORE, 'opens every project folder for its pinned-directory inventory, and writes the uploads, on the loop'),
    ('cowork/services/artifacts.py', 'mount_preview', LOOP_FILE_STORE, "opens the previewed artifact's folder on the loop"),
    ('cowork/api/v1/endpoints/files.py', 'retrieve_file_content', THREAD_FILE_STORE, "sync route: reads the stored file on anyio's default threads"),
    ('cowork/api/v1/endpoints/memory.py', 'list_memory', THREAD_FILE_STORE, "sync route: reads the memory files on anyio's default threads"),
    ('cowork/api/v1/endpoints/memory.py', 'update_memory', THREAD_FILE_STORE, "sync route: writes the memory files on anyio's default threads"),
    ('cowork/api/v1/endpoints/project_files.py', 'download_project_file', THREAD_FILE_STORE, "sync route: opens the project file it streams on anyio's default threads"),
    ('cowork/api/v1/endpoints/project_files.py', 'preview_asset', THREAD_FILE_STORE, "sync route: opens the project file it streams on anyio's default threads"),
    ('cowork/api/v1/endpoints/project_files.py', 'write_project_file', THREAD_FILE_STORE, "sync route: creates the project folder and writes the file on anyio's default threads"),
    ('cowork/api/v1/endpoints/projects.py', 'create_project', THREAD_FILE_STORE, "sync route: creates the project folder on anyio's default threads"),
    ('cowork/api/v1/endpoints/projects.py', 'list_projects', THREAD_FILE_STORE, "sync route: creates the general project's folder when it is missing, on anyio's default threads"),
    ('cowork/api/v1/endpoints/projects.py', 'update_project', THREAD_FILE_STORE, "sync route: checks the project's instructions before a move, on anyio's default threads"),
    ('cowork/api/v1/endpoints/skills.py', 'create_skill', THREAD_FILE_STORE, "sync route: resolves the skill folder on anyio's default threads"),
    ('cowork/api/v1/endpoints/skills.py', 'update_skill', THREAD_FILE_STORE, "sync route: reads the skill folder on anyio's default threads"),
    ('cowork/api/v1/endpoints/skills.py', '_upload_skill_in_new_session', THREAD_FILE_STORE, "a skill upload's file work, handed to anyio's default threads"),
    ('cowork/channels/runtime.py', 'AntonChannelRuntime._resolve_or_create_binding', THREAD_FILE_STORE, "a run_db unit that creates the default project's folder for a new binding"),
    ('cowork/handlers/responses.py', 'ResponsesHandler._read_turn_start', THREAD_FILE_STORE, "a run_db unit that also reads the turn's attachments"),
    ('cowork/services/connectors/vault_secrets.py', '_restore_from_vault', THREAD_FILE_STORE, "reads the organization's connection vault through asyncio.to_thread"),
    ('cowork/services/publish.py', 'publish_artifact', THREAD_FILE_STORE, "the agent's publish tool reads the artifact in its tool thread"),
)


def allowed_sites() -> frozenset[Site]:
    sites = [Site(path, function, kind) for path, function, kind, _reason in ALLOWLIST]
    duplicated = sorted({site for site in sites if sites.count(site) > 1})
    assert not duplicated, f"listed twice: {duplicated}"
    return frozenset(sites)


class _Watch:
    """The sites found while a watch is open, from every thread."""

    def __init__(self, roots: tuple[str, ...]) -> None:
        self.roots = roots
        self.sites: set[Site] = set()
        self._lock = threading.Lock()

    def add(self, site: Site) -> None:
        with self._lock:
            self.sites.add(site)


_watches: list[_Watch] = []
_state = threading.local()
_homes: dict[str, tuple[str, ...]] = {}


def _home_prefixes() -> tuple[str, ...]:
    """COWORK_HOME as given and with its links resolved, read on each call
    since tests point it elsewhere. Resolved once per value."""
    raw = os.environ.get("COWORK_HOME")
    if not raw:
        return ()
    prefixes = _homes.get(raw)
    if prefixes is None:
        given = os.path.abspath(os.path.expanduser(raw))
        prefixes = tuple({given, os.path.realpath(given)})
        _homes[raw] = prefixes
    return prefixes


def _reaches_the_file_store(args: tuple, kwargs: dict) -> bool:
    if kwargs.get("dir_fd") is not None:
        return True
    if not args:
        return False
    target = args[0]
    if isinstance(target, int):
        return True
    try:
        path = os.fspath(target)
    except TypeError:
        return False
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    path = os.path.abspath(path)
    for home in _home_prefixes():
        if path == os.path.join(home, ".env"):
            return False
        if path == home or path.startswith(home + os.sep):
            return True
    return False


def _on_a_loop() -> bool:
    return asyncio.events._get_running_loop() is not None


def _on_a_worker() -> bool:
    thread = threading.current_thread()
    return type(thread).__name__ == "WorkerThread" or thread.name.startswith("asyncio_")


def _in_org_tenancy() -> bool:
    """Read from the environment, as tests set it: get_app_settings() here
    would fill its cache before a test sets the deployment it builds."""
    return os.environ.get("COWORK_TENANCY_MODE") == "org"


def _on_efs_thread() -> bool:
    marked = getattr(units, "on_efs_thread", None)
    return bool(marked and marked())


def _site_of(code, roots: tuple[str, ...], kind: str) -> Site | None:
    filename = code.co_filename
    if not filename.startswith(roots):
        return None
    path = os.path.relpath(filename, _REPO_DIR).replace(os.sep, "/")
    return Site(path, code.co_qualname, kind)


def _innermost_coroutine(frame, roots: tuple[str, ...], kind: str) -> Site | None:
    while frame is not None:
        code = frame.f_code
        if code.co_flags & _ASYNC_CODE:
            return _site_of(code, roots, kind)
        frame = frame.f_back
    return None


def _outermost_function(frame, roots: tuple[str, ...], kind: str) -> Site | None:
    """The outermost cowork function on the thread's stack, or None when a
    test's function sits outside it."""
    found = None
    while frame is not None:
        code = frame.f_code
        if code.co_filename != _UNITS_FILE:
            site = _site_of(code, roots, kind)
            if site is not None:
                found = site
            elif code.co_filename.startswith(_TESTS_DIR):
                found = None
        frame = frame.f_back
    return found


def _record(frame, *, file_store: bool) -> None:
    if file_store and not _in_org_tenancy():
        return
    for watch in tuple(_watches):
        if _on_a_loop():
            kind = LOOP_FILE_STORE if file_store else LOOP_POOL_CHECKOUT
            site = _innermost_coroutine(frame, watch.roots, kind)
        elif file_store and _on_a_worker() and not _on_efs_thread():
            site = _outermost_function(frame, watch.roots, THREAD_FILE_STORE)
        else:
            site = None
        if site is not None:
            watch.add(site)


def _guarded(real):
    def call(*args, **kwargs):
        if _watches and not getattr(_state, "busy", False):
            _state.busy = True
            try:
                if _reaches_the_file_store(args, kwargs):
                    _record(sys._getframe(1), file_store=True)
            finally:
                _state.busy = False
        return real(*args, **kwargs)

    call.__wrapped__ = real
    return call


def _on_checkout(_dbapi_connection, _record_, _proxy) -> None:
    if not getattr(_state, "busy", False):
        _state.busy = True
        try:
            _record(sys._getframe(1), file_store=False)
        finally:
            _state.busy = False


_originals: dict[tuple[object, str], object] = {}


def install() -> None:
    """Wrap the calls and listen for checkouts. Once per process."""
    if _originals:
        return
    for name in _WRAPPED_OS:
        _originals[(os, name)] = getattr(os, name)
        setattr(os, name, _guarded(getattr(os, name)))
    real_open = io.open
    _originals[(io, "open")] = real_open
    _originals[(builtins, "open")] = builtins.open
    guarded_open = _guarded(real_open)
    io.open = guarded_open
    builtins.open = guarded_open
    event.listen(Pool, "checkout", _on_checkout)


@contextmanager
def watching(*, roots: tuple[str, ...] = (_COWORK_DIR,)) -> Iterator[set[Site]]:
    """The sites found while the block runs, in code under ``roots``."""
    install()
    watch = _Watch(roots)
    _watches.append(watch)
    try:
        yield watch.sites
    finally:
        _watches.remove(watch)
