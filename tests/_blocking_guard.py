"""A runtime guard for blocking work where it stops other requests.

cowork-server answers every request from one event loop. A stat, a listing or
an open under COWORK_HOME, an NFS hard mount on hosted pods, that runs on the
loop stops every request on the pod until the file server answers. The same
call on one of anyio's default threads takes a thread that every
authenticated request's sync dependencies need, and on a database unit's
thread it holds a pooled connection while it waits. A pool checkout on the
loop stops the loop for up to POOL_TIMEOUT when the pool is full. File store
work belongs in ``run_efs`` and database work in ``run_db`` (cowork/db/units.py).

The guard wraps the ``os`` calls that read or change the file system
(``_PATH_PARAMETERS``), ``open`` and ``io.open``, and listens for SQLAlchemy
pool checkouts. A call is file store work when a path it names, by position
or by keyword, is under COWORK_HOME, or when it names a file descriptor. The
guard records a site:

* on a thread running an event loop: a pool checkout, and file store work.
  The site is the innermost coroutine on the stack, the one that stopped the
  loop, when it is cowork's. Under a library's coroutine, such as FastAPI's
  running a cowork validator, it is the outermost cowork function that
  coroutine called, or when it called none, the next cowork coroutine out. In
  a loop callback, which runs under no coroutine, it is the outermost cowork
  function the callback ran, not the code that drives the loop.
* on a worker thread, anyio's or the loop's default executor's, that
  ``run_efs`` did not start: file store work. The site is the outermost
  cowork function on the thread's stack, the one handed to the thread, past
  ``run_db``'s own runner and past lambdas, so a unit handed over as
  ``lambda session: _list_projects(session, principal)`` is
  ``_list_projects``. When the thread runs no cowork function, as when a
  bare ``os.stat`` is handed over, the site is the coroutine that called
  ``anyio.to_thread.run_sync`` or ``asyncio.to_thread``. When that coroutine
  is a library's, the site is the route whose request it answers, as
  Starlette's ``FileResponse`` stats and opens its file after its route
  returned, or else the cowork coroutine that awaits it.

A site also names its leaf: ``path:qualname`` of the cowork function nearest
the call, past the file helpers in cowork/common/paths.py, or the helper
itself when the site's own function called it. For a call a library handed
over it is the library's coroutine that handed it over, and when nothing
nearer is cowork's it is the site's own function. So new work below a listed
function is a new site, even when it goes through a listed helper, and so is
a new caller of a listed function.

File store work counts only in org tenancy, where COWORK_HOME is the shared
file store: when the environment names it, as ``org_deployment`` does, or
when the settings ``get_app_settings()`` already holds name it, as a test
that patches their ``tenancy_mode`` does. A test that patches one module's
own check, such as its ``_org_mode``, switches that module and not the
deployment, so the guard doesn't count it. On a desktop COWORK_HOME is a
local disk, and its file work on the loop is the desktop's own trade. A stat
of ``<COWORK_HOME>/.env`` doesn't count either: pydantic-settings fixes the
dotenv chain when cowork is imported, and this suite imports it in local
tenancy, so an org-tenancy test stats a file an org pod's chain leaves out
(``_env_file_chain``).

Nothing is recorded for work a test arranged: a call whose innermost
coroutine is a test's, such as a stub that runs a unit inline, a function a
test handed to a thread, or a call a test's coroutine handed over.

Each site is checked against ``ALLOWLIST``. An entry names a function's sites
of one kind by their leaves, with its reason; ``(ANY_LEAF,)`` lists every
leaf, for a function whose reason covers its whole body, such as a
desktop-only route. A site no entry lists fails the test that reached it. A
listed leaf no test reached fails the session, but only on a full-suite run,
since what a runtime guard finds depends on which tests ran
(``StaleEntryCheck``).
"""
from __future__ import annotations

import asyncio
import builtins
import inspect
import io
import os
import sys
import sysconfig
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from inspect import CO_ASYNC_GENERATOR, CO_COROUTINE
from pathlib import Path
from types import CodeType, FrameType
from typing import NamedTuple

import anyio
import anyio.to_thread
import pytest
import starlette.concurrency
from sqlalchemy import event
from sqlalchemy.pool import Pool

import cowork
from cowork.common import paths
from cowork.common.settings.app_settings import get_app_settings
from cowork.db import units

LOOP_FILE_STORE = "file store on the loop"
THREAD_FILE_STORE = "file store off run_efs"
LOOP_POOL_CHECKOUT = "pool checkout on the loop"
ANY_LEAF = "*"

_COWORK_DIR = os.path.dirname(os.path.abspath(cowork.__file__)) + os.sep
_REPO_DIR = os.path.dirname(_COWORK_DIR.rstrip(os.sep)) + os.sep
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
_UNITS_FILE = os.path.abspath(units.__file__)
#: The shared file helpers: a leaf is named past them.
_FILE_HELPERS = os.path.abspath(paths.__file__)
#: What runs a loop callback: the frames below it drive the loop.
_CALLBACK_RUNNER = asyncio.events.Handle._run.__code__
#: Where a library's code lives: a library's site is named under these.
_LIBRARY_DIRS = tuple(
    sorted({sysconfig.get_paths()[key] + os.sep for key in ("purelib", "platlib", "stdlib")}, key=len, reverse=True)
)
#: Code that hands a call to a thread for its caller: the coroutine that
#: handed the call over is the first one past these.
_THREAD_HELPERS = (
    os.path.dirname(os.path.abspath(anyio.__file__)) + os.sep,
    os.path.abspath(starlette.concurrency.__file__),
    _UNITS_FILE,
)
_ASYNC_CODE = CO_COROUTINE | CO_ASYNC_GENERATOR
_LAMBDA = "<lambda>"

#: Each wrapped ``os`` call and its parameters that name a path or a file
#: descriptor, in order. ``None`` marks one that names neither: symlink's
#: first is the text of the link it makes.
_PATH_PARAMETERS: dict[str, tuple[str | None, ...]] = {
    "stat": ("path",),
    "lstat": ("path",),
    "fstat": ("fd",),
    "open": ("path",),
    "scandir": ("path",),
    "listdir": ("path",),
    "access": ("path",),
    "readlink": ("path",),
    "statvfs": ("path",),
    "mkdir": ("path",),
    "rmdir": ("path",),
    "unlink": ("path",),
    "remove": ("path",),
    "rename": ("src", "dst"),
    "replace": ("src", "dst"),
    "link": ("src", "dst"),
    "symlink": (None, "dst"),
    "chmod": ("path",),
    "utime": ("path",),
    "truncate": ("path",),
    "read": ("fd",),
    "write": ("fd",),
    "pread": ("fd",),
    "pwrite": ("fd",),
    "fsync": ("fd",),
    "fdatasync": ("fd",),
    "ftruncate": ("fd",),
    "fchmod": ("fd",),
    "fchown": ("fd",),
}
_OPEN_PARAMETERS = ("file",)
_DIR_FD_KEYWORDS = ("dir_fd", "src_dir_fd", "dst_dir_fd")


class Site(NamedTuple):
    path: str
    function: str
    kind: str
    leaf: str


class Allowed(NamedTuple):
    """One ALLOWLIST entry: a function's sites of one kind, by their leaves,
    and why the work may stay."""

    path: str
    function: str
    kind: str
    leaves: tuple[str, ...]
    reason: str

    def lists(self, site: Site) -> bool:
        if (self.path, self.function, self.kind) != (site.path, site.function, site.kind):
            return False
        return self.leaves == (ANY_LEAF,) or site.leaf in self.leaves


ALLOWLIST: tuple[Allowed, ...] = (
    Allowed('cowork/api/v1/endpoints/artifact_workspace.py', 'set_artifact_access', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/api/v1/endpoints/artifact_workspace.py', '_sync_live_artifact', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/api/v1/endpoints/artifacts.py', 'delete_artifact_for_request', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/handlers/responses.py', 'ResponsesHandler._run_turn', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/handlers/responses.py', 'ResponsesHandler._collect', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/harnesses/anton_harness/harness.py', 'AntonHarness._build_turn_session', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/services/artifact_autopublish.py', 'autopublish_project_artifacts', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/services/providers.py', 'ping_provider.<locals>._chat_probe', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/turnqueue/producer.py', 'stream_remote_replies', LOOP_POOL_CHECKOUT, ('cowork/services/settings.py:SettingService._fetch_all_rows',), "loads the caller's settings with get_user_settings, which checks out a connection on the loop"),
    Allowed('cowork/services/memory.py', 'MemoryService.update_memory', LOOP_POOL_CHECKOUT, (
        'cowork/db/scoped.py:ScopedSession.get',
        'cowork/services/memory.py:MemoryService._write',
        'cowork/services/settings.py:SettingService._fetch_all_rows',
        'cowork/services/shared_resources.py:SharedResourceAccess.mutation_lock',
        'cowork/services/shared_resources.py:SharedResourceAccess.release_claim',
    ), 'MemoryService runs its sync body on the loop: the project row and the memory files, and the settings through get_user_settings'),
    Allowed('cowork/services/memory.py', 'MemoryService.get_memory', LOOP_POOL_CHECKOUT, (
        'cowork/db/scoped.py:ScopedSession.exec',
        'cowork/db/scoped.py:ScopedSession.get',
    ), 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    Allowed('cowork/services/memory.py', 'MemoryService.list_memory', LOOP_POOL_CHECKOUT, ('cowork/db/scoped.py:ScopedSession.exec',), 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    Allowed('cowork/services/memory.py', 'MemoryService.delete_memory', LOOP_POOL_CHECKOUT, ('cowork/db/scoped.py:ScopedSession.get',), 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    Allowed('cowork/api/v1/endpoints/project_files.py', 'upload_project_files', LOOP_POOL_CHECKOUT, ('cowork/db/scoped.py:ScopedSession.exec',), 'lists the projects for its pinned-directory inventory on the loop, then writes the uploads through it'),
    Allowed('cowork/server.py', 'lifespan', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'startup work before uvicorn binds the port, when no request waits on the loop'),
    Allowed('cowork/api/v1/endpoints/coding.py', 'inference_proxy', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'desktop only (LoopbackDesktopOnly on the router)'),
    Allowed('cowork/api/v1/endpoints/publish.py', 'activate_version_endpoint', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'desktop only (DesktopOnly on the publish router)'),
    Allowed('cowork/api/v1/endpoints/publish.py', 'list_publishable_endpoint', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'desktop only (DesktopOnly on the publish router)'),
    Allowed('cowork/api/v1/endpoints/publish.py', 'list_versions_endpoint', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'desktop only (DesktopOnly on the publish router)'),
    Allowed('cowork/api/v1/endpoints/publish.py', 'unpublish_artifact', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'desktop only (DesktopOnly on the publish router)'),
    Allowed('cowork/api/v1/endpoints/settings.py', 'write_raw_settings', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'desktop only (LoopbackDesktopOnly)'),
    Allowed('cowork/services/providers.py', 'warm_enabled_model_map', LOOP_POOL_CHECKOUT, (ANY_LEAF,), 'desktop only: the boot warm and POST /api/v1/settings/raw, which org tenancy skips and refuses'),
    Allowed('cowork/services/memory.py', 'MemoryService.update_memory', LOOP_FILE_STORE, (
        'cowork/harnesses/memory/store.py:MemoryStore._lstat_slot',
        'cowork/harnesses/memory/store.py:MemoryStore._root_fd',
        'cowork/harnesses/memory/store.py:MemoryStore.read_checked',
        'cowork/harnesses/memory/store.py:MemoryStore.restore_exact',
        'cowork/harnesses/memory/store.py:MemoryStore.write',
    ), 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    Allowed('cowork/services/memory.py', 'MemoryService.get_memory', LOOP_FILE_STORE, (
        'cowork/harnesses/memory/store.py:MemoryStore._lstat_slot',
        'cowork/harnesses/memory/store.py:MemoryStore._root_fd',
        'cowork/harnesses/memory/store.py:MemoryStore.read_checked',
    ), 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    Allowed('cowork/services/memory.py', 'MemoryService.list_memory', LOOP_FILE_STORE, (
        'cowork/harnesses/memory/store.py:MemoryStore._lstat_slot',
        'cowork/harnesses/memory/store.py:MemoryStore._root_fd',
        'cowork/harnesses/memory/store.py:MemoryStore.read_checked',
    ), 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    Allowed('cowork/services/memory.py', 'MemoryService.delete_memory', LOOP_FILE_STORE, (
        'cowork/harnesses/memory/store.py:MemoryStore._lstat_slot',
        'cowork/harnesses/memory/store.py:MemoryStore._root_fd',
        'cowork/harnesses/memory/store.py:MemoryStore.delete',
        'cowork/harnesses/memory/store.py:MemoryStore.read_checked',
    ), 'MemoryService runs its sync body on the loop: the project row and the memory files'),
    Allowed('cowork/api/v1/endpoints/project_files.py', 'upload_project_files', LOOP_FILE_STORE, (
        'cowork/api/v1/endpoints/project_files.py:_create_project_temp',
        'cowork/api/v1/endpoints/project_files.py:_opened_project_directory_inventory',
        'cowork/api/v1/endpoints/project_files.py:_write_all',
        'cowork/api/v1/endpoints/project_files.py:_write_bytes_at_project_root',
        'cowork/services/projects.py:ProjectService.ensure_dir_exists',
    ), 'opens every project folder for its pinned-directory inventory, and writes the uploads, on the loop'),
    Allowed('cowork/services/artifacts.py', 'mount_preview', LOOP_FILE_STORE, (ANY_LEAF,), 'desktop only (DesktopOnly on POST /api/v1/artifacts/preview-mount)'),
    Allowed('cowork/api/v1/endpoints/artifact_workspace.py', '_draft_stream.<locals>.chunks', THREAD_FILE_STORE, ('cowork/common/paths.py:read_bounded',), "streams the draft file's bytes from a sync generator's body, which Starlette's StreamingResponse reads on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/files.py', 'retrieve_file_content', THREAD_FILE_STORE, (
        'cowork/services/files.py:FileService.get_file_content',
        'starlette/responses.py:FileResponse.__call__',
        'starlette/responses.py:FileResponse._handle_simple',
    ), "sync route: reads the stored file on anyio's default threads, where the FileResponse it returns stats and opens the file too"),
    Allowed('cowork/api/v1/endpoints/memory.py', 'list_memory', THREAD_FILE_STORE, (
        'cowork/harnesses/memory/store.py:MemoryStore._lstat_slot',
        'cowork/harnesses/memory/store.py:MemoryStore._root_fd',
        'cowork/harnesses/memory/store.py:MemoryStore.read_checked',
    ), "sync route: reads the memory files on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/memory.py', 'update_memory', THREAD_FILE_STORE, (
        'cowork/harnesses/memory/store.py:MemoryStore._lstat_slot',
        'cowork/harnesses/memory/store.py:MemoryStore._root_fd',
        'cowork/harnesses/memory/store.py:MemoryStore.read_checked',
        'cowork/harnesses/memory/store.py:MemoryStore.write',
    ), "sync route: writes the memory files on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/project_files.py', 'download_project_file', THREAD_FILE_STORE, (
        'cowork/api/v1/endpoints/project_files.py:_pinned_fd',
        'cowork/api/v1/endpoints/project_files.py:_pinned_regular_file',
    ), "sync route: opens the project file it streams on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/project_files.py', 'preview_asset', THREAD_FILE_STORE, (
        'cowork/api/v1/endpoints/project_files.py:_pinned_fd',
        'cowork/api/v1/endpoints/project_files.py:_pinned_regular_file',
    ), "sync route: opens the project file it streams on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/project_files.py', 'write_project_file', THREAD_FILE_STORE, (
        'cowork/api/v1/endpoints/project_files.py:_create_project_temp',
        'cowork/api/v1/endpoints/project_files.py:_opened_pinned_descendant',
        'cowork/api/v1/endpoints/project_files.py:_opened_project_directory_inventory',
        'cowork/api/v1/endpoints/project_files.py:_pinned_fd',
        'cowork/api/v1/endpoints/project_files.py:_pinned_regular_file',
        'cowork/api/v1/endpoints/project_files.py:_project_dir',
        'cowork/api/v1/endpoints/project_files.py:_read_project_bytes',
        'cowork/api/v1/endpoints/project_files.py:_resolved_project_parts',
        'cowork/api/v1/endpoints/project_files.py:_safe_relpath',
        'cowork/api/v1/endpoints/project_files.py:_write_all',
        'cowork/api/v1/endpoints/project_files.py:_write_bytes_at_project_root',
        'cowork/services/projects.py:ProjectService.ensure_dir_exists',
    ), "sync route: creates the project folder and writes the file on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/project_files.py', '_pinned_stream.<locals>._chunks', THREAD_FILE_STORE, ('cowork/common/paths.py:read_bounded',), "streams the project file's bytes from a sync generator's body, which Starlette's StreamingResponse reads on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/projects.py', 'create_project', THREAD_FILE_STORE, (
        'cowork/services/projects.py:ProjectService._mkdir_in_root',
        'cowork/services/projects.py:ProjectService._project_path',
        'cowork/services/projects.py:ProjectService._root_fd',
    ), "sync route: creates the project folder on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/projects.py', 'list_projects', THREAD_FILE_STORE, (
        'cowork/services/projects.py:ProjectService._in_scoped_root',
        'cowork/services/projects.py:ProjectService._mkdir_in_root',
        'cowork/services/projects.py:ProjectService._project_path',
        'cowork/services/projects.py:ProjectService._root_fd',
        'cowork/services/projects.py:ProjectService.ensure_dir_exists',
    ), "sync route: creates the general project's folder when it is missing, on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/projects.py', 'update_project', THREAD_FILE_STORE, (
        'cowork/api/v1/endpoints/projects.py:_guard_project_instructions_for_move',
        'cowork/harnesses/memory/store.py:MemoryStore._root_fd',
        'cowork/services/projects.py:ProjectService._project_path',
        'cowork/services/projects.py:ProjectService._rename_in_root',
        'cowork/services/projects.py:ProjectService._stage_project_rename',
        'cowork/services/projects.py:ProjectService._trusted_project_root',
        'cowork/services/skills.py:SkillService.list_skills',
    ), "sync route: checks the project's instructions before a move, on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/skills.py', 'create_skill', THREAD_FILE_STORE, (
        'cowork/api/v1/endpoints/skills.py:_reserve_free_slug',
        'cowork/services/skills.py:SkillService._ensure_root',
        'cowork/services/skills.py:SkillService._replace_direct_child',
        'cowork/services/skills.py:SkillService._skill_dir',
        'cowork/services/skills.py:SkillService._write',
        'cowork/services/skills.py:SkillService.create_skill',
        'cowork/services/skills.py:SkillService.get_skill',
        'cowork/services/skills.py:_skill_from_dir',
    ), "sync route: resolves the skill folder on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/skills.py', 'update_skill', THREAD_FILE_STORE, (
        'cowork/api/v1/endpoints/skills.py:update_skill',
        'cowork/services/skills.py:SkillService._ensure_root',
        'cowork/services/skills.py:SkillService._replace_direct_child',
        'cowork/services/skills.py:SkillService._skill_dir',
        'cowork/services/skills.py:SkillService._write',
        'cowork/services/skills.py:SkillService.get_skill',
        'cowork/services/skills.py:SkillService.update_skill',
        'cowork/services/skills.py:_skill_from_dir',
    ), "sync route: reads the skill folder on anyio's default threads"),
    Allowed('cowork/api/v1/endpoints/skills.py', '_upload_skill_in_new_session', THREAD_FILE_STORE, (
        'cowork/services/skills.py:SkillService._rename_direct_child',
        'cowork/services/skills.py:SkillService._replace_direct_child',
        'cowork/services/skills.py:SkillService._rmtree_direct_child',
        'cowork/services/skills.py:SkillService._write',
        'cowork/services/skills.py:SkillService.import_skill',
    ), "a skill upload's file work, handed to anyio's default threads"),
    Allowed('cowork/channels/runtime.py', 'AntonChannelRuntime._resolve_or_create_binding', THREAD_FILE_STORE, (
        'cowork/services/projects.py:ProjectService._in_scoped_root',
        'cowork/services/projects.py:ProjectService._mkdir_in_root',
        'cowork/services/projects.py:ProjectService._project_path',
        'cowork/services/projects.py:ProjectService._root_fd',
        'cowork/services/projects.py:ProjectService.ensure_dir_exists',
    ), "a run_db unit that creates the default project's folder for a new binding"),
    Allowed('cowork/handlers/responses.py', 'ResponsesHandler._read_turn_start', THREAD_FILE_STORE, (
        'cowork/services/projects.py:ProjectService._in_scoped_root',
        'cowork/services/projects.py:ProjectService.ensure_dir_exists',
    ), "a run_db unit that also reads the turn's attachments"),
    Allowed('cowork/services/connectors/vault_secrets.py', '_restore_from_vault', THREAD_FILE_STORE, ('cowork/services/connectors/vault_secrets.py:_restore_from_vault',), "reads the organization's connection vault through asyncio.to_thread"),
    Allowed('cowork/services/publish.py', 'publish_artifact', THREAD_FILE_STORE, (ANY_LEAF,), "the agent's publish tool reads the artifact in its tool thread"),
)


def unlisted(found: Iterable[Site]) -> list[Site]:
    """The sites in ``found`` that no ALLOWLIST entry lists."""
    return sorted(site for site in found if not any(entry.lists(site) for entry in ALLOWLIST))


def unreached(reached: Iterable[Site]) -> list[Allowed]:
    """The ALLOWLIST entries with a leaf no site in ``reached`` has, each with
    only those leaves. An ANY_LEAF entry needs one site of its function."""
    reached = tuple(reached)
    stale = []
    for entry in ALLOWLIST:
        found = {site.leaf for site in reached if entry.lists(site)}
        if entry.leaves == (ANY_LEAF,):
            missing = () if found else entry.leaves
        else:
            missing = tuple(leaf for leaf in entry.leaves if leaf not in found)
        if missing:
            stale.append(entry._replace(leaves=missing))
    return stale


def listed_twice() -> list[Allowed]:
    """Entries that list a site another entry or leaf lists too: a second
    entry for one function and kind, a leaf named twice, or a leaf beside
    ANY_LEAF."""
    keys = [(entry.path, entry.function, entry.kind) for entry in ALLOWLIST]
    return [
        entry
        for entry, key in zip(ALLOWLIST, keys)
        if keys.count(key) > 1
        or len(set(entry.leaves)) < len(entry.leaves)
        or (ANY_LEAF in entry.leaves and len(entry.leaves) > 1)
    ]


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


class _HandedOver(NamedTuple):
    """Who handed the running thread its call: the site's function and leaf."""

    function: CodeType
    leaf: CodeType


#: Set by anyio.to_thread.run_sync and asyncio.to_thread, both of which copy
#: the caller's context into the thread.
_handed_over: ContextVar[_HandedOver | None] = ContextVar("blocking_guard_handed_over", default=None)


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


def _names_the_file_store(target: object) -> bool:
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


def _reaches_the_file_store(*, parameters: tuple[str | None, ...], args: tuple, kwargs: dict) -> bool:
    if any(kwargs.get(name) is not None for name in _DIR_FD_KEYWORDS):
        return True
    for position, name in enumerate(parameters):
        if name is None:
            continue
        target = args[position] if position < len(args) else kwargs.get(name)
        if target is not None and _names_the_file_store(target):
            return True
    return False


def _on_a_loop() -> bool:
    return asyncio.events._get_running_loop() is not None


def _on_a_worker() -> bool:
    thread = threading.current_thread()
    return type(thread).__name__ == "WorkerThread" or thread.name.startswith("asyncio_")


def _in_org_tenancy() -> bool:
    """Read from the environment, as tests set it, or from the settings
    get_app_settings() already holds. The guard never fills that cache: a
    call here before a test sets the deployment it builds would keep the
    wrong one."""
    if os.environ.get("COWORK_TENANCY_MODE") == "org":
        return True
    return bool(get_app_settings.cache_info().currsize) and get_app_settings().tenancy_mode == "org"


def _on_efs_thread() -> bool:
    marked = getattr(units, "on_efs_thread", None)
    return bool(marked and marked())


def _path_of(code: CodeType) -> str:
    """The code's file under site-packages or the standard library for a
    library's, else under the repo."""
    filename = code.co_filename
    base = next((root for root in _LIBRARY_DIRS if filename.startswith(root)), _REPO_DIR)
    return os.path.relpath(filename, base).replace(os.sep, "/")


class _Leaf:
    """The cowork function nearest a call past the file helpers, and the
    outermost file helper below it, as a walk out from the call finds them."""

    def __init__(self) -> None:
        self.nearest: CodeType | None = None
        self.helper: CodeType | None = None

    def passing(self, code: CodeType) -> None:
        """Note a cowork frame the walk passes."""
        if self.nearest is not None:
            return
        if code.co_filename == _FILE_HELPERS:
            self.helper = code
        else:
            self.nearest = code

    def of(self, function: CodeType) -> CodeType:
        """The leaf of a site whose own function is ``function``: the file
        helper it called itself, rather than the function again."""
        if self.nearest is None or self.nearest is function:
            return self.helper or function
        return self.nearest


def _site(function: CodeType, *, leaf: CodeType, kind: str) -> Site:
    return Site(_path_of(function), function.co_qualname, kind, f"{_path_of(leaf)}:{leaf.co_qualname}")


def _is_a_tests(code: CodeType) -> bool:
    return code.co_filename.startswith(_TESTS_DIR)


def _loop_site(frame: FrameType | None, *, roots: tuple[str, ...], kind: str) -> Site | None:
    leaf = _Leaf()
    called = None  # the outermost cowork function since the last coroutine
    while frame is not None and frame.f_code is not _CALLBACK_RUNNER:
        code = frame.f_code
        if code.co_filename != _UNITS_FILE:
            ours = code.co_filename.startswith(roots)
            if ours:
                leaf.passing(code)
            if code.co_flags & _ASYNC_CODE:
                if ours:
                    return _site(code, leaf=leaf.of(code), kind=kind)
                if _is_a_tests(code):
                    return None
                if called is not None:
                    return _site(called, leaf=leaf.of(called), kind=kind)
            elif ours:
                if called is None or code.co_name != _LAMBDA:
                    called = code
            elif _is_a_tests(code):
                called = None
        frame = frame.f_back
    return _site(called, leaf=leaf.of(called), kind=kind) if called is not None else None


def _thread_site(frame: FrameType | None, *, roots: tuple[str, ...], kind: str) -> Site | None:
    leaf = _Leaf()
    called = None
    tests_own = False
    while frame is not None:
        code = frame.f_code
        if code.co_filename != _UNITS_FILE:
            if code.co_filename.startswith(roots):
                leaf.passing(code)
                if called is None or code.co_name != _LAMBDA:
                    called = code
            elif _is_a_tests(code):
                called = None
                tests_own = True
        frame = frame.f_back
    if called is not None:
        return _site(called, leaf=leaf.of(called), kind=kind)
    if tests_own:
        return None
    handed_over = _handed_over.get()
    if handed_over is None:
        return None
    filename = handed_over.function.co_filename
    if filename.startswith(roots) or not filename.startswith((_COWORK_DIR, _TESTS_DIR)):
        return _site(handed_over.function, leaf=handed_over.leaf, kind=kind)
    return None


def _record(frame: FrameType, *, file_store: bool) -> None:
    if file_store and not _in_org_tenancy():
        return
    for watch in tuple(_watches):
        if _on_a_loop():
            kind = LOOP_FILE_STORE if file_store else LOOP_POOL_CHECKOUT
            site = _loop_site(frame, roots=watch.roots, kind=kind)
        elif file_store and _on_a_worker() and not _on_efs_thread():
            site = _thread_site(frame, roots=watch.roots, kind=THREAD_FILE_STORE)
        else:
            site = None
        if site is not None:
            watch.add(site)


def _guarded(real, *, parameters: tuple[str | None, ...]):
    def call(*args, **kwargs):
        if _watches and not getattr(_state, "busy", False):
            _state.busy = True
            try:
                if _reaches_the_file_store(parameters=parameters, args=args, kwargs=kwargs):
                    _record(sys._getframe(1), file_store=True)
            finally:
                _state.busy = False
        return real(*args, **kwargs)

    call.__wrapped__ = real
    return call


def _endpoint_of_request(frame: FrameType | None, *, leaf: CodeType) -> _HandedOver | None:
    """The route whose request a library's coroutine is answering, from the
    ASGI scope Starlette's router puts it in."""
    while frame is not None:
        scope = frame.f_locals.get("scope")
        endpoint = scope.get("endpoint") if isinstance(scope, dict) else None
        code = getattr(inspect.unwrap(endpoint), "__code__", None) if endpoint is not None else None
        if code is not None:
            return _HandedOver(code, leaf)
        frame = frame.f_back
    return None


def _awaiting_coroutine(frame: FrameType | None, *, leaf: CodeType) -> _HandedOver | None:
    while frame is not None:
        code = frame.f_code
        if code.co_flags & _ASYNC_CODE and code.co_filename.startswith((_COWORK_DIR, _TESTS_DIR)):
            return _HandedOver(code, leaf)
        frame = frame.f_back
    return None


def _handing_coroutine(frame: FrameType | None) -> _HandedOver | None:
    """The first coroutine past the thread helpers. A library's is named by
    the route whose response it sends, as Starlette sends a FileResponse
    after its route returned, or else by the cowork or test coroutine that
    awaits it, with the library's code as the leaf."""
    while frame is not None:
        code = frame.f_code
        if code.co_flags & _ASYNC_CODE and not code.co_filename.startswith(_THREAD_HELPERS):
            if code.co_filename.startswith((_COWORK_DIR, _TESTS_DIR)):
                return _HandedOver(code, code)
            return (
                _endpoint_of_request(frame, leaf=code)
                or _awaiting_coroutine(frame.f_back, leaf=code)
                or _HandedOver(code, code)
            )
        frame = frame.f_back
    return None


def _handing_over(real):
    """Wrap a coroutine that runs a call in a thread, so the thread knows who
    handed it the call."""

    async def hand_over(*args, **kwargs):
        if not _watches:
            return await real(*args, **kwargs)
        token = _handed_over.set(_handing_coroutine(sys._getframe(1)))
        try:
            return await real(*args, **kwargs)
        finally:
            _handed_over.reset(token)

    hand_over.__wrapped__ = real
    return hand_over


def _on_checkout(_dbapi_connection, _record_, _proxy) -> None:
    if not getattr(_state, "busy", False):
        _state.busy = True
        try:
            _record(sys._getframe(1), file_store=False)
        finally:
            _state.busy = False


_SUPPORTS = (os.supports_fd, os.supports_dir_fd, os.supports_follow_symlinks, os.supports_effective_ids)
_originals: dict[tuple[object, str], object] = {}


def _replace(owner: object, *, name: str, wrapper) -> None:
    real = getattr(owner, name)
    _originals[(owner, name)] = real
    setattr(owner, name, wrapper)
    # Code that asks os which calls take a descriptor, a dir_fd or
    # follow_symlinks, as shutil and pytest's tmp_path do, gets the same answer.
    for supported in _SUPPORTS:
        if real in supported:
            supported.add(wrapper)


def install() -> None:
    """Wrap the calls and listen for checkouts. Once per process."""
    if _originals:
        return
    for name, parameters in _PATH_PARAMETERS.items():
        # Windows has no os.statvfs, and its desktop tests run under this guard too.
        if hasattr(os, name):
            _replace(os, name=name, wrapper=_guarded(getattr(os, name), parameters=parameters))
    guarded_open = _guarded(io.open, parameters=_OPEN_PARAMETERS)
    _replace(io, name="open", wrapper=guarded_open)
    _replace(builtins, name="open", wrapper=guarded_open)
    _replace(anyio.to_thread, name="run_sync", wrapper=_handing_over(anyio.to_thread.run_sync))
    _replace(asyncio, name="to_thread", wrapper=_handing_over(asyncio.to_thread))
    event.listen(Pool, "checkout", _on_checkout)


@contextmanager
def watching(*, roots: tuple[str, ...] = (_COWORK_DIR,)) -> Iterator[set[Site]]:
    """The sites found while the block runs, in code under ``roots``, and a
    library's that hands a thread a call no cowork or test code awaits."""
    install()
    watch = _Watch(roots)
    _watches.append(watch)
    try:
        yield watch.sites
    finally:
        _watches.remove(watch)


def site_line(site: Site) -> str:
    return f"  ({site.path!r}, {site.function!r}, {site.kind!r}, {site.leaf!r})"


class StaleEntryCheck:
    """Fails a full unit run that leaves a listed leaf unreached.

    A run is full when it collected every test file under ``tests_dir``
    outside integration/, deselected and filtered nothing, ran every test
    body, and was not stopped early: only then is a leaf no test reached
    stale rather than unreached. A test file that skips itself at import
    counts as collected. A run that is not full says so in one line and why,
    except a run that only lists, --collect-only, --fixtures or
    --fixtures-per-test, whose output tools read as a list.
    """

    def __init__(self, *, tests_dir: Path, reached: set[Site]) -> None:
        self.tests_dir = tests_dir
        self.reached = reached
        self.collected: set[Path] = set()
        self.deselected = False
        self.skipped_because: str | None = None
        self.stale: list[Allowed] = []

    def pytest_collectstart(self, collector: pytest.Collector) -> None:
        if isinstance(collector, pytest.Module):
            self.collected.add(collector.path)

    def pytest_deselected(self) -> None:
        self.deselected = True

    def not_a_full_run(self, session: pytest.Session, *, exitstatus: int) -> str | None:
        """Why this run can't tell a stale entry from an unreached one, or
        None when it can."""
        config = session.config
        option = config.option
        if exitstatus not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
            return f"the run ended with exit status {int(exitstatus)}"
        if option.setuponly:
            return "--setup-only runs no test bodies"
        if option.setupplan:
            return "--setup-plan runs no test bodies"
        if hasattr(config, "workerinput") or config.pluginmanager.hasplugin("dsession"):
            return "pytest-xdist splits the tests between workers"
        if self.deselected or option.keyword or option.markexpr:
            return "tests were deselected"
        if session.shouldstop or session.shouldfail:
            return "the run stopped early"
        every_file = {
            path
            for path in self.tests_dir.rglob("test_*.py")
            if "integration" not in path.relative_to(self.tests_dir).parts
        }
        collected = every_file & self.collected
        if collected != every_file:
            return f"it collected {len(collected)} of {len(every_file)} test files"
        return None

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        option = session.config.option
        if option.collectonly or option.showfixtures or option.show_fixtures_per_test:
            return
        self.skipped_because = self.not_a_full_run(session, exitstatus=exitstatus)
        if self.skipped_because is None:
            self.stale = unreached(self.reached)
        if self.stale:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter) -> None:
        if self.skipped_because is not None:
            terminalreporter.write_line(
                f"The blocking guard's stale-entry check skipped: {self.skipped_because} (tests/_blocking_guard.py)."
            )
        if self.stale:
            terminalreporter.write_line("tests/_blocking_guard.py's ALLOWLIST names sites no test reached:", red=True)
            for entry in self.stale:
                for leaf in entry.leaves:
                    terminalreporter.write_line(site_line(Site(entry.path, entry.function, entry.kind, leaf)), red=True)
