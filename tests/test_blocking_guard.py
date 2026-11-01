"""The blocking guard (tests/_blocking_guard.py) finds each kind of site.

Each route here is defined in this module, and the guard is told to treat
this module as cowork code, so a new route of the same shape anywhere in
cowork/ is found the same way. File store calls count in org tenancy, where
COWORK_HOME is the shared file store.
"""
from __future__ import annotations

import asyncio
import functools
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sqlalchemy import text

import _blocking_guard as guard
from cowork.common.paths import cowork_home, safe_join
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.session import get_open_session
from cowork.db.units import run_efs

_THIS_FILE = os.path.abspath(__file__)
_THIS_PATH = "tests/test_blocking_guard.py"
_ROOT = Path(__file__).resolve().parents[1]


def _leaf(qualname: str) -> str:
    return f"{_THIS_PATH}:{qualname}"


def _note() -> Path:
    return cowork_home() / "blocking-guard" / "note.txt"


@pytest.fixture
def note() -> Path:
    """A file under COWORK_HOME, written before any watch opens."""
    path = _note()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("hello")
    return path


def _stat_it(path: Path) -> bool:
    return path.exists()


def _list_it_by_descriptor(path: Path) -> list[str]:
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        with os.scandir(fd) as entries:
            return [entry.name for entry in entries]
    finally:
        os.close(fd)


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/stat-on-the-loop")
    async def stat_on_the_loop():
        return {"exists": _stat_it(_note())}

    @app.get("/scandir-a-descriptor-on-the-loop")
    async def scandir_a_descriptor_on_the_loop():
        return {"names": _list_it_by_descriptor(_note())}

    @app.get("/read-in-a-sync-route")
    def read_in_a_sync_route():
        return {"text": _note().read_text()}

    @app.get("/stat-through-run-efs")
    async def stat_through_run_efs():
        return {"exists": await run_efs(lambda: _stat_it(_note()), scope=LOCAL_SCOPE)}

    @app.get("/query-on-the-loop")
    async def query_on_the_loop():
        session = get_open_session()
        try:
            return {"one": session.execute(text("select 1")).scalar()}
        finally:
            session.close()

    @app.get("/file-response")
    async def file_response():
        return FileResponse(_note())

    @app.get("/stat-handed-to-anyio")
    async def stat_handed_to_anyio():
        return {"size": (await anyio.to_thread.run_sync(os.stat, _note())).st_size}

    @app.get("/stat-handed-to-asyncio")
    async def stat_handed_to_asyncio():
        return {"size": (await asyncio.to_thread(os.stat, _note())).st_size}

    @app.get("/stat-a-library-hands-to-a-thread")
    async def stat_a_library_hands_to_a_thread():
        return {"size": (await _LIBRARY["stat_on_a_thread"](_note())).st_size}

    return app


async def _sites_of(path: str) -> set[guard.Site]:
    transport = httpx.ASGITransport(app=_app())
    with guard.watching(roots=(_THIS_FILE,)) as found:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(path)
    assert response.status_code == 200, response.text
    return set(found)


async def test_a_coroutine_route_that_stats_under_cowork_home_is_found(org_deployment, note):
    assert await _sites_of("/stat-on-the-loop") == {
        guard.Site(_THIS_PATH, "_app.<locals>.stat_on_the_loop", guard.LOOP_FILE_STORE, _leaf("_stat_it"))
    }


async def test_a_coroutine_route_that_lists_a_directory_descriptor_is_found(org_deployment, note):
    assert await _sites_of("/scandir-a-descriptor-on-the-loop") == {
        guard.Site(
            _THIS_PATH, "_app.<locals>.scandir_a_descriptor_on_the_loop", guard.LOOP_FILE_STORE, _leaf("_list_it_by_descriptor")
        )
    }


async def test_a_sync_route_that_reads_under_cowork_home_is_found(org_deployment, note):
    route = "_app.<locals>.read_in_a_sync_route"
    assert await _sites_of("/read-in-a-sync-route") == {
        guard.Site(_THIS_PATH, route, guard.THREAD_FILE_STORE, _leaf(route))
    }


async def test_work_run_through_run_efs_is_not_found(org_deployment, note):
    assert await _sites_of("/stat-through-run-efs") == set()


async def test_a_pool_checkout_in_a_coroutine_route_is_found():
    route = "_app.<locals>.query_on_the_loop"
    assert await _sites_of("/query-on-the-loop") == {
        guard.Site(_THIS_PATH, route, guard.LOOP_POOL_CHECKOUT, _leaf(route))
    }


async def test_a_tests_own_file_work_is_not_found(org_deployment, note):
    """This coroutine is a test's, not cowork's, so nothing is blamed."""
    with guard.watching() as found:
        _stat_it(note)
        await anyio.to_thread.run_sync(os.stat, note)
    assert found == set()


async def test_file_store_calls_on_a_desktop_are_not_found(note):
    """On a desktop COWORK_HOME is a local disk."""
    assert await _sites_of("/stat-on-the-loop") == set()


async def test_a_unit_a_test_runs_inline_is_not_blamed_on_its_caller(org_deployment, note):
    """A test stub that runs work on the loop is the test's, not the route's."""

    async def inline_stub():
        _stat_it(note)

    async def route_like():
        await inline_stub()

    with guard.watching(roots=(_THIS_FILE,)) as found:
        await route_like()
    # inline_stub is in this module too, so it is blamed, not route_like.
    stub = "test_a_unit_a_test_runs_inline_is_not_blamed_on_its_caller.<locals>.inline_stub"
    assert found == {guard.Site(_THIS_PATH, stub, guard.LOOP_FILE_STORE, _leaf("_stat_it"))}

    with guard.watching() as found:
        await route_like()
    assert found == set()


# ── Entry and leaf ──


def _listed_leaf(path: Path) -> None:
    os.stat(path)


def _new_leaf(path: Path) -> None:
    list(os.scandir(path.parent))


def _entry_unit(*, leaf: Callable[[Path], None], path: Path) -> None:
    leaf(path)


def _other_entry_unit(*, leaf: Callable[[Path], None], path: Path) -> None:
    leaf(path)


async def _sites_on_a_worker(work: Callable[[], None], *, roots: tuple[str, ...] = (_THIS_FILE,)) -> set[guard.Site]:
    with guard.watching(roots=roots) as found:
        await anyio.to_thread.run_sync(work)
    return set(found)


async def test_new_work_below_a_listed_entry_is_a_new_site(org_deployment, note, monkeypatch):
    monkeypatch.setattr(
        guard,
        "ALLOWLIST",
        (guard.Allowed(_THIS_PATH, "_entry_unit", guard.THREAD_FILE_STORE, (_leaf("_listed_leaf"),), "listed here"),),
    )
    listed = await _sites_on_a_worker(lambda: _entry_unit(leaf=_listed_leaf, path=note))
    new = await _sites_on_a_worker(lambda: _entry_unit(leaf=_new_leaf, path=note))

    assert listed == {guard.Site(_THIS_PATH, "_entry_unit", guard.THREAD_FILE_STORE, _leaf("_listed_leaf"))}
    assert guard.unlisted(listed) == []
    assert new == {guard.Site(_THIS_PATH, "_entry_unit", guard.THREAD_FILE_STORE, _leaf("_new_leaf"))}
    assert guard.unlisted(new) == sorted(new)


async def test_a_new_entry_that_calls_a_listed_helper_is_a_new_site(org_deployment, note):
    listed = await _sites_on_a_worker(lambda: _entry_unit(leaf=_listed_leaf, path=note))
    other = await _sites_on_a_worker(lambda: _other_entry_unit(leaf=_listed_leaf, path=note))

    assert other == {guard.Site(_THIS_PATH, "_other_entry_unit", guard.THREAD_FILE_STORE, _leaf("_listed_leaf"))}
    assert other != listed


async def test_a_unit_handed_over_as_a_lambda_is_named_by_what_it_calls(org_deployment, note):
    """Every lambda in a function has one name, so the entry is the function
    the lambda calls. A lambda that does the work itself is still a site."""
    assert await _sites_on_a_worker(lambda: _entry_unit(leaf=_listed_leaf, path=note)) == {
        guard.Site(_THIS_PATH, "_entry_unit", guard.THREAD_FILE_STORE, _leaf("_listed_leaf"))
    }
    itself = "test_a_unit_handed_over_as_a_lambda_is_named_by_what_it_calls.<locals>.<lambda>"
    assert await _sites_on_a_worker(lambda: os.stat(note)) == {
        guard.Site(_THIS_PATH, itself, guard.THREAD_FILE_STORE, _leaf(itself))
    }


def _entry_unit_site(leaf: str, *, kind: str = guard.THREAD_FILE_STORE) -> guard.Site:
    return guard.Site(_THIS_PATH, "_entry_unit", kind, _leaf(leaf))


def _joins_a_name(path: Path) -> None:
    safe_join(path.parent, path.name)


def _joins_another_name(path: Path) -> None:
    safe_join(path.parent, path.name)


def _entry_that_joins(*, path: Path) -> None:
    safe_join(path.parent, path.name)


async def test_work_through_a_shared_file_helper_is_named_by_the_helpers_caller(org_deployment, note):
    """Most file work goes through the helpers in cowork/common/paths.py, so
    the leaf is the function that called the helper, and new work below a
    listed function through a listed helper is a new site. When the site's
    own function called the helper, the leaf is the helper."""
    roots = (_THIS_FILE, guard._COWORK_DIR)
    one = await _sites_on_a_worker(lambda: _entry_unit(leaf=_joins_a_name, path=note), roots=roots)
    another = await _sites_on_a_worker(lambda: _entry_unit(leaf=_joins_another_name, path=note), roots=roots)
    itself = await _sites_on_a_worker(lambda: _entry_that_joins(path=note), roots=roots)

    assert one == {_entry_unit_site("_joins_a_name")}
    assert another == {_entry_unit_site("_joins_another_name")}
    assert itself == {
        guard.Site(_THIS_PATH, "_entry_that_joins", guard.THREAD_FILE_STORE, "cowork/common/paths.py:safe_join")
    }


def test_an_entry_is_stale_by_each_leaf_no_test_reached(monkeypatch):
    entry = guard.Allowed(
        _THIS_PATH, "_entry_unit", guard.THREAD_FILE_STORE, (_leaf("_listed_leaf"), _leaf("_new_leaf")), "two calls"
    )
    monkeypatch.setattr(guard, "ALLOWLIST", (entry,))

    assert guard.unreached({_entry_unit_site("_listed_leaf"), _entry_unit_site("_new_leaf")}) == []
    assert guard.unreached({_entry_unit_site("_listed_leaf")}) == [entry._replace(leaves=(_leaf("_new_leaf"),))]


def test_an_any_leaf_entry_lists_every_leaf_of_its_function(monkeypatch):
    whole = guard.Allowed(_THIS_PATH, "_entry_unit", guard.THREAD_FILE_STORE, (guard.ANY_LEAF,), "its whole body")
    monkeypatch.setattr(guard, "ALLOWLIST", (whole,))
    sites = {_entry_unit_site("_listed_leaf"), _entry_unit_site("_new_leaf")}
    other_kind = _entry_unit_site("_new_leaf", kind=guard.LOOP_FILE_STORE)

    assert guard.unlisted(sites) == []
    assert guard.unlisted({other_kind}) == [other_kind]
    assert guard.unreached(sites) == []
    assert guard.unreached({other_kind}) == [whole]


def test_a_function_listed_twice_or_a_leaf_beside_any_leaf_is_refused(monkeypatch):
    one = guard.Allowed(_THIS_PATH, "_entry_unit", guard.THREAD_FILE_STORE, (_leaf("_new_leaf"),), "one call")
    mixed = guard.Allowed(
        _THIS_PATH, "_other_entry_unit", guard.THREAD_FILE_STORE, (guard.ANY_LEAF, _leaf("_new_leaf")), "both"
    )
    monkeypatch.setattr(guard, "ALLOWLIST", (one, one, mixed))

    assert guard.listed_twice() == [one, one, mixed]


def test_every_allowlist_entry_gives_its_reason():
    kinds = {guard.LOOP_FILE_STORE, guard.THREAD_FILE_STORE, guard.LOOP_POOL_CHECKOUT}
    for entry in guard.ALLOWLIST:
        assert entry.path.startswith("cowork/") and entry.function, entry
        assert entry.kind in kinds, entry
        assert entry.leaves and all(leaf == guard.ANY_LEAF or ":" in leaf for leaf in entry.leaves), entry
        assert entry.reason.strip(), entry
    assert guard.listed_twice() == []


# ── Every call that reads or changes the file system ──


def _calls(*, home: Path, outside: Path, fd: int) -> dict[str, Callable[[], object]]:
    """One call per kind: each touches COWORK_HOME through one argument only,
    a path under it or ``fd``, a descriptor of a file there."""
    target = home / "target"
    calls: dict[str, Callable[[], object]] = {
        "unlink": lambda: os.unlink(home / "file"),
        "remove": lambda: os.remove(home / "file"),
        "rename from the file store": lambda: os.rename(home / "file", outside / "moved"),
        "rename into the file store": lambda: os.rename(outside / "file", target),
        "replace from the file store": lambda: os.replace(home / "file", outside / "moved"),
        "replace into the file store": lambda: os.replace(outside / "file", target),
        "link into the file store": lambda: os.link(outside / "file", target),
        "symlink into the file store": lambda: os.symlink(outside / "file", target),
        "mkdir": lambda: os.mkdir(home / "new"),
        "Path.mkdir of a new folder": lambda: (home / "new" / "deeper").mkdir(parents=True),
        "rmdir": lambda: os.rmdir(home / "folder"),
        "access": lambda: os.access(home / "file", os.R_OK),
        "chmod": lambda: os.chmod(home / "file", 0o600),
        "utime": lambda: os.utime(home / "file"),
        "readlink": lambda: os.readlink(home / "link"),
        "truncate": lambda: os.truncate(home / "file", 0),
        "statvfs": lambda: os.statvfs(home),
        "stat by keyword": lambda: os.stat(path=home / "file"),
        "scandir by keyword": lambda: list(os.scandir(path=home)),
        "listdir by keyword": lambda: os.listdir(path=home),
        "open by keyword": lambda: open(file=home / "file").close(),
        "rename by its directory descriptors": lambda: _rename_by_directory_descriptors(outside),
        "read": lambda: os.read(fd, 1),
        "write": lambda: os.write(fd, b"x"),
        "pread": lambda: os.pread(fd, 1, 0),
        "pwrite": lambda: os.pwrite(fd, b"x", 0),
        "fsync": lambda: os.fsync(fd),
        "ftruncate": lambda: os.ftruncate(fd, 0),
        "fchmod": lambda: os.fchmod(fd, 0o600),
        "fchown": lambda: os.fchown(fd, -1, -1),
    }
    # macOS has no os.fdatasync.
    if hasattr(os, "fdatasync"):
        calls["fdatasync"] = lambda: os.fdatasync(fd)
    return calls


def _rename_by_directory_descriptors(outside: Path) -> None:
    fd = os.open(outside, os.O_RDONLY)
    try:
        os.rename("file", "moved", src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        os.close(fd)


async def _blocks_the_loop(call: Callable[[], object]) -> None:
    call()


_CALL_NAMES = list(_calls(home=Path("home"), outside=Path("outside"), fd=-1))


@pytest.mark.parametrize("name", _CALL_NAMES)
async def test_every_call_that_touches_the_file_store_is_found(org_deployment, tmp_path, name):
    home = cowork_home() / "blocking-guard-calls" / tmp_path.name
    outside = tmp_path / "outside"
    for folder in (home / "folder", outside):
        folder.mkdir(parents=True)
    for file in (home / "file", outside / "file"):
        file.write_text("hello")
    (home / "link").symlink_to(home / "file")
    fd = os.open(home / "file", os.O_RDWR)

    try:
        with guard.watching(roots=(_THIS_FILE,)) as found:
            await _blocks_the_loop(_calls(home=home, outside=outside, fd=fd)[name])
    finally:
        os.close(fd)

    assert {(site.function, site.kind) for site in found} == {("_blocks_the_loop", guard.LOOP_FILE_STORE)}


def test_the_wrapped_calls_keep_their_place_in_os_supports_sets(tmp_path):
    """shutil and pytest ask these sets which calls take a descriptor, a
    dir_fd or follow_symlinks; a wrapper missing from them sends shutil down
    a branch that fails."""
    guard.install()
    for name in guard._PATH_PARAMETERS:
        if not hasattr(os, name):  # os.fdatasync on macOS
            continue
        wrapper = getattr(os, name)
        for supported in (os.supports_fd, os.supports_dir_fd, os.supports_follow_symlinks, os.supports_effective_ids):
            assert (wrapper in supported) == (wrapper.__wrapped__ in supported), (name, supported)

    (tmp_path / "file").write_text("hello")
    (tmp_path / "a").symlink_to(tmp_path / "file")
    (tmp_path / "b").symlink_to(tmp_path / "file")
    shutil.copystat(tmp_path / "a", tmp_path / "b", follow_symlinks=False)


def test_the_guard_installs_where_os_lacks_a_wrapped_call(tmp_path):
    """Windows has no os.statvfs, and Windows runs the desktop's tests under
    this guard."""
    script = "import os\ndel os.statvfs\nimport _blocking_guard\n_blocking_guard.install()\nprint(os.stat.__wrapped__)\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": f"{_ROOT / 'tests'}{os.pathsep}{_ROOT}"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "built-in function stat" in result.stdout, result.stdout


# ── Org tenancy a test sets on the settings it holds ──


async def test_org_tenancy_patched_onto_the_held_settings_counts(monkeypatch, note):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    monkeypatch.setattr(get_app_settings(), "tenancy_mode", "org")

    assert await _sites_of("/stat-on-the-loop") == {
        guard.Site(_THIS_PATH, "_app.<locals>.stat_on_the_loop", guard.LOOP_FILE_STORE, _leaf("_stat_it"))
    }


async def test_the_guard_never_fills_the_settings_cache(monkeypatch, note):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()

    with guard.watching(roots=(_THIS_FILE,)) as found:
        await _blocks_the_loop(lambda: _stat_it(note))

    assert found == set()
    assert get_app_settings.cache_info().currsize == 0


# ── Cowork code that a library or the loop runs ──

#: A library's coroutines: code compiled as if installed under site-packages.
_LIBRARY: dict[str, object] = {}
exec(
    compile(
        "async def run_callback(callback, path):\n"
        "    return callback(path)\n"
        "\n"
        "async def stat(path):\n"
        "    import os\n"
        "    return os.stat(path)\n"
        "\n"
        "async def stat_on_a_thread(path):\n"
        "    import os, anyio.to_thread\n"
        "    return await anyio.to_thread.run_sync(os.stat, path)\n",
        os.path.join(sysconfig.get_paths()["purelib"], "blocking_guard_fake_library.py"),
        "exec",
    ),
    _LIBRARY,
)


async def test_cowork_code_a_library_coroutine_calls_is_found(org_deployment, note):
    """As FastAPI's coroutine runs a cowork validator: the site is the cowork
    function it called."""
    with guard.watching(roots=(_THIS_FILE,)) as found:
        await _LIBRARY["run_callback"](_stat_it, note)
    assert found == {guard.Site(_THIS_PATH, "_stat_it", guard.LOOP_FILE_STORE, _leaf("_stat_it"))}


async def _awaits_a_library(path: Path) -> None:
    await _LIBRARY["stat"](path)


async def test_a_library_coroutine_that_blocks_is_blamed_on_the_cowork_coroutine_awaiting_it(org_deployment, note):
    with guard.watching(roots=(_THIS_FILE,)) as found:
        await _awaits_a_library(note)
    assert found == {guard.Site(_THIS_PATH, "_awaits_a_library", guard.LOOP_FILE_STORE, _leaf("_awaits_a_library"))}


def _stat_in_a_callback(*, path: Path, done: asyncio.Event) -> None:
    try:
        _stat_it(path)
    finally:
        done.set()


async def test_cowork_code_in_a_loop_callback_is_found(org_deployment, note):
    done = asyncio.Event()
    with guard.watching(roots=(_THIS_FILE,)) as found:
        asyncio.get_running_loop().call_soon(functools.partial(_stat_in_a_callback, path=note, done=done))
        await done.wait()
    assert found == {guard.Site(_THIS_PATH, "_stat_in_a_callback", guard.LOOP_FILE_STORE, _leaf("_stat_it"))}


def test_cowork_code_in_a_loop_callback_a_sync_test_drives_is_found(org_deployment, note):
    """A sync test that runs the loop with asyncio.run sits below the loop's
    callback runner. The work is the callback's, not the code that drives the
    loop."""

    async def main() -> None:
        done = asyncio.Event()
        asyncio.get_running_loop().call_soon(functools.partial(_stat_in_a_callback, path=note, done=done))
        await done.wait()

    with guard.watching(roots=(_THIS_FILE,)) as found:
        asyncio.run(main())
    assert found == {guard.Site(_THIS_PATH, "_stat_in_a_callback", guard.LOOP_FILE_STORE, _leaf("_stat_it"))}


async def test_a_file_response_is_found_on_the_thread_it_hands_its_file_to(org_deployment, note):
    """Starlette's FileResponse stats and opens its file on anyio's default
    threads after the route returns, so no cowork function is on the thread.
    The site is the route that returned it, once for each FileResponse
    coroutine that hands a call over: the stat and the open."""
    route = "_app.<locals>.file_response"
    assert await _sites_of("/file-response") == {
        guard.Site(_THIS_PATH, route, guard.THREAD_FILE_STORE, "starlette/responses.py:FileResponse.__call__"),
        guard.Site(_THIS_PATH, route, guard.THREAD_FILE_STORE, "starlette/responses.py:FileResponse._handle_simple"),
    }


async def test_a_library_hand_over_in_a_request_is_named_by_the_library_coroutine(org_deployment, note):
    """Each library coroutine that hands a call over is its own leaf, so a
    second one under a listed route is a new site."""
    route = "_app.<locals>.stat_a_library_hands_to_a_thread"
    assert await _sites_of("/stat-a-library-hands-to-a-thread") == {
        guard.Site(_THIS_PATH, route, guard.THREAD_FILE_STORE, "blocking_guard_fake_library.py:stat_on_a_thread")
    }


async def _awaits_a_library_that_hands_over(path: Path) -> None:
    await _LIBRARY["stat_on_a_thread"](path)


async def test_a_library_hand_over_outside_a_request_is_blamed_on_the_coroutine_awaiting_it(org_deployment, note):
    with guard.watching(roots=(_THIS_FILE,)) as found:
        await _awaits_a_library_that_hands_over(note)
    assert found == {
        guard.Site(
            _THIS_PATH,
            "_awaits_a_library_that_hands_over",
            guard.THREAD_FILE_STORE,
            "blocking_guard_fake_library.py:stat_on_a_thread",
        )
    }


@pytest.mark.parametrize("route", ["stat_handed_to_anyio", "stat_handed_to_asyncio"])
async def test_a_bare_call_a_cowork_coroutine_hands_to_a_thread_is_found(org_deployment, note, route):
    function = f"_app.<locals>.{route}"
    assert await _sites_of(f"/{route.replace('_', '-')}") == {
        guard.Site(_THIS_PATH, function, guard.THREAD_FILE_STORE, _leaf(function))
    }


# ── The stale-entry check ──


def test_the_suites_cowork_home_is_a_resolved_path():
    """macOS links /var to /private/var. Under that link, a check that compares
    a path with its resolved form, as ProjectService._child_name does, touches
    the file store on macOS only, so a leaf listed from a macOS run would be
    stale on a Linux run."""
    home = os.environ["COWORK_HOME"]
    assert os.path.realpath(home) == home


_STALE_CONFTEST = """
import os
from pathlib import Path

import _blocking_guard


def pytest_configure(config):
    if os.environ.get("FAKE_XDIST_WORKER"):
        config.workerinput = {"workerid": "gw0"}


def pytest_sessionstart(session):
    session.config.pluginmanager.register(
        _blocking_guard.StaleEntryCheck(tests_dir=Path(__file__).parent, reached=set()), "stale-entries"
    )
"""


def _run_the_stale_check(tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """A run over two test files, one of which skips itself at import, with a
    check that reached no site, so every ALLOWLIST entry is stale when the
    check runs."""
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "conftest.py").write_text(_STALE_CONFTEST)
    (tests / "test_runs.py").write_text("def test_runs():\n    pass\n")
    (tests / "test_skips_itself.py").write_text(
        "import pytest\n\npytest.skip('needs what this machine lacks', allow_module_level=True)\n"
    )
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--rootdir", str(tmp_path), *args],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": f"{_ROOT / 'tests'}{os.pathsep}{_ROOT}",
            "COWORK_HOME": str(home),
            "DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
            "MASTER_KEY_PATH": str(tmp_path / "master.key"),
            **(env or {}),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("listing", ["--collect-only", "--fixtures", "--fixtures-per-test"])
def test_a_run_that_only_lists_says_nothing_of_the_stale_check(tmp_path, listing):
    result = _run_the_stale_check(tmp_path, listing, "tests")

    assert "stale-entry check skipped" not in result.stdout, result.stdout
    assert "ALLOWLIST names sites no test reached" not in result.stdout, result.stdout
    assert result.returncode == pytest.ExitCode.OK, result.stdout


def test_a_test_file_that_skips_itself_leaves_the_stale_check_on(tmp_path):
    assert guard.ALLOWLIST, "the check needs an entry to report"
    result = _run_the_stale_check(tmp_path, "tests")

    assert "ALLOWLIST names sites no test reached" in result.stdout, result.stdout
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout


@pytest.mark.parametrize(
    ("args", "env", "reason"),
    [
        (["tests/test_runs.py"], {}, "it collected 1 of 2 test files"),
        (["--setup-only", "tests"], {}, "--setup-only runs no test bodies"),
        (["tests"], {"FAKE_XDIST_WORKER": "1"}, "pytest-xdist splits the tests between workers"),
    ],
    ids=["one file", "setup only", "xdist worker"],
)
def test_a_run_that_is_not_full_skips_the_stale_check_in_one_line(tmp_path, args, env, reason):
    result = _run_the_stale_check(tmp_path, *args, env=env)

    assert f"The blocking guard's stale-entry check skipped: {reason} (tests/_blocking_guard.py)." in result.stdout
    assert "ALLOWLIST names sites no test reached" not in result.stdout, result.stdout
    assert result.returncode == pytest.ExitCode.OK, result.stdout
