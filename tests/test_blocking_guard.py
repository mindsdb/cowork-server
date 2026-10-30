"""The blocking guard (tests/_blocking_guard.py) finds each kind of site.

Each route here is defined in this module, and the guard is told to treat
this module as cowork code, so a new route of the same shape anywhere in
cowork/ is found the same way. File store calls count in org tenancy, where
COWORK_HOME is the shared file store.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from sqlalchemy import text

import _blocking_guard as guard
from cowork.common.paths import cowork_home
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.session import get_open_session
from cowork.db.units import run_efs

_THIS_FILE = os.path.abspath(__file__)
_THIS_PATH = "tests/test_blocking_guard.py"


def _a_file_under_cowork_home() -> Path:
    path = cowork_home() / "blocking-guard" / "note.txt"
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
        return {"exists": _stat_it(_a_file_under_cowork_home())}

    @app.get("/scandir-a-descriptor-on-the-loop")
    async def scandir_a_descriptor_on_the_loop():
        return {"names": _list_it_by_descriptor(_a_file_under_cowork_home())}

    @app.get("/read-in-a-sync-route")
    def read_in_a_sync_route():
        return {"text": _a_file_under_cowork_home().read_text()}

    @app.get("/stat-through-run-efs")
    async def stat_through_run_efs():
        path = await run_efs(_a_file_under_cowork_home, scope=LOCAL_SCOPE)
        return {"exists": await run_efs(lambda: _stat_it(path), scope=LOCAL_SCOPE)}

    @app.get("/query-on-the-loop")
    async def query_on_the_loop():
        session = get_open_session()
        try:
            return {"one": session.execute(text("select 1")).scalar()}
        finally:
            session.close()

    return app


async def _sites_of(path: str) -> set[guard.Site]:
    transport = httpx.ASGITransport(app=_app())
    with guard.watching(roots=(_THIS_FILE,)) as found:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(path)
    assert response.status_code == 200, response.text
    return set(found)


async def test_a_coroutine_route_that_stats_under_cowork_home_is_found(org_deployment):
    assert await _sites_of("/stat-on-the-loop") == {
        guard.Site(_THIS_PATH, "_app.<locals>.stat_on_the_loop", guard.LOOP_FILE_STORE)
    }


async def test_a_coroutine_route_that_lists_a_directory_descriptor_is_found(org_deployment):
    assert await _sites_of("/scandir-a-descriptor-on-the-loop") == {
        guard.Site(_THIS_PATH, "_app.<locals>.scandir_a_descriptor_on_the_loop", guard.LOOP_FILE_STORE)
    }


async def test_a_sync_route_that_reads_under_cowork_home_is_found(org_deployment):
    assert await _sites_of("/read-in-a-sync-route") == {
        guard.Site(_THIS_PATH, "_app.<locals>.read_in_a_sync_route", guard.THREAD_FILE_STORE)
    }


async def test_work_run_through_run_efs_is_not_found(org_deployment):
    assert await _sites_of("/stat-through-run-efs") == set()


async def test_a_pool_checkout_in_a_coroutine_route_is_found():
    assert await _sites_of("/query-on-the-loop") == {
        guard.Site(_THIS_PATH, "_app.<locals>.query_on_the_loop", guard.LOOP_POOL_CHECKOUT)
    }


async def test_a_tests_own_file_work_is_not_found(org_deployment):
    """This coroutine is a test's, not cowork's, so nothing is blamed."""
    with guard.watching() as found:
        _stat_it(_a_file_under_cowork_home())
    assert found == set()


async def test_file_store_calls_on_a_desktop_are_not_found():
    """On a desktop COWORK_HOME is a local disk."""
    assert await _sites_of("/stat-on-the-loop") == set()


async def test_a_unit_a_test_runs_inline_is_not_blamed_on_its_caller(org_deployment):
    """A test stub that runs work on the loop is the test's, not the route's."""

    async def inline_stub():
        _stat_it(_a_file_under_cowork_home())

    async def route_like():
        await inline_stub()

    with guard.watching(roots=(_THIS_FILE,)) as found:
        await route_like()
    # inline_stub is in this module too, so it is blamed, not route_like.
    assert found == {guard.Site(_THIS_PATH, "test_a_unit_a_test_runs_inline_is_not_blamed_on_its_caller.<locals>.inline_stub", guard.LOOP_FILE_STORE)}

    with guard.watching() as found:
        await route_like()
    assert found == set()


def test_every_allowlist_entry_gives_its_reason():
    kinds = {guard.LOOP_FILE_STORE, guard.THREAD_FILE_STORE, guard.LOOP_POOL_CHECKOUT}
    for path, function, kind, reason in guard.ALLOWLIST:
        assert path.startswith("cowork/") and function, (path, function)
        assert kind in kinds, (path, function, kind)
        assert reason.strip(), (path, function)
    guard.allowed_sites()
