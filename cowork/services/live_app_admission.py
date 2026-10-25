"""Deployment-wide admission for live app slots, outside ORM transactions.

The lock covers only final slot checks, reset, launch and launch recording.
Approval cards and app probes must run after releasing it. PostgreSQL uses a
separate physical session; SQLite workers coordinate beside their database.
"""
from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from cowork.db import session as db_session
from cowork.db.units import run_to_completion
from cowork.services.shared_resources import _database_lock_engine, advisory_lock_key

LAUNCH_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_RETRY_SECONDS = 0.05
_LOCK_KEY = advisory_lock_key(b"cowork\0live_app_admission_v1")
_MEMORY_LOCKS: dict[Engine, threading.Lock] = {}
_MEMORY_GUARD = threading.Lock()


class _LaunchLock:
    """Resources stay attached before acquisition, so cancellation can close
    a connection or descriptor even if it arrives during a worker call."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.connection: Connection | None = None
        self.fd: int | None = None
        self.memory: threading.Lock | None = None
        self.acquired = False

    def open(self) -> None:
        if self.engine.dialect.name == "postgresql":
            self.connection = _database_lock_engine(self.engine).connect()
            self.connection.execute(sa.text("SET statement_timeout = 5000"))
            return
        if self.engine.dialect.name != "sqlite":
            raise RuntimeError("Live app admission needs PostgreSQL or SQLite.")
        database = self.engine.url.database
        if not database or database == ":memory:" or self.engine.url.query.get("mode") == "memory":
            with _MEMORY_GUARD:
                self.memory = _MEMORY_LOCKS.setdefault(self.engine, threading.Lock())
            return
        # Resolve aliases to the actual DB path, so workers with different
        # working directories or symlinked paths still use the same lock.
        path = Path(database.removeprefix("file:")).resolve()
        self.fd = os.open(
            str(path) + ".live-app-launch.lock",
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )

    def try_acquire(self) -> bool:
        if self.connection is not None:
            self.acquired = bool(self.connection.execute(
                sa.text("SELECT pg_try_advisory_lock(:key)"), {"key": _LOCK_KEY}
            ).scalar())
        elif self.memory is not None:
            self.acquired = self.memory.acquire(blocking=False)
        else:
            import fcntl

            assert self.fd is not None
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            self.acquired = True
        return self.acquired

    def close(self) -> None:
        # NullPool closes the PostgreSQL session physically, releasing every
        # advisory lock even when an explicit unlock statement would fail.
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.memory is not None and self.acquired:
            self.memory.release()
        self.acquired = False


@asynccontextmanager
async def live_app_launch_lock() -> AsyncIterator[None]:
    from anton.core.artifacts.live_app_slots import LiveAppLaunchError

    engine = db_session.get_engine(db_session.settings.database.uri)
    lock = _LaunchLock(engine)
    deadline = asyncio.get_running_loop().time() + LAUNCH_LOCK_TIMEOUT_SECONDS
    try:
        await run_to_completion(asyncio.to_thread(lock.open))
        while True:
            if await run_to_completion(asyncio.to_thread(lock.try_acquire)):
                break
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise LiveAppLaunchError(
                    "Live app slots are busy. Try launching again shortly.", cause="slot_busy"
                )
            await asyncio.sleep(min(_LOCK_RETRY_SECONDS, remaining))
        yield
    finally:
        # A cancelled acquisition can complete in its thread; both it and a
        # cancelled holder must release before the caller unwinds.
        await run_to_completion(asyncio.to_thread(lock.close))
