"""Test bootstrap for the channel tests.

Isolates the app onto a throwaway SQLite DB + master key BEFORE any cowork
module is imported (settings/engine are read at import time), then builds the
schema directly and seeds the ``general`` project the runtime depends on.
"""
import importlib
import io
import logging
import os
import pkgutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TMP = Path(tempfile.mkdtemp(prefix="cowork-chan-test-"))
# Force isolation (assignment, not setdefault, never touch a real DB).
os.environ["DATABASE_URI"] = f"sqlite:///{TMP / 'test.db'}"
os.environ["MASTER_KEY_PATH"] = str(TMP / "master.key")
os.environ["COWORK_PUBLIC_BASE_URL"] = "https://hooks.example.com"
os.environ["COWORK_CONVERSATION_LINK_TEMPLATE"] = "https://app.example.com/c/{conversation_id}"
# Org mode roots every filesystem store at cowork_home() (see
# scoped_storage_root), not just at the per-store COWORK_*_DIR override below,
# so this has to be set too, or any test that runs in org mode writes into
# the developer's real ~/.cowork/<org_id>/ and orphans dirs there.
os.environ["COWORK_HOME"] = str(TMP)
os.environ["COWORK_PROJECTS_DIR"] = str(TMP / "projects")
# File bytes too - without this, any test using FileService writes into the
# developer's real ~/.cowork/files/ and orphans dirs there.
os.environ["COWORK_FILES_DIR"] = str(TMP / "files")
# Without this, org-scoped tests write into the developer's real ~/.cowork/.
os.environ["COWORK_SHARED_DIR"] = str(TMP / "shared")
os.environ["ENV"] = "test"
# require_auth now defaults on in local mode. Every test in this suite drives
# create_app() through TestClient/ASGITransport with no bearer token, so
# leaving the default on would 401 nearly everything. A test that means to
# exercise the auth-on behavior sets this back explicitly.
os.environ["COWORK_REQUIRE_AUTH"] = "false"

import pytest
from sqlmodel import Session, SQLModel


@pytest.fixture(scope="session", autouse=True)
def db_schema():
    # Import every model module so the mapper + metadata are complete.
    import cowork.models as models_pkg
    for _, name, _ in pkgutil.iter_modules(models_pkg.__path__):
        importlib.import_module(f"cowork.models.{name}")

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.session import get_engine
    from cowork.models.project import Project
    from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID

    engine = get_engine(get_app_settings().database.uri)
    SQLModel.metadata.create_all(engine)

    # Under the projects root so the artifacts scanner treats it as registered.
    general_dir = TMP / "projects" / "general"
    general_dir.mkdir(parents=True, exist_ok=True)
    with Session(engine) as session:
        if session.get(Project, GENERAL_PROJECT_ID) is None:
            session.add(Project(id=GENERAL_PROJECT_ID, name=GENERAL_PROJECT, path=str(general_dir)))
            session.commit()
    yield


@pytest.fixture(autouse=True)
def keep_seeded_general_path():
    """Put the seeded ``general`` row's path back after each test.

    The default-project resolver writes on a read path: it re-points this row
    onto whatever ``COWORK_PROJECTS_DIR`` currently names, so a test that moves
    the root and then reaches a route leaves the row inside its own
    ``tmp_path``, which pytest deletes. Later tests that resolve ``general``
    from settings rather than from the row then disagree with it.

    This repairs leakage only, at teardown. The write itself is asserted in
    tests/test_general_project_root_change.py, so hiding it here costs no
    coverage.
    """
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.session import get_engine
    from cowork.models.project import Project
    from cowork.services.projects import GENERAL_PROJECT_ID

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as read:
        seeded = read.get(Project, GENERAL_PROJECT_ID)
        original = seeded.path if seeded is not None else None
    yield
    if original is None:
        return
    with Session(engine) as write:
        row = write.get(Project, GENERAL_PROJECT_ID)
        if row is not None and row.path != original:
            row.path = original
            write.add(row)
            write.commit()


@pytest.fixture(autouse=True)
def close_coding_services():
    yield
    from coding_service_fakes import close_services

    close_services()


@pytest.fixture(scope="session", autouse=True)
def trust_test_client_host():
    """Starlette's TestClient sends ``Host: testserver``; only tests trust it."""
    from cowork.api.v1.endpoints import guards

    production = guards._TRUSTED_LOOPBACK_HOSTS
    guards._TRUSTED_LOOPBACK_HOSTS = production | {"testserver"}
    yield
    guards._TRUSTED_LOOPBACK_HOSTS = production


@pytest.fixture
def cleanup_tmp_projects(tmp_path):
    """Delete Project rows created under this test's `tmp_path`, plus their
    dependent TaskObject, Conversation, and SharedResourceAttribution rows.

    Several artifact-ownership test modules create real Project rows against
    the session-scoped test DB; a leaked row (especially one with a real
    `.anton/artifacts` dir) pollutes other modules that scan or query all
    projects. Not autouse: opt in per module with
    `pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")`.
    """
    from sqlalchemy import delete, or_
    from sqlmodel import select

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.session import get_engine
    from cowork.models.conversation import Conversation
    from cowork.models.project import Project
    from cowork.models.shared_resource import SharedResourceAttribution
    from cowork.models.task_object import TaskObject

    yield

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        leaked_ids = list(
            session.exec(
                select(Project.id).where(Project.path.startswith(tmp_path.as_posix()))
            ).all()
        )
        if not leaked_ids:
            return
        session.exec(delete(TaskObject).where(TaskObject.project_id.in_(leaked_ids)))
        session.exec(delete(Conversation).where(Conversation.project_id.in_(leaked_ids)))
        # `artifact_resource_key` always starts with "<project_id>/".
        session.exec(
            delete(SharedResourceAttribution).where(
                or_(
                    *(
                        SharedResourceAttribution.resource_key.startswith(f"{pid}/")
                        for pid in leaked_ids
                    )
                )
            )
        )
        session.exec(delete(Project).where(Project.id.in_(leaked_ids)))
        session.commit()


@pytest.fixture
def org_deployment(monkeypatch):
    """Hosted org tenancy for one test, through the real AppSettings."""
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def stub_landlock_kernel(monkeypatch, *, abi: int = 6) -> None:
    """Answer the code sandbox's kernel checks without a kernel: the ABI probe
    reports ``abi`` and the launcher's self-test passes. The one seam
    cowork.code_sandbox leaves for tests, since neither macOS nor every CI
    kernel has Landlock."""
    from cowork import code_sandbox

    monkeypatch.setattr(code_sandbox, "_probe_abi", lambda: abi)
    monkeypatch.setattr(code_sandbox, "_run_self_test", lambda: None)


@pytest.fixture
def landlock_kernel(monkeypatch):
    """stub_landlock_kernel for one test: a kernel at Landlock ABI 6 whose
    self-test passes."""
    stub_landlock_kernel(monkeypatch)


@pytest.fixture
def personal_deployment(monkeypatch):
    """A local deployment that trusts identity headers, for one test.

    Local tenancy with COWORK_TRUST_IDENTITY_HEADERS and COWORK_REQUIRE_AUTH
    on, every resource personal (the setting's default), and
    COWORK_CODE_SANDBOX=landlock, which identity headers require, with its
    kernel checks stubbed (stub_landlock_kernel). COWORK_AUTH_TOKEN is set,
    since identity headers take the token from the environment alone. A test
    that leaves a resource shared sets COWORK_PERSONAL_RESOURCES and clears
    the settings cache itself.
    """
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "true")
    monkeypatch.setenv("COWORK_REQUIRE_AUTH", "true")
    monkeypatch.setenv("COWORK_AUTH_TOKEN", "personal-deployment-test-token")
    monkeypatch.setenv("COWORK_CODE_SANDBOX", "landlock")
    monkeypatch.delenv("COWORK_PERSONAL_RESOURCES", raising=False)
    stub_landlock_kernel(monkeypatch)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def self_hosted_minds_url(monkeypatch):
    """User settings whose minds_url is self-hosted's https://minds.invalid.

    The stack sets MINDS_URL to a host that never resolves while its model
    calls go to MindsHub. Returns the settings object, so a test can point
    minds_url elsewhere.
    """
    from types import SimpleNamespace

    from cowork.common.settings import user_settings

    settings = SimpleNamespace(minds_url="https://minds.invalid")
    monkeypatch.setattr(user_settings, "get_user_settings", lambda: settings)
    return settings


@pytest.fixture
def managed_dir(monkeypatch, tmp_path):
    """An operator's managed folder (COWORK_MANAGED_DIR) for one test.

    ``memory/`` holds one rule and one lesson and an empty profile, as the
    image bakes them; ``skills/`` and ``data-vault/`` start empty. Turns read
    it through the installed anton's Cortex and its read-only ``managed_hc``
    layer. Yields the cowork.managed.ManagedFolder.
    """
    from anton.core.memory.hippocampus import Hippocampus

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.managed import ManagedFolder
    from tests._fakes import MANAGED_LESSON, MANAGED_RULE

    folder = ManagedFolder.at(tmp_path / "managed")
    for path in (folder.memory_dir, folder.skills_dir, folder.vault_dir):
        path.mkdir(parents=True)
    managed_hc = Hippocampus(folder.memory_dir)
    managed_hc.encode_rule(MANAGED_RULE, kind="always", confidence="high", source="user")
    managed_hc.encode_lesson(MANAGED_LESSON, topic="data", source="user")
    (folder.memory_dir / "profile.md").write_text("", encoding="utf-8")

    monkeypatch.setenv("COWORK_MANAGED_DIR", str(folder.root))
    get_app_settings.cache_clear()
    yield folder
    get_app_settings.cache_clear()


@pytest.fixture
def stored_memory_enabled():
    """Store ``memory_enabled`` for one test as the operator's Settings write
    does (SettingService), and remove the row afterwards.

    The key is operator-pinned, so the row is the deployment's global one,
    which every scope resolves ahead of MEMORY_ENABLED. Call the yielded
    function with ``"true"`` or ``"false"``.
    """
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    def store(value: str) -> None:
        with get_open_session() as db:
            SettingService(db, LOCAL_SCOPE).upsert_setting("memory_enabled", value)

    yield store
    with get_open_session() as db:
        SettingService(db, LOCAL_SCOPE).delete_setting("memory_enabled")


@pytest.fixture
def granted_product_permissions(monkeypatch):
    """Model built-in Member grants only for suites testing resource ownership.

    Tests of live role decisions opt out and exercise the internal HTTP boundary.
    This fixture does not override artifact ownership or tenancy resolution.
    """
    from cowork.services import product_permissions
    from cowork.api.v1.endpoints import artifact_workspace

    async def allowed(scope, permission):
        assert permission in {"product.execute", "artifact.manage"}
        return True

    monkeypatch.setattr(product_permissions, "has_product_permission", allowed)
    monkeypatch.setattr(artifact_workspace, "has_product_permission", allowed)


@pytest.fixture
def openai_compatible_api(monkeypatch):
    """Set COWORK_OPENAI_COMPATIBLE_API for one test, or unset it with None.

    Read through the real AppSettings. get_app_settings() is cached for the
    process, so the cache is cleared on both sides of the test.
    """
    from cowork.common.settings.app_settings import get_app_settings

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv("COWORK_OPENAI_COMPATIBLE_API", raising=False)
        else:
            monkeypatch.setenv("COWORK_OPENAI_COMPATIBLE_API", value)
        get_app_settings.cache_clear()

    yield _set
    get_app_settings.cache_clear()


@pytest.fixture
def anton_responses_ready(monkeypatch):
    """Set whether the installed anton reports its Responses path ready.

    Call it with True, the default, or with False for an anton that predates
    RESPONSES_TRANSPORT_READY. Either way the test pins the flag itself, so it
    holds whichever anton the lock installs. The name is resolved when the
    returned callable runs, so a test that has swapped anton's OpenAIProvider
    for a stand-in sets the flag on the stand-in.
    """

    def _set(ready: bool = True) -> None:
        monkeypatch.setattr(
            "anton.core.llm.openai.OpenAIProvider.RESPONSES_TRANSPORT_READY",
            ready,
            raising=False,
        )

    return _set


@pytest.fixture
def one_connection_pool(monkeypatch):
    """Point the app at a pool of one connection, and make POOL_TIMEOUT 2 s.

    The test database is a SQLite file, so SQLAlchemy gives it a QueuePool with
    the same wait-then-TimeoutError checkout Postgres gets. SQLite engines are
    built with no pool settings (cowork.db.session), hence the engine built
    here. POOL_TIMEOUT is set to the engine's own wait, so the refusal's
    sentence, its Retry-After and the unit-slot wait all match it.
    """
    from sqlalchemy import create_engine

    import cowork.db.session as db_session

    uri = db_session.settings.database.uri
    engine = create_engine(
        uri, connect_args={"check_same_thread": False},
        pool_size=1, max_overflow=0, pool_timeout=2,
    )
    monkeypatch.setitem(db_session._engines, uri, engine)
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 2)
    yield engine
    # The cached sessionmaker would keep the disposed engine alive.
    db_session._session_factories.pop(id(engine), None)
    engine.dispose()


OwnedHandlerKind = Literal["console", "all_file", "error_file"]


@dataclass(frozen=True)
class OwnedLog:
    """A logger whose records pass through one production handler, then caplog."""

    logger: logging.Logger
    handler: logging.Handler
    stream: io.StringIO

    def output(self) -> str:
        """What the production handler wrote: its stream, or its log file."""
        self.handler.flush()
        if isinstance(self.handler, logging.FileHandler):
            return Path(self.handler.baseFilename).read_text()
        return self.stream.getvalue()


@pytest.fixture
def owned_logger(monkeypatch, caplog, tmp_path):
    """Route a logger through one production handler, then caplog.

    The handler's filters rewrite a record before caplog sees it, as the root
    console handler does in a running server. The logger does not propagate,
    so a test proves the selected handler owns the filter instead of
    borrowing the root console's, and its level is set here, so a test does
    not depend on the level LOG_LEVEL gave root.
    """
    from cowork.common import logger as app_logger

    handlers: list[logging.Handler] = []
    levels: list[tuple[logging.Logger, int]] = []

    def install(name: str, *, kind: OwnedHandlerKind = "console", level: int = logging.ERROR) -> OwnedLog:
        stream = io.StringIO()
        if kind == "console":
            monkeypatch.setenv("RICH_LOGGING", "false")
            handler = app_logger.setup_console_handler()
            handler.setStream(stream)
            handlers.append(handler)
        else:
            files = app_logger.setup_file_logging(str(tmp_path / "owned-logs"))
            handlers.extend(files)
            handler = files[0 if kind == "all_file" else 1]
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [handler, caplog.handler])
        monkeypatch.setattr(logger, "propagate", False)
        monkeypatch.setattr(logger, "disabled", False)
        # setLevel, not an attribute patch: it also clears the cached
        # isEnabledFor answers, which would otherwise keep the old level.
        levels.append((logger, logger.level))
        logger.setLevel(level)
        return OwnedLog(logger=logger, handler=handler, stream=stream)

    yield install
    for logger, level in reversed(levels):
        logger.setLevel(level)
    for handler in handlers:
        handler.close()
