"""Code with no scope on a deployment that trusts identity headers.

There, LOCAL_SCOPE is the operator view, which lists every person's
conversations, running answers, projects, files, schedules and pins, and reads
skills, connections, artifacts and memory from the deployment's own stores. So
a helper that finds no scope bound skips its work, as it does in org tenancy,
instead of falling back to LOCAL_SCOPE. Desktop keeps the fallback. Boot paths name
LOCAL_SCOPE explicitly, so the stack still boots with the setting on.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.services import skill_links, task_objects


@pytest.fixture
def desktop(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture(params=["personal", "org"])
def scope_required(request):
    """Both deployments where a missing scope must not mean LOCAL_SCOPE."""
    request.getfixturevalue(f"{request.param}_deployment")
    return request.param


def _refuse_sessions(monkeypatch):
    import cowork.db.session as db_session

    def refuse(*args, **kwargs):
        raise AssertionError("opened a session with no scope")

    monkeypatch.setattr(db_session, "get_open_session", refuse)


# ── artifact indexing ───────────────────────────────────────────────────────

def test_indexing_without_a_scope_is_skipped(scope_required, monkeypatch, caplog, owned_logger):
    _refuse_sessions(monkeypatch)
    owned_logger(task_objects.logger.name, level=logging.WARNING)
    task_objects._index_new_slugs(SimpleNamespace(created_by=None), uuid4(), uuid4(), ["chart"], None)
    [record] = [
        r for r in caplog.records if r.name == task_objects.logger.name and r.levelno == logging.WARNING
    ]
    assert record.getMessage().startswith(
        "artifact indexing skipped: no tenant scope provided on a partitioned deployment"
    )


def test_indexing_without_a_scope_is_local_on_a_desktop(desktop, monkeypatch):
    seen = []
    monkeypatch.setattr(task_objects, "record_new_artifacts", lambda session, **kw: seen.append(session.scope))
    task_objects._index_new_slugs(SimpleNamespace(created_by=None), uuid4(), uuid4(), ["chart"], None)
    assert seen == [LOCAL_SCOPE]


# ── skill-draft seeding ─────────────────────────────────────────────────────

@pytest.fixture
def base_skill(monkeypatch, tmp_path) -> Path:
    """A skill in the base store, and an empty draft folder to seed."""
    root = tmp_path / "skills"
    (root / "report").mkdir(parents=True)
    (root / "report" / "SKILL.md").write_text("---\nname: report\n---\nbody\n")
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(root))
    get_app_settings.cache_clear()
    draft = tmp_path / "draft"
    draft.mkdir()
    return draft


def test_draft_seeding_without_a_scope_copies_nothing(scope_required, base_skill):
    get_app_settings.cache_clear()
    task_objects._seed_draft_from_store(base_skill, "report")
    assert list(base_skill.iterdir()) == []


def test_draft_seeding_without_a_scope_reads_the_base_store_on_a_desktop(desktop, base_skill):
    get_app_settings.cache_clear()
    task_objects._seed_draft_from_store(base_skill, "report")
    assert [p.name for p in base_skill.iterdir()] == ["SKILL.md"]


def test_draft_seeding_in_org_tenancy_with_a_non_org_scope_copies_nothing(base_skill, monkeypatch):
    """Hosted org tenancy with a non-org scope bound (which no current caller
    does) must seed nothing from the unkeyed shared store rather than reading
    it as the operator would."""
    from cowork.common.settings.user_settings import use_settings_scope

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()

    with use_settings_scope(LOCAL_SCOPE):
        task_objects._seed_draft_from_store(base_skill, "report")

    assert list(base_skill.iterdir()) == []


# ── attachments for an in-process turn ──────────────────────────────────────

async def test_attachments_without_a_scope_are_skipped(scope_required, monkeypatch, caplog, owned_logger):
    async def refuse(*args, **kwargs):
        raise AssertionError("read attachments with no scope")

    monkeypatch.setattr(harness_mod, "run_db", refuse)
    owned_logger(harness_mod.logger.name, level=logging.WARNING)
    listed = await harness_mod._read_attachments(SimpleNamespace(id=uuid4(), org_id=None), scope=None)
    assert listed is None
    assert any(
        r.name == harness_mod.logger.name
        and r.levelno == logging.WARNING
        and r.getMessage().startswith("attachments: no tenant scope is bound on a partitioned deployment")
        for r in caplog.records
    )


async def test_attachments_without_a_scope_are_local_on_a_desktop(desktop, monkeypatch):
    seen = []

    async def recording(work, *, scope):
        seen.append(scope)
        return []

    monkeypatch.setattr(harness_mod, "run_db", recording)
    assert await harness_mod._read_attachments(SimpleNamespace(id=uuid4(), org_id=None), scope=None) == []
    assert seen == [LOCAL_SCOPE]


# ── adopted-folder discovery for skill links ────────────────────────────────

def test_adopted_folders_are_not_read_with_scopes_required(scope_required, monkeypatch, caplog, owned_logger):
    # The read swallows its own failure into {}, so the signal is that it never
    # tried: no session opened, nothing logged.
    _refuse_sessions(monkeypatch)
    owned_logger(skill_links.logger.name, level=logging.WARNING)
    assert skill_links._external_project_dirs() == {}
    assert [r for r in caplog.records if r.name == skill_links.logger.name] == []


# ── boot ────────────────────────────────────────────────────────────────────

def test_the_stack_boots_with_trusted_identity(personal_deployment, monkeypatch, tmp_path):
    """run_dev_setup on a fresh database and home, with the setting on: the
    unscoped boot paths name LOCAL_SCOPE, so the base stores are seeded and
    nothing raises MissingTenantScopeError."""
    from cowork.dev_setup import run_dev_setup
    from cowork.services.skills import SkillService

    monkeypatch.setenv("DATABASE_URI", f"sqlite:///{tmp_path / 'boot.db'}")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    get_app_settings.cache_clear()

    # The guard the boot paths now step around.
    with pytest.raises(MissingTenantScopeError):
        SkillService()

    run_dev_setup()

    assert sorted(p.name for p in (tmp_path / "skills").iterdir() if p.is_dir())
    assert (tmp_path / "memory").is_dir()
    assert SkillService(LOCAL_SCOPE).root == tmp_path / "skills"
