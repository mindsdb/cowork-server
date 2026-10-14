"""Skills on a deployment that trusts identity headers: each person's own.

A person lists, edits and deletes only their own store. Nothing fans links out
of the unkeyed store, at boot or on a change; instead each in-process turn
links its project's skills from the asker's own store, with the selection rule
a remote turn uses, after seeding the read-only builtins. Skills left shared
keep today's links, and the desktop is unchanged.
"""
from __future__ import annotations

import shutil
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from sqlalchemy import delete

from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import Provider, UserSettings, use_settings_scope
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.schemas.skills import SkillUpdateRequest
from cowork.services.skill_links import sync_turn_skill_links
from cowork.services.skills import BUILTIN_SKILL_SLUGS, SkillService, link_turn_skills

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)


def _person(*, partition: str, sub: str, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


ALICE = _person(partition=PARTITION_A, sub=SUB_A)
BOB = _person(partition=PARTITION_B, sub=SUB_B)


@pytest.fixture
def stack(personal_deployment, monkeypatch, tmp_path):
    """A trusted-identity deployment whose stores live under the test's folder."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "home" / "skills"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    get_app_settings.cache_clear()
    yield tmp_path
    with get_open_session() as db:
        db.exec(delete(Conversation).where(Conversation.org_id.in_([PARTITION_A, PARTITION_B])))
        db.exec(delete(Project).where(Project.org_id.in_([PARTITION_A, PARTITION_B])))
        db.commit()
    get_app_settings.cache_clear()


@pytest.fixture
def scoped(stack):
    """One request session per scope, as production opens one per request: a
    raw session refuses a second partition once it carries one."""
    sessions: dict[TenantScope, ScopedSession] = {}

    def for_scope(scope: TenantScope) -> ScopedSession:
        if scope not in sessions:
            sessions[scope] = ScopedSession(get_open_session(), scope)
        return sessions[scope]

    try:
        yield for_scope
    finally:
        for session in sessions.values():
            session.close()


def _links(skills_dir: Path) -> dict[str, Path]:
    return {p.name: p.resolve() for p in skills_dir.iterdir() if p.is_symlink()}


# ── the store over the API ──────────────────────────────────────────────────


def test_a_person_lists_and_changes_only_their_own_skills(scoped):
    SkillService(ALICE).create_skill("probe-a-skill", "Alice's private procedure.")

    bobs = {s["id"] for s in skills_ep.list_skills(scoped(BOB), None)["skills"]}
    assert "probe-a-skill" not in bobs
    with pytest.raises(HTTPException) as edit:
        skills_ep.update_skill(
            "probe-a-skill", SkillUpdateRequest(instructions="Bob's now"), scoped(BOB), None
        )
    assert edit.value.status_code == 404
    with pytest.raises(HTTPException) as removal:
        skills_ep.delete_skill("probe-a-skill", scoped(BOB), None)
    assert removal.value.status_code == 404

    unchanged = skills_ep.get_skill("probe-a-skill", scoped(ALICE), None)
    assert unchanged["declarative"] == "Alice's private procedure."
    alices = {s["id"] for s in skills_ep.list_skills(scoped(ALICE), None)["skills"]}
    assert "probe-a-skill" in alices


def test_shared_skills_stay_one_store(scoped, monkeypatch):
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "")
    get_app_settings.cache_clear()
    shared_a = _person(partition=PARTITION_A, sub=SUB_A, resources=frozenset())
    shared_b = _person(partition=PARTITION_B, sub=SUB_B, resources=frozenset())
    SkillService(shared_a).create_skill("team-procedure", "Everyone's.")

    assert "team-procedure" in {s["id"] for s in skills_ep.list_skills(scoped(shared_b), None)["skills"]}
    assert "team-procedure" in {s["id"] for s in skills_ep.list_skills(scoped(LOCAL_SCOPE), None)["skills"]}


# ── links per turn ──────────────────────────────────────────────────────────


def test_each_turn_links_only_the_askers_own_skills(stack):
    SkillService(ALICE).create_skill("probe-a-skill", "Alice's private procedure.")
    a_skills = stack / "a-project" / "skills"
    b_skills = stack / "b-project" / "skills"

    link_turn_skills(scope=ALICE, project_name="general", skills_dir=a_skills)
    link_turn_skills(scope=BOB, project_name="general", skills_dir=b_skills)

    a_store = SkillService(ALICE).root.resolve()
    b_store = SkillService(BOB).root.resolve()
    assert _links(a_skills)["probe-a-skill"] == a_store / "probe-a-skill"
    assert "probe-a-skill" not in _links(b_skills)
    # Every link points into the asker's own store, builtins included: a
    # person's first turn seeds their read-only copies, as a remote turn does.
    assert set(BUILTIN_SKILL_SLUGS) <= set(_links(b_skills))
    assert {target.parent for target in _links(b_skills).values()} == {b_store}
    assert {target.parent for target in _links(a_skills).values()} == {a_store}


def test_a_turn_keeps_the_project_selection_rule(stack):
    service = SkillService(ALICE)
    service.create_skill("for-general", "Here.", projects=["general"])
    service.create_skill("for-elsewhere", "Not here.", projects=["elsewhere"])
    service.create_skill("switched-off", "Never.", enabled=False)
    skills_dir = stack / "a-project" / "skills"

    link_turn_skills(scope=ALICE, project_name="general", skills_dir=skills_dir)

    linked = set(_links(skills_dir))
    assert "for-general" in linked
    assert "for-elsewhere" not in linked
    assert "switched-off" not in linked


def test_a_turn_replaces_links_from_another_store_and_keeps_real_folders(stack):
    service = SkillService(ALICE)
    service.create_skill("probe-a-skill", "Alice's.")
    unkeyed = stack / "home" / "skills"
    (unkeyed / "probe-a-skill").mkdir(parents=True)
    (unkeyed / "legacy-shared").mkdir(parents=True)
    skills_dir = stack / "a-project" / "skills"
    skills_dir.mkdir(parents=True)
    # What an earlier fan-out from the unkeyed store left behind.
    (skills_dir / "probe-a-skill").symlink_to(unkeyed / "probe-a-skill", target_is_directory=True)
    (skills_dir / "legacy-shared").symlink_to(unkeyed / "legacy-shared", target_is_directory=True)
    # A skill the agent wrote straight into the folder: the draft relocation's.
    (skills_dir / "agent-written").mkdir()
    (skills_dir / "agent-written" / "SKILL.md").write_text("---\nname: agent-written\n---\n")

    link_turn_skills(scope=ALICE, project_name="general", skills_dir=skills_dir)

    links = _links(skills_dir)
    assert links["probe-a-skill"] == service.root.resolve() / "probe-a-skill"
    assert "legacy-shared" not in links and not (skills_dir / "legacy-shared").exists()
    assert (skills_dir / "agent-written" / "SKILL.md").is_file()


def test_a_link_that_cannot_be_made_does_not_stop_the_turn(stack):
    blocked = stack / "a-project" / "skills"
    blocked.mkdir(parents=True)
    (blocked / "docx").mkdir()  # a real folder squats one builtin's name

    link_turn_skills(scope=ALICE, project_name="general", skills_dir=blocked)

    assert not (blocked / "docx").is_symlink()
    assert set(BUILTIN_SKILL_SLUGS) - {"docx"} <= set(_links(blocked))


def test_sync_alone_removes_every_unwanted_link(tmp_path):
    store = tmp_path / "store"
    (store / "keep").mkdir(parents=True)
    (store / "drop").mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "drop").symlink_to(store / "drop", target_is_directory=True)
    (skills_dir / "dangling").symlink_to(tmp_path / "gone", target_is_directory=True)

    sync_turn_skill_links(skills_dir=skills_dir, wanted={"keep": store / "keep"})

    assert _links(skills_dir) == {"keep": (store / "keep").resolve()}
    assert not (skills_dir / "dangling").is_symlink()


# ── no fan-out where skills are partitioned ─────────────────────────────────


def test_no_change_or_boot_fans_skills_out_inside_a_partition(stack, monkeypatch):
    from cowork import dev_setup

    reconciled = []
    monkeypatch.setattr(
        "cowork.services.skill_links.reconcile_all", lambda skills: reconciled.append(skills)
    )

    dev_setup._distribute_skill_links()

    assert reconciled == []
    for scope in (ALICE, LOCAL_SCOPE):
        assert SkillService(scope)._link_projects is False


def test_boot_still_fans_out_where_skills_are_shared(stack, monkeypatch):
    from cowork import dev_setup

    monkeypatch.setenv(
        "COWORK_PERSONAL_RESOURCES", "projects,conversations,files,pins,schedules"
    )
    get_app_settings.cache_clear()
    reconciled = []
    monkeypatch.setattr(
        "cowork.services.skill_links.reconcile_all", lambda skills: reconciled.append(skills)
    )

    dev_setup._distribute_skill_links()

    assert len(reconciled) == 1


# ── the in-process turn ─────────────────────────────────────────────────────


@asynccontextmanager
async def _turn_config(*, monkeypatch, project_path: Path, scope: TenantScope | None, project_org: str | None):
    """Build one in-process turn's session config, bound to ``scope``."""
    saved = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI,
        coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI,
        planning_model="p", coding_model="c", router_model="r",
        openai_api_key=SecretStr("test-only-key"),
        episodic_memory=False,
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: saved)
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")

    project_path.mkdir(parents=True, exist_ok=True)
    with get_open_session() as db:
        project = Project(name=project_path.name, path=str(project_path), org_id=project_org)
        db.add(project)
        db.commit()
        conversation = Conversation(
            topic="skills",
            project_id=project.id,
            org_id=scope.org_id if scope is not None and scope.partitioned_for(
                PersonalResource.CONVERSATIONS
            ) else None,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        with use_settings_scope(scope) if scope is not None else nullcontext():
            config, scratch, _ = await harness.AntonHarness()._build_chat_session(conversation)
        try:
            yield config, scratch
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)
            client = config.llm_client
            for provider in (client.planning_provider, client.coding_provider, client.router_provider):
                await provider._client.close()


async def test_a_persons_turn_reads_their_own_skills_from_their_project(monkeypatch, stack):
    SkillService(ALICE).create_skill("probe-a-skill", "Alice's private procedure.")
    a_project = stack / "shared" / PARTITION_A / "projects" / f"a-{uuid4().hex[:6]}"
    b_project = stack / "shared" / PARTITION_B / "projects" / f"b-{uuid4().hex[:6]}"

    async with _turn_config(
        monkeypatch=monkeypatch, project_path=a_project, scope=ALICE, project_org=PARTITION_A
    ) as (config, _scratch):
        assert config.settings.skills_root == a_project / "skills"
        assert "probe-a-skill" in _links(a_project / "skills")
    async with _turn_config(
        monkeypatch=monkeypatch, project_path=b_project, scope=BOB, project_org=PARTITION_B
    ) as (config, _scratch):
        assert config.settings.skills_root == b_project / "skills"
        assert "probe-a-skill" not in _links(b_project / "skills")
        assert set(BUILTIN_SKILL_SLUGS) <= set(_links(b_project / "skills"))


async def test_a_shared_projects_folder_is_never_rewritten_by_a_persons_turn(monkeypatch, stack):
    """With skills personal and projects shared, two people work in one
    project folder at once, so each turn links its own skills in its scratch
    folder and leaves the project's skills folder alone."""
    shared_projects = frozenset(
        ALL_PERSONAL_RESOURCES
        - {PersonalResource.PROJECTS}
    )
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(shared_projects)))
    get_app_settings.cache_clear()
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=shared_projects)
    SkillService(alice).create_skill("probe-a-skill", "Alice's private procedure.")
    project = stack / "home" / "projects" / f"team-{uuid4().hex[:6]}"

    async with _turn_config(
        monkeypatch=monkeypatch, project_path=project, scope=alice, project_org=None
    ) as (config, scratch):
        assert scratch is not None
        assert config.settings.skills_root == scratch / "skills"
        assert "probe-a-skill" in _links(scratch / "skills")
        assert _links(project / "skills") == {}


async def test_desktop_turns_keep_the_fanned_out_project_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    get_app_settings.cache_clear()
    calls = []
    monkeypatch.setattr(harness, "link_turn_skills", lambda **kwargs: calls.append(kwargs))
    try:
        project = tmp_path / "projects" / f"desk-{uuid4().hex[:6]}"
        async with _turn_config(
            monkeypatch=monkeypatch, project_path=project, scope=None, project_org=None
        ) as (config, _scratch):
            assert config.settings.skills_root == project / "skills"
        assert calls == []
    finally:
        get_app_settings.cache_clear()
        with get_open_session() as db:
            db.exec(delete(Project).where(Project.path == str(project)))
            db.commit()
