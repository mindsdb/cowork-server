"""The operator commands, ``python -m cowork.ops.<module>``, on two people.

Each command acts in one person's space, worked out from the pair the proxy
sends: memory print shows what that person's agent learned and nothing of
the other's, and after memory clear it shows nothing learned. Where memory is
shared, memory print and clear take no person and act on the deployment's one
store alone; while memory is off, print shows nothing learned and clear
refuses. skills share
adds a person's skill to the shared-skill list in the shared root, with the
checks and texts the person's own share gets, where everyone else reads it
from the person's own store while the person still edits it, and unshare
takes it out again; neither runs as root. skills snapshot lists what the person's agent sees,
shared skills included, with a sha256 that stays the same while the skills
do. connections share adds a person's connection to the shared-connection
registry, where everyone else reads it from the person's own vault, and
unshare takes it out again.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from sqlalchemy import delete

from cowork import shared_skills
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession
from cowork.db.session import get_open_session
from cowork.harnesses.memory.registry import MemorySlot
from cowork.managed import readonly_skill_dirs
from cowork.models.project import Project
from cowork.ops import Person, personal_scope
from cowork.ops import connections as connections_ops
from cowork.ops import memory as memory_ops
from cowork.ops import skills as skills_ops
from cowork.schemas.memory import MemoryScope
from cowork.services.connectors.persist import own_vault_for_scope, vault_for_scope
from cowork.services.memory import MemoryService
from cowork.services.skills import BUILTIN_SKILL_SLUGS, SKILL_SHARING_UNAVAILABLE, SkillService
from cowork.shared_skills import shared_skills_file
from tests._fakes import shared_skill_list_lock_held

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
SUB_B = "22222222-2222-4222-8222-222222222222"
ALICE = Person(organization_id=DEPLOYMENT, user_id=SUB_A)
BOB = Person(organization_id=DEPLOYMENT, user_id=SUB_B)


def _arguments(person: Person) -> list[str]:
    return ["--organization-id", person.organization_id, "--user-id", person.user_id]


def _run(main, argv: list[str], capsys) -> tuple[int, str, str]:
    status = main(argv)
    captured = capsys.readouterr()
    return status, captured.out, captured.err


@pytest.fixture
def stack(personal_deployment, managed_dir, monkeypatch, tmp_path):
    """A trusted-identity deployment with a managed folder, its stores under
    the test's folder. The two people's rows are removed afterwards."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "home" / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "home" / "memory"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "home" / "data-vault"))
    # As Cowork's own user, whoever runs the suite: a skill share or unshare
    # refuses root.
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    get_app_settings.cache_clear()
    yield managed_dir
    with get_open_session() as db:
        db.exec(delete(Project).where(Project.org_id.in_([ALICE.partition_id, BOB.partition_id])))
        db.commit()
    get_app_settings.cache_clear()


def test_the_person_maps_to_the_pinned_partition_in_any_letter_case():
    person = Person(organization_id=DEPLOYMENT.upper(), user_id=SUB_A.upper())
    assert person.user_id == SUB_A
    assert person.partition_id == "44e92829-e701-599b-88f9-69f5bf1165eb"


def test_the_scope_is_the_one_the_persons_requests_get(stack):
    scope = personal_scope(ALICE)
    assert scope.personal and scope.org_id == ALICE.partition_id and scope.user_id == SUB_A


# ── memory ──────────────────────────────────────────────────────────────────


def _general(person: Person, root: Path) -> Project:
    raw = get_open_session()
    try:
        scoped = ScopedSession(raw, personal_scope(person))
        path = root / "shared" / person.partition_id / "projects" / "general"
        path.mkdir(parents=True, exist_ok=True)
        project = Project(name="general", path=str(path))
        scoped.add(project)
        scoped.commit()
        scoped.refresh(project)
        return project
    finally:
        raw.close()


def _remember(person: Person, project: Project, *, rule: str, lesson: str) -> None:
    raw = get_open_session()
    try:
        service = MemoryService(ScopedSession(raw, personal_scope(person)))
        service.update_memory_sync(MemoryScope.global_, MemorySlot.RULES, rule)
        service.update_memory_sync(MemoryScope.project, MemorySlot.LESSONS, lesson, project.id)
    finally:
        raw.close()


def test_memory_print_then_clear_on_two_people(stack, capsys, tmp_path):
    alice_general = _general(ALICE, tmp_path)
    bob_general = _general(BOB, tmp_path)
    _remember(ALICE, alice_general, rule="- Favorite part number is P-1111", lesson="- Give quantities in pallets")
    _remember(BOB, bob_general, rule="- Answer in cases", lesson="- The plant ships on Fridays")

    status, out, _ = _run(memory_ops.main, ["print", *_arguments(ALICE)], capsys)
    assert status == 0
    alice_tier = tmp_path / "shared" / ALICE.partition_id / "memory" / "users" / SUB_A
    assert f"Global, in {alice_tier}" in out
    assert f"Project general ({alice_general.id}), in {alice_tier / 'projects' / str(alice_general.id)}" in out
    assert "P-1111" in out and "pallets" in out
    assert "cases" not in out and "Fridays" not in out
    assert "Nothing learned." not in out

    status, out, _ = _run(memory_ops.main, ["clear", *_arguments(ALICE)], capsys)
    assert status == 0
    assert out == f"Cleared the memory of user {SUB_A}, in {alice_tier}.\n"

    status, out, _ = _run(memory_ops.main, ["print", *_arguments(ALICE)], capsys)
    assert status == 0
    assert out.splitlines()[-1] == "Nothing learned."
    assert "P-1111" not in out and "pallets" not in out

    # The other person's memory is untouched.
    _, out, _ = _run(memory_ops.main, ["print", *_arguments(BOB)], capsys)
    assert "Answer in cases" in out and "Fridays" in out

    status, out, _ = _run(memory_ops.main, ["clear", *_arguments(ALICE)], capsys)
    assert status == 0 and out.startswith(f"User {SUB_A} has no memory to clear")


SHARES_MEMORY = "conversations,projects,files,skills,connections,schedules,pins,settings"


def _share_memory(monkeypatch) -> None:
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", SHARES_MEMORY)
    get_app_settings.cache_clear()


def _tree(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes() if path.is_file() else b""
        for path in sorted(folder.rglob("*"))
    }


@pytest.mark.parametrize(
    "argv",
    [[], ["--organization-id", DEPLOYMENT], ["--user-id", SUB_A]],
    ids=["neither", "organization-only", "user-only"],
)
def test_memory_kept_per_person_needs_both_person_arguments(stack, capsys, argv):
    for action in ("print", "clear"):
        status, out, err = _run(memory_ops.main, [action, *argv], capsys)
        assert status == 1 and out == ""
        assert f"so {action} needs --organization-id and --user-id" in err


def test_shared_memory_print_shows_the_one_store_by_project_id(stack, monkeypatch, capsys, tmp_path):
    _share_memory(monkeypatch)
    alice_general = _general(ALICE, tmp_path)
    bob_general = _general(BOB, tmp_path)
    _remember(ALICE, alice_general, rule="- Favorite part number is P-1111", lesson="- Give quantities in pallets")
    _remember(BOB, bob_general, rule="- Favorite part number is P-1111", lesson="- The plant ships on Fridays")

    status, out, _ = _run(memory_ops.main, ["print"], capsys)

    assert status == 0
    store = tmp_path / "home" / "memory"
    assert out.splitlines()[0] == "Shared memory of this deployment"
    assert f"Global, in {store}" in out
    for project in (alice_general, bob_general):
        assert f"Project {project.id}, in {store / 'projects' / str(project.id)}" in out
    assert "P-1111" in out and "pallets" in out and "Fridays" in out
    # Each project is named by its id alone, never by its owner's name for it.
    assert "Project general" not in out


GLOBAL_SLOT_FILES = ("profile.md", "rules.md", "lessons.md")


def test_shared_memory_clear_keeps_every_persons_tier_and_the_managed_folder(
    stack, monkeypatch, capsys, tmp_path
):
    """Every global slot goes, profile included, and every project tier. A
    folder that is no project id stays, and so does a link named like a tier,
    with what it points at."""
    alice_general = _general(ALICE, tmp_path)
    _remember(ALICE, alice_general, rule="- Answer in cases", lesson="- Alice's own lesson")
    alice_tier = tmp_path / "shared" / ALICE.partition_id / "memory" / "users" / SUB_A
    personal = _tree(alice_tier)
    managed = _tree(stack.root)
    _share_memory(monkeypatch)
    _remember(ALICE, alice_general, rule="- Shared rule", lesson="- Shared lesson")
    raw = get_open_session()
    try:
        service = MemoryService(ScopedSession(raw, personal_scope(ALICE)))
        service.update_memory_sync(MemoryScope.global_, MemorySlot.PROFILE, "- Shared profile line")
        service.update_memory_sync(MemoryScope.global_, MemorySlot.LESSONS, "- Shared global lesson")
    finally:
        raw.close()
    store = tmp_path / "home" / "memory"
    assert all((store / name).is_file() for name in GLOBAL_SLOT_FILES)
    (store / "projects" / "notes").mkdir()
    (store / "projects" / "notes" / "keep.md").write_text("not a project tier\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "lessons.md").write_text("- Behind a link\n")
    link = store / "projects" / str(uuid4())
    os.symlink(outside, link, target_is_directory=True)
    pointed_at = _tree(outside)

    status, out, _ = _run(memory_ops.main, ["clear"], capsys)

    assert status == 0
    assert out == f"Cleared the deployment's shared memory, in {store}.\n"
    assert not any((store / name).exists() for name in GLOBAL_SLOT_FILES)
    assert not (store / "projects" / str(alice_general.id)).exists()
    assert (store / "projects" / "notes" / "keep.md").exists()
    assert link.is_symlink() and _tree(outside) == pointed_at
    assert _tree(alice_tier) == personal and _tree(stack.root) == managed
    status, out, _ = _run(memory_ops.main, ["print"], capsys)
    assert status == 0 and out.splitlines()[-1] == "Nothing learned."
    status, out, _ = _run(memory_ops.main, ["clear"], capsys)
    assert status == 0 and out == f"The deployment has no shared memory to clear, in {store}.\n"


def test_shared_memory_refuses_a_link_in_place_of_the_project_tiers(stack, monkeypatch, capsys, tmp_path):
    """A link where the shared store's projects/ folder belongs is refused,
    never followed: print and clear fail, and clear removes nothing."""
    _share_memory(monkeypatch)
    alice_general = _general(ALICE, tmp_path)
    _remember(ALICE, alice_general, rule="- Shared rule", lesson="- Shared lesson")
    store = tmp_path / "home" / "memory"
    outside = tmp_path / "outside"
    (store / "projects").rename(outside)
    os.symlink(outside, store / "projects", target_is_directory=True)
    stored, pointed_at = _tree(store), _tree(outside)

    for action in ("print", "clear"):
        status, out, err = _run(memory_ops.main, [action], capsys)
        assert status == 1 and out == ""
        assert "where its projects/ folder should be is refused, never followed" in err
    assert (store / "rules.md").is_file() and (store / "projects").is_symlink()
    assert _tree(store) == stored and _tree(outside) == pointed_at


@pytest.mark.parametrize(
    "argv",
    [_arguments(ALICE), ["--organization-id", DEPLOYMENT], ["--user-id", SUB_A]],
    ids=["both", "organization-only", "user-only"],
)
def test_shared_memory_refuses_a_person(stack, monkeypatch, capsys, tmp_path, argv):
    """A clear meant for one person never becomes a clear of everyone's store."""
    _share_memory(monkeypatch)
    alice_general = _general(ALICE, tmp_path)
    _remember(ALICE, alice_general, rule="- Shared rule", lesson="- Shared lesson")
    store = _tree(tmp_path / "home" / "memory")

    for action in ("print", "clear"):
        status, out, err = _run(memory_ops.main, [action, *argv], capsys)
        assert status == 1 and out == ""
        assert f"so {action} acts on its one store and takes no --organization-id or --user-id" in err
    assert _tree(tmp_path / "home" / "memory") == store


@pytest.mark.parametrize("shared", [False, True], ids=["personal", "shared"])
def test_memory_off_prints_nothing_learned_and_refuses_to_clear(stack, monkeypatch, capsys, tmp_path, shared):
    alice_general = _general(ALICE, tmp_path)
    _remember(ALICE, alice_general, rule="- Answer in cases", lesson="- Alice's own lesson")
    if shared:
        _share_memory(monkeypatch)
        _remember(ALICE, alice_general, rule="- Shared rule", lesson="- Shared lesson")
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    argv = [] if shared else _arguments(ALICE)
    stored = (_tree(tmp_path / "shared"), _tree(tmp_path / "home" / "memory"))

    status, out, err = _run(memory_ops.main, ["print", *argv], capsys)
    assert status == 0 and err == ""
    assert out == memory_ops.MEMORY_OFF_PRINT + "\n"
    assert out.startswith("Memory is off") and "kept as it is, unread, until memory goes back on" in out
    assert "Shared rule" not in out and "Alice's own lesson" not in out

    status, out, err = _run(memory_ops.main, ["clear", *argv], capsys)
    assert status == 1 and out == ""
    assert err == f"error: {memory_ops.MEMORY_OFF_CLEAR}\n"
    assert "Clearing it is refused" in err and "kept as it is" in err

    assert (_tree(tmp_path / "shared"), _tree(tmp_path / "home" / "memory")) == stored


@pytest.mark.parametrize("shared", [False, True], ids=["personal", "shared"])
def test_memory_off_still_checks_the_person_arguments_first(stack, monkeypatch, capsys, shared):
    if shared:
        _share_memory(monkeypatch)
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    argv = _arguments(ALICE) if shared else []

    status, out, err = _run(memory_ops.main, ["print", *argv], capsys)

    assert status == 1 and out == ""
    assert "--organization-id" in err and "Memory is off" not in err


def test_a_stored_memory_setting_turns_the_commands_off(stack, capsys, stored_memory_enabled):
    """The commands resolve memory_enabled as turns do: a stored row wins over
    MEMORY_ENABLED."""
    stored_memory_enabled("false")
    status, out, _ = _run(memory_ops.main, ["print", *_arguments(ALICE)], capsys)
    assert status == 0 and out == memory_ops.MEMORY_OFF_PRINT + "\n"


def test_commands_refuse_on_a_deployment_that_trusts_no_identity(stack, monkeypatch, capsys):
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    get_app_settings.cache_clear()
    status, out, err = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
    assert status == 1 and out == ""
    assert "COWORK_TRUST_IDENTITY_HEADERS is off" in err


def test_memory_commands_refuse_on_a_deployment_that_trusts_no_identity(stack, monkeypatch, capsys):
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    get_app_settings.cache_clear()
    for argv in (["print", *_arguments(ALICE)], ["clear"]):
        status, out, err = _run(memory_ops.main, argv, capsys)
        assert status == 1 and out == ""
        assert "COWORK_TRUST_IDENTITY_HEADERS is off" in err


def test_a_person_id_that_is_not_a_uuid_is_refused(stack, capsys):
    with pytest.raises(SystemExit) as refused:
        memory_ops.main(["print", "--organization-id", DEPLOYMENT, "--user-id", "not-a-uuid"])
    assert refused.value.code == 2
    assert "is not a UUID" in capsys.readouterr().err


# ── skills ──────────────────────────────────────────────────────────────────


def _skill_command(person: Person, slug: str, capsys, *, action: str = "share") -> tuple[int, str, str]:
    return _run(skills_ops.main, [action, *_arguments(person), "--skill", slug], capsys)


def _skill_list() -> list[dict]:
    return json.loads(shared_skills_file().read_text())


def _listed_skills(person: Person) -> dict[str, dict]:
    raw = get_open_session()
    try:
        scoped = ScopedSession(raw, personal_scope(person))
        return {s["id"]: s for s in skills_ep.list_skills(scoped, None)["skills"]}
    finally:
        raw.close()


def test_share_skill_adds_it_to_the_list_and_its_creator_keeps_edit(stack, capsys, tmp_path):
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "Do it the team's way.")

    status, out, _ = _skill_command(ALICE, "team-skill", capsys)

    assert status == 0
    list_file = tmp_path / "shared" / "shared" / "skills.json"
    assert shared_skills_file() == list_file
    assert out == (
        f"Shared skill team-skill of user {SUB_A} with everyone, read-only: "
        f"added it to {list_file}.\n"
    )
    assert _skill_list() == [
        {"user_id": SUB_A, "partition_id": ALICE.partition_id, "skill": "team-skill"}
    ]
    assert list_file.stat().st_mode & 0o777 == 0o644
    # Nothing is copied: the skill stays in Alice's store, the managed
    # skills folder stays empty, and nothing is written into the managed
    # folder.
    assert (alice.root / "team-skill" / "SKILL.md").is_file()
    assert list(stack.skills_dir.iterdir()) == []
    assert not stack.shared_dir.exists()

    bobs = _listed_skills(BOB)["team-skill"]
    assert (bobs["isBuiltin"], bobs["shared"]) == (False, True)
    assert bobs["capabilities"] == {"canEdit": False, "canDelete": False, "canDisable": False, "canShare": False}
    alices = _listed_skills(ALICE)["team-skill"]
    assert (alices["shared"], alices["capabilities"]["canEdit"], alices["capabilities"]["canShare"]) == (
        True,
        True,
        True,
    )
    alice.update_skill("team-skill", instructions="The team's new way.")
    assert _listed_skills(BOB)["team-skill"]["declarative"] == "The team's new way."


def test_unshare_skill_removes_it_and_keeps_the_persons_own(stack, capsys):
    SkillService(personal_scope(ALICE)).create_skill("team-skill", "Alice's.")
    SkillService(personal_scope(BOB)).create_skill("bob-skill", "Bob's.")
    _skill_command(ALICE, "team-skill", capsys)
    _skill_command(BOB, "bob-skill", capsys)

    status, out, _ = _skill_command(ALICE, "team-skill", capsys, action="unshare")

    assert status == 0
    assert out == (
        f"Stopped sharing skill team-skill of user {SUB_A}: removed it from "
        f"{shared_skills_file()}. Their own skill stays in their store.\n"
    )
    assert [entry["skill"] for entry in _skill_list()] == ["bob-skill"]
    assert "team-skill" not in _listed_skills(BOB)
    assert _listed_skills(ALICE)["team-skill"]["shared"] is False


BUILTIN = sorted(BUILTIN_SKILL_SLUGS)[0]


@pytest.mark.parametrize(
    ("slug", "message"),
    [
        (BUILTIN, f"{BUILTIN} is a built-in skill that everyone already has, so it can't be shared."),
        ("never-made", f"User {SUB_A} has no skill named never-made."),
    ],
)
def test_share_skill_refuses_what_it_cannot_share(stack, capsys, slug, message):
    SkillService(personal_scope(ALICE)).ensure_builtin_skills()
    status, out, err = _skill_command(ALICE, slug, capsys)
    assert status == 1 and out == ""
    assert err == f"error: {message}\n"
    assert not shared_skills_file().exists()


def _first_skill_in(root: Path) -> str:
    return sorted(readonly_skill_dirs(root).readable)[0]


@pytest.mark.parametrize("root", ["anton", "host"])
def test_share_skill_refuses_a_name_the_agent_already_has(stack, capsys, root):
    """anton reads its packaged skills after the shared ones and the
    harness's own before them, so a share of either name would replace
    anton's for everyone or reach nobody."""
    from anton.core.memory.skills import SkillStore

    from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR

    slug = _first_skill_in(SkillStore().builtin_root if root == "anton" else HOST_SKILLS_DIR)
    SkillService(personal_scope(ALICE)).create_skill(slug, "Alice's own version.")

    status, out, err = _skill_command(ALICE, slug, capsys)

    assert status == 1 and out == ""
    assert err == (
        f"error: Everyone's agent already has a skill named {slug}, so sharing yours would reach "
        "nobody. Rename your skill before you share it.\n"
    )
    assert not shared_skills_file().exists()


def test_share_skill_refuses_a_skill_holding_a_link(stack, capsys, tmp_path):
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "Do it.")
    secret = tmp_path / "secret.txt"
    secret.write_text("not for everyone")
    (alice.root / "team-skill" / "notes.txt").symlink_to(secret)

    status, _, err = _skill_command(ALICE, "team-skill", capsys)

    assert status == 1
    assert err == "error: This skill holds a link, notes.txt. Remove it before you share the skill.\n"
    assert (alice.root / "team-skill" / "SKILL.md").exists()
    assert not shared_skills_file().exists()


def test_share_skill_refuses_one_already_shared_or_hidden(stack, capsys):
    SkillService(personal_scope(ALICE)).create_skill("team-skill", "Alice's.")
    SkillService(personal_scope(BOB)).create_skill("team-skill", "Bob's.")
    SkillService(personal_scope(BOB)).create_skill("report", "Bob's.")
    (stack.skills_dir / "report").mkdir()
    (stack.skills_dir / "report" / "SKILL.md").write_text(
        "---\nname: report\ndescription: The operator's.\n---\nOperator way.\n"
    )
    _skill_command(ALICE, "team-skill", capsys)

    for person, slug, message in (
        (ALICE, "team-skill", f"Skill team-skill of user {SUB_A} is already shared."),
        (
            BOB,
            "team-skill",
            "Someone else already shares a skill named team-skill. Rename your skill before you share it.",
        ),
        (
            BOB,
            "report",
            "A built-in skill is already named report, so sharing yours would reach nobody. "
            "Rename your skill before you share it.",
        ),
    ):
        status, out, err = _skill_command(person, slug, capsys)
        assert (status, out) == (1, "")
        assert err == f"error: {message}\n"
    assert len(_skill_list()) == 1


def test_unshare_skill_refuses_a_name_the_person_shares_none_of(stack, capsys):
    SkillService(personal_scope(ALICE)).create_skill("team-skill", "Alice's.")
    _skill_command(ALICE, "team-skill", capsys)

    status, out, err = _skill_command(BOB, "team-skill", capsys, action="unshare")

    assert (status, out) == (1, "")
    assert err == f"error: User {SUB_B} shares no skill named team-skill.\n"
    assert len(_skill_list()) == 1


def test_unshare_skill_clears_the_entry_of_a_skill_that_is_gone(stack, capsys):
    """An account removed from the identity provider leaves its store and its
    entry behind: unshare clears the entry by the old sub."""
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "Alice's.")
    _skill_command(ALICE, "team-skill", capsys)
    SkillService._rmtree_direct_child(alice.root, "team-skill")

    status, out, _ = _skill_command(ALICE, "team-skill", capsys, action="unshare")

    assert status == 0 and out.startswith("Stopped sharing skill team-skill")
    assert _skill_list() == []


def test_a_malformed_skill_list_is_read_as_empty_and_replaced(stack, capsys, caplog):
    SkillService(personal_scope(ALICE)).create_skill("team-skill", "Alice's.")
    list_file = shared_skills_file()
    list_file.parent.mkdir(parents=True)
    list_file.write_text('{"not": "a list"}')
    caplog.set_level(logging.WARNING, logger="cowork.shared_skills")

    status, _, _ = _skill_command(ALICE, "team-skill", capsys)

    assert status == 0
    assert _skill_list() == [{"user_id": SUB_A, "partition_id": ALICE.partition_id, "skill": "team-skill"}]
    assert any(
        record.name == "cowork.shared_skills" and str(list_file) in record.getMessage()
        for record in caplog.records
    )


def test_share_and_unshare_answer_busy_while_another_change_holds_the_list(stack, monkeypatch, capsys):
    monkeypatch.setattr(shared_skills, "SHARED_SKILLS_LOCK_TIMEOUT", 0.2)
    SkillService(personal_scope(ALICE)).create_skill("team-skill", "Alice's.")
    SkillService(personal_scope(BOB)).create_skill("bob-skill", "Bob's.")
    _skill_command(BOB, "bob-skill", capsys)
    written = shared_skills_file().read_bytes()

    with shared_skill_list_lock_held():
        for person, slug, action in ((ALICE, "team-skill", "share"), (BOB, "bob-skill", "unshare")):
            status, out, err = _skill_command(person, slug, capsys, action=action)
            assert (status, out) == (1, "")
            assert err == "error: Sharing is busy right now. Try again in a moment.\n"
    assert shared_skills_file().read_bytes() == written


def test_share_skill_refuses_where_skills_are_shared(stack, monkeypatch, capsys):
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "connections,memory,settings")
    get_app_settings.cache_clear()

    for action in ("share", "unshare"):
        status, out, err = _skill_command(ALICE, "team-skill", capsys, action=action)
        assert (status, out) == (1, "")
        assert err == f"error: {SKILL_SHARING_UNAVAILABLE}\n"
    assert not shared_skills_file().exists()


def test_share_and_unshare_refuse_root(stack, monkeypatch, capsys):
    """Root would create the list's folder or lock file as root, and every
    later write of Cowork's own would fail."""
    SkillService(personal_scope(ALICE)).create_skill("team-skill", "Alice's.")
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)

    for action in ("share", "unshare"):
        status, out, err = _skill_command(ALICE, "team-skill", capsys, action=action)
        assert (status, out) == (1, "")
        assert err == (
            "error: Run this as the user Cowork runs as, not root: Cowork itself writes "
            f"{shared_skills_file()}, and a folder or lock file root creates there would stop it.\n"
        )
    assert not shared_skills_file().parent.exists()


@pytest.mark.parametrize("option", [["--managed-dir", "/tmp"], ["--registry-stdout"]])
def test_skill_share_takes_no_managed_folder_or_registry_stdout(stack, capsys, option):
    for action in ("share", "unshare"):
        with pytest.raises(SystemExit) as refused:
            skills_ops.main([action, *_arguments(ALICE), "--skill", "team-skill", *option])
        assert refused.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err


def _snapshot_lines(out: str) -> tuple[dict[str, str], str]:
    lines = out.splitlines()
    skills = dict(line.split("  ", 1) for line in lines[1:-1])
    return skills, lines[-1]


def _manifest_digest(skills: dict[str, str], roots: dict[str, Path]) -> str:
    """sha256 over sha256sum's line for every file of every skill, by slug and
    path, leaving out stats.json, hidden files and links."""
    lines = []
    for slug, source in sorted(skills.items()):
        skill_dir = roots[source] / slug
        for path in sorted(skill_dir.rglob("*")):
            relative = path.relative_to(skill_dir).as_posix()
            if (
                not path.is_file()
                or path.is_symlink()
                or relative == "stats.json"
                or any(part.startswith(".") for part in relative.split("/"))
            ):
                continue
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {slug}/{relative}\n")
    return f"sha256 {hashlib.sha256(''.join(lines).encode()).hexdigest()}"


def test_skills_snapshot_lists_what_the_agent_sees_with_a_stable_hash(stack, capsys):
    from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR

    SkillService(personal_scope(ALICE)).create_skill("my-notes", "Mine.")
    (stack.skills_dir / "team-skill").mkdir()
    (stack.skills_dir / "team-skill" / "SKILL.md").write_text(
        "---\nname: team-skill\ndescription: Team.\n---\nTeam way.\n"
    )

    status, out, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
    assert status == 0
    assert out.startswith(f"Skills user {SUB_A}'s agent sees in project general, space {ALICE.partition_id}:\n")
    skills, digest_line = _snapshot_lines(out)
    assert skills["my-notes"] == "(own)"
    assert skills["team-skill"] == "(managed)"
    assert skills["connect-datasource"] == "(host)"
    assert all(skills[slug] == "(own)" for slug in BUILTIN_SKILL_SLUGS)

    # The hash is sha256 over sha256sum's lines for every file of each skill.
    own_root = SkillService(personal_scope(ALICE)).root
    from anton.core.memory.skills import SkillStore

    roots = {"(own)": own_root, "(host)": HOST_SKILLS_DIR, "(managed)": stack.skills_dir, "(anton)": SkillStore().builtin_root}
    assert digest_line == _manifest_digest(skills, roots)

    _, again, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
    assert again == out

    SkillService(personal_scope(ALICE)).update_skill("my-notes", instructions="Changed.")
    _, changed, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
    assert _snapshot_lines(changed)[1] != digest_line


def test_a_snapshot_follows_every_file_of_a_skill_but_its_usage(stack, capsys):
    """A helper script or reference file is part of what the agent reads, so
    changing one changes the hash; the store's usage sidecar and hidden files
    are not."""
    own = SkillService(personal_scope(ALICE)).root / "my-notes"
    SkillService(personal_scope(ALICE)).create_skill("my-notes", "Mine.")
    (own / "helper.py").write_text("print('v1')\n")
    managed = stack.skills_dir / "team-skill"
    managed.mkdir()
    (managed / "SKILL.md").write_text("---\nname: team-skill\ndescription: Team.\n---\nTeam way.\n")
    (managed / "refs").mkdir()
    (managed / "refs" / "ref.md").write_text("Reference v1.\n")

    def digest() -> str:
        _, out, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
        return _snapshot_lines(out)[1]

    first = digest()
    (own / "stats.json").write_text('{"uses": 3}')
    (own / ".cache").write_text("anything")
    assert digest() == first

    (own / "helper.py").write_text("print('v2')\n")
    after_helper = digest()
    assert after_helper != first
    (managed / "refs" / "ref.md").write_text("Reference v2.\n")
    assert digest() != after_helper


def test_a_snapshot_takes_a_managed_skill_over_a_same_named_own_one(stack, capsys):
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "My own version.")
    (stack.skills_dir / "team-skill").mkdir()
    (stack.skills_dir / "team-skill" / "SKILL.md").write_text(
        "---\nname: team-skill\ndescription: Team.\n---\nTeam way.\n"
    )

    _, out, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)

    assert _snapshot_lines(out)[0]["team-skill"] == "(managed)"


def test_a_snapshot_includes_the_skills_others_share_and_follows_their_edits(stack, capsys):
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "Do it the team's way.")
    alice.create_skill("private-skill", "Alice's alone.")
    _skill_command(ALICE, "team-skill", capsys)

    _, out, _ = _run(skills_ops.main, ["snapshot", *_arguments(BOB)], capsys)
    skills, digest_line = _snapshot_lines(out)
    assert skills["team-skill"] == "(shared)"
    assert "private-skill" not in skills

    # The shared skill's lines hash the files in Alice's own store.
    from anton.core.memory.skills import SkillStore

    from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR

    roots = {
        "(own)": SkillService(personal_scope(BOB)).root,
        "(host)": HOST_SKILLS_DIR,
        "(managed)": stack.skills_dir,
        "(shared)": alice.root,
        "(anton)": SkillStore().builtin_root,
    }
    assert digest_line == _manifest_digest(skills, roots)

    alice.update_skill("team-skill", instructions="The team's new way.")
    _, changed, _ = _run(skills_ops.main, ["snapshot", *_arguments(BOB)], capsys)
    assert _snapshot_lines(changed)[1] != digest_line


# ── connections ─────────────────────────────────────────────────────────────


def _share(person: Person, name: str, folder: Path, capsys, *, action: str = "share") -> tuple[int, str, str]:
    return _run(
        connections_ops.main,
        [action, *_arguments(person), "--name", name, "--managed-dir", str(folder)],
        capsys,
    )


def _registry(folder: Path) -> list[dict]:
    return json.loads((folder / "shared" / "connections.json").read_text())


def test_share_connection_adds_it_to_the_registry_for_everyone(stack, capsys):
    own_vault_for_scope(personal_scope(ALICE)).save(
        "postgres", "erp", {"host": "erp.example", "password": "erpSecret"}, secure_keys=["password"]
    )

    status, out, _ = _share(ALICE, "erp", stack.root, capsys)

    assert status == 0
    registry_file = stack.root / "shared" / "connections.json"
    assert out == (
        f"Shared connection postgres/erp of user {SUB_A} with everyone, read-only: "
        f"added it to {registry_file}.\n"
    )
    assert _registry(stack.root) == [
        {"user_id": SUB_A, "partition_id": ALICE.partition_id, "engine": "postgres", "name": "erp"}
    ]
    assert registry_file.stat().st_mode & 0o777 == 0o644
    # Nothing is copied: the operator's own connections stay as they were.
    assert LocalDataVault(stack.vault_dir).list_connections() == []
    bob_vault = vault_for_scope(personal_scope(BOB))
    assert not bob_vault.can_change("postgres", "erp")
    assert bob_vault.is_shared("postgres", "erp")
    assert bob_vault.load("postgres", "erp")["host"] == "erp.example"
    assert vault_for_scope(personal_scope(ALICE)).can_change("postgres", "erp")


def test_unshare_connection_removes_it_and_keeps_the_persons_own(stack, capsys):
    own_vault_for_scope(personal_scope(ALICE)).save("postgres", "erp", {"host": "erp.example"})
    own_vault_for_scope(personal_scope(BOB)).save("mysql", "crm", {"host": "crm.example"})
    _share(ALICE, "erp", stack.root, capsys)
    _share(BOB, "crm", stack.root, capsys)

    status, out, _ = _share(ALICE, "erp", stack.root, capsys, action="unshare")

    assert status == 0
    assert out == (
        f"Stopped sharing connection postgres/erp of user {SUB_A}: removed it from "
        f"{stack.root / 'shared' / 'connections.json'}. Their own connection stays in their vault.\n"
    )
    assert [(entry["engine"], entry["name"]) for entry in _registry(stack.root)] == [("mysql", "crm")]
    assert vault_for_scope(personal_scope(BOB)).read_record("postgres", "erp") is None
    assert own_vault_for_scope(personal_scope(ALICE)).read_record("postgres", "erp") is not None


def test_share_connection_refuses_an_unknown_or_ambiguous_name(stack, capsys):
    own = own_vault_for_scope(personal_scope(ALICE))
    own.save("postgres", "shared-name", {"host": "a"})
    own.save("mysql", "shared-name", {"host": "b"})

    for name, message in (("missing", "no saved connection named missing"), ("shared-name", "mysql, postgres")):
        status, out, err = _share(ALICE, name, stack.root, capsys)
        assert status == 1 and out == ""
        assert message in err
    assert not (stack.root / "shared" / "connections.json").exists()


def test_an_ambiguous_name_is_refused_naming_the_engine_option(stack, capsys):
    """One Google account connected to Gmail and to Calendar saves two
    connections under one name."""
    own = own_vault_for_scope(personal_scope(ALICE))
    own.save("gmail", "ada-example-com", {"account_email": "ada@example.com"})
    own.save("google_calendar", "ada-example-com", {"account_email": "ada@example.com"})

    status, out, err = _share(ALICE, "ada-example-com", stack.root, capsys)

    assert (status, out) == (1, "")
    assert "for gmail, google_calendar. Name one with --engine (ENGINE= for make)." in err
    assert not (stack.root / "shared" / "connections.json").exists()


def test_the_engine_option_shares_and_unshares_one_of_two(stack, capsys):
    own = own_vault_for_scope(personal_scope(ALICE))
    own.save("gmail", "ada-example-com", {"account_email": "ada@example.com"})
    own.save("google_calendar", "ada-example-com", {"account_email": "ada@example.com"})

    def run(action: str, *engine: str) -> tuple[int, str, str]:
        return _run(
            connections_ops.main,
            [action, *_arguments(ALICE), "--name", "ada-example-com", *engine, "--managed-dir", str(stack.root)],
            capsys,
        )

    assert run("share", "--engine", "gmail")[0] == 0
    assert run("share", "--engine", "google_calendar")[0] == 0
    assert [entry["engine"] for entry in _registry(stack.root)] == ["gmail", "google_calendar"]
    status, _, err = run("share", "--engine", "slack")
    assert status == 1 and "has no saved slack connection named ada-example-com" in err

    status, out, err = run("unshare")
    assert (status, out) == (1, "")
    assert "shares 2 connections named ada-example-com, for gmail, google_calendar" in err
    assert "--engine (ENGINE= for make)" in err

    status, out, _ = run("unshare", "--engine", "gmail")
    assert status == 0 and "connection gmail/ada-example-com" in out
    assert [entry["engine"] for entry in _registry(stack.root)] == ["google_calendar"]


def test_share_connection_refuses_one_already_shared_or_hidden(stack, capsys):
    own_vault_for_scope(personal_scope(ALICE)).save("postgres", "erp", {"host": "alice.example"})
    own_vault_for_scope(personal_scope(BOB)).save("postgres", "erp", {"host": "bob.example"})
    own_vault_for_scope(personal_scope(BOB)).save("postgres", "warehouse", {"host": "bob.example"})
    LocalDataVault(stack.vault_dir).save("postgres", "warehouse", {"host": "operator.example"})
    _share(ALICE, "erp", stack.root, capsys)

    for person, name, message in (
        (ALICE, "erp", f"Connection postgres/erp of user {SUB_A} is already shared."),
        (BOB, "erp", f"User {SUB_A} already shares a postgres connection named erp."),
        (BOB, "warehouse", "The managed folder already provides a postgres connection named warehouse."),
    ):
        status, out, err = _share(person, name, stack.root, capsys)
        assert (status, out) == (1, "")
        assert message in err
    assert len(_registry(stack.root)) == 1


def test_unshare_connection_refuses_a_name_the_person_shares_none_of(stack, capsys):
    own_vault_for_scope(personal_scope(ALICE)).save("postgres", "erp", {"host": "erp.example"})
    _share(ALICE, "erp", stack.root, capsys)

    status, out, err = _share(BOB, "erp", stack.root, capsys, action="unshare")

    assert (status, out) == (1, "")
    assert f"User {SUB_B} shares no connection named erp." in err
    assert len(_registry(stack.root)) == 1


def test_a_malformed_registry_is_never_rewritten(stack, capsys):
    own_vault_for_scope(personal_scope(ALICE)).save("postgres", "erp", {"host": "erp.example"})
    registry_file = stack.root / "shared" / "connections.json"
    registry_file.parent.mkdir()
    registry_file.write_text('{"not": "a list"}')

    for action in ("share", "unshare"):
        status, out, err = _share(ALICE, "erp", stack.root, capsys, action=action)
        assert (status, out) == (1, "")
        assert "is not a list of shared connections" in err
    assert registry_file.read_text() == '{"not": "a list"}'


def test_share_connection_refuses_where_connections_are_shared(stack, monkeypatch, capsys):
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "skills,memory,settings")
    get_app_settings.cache_clear()
    own_vault_for_scope(personal_scope(ALICE)).save("postgres", "erp", {"host": "erp.example"})

    status, out, err = _share(ALICE, "erp", stack.root, capsys)

    assert (status, out) == (1, "")
    assert "COWORK_PERSONAL_RESOURCES leaves connections shared" in err
    assert not (stack.root / "shared" / "connections.json").exists()


# ── --registry-stdout ───────────────────────────────────────────────────────


def test_registry_stdout_prints_the_bytes_and_writes_nothing(stack, capsys):
    """The stack's operator scripts check the connection registry and write
    it on the host, so with the flag the command writes nothing, stdout holds
    exactly what it would have written, and its message goes to stderr."""
    main, name_option, registry_file = connections_ops.main, "--name", stack.shared_connections_file
    own_vault_for_scope(personal_scope(ALICE)).save("postgres", "alice-item", {"host": "a"})
    own_vault_for_scope(personal_scope(BOB)).save("mysql", "bob-item", {"host": "b"})
    message = "Shared connection postgres/alice-item"

    def run(person: Person, action: str, *flag: str) -> tuple[int, str, str]:
        name = "alice-item" if person == ALICE else "bob-item"
        return _run(
            main,
            [action, *_arguments(person), name_option, name, "--managed-dir", str(stack.root), *flag],
            capsys,
        )

    # Bob's entry, written the ordinary way, is what Alice's command reads.
    assert run(BOB, "share")[0] == 0
    written = registry_file.read_bytes()
    before = _tree(stack.root)

    status, out, err = run(ALICE, "share", "--registry-stdout")

    assert status == 0
    assert _tree(stack.root) == before
    assert message in err and "the registry on stdout" in err
    printed = json.loads(out)
    assert [entry["user_id"] for entry in printed] == [SUB_B, SUB_A]
    # Exactly the bytes the command writes without the flag.
    assert run(ALICE, "share")[0] == 0
    assert out.encode() == registry_file.read_bytes() != written

    status, out, err = run(ALICE, "unshare", "--registry-stdout")
    assert status == 0
    assert out.encode() == written
    assert "Stopped sharing" in err


def test_registry_stdout_refusals_print_nothing_on_stdout(stack, capsys):
    status, out, err = _run(
        connections_ops.main,
        ["share", *_arguments(ALICE), "--name", "never-made", "--managed-dir", str(stack.root), "--registry-stdout"],
        capsys,
    )
    assert (status, out) == (1, "")
    assert "has no saved connection named never-made" in err
    assert not (stack.root / "shared").exists()


@pytest.mark.parametrize("module", ["cowork.ops.memory", "cowork.ops.skills", "cowork.ops.connections"])
def test_each_command_loads_every_model_it_queries_through(module):
    """The operator runs a command in a fresh interpreter, unlike these tests, which import every
    model. A relationship named by string resolves at the first query, so a command that loads
    only the models it queries stops there."""
    probe = f"import {module}\nfrom sqlalchemy.orm import configure_mappers\nconfigure_mappers()\n"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
