"""The operator commands, ``python -m cowork.ops.<module>``, on two people.

Each command acts in one person's space, worked out from the pair the proxy
sends: memory print shows what that person's agent learned and nothing of
the other's, and after memory clear it shows nothing learned. skills share
moves a person's skill into the managed folder, where the author loses edit
like everyone else. skills snapshot lists what the person's agent sees, with
a sha256 that stays the same while the skills do. connections share copies a
person's connection into the managed folder for everyone.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException
from sqlalchemy import delete

from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession
from cowork.db.session import get_open_session
from cowork.harnesses.memory.registry import MemorySlot
from cowork.managed import MANAGED_SKILL_REFUSAL
from cowork.models.project import Project
from cowork.ops import Person, personal_scope
from cowork.ops import connections as connections_ops
from cowork.ops import memory as memory_ops
from cowork.ops import skills as skills_ops
from cowork.schemas.memory import MemoryScope
from cowork.schemas.skills import SkillUpdateRequest
from cowork.services.connectors.persist import own_vault_for_scope, vault_for_scope
from cowork.services.memory import MemoryService
from cowork.services.skills import BUILTIN_SKILL_SLUGS, SkillService

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


def test_memory_commands_refuse_when_memory_is_shared(stack, monkeypatch, capsys):
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "skills,connections")
    get_app_settings.cache_clear()
    for action in ("print", "clear"):
        status, out, err = _run(memory_ops.main, [action, *_arguments(ALICE)], capsys)
        assert status == 1 and out == ""
        assert "keeps memory shared" in err


def test_commands_refuse_on_a_deployment_that_trusts_no_identity(stack, monkeypatch, capsys):
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    get_app_settings.cache_clear()
    status, out, err = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
    assert status == 1 and out == ""
    assert "COWORK_TRUST_IDENTITY_HEADERS is off" in err


def test_a_person_id_that_is_not_a_uuid_is_refused(stack, capsys):
    with pytest.raises(SystemExit) as refused:
        memory_ops.main(["print", "--organization-id", DEPLOYMENT, "--user-id", "not-a-uuid"])
    assert refused.value.code == 2
    assert "is not a UUID" in capsys.readouterr().err


# ── skills ──────────────────────────────────────────────────────────────────


def test_share_skill_moves_it_and_its_author_loses_edit(stack, capsys):
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "Do it the team's way.")

    status, out, _ = _run(
        skills_ops.main,
        ["share", *_arguments(ALICE), "--skill", "team-skill", "--managed-dir", str(stack.skills_dir)],
        capsys,
    )

    assert status == 0
    assert out == (
        f"Moved skill team-skill of user {SUB_A} into {stack.skills_dir / 'team-skill'}. "
        "The person's own copy is deleted.\n"
    )
    assert "Do it the team's way." in (stack.skills_dir / "team-skill" / "SKILL.md").read_text()
    assert not (alice.root / "team-skill").exists()
    assert [entry.name for entry in stack.skills_dir.iterdir()] == ["team-skill"]

    for person in (ALICE, BOB):
        raw = get_open_session()
        try:
            scoped = ScopedSession(raw, personal_scope(person))
            listed = {s["id"]: s for s in skills_ep.list_skills(scoped, None)["skills"]}
            assert listed["team-skill"]["isBuiltin"] is True
            assert listed["team-skill"]["capabilities"]["canEdit"] is False
            with pytest.raises(HTTPException) as refused:
                skills_ep.update_skill("team-skill", SkillUpdateRequest(instructions="mine now"), scoped, None)
            assert refused.value.status_code == 403 and refused.value.detail == MANAGED_SKILL_REFUSAL
        finally:
            raw.close()


@pytest.mark.parametrize(
    ("slug", "message"),
    [
        (sorted(BUILTIN_SKILL_SLUGS)[0], "packaged built-in"),
        ("never-made", "has no skill named never-made"),
    ],
)
def test_share_skill_refuses_what_it_cannot_move(stack, capsys, slug, message):
    SkillService(personal_scope(ALICE)).ensure_builtin_skills()
    status, out, err = _run(
        skills_ops.main,
        ["share", *_arguments(ALICE), "--skill", slug, "--managed-dir", str(stack.skills_dir)],
        capsys,
    )
    assert status == 1 and out == ""
    assert message in err
    assert list(stack.skills_dir.iterdir()) == []


def test_share_skill_refuses_a_skill_holding_a_link(stack, capsys, tmp_path):
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "Do it.")
    secret = tmp_path / "secret.txt"
    secret.write_text("not for everyone")
    (alice.root / "team-skill" / "notes.txt").symlink_to(secret)

    status, _, err = _run(
        skills_ops.main,
        ["share", *_arguments(ALICE), "--skill", "team-skill", "--managed-dir", str(stack.skills_dir)],
        capsys,
    )

    assert status == 1 and "holds a link" in err
    assert (alice.root / "team-skill" / "SKILL.md").exists()
    assert list(stack.skills_dir.iterdir()) == []


def _snapshot_lines(out: str) -> tuple[dict[str, str], str]:
    lines = out.splitlines()
    skills = dict(line.split("  ", 1) for line in lines[1:-1])
    return skills, lines[-1]


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

    # The hash is sha256 over sha256sum's lines for each SKILL.md, by slug.
    own_root = SkillService(personal_scope(ALICE)).root
    from anton.core.memory.skills import SkillStore

    roots = {"(own)": own_root, "(host)": HOST_SKILLS_DIR, "(managed)": stack.skills_dir, "(anton)": SkillStore().builtin_root}
    manifest = "".join(
        f"{hashlib.sha256((roots[source] / slug / 'SKILL.md').read_bytes()).hexdigest()}  {slug}/SKILL.md\n"
        for slug, source in sorted(skills.items())
    )
    assert digest_line == f"sha256 {hashlib.sha256(manifest.encode()).hexdigest()}"

    _, again, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
    assert again == out

    SkillService(personal_scope(ALICE)).update_skill("my-notes", instructions="Changed.")
    _, changed, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)
    assert _snapshot_lines(changed)[1] != digest_line


def test_a_snapshot_takes_a_managed_skill_over_a_same_named_own_one(stack, capsys):
    alice = SkillService(personal_scope(ALICE))
    alice.create_skill("team-skill", "My own version.")
    (stack.skills_dir / "team-skill").mkdir()
    (stack.skills_dir / "team-skill" / "SKILL.md").write_text(
        "---\nname: team-skill\ndescription: Team.\n---\nTeam way.\n"
    )

    _, out, _ = _run(skills_ops.main, ["snapshot", *_arguments(ALICE)], capsys)

    assert _snapshot_lines(out)[0]["team-skill"] == "(managed)"


# ── connections ─────────────────────────────────────────────────────────────


def test_share_connection_copies_it_for_everyone(stack, capsys):
    own_vault_for_scope(personal_scope(ALICE)).save(
        "postgres", "erp", {"host": "erp.example", "password": "erpSecret"}, secure_keys=["password"]
    )

    status, out, _ = _run(
        connections_ops.main,
        ["share", *_arguments(ALICE), "--name", "erp", "--managed-dir", str(stack.vault_dir)],
        capsys,
    )

    assert status == 0
    assert out.startswith(f"Copied connection postgres/erp of user {SUB_A} into {stack.vault_dir}")
    copied = LocalDataVault(stack.vault_dir).read_record("postgres", "erp")
    assert copied["fields"]["password"] == "erpSecret" and copied["secure_keys"] == ["password"]
    bob_vault = vault_for_scope(personal_scope(BOB))
    assert bob_vault.is_managed("postgres", "erp")
    assert bob_vault.load("postgres", "erp")["host"] == "erp.example"
    # The author's own entry stays in their vault, behind the managed one.
    assert own_vault_for_scope(personal_scope(ALICE)).read_record("postgres", "erp") is not None


def test_share_connection_refuses_an_unknown_or_ambiguous_name(stack, capsys):
    own = own_vault_for_scope(personal_scope(ALICE))
    own.save("postgres", "shared-name", {"host": "a"})
    own.save("mysql", "shared-name", {"host": "b"})

    for name, message in (("missing", "no saved connection named missing"), ("shared-name", "mysql, postgres")):
        status, out, err = _run(
            connections_ops.main,
            ["share", *_arguments(ALICE), "--name", name, "--managed-dir", str(stack.vault_dir)],
            capsys,
        )
        assert status == 1 and out == ""
        assert message in err
    assert LocalDataVault(stack.vault_dir).list_connections() == []
