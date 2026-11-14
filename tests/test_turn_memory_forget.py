"""A cloud turn's request to remove saved memory (apply_turn_forget).

The pod holds a read-only copy of memory, so it only reports which saved lines
the model asked to remove; the server removes them through the caller's scope.
The caller's own tier always; the project tier only where a write to it passes
(_existing_slot_refusal), so a turn can't remove what its caller couldn't edit.
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import cowork.handlers.responses as responses_mod
import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.structured_logging import make_formatter
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.handlers.responses import ResponsesHandler
from cowork.models.shared_resource import SharedResourceMutation
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.services import memory as memory_mod
from cowork.services.memory import (
    apply_turn_forget,
    apply_turn_memory,
    build_turn_memory,
    normalize_memory_line,
)
from cowork.services.projects import ProjectService
from cowork.services.shared_resources import (
    PROJECT_MEMORY,
    SharedResourceAccess,
    project_memory_resource_key,
)
from cowork.turnqueue.remote_turn import remote_turn_events
from tests.test_remote_turn import _drain, _fake_handler
from tests.test_responses_remote_backend import _no_workspace_or_memory_units, _remote_handler


ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ADMIN = "33333333-3333-4333-8333-333333333333"


@pytest.fixture(autouse=True)
def _loggers_on(monkeypatch):
    """Turn the loggers these tests read back on for each test. A migration
    test that runs Alembic (cowork/db/alembic/env.py calls fileConfig, whose
    default disable_existing_loggers=True) leaves every logger built before
    it disabled for the rest of the session, and caplog then captures
    nothing from it."""
    for logger in (memory_mod.logger, responses_mod.logger, remote_turn_mod.logger):
        monkeypatch.setattr(logger, "disabled", False)


@pytest.fixture()
def org_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    get_app_settings.cache_clear()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    get_app_settings.cache_clear()


def _scope(user: str, org: str = ORG_A) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


def _scoped(engine, user: str, org: str = ORG_A) -> ScopedSession:
    return ScopedSession(Session(engine), _scope(user, org))


def _access(scoped: ScopedSession, *, admin: bool = False) -> SharedResourceAccess:
    scope = scoped.scope
    return SharedResourceAccess(
        scoped,
        Principal(
            user_id=scope.user_id,
            org_id=scope.org_id,
            email="member@example.com",
            roles=frozenset({ORG_MANAGE_ROLE}) if admin else frozenset(),
        ),
    )


def _global(scope: TenantScope) -> dict[str, str]:
    return build_turn_memory(scope).get("global", {})


def _project_rules(scope: TenantScope, project) -> str:
    return build_turn_memory(scope, project.path).get("project", {}).get("rules", "")


def _refusals(caplog) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == memory_mod.logger.name and record.levelno == logging.WARNING
    ]


# The matching rule anton's forget tool shares


# The form a saved entry is compared in, case by case. anton's
# tests/test_forget_memory.py holds this same table (_SHARED_FORM) for its
# normalize_entry_text; change both together.
_SHARED_FORM = [
    ("  Reply   in\tSpanish <!-- confidence:high source:user ts:2026-11-12 -->  ", "reply in spanish"),
    ("Reply in Spanish\n", "reply in spanish"),
    ("Straße", "strasse"),
    ("a <!-- b --> c", "a <!-- b --> c"),                  # only a tail is metadata
    ("- Reply in Spanish", "- reply in spanish"),          # an entry's own bullet stays
    ("<!-- ts:2026-11-12 -->", ""),
    ("Preferred chart color: teal (#008080).", "preferred chart color: teal (#008080)"),
    ("Preferred chart color: teal (#008080). <!-- confidence:high source:user ts:2026-11-14 -->",
     "preferred chart color: teal (#008080)"),
    ("Preferred chart color: teal (#008080)", "preferred chart color: teal (#008080)"),
    ("Reply in Spanish!", "reply in spanish"),
    ("Reply in Spanish?", "reply in spanish"),
    ("Reply in Spanish;", "reply in spanish"),
    ("Reply in Spanish:", "reply in spanish"),
    ("Reply in Spanish ?!.. ", "reply in spanish"),        # a run of it, spaces between
    ("Use v1.2 of the API: always.", "use v1.2 of the api: always"),
    ("Reply in Spanish,", "reply in spanish,"),            # a comma stays
    ("(see the docs.)", "(see the docs.)"),                # punctuation before a bracket stays
    ("Reply in Spanish…", "reply in spanish…"),            # so does an ellipsis character
    # Stripped, a note would end the form and a second pass would drop it.
    ("Reply in Spanish <!-- x --> .", "reply in spanish <!-- x --> ."),
    ("...", "..."),                                       # only punctuation keeps it
    ("Name: Ada", "name: ada"),
]


@pytest.mark.parametrize("text, form", _SHARED_FORM)
def test_normalize_memory_line_pins_the_shared_matching_rule(text, form):
    assert normalize_memory_line(text) == form
    # The pod sends an entry in this form, and it is normalized here once more.
    assert normalize_memory_line(form) == form


# anton's rule for a saved entry (normalize_entry_text in
# anton/core/memory/hippocampus.py, with _extract_metadata), restated here to
# pin cowork's to it.
_ANTON_TAIL = re.compile(r"\s*<!--([^>]*)-->\s*$")


def _anton_form(text: str) -> str:
    tail = _ANTON_TAIL.search(text)
    clean = text[: tail.start()].strip() if tail else text.strip()
    folded = " ".join(clean.split()).casefold()
    stripped = folded.rstrip(".!?;: ")
    return folded if not stripped or stripped.endswith("-->") else stripped


@pytest.mark.parametrize("text, form", _SHARED_FORM)
def test_anton_normalizes_an_entry_to_the_same_form(text, form):
    assert _anton_form(text) == form


# The caller's own tier


def test_a_global_rule_is_removed(org_engine):
    alice = _scope(ALICE)
    apply_turn_memory(alice, None, [
        {"text": "Reply in Spanish", "kind": "always", "scope": "global"},
        {"text": "Use metric units", "kind": "always", "scope": "global"},
    ])

    removed = apply_turn_forget(alice, None, [
        {"text": "  reply   IN spanish <!-- confidence:medium source:llm ts:2026-11-12 -->", "scope": "global"},
    ])

    assert removed == 1
    rules = _global(alice)["rules"]
    assert "Reply in Spanish" not in rules
    assert "Use metric units" in rules


def test_a_lesson_is_removed(org_engine):
    alice = _scope(ALICE)
    apply_turn_memory(alice, None, [
        {"text": "staging is read-only", "kind": "lesson", "scope": "global"},
        {"text": "the sales table is in EUR", "kind": "lesson", "scope": "global"},
    ])

    assert apply_turn_forget(alice, None, [{"text": "Staging is read-only", "scope": "global"}]) == 1

    lessons = _global(alice)["lessons"]
    assert "staging is read-only" not in lessons
    assert "the sales table is in EUR" in lessons


def test_a_profile_fact_is_removed(org_engine):
    alice = _scope(ALICE)
    apply_turn_memory(alice, None, [
        {"text": "Name: Zoran", "kind": "profile", "scope": "global"},
        {"text": "Role: analyst", "kind": "profile", "scope": "global"},
    ])

    assert apply_turn_forget(alice, None, [{"text": "Name: Zoran", "scope": "global"}]) == 1

    profile = _global(alice)["profile"]
    assert "Name: Zoran" not in profile
    assert "Role: analyst" in profile


@pytest.mark.parametrize("saved, kind, sent", [
    # The pod's form of the line the model sent without the saved period.
    ("Preferred chart color: teal (#008080).", "profile", "preferred chart color: teal (#008080)"),
    ("Label chart axes in English.", "always", "label chart axes in english"),
    ("Label chart axes in English", "always", "label chart axes in english."),
])
def test_an_entry_is_removed_whether_or_not_either_side_ends_in_a_period(org_engine, saved, kind, sent):
    alice = _scope(ALICE)
    apply_turn_memory(alice, None, [
        {"text": saved, "kind": kind, "scope": "global"},
        {"text": "Role: analyst", "kind": "profile", "scope": "global"},
    ])

    assert apply_turn_forget(alice, None, [{"text": sent, "scope": "global"}]) == 1

    memory = _global(alice)
    assert saved.rstrip(".") not in memory.get("profile", "") + memory.get("rules", "")
    assert "Role: analyst" in memory["profile"]


def test_two_entries_that_differ_only_in_a_final_period_are_both_removed(org_engine):
    """They say the same thing, so a person who asks to forget it wants neither
    kept. One request removes both, as it does two entries that differ only in
    case or spacing."""
    alice = _scope(ALICE)
    apply_turn_memory(alice, None, [
        {"text": "Use teal for bars.", "kind": "always", "scope": "global"},
        {"text": "Use teal for bars", "kind": "always", "scope": "global"},
        {"text": "Use metric units", "kind": "always", "scope": "global"},
    ])
    assert _global(alice)["rules"].count("Use teal for bars") == 2

    assert apply_turn_forget(alice, None, [{"text": "use teal for bars", "scope": "global"}]) == 2

    rules = _global(alice)["rules"]
    assert "teal" not in rules
    assert "Use metric units" in rules


def test_an_entry_that_matches_nothing_removes_nothing(org_engine, caplog):
    """No substring and no near match: only the whole normalized line."""
    alice = _scope(ALICE)
    apply_turn_memory(alice, None, [{"text": "Reply in Spanish", "kind": "always", "scope": "global"}])
    before = _global(alice)

    with caplog.at_level(logging.INFO):
        removed = apply_turn_forget(alice, None, [
            {"text": "Reply", "scope": "global"},
            {"text": "Reply in Spanish please", "scope": "global"},
            {"text": "Reply in Spanish", "scope": "project"},
        ])

    assert removed == 0
    assert _global(alice) == before
    # The project tier was named but the turn has no project.
    assert [(r.getMessage(), r.count) for r in _refusals(caplog)] == [
        ("Dropped remote-turn project memory removals: the turn has no project", 1),
    ]


def test_a_malformed_entry_is_skipped(org_engine, caplog):
    """The pod's payload is untrusted: an entry that does not parse costs only
    itself, and the valid entry after it still removes its line. One WARNING
    counts the skipped entries, so a dropped removal shows in the log."""
    alice = _scope(ALICE)
    apply_turn_memory(alice, None, [
        {"text": "Reply in Spanish", "kind": "always", "scope": "global"},
        {"text": "Use metric units", "kind": "always", "scope": "global"},
    ])

    with caplog.at_level(logging.WARNING, logger=memory_mod.logger.name):
        removed = apply_turn_forget(alice, None, [
            "not-a-dict",
            {"scope": "global"},                                   # no text
            {"text": "   ", "scope": "global"},                    # blank
            {"text": 42, "scope": "global"},                       # not text
            {"text": "Use metric units", "scope": "../../etc"},    # unknown scope
            {"text": "Use metric units", "scope": None},
            {"text": "<!-- ts:2026-11-12 -->", "scope": "global"},  # only a tail
            {"text": "Reply in Spanish", "scope": "global", "kind": "forget"},  # extra keys ignored
        ])

    assert removed == 1
    rules = _global(alice)["rules"]
    assert "Reply in Spanish" not in rules
    assert "Use metric units" in rules
    assert [(r.getMessage(), r.event, r.count) for r in _refusals(caplog)] == [
        ("Dropped remote-turn memory removals that name no saved entry", "turn_memory_forget_dropped", 7),
    ]


@pytest.mark.parametrize(
    "saved_by",
    ["a cloud turn, whose casefold grows it", "the Memories page, which takes any length"],
)
def test_a_long_entry_is_removed(org_engine, caplog, saved_by):
    """anton sends the saved entry's text casefolded, which can be longer than
    the 20,000 characters a cloud turn may save ("ß" becomes "ss"), and the
    Memories page saves a line of any length. A removal writes nothing, so its
    text has no bound, and a long entry is removed like any other."""
    from cowork.harnesses.memory.registry import MemorySlot
    from cowork.harnesses.memory.store import GlobalMemoryStore

    alice = _scope(ALICE)
    if saved_by.startswith("a cloud turn"):
        text = "Keep " + "ß" * 10_500
        assert apply_turn_memory(alice, None, [{"text": text, "kind": "always", "scope": "global"}]) == 1
    else:
        text = "Keep " + "x" * 70_000
        apply_turn_memory(alice, None, [{"text": "Use metric units", "kind": "always", "scope": "global"}])
        store = GlobalMemoryStore(scope=alice)
        store.write(MemorySlot.RULES, store.read(MemorySlot.RULES).replace("Use metric units", text))
    sent = normalize_memory_line(text)
    assert len(sent) > 20_000

    with caplog.at_level(logging.WARNING, logger=memory_mod.logger.name):
        removed = apply_turn_forget(alice, None, [{"text": sent, "scope": "global"}])

    assert removed == 1
    assert "Keep" not in _global(alice).get("rules", "")
    assert _refusals(caplog) == []


def test_another_orgs_memory_is_untouched(org_engine):
    """The scope decides whose memory changes, never the payload: the same
    person id in another org keeps the same line."""
    in_a, in_b = _scope(ALICE, ORG_A), _scope(ALICE, ORG_B)
    for scope in (in_a, in_b):
        apply_turn_memory(scope, None, [{"text": "Reply in Spanish", "kind": "always", "scope": "global"}])
    b_project = ProjectService(_scoped(org_engine, ALICE, ORG_B)).create_project("b-project")
    apply_turn_memory(
        in_b, b_project.path,
        [{"text": "Reply in Spanish", "kind": "always", "scope": "project"}],
        access=_access(_scoped(org_engine, ALICE, ORG_B)), project_id=b_project.id,
    )

    assert apply_turn_forget(in_a, None, [{"text": "Reply in Spanish", "scope": "global"}]) == 1
    # Org A's turn can't name org B's project: its session does not find it.
    a_session = _scoped(org_engine, ALICE, ORG_A)
    with pytest.raises(ValueError, match="not found"):
        apply_turn_forget(
            in_a, b_project.path, [{"text": "Reply in Spanish", "scope": "project"}],
            access=_access(a_session), project_id=b_project.id,
        )

    assert "Reply in Spanish" not in _global(in_a).get("rules", "")
    assert "Reply in Spanish" in _global(in_b)["rules"]
    assert "Reply in Spanish" in _project_rules(in_b, b_project)


# The project tier, under the gate a write passes


def _alices_project_rule(engine, text: str = "Deploy on green only"):
    alice = _scoped(engine, ALICE)
    project = ProjectService(alice).create_project("shared-project")
    assert apply_turn_memory(
        alice.scope, project.path, [{"text": text, "kind": "always", "scope": "project"}],
        access=_access(alice), project_id=project.id,
    ) == 1
    return project


def test_a_project_rule_is_removed_by_its_author(org_engine, caplog):
    project = _alices_project_rule(org_engine)
    alice = _scoped(org_engine, ALICE)
    # The same line in Alice's own tier stays: the request named the project.
    apply_turn_memory(alice.scope, None, [{"text": "Deploy on green only", "kind": "always", "scope": "global"}])

    with caplog.at_level(logging.INFO):
        removed = apply_turn_forget(
            alice.scope, project.path, [{"text": "deploy on GREEN only", "scope": "project"}],
            access=_access(alice), project_id=project.id,
        )

    assert removed == 1
    assert "Deploy on green only" not in _project_rules(alice.scope, project)
    assert "Deploy on green only" in _global(alice.scope)["rules"]
    assert _refusals(caplog) == []
    key = project_memory_resource_key(project.id, "rules")
    updates = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == PROJECT_MEMORY,
            SharedResourceMutation.resource_key == key,
            SharedResourceMutation.action == "update",
        )
    ).all()
    assert [update.actor_id for update in updates] == [ALICE]


def test_a_project_rule_is_refused_for_a_caller_the_write_gate_refuses(org_engine, caplog):
    """Bob is a member but not the slot's author or an admin, so the write
    gate drops his project entry; the removal gate refuses his request the
    same way, with one WARNING naming the slot, and the rule stays."""
    project = _alices_project_rule(org_engine)
    bob = _scoped(org_engine, BOB)
    key = project_memory_resource_key(project.id, "rules")

    with caplog.at_level(logging.INFO):
        written = apply_turn_memory(
            bob.scope, project.path, [{"text": "Bob's rule", "kind": "always", "scope": "project"}],
            access=_access(bob), project_id=project.id,
        )
        removed = apply_turn_forget(
            bob.scope, project.path,
            [
                {"text": "Deploy on green only", "scope": "project"},
                {"text": "Nothing like this is saved", "scope": "project"},
            ],
            access=_access(bob), project_id=project.id,
        )

    assert (written, removed) == (0, 0)
    assert "Deploy on green only" in _project_rules(bob.scope, project)
    dropped_write, refused = _refusals(caplog)
    assert dropped_write.getMessage() == (
        f"Dropped a remote-turn project memory entry for {key}: the turn's actor is not its author"
    )
    assert refused.getMessage() == "Refused a remote-turn project memory removal"
    # The deployment's JSON lines keep the message template and the allowed
    # fields only, so the slot and the reason ride in those fields.
    logged = json.loads(make_formatter().format(refused))
    assert {k: logged[k] for k in ("event", "reason", "slot", "project_id")} == {
        "event": "turn_memory_forget_refused",
        "reason": "the turn's actor is not its author",
        "slot": "rules",
        "project_id": str(project.id),
    }

    # An admin passes the write gate, and so passes this one.
    admin = _scoped(org_engine, ADMIN)
    assert apply_turn_forget(
        admin.scope, project.path, [{"text": "Deploy on green only", "scope": "project"}],
        access=_access(admin, admin=True), project_id=project.id,
    ) == 1
    assert "Deploy on green only" not in _project_rules(admin.scope, project)


def test_a_project_removal_needs_the_turns_trusted_identity(org_engine, caplog):
    """A bare tenant scope is no ownership proof for a write, nor for a removal."""
    project = _alices_project_rule(org_engine)
    alice = _scoped(org_engine, ALICE)

    with caplog.at_level(logging.INFO):
        removed = apply_turn_forget(
            alice.scope, project.path, [{"text": "Deploy on green only", "scope": "project"}],
        )

    assert removed == 0
    assert "Deploy on green only" in _project_rules(alice.scope, project)
    assert [(r.getMessage(), r.count) for r in _refusals(caplog)] == [
        ("Dropped remote-turn project memory removals: the turn carries no trusted organization identity", 1),
    ]


def test_a_desktop_turn_removes_a_project_lesson(org_engine, tmp_path):
    """Outside org tenancy the project tier has no authorship gate, as for a
    write (apply_turn_memory)."""
    project = tmp_path / "desktop-project"
    (project / ".anton" / "memory").mkdir(parents=True)
    apply_turn_memory(LOCAL_SCOPE, str(project), [
        {"text": "the API is paginated", "kind": "lesson", "scope": "project"},
    ])

    assert apply_turn_forget(LOCAL_SCOPE, str(project), [{"text": "The API is paginated", "scope": "project"}]) == 1
    assert "paginated" not in build_turn_memory(LOCAL_SCOPE, str(project)).get("project", {}).get("lessons", "")


def test_an_entry_is_removed_only_from_the_tier_it_names(org_engine, tmp_path):
    """anton sends each entry with the tier the pod found it in, "global" or
    "project" (MemoryForgetEntry in anton/cloud_turn/contract.py). An entry
    with no scope, or any other one such as "any", is skipped and removes the
    line from neither tier; one that names a tier keeps the other tier's
    copy of the same line."""
    project = tmp_path / "desktop-project"
    (project / ".anton" / "memory").mkdir(parents=True)
    apply_turn_memory(LOCAL_SCOPE, str(project), [
        {"text": "Reply in Spanish", "kind": "always", "scope": "global"},
        {"text": "Reply in Spanish", "kind": "always", "scope": "project"},
    ])

    def rules() -> tuple[str, str]:
        memory = build_turn_memory(LOCAL_SCOPE, str(project))
        return memory.get("global", {}).get("rules", ""), memory.get("project", {}).get("rules", "")

    assert apply_turn_forget(LOCAL_SCOPE, str(project), [
        {"text": "Reply in Spanish", "scope": "any"},
        {"text": "Reply in Spanish"},
    ]) == 0
    assert all("Reply in Spanish" in tier for tier in rules())

    assert apply_turn_forget(LOCAL_SCOPE, str(project), [{"text": "Reply in Spanish", "scope": "project"}]) == 1
    global_rules, project_rules = rules()
    assert ("Reply in Spanish" in global_rules, "Reply in Spanish" in project_rules) == (True, False)


def test_the_forget_entry_accepts_what_antons_contract_sends():
    """The wire model mirrors anton's MemoryForgetEntry: the same fields, the
    same scopes, and a scope it can't do without. Skipped while the installed
    anton sends no memory_forget."""
    from typing import get_args

    contract = pytest.importorskip("anton.cloud_turn.contract")
    entry = getattr(contract, "MemoryForgetEntry", None)
    if entry is None:
        pytest.skip("the installed anton has no MemoryForgetEntry")
    wire = memory_mod._WireForget.model_fields
    antons = entry.model_fields

    assert set(wire) == set(antons)
    assert set(get_args(wire["scope"].annotation)) == set(get_args(antons["scope"].annotation))
    assert {name: field.is_required() for name, field in wire.items()} == {
        name: field.is_required() for name, field in antons.items()
    }


# The wire: where the pod's memory_forget event is consumed


@pytest.fixture()
def forget_fakes(monkeypatch):
    """The collaborators of ``_persist_turn_memory_forget`` as fakes that
    record what reached apply_turn_forget, and in which order."""
    import cowork.services.shared_resources as shared_resources

    project_id = uuid4()
    project = SimpleNamespace(id=project_id, path="/projects/one", name="one")
    conversation = SimpleNamespace(project_id=project_id)
    events: list = []

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return conversation

    class FakeSession:
        scope = SimpleNamespace(org_mode=True, org_id="org-123", user_id="user-456")

        def get(self, model, ident):
            return project

        def refresh(self, row):
            pass

    class FakeAccess:
        def __init__(self, session, principal):
            pass

        @contextmanager
        def coordination_lock(self, kind, key):
            events.append("lock-enter")
            yield
            events.append("lock-exit")

    def fake_forget(scope, path, entries, **kwargs):
        events.append(("forget", path, entries, kwargs["project_id"]))
        return len(entries)

    def no_write(*args, **kwargs):
        raise AssertionError("a forget request must not write memory")

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "apply_turn_forget", fake_forget)
    monkeypatch.setattr(responses_mod, "apply_turn_memory", no_write)
    monkeypatch.setattr(shared_resources, "SharedResourceAccess", FakeAccess)
    return SimpleNamespace(session=FakeSession(), events=events, project=project)


def test_persist_forget_removes_under_the_project_lock(monkeypatch, forget_fakes, caplog):
    monkeypatch.setattr(
        responses_mod, "get_user_settings",
        lambda scope=None: SimpleNamespace(memory_enabled=True, memory_mode="autopilot"),
    )
    entries = [{"text": "Reply in Spanish", "scope": "global"}]
    conv_id = uuid4()

    with caplog.at_level(logging.INFO):
        ResponsesHandler._persist_turn_memory_forget(forget_fakes.session, conv_id, entries, object())

    assert forget_fakes.events == [
        "lock-enter", ("forget", "/projects/one", entries, forget_fakes.project.id), "lock-exit",
    ]
    logged = [
        (r.getMessage(), r.count, r.conversation_id) for r in caplog.records
        if r.name == responses_mod.logger.name and r.levelno == logging.INFO
    ]
    assert logged == [
        ("[responses] removed the memory entries a remote turn asked to forget", 1, str(conv_id)),
    ]


@pytest.mark.parametrize(
    "settings",
    [
        SimpleNamespace(memory_enabled=True, memory_mode="off"),
        SimpleNamespace(memory_enabled=False, memory_mode="autopilot"),
    ],
    ids=["mode-off", "memory-disabled"],
)
def test_memory_off_removes_nothing(monkeypatch, forget_fakes, settings, caplog):
    """While memory is off what is stored stays as it is, as the Memories
    page refuses a change then; a removal is a change."""
    monkeypatch.setattr(responses_mod, "get_user_settings", lambda scope=None: settings)
    conv_id = uuid4()

    with caplog.at_level(logging.INFO, logger=responses_mod.logger.name):
        ResponsesHandler._persist_turn_memory_forget(
            forget_fakes.session, conv_id, [{"text": "x"}, {"text": "y"}], object(),
        )

    assert forget_fakes.events == ["lock-enter", "lock-exit"]
    assert _dropped_line(caplog) == {
        "message": _READ_ONLY_LINE,
        "count": 2,
        "conversation_id": str(conv_id),
    }


def test_a_comparison_side_removes_nothing(monkeypatch, forget_fakes, caplog):
    monkeypatch.setattr(
        responses_mod, "get_user_settings",
        lambda scope=None: SimpleNamespace(memory_enabled=True, memory_mode="autopilot"),
    )
    monkeypatch.setattr(responses_mod, "is_comparison_sandbox", lambda name: True)
    conv_id = uuid4()

    with caplog.at_level(logging.INFO, logger=responses_mod.logger.name):
        ResponsesHandler._persist_turn_memory_forget(forget_fakes.session, conv_id, [{"text": "x"}], object())

    assert forget_fakes.events == ["lock-enter", "lock-exit"]
    assert _dropped_line(caplog) == {
        "message": _READ_ONLY_LINE,
        "count": 1,
        "conversation_id": str(conv_id),
    }


_READ_ONLY_LINE = (
    "[responses] memory is read-only in this conversation; dropped what a remote turn asked to save or remove"
)


def _dropped_line(caplog) -> dict:
    """The one INFO line the handler logged, as the hosted JSON log prints
    it: a fixed message, with the count and the conversation as fields."""
    [record] = [
        r for r in caplog.records if r.name == responses_mod.logger.name and r.levelno == logging.INFO
    ]
    line = json.loads(make_formatter().format(record))
    return {key: line.get(key) for key in ("message", "count", "conversation_id")}


@pytest.mark.parametrize("forget_first", [True, False], ids=["pod-order", "reversed"])
async def test_a_web_turn_applies_the_pods_forget_request(monkeypatch, forget_first):
    """Web and scheduled turns both stream through _produce_remote. The forget
    reply goes to the removal, the memory reply to the write, each with the
    turn's principal, in the order they arrive. anton sends memory_forget
    before memory, so the removal applies to memory as the turn found it and
    a line the turn saves again stays saved."""
    handler = _remote_handler(monkeypatch, {})
    _no_workspace_or_memory_units(monkeypatch)
    monkeypatch.setattr(ResponsesHandler, "_remote_artifacts_context", staticmethod(lambda session, conv_id: None))
    calls: list = []
    monkeypatch.setattr(
        ResponsesHandler, "_persist_turn_memory",
        staticmethod(lambda session, conv_id, entries, principal: calls.append(("memory", entries, principal))),
    )
    monkeypatch.setattr(
        ResponsesHandler, "_persist_turn_memory_forget",
        staticmethod(lambda session, conv_id, entries, principal: calls.append(("forget", entries, principal))),
    )
    remember = [{"text": "Use metric units", "kind": "always", "scope": "global"}]
    forget = [{"text": "Reply in Spanish", "scope": "global"}]

    replies = [("turn_memory_forget", {"entries": forget}), ("turn_memory", {"entries": remember})]
    if not forget_first:
        replies.reverse()

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "Done."}
        for reply in replies:
            yield reply
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)
    frames: list[str] = []

    class RecBuffer:
        async def append(self, kind, record):
            frames.append(record["sse"])

        async def close(self, reason):
            frames.append(f"CLOSE:{reason}")

    await handler._produce_remote(
        conv_id=uuid4(), input_text="forget that", original_content="forget that",
        model="anton", harness_id="anton", buffer=RecBuffer(),
    )

    applied = [("forget", forget, handler.principal), ("memory", remember, handler.principal)]
    assert calls == (applied if forget_first else applied[::-1])
    assert frames[-1] == "CLOSE:completed"


async def test_a_channel_turn_applies_the_pods_forget_request(monkeypatch):
    """Outside org tenancy a channel turn changes memory with no principal,
    the removal and the write in the order the pod sent them."""
    calls: list = []
    _fake_handler(
        monkeypatch,
        persist_turn_memory=lambda session, conv_id, entries, principal: calls.append(("memory", entries, principal)),
    )
    monkeypatch.setattr(
        responses_mod.ResponsesHandler, "_persist_turn_memory_forget",
        staticmethod(lambda session, conv_id, entries, principal: calls.append(("forget", entries, principal))),
    )
    forget = [{"text": "Reply in Spanish", "scope": "global"}]
    remember = [{"text": "Use metric units", "kind": "always", "scope": "global"}]

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_memory_forget", {"entries": forget}
        yield "turn_memory", {"entries": remember}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    _events, failure = await _drain(remote_turn_events(
        scope=LOCAL_SCOPE, conv_id=uuid4(), org_id=None, user_id=None,
        input_text="forget that", model="anton", turn_rows=[],
    ))

    assert failure is None
    assert calls == [("forget", forget, None), ("memory", remember, None)]


async def test_an_org_channel_turn_removes_nothing_and_says_so_at_info(monkeypatch, caplog):
    """An organization's channel turn has no member, so neither tier takes a
    removal: the same INFO line as for its memory, and no removal."""
    calls: list = []
    _fake_handler(monkeypatch)
    monkeypatch.setattr(
        responses_mod.ResponsesHandler, "_persist_turn_memory_forget",
        staticmethod(lambda session, conv_id, entries, principal: calls.append(entries)),
    )

    async def fake_replies(**kwargs):
        yield "turn_memory_forget", {"entries": [{"text": "Reply in Spanish"}]}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    with caplog.at_level(logging.INFO):
        _events, failure = await _drain(remote_turn_events(
            scope=TenantScope(org_mode=True, org_id="org-123"), conv_id=uuid4(), org_id="org-123",
            user_id=None, input_text="forget that", model="anton", turn_rows=[],
        ))

    assert failure is None
    assert calls == []
    assert [
        r.getMessage() for r in caplog.records
        if r.name == remote_turn_mod.logger.name and r.levelno == logging.INFO
    ] == ["channel turn memory not saved: an organization's channel turn has no member"]
