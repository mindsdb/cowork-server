"""Cross-tenant behaviour of the swept schedule services.

Schedules are a root table (own org_id) → direct org filtering. ScheduleRun
is a child (no org_id) → the two request-reachable run methods anchor on the
parent schedule being visible in scope. The scheduler loop runs under
SYSTEM_SCOPE (unscoped executor) and must still see every schedule.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService, ScheduleService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _scope(org: str, user: str = "user-1") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


@pytest.fixture()
def engine(monkeypatch):
    get_app_settings.cache_clear()
    import cowork.models.message, cowork.models.message_event  # noqa: F401
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    with Session(eng) as seed:
        seed.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/tmp/general"))
        seed.commit()
    yield eng
    get_app_settings.cache_clear()


def _svc(engine, scope: TenantScope) -> ScheduleService:
    return ScheduleService(ScopedSession(Session(engine), scope))


def _make(svc: ScheduleService) -> object:
    return svc.create_schedule(
        title="daily report", prompt="do it", cadence="daily",
        next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc), model="sonnet",
    )


def test_create_stamps_org_and_creator(engine):
    sched = _make(_svc(engine, _scope(ORG_A, "alice")))
    assert sched.org_id == ORG_A
    assert sched.created_by == "alice"


def test_other_org_cannot_see_or_touch(engine):
    a = _svc(engine, _scope(ORG_A))
    b = _svc(engine, _scope(ORG_B))
    sched = _make(a)

    assert sched.id not in {s.id for s in b.list_schedules()}
    with pytest.raises(ValueError, match="not found"):
        b.get_schedule(sched.id)
    with pytest.raises(ValueError, match="not found"):
        b.update_schedule(sched.id, title="stolen")
    with pytest.raises(ValueError, match="not found"):
        b.pause_schedule(sched.id)
    assert b.delete_schedule(sched.id) is False  # nonexistent-shaped
    assert a.get_schedule(sched.id).title == "daily report"  # untouched


def test_runs_are_anchored_to_parent_schedule(engine):
    a = _svc(engine, _scope(ORG_A))
    sched = _make(a)
    # a run exists under org A's schedule
    a_runs = ScheduleRunService(ScopedSession(Session(engine), _scope(ORG_A)))
    a_runs.create_run(sched.id)

    # org B, whose scope can't see the parent schedule, sees no runs / no active
    b_runs = ScheduleRunService(ScopedSession(Session(engine), _scope(ORG_B)))
    assert b_runs.list_runs(sched.id) == []
    assert b_runs.has_active_run(sched.id) is False
    # org A sees them
    assert len(a_runs.list_runs(sched.id)) == 1
    assert a_runs.has_active_run(sched.id) is True


def test_system_scope_sees_all_schedules(engine):
    _make(_svc(engine, _scope(ORG_A)))
    _make(_svc(engine, _scope(ORG_B)))
    # the executor (SYSTEM_SCOPE) scans the whole DB regardless of org
    system = _svc(engine, SYSTEM_SCOPE)
    assert len(system.list_schedules()) == 2


def test_local_scope_sees_all(engine):
    _make(_svc(engine, _scope(ORG_A)))
    _make(_svc(engine, _scope(ORG_B)))
    from cowork.db.scoped import LOCAL_SCOPE
    assert len(_svc(engine, LOCAL_SCOPE).list_schedules()) == 2
