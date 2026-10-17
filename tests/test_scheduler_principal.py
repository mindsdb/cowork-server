"""Service-principal resolution for scheduled runs.

A scheduled run has no HTTP request and so no gateway-injected principal, so
`_principal_for_schedule` derives one from the schedule row instead — letting
the turn (and the remote backend's per-tenant key mint) run in org mode with
nothing in flight. Local mode stays unscoped.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import MissingTenantScopeError
from cowork.models.schedule import Schedule
from cowork.scheduler import _principal_for_schedule

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
USER = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _schedule(org_id: str | None, created_by: str | None) -> Schedule:
    return Schedule(
        title="daily report",
        prompt="do it",
        cadence="daily",
        next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        project_id=uuid4(),
        model="sonnet",
        org_id=org_id,
        created_by=created_by,
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_local_mode_has_no_principal(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    # Even a row carrying ids (e.g. copied from a cloud DB) stays unscoped on
    # desktop — the deployment mode decides, not the data.
    assert _principal_for_schedule(_schedule(ORG, USER)) is None


def test_org_mode_builds_principal_from_the_row(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    principal = _principal_for_schedule(_schedule(ORG, USER))
    assert principal is not None
    assert principal.org_id == ORG
    assert principal.user_id == USER


@pytest.mark.parametrize(
    "org_id, created_by",
    [(None, USER), (ORG, None), (None, None)],
    ids=["no-org", "no-creator", "neither"],
)
def test_org_mode_fails_loud_on_missing_identity(monkeypatch, org_id, created_by):
    # A NULL id on an org-mode row is corrupt data: running anyway would write
    # rows the owning user can never see. Fail loud instead.
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    with pytest.raises(MissingTenantScopeError):
        _principal_for_schedule(_schedule(org_id, created_by))


# ── trusted identity headers (local tenancy, personal partitions) ────────────
# ORG stands in for the stored partition id: the row's org_id is used as it is.

def test_personal_schedule_runs_as_its_creator(personal_deployment):
    principal = _principal_for_schedule(_schedule(ORG, USER))
    assert principal is not None
    assert (principal.org_id, principal.user_id) == (ORG, USER)


@pytest.mark.parametrize(
    "org_id, created_by",
    [(None, USER), (ORG, None), (None, None)],
    ids=["no-org", "no-creator", "neither"],
)
def test_personal_schedule_with_no_owner_fails_loud(personal_deployment, org_id, created_by):
    # A row made before the update has no owner. Running it would act as no
    # one, so it fails closed and the scheduler turns it off, as in org mode.
    with pytest.raises(MissingTenantScopeError):
        _principal_for_schedule(_schedule(org_id, created_by))


def test_shared_schedules_stay_unscoped_with_trusted_identity(personal_deployment, monkeypatch):
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "memory")
    get_app_settings.cache_clear()
    assert _principal_for_schedule(_schedule(ORG, USER)) is None
    assert _principal_for_schedule(_schedule(None, None)) is None
