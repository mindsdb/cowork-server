from __future__ import annotations

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.project_collaboration import ProjectCollaborator
from cowork.services.project_permissions import (
    ProjectPermissionError,
    get_project_principal,
    has_project_permission,
    normalize_role,
    require_project_permission,
    role_allows,
)
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture(autouse=True)
def clean_permission_rows():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        rows = session.exec(select(ProjectCollaborator)).all()
        for row in rows:
            session.delete(row)
        session.commit()
    yield
    with Session(engine) as session:
        rows = session.exec(select(ProjectCollaborator)).all()
        for row in rows:
            session.delete(row)
        session.commit()


def test_role_capability_matrix():
    assert normalize_role("admin") == "owner"
    assert normalize_role("unknown") == "viewer"

    assert role_allows("viewer", "view")
    assert not role_allows("viewer", "comment")
    assert role_allows("commenter", "comment")
    assert not role_allows("commenter", "review")
    assert role_allows("reviewer", "review")
    assert not role_allows("reviewer", "edit")
    assert role_allows("editor", "edit")
    assert not role_allows("editor", "manage")
    assert role_allows("owner", "manage")
    assert role_allows("admin", "manage")
    assert not role_allows("owner", "unknown")


def test_project_principal_uses_normalized_collaborator_email():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="ada@example.com", role="reviewer"))
        session.commit()

        principal = get_project_principal(session, GENERAL_PROJECT_ID, " Ada@Example.COM ")

    assert principal is not None
    assert principal.email == "ada@example.com"
    assert principal.role == "reviewer"


def test_permission_checks_and_require_helper():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="editor@example.com", role="editor"))
        session.commit()

        assert has_project_permission(session, GENERAL_PROJECT_ID, "editor@example.com", "comment") is True
        assert has_project_permission(session, GENERAL_PROJECT_ID, "editor@example.com", "manage") is False
        principal = require_project_permission(session, GENERAL_PROJECT_ID, "editor@example.com", "edit")
        assert principal.role == "editor"

        with pytest.raises(ProjectPermissionError) as exc:
            require_project_permission(session, GENERAL_PROJECT_ID, "missing@example.com", "view")

    assert exc.value.capability == "view"
    assert exc.value.project_id == GENERAL_PROJECT_ID
