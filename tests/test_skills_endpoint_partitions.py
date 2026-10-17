"""The skills endpoint inside a person's partition.

Builtins are read-only wherever the skill store is partitioned, so the endpoint
refuses edits, renames into and deletes of a builtin slug there, and reports no
capability for them. A rename of the person's own skill works: the target slug
is computed for every scope, not only inside the org-mode lock block.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.schemas.skills import SkillUpdateRequest
from cowork.services.skills import BUILTIN_SKILL_SLUGS, SkillService

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
SUB_A = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _person(*, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=resources)


@pytest.fixture()
def skills_root(tmp_path, monkeypatch, personal_deployment):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    get_app_settings.cache_clear()
    yield tmp_path / "skills"
    get_app_settings.cache_clear()


@pytest.fixture()
def scoped(skills_root):
    raw = get_open_session()
    try:
        yield lambda scope: ScopedSession(raw, scope)
    finally:
        raw.close()


def _builtin() -> str:
    return sorted(BUILTIN_SKILL_SLUGS)[0]


def test_listing_seeds_read_only_builtins_with_no_author(scoped):
    listed = skills_ep.list_skills(scoped(_person()), None)["skills"]

    by_id = {skill["id"]: skill for skill in listed}
    assert set(BUILTIN_SKILL_SLUGS) <= set(by_id)
    builtin = by_id[_builtin()]
    assert builtin["isBuiltin"] is True
    assert builtin["capabilities"] == {"canEdit": False, "canDelete": False, "canDisable": False, "canShare": False}
    # No "Another member": the caller is the only actor a fallback could name.
    assert builtin["attribution"]["createdBy"] is None


def test_renaming_your_own_skill_works(scoped):
    scope = _person()
    SkillService(scope).create_skill("my-notes", "Do the thing.")

    renamed = skills_ep.update_skill("my-notes", SkillUpdateRequest(label="my-renamed"), scoped(scope), None)

    assert renamed["id"] == "my-renamed"
    assert renamed["capabilities"]["canEdit"] is True
    assert renamed["attribution"]["createdBy"] is None


def test_renaming_into_or_editing_a_builtin_is_refused(scoped):
    scope = _person()
    SkillService(scope).ensure_builtin_skills()
    SkillService(scope).create_skill("my-notes", "Do the thing.")

    with pytest.raises(HTTPException) as into:
        skills_ep.update_skill("my-notes", SkillUpdateRequest(label=_builtin()), scoped(scope), None)
    assert into.value.status_code == 403
    with pytest.raises(HTTPException) as edit:
        skills_ep.update_skill(_builtin(), SkillUpdateRequest(instructions="x"), scoped(scope), None)
    assert edit.value.status_code == 403
    with pytest.raises(HTTPException) as delete:
        skills_ep.delete_skill(_builtin(), scoped(scope), None)
    assert delete.value.status_code == 403


def test_with_shared_skills_a_builtin_copy_is_the_operator_views(scoped, skills_root):
    """Skills shared: the base store's builtin copy is mutable, but nobody
    created it, so only the operator view edits it."""
    import shutil

    from cowork.services.skills import BUILTIN_SKILLS_DIR, SHARED_SKILL_REFUSAL

    shutil.copytree(BUILTIN_SKILLS_DIR / _builtin(), skills_root / _builtin())
    person = _person(resources=frozenset())
    for scope, editable in ((person, False), (LOCAL_SCOPE, True)):
        listed = {s["id"]: s for s in skills_ep.list_skills(scoped(scope), None)["skills"]}
        assert listed[_builtin()]["capabilities"]["canEdit"] is editable

    with pytest.raises(HTTPException) as refused:
        skills_ep.update_skill(_builtin(), SkillUpdateRequest(instructions="mine now"), scoped(person), None)
    assert (refused.value.status_code, refused.value.detail) == (403, SHARED_SKILL_REFUSAL)
    edited = skills_ep.update_skill(
        _builtin(), SkillUpdateRequest(instructions="the operator's"), scoped(LOCAL_SCOPE), None
    )
    assert edited["declarative"] == "the operator's"


@pytest.fixture()
def desktop(tmp_path, monkeypatch):
    """The desktop sidecar: local tenancy, no trusted identity, no managed
    folder, every store under tmp_path. Yields its LOCAL_SCOPE session."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    raw = get_open_session()
    try:
        yield ScopedSession(raw, LOCAL_SCOPE)
    finally:
        raw.close()
        get_app_settings.cache_clear()


def test_a_desktop_rename_into_an_existing_name_answers_409(desktop):
    service = SkillService(LOCAL_SCOPE)
    service.create_skill("a-skill", "A.")
    service.create_skill("b-skill", "B.")

    with pytest.raises(HTTPException) as taken:
        skills_ep.update_skill("a-skill", SkillUpdateRequest(label="b-skill"), desktop, None)

    assert taken.value.status_code == 409
    assert service.get_skill("a-skill").instructions.strip() == "A."
    assert service.get_skill("b-skill").instructions.strip() == "B."
