"""The access a publish that names none gets, and the shared publish token."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from pydantic import SecretStr

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.models.artifact_owner_choice import AccessMode
from cowork.schemas.artifact_access import OWNER_ONLY_ACCESS, ArtifactAccessRequest
from cowork.services import publish
from cowork.services.artifact_autopublish import AUTOPUBLISH_ACCESS

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
PASSWORD_ENTRY = {"report_id": "rid-1", "mode": "password", "requires_password": True, "access_password": "pw"}


@pytest.fixture
def publish_token(monkeypatch):
    def set_token(value: str) -> None:
        monkeypatch.setenv("COWORK_PUBLISH_TOKEN", value)
        get_app_settings.cache_clear()

    yield set_token
    monkeypatch.undo()
    get_app_settings.cache_clear()


def test_autopublish_uses_the_shared_owner_only_access():
    assert AUTOPUBLISH_ACCESS is OWNER_ONLY_ACCESS
    assert OWNER_ONLY_ACCESS.publisher_access()["owner_only"] is True


def test_a_named_access_or_password_is_kept():
    public = ArtifactAccessRequest(mode=AccessMode.PUBLIC)
    assert publish.default_publish_access(public, None, report_id=None, previous=None, scope=None) is public
    assert publish.default_publish_access(None, "pw", report_id=None, previous=None, scope=None) is None


def test_a_first_publish_is_owner_only():
    assert publish.default_publish_access(None, None, report_id=None, previous=None, scope=None) is OWNER_ONLY_ACCESS
    assert publish.default_publish_access(None, None, report_id=None, previous=None, scope=ORG_SCOPE) is OWNER_ONLY_ACCESS


def test_a_republish_keeps_its_access():
    kept = publish.default_publish_access(None, None, report_id="rid-1", previous=PASSWORD_ENTRY, scope=None)
    assert kept.publisher_access() == {"mode": "password", "password": "pw"}


def test_a_taken_down_page_comes_back_owner_only():
    taken_down = {**PASSWORD_ENTRY, "mode": "public", "requires_password": False, "published": False}
    assert publish.default_publish_access(
        None, None, report_id="rid-1", previous=taken_down, scope=None
    ) is OWNER_ONLY_ACCESS


def test_a_taken_down_page_through_a_shared_token_must_choose(publish_token):
    publish_token("tok")
    taken_down = {**PASSWORD_ENTRY, "published": False}
    with pytest.raises(publish.AccessChoiceRequired):
        publish.default_publish_access(None, None, report_id="rid-1", previous=taken_down, scope=LOCAL_SCOPE)


def test_a_first_publish_through_a_shared_token_must_choose(publish_token):
    publish_token("tok")
    with pytest.raises(publish.AccessChoiceRequired) as exc:
        publish.default_publish_access(None, None, report_id=None, previous=None, scope=LOCAL_SCOPE)
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == publish.ACCESS_CHOICE_REQUIRED_CODE


def test_an_owner_only_first_publish_through_a_shared_token_is_refused(publish_token):
    publish_token("tok")
    with pytest.raises(publish.AccessChoiceRequired):
        publish.default_publish_access(OWNER_ONLY_ACCESS, None, report_id=None, previous=None, scope=LOCAL_SCOPE)
    named = ArtifactAccessRequest(mode=AccessMode.RESTRICTED, allowed_emails=["a@x.com"], owner_only=True)
    assert publish.default_publish_access(named, None, report_id=None, previous=None, scope=LOCAL_SCOPE) is named
    # A re-publish keeps the link it has.
    assert publish.default_publish_access(
        OWNER_ONLY_ACCESS, None, report_id="rid-1", previous=None, scope=LOCAL_SCOPE
    ) is OWNER_ONLY_ACCESS


def test_an_owner_only_first_publish_without_a_shared_token_is_kept():
    assert publish.default_publish_access(
        OWNER_ONLY_ACCESS, None, report_id=None, previous=None, scope=LOCAL_SCOPE
    ) is OWNER_ONLY_ACCESS


def test_the_org_path_ignores_the_publish_token_setting(publish_token):
    publish_token("tok")
    assert publish.default_publish_access(None, None, report_id=None, previous=None, scope=ORG_SCOPE) is OWNER_ONLY_ACCESS


@pytest.mark.parametrize(("token", "shared"), [("", False), ("   ", False), ("tok", True)])
def test_the_shared_token_check_matches_the_key_the_endpoint_uses(publish_token, token, shared):
    publish_token(token)
    settings = SimpleNamespace(
        openai_base_url="", minds_url="https://api.mindshub.ai", minds_api_key=SecretStr("provider"), publish_url=""
    )
    _url, api_key = publish._resolve_publish_endpoint(settings)
    assert publish.uses_shared_publish_token(LOCAL_SCOPE) is shared
    assert publish.uses_shared_publish_token(None) is shared
    assert (api_key == "tok") is shared
    assert publish.uses_shared_publish_token(ORG_SCOPE) is False


def test_an_approval_refusal_lists_what_is_pending():
    pending = [{"engine": "postgres", "name": "scm", "label": "postgres/scm"}]
    exc = publish.DatasourceApprovalRequired(pending)
    assert exc.status_code == 409
    assert exc.detail == {
        "code": publish.DATASOURCE_APPROVAL_REQUIRED_CODE,
        "message": publish.DATASOURCE_APPROVAL_REQUIRED_MESSAGE,
        "pending": pending,
    }
    assert exc.pending == pending


@pytest.fixture
def artifacts_base(tmp_path: Path) -> Path:
    base = tmp_path / "proj" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    return base


@pytest.fixture
def page(artifacts_base: Path) -> Path:
    folder = artifacts_base / "report"
    folder.mkdir()
    (folder / "report.html").write_text("<html>report</html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "report", "name": "report", "type": "html-app"}))
    return folder


@pytest.fixture
def upload(monkeypatch):
    fake = mock.Mock(return_value={"view_url": "https://view.example/r/1", "report_id": "rid-1", "md5": "d"})
    monkeypatch.setattr("anton.publisher.publish", fake)
    monkeypatch.setattr(publish, "vault_for_scope", mock.Mock(return_value=object()))
    return fake


def _publish(page, artifacts_base, access=None):
    return publish.publish_artifact(
        page, artifacts_base=artifacts_base, api_key="k", publish_url="https://api.mindshub.ai", access=access
    )


def _sent_mode(upload) -> str:
    return upload.call_args.kwargs["access"]["mode"]


def test_a_first_publish_without_access_sends_owner_only(page, artifacts_base, upload):
    out = _publish(page, artifacts_base)
    assert _sent_mode(upload) == "restricted"
    assert out["accessMode"] == "restricted"


def test_a_republish_without_access_keeps_the_password(page, artifacts_base, upload):
    _publish(page, artifacts_base, ArtifactAccessRequest(mode=AccessMode.PASSWORD, password="pw"))
    _publish(page, artifacts_base)
    assert _sent_mode(upload) == "password"


@pytest.mark.parametrize("before", [
    ArtifactAccessRequest(mode=AccessMode.PASSWORD, password="pw"),
    OWNER_ONLY_ACCESS,
    None,
])
def test_an_explicit_public_publishes_public(page, artifacts_base, upload, before):
    if before is not None:
        _publish(page, artifacts_base, before)
    _publish(page, artifacts_base, ArtifactAccessRequest(mode=AccessMode.PUBLIC))
    assert _sent_mode(upload) == "public"


def test_a_first_publish_through_a_shared_token_uploads_nothing(page, artifacts_base, upload, publish_token):
    publish_token("tok")
    with pytest.raises(publish.AccessChoiceRequired):
        _publish(page, artifacts_base)
    upload.assert_not_called()
    assert not (page / ".published.json").exists()
