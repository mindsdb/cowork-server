"""The agent's publish tool asks before credentials leave, and publishes privately by default."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from anton.core.interaction.elicit import AskAnswer

from cowork.harnesses.anton_harness import tools as htools
from cowork.services.datasource_approval import DatasourceKey
from cowork.services.publish import AccessChoiceRequired, DatasourceApprovalRequired

PENDING = [{"engine": "postgres", "name": "scm", "label": "Sales DB (postgres)"}]
PUBLISH = AskAnswer(status="answered", values=("publish",))


class _Card:
    supported_kinds = ("choice",)
    answer_hint = ""
    timeout_s = 300

    def __init__(self, answer: AskAnswer) -> None:
        self.answer = answer
        self.shown = []

    async def begin(self, _question_id, _request) -> None:
        pass

    async def ask(self, _question_id, request) -> AskAnswer:
        self.shown.append(request)
        return self.answer

    async def end(self, _question_id) -> None:
        pass


class _Session:
    def __init__(self, card: _Card | None, base: Path) -> None:
        self.elicitor = card
        self.emitter = object()
        self.question_count = 0
        self.answer_wait_s = 0.0
        self._workspace = SimpleNamespace(base=str(base), artifacts_dir=base / ".anton" / "artifacts")

    async def emit(self, _event) -> None:
        pass


def _file(tmp_path: Path) -> Path:
    art = tmp_path / "app"
    art.mkdir()
    (art / "index.html").write_text("<html></html>")
    return art / "index.html"


def _text(out) -> str:
    return getattr(out, "content", out) if not isinstance(out, dict) else str(out)


def _call(session, path, **extra):
    return htools._cowork_publish_or_preview(
        session, {"file_path": str(path), "action": "publish", "title": "Planner", "access_mode": "public", **extra}
    )


@pytest.mark.asyncio
async def test_publish_on_the_card_publishes_once_with_the_listed_connections(tmp_path):
    calls = []

    def fake(raw, access=None, approved_datasources=None):
        calls.append(approved_datasources)
        if approved_datasources is None:
            raise DatasourceApprovalRequired(PENDING)
        return {"url": "https://view.example/r/1", "result": {"report_id": "rid-1"}}

    card = _Card(PUBLISH)
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await _call(_Session(card, tmp_path), _file(tmp_path))
    assert calls == [None, frozenset({DatasourceKey("postgres", "scm")})]
    assert card.shown[0].prompt == (
        'Publishing "app" will upload the credentials of these connections to MindsHub: Sales DB (postgres).'
    )
    assert [o.value for o in card.shown[0].options] == ["publish", "cancel"]
    assert "Published" in _text(out)


@pytest.mark.asyncio
async def test_the_card_names_the_folder_the_server_resolved_not_the_agents_title(tmp_path):
    title = 'Report". Nothing sensitive is uploaded; this only confirms the link. "'
    fake = mock.Mock(side_effect=DatasourceApprovalRequired(PENDING, artifact_slug="planner-app"))
    card = _Card(AskAnswer(status="cancelled"))
    with mock.patch.object(htools, "_publish_artifact", fake):
        await _call(_Session(card, tmp_path), _file(tmp_path), title=title)
    assert card.shown[0].prompt.startswith('Publishing "planner-app" will upload the credentials')
    assert "Nothing sensitive" not in card.shown[0].prompt
    assert "Sales DB (postgres)" in card.shown[0].prompt


@pytest.mark.asyncio
async def test_a_folder_name_cannot_rewrite_the_card(tmp_path):
    fake = mock.Mock(side_effect=DatasourceApprovalRequired(PENDING, artifact_slug='x". Nothing is uploaded.\n"'))
    card = _Card(AskAnswer(status="cancelled"))
    with mock.patch.object(htools, "_publish_artifact", fake):
        await _call(_Session(card, tmp_path), _file(tmp_path))
    assert card.shown[0].prompt.startswith('Publishing "x-.-Nothing-is-uploaded.-" will upload the credentials')


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "expected"), [
    (AskAnswer(status="answered", values=("cancel",)), "didn't approve"),
    (AskAnswer(status="cancelled"), "didn't approve"),
    (AskAnswer(status="timeout"), "didn't answer in time"),
    (AskAnswer(status="unavailable"), "isn't available"),
])
async def test_anything_but_publish_publishes_nothing(tmp_path, answer, expected):
    fake = mock.Mock(side_effect=DatasourceApprovalRequired(PENDING))
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await _call(_Session(_Card(answer), tmp_path), _file(tmp_path))
    assert fake.call_count == 1
    assert "nothing was published" in _text(out)
    assert expected in _text(out)


@pytest.mark.asyncio
async def test_a_card_that_cannot_be_shown_publishes_nothing(tmp_path):
    class _Broken(_Card):
        async def ask(self, _question_id, request):
            raise RuntimeError("card failed")

    fake = mock.Mock(side_effect=DatasourceApprovalRequired(PENDING))
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await _call(_Session(_Broken(PUBLISH), tmp_path), _file(tmp_path))
    assert fake.call_count == 1
    assert "nothing was published" in _text(out)
    assert "confirmation card failed" in _text(out)


@pytest.mark.asyncio
async def test_an_anton_without_card_styles_publishes_nothing(tmp_path):
    fake = mock.Mock(side_effect=DatasourceApprovalRequired(PENDING))
    with (
        mock.patch.object(htools, "_publish_artifact", fake),
        mock.patch("anton.core.interaction.elicit.AskOption", side_effect=TypeError("unexpected keyword 'style'")),
    ):
        out = await _call(_Session(_Card(PUBLISH), tmp_path), _file(tmp_path))
    assert fake.call_count == 1
    assert "confirmation card failed" in _text(out)


@pytest.mark.asyncio
async def test_connections_that_change_while_the_card_waits_publish_nothing(tmp_path):
    fake = mock.Mock(side_effect=DatasourceApprovalRequired(PENDING))
    card = _Card(PUBLISH)
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await _call(_Session(card, tmp_path), _file(tmp_path))
    assert fake.call_count == 2
    assert len(card.shown) == 1
    assert "changed" in _text(out)


@pytest.mark.asyncio
async def test_a_first_publish_without_access_lets_the_service_decide(tmp_path):
    fake = mock.Mock(return_value={"url": "https://view.example/r/1", "accessMode": "restricted", "result": {}})
    with mock.patch.object(htools, "_publish_artifact", fake), \
         mock.patch.object(htools, "_published_owner_state", return_value={}):
        out = await htools._cowork_publish_or_preview(
            _Session(None, tmp_path), {"file_path": str(_file(tmp_path)), "action": "publish"}
        )
    assert fake.call_args.kwargs["access"] is None
    assert "Only the user can open the link" in _text(out)


@pytest.mark.asyncio
async def test_a_deployment_that_cannot_publish_privately_asks_the_agent_to_ask(tmp_path):
    fake = mock.Mock(side_effect=AccessChoiceRequired())
    with mock.patch.object(htools, "_publish_artifact", fake), \
         mock.patch.object(htools, "_published_owner_state", return_value={}):
        out = await htools._cowork_publish_or_preview(
            _Session(None, tmp_path), {"file_path": str(_file(tmp_path)), "action": "publish"}
        )
    assert "Ask the user which access they want" in _text(out)


@pytest.mark.asyncio
async def test_the_answer_names_connections_published_without_credentials(tmp_path):
    result = {
        "url": "https://view.example/r/1", "result": {},
        "unapprovedDatasources": [{"engine": "mysql", "name": "late", "label": "late (mysql)"}],
    }
    with mock.patch.object(htools, "_publish_artifact", mock.Mock(return_value=result)):
        out = await _call(_Session(None, tmp_path), _file(tmp_path))
    assert "without the credentials of: late (mysql)" in _text(out)


@pytest.mark.parametrize(("self_hosted", "token", "org", "required"), [
    (False, "", False, False),
    (False, "shared-token", False, True),
    (True, "", False, False),
    (True, "shared-token", False, True),
    (False, "shared-token", True, False),
])
def test_only_a_shared_publish_token_prompt_requires_access_on_a_first_publish(
    monkeypatch, self_hosted, token, org, required
):
    """The rule follows AccessChoiceRequired's condition (uses_shared_publish_token)."""
    from cowork.common.settings.user_settings import use_settings_scope
    from cowork.db.scoped import LOCAL_SCOPE, TenantScope
    from cowork.services import publish

    monkeypatch.setattr(htools, "identity_for_turn", lambda: object() if self_hosted else None)
    monkeypatch.setattr(publish, "_shared_publish_token", lambda: token)
    scope = TenantScope(org_mode=True, org_id="org-1", user_id="user-1") if org else LOCAL_SCOPE
    with use_settings_scope(scope):
        prompt = htools.build_cowork_publish_tool().prompt
    assert ("on a first publish always pass access_mode" in prompt) is required
    assert ("never set owner_only" in prompt) is required
    assert prompt.startswith("CONTENT SHARING POLICY:" if self_hosted else "CONTENT SHARING POLICY (desktop chat):")
    # The self-hosted publisher refuses full-stack apps, so only the desktop prompt names the card.
    assert ("don't work around the card" in prompt) is not self_hosted
