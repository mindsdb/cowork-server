import json
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select
from cowork.api.v1.endpoints import content_reports as endpoint
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.content_report import ContentReport
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.principal import Principal, get_principal

@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv('COWORK_TENANCY_MODE', 'org')
    get_app_settings.cache_clear()
    engine = get_engine(get_app_settings().database.uri)
    org, user = str(uuid4()), str(uuid4())
    caller = [Principal(user_id=user, org_id=org)]
    with Session(engine) as db:
        project = Project(name=str(uuid4()), path='/unused', org_id=org, created_by=user)
        db.add(project)
        db.flush()
        conversation = Conversation(topic='Report test', project_id=project.id, org_id=org, created_by=user)
        db.add(conversation)
        db.flush()
        message = Message(conversation_id=conversation.id, role='assistant', content='Reported response')
        db.add(message)
        db.commit()
        body = {'conversation_id': str(conversation.id), 'message_id': str(message.id)}
    app = FastAPI()
    app.include_router(endpoint.router, prefix='/reports')
    app.dependency_overrides[get_principal] = lambda: caller[0]
    with TestClient(app) as client:
        yield client, caller, engine, body
    get_app_settings.cache_clear()


def test_durable_report_and_retry_return_same_receipt(setup):
    client, caller, engine, body = setup
    first = client.post('/reports/', json=body)
    assert first.status_code == 201, first.text
    assert client.post('/reports/', json=body).json() == first.json()
    with Session(engine) as db:
        rows = db.exec(select(ContentReport).where(ContentReport.user_id == caller[0].user_id)).all()
        assert len(rows) == 1
        assert rows[0].evidence == '"Reported response"'
        assert rows[0].status == 'pending'
    assert 'evidence' not in first.json()
    assert client.get('/reports/').status_code == 405

def test_event_only_report_keeps_visible_card_after_message_deletion(setup):
    from cowork.models.message_event import MessageEvent

    client, _, engine, body = setup
    with Session(engine) as db:
        message = db.get(Message, UUID(body["message_id"]))
        message.content = ""
        for sequence, data in enumerate([
            {"type": "response.turn_history", "rows": ["private history"]},
            {"type": "response.quick_edit_group", "edits": [
                {"summary": "Updated heading", "request": "private original request"},
            ]},
            {"type": "response.artifact_created", "artifact": {
                "id": "abc", "title": "Quarterly results", "type": "document",
                "accessPassword": "secret", "path": "/private/path",
            }},
        ]):
            db.add(MessageEvent(message_id=message.id, sequence_number=sequence, event_data=data))
        db.commit()
    receipt = client.post("/reports/", json=body)
    assert receipt.status_code == 201, receipt.text
    with Session(engine) as db:
        message = db.get(Message, UUID(body["message_id"]))
        for event in message.message_events:
            db.delete(event)
        db.delete(message)
        db.commit()
        evidence = json.loads(db.get(ContentReport, UUID(receipt.json()["id"])).evidence)
        assert evidence == {"cards": [{"event": "response.quick_edit_group", "summary": "Updated heading"}, {
            "event": "response.artifact_created", "id": "abc", "title": "Quarterly results", "type": "document",
        }]}


def test_event_report_caps_large_unicode_cards_as_valid_json(setup):
    from cowork.models.message_event import MessageEvent

    client, _, engine, body = setup
    with Session(engine) as db:
        message = db.get(Message, UUID(body["message_id"]))
        message.content = ""
        text = "🌱" * 2048
        for sequence in range(20):
            db.add(MessageEvent(
                message_id=message.id, sequence_number=sequence, event_data={
                    "type": "response.artifact_created",
                    "artifact": dict.fromkeys(("id", "slug", "projectId", "title", "description", "type"), text),
                },
            ))
        db.commit()
    receipt = client.post("/reports/", json=body)
    assert receipt.status_code == 201
    with Session(engine) as db:
        evidence = db.get(ContentReport, UUID(receipt.json()["id"])).evidence
        assert len(evidence) <= 65536
        cards = json.loads(evidence)["cards"]
        assert 0 < len(cards) < 20
        assert cards[0]["title"] == text[:512]


@pytest.mark.parametrize('other_org', [False, True])
def test_cannot_report_another_persons_task(setup, other_org):
    client, caller, _, body = setup
    caller[0] = Principal(user_id=str(uuid4()), org_id=str(uuid4()) if other_org else caller[0].org_id)
    assert client.post('/reports/', json=body).status_code == 404


def test_cannot_substitute_evidence_or_target(setup):
    client, _, _, body = setup
    assert client.post('/reports/', json={**body, 'evidence': 'fabricated'}).status_code == 422
    assert client.post('/reports/', json={**body, 'message_id': str(uuid4())}).status_code == 404
    assert client.post('/reports/', json={**body, 'conversation_id': str(uuid4())}).status_code == 404


def _reports_by(engine, caller) -> list:
    with Session(engine) as db:
        return db.exec(select(ContentReport).where(ContentReport.user_id == caller[0].user_id)).all()


def _add_message(engine, conversation_id, **fields) -> str:
    with Session(engine) as db:
        message = Message(conversation_id=conversation_id, **fields)
        db.add(message)
        db.commit()
        return str(message.id)


def test_cannot_pair_own_task_with_a_message_from_another(setup):
    client, caller, engine, body = setup
    with Session(engine) as db:
        # Another organization's task: messages carry no org to filter on.
        org, user = str(uuid4()), str(uuid4())
        project = Project(name=str(uuid4()), path='/unused', org_id=org, created_by=user)
        db.add(project)
        db.flush()
        other = Conversation(topic='Other', project_id=project.id, org_id=org, created_by=user)
        db.add(other)
        db.commit()
        other_id = other.id
    foreign = _add_message(engine, other_id, role='assistant', content='Someone else\'s reply')
    assert client.post('/reports/', json={**body, 'message_id': foreign}).status_code == 404
    assert _reports_by(engine, caller) == []


@pytest.mark.parametrize('fields', [
    {'role': 'user', 'content': 'My own question'},
    # Hidden replay rows are saved as assistant and hold raw tool arguments.
    {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'run', 'input': {'token': 'secret'}}]},
    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Checking.'},
                                      {'type': 'tool_use', 'id': 't2', 'name': 'run', 'input': {}}]},
])
def test_only_a_visible_assistant_reply_can_be_reported(setup, fields):
    client, caller, engine, body = setup
    message_id = _add_message(engine, UUID(body['conversation_id']), **fields)
    assert client.post('/reports/', json={**body, 'message_id': message_id}).status_code == 404
    assert _reports_by(engine, caller) == []


@pytest.mark.parametrize('route', ['/reports/', '/reports/artifact'])
def test_report_racing_its_own_retry_returns_the_first_receipt(setup, monkeypatch, route):
    from sqlalchemy.exc import IntegrityError
    client, _, _, body = setup
    receipt = {'id': str(uuid4()), 'status': 'received'}
    calls = []

    async def run_db(operation, *, scope):
        calls.append(operation)
        if len(calls) == 1:
            raise IntegrityError('insert', {}, Exception('uq_content_report_target'))
        return receipt

    async def reviewed(*args, **kwargs):
        return None, ('generation', '{}')

    monkeypatch.setattr(endpoint, 'run_db', run_db)
    monkeypatch.setattr(endpoint, 'review_artifact_and_read', reviewed)
    if route.endswith('artifact'):
        body = {'project_id': str(uuid4()), 'artifact_id': str(uuid4())}
    result = client.post(route, json=body)
    assert (result.status_code, result.json()) == (201, receipt)
    assert len(calls) == 2


def test_anonymous_cannot_report(setup):
    client, caller, _, body = setup
    caller[0] = None
    assert client.post('/reports/', json=body).status_code == 401


def test_daily_report_limit(setup):
    client, caller, engine, body = setup
    with Session(engine) as db:
        now = datetime.now(timezone.utc)
        # The extra report models two requests admitted concurrently. Capacity
        # returns when the 100th newest report expires, after about two hours.
        times = [now - timedelta(hours=23), now - timedelta(hours=22)] + [now] * 99
        for created_at in times:
            db.add(ContentReport(org_id=caller[0].org_id, user_id=caller[0].user_id,
                target_key=str(uuid4()), conversation_id=uuid4(), message_id=uuid4(), evidence='test', created_at=created_at))
        db.commit()
    result = client.post('/reports/', json=body)
    assert result.status_code == 429
    assert 7195 <= int(result.headers['Retry-After']) <= 7200

@pytest.mark.parametrize('allowed', [False, True])
def test_artifact_report_uses_viewers_access_check(setup, monkeypatch, tmp_path, allowed):
    from unittest.mock import AsyncMock
    from fastapi import HTTPException
    from cowork.api.v1.artifact_scope import ResolvedArtifact
    from cowork.services.artifact_metadata import ArtifactMetadata
    client, caller, engine, _ = setup
    project_id, artifact_id = uuid4(), uuid4()
    source = tmp_path / "index.html"
    source.write_text("First version")
    extra = tmp_path / "style.css"
    extra.write_text("p {color: red}")
    found = ResolvedArtifact(None, tmp_path, ArtifactMetadata.model_validate({
        "title": "Reported artifact", "primary": "index.html",
    }), False)
    async def read_authorized(scope, project_id, artifact_id, *, read):
        return found, read(found)
    resolver = AsyncMock(side_effect=read_authorized)
    if not allowed:
        resolver.side_effect = HTTPException(404, 'Artifact not found')
    monkeypatch.setattr(endpoint, 'review_artifact_and_read', resolver)
    body = {'project_id': str(project_id), 'artifact_id': str(artifact_id)}
    response = client.post('/reports/artifact', json=body)
    assert response.status_code == (201 if allowed else 404), response.text
    assert resolver.await_args.args[0].user_id == caller[0].user_id
    assert resolver.await_args.args[1:] == (str(project_id), artifact_id.hex)
    with Session(engine) as db:
        rows=db.exec(select(ContentReport).where(ContentReport.user_id == caller[0].user_id)).all()
        assert len(rows) == int(allowed)
        if allowed:
            assert rows[0].artifact_id == artifact_id
            assert rows[0].message_id is None
    if allowed:
        assert client.post('/reports/artifact',json=body).json() == response.json()
        with Session(engine) as db:
            original = db.get(ContentReport, UUID(response.json()["id"]))
            original.status = "resolved"
            generation = json.loads(original.evidence)["generation"]
            assert generation
            db.commit()
        assert client.post('/reports/artifact',json=body).json() == response.json()
        source.write_text("Second version")
        changed = client.post('/reports/artifact', json=body)
        assert changed.status_code == 201
        assert changed.json()["id"] != response.json()["id"]
        with Session(engine) as db:
            updated = db.get(ContentReport, UUID(changed.json()["id"]))
            assert updated.status == "pending"
            assert json.loads(updated.evidence)["generation"] != generation
        extra.write_text("p {color: black}")
        supporting_asset_changed = client.post('/reports/artifact', json=body)
        assert supporting_asset_changed.status_code == 201
        assert supporting_asset_changed.json() != changed.json()
        assert client.post('/reports/artifact', json=body).json() == supporting_asset_changed.json()
        assert set(tmp_path.iterdir()) == {source, extra}


def test_report_migration_preserves_other_tables_and_round_trips():
    import importlib
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from unittest.mock import patch
    migration=importlib.import_module('cowork.db.alembic.versions.a5c8721e09dd_content_reports')
    engine=sa.create_engine('sqlite://')
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE existing_data (value TEXT)')
        connection.exec_driver_sql("INSERT INTO existing_data VALUES ('preserve')")
        with patch.object(migration, 'op', Operations(MigrationContext.configure(connection))):
            migration.upgrade()
            migration.upgrade()  # databases bootstrapped from metadata are safe too
            names={column['name'] for column in sa.inspect(connection).get_columns('content_reports')}
            assert {'target_key','artifact_id','message_id','reviewed_by','resolution'} <= names
            migration.downgrade()
            assert 'content_reports' not in sa.inspect(connection).get_table_names()
            assert connection.exec_driver_sql('SELECT value FROM existing_data').scalar() == 'preserve'


def test_operator_can_inspect_and_record_disposition_once(setup, monkeypatch, capsys):
    from cowork import content_report_admin
    client, _, engine, body = setup
    report_id=client.post('/reports/',json=body).json()['id']
    monkeypatch.setattr('sys.argv',['content-report-admin','show',report_id])
    content_report_admin.main()
    assert 'Reported response' in capsys.readouterr().out
    monkeypatch.setattr('sys.argv',['content-report-admin','resolve',report_id,'--reviewer','test-operator','--resolution','Reviewed synthetic test evidence.'])
    content_report_admin.main()
    with Session(engine) as db:
        from uuid import UUID
        row=db.get(ContentReport,UUID(report_id))
        assert row.status == 'resolved'
        assert row.reviewed_by == 'test-operator'
        assert row.resolution == 'Reviewed synthetic test evidence.'
    with pytest.raises(SystemExit):
        content_report_admin.main()
