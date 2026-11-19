"""A submission validated against a stored spec is always probed against it.

submit_form decides whether a submission is checked against a stored spec
(an explicit ``connector_id`` or a stamped ``_connector_id``) or against its
own handcrafted form. The probe stream must reach the same decision, or a
request validated as a built-in connector could be saved untested.
"""
from types import SimpleNamespace

import pytest

from cowork.api.v1.endpoints.connectors.submissions import submit_form
from cowork.db.scoped import LOCAL_SCOPE
from cowork.handlers import probe as probe_handler
from cowork.schemas.connectors import SubmitFormRequest
from cowork.services.connectors.probe import ProbeOutcome

POSTGRES_VALUES = {
    "host": "db.example.com", "port": "5432", "database": "app",
    "username": "app", "password": "not-a-real-password",
}


class _RecordingProbe:
    runs: list[str] = []

    def __init__(self, *, engine, **_kwargs) -> None:
        self.engine = engine

    async def run(self):
        type(self).runs.append(self.engine)
        yield "verdict", ProbeOutcome(status="failure", error="rejected")


@pytest.fixture
def saved(monkeypatch):
    _RecordingProbe.runs = []
    monkeypatch.setattr(probe_handler, "CredentialProbe", _RecordingProbe)
    monkeypatch.setattr(probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object()))
    monkeypatch.setattr("anton.workspace.Workspace", lambda path: SimpleNamespace(path=path))
    calls: list[tuple] = []
    monkeypatch.setattr(probe_handler, "persist_connection", lambda *a, **kw: calls.append(a) or "slug")
    return calls


async def _stream(req: SubmitFormRequest) -> str:
    response = await submit_form(req, LOCAL_SCOPE)
    return "".join([chunk if isinstance(chunk, str) else chunk.decode() async for chunk in response.body_iterator])


@pytest.mark.asyncio
async def test_an_explicit_connector_id_with_an_unstamped_form_is_still_probed(saved):
    req = SubmitFormRequest(
        connector_id="postgres",
        method="host-port",
        form_id="fm_aaaaaaaaaa",
        form_spec={"form_id": "fm_aaaaaaaaaa", "fields": []},
        values=POSTGRES_VALUES,
    )

    stream = await _stream(req)

    assert _RecordingProbe.runs == ["postgres"]
    assert "no live probe" not in stream
    assert saved == []


@pytest.mark.asyncio
async def test_a_stamped_builtin_submission_is_probed(saved):
    req = SubmitFormRequest(
        method="host-port",
        form_id="postgres-connector",
        form_spec={"_connector_id": "postgres", "form_id": "postgres-connector"},
        values=POSTGRES_VALUES,
    )

    await _stream(req)

    assert _RecordingProbe.runs == ["postgres"]
