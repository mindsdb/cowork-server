"""The form submission stream keeps a multi-step handcrafted connect in one record.

Runs ProbeHandler end to end with no conversation against a real vault, with
a stand-in probe that records what it was built with and accepts: the second
form carries ``_extends_connection``, is probed with the first record's fields
too, and must add its fields to that record.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError
from cowork.handlers import probe as probe_handler
from cowork.handlers.probe import ProbeHandler
from cowork.services.connectors.probe import CredentialProbe, ProbeOutcome
from cowork.services.connectors.submissions import store

_STEP_ONE = {"form_id": "fm_aaaaaaaaaa", "engine": "linkedin", "title": "LinkedIn", "fields": []}


@pytest.fixture
def probes(monkeypatch):
    record = SimpleNamespace(built=[], events=[])

    class RecordingProbe:
        def __init__(self, **kwargs) -> None:
            record.built.append(kwargs)

        async def run(self):
            for event in record.events:
                yield event
            yield "verdict", ProbeOutcome(status="success", summary="Connection works.")

    monkeypatch.setattr(probe_handler, "CredentialProbe", RecordingProbe)
    return record


@pytest.fixture
def definitions(monkeypatch):
    saved: list[str] = []

    async def save_definition(self, connector_id, form_spec):
        saved.append(connector_id)

    monkeypatch.setattr(ProbeHandler, "_save_definition", save_definition)
    return saved


@pytest.fixture
def vault(tmp_path, monkeypatch, definitions, probes):
    vault = LocalDataVault(Path(tmp_path) / "vault")
    monkeypatch.setattr("cowork.handlers.probe.vault_for_scope", lambda scope: vault)
    monkeypatch.setattr(ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object()))
    monkeypatch.setattr("anton.workspace.Workspace", lambda path: SimpleNamespace(path=path))
    return vault


async def _submit(values: dict, form_spec: dict, name: str = "") -> str:
    submission_id = store.stage(
        form_id=form_spec["form_id"],
        connector_id="linkedin",
        conversation_id=None,
        values=values,
        form_spec=form_spec,
    )
    events = [
        event async for event in ProbeHandler(scope=LOCAL_SCOPE).run(
            submission_id, "linkedin", None, name, None,
        )
    ]
    return "".join(events)


def _follow_up(slug: str, fields: list | None = None) -> dict:
    return {
        "form_id": "fm_bbbbbbbbbb", "engine": "linkedin", "title": "LinkedIn", "fields": fields or [],
        "_extends_connection": slug, "_existing_name": slug,
    }


def _stream_payloads(stream: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in stream.splitlines() if line.startswith("data: ")]


def _patches(stream: str) -> list[dict]:
    text = "".join(p.get("delta", "") for p in _stream_payloads(stream))
    return [json.loads(b) for b in re.findall(r"```data-vault-form-patch\n(.*?)\n```", text, re.S)]


def _final_status(stream: str) -> str | None:
    [completed] = [p for p in _stream_payloads(stream) if p.get("type") == "response.completed"]
    return completed["response"].get("status")


@pytest.mark.asyncio
async def test_follow_up_form_lands_in_the_first_record(vault):
    await _submit(
        {"client_id": "86nwdt9sl34cuy", "client_secret": "app-secret"},
        {"form_id": "fm_aaaaaaaaaa", "engine": "linkedin", "title": "LinkedIn", "fields": []},
    )
    [first] = vault.list_connections()
    slug = first["name"]

    stream = await _submit(
        {"access_token": "access-1", "refresh_token": "refresh-1"},
        {
            "form_id": "fm_bbbbbbbbbb",
            "engine": "linkedin",
            "fields": [],
            "_extends_connection": slug,
            "_existing_name": slug,
        },
        name=slug,
    )

    assert f"Saved as `{slug}`" in stream
    assert len(vault.list_connections()) == 1
    fields = vault.read_record("linkedin", slug)["fields"]
    assert {"client_id", "client_secret", "access_token", "refresh_token"} <= set(fields)


@pytest.mark.asyncio
async def test_follow_up_form_without_extends_is_a_separate_record(vault):
    await _submit(
        {"client_id": "86nwdt9sl34cuy", "client_secret": "app-secret"},
        {"form_id": "fm_aaaaaaaaaa", "engine": "linkedin", "title": "LinkedIn", "fields": []},
    )
    await _submit(
        {"client_id": "86nwdt9sl34cuy", "access_token": "access-1"},
        {"form_id": "fm_bbbbbbbbbb", "engine": "linkedin", "title": "LinkedIn", "fields": []},
    )
    assert len(vault.list_connections()) == 2


@pytest.mark.asyncio
async def test_only_the_first_step_saves_the_connector_definition(vault, definitions):
    await _submit(
        {"client_id": "86nwdt9sl34cuy", "client_secret": "app-secret"},
        {"form_id": "fm_aaaaaaaaaa", "engine": "linkedin", "title": "LinkedIn", "fields": []},
    )
    [first] = vault.list_connections()
    await _submit(
        {"access_token": "access-1"},
        {
            "form_id": "fm_bbbbbbbbbb", "engine": "linkedin", "title": "LinkedIn", "fields": [],
            "_extends_connection": first["name"], "_existing_name": first["name"],
        },
        name=first["name"],
    )

    assert definitions == ["linkedin"]


@pytest.mark.asyncio
async def test_follow_up_step_is_probed_with_the_connection_it_extends(vault, probes):
    await _submit({"client_id": "86nwdt9sl34cuy", "client_secret": "app-secret"}, _STEP_ONE)
    [first] = vault.list_connections()

    await _submit({"access_token": "access-1"}, _follow_up(first["name"]), name=first["name"])

    assert probes.built[1]["credentials"] == {
        "client_id": "86nwdt9sl34cuy",
        "client_secret": "app-secret",
        "access_token": "access-1",
    }


@pytest.mark.asyncio
async def test_a_resubmitted_field_is_probed_with_its_new_value(vault, probes):
    await _submit({"client_id": "86nwdt9sl34cuy", "client_secret": "app-secret"}, _STEP_ONE)
    [first] = vault.list_connections()

    await _submit(
        {"client_secret": "rotated-secret", "access_token": "access-1"},
        _follow_up(first["name"]),
        name=first["name"],
    )

    assert probes.built[1]["credentials"]["client_secret"] == "rotated-secret"


@pytest.mark.asyncio
async def test_extending_a_missing_connection_fails_before_any_probe(vault, probes):
    stream = await _submit({"access_token": "access-1"}, _follow_up("linkedin-gone"), name="linkedin-gone")

    assert probes.built == []
    assert vault.list_connections() == []
    assert _final_status(stream) == "failed"
    assert any(
        "No linkedin connection named 'linkedin-gone'" in (patch.get("form_error") or "")
        for patch in _patches(stream)
    )


@pytest.mark.asyncio
async def test_a_saved_field_the_step_does_not_show_gets_no_form_patch(vault, probes):
    await _submit({"client_id": "86nwdt9sl34cuy", "client_secret": "app-secret"}, _STEP_ONE)
    [first] = vault.list_connections()
    probes.events = [
        ("field_status", {"name": "client_id", "status": "ok", "method_id": None}),
        ("remove_field", {"name": "client_secret", "method_id": None}),
        ("field_status", {"name": "CLIENT_ID", "status": "ok", "method_id": None}),
        ("field_status", {"name": "DS_CLIENT_SECRET", "status": "ok", "method_id": None}),
        ("field_status", {"name": "access_token", "status": "ok", "method_id": None}),
    ]

    stream = await _submit(
        {"access_token": "access-1"},
        _follow_up(first["name"], fields=[{"name": "access_token", "type": "password"}]),
        name=first["name"],
    )

    patched = {name for patch in _patches(stream) for name in (patch.get("fields") or {})}
    assert patched == {"access_token"}


@pytest.mark.asyncio
async def test_the_probe_is_told_which_fields_are_already_saved(vault, probes):
    await _submit({"client_id": "86nwdt9sl34cuy", "client_secret": "app-secret"}, _STEP_ONE)
    [first] = vault.list_connections()

    await _submit(
        {"access_token": "access-1"},
        _follow_up(first["name"], fields=[{"name": "access_token", "type": "password"}]),
        name=first["name"],
    )

    saved_fields = probes.built[1]["saved_fields"]
    assert saved_fields == ["client_id", "client_secret"]
    prompt = CredentialProbe(
        engine="linkedin", credentials={}, llm_client=None, workspace=None, saved_fields=saved_fields,
    )._build_prompt("/tmp/probe.env", ["DS_CLIENT_ID", "DS_CLIENT_SECRET", "DS_ACCESS_TOKEN"])
    assert "ALREADY SAVED ON THIS CONNECTION" in prompt
    assert "`client_id`, `client_secret`" in prompt


@pytest.mark.asyncio
async def test_a_connection_that_cannot_be_read_fails_on_the_form(vault, probes, monkeypatch):
    def no_scope(scope):
        raise MissingTenantScopeError("no organization in scope")

    monkeypatch.setattr("cowork.handlers.probe.vault_for_scope", no_scope)

    stream = await _submit({"access_token": "access-1"}, _follow_up("linkedin-1"), name="linkedin-1")

    assert probes.built == []
    assert _final_status(stream) == "failed"
    assert any(
        "Could not read `linkedin-1`" in (patch.get("form_error") or "")
        for patch in _patches(stream)
    )
