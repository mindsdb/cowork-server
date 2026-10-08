"""The form submission stream keeps a multi-step handcrafted connect in one record.

Runs ProbeHandler end to end with no conversation against a real vault, with
a stand-in probe that accepts the credentials: the second form carries
``_extends_connection`` and must add its fields to the first record.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.db.scoped import LOCAL_SCOPE
from cowork.handlers import probe as probe_handler
from cowork.handlers.probe import ProbeHandler
from cowork.services.connectors.probe import ProbeOutcome
from cowork.services.connectors.submissions import store


class _AcceptingProbe:
    def __init__(self, **_kwargs) -> None:
        pass

    async def run(self):
        yield "verdict", ProbeOutcome(status="success", summary="Connection works.")


@pytest.fixture
def definitions(monkeypatch):
    saved: list[str] = []

    async def save_definition(self, connector_id, form_spec):
        saved.append(connector_id)

    monkeypatch.setattr(ProbeHandler, "_save_definition", save_definition)
    return saved


@pytest.fixture
def vault(tmp_path, monkeypatch, definitions):
    vault = LocalDataVault(Path(tmp_path) / "vault")
    monkeypatch.setattr("cowork.handlers.probe.vault_for_scope", lambda scope: vault)
    monkeypatch.setattr(probe_handler, "CredentialProbe", _AcceptingProbe)
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
