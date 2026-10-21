"""An edit of a saved connection is checked with its stored secrets.

The edit form sends each secret the user left alone as the keep sentinel.
The LLM probe must see the stored value instead, only for a record the caller
can change, and the save must still update that record in place.
"""
from __future__ import annotations

import json
import re

import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.api.v1.endpoints.connectors import submissions as submissions_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.handlers import probe as probe_handler
from cowork.principal import personal_partition_id
from cowork.schemas.connectors import SubmitFormRequest
from cowork.services.connectors import databricks_genie
from cowork.services.connectors.connections import KEPT_VALUES_NOT_FOUND
from cowork.services.connectors.identity import KEPT_SECRET_MOVED, VAULT_KEEP_SENTINEL
from cowork.services.connectors.layered_vault import SHARED_CONNECTION_REFUSAL
from cowork.services.connectors.persist import KeptSecretMovedError, own_vault_for_scope, persist_connection
from cowork.services.connectors.probe import ProbeOutcome

PAT = "personal-access-token"
HOST = "https://adb-1234567890123456.7.azuredatabricks.net"
OTHER_HOST = "https://collector.example.net"
TOKEN = "dapi-stored-token-4b8e"
OLD_PATH = "/sql/1.0/warehouses/old"
NEW_PATH = "/sql/1.0/warehouses/new"
SAVED = {"workspace_url": HOST, "access_token": TOKEN, "http_path": OLD_PATH}

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
ALICE = TenantScope(
    personal=True, org_id=personal_partition_id(DEPLOYMENT, SUB_A), user_id=SUB_A,
    personal_resources=ALL_PERSONAL_RESOURCES,
)
BOB = TenantScope(
    personal=True, org_id=personal_partition_id(DEPLOYMENT, SUB_B), user_id=SUB_B,
    personal_resources=ALL_PERSONAL_RESOURCES,
)


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """Every store this test writes under its own tmp_path."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "home" / "data-vault"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "home" / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "home" / "memory"))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


@pytest.fixture
def probes(monkeypatch, stores) -> list[dict]:
    """Patch the LLM probe to record the arguments of every probe built and succeed."""
    built: list[dict] = []

    class _Probe:
        """A CredentialProbe that records its arguments and succeeds."""

        def __init__(self, **kwargs):
            """Record the probe's arguments."""
            built.append(kwargs)

        async def run(self):
            """Yield a success verdict."""
            yield "verdict", ProbeOutcome(status="success", summary="Connection works.")

    monkeypatch.setattr(probe_handler, "CredentialProbe", _Probe)
    monkeypatch.setattr(
        probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object())
    )
    return built


async def _submit(
    values: dict, *, name: str = "", scope: TenantScope = LOCAL_SCOPE, method: str = PAT,
) -> str:
    """Submit the Databricks form through the submissions route and return the whole stream."""
    response = await submissions_ep.submit_form(
        SubmitFormRequest(connector_id="databricks", method=method, name=name, values=dict(values)),
        scope,
    )
    return "".join([chunk async for chunk in response.body_iterator])


def _text(stream: str) -> str:
    """The reply's text: every output delta in the stream, joined."""
    events = [json.loads(line[len("data: "):]) for line in stream.splitlines() if line.startswith("data: ")]
    return "".join(e["delta"] for e in events if e.get("type") == "response.output_text.delta")


def _outcome(stream: str) -> dict:
    """The last form patch the reply carries, which states the outcome."""
    blocks = re.findall(r"```data-vault-form-patch\n(.*?)\n```", _text(stream), flags=re.DOTALL)
    return json.loads(blocks[-1])


def _records(scope: TenantScope = LOCAL_SCOPE) -> list[dict]:
    """Every databricks record in ``scope``'s own vault."""
    vault = own_vault_for_scope(scope)
    return [
        vault.read_record("databricks", c["name"])
        for c in vault.list_connections()
        if c["engine"] == "databricks"
    ]


def _edit(**changes: str) -> dict:
    """The edit form's payload for SAVED: the secret kept, the rest as saved, then ``changes``."""
    return {**SAVED, "access_token": VAULT_KEEP_SENTINEL, "user_label": "", **changes}


async def test_an_edit_that_keeps_the_token_and_changes_only_the_label_is_checked_with_the_stored_token(probes):
    slug = persist_connection("databricks", PAT, "", SAVED)

    stream = await _submit(_edit(user_label="Warehouse"), name=slug)

    assert _outcome(stream)["_is_success"] is True
    [probe] = probes
    assert probe["credentials"] == SAVED
    [record] = _records()
    assert record["name"] == slug
    assert record["fields"]["access_token"] == TOKEN
    assert record["fields"]["_user_label"] == "Warehouse"
    assert VAULT_KEEP_SENTINEL not in json.dumps(record)


@pytest.mark.parametrize(
    "changed",
    [{"workspace_url": OTHER_HOST}, {"http_path": NEW_PATH}, {"catalog": "main"}],
    ids=["another-host", "another-warehouse", "an-added-catalog"],
)
async def test_an_edit_that_keeps_the_token_and_changes_the_account_is_refused_before_any_probe(
    probes, changed,
):
    slug = persist_connection("databricks", PAT, "", SAVED)
    [before] = _records()

    stream = await _submit(_edit(**changed), name=slug)

    assert KEPT_SECRET_MOVED in _text(stream)
    assert _outcome(stream)["_is_success"] is False
    assert probes == []
    assert TOKEN not in stream
    assert _records() == [before]


def test_a_direct_save_that_keeps_the_token_for_another_host_is_refused_and_writes_nothing(stores):
    slug = persist_connection("databricks", PAT, "", SAVED)
    [before] = _records()
    values = {**SAVED, "access_token": VAULT_KEEP_SENTINEL, "workspace_url": OTHER_HOST}

    with pytest.raises(KeptSecretMovedError) as refused:
        persist_connection("databricks", PAT, slug, values)

    assert refused.value.message == KEPT_SECRET_MOVED
    assert _records() == [before]


def test_a_direct_save_that_keeps_the_token_and_changes_no_identity_field_saves_in_place(stores):
    slug = persist_connection("databricks", PAT, "", SAVED)

    saved = persist_connection("databricks", PAT, slug, {**SAVED, "access_token": VAULT_KEEP_SENTINEL})

    assert saved == slug
    [record] = _records()
    assert record["fields"]["access_token"] == TOKEN


def test_a_save_whose_caller_bound_the_kept_secret_may_change_other_fields(stores):
    genie = {"workspace_url": HOST, "client_id": "principal", "_client_secret": "stored-secret"}
    slug = persist_connection("databricks", "genie-mcp", "", genie)
    edit = {**genie, "_client_secret": VAULT_KEEP_SENTINEL, "genie_space_id": "a" * 32}

    with pytest.raises(KeptSecretMovedError):
        persist_connection("databricks", "genie-mcp", slug, edit)
    persist_connection("databricks", "genie-mcp", slug, edit, secret_bound_by_caller=True)

    [record] = _records()
    assert record["fields"]["_client_secret"] == "stored-secret"
    assert record["fields"]["genie_space_id"] == "a" * 32


async def test_an_edit_that_changes_the_token_is_checked_with_the_new_token(probes):
    slug = persist_connection("databricks", PAT, "", SAVED)

    stream = await _submit(_edit(access_token="dapi-rotated-token"), name=slug)

    assert _outcome(stream)["_is_success"] is True
    assert probes[0]["credentials"]["access_token"] == "dapi-rotated-token"
    [record] = _records()
    assert record["fields"]["access_token"] == "dapi-rotated-token"


async def test_a_kept_non_secret_field_resolves_to_its_stored_value_as_the_save_does(probes):
    slug = persist_connection("databricks", PAT, "", SAVED)

    stream = await _submit(_edit(http_path=VAULT_KEEP_SENTINEL), name=slug)

    assert _outcome(stream)["_is_success"] is True
    assert probes[0]["credentials"]["http_path"] == OLD_PATH
    [record] = _records()
    assert record["fields"]["http_path"] == OLD_PATH


@pytest.mark.parametrize("name", ["", "no-such-connection"], ids=["unnamed", "missing-record"])
async def test_a_kept_secret_with_no_record_to_resolve_against_is_refused_before_any_probe(probes, name):
    stream = await _submit(_edit(), name=name)

    assert KEPT_VALUES_NOT_FOUND in _text(stream)
    assert _outcome(stream)["_is_success"] is False
    assert probes == []
    assert _records() == []


async def test_a_kept_secret_is_never_logged(probes, monkeypatch):
    lines: list[str] = []

    class _Log:
        """A logger that keeps every line it is given."""

        def _record(self, msg: str, *args: object, **kwargs: object) -> None:
            """Keep the formatted message."""
            lines.append(str(msg) % args if args else str(msg))

        debug = info = warning = error = exception = _record

    monkeypatch.setattr(probe_handler, "logger", _Log())
    slug = persist_connection("databricks", PAT, "", SAVED)

    stream = await _submit(_edit(user_label="Warehouse"), name=slug)

    assert TOKEN not in stream
    assert all(TOKEN not in line for line in lines)


@pytest.fixture
def alices_shared_connection(personal_deployment, managed_dir, probes) -> str:
    """Alice's PAT connection, shared with everyone through the managed registry."""
    get_app_settings.cache_clear()
    slug = persist_connection("databricks", PAT, "", SAVED, scope=ALICE)
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_connections_file.write_text(json.dumps([
        {"user_id": SUB_A, "partition_id": ALICE.org_id, "engine": "databricks", "name": slug},
    ]))
    return slug


async def test_an_edit_of_a_connection_shared_with_the_caller_is_refused_before_any_probe(
    alices_shared_connection, probes,
):
    alices = own_vault_for_scope(ALICE).read_record("databricks", alices_shared_connection)

    stream = await _submit(_edit(http_path=NEW_PATH), name=alices_shared_connection, scope=BOB)

    assert SHARED_CONNECTION_REFUSAL in _text(stream)
    assert _outcome(stream)["_is_success"] is False
    assert probes == []
    assert TOKEN not in stream
    assert _records(BOB) == []
    assert own_vault_for_scope(ALICE).read_record("databricks", alices_shared_connection) == alices


async def test_an_edit_of_a_managed_connection_is_refused_before_any_probe(managed_dir, probes):
    get_app_settings.cache_clear()
    managed = LocalDataVault(managed_dir.vault_dir)
    managed.save("databricks", "operators", {**SAVED, "_method": PAT}, secure_keys=["access_token"])

    stream = await _submit(_edit(http_path=NEW_PATH), name="operators")

    assert SHARED_CONNECTION_REFUSAL in _text(stream)
    assert probes == []
    assert TOKEN not in stream
    assert managed.read_record("databricks", "operators")["fields"]["http_path"] == OLD_PATH


async def test_the_genie_edit_path_keeps_its_own_check(probes, monkeypatch):
    checked: list[dict] = []

    async def _save_genie(scope, credentials, name, *, label="", user_label=""):
        """Record what the Genie save received and refuse, as a failed check does."""
        checked.append(dict(credentials))
        raise databricks_genie.GenieConnectionError(databricks_genie.SECRET_NEEDED)

    monkeypatch.setattr(probe_handler, "save_genie_connection", _save_genie)
    values = {"workspace_url": HOST, "client_id": "principal", "client_secret": VAULT_KEEP_SENTINEL}

    stream = await _submit(values, method="genie-mcp")

    assert checked == [values]
    assert databricks_genie.SECRET_NEEDED in _text(stream)
    assert KEPT_VALUES_NOT_FOUND not in _text(stream)
    assert probes == []


@pytest.mark.parametrize("ending", ["verdict", "crash"])
async def test_a_kept_secret_echoed_by_the_probe_never_reaches_the_stream(stores, monkeypatch, ending):
    class _EchoingProbe:
        """A CredentialProbe whose cells and verdict all repeat the token it was given."""

        def __init__(self, **kwargs):
            """Keep the token the probe was handed."""
            self.token = kwargs["credentials"]["access_token"]

        async def run(self):
            """Yield every event kind the handler streams, each carrying the token."""
            token = self.token
            yield "text", f"Using {token}.\n"
            yield "status", f"Trying {token}"
            yield "switch_method", {"method_id": PAT, "reason": f"Token {token} first"}
            yield "scratchpad", {"action": "start"}
            yield "scratchpad", {"action": "end", "name": "c", "code": f"print('{token}')"}
            yield "scratchpad", {"action": "result", "content": json.dumps({"stdout": token})}
            if ending == "crash":
                raise RuntimeError(f"rejected {token}")
            yield "verdict", ProbeOutcome(status="failure", error=f"401 for {token}", follow_up=token)

    monkeypatch.setattr(probe_handler, "CredentialProbe", _EchoingProbe)
    monkeypatch.setattr(
        probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object())
    )
    monkeypatch.setattr(probe_handler, "logger", _SilentLog())
    slug = persist_connection("databricks", PAT, "", SAVED)

    stream = await _submit(_edit(user_label="Warehouse"), name=slug)

    assert TOKEN not in stream
    assert "[saved secret]" in stream
    assert "print(" in stream
    assert _outcome(stream)["_is_success"] is False


class _SilentLog:
    """A logger that drops every line, so a crash's traceback stays out of the test output."""

    def _drop(self, msg: str, *args: object, **kwargs: object) -> None:
        """Drop the line."""

    debug = info = warning = error = exception = _drop
