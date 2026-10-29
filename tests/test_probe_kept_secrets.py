"""An edit of a saved connection is checked with its stored secrets.

The edit form sends each secret the user left alone as the keep sentinel.
The LLM probe must see the stored value instead, only for a record the caller
can change, and the save must still update that record in place.
"""
from __future__ import annotations

import base64
import json
import re
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import connections as connections_ep
from cowork.api.v1.endpoints.connectors import submissions as submissions_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.handlers import probe as probe_handler
from cowork.principal import personal_partition_id
from cowork.schemas.connectors import DirectSaveRequest, SubmitFormRequest
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
    [{"http_path": NEW_PATH}, {"catalog": "main"}, {"schema": "sales"}],
    ids=["another-warehouse", "an-added-catalog", "an-added-schema"],
)
async def test_an_edit_that_keeps_the_token_and_changes_a_listed_safe_field_saves_in_place(
    probes, changed,
):
    slug = persist_connection("databricks", PAT, "", SAVED)

    stream = await _submit(_edit(**changed), name=slug)

    assert _outcome(stream)["_is_success"] is True
    [probe] = probes
    assert probe["credentials"] == {**SAVED, **changed}
    [record] = _records()
    assert record["name"] == slug
    assert record["fields"]["access_token"] == TOKEN
    assert all(record["fields"][key] == value for key, value in changed.items())


@pytest.mark.parametrize(
    "changed",
    [
        {"workspace_url": OTHER_HOST},
        {"workspace_url": OTHER_HOST, "http_path": NEW_PATH},
        {"server_hostname": "collector.example.net"},
        {"ssl_verify": "false"},
    ],
    ids=["another-host", "another-host-and-warehouse", "an-added-host-field", "an-added-tls-flag"],
)
async def test_an_edit_that_keeps_the_token_and_changes_an_unlisted_field_is_refused_before_any_probe(
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


def test_a_desktop_reconnect_that_leaves_out_the_token_cannot_move_it_to_another_host(stores):
    slug = persist_connection("databricks", PAT, "", SAVED)
    [before] = _records()
    body = DirectSaveRequest(
        connector_id="databricks", method=PAT, name=slug, replace_existing=True,
        values={"workspace_url": OTHER_HOST, "http_path": OLD_PATH},
    )

    with pytest.raises(HTTPException) as refused:
        connections_ep.save_connection_direct(body, LOCAL_SCOPE)

    assert refused.value.status_code == 422
    assert refused.value.detail == KEPT_SECRET_MOVED
    assert _records() == [before]


def test_a_desktop_reconnect_that_leaves_out_the_token_on_the_same_host_keeps_it(stores):
    slug = persist_connection("databricks", PAT, "", SAVED)
    body = DirectSaveRequest(
        connector_id="databricks", method=PAT, name=slug, replace_existing=True,
        values={"workspace_url": HOST, "http_path": NEW_PATH},
    )

    saved = connections_ep.save_connection_direct(body, LOCAL_SCOPE)

    assert saved["name"] == slug
    [record] = _records()
    assert record["fields"]["access_token"] == TOKEN
    assert record["fields"]["http_path"] == NEW_PATH


def test_a_desktop_reconnect_of_a_connection_marked_needs_reconnect_keeps_its_stored_secret(stores):
    signed_in = {
        "access_token": "ads-oauth-token-old", "developer_token": "ads-developer-token-9d1e",
        "account_email": "ops@example.net", "token_url": "https://oauth2.googleapis.com/token",
        "auth_type": "oauth",
    }
    slug = persist_connection("google_ads", "browser_oauth_builtin", "", signed_in)
    connections_ep.patch_connection_token(
        "google_ads", slug, connections_ep.PatchTokenBody(status="needs_reconnect"), LOCAL_SCOPE,
    )
    reconnect = {k: v for k, v in signed_in.items() if k != "developer_token"}
    body = DirectSaveRequest(
        connector_id="google_ads", method="browser_oauth_builtin", name=slug, replace_existing=True,
        values={**reconnect, "access_token": "ads-oauth-token-new"},
    )

    saved = connections_ep.save_connection_direct(body, LOCAL_SCOPE)

    assert saved["name"] == slug
    fields = own_vault_for_scope(LOCAL_SCOPE).read_record("google_ads", slug)["fields"]
    assert fields["access_token"] == "ads-oauth-token-new"
    assert fields["developer_token"] == "ads-developer-token-9d1e"
    assert "status" not in fields


def test_a_direct_save_of_a_genie_connection_is_refused_and_writes_nothing(stores):
    body = DirectSaveRequest(
        connector_id="databricks", method="genie-mcp", name="",
        values={"workspace_url": "adb-1234567890123456.7.azuredatabricks.net", "client_id": "principal",
                "client_secret": "submitted-secret"},
    )

    with pytest.raises(HTTPException) as refused:
        connections_ep.save_connection_direct(body, LOCAL_SCOPE)

    assert refused.value.status_code == 400
    assert refused.value.detail == databricks_genie.DIRECT_SAVE_REFUSAL
    assert _records() == []

GENIE_HOST = "adb-1234567890123456.7.azuredatabricks.net"
GENIE_SAVED = {"workspace_url": GENIE_HOST, "client_id": "principal", "_client_secret": "stored-secret"}


def _fake_genie_support(mints: list[str]) -> databricks_genie.GenieSupport:
    """Genie support whose host check strips the scheme and whose token mint only records the call."""

    class _Invalid(ValueError):
        """A refused workspace address or space ID."""

    async def _mint(host: str, client_id: str, secret: str) -> str:
        """Record that a token was minted."""
        mints.append(client_id)
        return "minted"

    helpers = SimpleNamespace(
        canonical_workspace_host=lambda url: url.removeprefix("https://"),
        genie_mcp_url=lambda host, space_id: f"https://{host}/api/2.0/mcp/genie",
        mint_token=_mint,
        InvalidWorkspaceHost=_Invalid,
        InvalidGenieSpace=_Invalid,
    )
    return databricks_genie.GenieSupport(databricks=helpers, mcp_session=None)


async def test_a_kept_genie_secret_is_not_reused_for_another_client(stores, monkeypatch):
    mints: list[str] = []
    monkeypatch.setattr(databricks_genie, "_anton_genie", lambda: _fake_genie_support(mints))
    slug = persist_connection("databricks", "genie-mcp", "", GENIE_SAVED)
    [before] = _records()
    edit = {"workspace_url": GENIE_HOST, "client_id": "another-principal", "client_secret": VAULT_KEEP_SENTINEL}

    with pytest.raises(databricks_genie.GenieConnectionError) as refused:
        await databricks_genie.save_genie_connection(LOCAL_SCOPE, edit, slug)

    assert refused.value.message == databricks_genie.SECRET_NEEDED
    assert mints == []
    assert _records() == [before]


def test_a_genie_save_binds_a_kept_secret_like_every_connection(stores):
    slug = persist_connection("databricks", "genie-mcp", "", GENIE_SAVED)
    kept = {**GENIE_SAVED, "_client_secret": VAULT_KEEP_SENTINEL}

    with pytest.raises(KeptSecretMovedError):
        persist_connection("databricks", "genie-mcp", slug, {**kept, "client_id": "another-principal"})
    persist_connection("databricks", "genie-mcp", slug, {**kept, "genie_space_id": "a" * 32})

    [record] = _records()
    assert record["fields"]["_client_secret"] == "stored-secret"
    assert record["fields"]["client_id"] == "principal"
    assert record["fields"]["genie_space_id"] == "a" * 32


def test_a_connector_with_no_listed_safe_field_keeps_a_secret_only_when_every_field_matches(stores):
    vault = own_vault_for_scope(LOCAL_SCOPE)
    saved = {"api_key": "stored-api-key-7f3a", "workspace_gid": "1201"}
    slug = persist_connection("example-tool", None, "", saved)

    with pytest.raises(KeptSecretMovedError):
        persist_connection("example-tool", None, slug, {"api_key": VAULT_KEEP_SENTINEL, "workspace_gid": "1202"})
    persist_connection("example-tool", None, slug, {"api_key": VAULT_KEEP_SENTINEL, "workspace_gid": "1201"})

    assert vault.read_record("example-tool", slug)["fields"]["api_key"] == "stored-api-key-7f3a"


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


@pytest.mark.parametrize("ending", ["success", "crash", "save-failure"])
async def test_a_kept_secret_is_never_logged(probes, monkeypatch, ending):
    calls: list[tuple[str, tuple, dict]] = []

    class _Log:
        """A logger that keeps every call it is given, with its arguments."""

        def _record(self, msg: str, *args: object, **kwargs: object) -> None:
            """Keep the message, its arguments and its keyword arguments."""
            calls.append((str(msg), args, kwargs))

        debug = info = warning = error = exception = _record

    class _CrashingProbe:
        """A CredentialProbe that fails with the token it was given in its error."""

        def __init__(self, **kwargs):
            """Keep the token the probe was handed."""
            self.token = kwargs["credentials"]["access_token"]

        async def run(self):
            """Raise before any verdict, quoting the token."""
            if self.token:
                raise RuntimeError(f"rejected {self.token}")
            yield "verdict", ProbeOutcome(status="success")

    if ending == "crash":
        monkeypatch.setattr(probe_handler, "CredentialProbe", _CrashingProbe)
    monkeypatch.setattr(probe_handler, "logger", _Log())
    slug = persist_connection("databricks", PAT, "", SAVED)

    def _failing_save(*args: object, **kwargs: object) -> str:
        """Fail the save after a working check, quoting the stored token."""
        raise RuntimeError(f"vault refused {TOKEN}")

    if ending == "save-failure":
        monkeypatch.setattr(probe_handler, "persist_connection", _failing_save)

    stream = await _submit(_edit(user_label="Warehouse"), name=slug)

    assert TOKEN not in stream
    for msg, args, kwargs in calls:
        assert TOKEN not in msg
        assert all(TOKEN not in str(arg) for arg in args)
        assert not kwargs.get("exc_info")
    if ending in ("crash", "save-failure"):
        assert any("RuntimeError" in str(arg) for _, args, _ in calls for arg in args)


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
            yield "scratchpad", {"action": "end", "name": "b", "code": "print(base64.b64encode(t))"}
            yield "scratchpad", {"action": "result", "content": json.dumps({"stdout": _b64(token)})}
            yield "text", f"Encoded {quote(token, safe='')} and {_b64(token)}."
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
    assert _b64(TOKEN) not in stream
    assert "[DS_CONNECTION_CHECK_KEPT__KEPT_0]" in stream
    assert "print(" in stream
    assert _outcome(stream)["_is_success"] is False


class _SilentLog:
    """A logger that drops every line, so a crash's traceback stays out of the test output."""

    def _drop(self, msg: str, *args: object, **kwargs: object) -> None:
        """Drop the line."""

    debug = info = warning = error = exception = _drop


def _b64(value: str) -> str:
    """``value`` in standard base64."""
    return base64.b64encode(value.encode()).decode()


def test_a_later_step_of_a_multi_step_save_cannot_move_a_stored_secret(stores):
    vault = own_vault_for_scope(LOCAL_SCOPE)
    slug = persist_connection("example-tool", None, "", {"host": "a.example.net", "api_key": "stored-api-key-7f3a"})
    before = vault.read_record("example-tool", slug)

    with pytest.raises(KeptSecretMovedError):
        persist_connection("example-tool", None, slug, {"host": "b.example.net"}, extends=True)

    assert vault.read_record("example-tool", slug) == before


def test_a_later_step_of_a_multi_step_save_may_add_fields_beside_a_stored_secret(stores):
    vault = own_vault_for_scope(LOCAL_SCOPE)
    slug = persist_connection("example-tool", None, "", {"host": "a.example.net", "api_key": "stored-api-key-7f3a"})

    persist_connection("example-tool", None, slug, {"account_email": "ops@example.net"}, extends=True)

    fields = vault.read_record("example-tool", slug)["fields"]
    assert fields["api_key"] == "stored-api-key-7f3a"
    assert fields["host"] == "a.example.net"
    assert fields["account_email"] == "ops@example.net"


async def test_an_edit_submit_of_a_browser_sign_in_connection_is_refused_and_changes_nothing(probes):
    vault = own_vault_for_scope(LOCAL_SCOPE)
    slug = persist_connection(
        "hubspot", "mcp", "", {"access_token": "hubspot-oauth-token-51c2", "account_email": "ops@example.net"},
        default_fields={"_access_mode": "none"},
    )
    before = vault.read_record("hubspot", slug)
    request = SubmitFormRequest(
        connector_id="hubspot", method="mcp", name=slug,
        values={"access_token": VAULT_KEEP_SENTINEL, "account_email": "ops@example.net"},
    )

    with pytest.raises(HTTPException) as refused:
        await submissions_ep.submit_form(request, LOCAL_SCOPE)

    assert refused.value.status_code == 400
    assert refused.value.detail == submissions_ep.BROWSER_SIGN_IN_EDIT_REFUSAL
    assert probes == []
    assert vault.read_record("hubspot", slug) == before
    assert before["fields"]["_access_mode"] == "none"
