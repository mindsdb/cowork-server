"""Connecting Databricks Genie with a service principal.

The form's submission runs a fixed check instead of the LLM probe: mint a
token for the service principal, open Genie's MCP server with it, and list its
tools. The installed anton does not ship the Databricks helpers yet, so each
test patches the gate to a fake of them; the fake mirrors their contract
(errors raised, values returned) and records every call it gets.
"""
from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from fastapi import HTTPException
from sqlalchemy import delete

from cowork.api.v1.endpoints.connectors import submissions as submissions_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_open_session
from cowork.handlers import probe as probe_handler
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.schemas.connectors import SubmitFormRequest
from cowork.services.connectors import databricks_genie
from cowork.services.connectors.identity import VAULT_KEEP_SENTINEL
from cowork.services.connectors.layered_vault import SHARED_CONNECTION_REFUSAL
from cowork.services.connectors.persist import own_vault_for_scope, persist_connection

SECRET = "sp-secret-value-7f3a"
TOKEN = "minted-access-token-9c1e"
HOST = "adb-1234567890123456.7.azuredatabricks.net"
CLIENT_ID = "6f1c2a9e-0b4d-4f7e-9a3c-1d2e3f4a5b6c"
SPACE_ID = "01f0a1b2c3d4e5f60718293a4b5c6d7e"
GENIE_TOOLS = ("genie_ask", "genie_poll_response", "genie_get_query_result", "genie_cancel_response")
GENIE_VALUES = {"workspace_url": f"https://{HOST}", "client_id": CLIENT_ID, "client_secret": SECRET}

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
ALICE = TenantScope(personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES)
BOB = TenantScope(personal=True, org_id=PARTITION_B, user_id=SUB_B, personal_resources=ALL_PERSONAL_RESOURCES)

_SUFFIXES = (".cloud.databricks.com", ".azuredatabricks.net", ".gcp.databricks.com")


class InvalidWorkspaceHost(ValueError):
    """Fake of anton's error for an address that is not a Databricks workspace."""


class InvalidGenieSpace(ValueError):
    """Fake of anton's error for a value that is not a Genie space id."""


class GenieCredentialsRefused(RuntimeError):
    """Fake of anton's error for refused client credentials."""


class GenieRequestRefused(RuntimeError):
    """Fake of anton's error for any other refused token request."""


class GenieUnavailable(RuntimeError):
    """Fake of anton's error for an unreachable workspace."""


class FakeGenie:
    """A stand-in for anton's Databricks helpers and MCP session.

    ``mint_error`` and ``connect_error`` make the token request or the MCP
    connect fail; ``tools`` is what the session lists. ``mints`` and
    ``sessions`` record each call, so a test can tell nothing was sent.
    """

    def __init__(self) -> None:
        """Start with a working workspace whose Genie lists every tool."""
        self.mint_error: Exception | None = None
        self.connect_error: Exception | None = None
        self.tools: tuple[str, ...] = GENIE_TOOLS
        self.mints: list[tuple[str, str, str]] = []
        self.sessions: list[tuple[str, str]] = []
        self.databricks = SimpleNamespace(
            InvalidWorkspaceHost=InvalidWorkspaceHost,
            InvalidGenieSpace=InvalidGenieSpace,
            GenieCredentialsRefused=GenieCredentialsRefused,
            GenieRequestRefused=GenieRequestRefused,
            GenieUnavailable=GenieUnavailable,
            canonical_workspace_host=self.canonical_workspace_host,
            genie_mcp_url=self.genie_mcp_url,
            mint_token=self.mint_token,
        )
        fake = self

        class _Session:
            """Fake McpSession: records its url and token, fails on connect when told to."""

            def __init__(self, *, url: str, access_token: str) -> None:
                """Record the address and token the session is opened with."""
                fake.sessions.append((url, access_token))

            async def __aenter__(self) -> _Session:
                """Connect, or raise ``connect_error``."""
                if fake.connect_error is not None:
                    raise fake.connect_error
                return self

            async def __aexit__(self, *exc_info: object) -> None:
                """Close; nothing to release."""
                return None

            async def list_tools(self) -> list[SimpleNamespace]:
                """The configured tools, each with a ``name``."""
                return [SimpleNamespace(name=name) for name in fake.tools]

        self.support = databricks_genie.GenieSupport(databricks=self.databricks, mcp_session=_Session)

    @staticmethod
    def canonical_workspace_host(value: str) -> str:
        """The bare lowercase host of a Databricks address, as anton's does."""
        text = value.strip()
        parts = urlsplit(text if "://" in text else f"https://{text}")
        host = parts.netloc.lower()
        if parts.scheme != "https" or parts.path not in ("", "/") or not host.endswith(_SUFFIXES):
            raise InvalidWorkspaceHost("The workspace host must be a Databricks workspace host.")
        return host

    def genie_mcp_url(self, host: str, space_id: str | None = None) -> str:
        """Genie One's address, or one space's when ``space_id`` is 32 hex characters."""
        base = f"https://{self.canonical_workspace_host(host)}"
        if space_id is None:
            return base + "/ai-gateway/mcp-services/system.ai.genie_one_mcp"
        cleaned = space_id.strip()
        if len(cleaned) != 32 or any(c not in "0123456789abcdef" for c in cleaned):
            raise InvalidGenieSpace("A Genie space id is 32 lowercase hex characters.")
        return base + f"/api/2.0/mcp/genie/{cleaned}"

    async def mint_token(self, host: str, client_id: str, client_secret: str) -> str:
        """Record the request; raise ``mint_error`` or return the token."""
        self.mints.append((host, client_id, client_secret))
        if self.mint_error is not None:
            raise self.mint_error
        return TOKEN


class RecordingLogger:
    """A module logger that keeps every message it is given, formatted."""

    def __init__(self) -> None:
        """Start with no lines."""
        self.lines: list[str] = []

    def _record(self, msg: str, *args: object, **kwargs: object) -> None:
        """Keep the formatted message and any keyword arguments."""
        self.lines.append(str(msg) % args if args else str(msg))
        self.lines.append(repr(kwargs))

    debug = info = warning = error = exception = _record


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
    with get_open_session() as db:
        db.exec(delete(Conversation).where(Conversation.org_id.in_([PARTITION_A, PARTITION_B])))
        db.exec(delete(Project).where(Project.org_id.in_([PARTITION_A, PARTITION_B])))
        db.commit()
    get_app_settings.cache_clear()


@pytest.fixture
def genie(monkeypatch, stores) -> FakeGenie:
    """The gate patched to a working fake, and the LLM probe made to fail the test if it runs."""
    fake = FakeGenie()
    monkeypatch.setattr(databricks_genie, "_anton_genie", lambda: fake.support)
    return fake


@pytest.fixture
def no_llm_probe(monkeypatch) -> list[str]:
    """Record any attempt to build the LLM client or the credential probe."""
    attempts: list[str] = []

    def _probe(**kwargs):
        """Stand-in for CredentialProbe that records the attempt and fails."""
        attempts.append("probe")
        raise AssertionError("the LLM probe ran")

    def _llm(settings=None):
        """Stand-in for the LLM client builder that records the attempt and fails."""
        attempts.append("llm")
        raise AssertionError("an LLM client was built")

    monkeypatch.setattr(probe_handler, "CredentialProbe", _probe)
    monkeypatch.setattr(probe_handler.ProbeHandler, "_build_llm_client", staticmethod(_llm))
    return attempts


async def _connect(
    values: dict, *, scope: TenantScope = LOCAL_SCOPE, name: str = "", method: str = "genie-mcp",
    conversation_id: str | None = None,
) -> str:
    """Submit the Databricks form through the submissions route and return the whole stream."""
    response = await submissions_ep.submit_form(
        SubmitFormRequest(
            connector_id="databricks", method=method, name=name, values=dict(values),
            conversation_id=conversation_id,
        ),
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


async def test_a_working_service_principal_is_saved_without_the_llm_probe(genie, no_llm_probe):
    stream = await _connect(GENIE_VALUES)

    assert no_llm_probe == []
    assert _outcome(stream)["_is_success"] is True
    [record] = _records()
    fields = record["fields"]
    assert fields["_method"] == "genie-mcp"
    assert fields["workspace_url"] == HOST
    assert fields["client_id"] == CLIENT_ID
    assert fields["_client_secret"] == SECRET
    assert "client_secret" not in fields
    assert "genie_space_id" not in fields
    assert "_client_secret" in record["secure_keys"]
    assert genie.mints == [(HOST, CLIENT_ID, SECRET)]
    assert genie.sessions == [(f"https://{HOST}/ai-gateway/mcp-services/system.ai.genie_one_mcp", TOKEN)]


async def test_a_space_id_is_stored_and_checked_against_its_space(genie, no_llm_probe):
    await _connect({**GENIE_VALUES, "genie_space_id": SPACE_ID})

    [record] = _records()
    assert record["fields"]["genie_space_id"] == SPACE_ID
    assert genie.sessions == [(f"https://{HOST}/api/2.0/mcp/genie/{SPACE_ID}", TOKEN)]


async def test_it_works_without_a_model_key(genie, no_llm_probe, monkeypatch):
    def _no_settings(session):
        """A settings read that fails, as it does with no model key."""
        raise RuntimeError("no model key is configured")

    monkeypatch.setattr(probe_handler, "_read_probe_settings", _no_settings)

    stream = await _connect(GENIE_VALUES)

    assert _outcome(stream)["_is_success"] is True
    assert len(_records()) == 1


async def test_the_secret_is_not_in_the_reply_the_turn_or_the_log(genie, no_llm_probe, monkeypatch, tmp_path):
    handler_log, genie_log = RecordingLogger(), RecordingLogger()
    monkeypatch.setattr(probe_handler, "logger", handler_log)
    monkeypatch.setattr(databricks_genie, "logger", genie_log)
    conversation_id = uuid.uuid4()
    saved_turns: list[dict] = []

    def _read_conversation(session, *, conversation_id):
        """The probe's conversation, without a database row."""
        return probe_handler._ProbeConversation(id=conversation_id, workspace_path=tmp_path)

    def _save_probe_turn(session, *, conversation_id, text, events):
        """Keep the turn the handler would save, and give it an id."""
        saved_turns.append({"text": text, "events": events})
        return uuid.uuid4()

    monkeypatch.setattr(probe_handler, "_read_conversation", _read_conversation)
    monkeypatch.setattr(probe_handler, "_save_probe_turn", _save_probe_turn)

    stream = await _connect(GENIE_VALUES, conversation_id=str(conversation_id))
    genie.mint_error = GenieCredentialsRefused(f"refused {SECRET}")
    refused = await _connect({**GENIE_VALUES, "client_id": "another-principal"})

    [saved_turn] = saved_turns
    assert HOST in saved_turn["text"]
    for text in (stream, refused, json.dumps(saved_turns, default=str), *handler_log.lines, *genie_log.lines):
        assert SECRET not in text
        assert TOKEN not in text
    [record] = _records()
    assert [k for k, v in record["fields"].items() if v == SECRET] == ["_client_secret"]


async def test_refused_credentials_say_so_and_save_nothing(genie, no_llm_probe):
    genie.mint_error = GenieCredentialsRefused("Databricks refused the service principal's client ID or secret")

    stream = await _connect(GENIE_VALUES)

    assert databricks_genie.CREDENTIALS_REFUSED in _text(stream)
    assert _outcome(stream)["_is_success"] is False
    assert _records() == []


async def test_any_other_refusal_says_databricks_refused_and_saves_nothing(genie, no_llm_probe):
    genie.mint_error = GenieRequestRefused("Databricks refused the token request (HTTP 403)")

    stream = await _connect(GENIE_VALUES)

    assert databricks_genie.REQUEST_REFUSED.format(host=HOST) in _text(stream)
    assert "HTTP 403" not in stream
    assert _records() == []


async def test_a_rejected_token_at_connect_says_databricks_refused(genie, no_llm_probe):
    genie.connect_error = ExceptionGroup("connect failed", [RuntimeError("401 Unauthorized for token xyz")])

    stream = await _connect(GENIE_VALUES)

    assert databricks_genie.REQUEST_REFUSED.format(host=HOST) in _text(stream)
    assert "Unauthorized" not in stream
    assert _records() == []


async def test_an_unreachable_workspace_says_so_and_saves_nothing(genie, no_llm_probe):
    genie.mint_error = GenieUnavailable("The Databricks workspace could not be reached")

    stream = await _connect(GENIE_VALUES)

    assert databricks_genie.UNREACHABLE.format(host=HOST) in _text(stream)
    assert _records() == []


async def test_genie_not_enabled_says_so(genie, no_llm_probe):
    genie.tools = ("some_other_tool",)

    stream = await _connect(GENIE_VALUES)

    assert databricks_genie.GENIE_NOT_ENABLED.format(host=HOST) in _text(stream)
    assert _records() == []


async def test_an_invalid_space_id_is_refused_before_any_request(genie, no_llm_probe):
    stream = await _connect({**GENIE_VALUES, "genie_space_id": "../../api/2.0/secrets"})

    assert databricks_genie.INVALID_SPACE in _text(stream)
    assert "secrets" not in stream
    assert genie.mints == []
    assert _records() == []


async def test_a_pasted_browser_address_is_normalized(genie, no_llm_probe):
    await _connect({**GENIE_VALUES, "workspace_url": f"  HTTPS://{HOST.upper()}/  "})

    [record] = _records()
    assert record["fields"]["workspace_url"] == HOST
    assert genie.mints[0][0] == HOST


async def test_a_non_databricks_address_is_refused_before_any_request(genie, no_llm_probe):
    stream = await _connect({**GENIE_VALUES, "workspace_url": "https://attacker.example.com"})

    assert databricks_genie.WRONG_ADDRESS in _text(stream)
    assert "attacker" not in stream
    assert genie.mints == []
    assert genie.sessions == []
    assert _records() == []


async def test_reconnecting_the_same_principal_and_workspace_updates_the_connection(genie, no_llm_probe):
    await _connect(GENIE_VALUES)
    await _connect({**GENIE_VALUES, "workspace_url": f"{HOST}", "client_secret": "rotated-secret"})

    [record] = _records()
    assert record["fields"]["_client_secret"] == "rotated-secret"


async def test_an_unchanged_secret_on_edit_keeps_the_stored_secret(genie, no_llm_probe):
    await _connect(GENIE_VALUES)
    [first] = _records()

    stream = await _connect(
        {**GENIE_VALUES, "client_secret": VAULT_KEEP_SENTINEL, "genie_space_id": SPACE_ID},
        name=first["name"],
    )

    assert _outcome(stream)["_is_success"] is True
    assert genie.mints[-1] == (HOST, CLIENT_ID, SECRET)
    [record] = _records()
    assert record["name"] == first["name"]
    assert record["fields"]["_client_secret"] == SECRET
    assert record["fields"]["genie_space_id"] == SPACE_ID
    assert VAULT_KEEP_SENTINEL not in json.dumps(record)


async def test_an_unchanged_secret_with_nothing_stored_asks_for_the_secret(genie, no_llm_probe):
    stream = await _connect({**GENIE_VALUES, "client_secret": VAULT_KEEP_SENTINEL})

    assert databricks_genie.SECRET_NEEDED in _text(stream)
    assert genie.mints == []
    assert _records() == []


@pytest.fixture
def sharing(personal_deployment, managed_dir, genie):
    """Alice's Genie connection, shared with everyone through the managed registry."""
    get_app_settings.cache_clear()
    slug = persist_connection(
        "databricks", "genie-mcp", "",
        {"workspace_url": HOST, "client_id": CLIENT_ID, "_client_secret": SECRET},
        scope=ALICE,
    )
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_connections_file.write_text(json.dumps([
        {"user_id": SUB_A, "partition_id": PARTITION_A, "engine": "databricks", "name": slug},
    ]))
    return slug


async def test_a_shared_or_managed_match_is_refused_with_a_message(sharing, genie, no_llm_probe):
    alices = own_vault_for_scope(ALICE).read_record("databricks", sharing)

    fresh = await _connect({**GENIE_VALUES, "client_secret": "bobs-own-secret"}, scope=BOB)
    kept = await _connect({**GENIE_VALUES, "client_secret": VAULT_KEEP_SENTINEL}, scope=BOB, name=sharing)

    for stream in (fresh, kept):
        assert SHARED_CONNECTION_REFUSAL in _text(stream)
        assert _outcome(stream)["_is_success"] is False
    assert all(mint[2] != SECRET for mint in genie.mints)
    assert _records(BOB) == []
    assert own_vault_for_scope(ALICE).read_record("databricks", sharing) == alices


async def test_org_mode_refuses_genie_mcp(genie, no_llm_probe, org_deployment, monkeypatch):
    staged: list = []
    monkeypatch.setattr(submissions_ep.store, "stage", lambda **kwargs: staged.append(kwargs))
    scope = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")

    with pytest.raises(HTTPException) as refused:
        await _connect(GENIE_VALUES, scope=scope)

    assert refused.value.status_code == 400
    assert refused.value.detail == databricks_genie.ORG_MODE_REFUSAL
    assert staged == []
    assert genie.mints == []


async def test_an_anton_without_genie_support_says_this_build_cannot_connect_it(stores, no_llm_probe, monkeypatch):
    monkeypatch.setattr(databricks_genie, "_anton_genie", lambda: None)

    stream = await _connect(GENIE_VALUES)

    assert databricks_genie.NO_GENIE_SUPPORT in _text(stream)
    assert _records() == []


def test_the_installed_anton_without_the_helpers_reads_as_no_support(monkeypatch):
    monkeypatch.setattr(databricks_genie, "_import_genie_modules", lambda: (SimpleNamespace(), object))

    assert databricks_genie._anton_genie() is None


async def test_other_databricks_methods_still_use_the_llm_probe(genie, monkeypatch):
    from cowork.services.connectors.probe import ProbeOutcome

    probes: list[dict] = []

    class _Probe:
        """A CredentialProbe that records its arguments and succeeds."""

        def __init__(self, **kwargs):
            """Record the probe's arguments."""
            probes.append(kwargs)

        async def run(self):
            """Yield a success verdict."""
            yield "verdict", ProbeOutcome(status="success", summary="Connection works.")

    monkeypatch.setattr(probe_handler, "CredentialProbe", _Probe)
    monkeypatch.setattr(probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object()))

    stream = await _connect(
        {"workspace_url": f"https://{HOST}", "access_token": "dapi-token", "http_path": "/sql/1.0/warehouses/abc"},
        method="personal-access-token",
    )

    assert len(probes) == 1
    assert probes[0]["engine"] == "databricks"
    assert genie.mints == []
    assert _outcome(stream)["_is_success"] is True
    [record] = _records()
    assert record["fields"]["_method"] == "personal-access-token"
