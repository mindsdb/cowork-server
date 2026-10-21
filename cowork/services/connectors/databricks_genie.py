"""The save-time check for a Databricks Genie connection.

A Genie connection is checked without the LLM probe: a token is minted for the
service principal, Genie's MCP server is opened with it, and its tools are
listed. Each failure maps to one fixed message for the form. No message
carries an exception's text, the token, the secret, or any submitted value
except the canonical host. ``save_genie_connection`` stores the secret under
the hidden name ``_client_secret``, which every vault leaves out of the
agent's environment.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.identity import VAULT_KEEP_SENTINEL, derive_connection_name
from cowork.services.connectors.layered_vault import SHARED_CONNECTION_REFUSAL, SharedConnectionError
from cowork.services.connectors.persist import persist_connection, vault_for_scope

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope

logger = logging.getLogger(__name__)

CONNECTOR_ID = "databricks"
GENIE_METHOD = "genie-mcp"
SECRET_FIELD = "_client_secret"
GENIE_ASK_TOOL = "genie_ask"

WRONG_ADDRESS = (
    "That is not a Databricks workspace address. Use the address of your workspace, "
    "such as https://adb-1234567890123456.7.azuredatabricks.net."
)
CREDENTIALS_REFUSED = (
    "Databricks refused the service principal's client ID or secret. Check both and try again."
)
REQUEST_REFUSED = (
    "Databricks refused the request on {host}. Check that the service principal has Can use "
    "on the SQL warehouse and Can run on the Genie Agent."
)
UNREACHABLE = "The Databricks workspace {host} could not be reached. Try again in a moment."
GENIE_NOT_ENABLED = "Genie is not enabled on {host}. Ask a workspace admin to turn it on, then try again."
INVALID_SPACE = (
    "That is not a Genie space ID. It is the 32 character ID in the space's address; "
    "leave it blank to use every space."
)
NO_GENIE_SUPPORT = "This build cannot connect Databricks Genie. Update the app and try again."
SECRET_NEEDED = "Enter the service principal's secret."
SAVE_FAILED = "The connection worked but could not be saved. Try again."
ORG_MODE_REFUSAL = "Databricks Genie is not available on this deployment."


@dataclass(frozen=True)
class GenieSupport:
    """anton's Databricks helpers and the MCP session class they are checked with."""

    databricks: Any
    mcp_session: Any


@dataclass(frozen=True)
class GenieCheck:
    """A Genie connection that passed the check.

    ``host`` is the canonical workspace host, ``client_secret`` the secret the
    token was minted with (the stored one when the form kept it), and
    ``space_id`` the stripped space id, or None for Genie One.
    """

    host: str
    client_id: str
    client_secret: str
    space_id: str | None


@dataclass(frozen=True)
class SavedGenieConnection:
    """A checked and saved Genie connection: its slug, its stored user label, and its canonical host."""

    slug: str
    user_label: str | None
    host: str


class GenieConnectionError(Exception):
    """The check failed; ``message`` is the fixed text the form shows."""

    def __init__(self, message: str) -> None:
        """Keep ``message`` as both the error text and its ``message`` attribute."""
        super().__init__(message)
        self.message = message


def _import_genie_modules() -> tuple[ModuleType, Any]:
    """Import anton's Databricks module and McpSession. Raises ImportError when absent."""
    # Optional dependency: the anton release with the Databricks helpers may not be installed.
    from anton.core.mcp import databricks
    from anton.core.mcp.client import McpSession

    return databricks, McpSession


def _anton_genie() -> GenieSupport | None:
    """anton's Genie support, or None when the installed anton predates it.

    Same shape as the harness's ``_anton_mcp_wiring`` gate: a missing
    capability degrades this one method instead of raising. An anton whose
    ``databricks`` module lacks any helper the check calls counts as absent.
    """
    try:
        databricks, mcp_session = _import_genie_modules()
    except ImportError:
        logger.info("Installed anton has no Databricks Genie support; the Genie method is inactive.")
        return None
    needed = ("mint_token", "genie_mcp_url", "canonical_workspace_host")
    if not all(hasattr(databricks, name) for name in needed):
        logger.info("Installed anton's Databricks module lacks the Genie helpers; the Genie method is inactive.")
        return None
    return GenieSupport(databricks=databricks, mcp_session=mcp_session)


def is_genie_method(connector_id: str, method: str | None) -> bool:
    """Whether a submission is the Databricks Genie method."""
    return connector_id == CONNECTOR_ID and method == GENIE_METHOD


def _clean(values: dict[str, str], key: str) -> str:
    """The stripped text of ``values[key]``, or an empty string."""
    return str(values.get(key) or "").strip()


async def check_genie_connection(
    values: dict[str, str],
    *,
    stored_secret: Callable[[str], str | None] | None = None,
) -> GenieCheck:
    """Check a Genie connection the way its form submitted it.

    ``values`` holds ``workspace_url``, ``client_id``, ``client_secret`` and an
    optional ``genie_space_id``. A ``client_secret`` equal to the keep sentinel
    is replaced by ``stored_secret(host)``, called with the canonical host.
    The host and space id are checked before any request; then one token is
    minted and Genie's MCP server is opened and its tools listed. Returns the
    checked GenieCheck. Raises GenieConnectionError with a fixed message,
    keeping the cause. Sends network requests to the workspace; saves nothing.
    """
    support = _anton_genie()
    if support is None:
        raise GenieConnectionError(NO_GENIE_SUPPORT)
    databricks = support.databricks

    try:
        host = databricks.canonical_workspace_host(_clean(values, "workspace_url"))
    except databricks.InvalidWorkspaceHost as exc:
        raise GenieConnectionError(WRONG_ADDRESS) from exc

    space_id = _clean(values, "genie_space_id") or None
    try:
        url = databricks.genie_mcp_url(host, space_id)
    except databricks.InvalidGenieSpace as exc:
        raise GenieConnectionError(INVALID_SPACE) from exc

    client_id = _clean(values, "client_id")
    secret = str(values.get("client_secret") or "")
    if secret == VAULT_KEEP_SENTINEL:
        secret = (stored_secret(host) if stored_secret is not None else None) or ""
    if not secret:
        raise GenieConnectionError(SECRET_NEEDED)

    try:
        token = await databricks.mint_token(host, client_id, secret)
    except databricks.GenieCredentialsRefused as exc:
        raise GenieConnectionError(CREDENTIALS_REFUSED) from exc
    except databricks.GenieUnavailable as exc:
        raise GenieConnectionError(UNREACHABLE.format(host=host)) from exc
    except databricks.GenieRequestRefused as exc:
        raise GenieConnectionError(REQUEST_REFUSED.format(host=host)) from exc

    try:
        async with support.mcp_session(url=url, access_token=token) as session:
            tools = await session.list_tools()
    except Exception as exc:
        # A rejected token surfaces at connect as an ExceptionGroup whose text can carry the reply.
        raise GenieConnectionError(REQUEST_REFUSED.format(host=host)) from exc
    if GENIE_ASK_TOOL not in {tool.name for tool in tools}:
        raise GenieConnectionError(GENIE_NOT_ENABLED.format(host=host))
    return GenieCheck(host=host, client_id=client_id, client_secret=secret, space_id=space_id)


def genie_connection_name(name: str, check_values: dict[str, str]) -> str | None:
    """The slug a Genie save targets: ``name`` when given, else the one derived from the spec's ``name_from``."""
    return name.strip() or derive_connection_name(CONNECTOR_ID, GENIE_METHOD, check_values)


def stored_secret_reader(
    vault: Any, *, name: str, client_id: str, can_change: Callable[[str], bool]
) -> Callable[[str], str | None]:
    """A ``stored_secret`` callback for check_genie_connection.

    It reads ``_client_secret`` from the record the save would update: the one
    named ``name``, else the one ``name_from`` derives from ``client_id`` and
    the canonical host. A record the caller can't change, such as one shared
    with them, raises GenieConnectionError with the shared refusal, so its
    secret is never used. The secret is returned only when the record's
    ``workspace_url`` and ``client_id`` equal the submitted ones, else None.
    Reads the vault; writes nothing.
    """

    def _read(host: str) -> str | None:
        slug = genie_connection_name(name, {"client_id": client_id, "workspace_url": host})
        if not slug:
            return None
        record = vault.read_record(CONNECTOR_ID, slug)
        if record is None:
            return None
        if not can_change(slug):
            raise GenieConnectionError(SHARED_CONNECTION_REFUSAL)
        fields = record.get("fields") or {}
        # A kept secret must not be sent to a different workspace or as another principal.
        if fields.get("workspace_url") != host or fields.get("client_id") != client_id:
            return None
        value = fields.get(SECRET_FIELD)
        return value if isinstance(value, str) else None

    return _read


def stored_fields(check: GenieCheck, *, keep_secret: bool) -> dict[str, str]:
    """The fields a checked Genie connection is saved with.

    With ``keep_secret`` the secret is saved as the keep sentinel, so
    persist_connection resolves it against the stored record and treats the
    save as an edit of that record. ``_method`` comes from persist_connection.
    """
    fields = {
        "workspace_url": check.host,
        "client_id": check.client_id,
        SECRET_FIELD: VAULT_KEEP_SENTINEL if keep_secret else check.client_secret,
    }
    if check.space_id:
        fields["genie_space_id"] = check.space_id
    return fields


async def save_genie_connection(
    scope: TenantScope,
    credentials: dict[str, str],
    name: str,
    *,
    label: str = "",
    user_label: str = "",
) -> SavedGenieConnection:
    """Check a submitted Genie connection and save it to ``scope``'s vault.

    ``credentials`` are the form's values, ``name`` the connection being
    edited or an empty string. A kept secret resolves against the record the
    save updates, and only when the caller can change that record. Returns
    the SavedGenieConnection. Raises GenieConnectionError with a fixed
    message when the check fails, the record is shared or managed, or the
    save fails. Writes one vault record; sends network requests through
    check_genie_connection.
    """
    vault = vault_for_scope(scope)
    connections = ConnectionsService(scope)
    reader = stored_secret_reader(
        vault,
        name=name,
        client_id=_clean(credentials, "client_id"),
        can_change=lambda slug: connections.can_change(CONNECTOR_ID, slug),
    )
    check = await check_genie_connection(credentials, stored_secret=reader)
    keep_secret = credentials.get("client_secret") == VAULT_KEEP_SENTINEL
    try:
        slug = persist_connection(
            CONNECTOR_ID, GENIE_METHOD, name, stored_fields(check, keep_secret=keep_secret),
            label=label, user_label=user_label, vault=vault, scope=scope,
            # stored_secret_reader already bound a kept secret to the record's host and client ID.
            secret_bound_by_caller=keep_secret,
        )
    except SharedConnectionError as exc:
        raise GenieConnectionError(exc.detail) from exc
    except Exception as exc:
        # The error's text is not logged: a vault error can quote the record it failed on.
        logger.error("Could not save a Databricks Genie connection (%s)", type(exc).__name__)
        raise GenieConnectionError(SAVE_FAILED) from exc
    record = vault.read_record(CONNECTOR_ID, slug) or {}
    saved_label = str((record.get("fields") or {}).get("_user_label", "")).strip() or None
    return SavedGenieConnection(slug=slug, user_label=saved_label, host=check.host)
