"""The set of routes that ask for the launch hand-off's window proof, pinned.

``WindowProof`` (cowork/api/v1/permissions.py) refuses a caller that holds
the bearer token and nothing else, on a server the desktop app launched with
the hand-off. A route that approves an agent's action, reveals or writes a
credential, or changes a gate on what the agent may do declares it, so agent
code on the same machine can't do those things for itself.

The pin matters in both directions, so the assertions run both ways. A route
that loses the declaration lets an agent holding the bearer token act as the
person again, and nothing else would notice. A route that gains it stops
working for every caller except Cowork's window and the desktop app's main
process, so adding one needs a written reason too. An entry with no route is
a stale pin that has stopped protecting anything.

tests/test_window_proof_matrix.py checks the other half: that the routes
declaring the class are the routes that refuse, and no other route does.
"""
from __future__ import annotations

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute

from cowork.api.v1.permissions import WindowProof
from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.server import create_app

#: (path, methods) -> what an agent holding the bearer token could do with
#: the route if it did not ask for the window proof.
WINDOW_PROOF_REASONS: dict[tuple[str, tuple[str, ...]], str] = {
    # Code mode.
    ("/api/v1/coding/sessions/{session_id}/approvals/{approval_id}", ("POST",)): (
        "decide the approval a Code task waits on, so the agent approves its own command or file change"
    ),
    ("/api/v1/coding/sessions/{session_id}/questions/{question_id}", ("POST",)): (
        "answer the question a Code task asked the person, so the agent answers itself"
    ),
    ("/api/v1/coding/sessions/{session_id}/mode-turns", ("POST",)): (
        "switch a Code task from plan to build and start the turn, which approves the agent's own plan"
    ),
    ("/api/v1/coding/sessions/{session_id}", ("PATCH",)): (
        "change a Code task's permission mode, network access, extra folders and writes switch, "
        "so the agent widens its own sandbox"
    ),
    ("/api/v1/coding/runtime/registration-token", ("POST",)): (
        "issue a registration token that lets any computer register itself as a Code runtime"
    ),
    ("/api/v1/coding/runs/{run_id}/connector-capabilities", ("POST",)): (
        "mint a connector grant token for a run, so the agent grants itself a connector's actions"
    ),
    # Settings and the agent's credentials.
    ("/api/v1/settings/reveal-key/{name}", ("GET",)): "read a provider's API key unmasked",
    ("/api/v1/settings/raw", ("GET",)): "read the server's dotenv verbatim, every provider secret in it",
    ("/api/v1/settings/raw", ("POST",)): (
        "write the server's dotenv and sync its keys into settings, so the agent sets any server setting"
    ),
    ("/api/v1/settings/", ("PUT",)): (
        "write any settings, provider keys and endpoints among them, so the agent points the model "
        "at a server it chose or lifts a limit the person set"
    ),
    ("/api/v1/settings/{key}", ("PUT",)): "write one setting, such as a provider key or endpoint",
    ("/api/v1/settings/{key}", ("DELETE",)): "delete a setting, which puts a limit the person set back to its default",
    ("/api/v1/runtime-credential/minds", ("PUT",)): "replace or clear the MindsHub credential the agent's turns spend",
    ("/api/v1/runtime-credential/browser", ("PUT",)): (
        "point the agent's browser at a loopback address and key the caller chose, or clear it"
    ),
    # Connections and their credentials.
    ("/api/v1/connectors/oauth/{engine}/credentials", ("GET",)): "read a built-in connector's OAuth client secret",
    ("/api/v1/connectors/oauth/{service}/start", ("POST",)): (
        "start a connector's OAuth sign-in, whose callback saves the account as a connection the agent can use"
    ),
    ("/api/v1/connectors/oauth/{engine}/mcp/identity", ("POST",)): (
        "make the server call a provider's MCP identity tools with an access token the caller supplies"
    ),
    ("/api/v1/connectors/connections/save", ("POST",)): (
        "write connector credentials to the vault unchecked, planting a connection in the person's name"
    ),
    ("/api/v1/connectors/connections/validate-and-save", ("POST",)): (
        "store a developer-tool token as a connection in the person's name"
    ),
    ("/api/v1/connectors/connections/{engine}/{name}/token", ("PATCH",)): "replace a connection's access token",
    ("/api/v1/connectors/connections/{engine}/{name}/app-held", ("PUT",)): (
        "replace the token and access mode the server holds for an MCP connection"
    ),
    ("/api/v1/connectors/connections/{engine}/{name}/access-mode", ("PATCH",)): (
        "raise a connection's tool access from read to write"
    ),
    ("/api/v1/connectors/connections/{engine}/{name}", ("DELETE",)): (
        "delete a connection and revoke its token at the provider"
    ),
    ("/api/v1/connectors/connections/custom_mcp", ("POST",)): (
        "create a custom MCP connection to any server address the caller names"
    ),
    ("/api/v1/connectors/connections/custom_mcp/{name}/held", ("PUT",)): (
        "replace a custom MCP connection's held settings and token"
    ),
    ("/api/v1/connectors/connections/custom_mcp/{name}/allowed-tools", ("PUT",)): (
        "widen the tools the agent may call on a custom MCP connection"
    ),
    ("/api/v1/connectors/custom/{connector_id}/connect", ("POST",)): (
        "store credentials for a custom connector as a connection in the person's name"
    ),
    ("/api/v1/connectors/submissions/allow-private", ("POST",)): (
        "allow a connector test to reach the private host the person was asked about"
    ),
    ("/api/v1/connectors/submissions/confirm", ("POST",)): (
        "save the connection whose test result the person was asked to confirm"
    ),
    ("/api/v1/connectors/submissions/cancel", ("POST",)): (
        "end the connector test the person was asked about, answering for them"
    ),
    ("/api/v1/connectors/sharepoint/connections", ("POST",)): (
        "create a SharePoint connection with a key the caller supplies"
    ),
    ("/api/v1/connectors/sharepoint/connections/{name}", ("PUT",)): "replace a SharePoint connection's key and sites",
    # Questions and approvals in chat, artifacts and the browser.
    ("/api/v1/responses/answer", ("POST",)): (
        "answer the question a turn is blocked on, so the agent answers itself"
    ),
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/agent-repairs/{repair_id}/decision", ("POST",)): (
        "accept or reject the agent's own repair of an artifact"
    ),
    ("/api/v1/browse/approve", ("POST",)): "approve a step the agent's browser held for the person's approval",
    ("/api/v1/browse/decline", ("POST",)): "skip a step the agent's browser held, answering for the person",
    # Messaging channels.
    ("/api/v1/channels/{channel_type}/config", ("PUT",)): (
        "point a messaging channel at a workspace and token the caller supplies, so outside messages reach the agent"
    ),
    ("/api/v1/channels/{channel_type}/config", ("DELETE",)): "remove a messaging channel's configuration and token",
    ("/api/v1/channels/{channel_type}/setup", ("POST",)): (
        "start a messaging channel, so people outside the app can message the agent"
    ),
    ("/api/v1/channels/{channel_type}/teardown", ("POST",)): "stop a messaging channel",
}


@pytest.fixture(scope="module")
def proof_routes() -> set[tuple[str, tuple[str, ...]]]:
    app = create_app()
    found = set()
    for route in app.routes:
        if not isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue
        if WindowProof not in declared_permissions(route):
            continue
        path, methods = route_key(route)
        found.add((path, tuple(m for m in methods if m != "HEAD")))
    return found


def test_no_route_asks_for_the_proof_without_a_written_reason(proof_routes):
    unpinned = sorted(proof_routes - set(WINDOW_PROOF_REASONS))
    assert unpinned == [], (
        "these routes ask for the window proof and have no recorded reason. Add an "
        "entry to WINDOW_PROOF_REASONS saying what an agent could do with the route "
        f"without it: {unpinned}"
    )


def test_no_reason_outlives_its_route(proof_routes):
    stale = sorted(set(WINDOW_PROOF_REASONS) - proof_routes)
    assert stale == [], (
        "these pins name a route that no longer asks for the window proof (or no "
        "longer exists). Declare WindowProof on it again, or delete the entry and "
        f"say why in the same commit: {stale}"
    )


def test_the_proof_set_is_small_enough_to_read(proof_routes):
    """A number someone notices moving, as the open-by-design pin keeps one.
    The bound is loose on purpose: it is a tripwire for a bulk change, not a
    budget to spend down."""
    assert len(proof_routes) <= 48, (
        f"{len(proof_routes)} routes now ask for the window proof. If that is "
        "right, raise the bound in the same commit that explains why."
    )
