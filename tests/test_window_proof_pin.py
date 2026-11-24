"""The set of routes that ask for the launch hand-off's window proof, pinned.

``WindowProof`` (cowork/api/v1/permissions.py) refuses a caller that holds
the bearer token and nothing else, on a server the desktop app launched with
the hand-off. A route that approves an agent's action, reveals or writes a
credential, or changes a gate on what the agent may do declares it, and so
does every Code mode write route except those whose callers hold a credential
of their own and never the proof (CODE_MODE_WRITES_WITHOUT_THE_PROOF), so
agent code on the same machine can't do those things for itself.

The pin matters in both directions, so the assertions run both ways. A route
that loses the declaration lets an agent holding the bearer token act as the
person again, and nothing else would notice. A route that gains it stops
working for every caller except Cowork's window and the desktop app's main
process, so adding one needs a written reason too. An entry with no route is
a stale pin that has stopped protecting anything.

tests/test_window_proof_matrix.py checks the other half: that the routes
declaring the class are the routes that refuse, and no other route does.

Neither file knows which routes are privileged. The matrix proves that the
routes that refuse are the routes that declare WindowProof, not that every
privileged route declares it: a privileged route that never declares the
proof passes the pin and the matrix alike. So check a new route against the
criterion above when you review it. Only one test proves coverage of a whole
surface: ``test_every_code_mode_write_route_asks_for_the_proof`` fails on any
write route under /api/v1/coding/ that neither declares the proof nor names
its caller in CODE_MODE_WRITES_WITHOUT_THE_PROOF.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute

from cowork.api.v1.permissions import WindowProof
from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.server import create_app

CODE_MODE_PREFIX = "/api/v1/coding/"
READ_METHODS = frozenset({"GET", "HEAD"})

# Reasons several routes share, one per harm.

#: A Full access task runs its turns with no sandbox and no approvals.
PROMPT_A_CODE_TASK = (
    "send a prompt into a Code task, or run one queued there, so the agent runs work the person never "
    "asked for, outside any sandbox in a Full access task"
)
#: A Code task's terminal and its project commands run with no sandbox,
#: whatever the task's permission mode.
RUN_IN_A_CODE_TERMINAL = (
    "start or type into a Code task's terminal, whose commands run outside any sandbox whatever the "
    "task's permission mode"
)
RUN_A_CODE_PROJECT_COMMAND = (
    "run the project's commands in a Code task's workspace, outside any sandbox whatever the task's "
    "permission mode"
)
MANAGE_A_CODE_TERMINAL = "open, rename, resize, stop or close a terminal in the person's Code task"
#: Skills and playbooks are instructions a Code task loads with its prompt.
CHANGE_CODE_INSTRUCTIONS = (
    "add, change or remove the skills and playbooks Code tasks load as instructions, so the agent "
    "writes its own instructions"
)
#: The only confirmation these routes take is ``confirmed: true`` in the body.
ACT_THROUGH_A_CODE_CONNECTION = (
    "act in the person's name through a connection (push and open pull requests, mark one ready, "
    "merge it, resolve a review thread, post an update, or close or complete an issue) on a "
    "confirmation the caller writes into the request body itself"
)
READ_THROUGH_A_CODE_CONNECTION = (
    "read or search issues and pull requests through the person's GitHub or Linear connection, "
    "outside the grants a task holds"
)
ORGANIZE_A_CODE_TASK = (
    "rename, pin, archive or reopen a Code task in the person's list; archiving also removes the "
    "task's workspace from disk"
)
END_A_REPAIR_REVIEW = "end the person's review of the agent's repair with the agent's edit kept"

#: (path, methods) -> what an agent holding the bearer token could do with
#: the route if it did not ask for the window proof.
WINDOW_PROOF_REASONS: dict[tuple[str, tuple[str, ...]], str] = {
    # Code mode: answer for the person, or widen a task's own sandbox.
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
    ("/api/v1/coding/runs/{run_id}/connector-capabilities", ("POST",)): (
        "mint a connector grant token for a run, so the agent grants itself a connector's actions"
    ),
    # Code mode: start or steer a task's work.
    ("/api/v1/coding/sessions", ("POST",)): (
        "start a Code task with a prompt, folders and permission mode the caller chose, Full access "
        "included, so the agent runs work outside its sandbox"
    ),
    ("/api/v1/coding/sessions/{session_id}/turns", ("POST",)): PROMPT_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/steer", ("POST",)): PROMPT_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/queue", ("POST",)): PROMPT_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/queue/{instruction_id}/steer", ("POST",)): PROMPT_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/queue/run", ("POST",)): PROMPT_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/fork", ("POST",)): (
        "copy a Code task, its permission mode and folders included, into a new task the caller can then prompt"
    ),
    ("/api/v1/coding/sessions/{session_id}/recover", ("POST",)): (
        "restore an interrupted Code task on a computer the caller picks, or recreate its workspace"
    ),
    # Code mode: run commands outside any sandbox.
    ("/api/v1/coding/sessions/{session_id}/terminals/{terminal_id}/start", ("POST",)): RUN_IN_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminals/{terminal_id}/input", ("POST",)): RUN_IN_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminal/start", ("POST",)): RUN_IN_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminal/input", ("POST",)): RUN_IN_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/project-actions/run", ("POST",)): RUN_A_CODE_PROJECT_COMMAND,
    ("/api/v1/coding/sessions/{session_id}/validate", ("POST",)): RUN_A_CODE_PROJECT_COMMAND,
    ("/api/v1/coding/sessions/{session_id}/windows-sandbox/setup", ("POST",)): (
        "start the elevated Windows sandbox setup for a Code task's engine on this computer"
    ),
    # Code mode: act in the person's name through a connection.
    ("/api/v1/coding/sessions/{session_id}/draft-pull-requests", ("POST",)): ACT_THROUGH_A_CODE_CONNECTION,
    ("/api/v1/coding/sessions/{session_id}/pull-request-action", ("POST",)): ACT_THROUGH_A_CODE_CONNECTION,
    ("/api/v1/coding/sessions/{session_id}/publish", ("POST",)): ACT_THROUGH_A_CODE_CONNECTION,
    ("/api/v1/coding/sessions/{session_id}/source-action", ("POST",)): ACT_THROUGH_A_CODE_CONNECTION,
    ("/api/v1/coding/sessions/{session_id}/delivery-policy", ("PUT",)): (
        "turn on a Code task's delivery automation, so the window marks its pull request ready, merges it "
        "or closes its issue in the person's name without asking"
    ),
    ("/api/v1/coding/sessions/{session_id}/delivery-automation/claim", ("POST",)): (
        "use up the automatic fix attempts the person allowed a Code task"
    ),
    # Code mode: read through the person's connections.
    ("/api/v1/coding/source-context", ("POST",)): READ_THROUGH_A_CODE_CONNECTION,
    ("/api/v1/coding/projects/{project_id}/source-context", ("POST",)): READ_THROUGH_A_CODE_CONNECTION,
    ("/api/v1/coding/work-items/search", ("POST",)): READ_THROUGH_A_CODE_CONNECTION,
    ("/api/v1/coding/projects/{project_id}/work-items/search", ("POST",)): READ_THROUGH_A_CODE_CONNECTION,
    # Code mode: change the files, instructions or settings a task runs with.
    ("/api/v1/coding/projects", ("POST",)): (
        "create a Code project over any folders on this computer, with the commands, environment variables "
        "and default permission mode the caller chose"
    ),
    ("/api/v1/coding/projects/{project_id}", ("PATCH",)): (
        "change a Code project's folders, commands, environment variables, connections or default "
        "permission mode, which its new tasks take"
    ),
    ("/api/v1/coding/projects/{project_id}", ("DELETE",)): "delete a Code project",
    ("/api/v1/coding/sessions/{session_id}/project-commands/refresh", ("POST",)): (
        "copy a Code project's current commands onto an existing task"
    ),
    ("/api/v1/coding/project-resources/inspect", ("POST",)): (
        "learn whether any path on this computer is a folder or a Git repository"
    ),
    ("/api/v1/coding/workspace/initialize-git", ("POST",)): "make any folder on this computer a Git repository",
    ("/api/v1/coding/git/identity", ("PUT",)): (
        "set the Git author name and email for commits on this computer where none is set"
    ),
    ("/api/v1/coding/sessions/{session_id}/review/file", ("POST",)): (
        "stage, unstage or discard a file's changes in a Code task's workspace"
    ),
    ("/api/v1/coding/sessions/{session_id}/branch", ("POST",)): "create a branch in a Code task's repository",
    ("/api/v1/coding/sessions/{session_id}/commit", ("POST",)): (
        "commit a Code task's changes under the person's Git identity"
    ),
    ("/api/v1/coding/sessions/{session_id}/apply", ("POST",)): (
        "copy a Code task's changes into the person's own source folder"
    ),
    ("/api/v1/coding/skills/sources", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/sources/{source_id}/refresh", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/sources/{source_id}/apply", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/sources/{source_id}", ("DELETE",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/sources/{source_id}/projects", ("PUT",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/projects/{project_id}/skills/{source_id}", ("PUT",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/projects/{project_id}/playbook", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/projects/{project_id}/playbook", ("DELETE",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/projects/{project_id}/playbook/refresh", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/projects/{project_id}/playbook/apply", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/projects/{project_id}/playbook/items", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/personal", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/personal/import", ("POST",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/personal/{skill_id}", ("PUT",)): CHANGE_CODE_INSTRUCTIONS,
    ("/api/v1/coding/skills/personal/{skill_id}", ("DELETE",)): CHANGE_CODE_INSTRUCTIONS,
    # Code mode: manage, stop or remove the person's tasks and terminals.
    ("/api/v1/coding/sessions/{session_id}/rename", ("POST",)): ORGANIZE_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/pin", ("POST",)): ORGANIZE_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/unpin", ("POST",)): ORGANIZE_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/archive", ("POST",)): ORGANIZE_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}/unarchive", ("POST",)): ORGANIZE_A_CODE_TASK,
    ("/api/v1/coding/sessions/{session_id}", ("DELETE",)): "delete a Code task, with its workspace and history",
    ("/api/v1/coding/sessions/{session_id}/cancel", ("POST",)): "stop a Code task's running turn",
    ("/api/v1/coding/sessions/{session_id}/queue/{instruction_id}", ("DELETE",)): (
        "remove an instruction the person queued for a Code task"
    ),
    ("/api/v1/coding/sessions/{session_id}/terminals", ("POST",)): MANAGE_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminals/{terminal_id}", ("PATCH",)): MANAGE_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminals/{terminal_id}", ("DELETE",)): MANAGE_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminals/{terminal_id}/resize", ("POST",)): MANAGE_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminals/{terminal_id}/stop", ("POST",)): MANAGE_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminal/resize", ("POST",)): MANAGE_A_CODE_TERMINAL,
    ("/api/v1/coding/sessions/{session_id}/terminal/stop", ("POST",)): MANAGE_A_CODE_TERMINAL,
    # Code mode: the computers that run tasks, and the local runtime.
    ("/api/v1/coding/runtime/registration-token", ("POST",)): (
        "issue a registration token that lets any computer register itself as a Code runtime"
    ),
    ("/api/v1/coding/computers/{computer_id}", ("PATCH",)): "rename a computer registered to run Code tasks",
    ("/api/v1/coding/computers/{computer_id}", ("DELETE",)): "revoke a computer registered to run Code tasks",
    ("/api/v1/coding/runtime/prepare-shutdown", ("POST",)): (
        "interrupt every running Code task, as the desktop app does before it stops the server"
    ),
    ("/api/v1/coding/workspace-cleanup/retry", ("POST",)): (
        "retry removing the task workspaces Cowork queued for deletion"
    ),
    # Settings and the agent's credentials.
    ("/api/v1/settings/reveal-key/{name}", ("GET",)): "read a provider's API key unmasked",
    ("/api/v1/settings/raw", ("GET",)): "read the server's dotenv verbatim, every provider secret in it",
    ("/api/v1/settings/raw", ("POST",)): (
        "write the server's dotenv and sync its keys into settings, so the agent sets any server setting"
    ),
    ("/api/v1/settings/", ("PUT",)): (
        "write any settings over the API, provider keys and endpoints among them, to point the model at "
        "a server the agent chose or lift a limit the person set. The proof closes this path only: a "
        "settings row written to the database file, or a line in the account's .env for a setting with "
        "no row, still does both, until the server stops reading the settings that steer a credential "
        "or an approval from plain storage"
    ),
    ("/api/v1/settings/{key}", ("PUT",)): "write one setting, such as a provider key or endpoint",
    ("/api/v1/settings/{key}", ("DELETE",)): "delete a setting, which puts a limit the person set back to its default",
    ("/api/v1/settings/logout", ("POST",)): "delete every stored credential, signing the person out",
    ("/api/v1/runtime-credential/minds", ("PUT",)): "replace or clear the MindsHub credential the agent's turns spend",
    ("/api/v1/runtime-credential/browser", ("PUT",)): (
        "point the agent's browser at a loopback address and key the caller chose, or clear it"
    ),
    # Connections and their credentials.
    ("/api/v1/connectors/oauth/{engine}/credentials", ("GET",)): (
        "read a built-in connector's bundled OAuth client values, which ship in every installer and "
        "are not confidential"
    ),
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
    ("/api/v1/connectors/submissions/", ("POST",)): (
        "save or overwrite a connector's credentials, planting the agent's account in the person's name"
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
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/agent-repairs/release", ("POST",)): END_A_REPAIR_REVIEW,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/agent-repairs/{repair_id}/cancel", ("POST",)): (
        END_A_REPAIR_REVIEW
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

#: (path, methods) -> the caller of a Code mode write route that doesn't ask
#: for the window proof. Each caller holds a credential of its own and never
#: the proof, so the route can't ask for it. Every other write route under
#: CODE_MODE_PREFIX declares WindowProof.
CODE_MODE_WRITES_WITHOUT_THE_PROOF: dict[tuple[str, tuple[str, ...]], str] = {
    ("/api/v1/coding/inference/{path:path}", ("GET", "POST")): (
        "the Codex app-server of a local Code task (codex_config.local_inference_base_url), with its "
        "session's inference token, which coding._require_inference_client checks"
    ),
    ("/api/v1/coding/runtime/register", ("POST",)): (
        "a remote Code runtime registering itself (RemoteRuntimeClient.register in "
        "cowork/coding/runtime_client.py), with a registration token in the body"
    ),
    ("/api/v1/coding/runtime/computers/{computer_id}/heartbeat", ("POST",)): (
        "a remote Code runtime (RemoteRuntimeClient.heartbeat), with its runtime token"
    ),
    ("/api/v1/coding/runtime/computers/{computer_id}/lease", ("POST",)): (
        "a remote Code runtime (RemoteRuntimeClient.lease), with its runtime token"
    ),
    ("/api/v1/coding/runtime/runs/{run_id}/events", ("POST",)): (
        "a remote Code runtime (RemoteRuntimeClient.event), with its runtime token"
    ),
    ("/api/v1/coding/runtime/runs/{run_id}/commands/claim", ("POST",)): (
        "a remote Code runtime (RemoteRuntimeClient.commands), with its runtime token"
    ),
    ("/api/v1/coding/runtime/runs/{run_id}/commands/ack", ("POST",)): (
        "a remote Code runtime (RemoteRuntimeClient.acknowledge), with its runtime token"
    ),
    ("/api/v1/coding/runtime/computers/{computer_id}/connector-capabilities/{grant_id}/invoke", ("POST",)): (
        "the developer-tool MCP server of a task on a remote Code runtime "
        "(cowork/coding/remote_integration_mcp.py), with the run's agent token and a connector grant token"
    ),
    ("/api/v1/coding/runtime/computers/{computer_id}/runs/{run_id}/inference/{path:path}", ("GET", "POST")): (
        "the engine of a task on a remote Code runtime (RemoteRuntimeClient.inference_endpoint), with "
        "the run's agent token"
    ),
}


@pytest.fixture(scope="module")
def app() -> FastAPI:
    return create_app()


def _key(*, route: APIRoute | APIWebSocketRoute) -> tuple[str, tuple[str, ...]]:
    path, methods = route_key(route)
    return path, tuple(m for m in methods if m != "HEAD")


@pytest.fixture(scope="module")
def proof_routes(app) -> set[tuple[str, tuple[str, ...]]]:
    return {
        _key(route=route)
        for route in app.routes
        if isinstance(route, (APIRoute, APIWebSocketRoute)) and WindowProof in declared_permissions(route)
    }


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


def test_every_code_mode_write_route_asks_for_the_proof(app):
    """Code mode runs a Full access task's work and every terminal with no
    sandbox, and delivers in the person's name, so every route under
    CODE_MODE_PREFIX that takes anything but GET declares WindowProof, an
    ``api_route`` with GET among its methods included. The exceptions name a
    caller that holds its own credential and never the proof. A new write
    route lands in WINDOW_PROOF_REASONS, or here with its caller."""
    writes = {
        _key(route=route): WindowProof in declared_permissions(route)
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path.startswith(CODE_MODE_PREFIX)
        and route.methods - READ_METHODS
    }
    unasked = sorted(
        key for key, asks in writes.items() if not asks and key not in CODE_MODE_WRITES_WITHOUT_THE_PROOF
    )
    assert unasked == [], (
        "these Code mode write routes neither declare WindowProof nor name their "
        "caller in CODE_MODE_WRITES_WITHOUT_THE_PROOF. Declare the proof and pin a "
        f"reason, or name the caller that can't send it: {unasked}"
    )
    stale = sorted(key for key in CODE_MODE_WRITES_WITHOUT_THE_PROOF if writes.get(key, True))
    assert stale == [], (
        "these exemptions name a route that no longer exists, takes only GET, or "
        f"now declares WindowProof. Delete the entry: {stale}"
    )


def test_the_proof_set_is_small_enough_to_read(proof_routes):
    """A number someone notices moving, as the open-by-design pin keeps one.
    The bound is loose on purpose: it is a tripwire for a bulk change, not a
    budget to spend down. Code mode's write routes are most of the set, and
    the Code mode walker above puts each new one in it."""
    assert len(proof_routes) <= 128, (
        f"{len(proof_routes)} routes now ask for the window proof. If that is "
        "right, raise the bound in the same commit that explains why."
    )
