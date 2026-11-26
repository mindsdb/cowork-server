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
proof passes the pin and the matrix alike. Two walkers cover the write
routes, those that take any method but GET and HEAD.
``test_every_code_mode_write_route_asks_for_the_proof`` fails on any write
route under /api/v1/coding/ that neither declares the proof nor names its
caller in CODE_MODE_WRITES_WITHOUT_THE_PROOF.
``test_every_write_route_asks_for_the_proof_or_says_why`` fails on any write
route in the app that neither declares the proof nor appears with its reason
in WRITES_WITHOUT_THE_PROOF, CODE_MODE_WRITES_WITHOUT_THE_PROOF or the
open-by-design pin, so a new one has to be decided against the criterion
above. GET routes stay outside both walkers: a new privileged read, such as
one that reveals a credential, still depends on review.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute

from cowork.api.v1.permissions import WindowProof
from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.server import create_app
from tests.test_open_by_design_pin import OPEN_BY_DESIGN_REASONS

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
#: Anton's twins of the Code mode prompt routes. A chat's turn runs with the
#: person's files and connections.
PROMPT_AN_ANTON_CHAT = (
    "start a turn in a chat, or add a message to the one running there, so the agent runs work the "
    "person never asked for, with the person's files and connections"
)
#: A scheduled task's prompt runs as an Anton turn at the time the task names.
RUN_AN_ANTON_TASK = (
    "create, change, turn back on or run a scheduled task, whose prompt then runs as an Anton turn "
    "without the confirmation card Anton's own scheduling tool shows"
)
#: Skills are instructions Anton loads with its prompt, as Code mode's are.
CHANGE_ANTON_INSTRUCTIONS = (
    "add, change or remove the skills Anton loads as instructions, or turn one on or off for the "
    "person, so the agent writes its own instructions"
)
#: The memory's rules slot holds the person's behavioral gates.
CHANGE_ANTON_MEMORY = (
    "write or remove the memory every Anton turn reads, the rules of what the agent may do included, "
    "so the agent rewrites its own gates"
)
CHANGE_A_PROJECTS_FILES = (
    "write or delete any file in a project folder, its instructions (.anton/anton.md) and, on the "
    "desktop, its memory rules included, which the agent reads every turn"
)
CHANGE_A_CUSTOM_CONNECTOR = (
    "change or remove a custom connector's spec and usage notes, which the agent reads to call it"
)
OPEN_A_FOLDER_TO_ANTON = (
    "attach any folder on this computer to a chat, or make one a project, so the agent's turns there "
    "read and write it"
)
#: The desktop publishes with the person's own publishing key. Saving a
#: published artifact's source, or restoring a revision of it, publishes it
#: again, and so does editing a shared doc or restoring a version of it.
PUBLISH_IN_THE_PERSONS_NAME = (
    "publish an artifact, or change what a published one shows or who may open it, with the person's "
    "publishing key, so the agent makes its own content public in the person's name"
)
ROUTE_OUTSIDE_MESSAGES = (
    "bind a messaging channel's outside group or thread to a project or chat, with the instructions and "
    "trigger rule the caller wrote, so outside messages prompt the agent there"
)

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
    ("/api/v1/channels/bindings", ("POST",)): ROUTE_OUTSIDE_MESSAGES,
    ("/api/v1/channels/bindings/{binding_id}", ("PATCH",)): ROUTE_OUTSIDE_MESSAGES,
    # Anton: start or steer a chat's work, as the Code mode prompt routes do.
    ("/api/v1/responses/", ("POST",)): PROMPT_AN_ANTON_CHAT,
    ("/api/v1/responses/addon", ("POST",)): PROMPT_AN_ANTON_CHAT,
    ("/api/v1/schedules/", ("POST",)): RUN_AN_ANTON_TASK,
    ("/api/v1/schedules/{schedule_id}", ("PUT",)): RUN_AN_ANTON_TASK,
    ("/api/v1/schedules/{schedule_id}", ("PATCH",)): RUN_AN_ANTON_TASK,
    ("/api/v1/schedules/{schedule_id}/resume", ("POST",)): RUN_AN_ANTON_TASK,
    ("/api/v1/schedules/{schedule_id}/run-now", ("POST",)): RUN_AN_ANTON_TASK,
    # Anton: change the instructions, memory and connector notes a turn reads.
    ("/api/v1/skills/", ("POST",)): CHANGE_ANTON_INSTRUCTIONS,
    ("/api/v1/skills/upload", ("POST",)): CHANGE_ANTON_INSTRUCTIONS,
    ("/api/v1/skills/{skill_id}", ("PUT",)): CHANGE_ANTON_INSTRUCTIONS,
    ("/api/v1/skills/{skill_id}", ("DELETE",)): CHANGE_ANTON_INSTRUCTIONS,
    ("/api/v1/skills/{skill_id}/me", ("PUT",)): CHANGE_ANTON_INSTRUCTIONS,
    ("/api/v1/memory/", ("PUT",)): CHANGE_ANTON_MEMORY,
    ("/api/v1/memory/", ("DELETE",)): CHANGE_ANTON_MEMORY,
    ("/api/v1/memory/all", ("DELETE",)): CHANGE_ANTON_MEMORY,
    ("/api/v1/projects/{project_name}/files/{path:path}", ("PUT",)): CHANGE_A_PROJECTS_FILES,
    ("/api/v1/projects/{project_name}/files/{path:path}", ("DELETE",)): CHANGE_A_PROJECTS_FILES,
    ("/api/v1/connectors/custom/{connector_id}", ("PATCH",)): CHANGE_A_CUSTOM_CONNECTOR,
    ("/api/v1/connectors/custom/{connector_id}", ("DELETE",)): CHANGE_A_CUSTOM_CONNECTOR,
    # Anton: open a folder on this computer to a chat's turns.
    ("/api/v1/conversations/{conversation_id}/folders", ("POST",)): OPEN_A_FOLDER_TO_ANTON,
    ("/api/v1/tasks/{conversation_id}/folders", ("POST",)): OPEN_A_FOLDER_TO_ANTON,
    ("/api/v1/projects/", ("POST",)): OPEN_A_FOLDER_TO_ANTON,
    # Anton: publish in the person's name.
    ("/api/v1/publish/", ("POST",)): PUBLISH_IN_THE_PERSONS_NAME,
    ("/api/v1/publish/update", ("POST",)): PUBLISH_IN_THE_PERSONS_NAME,
    ("/api/v1/publish/activate", ("POST",)): PUBLISH_IN_THE_PERSONS_NAME,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}", ("PUT",)): PUBLISH_IN_THE_PERSONS_NAME,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/access", ("PUT",)): PUBLISH_IN_THE_PERSONS_NAME,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/republish", ("POST",)): PUBLISH_IN_THE_PERSONS_NAME,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/revisions/{revision_id}/restore", ("POST",)): (
        PUBLISH_IN_THE_PERSONS_NAME
    ),
    ("/api/v1/docs/{project_ref}/{artifact_id}/ops", ("POST",)): PUBLISH_IN_THE_PERSONS_NAME,
    ("/api/v1/docs/{project_ref}/{artifact_id}/history/{rev}/restore", ("POST",)): PUBLISH_IN_THE_PERSONS_NAME,
    # Anton: run a program on this computer.
    ("/api/v1/artifacts/preview-mount", ("POST",)): (
        "start a full-stack artifact's backend, a program that runs on this computer outside any sandbox"
    ),
    ("/api/v1/artifacts/open", ("POST",)): (
        "open an artifact file with this computer's default app for it, which runs the file outside any "
        "sandbox when it is an app or a script"
    ),
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

# Why a write route outside Code mode doesn't ask for the proof. Each reason
# says why the criterion at the top of this file doesn't cover the route.

CHANGES_THE_PERSONS_OWN_WORK = (
    "creates, edits, organizes or deletes the person's own chats, tasks, projects, files, artifacts, "
    "comments, pins, comparisons or analytics; it approves no agent action, reveals or writes no "
    "credential, and widens no gate on what the agent may do"
)
STOPS_OR_NARROWS = (
    "only stops, pauses, withdraws or narrows something (a turn, an added message, a scheduled task, a "
    "publication, an attached folder, a channel binding, a picked file), so it can't make the agent do more"
)
CHANGES_NOTHING = (
    "reads, checks, tests, searches, shows or reloads what is already there, or gives a fixed answer; "
    "it stores and approves nothing, and a stored key goes only to the address saved with it"
)
ASKS_A_MODEL = (
    "asks a model, with the MindsHub credential the caller sends, for an answer only (a voice call, or "
    "the app actions a voice request maps to), with no agent turn; the app takes any action through its "
    "own routes"
)
CHANGES_A_PREFERENCE = (
    "changes a preference of the person or their organization (the active MindsHub workspace, the "
    "app's look, the harness channel chats run on, a published artifact's analytics) that approves "
    "nothing, holds no credential and widens no gate"
)
RECORDS_A_PROVIDER_GRANT = (
    "records the files the person picked in a provider's own picker; the provider's grant, not this "
    "record, decides what the connection reaches"
)
ORG_TENANCY_ONLY = (
    "acts only in org tenancy, which create_app refuses to start while it holds the launch hand-off, "
    "so the proof would never apply"
)
SIGNED_IN_ONLY = (
    "acts only for a signed-in person, whom a local server names only from identity headers it was "
    "set to trust, which the desktop app never sets"
)
CALLER_WITHOUT_THE_PROOF = (
    "its caller is a messaging platform, which signs the body, or a cloud pod, with its turn's key, "
    "and neither ever holds the proof"
)

#: (path, methods) -> why a write route outside Code mode doesn't declare
#: WindowProof. Every other write route outside CODE_MODE_PREFIX declares
#: it, or is open by design (tests/test_open_by_design_pin.py).
WRITES_WITHOUT_THE_PROOF: dict[tuple[str, tuple[str, ...]], str] = {
    # Artifacts.
    ("/api/v1/artifacts/checks/{check_id}", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/comment-images", ("POST",)): (
        CHANGES_THE_PERSONS_OWN_WORK
    ),
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/quick-edits", ("POST",)): (
        CHANGES_THE_PERSONS_OWN_WORK
    ),
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/quick-edits/{edit_id}/undo", ("POST",)): (
        CHANGES_THE_PERSONS_OWN_WORK
    ),
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/export", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/stats/tracking", ("PUT",)): CHANGES_A_PREFERENCE,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/stats", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/comments-access", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/artifacts/shared-preview", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/agent-repairs", ("POST",)): (
        CHANGES_THE_PERSONS_OWN_WORK
    ),
    ("/api/v1/artifacts/drafts/{project_ref}/{artifact_id}/link", ("POST",)): SIGNED_IN_ONLY,
    ("/api/v1/artifacts/{slug}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/artifacts/reveal", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/artifacts/", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/artifacts/live-app-share", ("POST",)): SIGNED_IN_ONLY,
    ("/api/v1/artifacts/live-app-share/{share_id}", ("DELETE",)): SIGNED_IN_ONLY,
    ("/api/v1/artifacts/live-app-duplicate", ("POST",)): SIGNED_IN_ONLY,
    (
        "/api/v1/artifact-comments/{user_dir}/{report_id}/{subpath:path}",
        ("DELETE", "GET", "OPTIONS", "PATCH", "POST"),
    ): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/publish/", ("DELETE",)): STOPS_OR_NARROWS,
    ("/api/v1/publish/stats/tracking", ("PUT",)): CHANGES_A_PREFERENCE,
    ("/api/v1/publish/stats", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/content-reports/", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/content-reports/artifact", ("POST",)): ORG_TENANCY_ONLY,
    # The browser: on the web only, where it is the browser router's.
    ("/api/v1/browse/provision", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/embed", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/open", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/logins", ("DELETE",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/browsers", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/tabs/switch", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/tabs/close", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/close", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/browse/decisions", ("POST",)): CALLER_WITHOUT_THE_PROOF,
    # Messaging channels.
    ("/api/v1/channels/agent", ("PUT",)): CHANGES_A_PREFERENCE,
    ("/api/v1/channels/{channel_type}/reload", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/channels/{channel_type}/test-connection", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/channels/bindings/{binding_id}", ("DELETE",)): STOPS_OR_NARROWS,
    ("/api/v1/channels/discord/interactions", ("POST",)): CALLER_WITHOUT_THE_PROOF,
    ("/api/v1/channels/slack/events", ("POST",)): CALLER_WITHOUT_THE_PROOF,
    ("/api/v1/channels/telegram/webhook", ("POST",)): CALLER_WITHOUT_THE_PROOF,
    ("/api/v1/channels/whatsapp/webhook", ("GET", "POST")): CALLER_WITHOUT_THE_PROOF,
    # Chats, their turns, files and attachments.
    ("/api/v1/conversations/", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/tasks/", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/conversations/{conversation_id}", ("PATCH",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/tasks/{conversation_id}", ("PATCH",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/conversations/{conversation_id}/move", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/tasks/{conversation_id}/move", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/conversations/{conversation_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/tasks/{conversation_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/conversations/{conversation_id}/turns/{message_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/tasks/{conversation_id}/turns/{message_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/conversations/{conversation_id}/folders/{folder_id}", ("DELETE",)): STOPS_OR_NARROWS,
    ("/api/v1/tasks/{conversation_id}/folders/{folder_id}", ("DELETE",)): STOPS_OR_NARROWS,
    ("/api/v1/responses/cancel", ("POST",)): STOPS_OR_NARROWS,
    ("/api/v1/responses/addon/withdraw", ("POST",)): STOPS_OR_NARROWS,
    ("/api/v1/comparisons/", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/comparisons/{comparison_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/comparisons/{comparison_id}/verdicts/{turn_index}", ("PUT",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/comparisons/{comparison_id}/sides/{label}/continue", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/files/", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/files/{file_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/attachments/{project_name}/{session_id}/upload", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/attachments/{attachment_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/attachments/{project_name}/{session_id}/{attachment_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/attachments/{project_name}/{session_id}/{attachment_id}/move-to-project", ("POST",)): (
        CHANGES_THE_PERSONS_OWN_WORK
    ),
    ("/api/v1/pins/", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/pins/{item_id}/visit", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/pins/{item_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    # Docs. A new doc gets a fresh folder and isn't shared, so on the desktop
    # its republish sends nothing. An image is named by its hash and never
    # replaces a file; it shows in the doc only once an edit, which asks for
    # the proof, puts it there.
    ("/api/v1/docs/", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/docs/{project_ref}/{artifact_id}/images", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    # Projects and their files. Upload writes only top-level names that don't
    # start with a dot, so it can't reach .anton/.
    ("/api/v1/projects/{project_id}", ("PATCH",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/projects/{project_id}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/projects/{project_name}/files/upload", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/projects/{project_name}/files/import/sharepoint", ("POST",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/projects/{project_name}/skill_drafts/{slug}", ("DELETE",)): CHANGES_THE_PERSONS_OWN_WORK,
    ("/api/v1/projects/preview-mount-file", ("POST",)): CHANGES_NOTHING,
    # Schedules: these two only stop work. The rest ask for the proof.
    ("/api/v1/schedules/{schedule_id}", ("DELETE",)): STOPS_OR_NARROWS,
    ("/api/v1/schedules/{schedule_id}/pause", ("POST",)): STOPS_OR_NARROWS,
    # Connectors.
    ("/api/v1/connectors/connections/{engine}/{name}/picked-files", ("PATCH",)): RECORDS_A_PROVIDER_GRANT,
    ("/api/v1/connectors/connections/{engine}/{name}/picked-files/{file_id}", ("DELETE",)): STOPS_OR_NARROWS,
    ("/api/v1/connectors/custom", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/connectors/custom-connections/{connection_id}", ("DELETE",)): ORG_TENANCY_ONLY,
    ("/api/v1/connectors/connections/custom_mcp/check-url", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/connectors/datasources/", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/connectors/datasources/{connection_id}", ("PATCH",)): ORG_TENANCY_ONLY,
    ("/api/v1/connectors/datasources/{connection_id}", ("DELETE",)): ORG_TENANCY_ONLY,
    ("/api/v1/connectors/datasources/{connection_id}/validation-retry", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/connectors/notion/search", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/connectors/notion/fetch", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/connectors/oauth/{engine}/picker/session", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/connectors/oauth/{engine}/picker/token", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/connectors/posthog/projects", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/connectors/specs/match", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/connectors/submissions/replaces", ("POST",)): CHANGES_NOTHING,
    # Settings checks: each sends a stored key only to the address saved with it.
    ("/api/v1/settings/validate", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/settings/test-providers", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/settings/test-decision-model", ("POST",)): CHANGES_NOTHING,
    ("/api/v1/settings/validate-provider", ("POST",)): CHANGES_NOTHING,
    # Skills: sharing needs a signed-in person. The rest ask for the proof.
    ("/api/v1/skills/{skill_id}/sharing", ("PUT",)): SIGNED_IN_ONLY,
    # The person's account and devices.
    ("/api/v1/hub/deletion-request/", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/hub/workspaces/active", ("PUT",)): CHANGES_A_PREFERENCE,
    ("/api/v1/org-branding/appearance", ("PUT",)): CHANGES_A_PREFERENCE,
    ("/api/v1/push/devices", ("POST",)): ORG_TENANCY_ONLY,
    ("/api/v1/push/devices", ("DELETE",)): ORG_TENANCY_ONLY,
    # Live apps: each needs a signed-in person with an email.
    ("/api/v1/live-apps/availability/{notice_id}/ack", ("POST",)): SIGNED_IN_ONLY,
    ("/api/v1/live-apps/previews", ("POST",)): SIGNED_IN_ONLY,
    ("/api/v1/live-apps/previews/{record_id}/keepalive", ("POST",)): SIGNED_IN_ONLY,
    ("/api/v1/live-apps/previews", ("DELETE",)): SIGNED_IN_ONLY,
    ("/api/v1/live-apps/previews/{record_id}", ("DELETE",)): SIGNED_IN_ONLY,
    # Voice.
    ("/api/v1/voice/session", ("POST",)): ASKS_A_MODEL,
    ("/api/v1/voice/operate", ("POST",)): ASKS_A_MODEL,
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


def _write_routes(app: FastAPI) -> dict[tuple[str, tuple[str, ...]], bool]:
    """Each route that takes a method other than GET and HEAD, an
    ``api_route`` with GET among its methods included, and whether it
    declares WindowProof."""
    return {
        _key(route=route): WindowProof in declared_permissions(route)
        for route in app.routes
        if isinstance(route, APIRoute) and route.methods - READ_METHODS
    }


def test_every_code_mode_write_route_asks_for_the_proof(app):
    """Code mode runs a Full access task's work and every terminal with no
    sandbox, and delivers in the person's name, so every route under
    CODE_MODE_PREFIX that takes anything but GET declares WindowProof. The
    exceptions name a caller that holds its own credential and never the
    proof. A new write route lands in WINDOW_PROOF_REASONS, or here with its
    caller."""
    writes = {key: asks for key, asks in _write_routes(app).items() if key[0].startswith(CODE_MODE_PREFIX)}
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


def test_every_write_route_asks_for_the_proof_or_says_why(app):
    """Every write route in the app declares WindowProof or appears with its
    reason in WRITES_WITHOUT_THE_PROOF, CODE_MODE_WRITES_WITHOUT_THE_PROOF or
    the open-by-design pin. A new write route lands in WINDOW_PROOF_REASONS
    when the criterion at the top of this file covers it, and otherwise in
    WRITES_WITHOUT_THE_PROOF with the reason it doesn't."""
    explained = set(WRITES_WITHOUT_THE_PROOF) | set(CODE_MODE_WRITES_WITHOUT_THE_PROOF) | set(OPEN_BY_DESIGN_REASONS)
    unasked = sorted(key for key, asks in _write_routes(app).items() if not asks and key not in explained)
    assert unasked == [], (
        "these write routes neither declare WindowProof nor say why. Declare the "
        "proof and pin a reason in WINDOW_PROOF_REASONS, or add the route to "
        f"WRITES_WITHOUT_THE_PROOF with the reason it needs none: {unasked}"
    )


def test_no_write_without_the_proof_outlives_its_route(app):
    """Each entry names a write route outside Code mode that exists, doesn't
    declare WindowProof and isn't open by design, so a route has one answer."""
    writes = _write_routes(app)
    stale = sorted(
        key
        for key in WRITES_WITHOUT_THE_PROOF
        if writes.get(key, True) or key[0].startswith(CODE_MODE_PREFIX) or key in OPEN_BY_DESIGN_REASONS
    )
    assert stale == [], (
        "these entries name a route that no longer exists, takes only GET, now "
        "declares WindowProof, sits under Code mode or is open by design. Delete "
        f"the entry: {stale}"
    )


def test_the_proof_set_is_small_enough_to_read(proof_routes):
    """A number someone notices moving, as the open-by-design pin keeps one.
    The bound is loose on purpose: it is a tripwire for a bulk change, not a
    budget to spend down. Code mode's write routes are most of the set;
    Anton's twins of them (its turns, scheduled tasks, skills, memory and
    publishing) and the other writes the criterion covers are the rest. Both
    walkers put each new write route in it unless the route says why not."""
    assert len(proof_routes) <= 160, (
        f"{len(proof_routes)} routes now ask for the window proof. If that is "
        "right, raise the bound in the same commit that explains why."
    )
