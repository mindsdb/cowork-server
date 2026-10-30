from __future__ import annotations

from pathlib import Path

import pytest
from anton.core.mcp.client import McpSession
from mcp.server.mcpserver import MCPServer

from cowork.coding.project_models import CodeProject, ProjectConnection, ProjectFolder
from cowork.coding.provider_mcp import ProviderTools
from cowork.coding.session_factory import project_instructions
from cowork.coding.workspace import WorkspaceError


def linear_server() -> MCPServer:
    server = MCPServer("linear")

    @server.tool()
    def list_issues(project: str) -> str:
        return f"ENG-1 in {project}"

    @server.tool()
    def save_issue(title: str) -> str:
        return "created"

    @server.tool()
    def a_tool_added_later() -> str:
        return "unclassified"

    return server


def project(*connections: ProjectConnection) -> CodeProject:
    return CodeProject(
        id="product",
        name="Product",
        folders=[ProjectFolder(id="app", name="App", path="/tmp/app")],
        connections=list(connections),
    )


def providers(current: CodeProject, token_for=lambda _connection: "token") -> ProviderTools:
    tokens: list[str] = []

    def session(_url: str, token: str) -> McpSession:
        tokens.append(token)
        return McpSession(server=linear_server())

    tools = ProviderTools(current, token_for, session_factory=session)
    tools.tokens = tokens  # type: ignore[attr-defined]
    return tools


def test_only_classified_read_tools_are_exposed_and_forwarded() -> None:
    tools = providers(project(ProjectConnection(provider="linear", name="work", label="Linear work")))
    try:
        assert [item["name"] for item in tools.tools()] == ["linear__list_issues"]
        assert tools.tools()[0]["inputSchema"]["required"] == ["project"]
        result = tools.call("linear__list_issues", {"project": "Scheduled Tasks"})
        assert result == {"content": [{"type": "text", "text": "ENG-1 in Scheduled Tasks"}], "isError": False}
        # Write and unclassified tools are never routed, so a guessed name reaches nothing.
        assert not tools.owns("linear__save_issue")
        assert not tools.owns("linear__a_tool_added_later")
        assert tools.status == {"Linear work": "1 read-only tools"}
    finally:
        tools.close()


def test_two_connections_for_one_provider_are_named_by_connection() -> None:
    tools = providers(project(
        ProjectConnection(provider="linear", name="work"),
        ProjectConnection(provider="linear", name="oss"),
    ))
    try:
        assert {item["name"] for item in tools.tools()} == {"linear__work__list_issues", "linear__oss__list_issues"}
    finally:
        tools.close()


def test_a_failed_connection_is_reported_and_does_not_hide_the_others() -> None:
    def token_for(connection: ProjectConnection) -> str | None:
        if connection.name == "broken":
            raise WorkspaceError("Reconnect Linear before using this project source")
        if connection.provider == "github":
            return None
        return "token"

    tools = providers(
        project(
            ProjectConnection(provider="linear", name="broken"),
            ProjectConnection(provider="linear", name="work"),
            ProjectConnection(provider="github", name="enterprise"),
            ProjectConnection(provider="slack", name="team"),
        ),
        token_for,
    )
    try:
        assert [item["name"] for item in tools.tools()] == ["linear__work__list_issues"]
        assert tools.status == {
            "broken": "Unavailable",
            "work": "1 read-only tools",
            "enterprise": "No hosted tools for this connection",
        }
        # Slack has no hosted server, so no token is ever requested for it.
        assert tools.tokens == ["token"]  # type: ignore[attr-defined]
    finally:
        tools.close()


@pytest.mark.parametrize("local", [True, False])
def test_prompt_names_the_forwarded_tools_only_on_the_local_plane(local: bool, tmp_path: Path) -> None:
    from cowork.coding.contracts import TaskWorkspace, WorkspaceKind

    current = project(
        ProjectConnection(provider="linear", name="work", label="MindsDB"),
        ProjectConnection(provider="slack", name="team", label="Team"),
    )
    workspaces = [TaskWorkspace(
        folder_id="app",
        folder_name="App",
        source_path=str(tmp_path),
        workspace_path=str(tmp_path),
        workspace_kind=WorkspaceKind.git_worktree,
    )] if local else []
    text = project_instructions(current, workspaces, [], "")

    assert "- slack: Team — read one item by URL with mindshub_read_developer_context" in text
    if local:
        assert "- linear: MindsDB — linear's own read-only tools on the mindshub_code server, named linear__*" in text
    else:
        assert "- linear: MindsDB — read one item by URL with mindshub_read_developer_context" in text
