from __future__ import annotations

import json
from pathlib import Path

from cowork.coding.contracts import CodingSession, TaskWorkspace, WorkspaceKind
from cowork.coding.integration_mcp import CodeIntegrationMcp
from cowork.coding.project_models import CodeProject, ProjectConnection, ProjectFolder
from cowork.coding.project_store import CodeProjectStore
from cowork.coding.store import CodingStore


def test_agent_neutral_mcp_exposes_project_context_as_read_only_tools(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    project = CodeProject(
        id="product",
        name="Product",
        folders=[ProjectFolder(id="app", name="App", path=str(source), base_branch="staging")],
        connections=[ProjectConnection(provider="github", name="work", label="Work GitHub")],
    )
    CodeProjectStore(tmp_path).create(project)
    CodingStore(tmp_path).save_session(
        CodingSession(
            id="task-1",
            title="Task",
            engine_id="codex",
            engine_adapter_version="1",
            model="gpt-5.6-sol",
            project_id=project.id,
            project_name=project.name,
            source_path=str(source),
            workspace_path=str(workspace),
            workspace_kind=WorkspaceKind.git_worktree,
            workspaces=[
                TaskWorkspace(
                    folder_id="app",
                    folder_name="App",
                    source_path=str(source),
                    workspace_path=str(workspace),
                    workspace_kind=WorkspaceKind.git_worktree,
                    base_branch="staging",
                    task_branch="cowork/product/task-1",
                )
            ],
            guidance_summary="Team playbook · 3 active items",
            source_contexts=[{"provider": "github", "kind": "issue", "url": "https://github.com/example/repo/issues/1"}],
        )
    )

    server = CodeIntegrationMcp(tmp_path, "task-1")
    tools = server.handle("tools/list", {})
    assert tools is not None
    names = {item["name"] for item in tools["tools"]}
    assert names == {"mindshub_project_context", "mindshub_read_developer_context"}
    assert all("write" not in name and "publish" not in name for name in names)

    result = server.handle("tools/call", {"name": "mindshub_project_context", "arguments": {}})
    assert result is not None
    payload = json.loads(result["content"][0]["text"])
    assert payload["project"] == "Product"
    assert payload["folders"][0]["taskBranch"] == "cowork/product/task-1"
    assert payload["connectedTools"] == [{"provider": "github", "label": "Work GitHub"}]
    server.close()
