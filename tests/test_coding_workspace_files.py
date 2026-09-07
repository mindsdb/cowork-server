from __future__ import annotations

from pathlib import Path

import pytest

from cowork.coding.context import validate_references
from cowork.coding.contracts import (
    CodingSession,
    InputReference,
    PermissionMode,
    TaskWorkspace,
    WorkspaceKind,
)
from cowork.coding.workspace_files import WorkspaceFileBrowser


def _workspace(resource_id: str, name: str, path: Path) -> TaskWorkspace:
    return TaskWorkspace(
        folder_id=resource_id,
        folder_name=name,
        source_path=str(path),
        workspace_path=str(path),
        workspace_kind=WorkspaceKind.direct_folder,
    )


def _session(*workspaces: TaskWorkspace) -> CodingSession:
    primary = workspaces[0]
    return CodingSession(
        id="task-1",
        title="Browse files",
        engine_id="codex",
        engine_adapter_version="1",
        model="gpt",
        permission_mode=PermissionMode.supervised,
        source_path=primary.source_path,
        workspace_path=primary.workspace_path,
        workspace_kind=primary.workspace_kind,
        workspaces=list(workspaces),
    )


def test_browser_labels_a_standalone_workspace_with_its_source_folder(tmp_path: Path) -> None:
    source = tmp_path / "customer-app"
    workspace = tmp_path / "workspaces" / "af8922aa-605c-4064-b639-4e95fbf2524e"
    source.mkdir()
    workspace.mkdir(parents=True)
    (workspace / "package.json").write_text("{}\n", encoding="utf-8")
    session = CodingSession(
        id="task-1",
        title="Browse files",
        engine_id="codex",
        engine_adapter_version="1",
        model="gpt",
        permission_mode=PermissionMode.supervised,
        source_path=str(source),
        workspace_path=str(workspace),
        workspace_kind=WorkspaceKind.local_copy,
        workspaces=[],
    )

    browser = WorkspaceFileBrowser(session)

    assert [(item.id, item.name) for item in browser.resources().items] == [
        ("folder", "customer-app"),
    ]
    [entry] = browser.entries("folder").items
    assert entry.resource_name == "customer-app"
    assert browser.file("folder", "package.json").resource_name == "customer-app"


def test_browser_lists_multiple_resources_and_reads_bounded_lines(tmp_path: Path) -> None:
    api = tmp_path / "api"
    web = tmp_path / "web"
    (api / "src").mkdir(parents=True)
    web.mkdir()
    (api / "src" / "main.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (web / "index.ts").write_text("export const app = true;\n", encoding="utf-8")

    browser = WorkspaceFileBrowser(_session(
        _workspace("api", "API", api),
        _workspace("web", "Web", web),
    ))

    assert [(item.id, item.name) for item in browser.resources().items] == [
        ("api", "API"),
        ("web", "Web"),
    ]
    root = browser.entries("api")
    assert [(item.path, item.kind) for item in root.items] == [("src", "directory")]
    content = browser.file("api", "src/main.py", 2, 3)
    assert content.content == "two\nthree\n"
    assert content.line_start == 2
    assert content.line_end == 3
    assert content.truncated is True
    assert len(content.content_hash) == 64


def test_browser_searches_paths_and_content_without_build_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / "src" / "router.ts").write_text("export const route = 'dashboard';\n", encoding="utf-8")
    (root / "src" / "view.ts").write_text("export const dashboard = true;\n", encoding="utf-8")
    (root / "node_modules" / "dashboard.js").write_text("ignored", encoding="utf-8")
    browser = WorkspaceFileBrowser(_session(_workspace("root", "Workspace", root)))

    matches = browser.search("dashboard").items

    assert {(item.path, item.match_kind) for item in matches} == {
        ("src/router.ts", "content"),
        ("src/view.ts", "content"),
    }


def test_browser_returns_a_valid_empty_range_for_an_empty_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "empty.txt").write_text("", encoding="utf-8")

    content = WorkspaceFileBrowser(_session(_workspace("root", "Workspace", root))).file(
        "root",
        "empty.txt",
    )

    assert content.content == ""
    assert content.line_count == 0
    assert (content.line_start, content.line_end) == (0, 0)
    assert content.truncated is False


def test_browser_rejects_traversal_symlinks_binary_and_large_files(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "escape").symlink_to(outside)
    (root / "binary.dat").write_bytes(b"before\0after")
    (root / "large.txt").write_bytes(b"x" * (1024 * 1024 + 1))
    browser = WorkspaceFileBrowser(_session(_workspace("root", "Workspace", root)))

    with pytest.raises(ValueError, match="stay inside"):
        browser.file("root", "../outside.txt")
    with pytest.raises(ValueError, match="unavailable"):
        browser.file("root", "escape")
    with pytest.raises(ValueError, match="Binary"):
        browser.file("root", "binary.dat")
    with pytest.raises(ValueError, match="too large"):
        browser.file("root", "large.txt")
    assert [item.name for item in browser.entries("root").items] == ["binary.dat", "large.txt"]


def test_precise_reference_is_resolved_and_hash_checked(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "main.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    session = _session(_workspace("root", "Workspace", root))
    content = WorkspaceFileBrowser(session).file("root", "main.py")
    reference = InputReference(
        name="main.py:1",
        path=str(target),
        resource_id="root",
        relative_path="main.py",
        line_start=1,
        line_end=1,
        content_hash=content.content_hash,
    )

    [resolved] = validate_references(session, [reference])
    assert resolved.path == str(target)
    assert resolved.line_start == 1
    assert resolved.line_end == 1

    target.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_references(session, [reference])
