from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cowork.coding.workspace import WorkspaceManager


def copy_source_changes(
    manager: WorkspaceManager, source: Path, target: Path, revision: str
) -> None:
    """Copy the working tree through a private index; never stage the user's files."""
    from cowork.coding.workspace import MAX_TOTAL_DIFF_BYTES, WorkspaceError

    if manager.git.run(source, "ls-files", "--unmerged").stdout:
        raise WorkspaceError("Resolve merge conflicts before including local changes")
    # Git patches cannot carry uncommitted files inside a submodule.
    gitlinks = {
        line.split("\t", 1)[1]
        for line in manager.git.run(source, "ls-files", "--stage", "-z").stdout.split(
            "\0"
        )
        if line.startswith("160000 ") and "\t" in line
    }
    if gitlinks.intersection(manager.source_change_paths(source)):
        raise WorkspaceError(
            "Commit changes inside submodules before including local changes"
        )
    with tempfile.TemporaryDirectory(prefix="cowork-source-changes-") as temporary:
        root = Path(temporary)
        environment = {"GIT_INDEX_FILE": str(root / "index"), "GIT_OPTIONAL_LOCKS": "0"}
        manager.git.run(source, "read-tree", revision, environment=environment)
        manager.git.run(source, "add", "--all", "--", ".", environment=environment)
        patch = root / "changes.patch"
        manager.git.run(
            source,
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            f"--output={patch}",
            revision,
            "--",
            environment=environment,
        )
        if manager.git.run(source, "rev-parse", "HEAD").stdout.strip() != revision:
            raise WorkspaceError(
                "The source branch changed. Refresh repositories and try again"
            )
        if patch.stat().st_size > MAX_TOTAL_DIFF_BYTES:
            raise WorkspaceError(
                "Local changes are too large to copy safely. Commit them first or start from committed code"
            )
        if patch.stat().st_size:
            with patch.open("rb") as handle:
                checked = manager.git.run(
                    target, "apply", "--check", "-", stdin=handle, check=False
                )
            if checked.returncode:
                raise WorkspaceError(
                    "Local changes conflict with the selected base branch. Choose the current branch or start from committed code"
                )
            with patch.open("rb") as handle:
                manager.git.run(
                    target, "apply", "--whitespace=nowarn", "-", stdin=handle
                )
