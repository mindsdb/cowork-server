from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cowork.coding.workspace import WorkspaceManager


def copy_source_changes(
    manager: WorkspaceManager, source: Path, target: Path, revision: str
) -> None:
    """Copy through a private index and object store; never stage the user's files."""
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
        objects = root / "objects"
        objects.mkdir()
        source_objects = (source / manager.git.run(source, "rev-parse", "--git-path", "objects").stdout.strip()).resolve()
        # Git reads existing objects through alternates but writes new blobs
        # only to this temporary store, even if we later reject an oversized patch.
        alternates = json.dumps(str(source_objects), ensure_ascii=False)
        inherited = os.environ.get("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        if inherited:
            alternates += os.pathsep + inherited
        environment = {
            "GIT_INDEX_FILE": str(root / "index"),
            "GIT_OBJECT_DIRECTORY": str(objects),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": alternates,
            "GIT_OPTIONAL_LOCKS": "0",
        }
        manager.git.run(source, "read-tree", revision, environment=environment)
        manager.git.run(source, "add", "--all", "--", ".", environment=environment)
        # An untracked nested repository becomes a new gitlink in this private
        # index. A patch would copy only the link, silently dropping its files.
        staged = manager.git.run(
            source, "ls-files", "--stage", "-z", environment=environment
        ).stdout
        if any(
            line.startswith("160000 ") and line.split("\t", 1)[1] not in gitlinks
            for line in staged.split("\0") if "\t" in line
        ):
            raise WorkspaceError(
                "Local changes contain embedded repositories. Add them as project resources or start from committed code"
            )
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
