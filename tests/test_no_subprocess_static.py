"""No new code-execution sites in cowork-server.

Artifact and attachment files live on shared EFS and are written by any org's
agent. Every way of running one is a cross-tenant code execution. The desktop
build legitimately needs a few, so they are listed here by exact location; the
test asserts the set of sites EQUALS the allowlist, so a new one fails and a
removed one also fails (keeping the list honest).

Each allowlisted site must be guarded by `_cloud_mode()`; see
tests/test_no_execution_in_org_mode.py for the runtime half of this.
"""

import ast
from pathlib import Path

COWORK_ROOT = Path(__file__).resolve().parent.parent / "cowork"

BANNED_MODULES = {"subprocess", "runpy", "pty", "multiprocessing"}
BANNED_ATTRS = {
    "system", "popen", "execv", "execve", "execvp", "execvpe", "execl",
    "execle", "execlp", "spawnv", "spawnve", "spawnl", "spawnlp",
    "posix_spawn", "posix_spawnp", "fork", "forkpty",
}
BANNED_NAMES = {"eval", "exec", "compile"}
BANNED_IMPORT_PATHS = {"anton.core.artifacts.backend_launcher"}

#: (relative path, line, short reason). Desktop-only, each guarded by _cloud_mode().
#
# The bare "subprocess" entries are the module-level `import subprocess` in
# each file, not a second execution capability: every use of that import in
# both files is one of the `subprocess.run` calls listed below, and each of
# those is behind the same `_cloud_mode()` check. The AST walker records the
# `import` statement and each `Call` as separate sites, so the same guarded
# capability shows up under two different "what" strings.
ALLOWLIST = {
    ("services/artifacts.py", "subprocess", "import backing reveal_in_file_manager's subprocess.run calls"),
    ("services/artifacts.py", "subprocess.run", "reveal_in_file_manager, macOS"),
    ("services/artifacts.py", "subprocess.run", "reveal_in_file_manager, Windows"),
    ("services/artifacts.py", "subprocess.run", "reveal_in_file_manager, Linux"),
    ("services/artifacts.py", "anton.core.artifacts.backend_launcher",
     "_launch_backend_locked, desktop artifact preview"),
    ("api/v1/endpoints/artifacts.py", "subprocess", "import backing open_artifact's subprocess.run call"),
    ("api/v1/endpoints/artifacts.py", "subprocess.run", "open_artifact endpoint"),
}


def _sites() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        rel = str(path.relative_to(COWORK_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in BANNED_MODULES:
                        found.add((rel, alias.name, f"line {node.lineno}"))
                    if alias.name in BANNED_IMPORT_PATHS:
                        found.add((rel, alias.name, f"line {node.lineno}"))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.split(".")[0] in BANNED_MODULES or mod in BANNED_IMPORT_PATHS:
                    found.add((rel, mod, f"line {node.lineno}"))
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    owner = fn.value.id if isinstance(fn.value, ast.Name) else ""
                    if fn.attr in BANNED_ATTRS and owner == "os":
                        found.add((rel, f"os.{fn.attr}", f"line {node.lineno}"))
                    if fn.attr.startswith("create_subprocess_"):
                        found.add((rel, f"asyncio.{fn.attr}", f"line {node.lineno}"))
                    if owner == "subprocess":
                        found.add((rel, f"subprocess.{fn.attr}", f"line {node.lineno}"))
                elif isinstance(fn, ast.Name) and fn.id in BANNED_NAMES:
                    found.add((rel, fn.id, f"line {node.lineno}"))
    return found


def test_execution_sites_match_the_allowlist():
    found = {(rel, what) for rel, what, _ in _sites()}
    allowed = {(rel, what) for rel, what, _ in ALLOWLIST}

    new = found - allowed
    assert not new, (
        "New code-execution site(s) in cowork-server. Artifact files come from "
        "shared EFS and are untrusted. If this is desktop-only, guard it with "
        "_cloud_mode() and add it to ALLOWLIST with a reason:\n"
        + "\n".join(f"  {rel}: {what}" for rel, what in sorted(new))
    )

    gone = allowed - found
    assert not gone, (
        "ALLOWLIST names execution site(s) that no longer exist. Remove them so "
        "the list stays an accurate inventory:\n"
        + "\n".join(f"  {rel}: {what}" for rel, what in sorted(gone))
    )
