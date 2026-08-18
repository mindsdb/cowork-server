"""No new code-execution sites in cowork-server.

Artifact and attachment files live on shared EFS and are written by any org's
agent. Every way of running one is a cross-tenant code execution. The desktop
build legitimately needs a few, so they are listed here by exact site; the
test asserts the set of sites EQUALS the allowlist, so a new one fails and a
removed one also fails (keeping the list honest).

Each allowlisted site must be guarded by `_org_mode()`; see
tests/test_no_execution_in_org_mode.py for the runtime half of this.

A site is keyed by (file, enclosing function/method, labeled call), NOT by
(file, callable name). Keying by (file, callable) alone lets two different
defects through, both confirmed by mutation:

  1. A brand-new unguarded `subprocess.run` call anywhere in an already-
     allowlisted file collapses onto the same "callable" as an existing,
     guarded call and is silently accepted.
  2. Deleting one of several allowlisted calls in the same file leaves the
     "callable" entry still present (another call of the same kind remains),
     so the deletion is not noticed.

The enclosing function name distinguishes (1): a call in a new function is a
new site regardless of which banned callable it uses. It does not by itself
distinguish (2) when multiple allowlisted calls share one function (as
`reveal_in_file_manager`'s three OS branches do), so each labeled call also
carries an argument hint, the literal first argument when it is a plain
string or a list/tuple starting with one (e.g. the program name passed to
`subprocess.run`). That hint is what actually varies between the three
branches, so it is what makes each one a distinct, honestly-nameable site,
without resorting to a line number that churns on unrelated edits.
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

#: (relative path, enclosing function/method or "<module>", labeled call,
#: reason). Desktop-only, each guarded by _org_mode().
ALLOWLIST = {
    ("services/artifacts.py", "<module>", "subprocess",
     "import backing reveal_in_file_manager's subprocess.run calls below"),
    ("services/artifacts.py", "reveal_in_file_manager", "subprocess.run(open)",
     "reveal_in_file_manager, macOS"),
    ("services/artifacts.py", "reveal_in_file_manager", "subprocess.run(explorer)",
     "reveal_in_file_manager, Windows"),
    ("services/artifacts.py", "reveal_in_file_manager", "subprocess.run(xdg-open)",
     "reveal_in_file_manager, Linux"),
    ("services/artifacts.py", "_launch_backend_locked",
     "anton.core.artifacts.backend_launcher",
     "_launch_backend_locked, desktop artifact preview"),
    ("api/v1/endpoints/artifacts.py", "<module>", "subprocess",
     "import backing open_artifact's subprocess.run call below"),
    ("api/v1/endpoints/artifacts.py", "open_artifact", "subprocess.run(open)",
     "open_artifact endpoint"),
}


def _arg_hint(node: ast.Call) -> str | None:
    """The literal first argument of a call, when it's a plain string or a
    list/tuple starting with one (e.g. `subprocess.run(["open", ...])` ->
    "open"). None when the argument isn't a simple literal."""
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    if isinstance(first, (ast.List, ast.Tuple)) and first.elts:
        elt = first.elts[0]
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            return elt.value
    return None


class _SiteVisitor(ast.NodeVisitor):
    """Walks one file's AST, tracking the enclosing function/class stack so
    each site can be labeled by where it lives, not just what it calls."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self._stack: list[str] = []
        self.found: set[tuple[str, str, str]] = set()

    def _enclosing(self) -> str:
        return ".".join(self._stack) if self._stack else "<module>"

    def _push(self, name: str, node: ast.AST) -> None:
        self._stack.append(name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push(node.name, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push(node.name, node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push(node.name, node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in BANNED_MODULES or alias.name in BANNED_IMPORT_PATHS:
                self.found.add((self.rel, self._enclosing(), alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod.split(".")[0] in BANNED_MODULES or mod in BANNED_IMPORT_PATHS:
            self.found.add((self.rel, self._enclosing(), mod))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        what: str | None = None
        if isinstance(fn, ast.Attribute):
            owner = fn.value.id if isinstance(fn.value, ast.Name) else ""
            if fn.attr in BANNED_ATTRS and owner == "os":
                what = f"os.{fn.attr}"
            elif fn.attr.startswith("create_subprocess_"):
                what = f"asyncio.{fn.attr}"
            elif owner == "subprocess":
                what = f"subprocess.{fn.attr}"
        elif isinstance(fn, ast.Name) and fn.id in BANNED_NAMES:
            what = fn.id
        if what is not None:
            hint = _arg_hint(node)
            label = f"{what}({hint})" if hint else what
            self.found.add((self.rel, self._enclosing(), label))
        self.generic_visit(node)


def _sites() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        rel = str(path.relative_to(COWORK_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _SiteVisitor(rel)
        visitor.visit(tree)
        found |= visitor.found
    return found


def test_execution_sites_match_the_allowlist():
    found = _sites()
    allowed = {(rel, enclosing, label) for rel, enclosing, label, _ in ALLOWLIST}

    new = found - allowed
    assert not new, (
        "New code-execution site(s) in cowork-server. Artifact files come from "
        "shared EFS and are untrusted. If this is desktop-only, guard it with "
        "_org_mode() and add it to ALLOWLIST with a reason:\n"
        + "\n".join(f"  {rel}: {enclosing}: {label}" for rel, enclosing, label in sorted(new))
    )

    gone = allowed - found
    assert not gone, (
        "ALLOWLIST names execution site(s) that no longer exist. Remove them so "
        "the list stays an accurate inventory:\n"
        + "\n".join(f"  {rel}: {enclosing}: {label}" for rel, enclosing, label in sorted(gone))
    )
