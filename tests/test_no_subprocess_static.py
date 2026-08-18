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

Two further evasions of that keying were confirmed by mutation and are closed
here:

  3. A set of keys still collapses two IDENTICAL keys onto one, so a second
     unguarded call in an already-allowlisted function with the same
     first-argument literal was accepted: inserting
     `subprocess.run(["open", "/mnt/cowork-shared/evil"])` above the
     `_org_mode()` guard in `reveal_in_file_manager` passed the test, because
     the guarded macOS branch below it already claims `subprocess.run(open)`.
     Sites are therefore COUNTED, not collected: ALLOWLIST is a sequence, one
     entry per occurrence, and the test compares multiplicities. A duplicate
     site needs its own entry, with its own reason, to pass.

  4. Matching the owner of a call by its literal spelling missed every
     aliased import: `import subprocess as _sp` followed by `_sp.run(...)`
     was invisible, as would be `from subprocess import run as _r; _r(...)`.
     Imports are resolved to their canonical module/callable in a pre-pass
     over the whole file, so an aliased call is reported under its real name
     and cannot hide behind a local rename.

`ChatSession` is banned as a constructor for the same reason a subprocess
spawn is banned: anton registers the `scratchpad` tool on it unconditionally
and, absent a runtime_factory, runs LLM-written Python in a subprocess of this
process with `cwd` in the workspace. Two independent constructors of it were
found in review, one of them unguarded, so the single allowlisted construction
is `cowork/common/chat_session.py::build_chat_session`, which refuses in org
mode; every other caller must route through it.

`LocalScratchpadRuntime` is banned as a constructor for the same reason:
it is the subprocess executor the `ChatSession` ban exists to keep out. Its
one construction, in `cowork/services/scratchpad_runtime.py::_make_runtime`,
is unguarded itself, but every path to it runs through
`artifacts.py::_launch_backend_locked`, which is only reached after the
caller's `_org_mode()` check. That indirection makes it a hole in the
mechanism rather than a live one: a new, closer caller added later would
spawn it unguarded with this test still green, which is exactly what listing
it here as a named site prevents.
"""

import ast
from collections import Counter
from pathlib import Path

COWORK_ROOT = Path(__file__).resolve().parent.parent / "cowork"

BANNED_MODULES = {"subprocess", "runpy", "pty", "multiprocessing"}
BANNED_ATTRS = {
    "system", "popen", "execv", "execve", "execvp", "execvpe", "execl",
    "execle", "execlp", "spawnv", "spawnve", "spawnl", "spawnlp",
    "posix_spawn", "posix_spawnp", "fork", "forkpty",
}
BANNED_NAMES = {"eval", "exec", "compile"}
BANNED_CONSTRUCTORS = {"ChatSession", "LocalScratchpadRuntime"}
BANNED_IMPORT_PATHS = {"anton.core.artifacts.backend_launcher"}

#: (relative path, enclosing function/method or "<module>", labeled call,
#: reason). Desktop-only, each guarded by _org_mode(), EXCEPT the
#: build_chat_session entry, which is itself the guard. One entry per
#: occurrence: two identical sites need two entries (see the module docstring).
ALLOWLIST = (
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
    ("common/chat_session.py", "build_chat_session", "ChatSession",
     "the one sanctioned ChatSession construction, refuses in org mode"),
    ("services/scratchpad_runtime.py", "_make_runtime", "LocalScratchpadRuntime",
     "reachable only through _launch_backend_locked, guarded by its caller's "
     "_org_mode() check; not itself guarded, so it must stay a named, "
     "single-entry site rather than an unlisted local import"),
)


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


def _collect_aliases(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """Every local name in a file that refers to an imported module or
    callable, mapped to the canonical dotted name it was imported from.

    Returns `(modules, callables)`: `import subprocess as _sp` puts
    `_sp -> subprocess` in the first; `from subprocess import run as _r` puts
    `_r -> subprocess.run` in the second. Without this, a banned call hides
    behind any local rename.

    Collected in a pre-pass over the whole file, not while walking calls,
    because a module-level function body is visited before an import that
    appears further down the file, and because a function-local import is in
    scope for the call that follows it. Merging every scope's imports into one
    flat map over-approximates (an alias imported in one function is treated
    as known in another), which can only ever report MORE sites, never fewer.
    """
    modules: dict[str, str] = {}
    callables: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                modules[alias.asname or root] = root
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                callables[local] = f"{mod}.{alias.name}" if mod else alias.name
    return modules, callables


class _SiteVisitor(ast.NodeVisitor):
    """Walks one file's AST, tracking the enclosing function/class stack so
    each site can be labeled by where it lives, not just what it calls."""

    def __init__(self, rel: str, modules: dict[str, str], callables: dict[str, str]) -> None:
        self.rel = rel
        self._modules = modules
        self._callables = callables
        self._stack: list[str] = []
        self.found: list[tuple[str, str, str]] = []

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

    def _record(self, node: ast.Call | ast.stmt, what: str, *, hint: str | None = None) -> None:
        label = f"{what}({hint})" if hint else what
        self.found.append((self.rel, self._enclosing(), label))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in BANNED_MODULES or alias.name in BANNED_IMPORT_PATHS:
                self._record(node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod.split(".")[0] in BANNED_MODULES or mod in BANNED_IMPORT_PATHS:
            self._record(node, mod)
        self.generic_visit(node)

    def _label_for(self, fn: ast.expr) -> str | None:
        """The canonical name of a banned callable, or None.

        Both spellings resolve through the file's import aliases, so
        `_sp.run(...)` after `import subprocess as _sp` reports as
        `subprocess.run`, and the ALLOWLIST never has to know a local name.
        """
        if isinstance(fn, ast.Attribute):
            raw_owner = fn.value.id if isinstance(fn.value, ast.Name) else ""
            owner = self._modules.get(raw_owner, raw_owner)
            if fn.attr in BANNED_ATTRS and owner == "os":
                return f"os.{fn.attr}"
            if fn.attr.startswith("create_subprocess_"):
                return f"asyncio.{fn.attr}"
            if owner in BANNED_MODULES:
                return f"{owner}.{fn.attr}"
            if fn.attr in BANNED_CONSTRUCTORS:
                return fn.attr
            return None
        if isinstance(fn, ast.Name):
            if fn.id in BANNED_NAMES:
                return fn.id
            canonical = self._callables.get(fn.id)
            if canonical is None:
                return None
            owner, _, attr = canonical.rpartition(".")
            if attr in BANNED_CONSTRUCTORS:
                return attr
            if attr.startswith("create_subprocess_"):
                return f"asyncio.{attr}"
            root = owner.split(".")[0]
            if root in BANNED_MODULES:
                return f"{root}.{attr}"
            if root == "os" and attr in BANNED_ATTRS:
                return f"os.{attr}"
            if attr in BANNED_NAMES:
                return attr
        return None

    def visit_Call(self, node: ast.Call) -> None:
        what = self._label_for(node.func)
        if what is not None:
            self._record(node, what, hint=_arg_hint(node))
        self.generic_visit(node)


def _sites() -> Counter[tuple[str, str, str]]:
    """Every execution site under cowork/, counted. A count, not a set: two
    identical sites are two sites, and collapsing them is how a second
    unguarded call slipped in beside an allowlisted one."""
    found: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        rel = str(path.relative_to(COWORK_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules, callables = _collect_aliases(tree)
        visitor = _SiteVisitor(rel, modules, callables)
        visitor.visit(tree)
        found.update(visitor.found)
    return found


def _render(counts: Counter[tuple[str, str, str]]) -> str:
    return "\n".join(
        f"  {rel}: {enclosing}: {label}" + (f" (x{n})" if n > 1 else "")
        for (rel, enclosing, label), n in sorted(counts.items())
    )


def test_execution_sites_match_the_allowlist():
    found = _sites()
    allowed: Counter[tuple[str, str, str]] = Counter(
        (rel, enclosing, label) for rel, enclosing, label, _ in ALLOWLIST
    )

    new = found - allowed
    assert not new, (
        "New code-execution site(s) in cowork-server. Artifact files come from "
        "shared EFS and are untrusted. If this is desktop-only, guard it with "
        "_org_mode() and add it to ALLOWLIST with a reason (one entry per "
        "occurrence):\n" + _render(new)
    )

    gone = allowed - found
    assert not gone, (
        "ALLOWLIST names execution site(s) that no longer exist. Remove them so "
        "the list stays an accurate inventory:\n" + _render(gone)
    )
