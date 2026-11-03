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

The same sites matter on a self-hosted deployment, where every person's agent
code runs beside this server. There COWORK_CODE_SANDBOX=landlock starts each
of anton's spawns through its Landlock launcher, under the policy
cowork.code_sandbox.build_cell_sandbox builds, and a construction that drops
that policy runs the agent's code unlocked. So every construction that leads
to an anton spawn (ChatSessionConfig, anton's ScratchpadManager and its
default runtime factory, LocalScratchpadRuntime, the backend pool and its
registry, and the credential probe) must pass ``cell_sandbox`` too:
test_every_agent_code_construction_passes_the_policy reads each one, through
the file's import aliases. At runtime, build_chat_session and the pool's
_make_runtime and get_or_create also refuse a missing policy under the
sandbox (cowork.code_sandbox.refuse_a_missing_policy). The launcher's startup
self-test is a spawn of its own, listed below; Code mode's git,
project-command and Codex sites are refused under the sandbox
(refuse_code_mode_in_sandbox), and app backends too (_ensure_backend_running).
"""

import ast
from collections import Counter
from pathlib import Path

import pytest

COWORK_ROOT = Path(__file__).resolve().parent.parent / "cowork"

BANNED_MODULES = {
    "subprocess",
    "runpy",
    "pty",
    "multiprocessing",
    # Deserialisers: loading one on agent-authored shared-storage bytes is code
    # execution (`pickle.load(session_path)` is the highest-value example, since
    # that path is hardcoded here).
    "pickle",
    "dill",
    "marshal",
    "shelve",
    "joblib",
    # URI-resolving renderers: they fetch <img>/<link>/@import out of
    # attacker-authored HTML (SSRF / local-file read) while rendering.
    "xhtml2pdf",
    "htmldocx",
    # A fork of htmldocx that fetches images the same way.
    "html4docx",
    "weasyprint",
    "pdfkit",
    "imgkit",
    # Runs the pandoc binary in a child process, and pandoc fetches the
    # images and stylesheets the source names.
    "pypandoc",
}
BANNED_ATTRS = {
    "system",
    "popen",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "execl",
    "execle",
    "execlp",
    "spawnv",
    "spawnve",
    "spawnl",
    "spawnlp",
    "posix_spawn",
    "posix_spawnp",
    "fork",
    "forkpty",
    "execlpe",
    "spawnle",
    "spawnlpe",
    "spawnvp",
    "spawnvpe",
    "startfile",
}
#: An event loop's own spawns, on whatever receiver: `loop.subprocess_exec`,
#: `asyncio.get_running_loop().subprocess_shell(...)`.
LOOP_SPAWN_ATTRS = {"subprocess_exec", "subprocess_shell"}
BANNED_NAMES = {"eval", "exec", "compile", "__import__"}
#: Dangerous callables named by their canonical dotted path, so the module's
#: SAFE helpers are not swept up: `importlib.import_module` is banned but
#: `importlib.metadata.version` is not; `yaml.load` is banned but
#: `yaml.safe_load` is not.
BANNED_CALLS = {
    "importlib.import_module",
    "yaml.load",
    "yaml.unsafe_load",
    # Spawns anton's Landlock launcher in a child process.
    "anton.core.backends.sandbox.self_test",
    "anyio.open_process",
    "anyio.run_process",
    # Forks this process into workers.
    "concurrent.futures.ProcessPoolExecutor",
}
BANNED_CONSTRUCTORS = {"ChatSession", "LocalScratchpadRuntime"}
BANNED_IMPORT_PATHS = {"anton.core.artifacts.backend_launcher"}

#: (relative path, enclosing function/method or "<module>", labeled call,
#: reason). Desktop-only, each guarded by _org_mode(), EXCEPT the
#: build_chat_session entry, which is itself the guard. One entry per
#: occurrence: two identical sites need two entries (see the module docstring).
ALLOWLIST = (
    (
        "coding/project_actions.py",
        "<module>",
        "subprocess",
        "import used only for Windows command-line serialization; task execution still routes through the guarded terminal service",
    ),
    (
        "coding/project_actions.py",
        "terminal_command_line",
        "subprocess.list2cmdline",
        "serialize the trusted working-directory argument for cmd.exe without starting a process",
    ),
    (
        "coding/project_actions.py",
        "terminal_command_line",
        "subprocess.list2cmdline",
        "serialize trusted project-action arguments for cmd.exe without starting a process",
    ),
    (
        "coding/project_workspaces.py",
        "<module>",
        "subprocess",
        "import backing desktop-only Code Project setup and validation commands",
    ),
    (
        "coding/project_workspaces.py",
        "ProjectCommandRunner._run_one",
        "subprocess.run",
        "shell-free local Code Project commands, refused before execution in org mode",
    ),
    (
        "coding/workspace.py",
        "<module>",
        "subprocess",
        "import backing the local-only GitRunner; every run is refused by _org_mode()",
    ),
    (
        "coding/workspace.py",
        "GitRunner.run",
        "subprocess.run(git)",
        "shell-free local Git operations, refused before spawn by _org_mode()",
    ),
    (
        "server.py",
        "_warm_agent_runtime",
        "importlib.import_module",
        "imports the fixed AGENT_RUNTIME_MODULES tuple of anton modules at startup; no module name comes from input or files",
    ),
    (
        "services/artifacts.py",
        "<module>",
        "subprocess",
        "import backing reveal_in_file_manager's subprocess.run calls below",
    ),
    (
        "services/artifacts.py",
        "reveal_in_file_manager",
        "subprocess.run(open)",
        "reveal_in_file_manager, macOS",
    ),
    (
        "services/artifacts.py",
        "reveal_in_file_manager",
        "subprocess.run(explorer)",
        "reveal_in_file_manager, Windows",
    ),
    (
        "services/artifacts.py",
        "reveal_in_file_manager",
        "subprocess.run(xdg-open)",
        "reveal_in_file_manager, Linux",
    ),
    (
        "services/artifacts.py",
        "_launch_backend_locked",
        "anton.core.artifacts.backend_launcher",
        "_launch_backend_locked, desktop artifact preview",
    ),
    (
        "api/v1/endpoints/artifacts.py",
        "<module>",
        "subprocess",
        "import backing open_artifact's subprocess.run call below",
    ),
    (
        "api/v1/endpoints/artifacts.py",
        "open_artifact",
        "subprocess.run(open)",
        "open_artifact endpoint",
    ),
    (
        "common/chat_session.py",
        "build_chat_session",
        "ChatSession",
        "the one sanctioned ChatSession construction, refuses in org mode",
    ),
    (
        "services/scratchpad_runtime.py",
        "_make_runtime",
        "LocalScratchpadRuntime",
        "reachable only through _launch_backend_locked, guarded by its caller's "
        "_org_mode() check; not itself guarded, so it must stay a named, "
        "single-entry site rather than an unlisted local import",
    ),
    (
        "code_sandbox.py",
        "_run_self_test",
        "anton.core.backends.sandbox.self_test",
        "the code sandbox's startup self-test: runs anton's launcher once, in a "
        "child process, against a folder it creates, and only with "
        "COWORK_CODE_SANDBOX=landlock",
    ),
    (
        "channels/registry.py",
        "load_first_party_plugins",
        "importlib.import_module",
        "imports packaged cowork.channels.plugins modules discovered via "
        "pkgutil.iter_modules, never an attacker-supplied name",
    ),
)


#: Constructions that lead to an anton spawn of agent code, and so must carry
#: the code sandbox's policy: the session config of a turn and of the probe,
#: anton's pad manager and its default runtime factory, the backend pool's
#: runtime, the pool and its registry, and the probe itself. Matched by the
#: callable's own name through the file's import aliases; a dotted entry
#: matches only through its module, since `get_or_create` alone names many
#: methods.
POLICY_CARRIERS = {
    "ChatSessionConfig",
    "ScratchpadManager",
    "local_scratchpad_runtime_factory",
    "LocalScratchpadRuntime",
    "WorkspaceScopedPool",
    "CredentialProbe",
    "_make_runtime",
    "scratchpad_runtime.get_or_create",
}

#: Where each carrier is constructed today, so the scan below is known to see
#: them: (relative path, enclosing function, carrier).
POLICY_CARRIER_SITES = (
    ("harnesses/anton_harness/harness.py", "AntonHarness._build_turn_session", "ChatSessionConfig"),
    ("services/connectors/probe.py", "CredentialProbe.run", "ChatSessionConfig"),
    ("handlers/probe.py", "ProbeHandler.run", "CredentialProbe"),
    ("services/scratchpad_runtime.py", "_make_runtime", "LocalScratchpadRuntime"),
    ("services/scratchpad_runtime.py", "get_or_create", "_make_runtime"),
    (
        "services/scratchpad_runtime.py",
        "WorkspaceScopedPool._pad",
        "scratchpad_runtime.get_or_create",
    ),
    ("services/artifacts.py", "_launch_backend_locked", "WorkspaceScopedPool"),
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
    `_sp -> subprocess` in the first, and `import concurrent.futures as cf`
    puts `cf -> concurrent.futures`; `from subprocess import run as _r` puts
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
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    modules[root] = root
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                callables[local] = f"{mod}.{alias.name}" if mod else alias.name
    return modules, callables


def _owner(node: ast.expr, *, modules: dict[str, str], callables: dict[str, str]) -> str:
    """The dotted name an attribute's owner refers to, through the file's
    import aliases (`cf` -> `concurrent.futures`, `futures.x` ->
    `concurrent.futures.x`), or "" when it is neither a name nor a chain of
    attributes on one."""
    if isinstance(node, ast.Name):
        return modules.get(node.id) or callables.get(node.id) or node.id
    if isinstance(node, ast.Attribute):
        owner = _owner(node.value, modules=modules, callables=callables)
        return f"{owner}.{node.attr}" if owner else ""
    return ""


def _banned_label(
    fn: ast.expr, modules: dict[str, str], callables: dict[str, str]
) -> str | None:
    """The canonical name of a banned callable expression, or None.

    Both an attribute (`subprocess.run`) and a bare name (`run`, from
    `from subprocess import run`) resolve through the file's import aliases, so
    an aliased call reports under its real name and the ALLOWLIST never has to
    know a local rename.
    """
    if isinstance(fn, ast.Attribute):
        owner = _owner(fn.value, modules=modules, callables=callables)
        root = owner.split(".")[0]
        if fn.attr in LOOP_SPAWN_ATTRS:
            return f"loop.{fn.attr}"
        if fn.attr in BANNED_ATTRS and root == "os":
            return f"os.{fn.attr}"
        if fn.attr.startswith("create_subprocess_"):
            return f"asyncio.{fn.attr}"
        if root in BANNED_MODULES:
            return f"{root}.{fn.attr}"
        if f"{owner}.{fn.attr}" in BANNED_CALLS:
            return f"{owner}.{fn.attr}"
        if fn.attr in BANNED_CONSTRUCTORS:
            return fn.attr
        return None
    if isinstance(fn, ast.Name):
        if fn.id in BANNED_NAMES:
            return fn.id
        canonical = callables.get(fn.id)
        if canonical is None:
            return None
        if canonical in BANNED_CALLS:
            return canonical
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


def _collect_var_aliases(
    tree: ast.AST, modules: dict[str, str], callables: dict[str, str]
) -> dict[str, str]:
    """Local names bound to a banned callable by assignment, e.g.
    `fn = subprocess.run` -> `{fn: "subprocess.run"}`.

    Closes the `fn = subprocess.run; fn(...)` evasion, where the dangerous
    attribute lives in an `Assign` the call-walker never sees. Over-approximates
    across scopes (a name bound in one function is treated as known in another),
    which can only ever report MORE sites, never fewer.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(
            node.value, (ast.Attribute, ast.Name)
        ):
            label = _banned_label(node.value, modules, callables)
            if label is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = label
    return aliases


class _SiteVisitor(ast.NodeVisitor):
    """Walks one file's AST, tracking the enclosing function/class stack so
    each site can be labeled by where it lives, not just what it calls."""

    def __init__(
        self,
        rel: str,
        modules: dict[str, str],
        callables: dict[str, str],
        var_aliases: dict[str, str],
    ) -> None:
        self.rel = rel
        self._modules = modules
        self._callables = callables
        self._var_aliases = var_aliases
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

    def _record(
        self, node: ast.Call | ast.stmt, what: str, *, hint: str | None = None
    ) -> None:
        label = f"{what}({hint})" if hint else what
        self.found.append((self.rel, self._enclosing(), label))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if (
                alias.name.split(".")[0] in BANNED_MODULES
                or alias.name in BANNED_IMPORT_PATHS
            ):
                self._record(node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod.split(".")[0] in BANNED_MODULES or mod in BANNED_IMPORT_PATHS:
            self._record(node, mod)
        self.generic_visit(node)

    def _label_for(self, fn: ast.expr) -> str | None:
        """The canonical name of a banned callable, or None. A bare name bound
        to a banned callable by assignment (`fn = subprocess.run`) resolves via
        the pre-collected var-aliases; everything else via `_banned_label`."""
        if isinstance(fn, ast.Name) and fn.id in self._var_aliases:
            return self._var_aliases[fn.id]
        return _banned_label(fn, self._modules, self._callables)

    def _getattr_target(self, node: ast.Call) -> str | None:
        """`getattr(<banned>, "<attr>")` resolved to the callable it names.

        Closes the `getattr(subprocess, "run")(...)` / `getattr(os, "system")`
        evasion, where the attribute is a string the attribute-walker never
        sees. Only a plain-string second argument is resolvable; a computed one
        is left to the runtime guard.
        """
        fn = node.func
        if not (isinstance(fn, ast.Name) and fn.id == "getattr") or len(node.args) < 2:
            return None
        owner_node, attr_node = node.args[0], node.args[1]
        if not (
            isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str)
        ):
            return None
        attr = attr_node.value
        owner = _owner(owner_node, modules=self._modules, callables=self._callables)
        root = owner.split(".")[0]
        if attr in LOOP_SPAWN_ATTRS:
            return f"loop.{attr}"
        if root == "os" and attr in BANNED_ATTRS:
            return f"os.{attr}"
        if root in BANNED_MODULES:
            return f"{root}.{attr}"
        if f"{owner}.{attr}" in BANNED_CALLS:
            return f"{owner}.{attr}"
        return None

    def visit_Call(self, node: ast.Call) -> None:
        what = self._label_for(node.func) or self._getattr_target(node)
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
        var_aliases = _collect_var_aliases(tree, modules, callables)
        visitor = _SiteVisitor(rel, modules, callables, var_aliases)
        visitor.visit(tree)
        found.update(visitor.found)
    return found


def _render(counts: Counter[tuple[str, str, str]]) -> str:
    return "\n".join(
        f"  {rel}: {enclosing}: {label}" + (f" (x{n})" if n > 1 else "")
        for (rel, enclosing, label), n in sorted(counts.items())
    )


def _carrier(
    fn: ast.expr, *, module: str, modules: dict[str, str], callables: dict[str, str]
) -> str | None:
    """The POLICY_CARRIERS entry a call's callable names, or None.

    A bare name resolves through the file's imports, or else to the file's
    own ``module``: `Cfg` imported `as` an alias of ChatSessionConfig is
    ChatSessionConfig, and `get_or_create` inside scratchpad_runtime.py is
    `scratchpad_runtime.get_or_create`. An attribute's owner resolves the
    same way (_owner)."""
    if isinstance(fn, ast.Attribute):
        owner = _owner(fn.value, modules=modules, callables=callables)
        dotted = f"{owner}.{fn.attr}" if owner else fn.attr
    elif isinstance(fn, ast.Name):
        dotted = callables.get(fn.id) or f"{module}.{fn.id}"
    else:
        return None
    parts = dotted.split(".")
    for candidate in (".".join(parts[-2:]), parts[-1]):
        if candidate in POLICY_CARRIERS:
            return candidate
    return None


class _CarrierVisitor(ast.NodeVisitor):
    """Each construction of a POLICY_CARRIERS entry in one file, labeled by
    where it lives, and whether it passes ``cell_sandbox``: as a keyword, or
    inside a ``**`` mapping that names it. ``module`` is the file's dotted
    module name, and ``modules`` and ``callables`` its import aliases
    (_collect_aliases)."""

    def __init__(
        self, rel: str, *, module: str, modules: dict[str, str], callables: dict[str, str]
    ) -> None:
        self.rel = rel
        self._module = module
        self._modules = modules
        self._callables = callables
        self._stack: list[str] = []
        self.found: list[tuple[str, str, str, bool]] = []

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

    def visit_Call(self, node: ast.Call) -> None:
        name = _carrier(
            node.func, module=self._module, modules=self._modules, callables=self._callables
        )
        if name is not None:
            passes = any(
                keyword.arg == "cell_sandbox"
                or (keyword.arg is None and "cell_sandbox" in ast.unparse(keyword.value))
                for keyword in node.keywords
            )
            enclosing = ".".join(self._stack) if self._stack else "<module>"
            self.found.append((self.rel, enclosing, name, passes))
        self.generic_visit(node)


def _carrier_visitor(rel: str, *, tree: ast.AST) -> _CarrierVisitor:
    """The carrier scan of one file at ``rel`` below cowork/, run."""
    modules, callables = _collect_aliases(tree)
    module = ".".join(("cowork", *Path(rel).with_suffix("").parts))
    visitor = _CarrierVisitor(rel, module=module, modules=modules, callables=callables)
    visitor.visit(tree)
    return visitor


def _carrier_sites() -> list[tuple[str, str, str, bool]]:
    found: list[tuple[str, str, str, bool]] = []
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        rel = str(path.relative_to(COWORK_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found.extend(_carrier_visitor(rel, tree=tree).found)
    return found


def test_every_agent_code_construction_passes_the_policy():
    """A construction that leads to an anton spawn and passes no
    ``cell_sandbox`` runs the agent's code unlocked under the sandbox."""
    found = _carrier_sites()
    assert {site[:3] for site in found} >= set(POLICY_CARRIER_SITES), (
        "the scan no longer sees a known construction; update POLICY_CARRIER_SITES"
    )
    dropping = [site[:3] for site in found if not site[3]]
    assert dropping == [], (
        "construction(s) that pass no cell_sandbox, so the agent's code they "
        "lead to would run outside the code sandbox:\n"
        + "\n".join(f"  {rel}: {enclosing}: {name}" for rel, enclosing, name in dropping)
    )


def test_a_construction_without_the_policy_is_caught():
    visitor = _carrier_visitor("snippet.py", tree=ast.parse(
        "def build():\n"
        "    ChatSessionConfig(llm_client=c)\n"
        "    ChatSessionConfig(llm_client=c, **({'cell_sandbox': p} if p else {}))\n"
        "    WorkspaceScopedPool(root, cell_sandbox=p)\n"
    ))
    assert [passes for *_, passes in visitor.found] == [False, True, True]


@pytest.mark.parametrize(
    ("source", "carrier"),
    [
        (
            "from anton.core.session import ChatSessionConfig as Cfg\n"
            "Cfg(llm_client=None)\n",
            "ChatSessionConfig",
        ),
        (
            "import anton.core.session as session_module\n"
            "session_module.ChatSessionConfig(llm_client=None)\n",
            "ChatSessionConfig",
        ),
        (
            "from anton.core.backends.manager import ScratchpadManager\n"
            "ScratchpadManager(runtime_factory=f).get_or_create('x')\n",
            "ScratchpadManager",
        ),
        (
            "from anton.core.backends.local import local_scratchpad_runtime_factory as factory\n"
            "factory(name='x')\n",
            "local_scratchpad_runtime_factory",
        ),
        (
            "from cowork.services import scratchpad_runtime\n"
            "scratchpad_runtime.get_or_create('x', workspace_path=p)\n",
            "scratchpad_runtime.get_or_create",
        ),
        (
            "from cowork.services.scratchpad_runtime import get_or_create as pad\n"
            "pad('x', workspace_path=p)\n",
            "scratchpad_runtime.get_or_create",
        ),
    ],
)
def test_a_construction_under_another_name_is_caught(source, carrier):
    """The carrier scan resolves import aliases and reaches anton's own pad
    manager, its runtime factory, and the backend pool's registry."""
    visitor = _carrier_visitor("snippet.py", tree=ast.parse(source))
    assert [(name, passes) for _rel, _enclosing, name, passes in visitor.found] == [
        (carrier, False)
    ]


def test_another_objects_get_or_create_is_no_carrier():
    visitor = _carrier_visitor("snippet.py", tree=ast.parse(
        "pool.get_or_create('app')\nself._scratchpads.get_or_create('main')\n"
    ))
    assert visitor.found == []


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


# --- unit coverage for the extended detection (I4) --------------------------
#
# These parse tiny snippets rather than the real tree, so a regression in the
# getattr / variable-alias / deserialiser / dynamic-import detection fails here
# with a clear message instead of silently shrinking what the tree scan catches.


def _snippet_labels(src: str) -> list[str]:
    tree = ast.parse(src)
    modules, callables = _collect_aliases(tree)
    var_aliases = _collect_var_aliases(tree, modules, callables)
    visitor = _SiteVisitor("snippet.py", modules, callables, var_aliases)
    visitor.visit(tree)
    return [label for _rel, _enc, label in visitor.found]


def test_getattr_evasion_is_caught():
    labels = _snippet_labels("import subprocess\ngetattr(subprocess, 'run')(['x'])\n")
    assert any(label.startswith("subprocess.run") for label in labels), labels


def test_getattr_os_system_is_caught_even_uncalled():
    assert "os.system" in _snippet_labels("import os\nfn = getattr(os, 'system')\n")


def test_variable_alias_call_is_caught():
    labels = _snippet_labels("import subprocess\nfn = subprocess.run\nfn(['x'])\n")
    assert any(label.startswith("subprocess.run") for label in labels), labels


def test_deserialiser_load_is_caught():
    assert any(
        label.startswith("pickle.load")
        for label in _snippet_labels("import pickle\npickle.load(open('x', 'rb'))\n")
    )


def test_dynamic_import_is_caught_but_metadata_is_not():
    assert any(
        label.startswith("importlib.import_module")
        for label in _snippet_labels("import importlib\nimportlib.import_module('x')\n")
    )
    assert (
        _snippet_labels("from importlib.metadata import version\nversion('x')\n") == []
    )


def test_yaml_load_is_caught_but_safe_load_is_not():
    assert any(
        label.startswith("yaml.load")
        for label in _snippet_labels("import yaml\nyaml.load('x')\n")
    )
    assert _snippet_labels("import yaml\nyaml.safe_load('x')\n") == []


def test_dunder_import_is_caught():
    assert any(
        label.startswith("__import__")
        for label in _snippet_labels("__import__('os')\n")
    )


@pytest.mark.parametrize(
    "source",
    [
        "import asyncio\nasyncio.get_running_loop().subprocess_exec(Protocol, 'py')\n",
        "loop.subprocess_shell(Protocol, 'py -c 1')\n",
        "fn = getattr(loop, 'subprocess_exec')\n",
    ],
)
def test_an_event_loops_spawn_is_caught_on_any_receiver(source):
    assert any(label.startswith("loop.subprocess_") for label in _snippet_labels(source))


@pytest.mark.parametrize("name", ["execlpe", "spawnle", "spawnlpe", "spawnvp", "spawnvpe", "startfile"])
def test_every_os_spawn_is_caught(name):
    assert f"os.{name}" in _snippet_labels(f"import os\nos.{name}(py)\n")
    assert f"os.{name}" in _snippet_labels(f"from os import {name}\n{name}(py)\n")


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("import anyio\nanyio.run_process(command)\n", "anyio.run_process"),
        ("from anyio import open_process\nopen_process(command)\n", "anyio.open_process"),
        (
            "import concurrent.futures\nconcurrent.futures.ProcessPoolExecutor()\n",
            "concurrent.futures.ProcessPoolExecutor",
        ),
        (
            "import concurrent.futures as cf\ncf.ProcessPoolExecutor()\n",
            "concurrent.futures.ProcessPoolExecutor",
        ),
        (
            "from concurrent import futures\nfutures.ProcessPoolExecutor()\n",
            "concurrent.futures.ProcessPoolExecutor",
        ),
        (
            "from concurrent.futures import ProcessPoolExecutor\nProcessPoolExecutor()\n",
            "concurrent.futures.ProcessPoolExecutor",
        ),
    ],
)
def test_anyio_spawns_and_a_process_pool_are_caught(source, label):
    assert label in _snippet_labels(source)


def test_a_thread_pool_is_not_a_spawn():
    assert _snippet_labels(
        "from concurrent.futures import ThreadPoolExecutor\nThreadPoolExecutor()\n"
    ) == []


@pytest.mark.parametrize("module", ["xhtml2pdf", "htmldocx", "html4docx", "pypandoc"])
def test_a_document_converter_that_fetches_is_caught(module):
    """Export builds Word files from the artifact's own bytes, so no converter
    that fetches what the source names may come back under cowork/."""
    assert _snippet_labels(f"import {module}\n") == [module]
    assert _snippet_labels(f"from {module} import convert\nconvert(source)\n") == [
        module,
        f"{module}.convert",
    ]
