"""The SQL a scratchpad cell runs, the HTTP requests it makes, the files it
writes and the names it binds from data it reads, read from its code without
running it.

The code is parsed with ``ast`` and nothing in it is evaluated. A string the
code spells is rendered from what it states:

- an f-string keeps each ``{expr}`` as written, except an environment lookup
  such as ``os.environ["DS_ERP_URL"]``, which reads ``{DS_ERP_URL}``, and a
  bare name bound to a saved connection's lookup or to a query, which reads as
  that string;
- ``"..." % value`` keeps each field (``%s``, ``%(key)s``) as ``{expr}`` by the
  same rule;
- ``+``, ``+=`` and ``sep.join([...])`` join their parts, where a bound name
  reads as ``{name}`` unless it holds a saved connection's lookup, a query or
  a base URL, so a token joined into a URL never reaches its text;
- ``os.path.join(...)``, ``Path(...)`` and ``/`` join a path, where a bound
  name reads as its string, except in a join that starts with a URL, where it
  reads as it does in a ``+``;
- ``text(...)``, ``dedent(...)`` and ``"...".format(...)`` give back the
  string they wrap;
- any other part renders as ``{expr}``.

A name bound to a string earlier in the cell, or in an earlier cell of the
turn (a scratchpad keeps one namespace across cells), stands for that string
where the code reads it on its own, and inside a template or join as the list
above says. Code that may not run (an ``if`` or ``else`` body, a loop body, an
``except`` handler or a ``case``) may leave a name holding what it held
before, so after that code a read of the name stands for each string it may
hold. A function's parameters and locals, a lambda's parameters and a
comprehension's targets are names of their own: they hide the cell's names of
the same spelling and never leak into the cell. Comments never reach the
parsed tree, and a bare string statement (a docstring) is skipped. Code inside
an f-string's ``{...}`` or a joined part still counts: a call there is a call
the cell makes.

Every string a cell spells or names is a SQL candidate, except one passed
straight to ``print`` or a plot title, so the statement a ``query_minds_data``
call runs (its first argument on the desktop, its second in the cloud), or an
``.execute``, ``read_sql`` or ``.sql`` call, is one whichever argument carries
it. A string bound to a name is a candidate where the code reads that name on
its own, not where it binds it, so a base query that ``+``, ``+=`` or an
f-string extends is not one. A candidate counts when, past its comments, it
starts with SELECT or a WITH ... AS ( common table expression, reads FROM, and
does not end like a sentence: that drops ``df.query('qty > 5')``, DDL, writes
and a printed label such as ``"With rows from stock.csv:"``. A candidate that
counts is split at each ``;`` outside a string literal, a quoted name, a
dollar-quoted string or a comment, and a statement is listed only when it
counts by the same rule, so a statement after a query in the same string (DDL
that sets a password, a write) is not. The split is read with both backslash
quote rules and both line-comment rules: MySQL requires whitespace or a
control character after ``--`` and also recognizes ``#``. When any reading
splits a candidate differently, nothing from it is listed. Dollar quoting
is checked both ways too: MySQL may read ``$tag$`` as a bare alias instead.

A function the cell defines keeps the queries, requests and environment reads
its body shows, so a later cell that calls it makes them too. Rebinding one
of its global inputs drops these details, since Python resolves globals at
the call and the saved details may be stale. Client base URLs also carry
between cells, within a bounded namespace. Each name the cell binds notes
whether its value comes from data the cell read and which
names from earlier cells it is computed from, so a caller can trace a figure a
later cell prints back to the cell that read its data.
"""

from __future__ import annotations

import ast
import posixpath
import re
import textwrap
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import islice, pairwise
from types import MappingProxyType

MAX_QUERIES = 3
MAX_REQUESTS = 3
# The characters a cell's joins (+, +=, str.join and path joins), %-templates
# and f-string query substitutions may render in all. Past it they render as
# nothing, so code that doubles a string on every line reads in bounded time
# and memory.
MAX_RENDERED_CHARS = 1_000_000
# The longest string a name stays bound to, and the longest query or request a
# function keeps. Both carry into the turn's later cells, whose budgets start
# afresh, so this bounds what one name or function brings in.
MAX_STRING_CHARS = 65_536
# The most strings a name keeps from before code that may not have run.
MAX_ALTERNATIVES = 3
# The most names from earlier cells one binding lists as its inputs.
MAX_BINDING_INPUTS = 64
# The most global inputs one function tracks, and the most HTTP clients one
# scratchpad carries. Past either cap, omit details rather than retain a
# partially tracked source.
MAX_FUNCTION_GLOBALS = 64
MAX_CLIENTS = 64
# Non-SQL strings kept with a function so its file/URL details appear only
# when called. Each is also bounded by MAX_STRING_CHARS.
MAX_FUNCTION_STRINGS = 64

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})
# Modules whose request functions make an HTTP call whatever their URL reads.
_HTTP_MODULES = frozenset({"requests", "httpx"})
# The web tools, which a cell calls as functions and the model as tools.
WEB_TOOLS = frozenset({"web_search", "web_fetch"})
_UNWRAPPERS = frozenset({"text", "dedent"})
# httpx clients, whose requests take a URL relative to the base_url they were
# made with.
_BASE_URL_CLIENTS = frozenset({"Client", "AsyncClient"})
# Calls that join a path from their positional arguments.
_PATH_JOINS = frozenset({"os.path.join", "path.join", "posixpath.join"})
_PATH_TYPES = frozenset({"Path", "PurePath", "PurePosixPath"})
# Calls whose string arguments are printed or drawn as a label, not run.
_LABEL_CALLS = frozenset({"print", "title", "suptitle", "set_title"})
# Calls that write the file one argument names, with that argument's position
# and keyword: pandas, matplotlib and numpy saves, an openpyxl, python-docx or
# PIL save, a pandas ExcelWriter or xlsxwriter Workbook, and pyarrow's
# write_table.
_FILE_WRITERS = {
    "to_csv": (0, "path_or_buf"),
    "to_json": (0, "path_or_buf"),
    "to_excel": (0, "excel_writer"),
    "to_parquet": (0, "path"),
    "to_markdown": (0, "buf"),
    "to_html": (0, "buf"),
    "savefig": (0, "fname"),
    "save": (0, "filename"),
    "savetxt": (0, "fname"),
    "ExcelWriter": (0, "path"),
    "Workbook": (0, "filename"),
    "write_table": (1, "where"),
}
# Path methods that write the file their receiver names.
_PATH_WRITERS = frozenset({"write_text", "write_bytes"})
# An open() mode holding one of these writes the file.
_WRITE_MODES = frozenset("wax")
# Calls that read data, besides a request, a web tool, the data helper and an
# open() that does not write: a cursor's execute, a DuckDB or Spark sql, numpy
# and pickle loads, workbook and PDF readers, and any read_* or fetch* call.
_READ_CALLS = frozenset({"execute", "sql", "load", "load_workbook", "loadtxt", "genfromtxt", "ExcelFile", "PdfReader"})
_READ_PREFIXES = ("read_", "fetch")
# Methods that put their arguments into the object they are called on.
_MUTATORS = frozenset({"append", "extend", "insert", "update", "add"})

# An unclosed /* comments out the rest of the text, as in SQL; ending the
# match there keeps the scan linear.
_SQL_COMMENT = re.compile(r"--[^\r\n]*|/\*.*?(?:\*/|\Z)", re.DOTALL)
_SQL_SELECT = re.compile(r"select\b", re.IGNORECASE)
# WITH only opens a query as a CTE: a name (bare, quoted or a placeholder),
# optional column list, then AS (.
_SQL_CTE = re.compile(
    r"with\s+(?:recursive\s+)?(?:\w+|\"[^\"]+\"|`[^`]+`|\{[^{}]*\})(?:\s*\([^)]*\))?\s+as\s*\(",
    re.IGNORECASE,
)
_SQL_FROM = re.compile(r"\bfrom\b", re.IGNORECASE)
# A name in backticks or brackets, a dollar-quoted string, which only the
# $tag$ that opened it closes (a $ inside a name and a $1 parameter open
# none), or a comment, inside which a ; ends no statement, or a ; that does.
# A name, string or comment left open runs to the end of the text, so the
# scan stays linear.
_SQL_QUOTED_NAME = r"`[^`]*+`?|\[[^\]]*+\]?"
_SQL_DOLLAR_STRING = r"(?<![\w$])\$(?P<tag>(?:[A-Za-z_]\w*)?)\$.*?(?:\$(?P=tag)\$|\Z)"
# MySQL recognizes -- only before a space or control character, and # starts
# a comment there. Its NO_BACKSLASH_ESCAPES mode makes quote and comment
# rules independent, so check every combination rather than only two scans.
_SQL_TOKENS = tuple(
    re.compile(quote + "|" + quoted_name + "|" + comment + r"|/\*.*?(?:\*/|\Z)|;", re.DOTALL)
    for quote in (
        r"'(?:''|\\.|[^'\\])*+'?|\"(?:\"\"|\\.|[^\"\\])*+\"?",
        r"'(?:''|[^'])*+'?|\"(?:\"\"|[^\"])*+\"?",
    )
    for comment in (r"--[^\r\n]*+", r"--(?=[\x00-\x20\x7f])[^\r\n]*+|\#[^\r\n]*+")
    for quoted_name in (_SQL_QUOTED_NAME + "|" + _SQL_DOLLAR_STRING, _SQL_QUOTED_NAME)
)

URL_START = re.compile(r"https?://", re.IGNORECASE)
# A URL with a scheme and a host but no path, which can carry no token.
_BARE_ORIGIN = re.compile(r"https?://[^/?#\s]+/?", re.IGNORECASE)
# A URL's user and password sit between :// and @, whatever comes before the
# ://, a placeholder such as {proto} included. Once a : has ended the user
# name, the password may hold /, ? or #, as a database URI's may. Each part
# is bounded, so a URL with many schemes in it strips in linear time.
_USERINFO = re.compile(r"(://)(?:[^/?#@:\s]{0,256}:\S{0,1024}|[^/?#@\s]{0,256})@")
_LEADING_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
# The words of an identifier, split on _, ., brackets, quotes and camelCase.
_IDENTIFIER_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_URL_WORDS = frozenset({"url", "urls", "uri", "uris", "endpoint", "endpoints", "base", "host"})
_CONNECTION_PLACEHOLDER = re.compile(r"\{DS_[A-Z0-9_]+\}")
# A field of a "..." % value template: %s, %(key)s, %-5.2f; %% is a literal %.
_PERCENT_FIELD = re.compile(
    r"%(?:\((?P<key>[^()]*)\))?[-#0 +]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[hlL]?(?P<conversion>[diouxXeEfFgGcrsa%])"
)
# Line ends as the parser splits a cell; a form feed is not one.
_LINE_END = re.compile(rb"\r\n|\n|\r")


@dataclass(frozen=True)
class HttpRequest:
    """One HTTP call a cell makes: its method and the URL template it names,
    with no userinfo, query string or fragment (``strip_url``)."""

    method: str
    url: str

    def __str__(self) -> str:
        return f"{self.method} {self.url}"


@dataclass(frozen=True)
class FunctionReading:
    """What a function's body shows of the data and the web it reaches: a
    call to the function, in its cell or a later one, makes the same."""

    # Up to MAX_QUERIES SQL statements, as CellReading.queries has them, each
    # at most MAX_STRING_CHARS.
    queries: tuple[str, ...] = ()
    # Up to MAX_REQUESTS requests, each URL at most MAX_STRING_CHARS.
    requests: tuple[HttpRequest, ...] = ()
    # Non-SQL strings its body reads, excluding labels and write targets,
    # bounded by MAX_FUNCTION_STRINGS and MAX_STRING_CHARS per string.
    strings: tuple[str, ...] = ()
    # Strings used only as printed/drawn labels, with the same bounds as
    # strings. Keep them separate so a caller does not treat a caption as
    # a document or as a pandas index name.
    labels: tuple[str, ...] = ()
    # Labels literally spelled by this helper or a helper it calls survive
    # data-global changes. Their callable dependencies still invalidate
    # them if a wrapper's callee (or print/title itself) is rebound.
    literal_labels: tuple[str, ...] = ()
    literal_label_names: tuple[str, ...] = ()
    environment_variables: tuple[str, ...] = ()
    calls_query_minds_data: bool = False
    calls_web_tool: bool = False
    # Whether the body reads data: a file, a query, a request or a web tool.
    reads_data: bool = False
    # Global names the saved details depend on, including called helpers'
    # inputs. Rebinding any one invalidates the details. At most
    # MAX_FUNCTION_GLOBALS; an overflow keeps no query/request details.
    global_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class NameBinding:
    """Where the value a cell leaves bound to a name comes from."""

    # The value comes from data the cell read (a file, a query, a request or
    # a web tool), directly or through a name the cell bound from one. A
    # function's value does when its body reads data or draws on such a name.
    from_read: bool = False
    # The names from earlier cells the value is computed from, in code order,
    # at most MAX_BINDING_INPUTS. Code that may not have run may leave the
    # name holding the earlier cell's value, so then the name lists itself.
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CellReading:
    """What a cell's code shows of the data and the web it reaches."""

    # Up to MAX_QUERIES SQL statements in code order, deduplicated, dedented
    # and stripped.
    queries: tuple[str, ...] = ()
    # Up to MAX_REQUESTS requests in code order, deduplicated.
    requests: tuple[HttpRequest, ...] = ()
    # Every distinct string the code spells or names, in code order.
    strings: tuple[str, ...] = ()
    calls_query_minds_data: bool = False
    calls_web_tool: bool = False
    # The environment variables the code reads, in code order: a saved
    # connection's DS_ variables among them.
    environment_variables: tuple[str, ...] = ()
    # The strings the code only passes straight to print or a plot title, in
    # code order: printed or drawn, never read or run.
    labels: tuple[str, ...] = ()
    # The strings the code names as a file it writes (a save in _FILE_WRITERS,
    # a Path's write_text or write_bytes, or an open() in a write, append or
    # create mode), in code order: a file it wrote is not one it read. When
    # the file's name does not render, every string inside it is one.
    written_files: tuple[str, ...] = ()
    # The names bound to a string once the cell has run, for the next cell.
    names: Mapping[str, str] = field(default_factory=dict)
    # The functions defined once the cell has run, earlier cells' and this
    # one's, for the next cell.
    functions: Mapping[str, FunctionReading] = field(default_factory=dict)
    # HTTP client names and sanitized base URLs for the next cell, bounded
    # by MAX_CLIENTS entries and MAX_STRING_CHARS per URL.
    clients: Mapping[str, str] = field(default_factory=dict)
    # Each name the cell binds at its top level, as the cell leaves it: by an
    # assignment, a loop or with target, a walrus, a def, a class, an import
    # or a del. A name the cell fills in place (an item or attribute store, or
    # append, extend, insert, update or add) with something read or from an
    # earlier cell counts too.
    bindings: Mapping[str, NameBinding] = field(default_factory=dict)
    # The names from earlier cells the code that runs with the cell draws on
    # (a function body runs only when called), in code order: a name it reads
    # that it has not bound itself, and the inputs of a name it bound and then
    # reads. A function the cell defines draws on its inputs where the cell
    # reads the function's name, such as to call it.
    loaded_names: tuple[str, ...] = ()


_NO_FUNCTIONS: Mapping[str, FunctionReading] = MappingProxyType({})
_NO_CLIENTS: Mapping[str, str] = MappingProxyType({})


def read_cell(
    code: str, *, names: Mapping[str, str], functions: Mapping[str, FunctionReading] = _NO_FUNCTIONS,
    clients: Mapping[str, str] = _NO_CLIENTS,
) -> CellReading:
    """Read ``code`` with ``names`` (the strings earlier cells bound) and
    ``functions`` (the functions they defined) and ``clients`` (their HTTP
    client base URLs) in scope. Code that does not
    parse never ran, so it reads as nothing; code nested deeper than the
    parser or the reader can walk reads as nothing too."""
    try:
        reader = _CellReader(code, names=names, functions=functions, clients=clients)
        reader.visit(ast.parse(code))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return CellReading(names=dict(names), functions=dict(functions), clients=_bounded_clients(clients))
    return CellReading(
        queries=_queries(text for index, text in enumerate(reader.strings) if index not in reader.skipped),
        requests=tuple(_first_unique(reader.requests, limit=MAX_REQUESTS)),
        strings=tuple(dict.fromkeys(reader.strings)),
        calls_query_minds_data=reader.data_helper_calls > 0,
        calls_web_tool=reader.web_tool_calls > 0,
        environment_variables=tuple(dict.fromkeys(reader.environment_variables)),
        written_files=tuple(dict.fromkeys(reader.written_files)),
        labels=_only_labels(reader.strings, labels=reader.labels, bound=reader._bound_strings),
        names=reader.names,
        functions=reader.functions,
        clients=reader.clients,
        bindings=reader.bindings,
        loaded_names=tuple(reader.loaded_names),
    )


def is_sql(text: str) -> bool:
    """Whether ``text`` reads as a query: past its comments it starts with
    SELECT or a WITH ... AS ( common table expression, reads FROM, and does
    not end like a sentence."""
    statement = _SQL_COMMENT.sub(" ", text).strip()
    opens = bool(_SQL_SELECT.match(statement) or _SQL_CTE.match(statement))
    return opens and bool(_SQL_FROM.search(statement)) and not _ends_like_a_sentence(statement)


def strip_url(url: str) -> str:
    """``url`` without its user and password, query string or fragment."""
    return re.split(r"[?#]", _USERINFO.sub(r"\1", url.strip()), maxsplit=1)[0]


@dataclass
class _Origin:
    """What a value the reader walks draws on."""

    # A call in it reads data: a file, a query, a request or a web tool.
    calls_read: bool = False
    # It reads a name the cell bound from data it read.
    uses_read: bool = False
    # The names from earlier cells it reads, in code order, as an ordered set
    # of at most MAX_BINDING_INPUTS.
    inputs: dict[str, None] = field(default_factory=dict)

    def add_inputs(self, names: Iterable[str]) -> None:
        for name in names:
            if len(self.inputs) >= MAX_BINDING_INPUTS:
                return
            self.inputs.setdefault(name, None)

    def absorb(self, other: _Origin) -> None:
        self.calls_read = self.calls_read or other.calls_read
        self.uses_read = self.uses_read or other.uses_read
        self.add_inputs(other.inputs)

    def binding(self) -> NameBinding:
        return NameBinding(from_read=self.calls_read or self.uses_read, inputs=tuple(self.inputs))


@dataclass
class _Scope:
    """A function, lambda, comprehension or class body the reader is inside."""

    # Whether the body runs where the code spells it: a comprehension's or a
    # class's does, a function's or a lambda's runs when it is called.
    runs_now: bool
    # Its parameters and the names it has bound so far.
    names: set[str] = field(default_factory=set)
    # For each reader table entry the body changed, what puts it back as it
    # was on entry.
    undo: dict[tuple[int, str], Callable[[], None]] = field(default_factory=dict)
    # The nonlocal names a deferred body reads. Overflow prevents retaining
    # query/request details that could outlive an untracked global rebind.
    global_names: dict[str, None] = field(default_factory=dict)
    literal_label_names: dict[str, None] = field(default_factory=dict)
    globals_complete: bool = True


@dataclass(frozen=True)
class _Position:
    """How far the reader's lists and counts had grown at a point in the code."""

    strings: int
    requests: int
    environment_variables: int
    written_files: int
    data_helper_calls: int
    web_tool_calls: int


class _CellReader(ast.NodeVisitor):
    """Walks a cell in code order, binding each name assigned a string before
    the code after it reads that name."""

    def __init__(
        self, code: str, *, names: Mapping[str, str], functions: Mapping[str, FunctionReading],
        clients: Mapping[str, str],
    ) -> None:
        # The cell as UTF-8 and the byte offset each line starts at: a node's
        # col_offset counts bytes, and slicing here costs only the node's size.
        self._code = code.encode()
        self._line_starts = [0, *(match.end() for match in _LINE_END.finditer(self._code))]
        # The names bound to a string, the functions defined and the httpx
        # clients' base URLs, as the code where the reader stands sees them.
        self.names: dict[str, str] = dict(names)
        self.functions: dict[str, FunctionReading] = dict(functions)
        self.clients: dict[str, str] = _bounded_clients(clients)
        # A reverse index avoids walking every function for every assignment.
        self._dependents: dict[str, set[str]] = {}
        for name, function in self.functions.items():
            self._remember_dependencies(name, function=function)
        # Strings a name may still hold from before code that may not have
        # run and that rebound it, besides the one in names.
        self._alternatives: dict[str, tuple[str, ...]] = {}
        self.bindings: dict[str, NameBinding] = {}
        self.loaded_names: dict[str, None] = {}
        self.strings: list[str] = []
        # Literal strings, rather than rendered globals: a helper's defining
        # cell retains these descriptive labels for its lineage, even though
        # its query/request details belong only to a call.
        self._literal_strings: set[int] = set()
        # A binding spells a string without reading it. These occurrences
        # do not prevent a later print(name) from being label-only.
        self._bound_strings: set[int] = set()
        # Indexes into strings of the ones that are not SQL candidates where
        # they stand: a string a name was bound to runs where the code reads
        # the name on its own, and a label is printed or drawn, not run.
        self.skipped: set[int] = set()
        # Indexes into strings of the ones passed straight to a label call.
        self.labels: set[int] = set()
        self.requests: list[HttpRequest] = []
        self.data_helper_calls = 0
        self.web_tool_calls = 0
        self.environment_variables: list[str] = []
        self.written_files: list[str] = []
        # Per open piece of code that may not run, innermost last: the strings
        # each name it rebound held before.
        self._branches: list[dict[str, list[str]]] = []
        self._scopes: list[_Scope] = []
        # Per value being walked for a name it binds, innermost last.
        self._origins: list[_Origin] = []
        self._rendered_chars = 0
        # Whether each bound string reads as a query, worked out once: a cell
        # can read one name in many f-string placeholders.
        self._queries: dict[str, bool] = {}

    def visit_Expr(self, node: ast.Expr) -> None:
        # A bare string statement is a docstring, not something the cell runs.
        if not _is_string_literal(node.value):
            self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        first = len(self.strings)
        with self._collecting() as origin:
            self.visit(node.value)
        rendered = self._string(node.value)
        client = self._client_base(node.value)
        for target in node.targets:
            self._assign_target(target, origin=origin, value=rendered, client=client)
        if all(isinstance(target, ast.Name) and target.id in self.names for target in node.targets):
            self._mark_bound(index=first, rendered=self.names[node.targets[0].id])

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        first = len(self.strings)
        with self._collecting() as origin:
            self.visit(node.value)
        self._assign_target(
            node.target, origin=origin, value=self._string(node.value), client=self._client_base(node.value),
        )
        if isinstance(node.target, ast.Name) and node.target.id in self.names:
            self._mark_bound(index=first, rendered=self.names[node.target.id])

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        with self._collecting() as origin:
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                # `x += v` reads x as well.
                self._note_load(node.target.id)
        if not isinstance(node.target, ast.Name):
            self._assign_target(node.target, origin=origin, value=None, client=None)
            return
        name = node.target.id
        joined = None
        if isinstance(node.op, ast.Add) and name in self.names:
            joined = self._joined([self.names[name], self._operand_text(node.value)])
        self._store(name, origin=origin, value=joined)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        with self._collecting() as origin:
            self.visit(node.value)
        self._assign_target(node.target, origin=origin, value=self._string(node.value), client=None)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        with self._collecting() as origin:
            self.visit(node.iter)
        self._assign_target(node.target, origin=origin, value=None, client=None)
        with self._branch():
            for statement in (*node.body, *node.orelse):
                self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        with self._branch():
            for statement in (*node.body, *node.orelse):
                self.visit(statement)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        with self._branch():
            for statement in (*node.body, *node.orelse):
                self.visit(statement)

    def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
        for statement in node.body:
            self.visit(statement)
        # Either a handler runs or the else body does.
        with self._branch():
            for handler in node.handlers:
                self.visit(handler)
            for statement in node.orelse:
                self.visit(statement)
        for statement in node.finalbody:
            self.visit(statement)

    visit_TryStar = visit_Try

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        with self._branch():
            for case in node.cases:
                self.visit(case)

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            with self._collecting() as origin:
                self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._assign_target(
                    item.optional_vars, origin=origin, value=None, client=self._client_base(item.context_expr),
                )
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self._store(alias.asname or alias.name.split(".")[0], origin=_Origin(), value=None)

    visit_ImportFrom = visit_Import

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_signature(node.args, returns=node.returns)
        start = self._position()
        with self._collecting() as origin, self._scope(parameters=_parameters(node.args), runs_now=False) as scope:
            for statement in node.body:
                self.visit(statement)
        function = self._function_reading(start, reads_data=origin.calls_read, scope=scope)
        literal_strings = tuple(dict.fromkeys(
            text for index, text in enumerate(self.strings[start.strings:], start.strings)
            if index in self._literal_strings and text in function.strings
        ))
        # A def evaluates its signature, not its body. Only a later call may
        # emit query/request details, after intervening global rebindings
        # clear them. Keep only non-SQL source literals for the defining
        # cell's lineage, such as the file its body spells. Rendered globals
        # here would still expose stale same-cell details.
        self._rewind(start)
        self.strings.extend(literal_strings)
        retained = range(start.strings, len(self.strings))
        self.skipped.update(retained)
        self._literal_strings.update(retained)
        self._store(node.name, origin=origin, value=None)
        self._put(self.functions, key=node.name, value=function)
        if not self._scopes:
            self._remember_dependencies(node.name, function=function)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_signature(node.args, returns=None)
        with self._scope(parameters=_parameters(node.args), runs_now=False):
            self.visit(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *(keyword.value for keyword in node.keywords)):
            self.visit(expression)
        with self._collecting() as origin, self._scope(parameters=(), runs_now=True):
            for statement in node.body:
                self.visit(statement)
        self._store(node.name, origin=origin, value=None)

    def visit_ListComp(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, node.elt)

    visit_SetComp = visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, node.key, node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._note_load(node.id)
            self.strings.extend(self._alternatives.get(node.id, ()))
            if node.id in self.names:
                self.strings.append(self.names[node.id])
        else:
            # Deleted: no longer the string, function or value it was.
            self._store(node.id, origin=_Origin(), value=None)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._literal_strings.add(len(self.strings))
            self.strings.append(node.value)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.strings.append(self._text(node))
        self._visit_placeholders(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        rendered = self._string(node)
        if rendered is not None:
            self.strings.append(rendered)
        if isinstance(node.op, (ast.Add, ast.Mod, ast.Div)):
            # A string's literal parts are the string itself, rendered or past
            # the budget, and never strings of their own.
            self._visit_placeholders(node)
        else:
            self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._environment_variable(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._environment_variable(node)
        name = _called_name(node.func)
        if name == "query_minds_data":
            self.data_helper_calls += 1
        elif name in WEB_TOOLS:
            self.web_tool_calls += 1
        request = self._request(node)
        if request is not None:
            self.requests.append(request)
        target = self._write_target(node)
        if target is not None:
            self.written_files.extend(self._written(target))
        if request is not None or _reads_data(name, writes=target is not None):
            self._note_read()
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            self._run(self.functions[node.func.id], name=node.func.id)
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        if name in _MUTATORS and isinstance(receiver, ast.Name):
            with self._collecting() as origin:
                self._visit_call_parts(node, name=name)
            self._mutate(receiver.id, origin=origin)
        else:
            self._visit_call_parts(node, name=name)

    def _visit_call_parts(self, node: ast.Call, *, name: str | None) -> None:
        if _joins_a_path(node) or _joins_a_sequence(node):
            rendered = self._string(node)
            if rendered is not None:
                # A joined path or string is one string: its parts are not
                # strings of their own.
                self.strings.append(rendered)
                self._visit_rendered_call(node)
                return
        if name in _LABEL_CALLS:
            self.visit(node.func)
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                first = len(self.strings)
                self.visit(argument)
                if isinstance(argument, ast.Name):
                    # A bound name may have several conditional values, all
                    # printed here. Unlike a call/f-string, visiting a Name
                    # cannot append strings from nested reads.
                    printed = range(first, len(self.strings))
                    self.skipped.update(printed)
                    self.labels.update(printed)
                elif _is_string_literal(argument) and first < len(self.strings):
                    self.skipped.add(first)
                    self.labels.add(first)
                    if first in self._literal_strings:
                        callable_name = _root_name(node.func)
                        if callable_name is not None:
                            self._note_global(callable_name, literal_label=True)
            return
        self.generic_visit(node)

    def _visit_comprehension(self, generators: Sequence[ast.comprehension], *elements: ast.expr) -> None:
        """Walk a comprehension as it runs: its first iterable in the
        enclosing scope, then its targets, conditions and elements in a scope
        of their own."""
        self.visit(generators[0].iter)
        with self._scope(parameters=(), runs_now=True):
            for index, generator in enumerate(generators):
                if index:
                    self.visit(generator.iter)
                self._assign_target(generator.target, origin=_Origin(), value=None, client=None)
                for condition in generator.ifs:
                    self.visit(condition)
            for element in elements:
                self.visit(element)

    def _visit_signature(self, arguments: ast.arguments, *, returns: ast.expr | None) -> None:
        """Visit what a def or lambda evaluates where it is defined: its
        defaults and annotations."""
        parameters = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        optional = (arguments.vararg, arguments.kwarg)
        annotations = (parameter.annotation for parameter in (*parameters, *optional) if parameter is not None)
        for expression in (*arguments.defaults, *arguments.kw_defaults, *annotations, returns):
            if expression is not None:
                self.visit(expression)

    def _visit_placeholders(self, node: ast.expr) -> None:
        """Visit each expression the string ``node`` renders as ``{expr}``, so
        a call, an environment read or a string inside one still counts. An
        expression that renders as a string itself, such as a bound name, is
        part of this string and not visited again."""
        if isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    self._visit_placeholders(part.value)
                    if part.format_spec is not None:
                        self._visit_placeholders(part.format_spec)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            for operand in _operands(node):
                self._visit_placeholders(operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            # "... %s" % values fills each field with a value, as an f-string does.
            self._visit_placeholders(node.left)
            for value in node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]:
                self._visit_placeholders(value)
        elif self._string(node) is None:
            self.visit(node)
        elif isinstance(node, ast.Call):
            self._visit_rendered_call(node)

    def _visit_rendered_call(self, node: ast.Call) -> None:
        """Visit the code inside a call that renders as a string, and not its
        text again: a part the string is made of counts only for the code
        inside it, while a .format(...) argument is code the call runs."""
        if _joins_a_sequence(node) and isinstance(node.func, ast.Attribute):
            self._visit_placeholders(node.func.value)
            for element in node.args[0].elts:
                self._visit_placeholders(element)
            return
        if _called_name(node.func) == "format" and isinstance(node.func, ast.Attribute):
            self._visit_placeholders(node.func.value)
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                self.visit(argument)
            return
        for argument in node.args:
            self._visit_placeholders(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def _environment_variable(self, node: ast.expr) -> str | None:
        """The variable ``node`` reads from the environment, noted as one the
        cell reads."""
        variable = _environment_variable(node)
        if variable is not None:
            self.environment_variables.append(variable)
        return variable

    def _assign_target(
        self, target: ast.expr, *, origin: _Origin, value: str | None, client: str | None,
    ) -> None:
        """Bind what ``target`` names: a name to ``value`` and to the httpx
        ``client`` base URL when there is one, each name an unpacking binds to
        no string, and the object an item or attribute store fills in place."""
        if isinstance(target, ast.Name):
            self._store(target.id, origin=origin, value=value)
            if client is not None and len(self.clients) < MAX_CLIENTS:
                self._put(self.clients, key=target.id, value=client)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._assign_target(element, origin=origin, value=None, client=None)
        elif isinstance(target, ast.Starred):
            self._assign_target(target.value, origin=origin, value=None, client=None)
        else:
            self.visit(target)
            root = _root_name(target)
            if root is not None:
                self._mutate(root, origin=origin)

    def _store(self, name: str, *, origin: _Origin, value: str | None) -> None:
        """Bind ``name`` where the reader stands: to ``value`` when that is a
        string, and at the cell's top level to what ``origin`` drew on. It no
        longer names a function or a client."""
        if not self._scopes:
            self._invalidate_dependents(name)
        self._set_name(name, value=value)
        self._put(self.functions, key=name, value=None)
        self._put(self.clients, key=name, value=None)
        if self._scopes:
            self._scopes[-1].names.add(name)
            return
        binding = origin.binding()
        if self._branches:
            binding = _merged(self.bindings.get(name, NameBinding(inputs=(name,))), binding)
        self.bindings[name] = binding

    def _remember_dependencies(self, name: str, *, function: FunctionReading) -> None:
        for global_name in function.global_names:
            self._dependents.setdefault(global_name, set()).add(name)

    def _invalidate_dependents(self, name: str) -> None:
        for dependent in self._dependents.pop(name, ()):
            function = self.functions.get(dependent)
            if function is not None and name in function.global_names:
                literal_labels = () if name in function.literal_label_names else function.literal_labels
                self.functions[dependent] = replace(
                    function, queries=(), requests=(), strings=(), labels=literal_labels,
                    literal_labels=literal_labels, environment_variables=(),
                )

    def _mutate(self, name: str, *, origin: _Origin) -> None:
        """Note that top-level code put what ``origin`` drew on into the object
        ``name`` holds."""
        if self._is_local(name) or self._deferred():
            return
        self._invalidate_dependents(name)
        self._put(self.clients, key=name, value=None)
        if not (origin.calls_read or origin.uses_read or origin.inputs):
            return
        self.bindings[name] = _merged(self.bindings.get(name, NameBinding(inputs=(name,))), origin.binding())

    def _set_name(self, name: str, *, value: str | None) -> None:
        """Bind ``name`` to ``value``, or unbind it when the code gave it no
        string or one longer than MAX_STRING_CHARS. In code that may not run,
        keep what it held before for the code after."""
        if self._branches:
            earlier = self._branches[-1].setdefault(name, [])
            held = (*self._alternatives.get(name, ()), *((self.names[name],) if name in self.names else ()))
            for text in held:
                if text not in earlier and len(earlier) < MAX_ALTERNATIVES:
                    earlier.append(text)
        self._put(self._alternatives, key=name, value=None)
        self._put(self.names, key=name, value=value if value is not None and len(value) <= MAX_STRING_CHARS else None)

    def _put[V](self, table: dict[str, V], *, key: str, value: V | None) -> None:
        """Set ``table[key]`` to ``value``, or remove it for None, noting the
        entry as it was for the innermost scope to put back."""
        if self._scopes:
            undo = self._scopes[-1].undo
            if (id(table), key) not in undo:
                undo[(id(table), key)] = partial(_set_entry, table, key=key, value=table.get(key))
        _set_entry(table, key=key, value=value)

    @contextmanager
    def _branch(self) -> Iterator[None]:
        """Walk code that may not run: after it, a name it rebound may still
        hold what it held before."""
        earlier_by_name: dict[str, list[str]] = {}
        self._branches.append(earlier_by_name)
        try:
            yield
        finally:
            self._branches.pop()
        for name, earlier in earlier_by_name.items():
            current = self.names.get(name)
            texts = dict.fromkeys((*earlier, *self._alternatives.get(name, ())))
            kept = tuple(text for text in texts if text != current)[:MAX_ALTERNATIVES]
            self._put(self._alternatives, key=name, value=kept or None)
            # A conditional client rebind may leave either client alive. Do
            # not carry the last branch's base URL as though it were certain.
            self._put(self.clients, key=name, value=None)

    @contextmanager
    def _scope(self, *, parameters: Iterable[str], runs_now: bool) -> Iterator[_Scope]:
        """Walk a body whose parameters and locals are its own: what it binds
        is put back as it was once the body ends."""
        scope = _Scope(runs_now=runs_now)
        self._scopes.append(scope)
        branches, self._branches = self._branches, []
        try:
            for parameter in parameters:
                self._store(parameter, origin=_Origin(), value=None)
            yield scope
        finally:
            self._branches = branches
            self._scopes.pop()
            for restore in scope.undo.values():
                restore()

    @contextmanager
    def _collecting(self) -> Iterator[_Origin]:
        """Gather what the code walked inside draws on, for the name it binds;
        the enclosing value draws on it too."""
        origin = _Origin()
        self._origins.append(origin)
        try:
            yield origin
        finally:
            self._origins.pop()
        if self._origins:
            self._origins[-1].absorb(origin)

    def _note_load(self, name: str) -> None:
        """Note what reading ``name`` draws on: the name itself when it comes
        from an earlier cell, or the inputs of the binding this cell gave it."""
        self._note_global(name)
        if self._is_local(name):
            return
        binding = self.bindings.get(name)
        inputs = (name,) if binding is None else binding.inputs
        if not self._deferred():
            for input_name in inputs:
                self.loaded_names.setdefault(input_name, None)
        if self._origins:
            origin = self._origins[-1]
            origin.uses_read = origin.uses_read or (binding is not None and binding.from_read)
            origin.add_inputs(inputs)

    def _note_global(self, name: str, *, from_function: bool = False, literal_label: bool = False) -> None:
        """Track globals used by a deferred body, including rendered names
        that are not visited again as standalone strings."""
        for index in range(len(self._scopes) - 1, -1, -1):
            scope = self._scopes[index]
            if not scope.runs_now:
                if not from_function:
                    if any(name in local.names for local in self._scopes[index:]):
                        return
                    if any(name in outer.names for outer in self._scopes[:index]):
                        # A closure can also change before the call. Tracking
                        # its local namespace would require executing the
                        # enclosing function, so do not freeze its details.
                        scope.globals_complete = False
                        return
                if len(scope.global_names) < MAX_FUNCTION_GLOBALS:
                    scope.global_names.setdefault(name, None)
                elif name not in scope.global_names:
                    scope.globals_complete = False
                if literal_label and name in scope.global_names:
                    scope.literal_label_names.setdefault(name, None)
                return

    def _note_read(self) -> None:
        if self._origins:
            self._origins[-1].calls_read = True

    def _is_local(self, name: str) -> bool:
        return any(name in scope.names for scope in self._scopes)

    def _deferred(self) -> bool:
        """Whether the reader is inside a function or lambda body, which runs
        when called rather than where the code spells it."""
        return any(not scope.runs_now for scope in self._scopes)

    def _run(self, function: FunctionReading, *, name: str) -> None:
        """Count a call to a function the cell or an earlier one defined: it
        runs the queries and requests its body shows."""
        self.strings.extend(function.queries)
        start = len(self.strings)
        self.strings.extend(function.strings)
        self.skipped.update(range(start, len(self.strings)))
        start = len(self.strings)
        self.strings.extend(function.labels)
        printed = range(start, len(self.strings))
        self.skipped.update(printed)
        self.labels.update(printed)
        self._literal_strings.update(
            index for index in printed if self.strings[index] in function.literal_labels
        )
        if function.literal_labels:
            self._note_global(name, literal_label=True)
            for dependency in function.literal_label_names:
                self._note_global(dependency, from_function=True, literal_label=True)
        for name in function.global_names:
            # A helper resolves its own globals even when the calling
            # function has locals with those names.
            self._note_global(name, from_function=True)
        self.requests.extend(function.requests)
        self.environment_variables.extend(function.environment_variables)
        self.data_helper_calls += function.calls_query_minds_data
        self.web_tool_calls += function.calls_web_tool
        if function.reads_data:
            self._note_read()

    def _position(self) -> _Position:
        return _Position(
            strings=len(self.strings),
            requests=len(self.requests),
            environment_variables=len(self.environment_variables),
            written_files=len(self.written_files),
            data_helper_calls=self.data_helper_calls,
            web_tool_calls=self.web_tool_calls,
        )

    def _function_reading(self, start: _Position, *, reads_data: bool, scope: _Scope) -> FunctionReading:
        """What the code walked since ``start``, a function's body, shows."""
        candidates = (
            text for index, text in enumerate(self.strings[start.strings:], start.strings)
            if index not in self.skipped and len(text) <= MAX_STRING_CHARS
        )
        requests = (request for request in self.requests[start.requests:] if len(request.url) <= MAX_STRING_CHARS)
        body_strings = self.strings[start.strings:]
        labels = {index for index in range(len(body_strings)) if index + start.strings in self.labels}
        bound = {index for index in range(len(body_strings)) if index + start.strings in self._bound_strings}
        label_strings = _only_labels(body_strings, labels=labels, bound=bound)
        literal_labels = {
            text for index, text in enumerate(body_strings, start.strings)
            if index in self.labels and index in self._literal_strings
        }
        not_read = {*self.written_files[start.written_files:], *label_strings}
        strings = (
            text for text in body_strings
            if len(text) <= MAX_STRING_CHARS and text not in not_read and not self._is_query(text)
        )
        return FunctionReading(
            queries=_queries(candidates) if scope.globals_complete else (),
            requests=tuple(_first_unique(requests, limit=MAX_REQUESTS)) if scope.globals_complete else (),
            strings=tuple(_first_unique(strings, limit=MAX_FUNCTION_STRINGS)) if scope.globals_complete else (),
            labels=tuple(
                text for text in label_strings if len(text) <= MAX_STRING_CHARS
            )[:MAX_FUNCTION_STRINGS] if scope.globals_complete else (),
            literal_labels=tuple(
                text for text in label_strings if text in literal_labels and len(text) <= MAX_STRING_CHARS
            )[:MAX_FUNCTION_STRINGS] if scope.globals_complete else (),
            literal_label_names=tuple(scope.literal_label_names),
            environment_variables=(
                tuple(dict.fromkeys(self.environment_variables[start.environment_variables:]))
                if scope.globals_complete else ()
            ),
            calls_query_minds_data=self.data_helper_calls > start.data_helper_calls,
            calls_web_tool=self.web_tool_calls > start.web_tool_calls,
            reads_data=reads_data,
            global_names=tuple(scope.global_names),
        )

    def _rewind(self, start: _Position) -> None:
        removed = range(start.strings, len(self.strings))
        self.skipped.difference_update(removed)
        self.labels.difference_update(removed)
        self._literal_strings.difference_update(removed)
        self._bound_strings.difference_update(removed)
        del self.strings[start.strings:]
        del self.requests[start.requests:]
        del self.environment_variables[start.environment_variables:]
        del self.written_files[start.written_files:]
        self.data_helper_calls = start.data_helper_calls
        self.web_tool_calls = start.web_tool_calls

    def _mark_bound(self, *, index: int, rendered: str) -> None:
        """Note ``strings[index]`` as the string a name was just bound to, when
        it is: it runs where the code reads the name on its own. A string from
        code inside the value, such as a call in an f-string's placeholder,
        comes after it and still runs here."""
        if index < len(self.strings) and self.strings[index] == rendered:
            self.skipped.add(index)
            self._bound_strings.add(index)

    def _string(self, node: ast.expr) -> str | None:
        """The string ``node`` stands for, or None when it isn't one."""
        variable = self._environment_variable(node)
        if variable is not None:
            return "{" + variable + "}"
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.JoinedStr):
            return "".join(self._fstring_part(part) for part in node.values)
        if isinstance(node, ast.Name):
            self._note_global(node.id)
            return self.names.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._concatenated(_operands(node))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return self._path(_operands(node))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return self._formatted(node)
        if isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name in _UNWRAPPERS and node.args:
                return self._string(node.args[0])
            if _joins_a_path(node):
                return self._path(node.args)
            if _joins_a_sequence(node) and isinstance(node.func, ast.Attribute):
                separator = self._string(node.func.value)
                if separator is not None:
                    return self._concatenated(node.args[0].elts, separator=separator)
            if name == "format" and isinstance(node.func, ast.Attribute):
                return self._string(node.func.value)
        return None

    def _operand(self, node: ast.expr) -> str | None:
        """How ``node`` reads as a part of a ``+``, ``+=`` or str.join: a bound
        name as ``{name}`` unless it holds a saved connection's lookup, a query
        or a base URL; anything else as it renders, or None when it isn't a
        string."""
        if isinstance(node, ast.Name) and node.id in self.names:
            self._note_global(node.id)
            bound = self.names[node.id]
            reads_inline = _CONNECTION_PLACEHOLDER.fullmatch(bound) or self._is_query(bound) or _names_a_url(bound)
            return bound if reads_inline else "{" + node.id + "}"
        return self._string(node)

    def _operand_text(self, node: ast.expr) -> str:
        rendered = self._operand(node)
        return rendered if rendered is not None else self._placeholder(node)

    def _concatenated(self, parts: Sequence[ast.expr], *, separator: str = "") -> str | None:
        """``parts`` joined by ``separator``, or None when no part is a string
        or the cell's renders would pass MAX_RENDERED_CHARS."""
        rendered = [self._operand(part) for part in parts]
        if all(text is None for text in rendered):
            return None
        texts = [text if text is not None else self._placeholder(part) for text, part in zip(rendered, parts)]
        return self._joined(texts, separator=separator)

    def _path(self, parts: Sequence[ast.expr]) -> str | None:
        """``parts`` joined as a path, as os.path.join joins them, or None when
        no part is a string or the cell's renders would pass
        MAX_RENDERED_CHARS. A bound name reads as its string, except in a
        join that starts with a URL, where it reads as it does in a ``+``."""
        rendered = [self._string(part) for part in parts]
        if all(text is None for text in rendered):
            return None
        if rendered[0] is not None and URL_START.match(rendered[0]):
            rendered = [rendered[0], *(self._operand(part) for part in parts[1:])]
        texts = (text if text is not None else self._placeholder(part) for text, part in zip(rendered, parts))
        joined = posixpath.join(*texts)
        return joined if self._spend(len(joined)) else None

    def _formatted(self, node: ast.BinOp) -> str | None:
        """``"..." % value`` with each field read as an f-string reads
        ``{expr}``, or None when the template isn't a string or the cell's
        renders would pass MAX_RENDERED_CHARS."""
        template = self._string(node.left)
        if template is None:
            return None
        values = list(node.right.elts) if isinstance(node.right, ast.Tuple) else [node.right]
        position = 0

        def field_text(match: re.Match[str]) -> str:
            nonlocal position
            if match.group("conversion") == "%":
                return "%"
            if match.group("key") is not None:
                return "{" + match.group("key") + "}"
            value = values[position] if position < len(values) else None
            position += 1
            return self._field(value) if value is not None else match.group(0)

        rendered = _PERCENT_FIELD.sub(field_text, template)
        return rendered if self._spend(len(rendered)) else None

    def _joined(self, texts: Sequence[str], *, separator: str = "") -> str | None:
        """``texts`` joined by ``separator``, or None once the cell's renders
        would pass MAX_RENDERED_CHARS in all. A join is charged once, for what
        it renders."""
        chars = sum(len(text) for text in texts) + len(separator) * max(len(texts) - 1, 0)
        return separator.join(texts) if self._spend(chars) else None

    def _spend(self, chars: int) -> bool:
        """Count ``chars`` against the cell's MAX_RENDERED_CHARS; False once
        they pass it."""
        self._rendered_chars += chars
        return self._rendered_chars <= MAX_RENDERED_CHARS

    def _fstring_part(self, part: ast.expr) -> str:
        """A literal part as it reads, and a placeholder as ``_field`` reads
        it; one with a conversion or format spec reads as written unless it
        is an environment lookup."""
        if isinstance(part, ast.Constant):
            return str(part.value)
        if not isinstance(part, ast.FormattedValue):
            return self._source(part)
        if part.conversion == -1 and part.format_spec is None:
            return self._field(part.value, written=self._source(part))
        variable = self._environment_variable(part.value)
        return "{" + variable + "}" if variable is not None else self._source(part)

    def _field(self, node: ast.expr, *, written: str | None = None) -> str:
        """An expression a template substitutes, as it reads there: an
        environment lookup as ``{KEY}``, a bare name bound to a saved
        connection's lookup or to a query as that string, and anything else
        as ``written``, by default ``{expr}``."""
        variable = self._environment_variable(node)
        if variable is not None:
            return "{" + variable + "}"
        if isinstance(node, ast.Name):
            self._note_global(node.id)
            bound = self.names.get(node.id)
            if bound is not None and (
                _CONNECTION_PLACEHOLDER.fullmatch(bound) or (self._is_query(bound) and self._spend(len(bound)))
            ):
                return bound
        return written if written is not None else self._placeholder(node)

    def _is_query(self, text: str) -> bool:
        if text not in self._queries:
            self._queries[text] = is_sql(text)
        return self._queries[text]

    def _text(self, node: ast.expr) -> str:
        """The string ``node`` stands for, or its source as a placeholder."""
        rendered = self._string(node)
        return rendered if rendered is not None else self._placeholder(node)

    def _placeholder(self, node: ast.expr) -> str:
        return "{" + self._source(node) + "}"

    def _source(self, node: ast.AST) -> str:
        try:
            start = self._line_starts[node.lineno - 1] + node.col_offset
            end = self._line_starts[node.end_lineno - 1] + node.end_col_offset
            segment = self._code[start:end].decode()
        except (AttributeError, TypeError, IndexError, UnicodeDecodeError):
            return ast.unparse(node)
        return segment or ast.unparse(node)

    def _client_base(self, node: ast.expr) -> str | None:
        """The base URL an httpx ``Client(base_url=...)`` is made with, as a
        request through it shows it: a saved connection's lookup, or a scheme
        and host with no path, as written; any other base as ``{base_url}``,
        so a token in its path never reaches the request."""
        if not (isinstance(node, ast.Call) and _called_name(node.func) in _BASE_URL_CLIENTS):
            return None
        base = next((keyword.value for keyword in node.keywords if keyword.arg == "base_url"), None)
        if base is None:
            return None
        rendered = self._string(base)
        if rendered is not None and len(rendered) <= MAX_STRING_CHARS:
            stripped = strip_url(rendered).rstrip("/")
            if _CONNECTION_PLACEHOLDER.fullmatch(stripped) or _BARE_ORIGIN.fullmatch(stripped):
                return stripped
        return "{base_url}"

    def _request(self, node: ast.Call) -> HttpRequest | None:
        """The request ``node`` makes, from its method and URL alone: a call's
        headers, auth, body, params and cookies never enter it. A relative URL
        on an httpx client made with a base_url is joined onto that base."""
        func = node.func
        name = _called_name(func)
        is_method = isinstance(func, ast.Attribute)
        receiver = _dotted(func.value) if is_method else None
        if is_method and name in _HTTP_METHODS:
            url, method = _argument(node, position=0, keyword="url"), name.upper()
        elif is_method and name == "request":
            url = _argument(node, position=1, keyword="url")
            method = self._method(_argument(node, position=0, keyword="method"))
        elif name == "urlopen":
            url, method = _argument(node, position=0, keyword="url"), _get_or_post(node)
            # urlopen(Request(...)): the Request call is the request.
            if isinstance(url, ast.Call):
                return None
        elif name == "Request" and receiver in (None, "request", "urllib.request"):
            url = _argument(node, position=0, keyword="url")
            method_argument = _argument(node, position=5, keyword="method")
            method = _get_or_post(node) if method_argument is None else self._method(method_argument)
        else:
            return None
        if url is None:
            return None
        template = strip_url(self._text(url))
        base = self.clients.get(receiver) if receiver is not None else None
        if base is not None and not URL_START.match(template):
            template = strip_url(base + "/" + template.lstrip("/"))
        if not template or not (receiver in _HTTP_MODULES or base is not None or _names_a_url(template)):
            return None
        return HttpRequest(method=method, url=template)

    def _method(self, node: ast.expr | None) -> str:
        """The method a call names, upper-cased, or its source as a
        placeholder when the code doesn't spell it."""
        if node is None:
            return "GET"
        rendered = self._string(node)
        return rendered.upper() if rendered is not None else self._placeholder(node)

    def _write_target(self, node: ast.Call) -> ast.expr | None:
        """The expression naming the file ``node`` writes, when it writes one:
        a save in _FILE_WRITERS, a Path's write_text or write_bytes, or an
        open() in a write, append or create mode. A receiver that renders as a
        string is a Path, whose open takes its mode first."""
        name = _called_name(node.func)
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        if name in _FILE_WRITERS:
            position, keyword = _FILE_WRITERS[name]
            return _argument(node, position=position, keyword=keyword)
        if name in _PATH_WRITERS:
            return receiver
        if name == "open" and receiver is not None and self._string(receiver) is not None:
            return receiver if self._writes(_argument(node, position=0, keyword="mode")) else None
        if name == "open" and self._writes(_argument(node, position=1, keyword="mode")):
            return _argument(node, position=0, keyword="file")
        return None

    def _written(self, target: ast.expr) -> list[str]:
        """The files ``target`` names: the string it renders as, or when it
        doesn't render, every string inside it."""
        rendered = self._string(target)
        if rendered is not None:
            return [rendered]
        texts = (self._string(node) for node in ast.walk(target) if _is_string_literal(node))
        return [text for text in texts if text is not None]

    def _writes(self, mode: ast.expr | None) -> bool:
        """Whether an open() mode writes its file: it holds w, a or x."""
        rendered = self._string(mode) if mode is not None else None
        return rendered is not None and bool(_WRITE_MODES & set(rendered))


def _queries(texts: Iterable[str]) -> tuple[str, ...]:
    """Up to MAX_QUERIES distinct SQL statements among ``texts``, in order,
    dedented and stripped. A text that reads as a query is split into its
    statements (_statements), and only those that read as a query themselves
    are listed, so a statement after a query in the same text (DDL that sets
    a password, a write) is listed only where the database ends a statement
    at a ``;`` that _statements reads as inside a literal, a quoted name or a
    comment."""
    statements = (
        textwrap.dedent(statement).strip()
        for text in dict.fromkeys(texts) if is_sql(text)
        for statement in _statements(text) if is_sql(statement)
    )
    return tuple(_first_unique(statements, limit=MAX_QUERIES))


def _statements(text: str) -> list[str]:
    """``text`` split after each ``;`` outside a string literal, a quoted
    name, a dollar-quoted string or a comment; each statement keeps its
    ``;``. Empty when supported quote or comment rules disagree about a
    split, because no database dialect accompanies the extracted text."""
    ends = _statement_ends(text, token=_SQL_TOKENS[0])
    if ends is None or any(ends != _statement_ends(text, token=token) for token in _SQL_TOKENS[1:]):
        return []
    return [text[start:end] for start, end in pairwise([0, *ends, len(text)])]


def _statement_ends(text: str, *, token: re.Pattern[str]) -> list[int] | None:
    """Where each statement in ``text`` ends, just past each ``;`` that
    ``token`` reads as one."""
    ends = []
    for match in token.finditer(text):
        value = match.group()
        if value == ";":
            ends.append(match.end())
        elif value.startswith("/*") and (
            (value.endswith("*/") and "/*" in value[2:]) or value.startswith(("/*!", "/*M!"))
        ):
            # PostgreSQL nests block comments; MySQL does not, and MySQL /
            # MariaDB executable comments can run their contents. A dialect
            # is not available here, so do not attach this batch as SQL.
            return None
    return ends


def _bounded_clients(clients: Mapping[str, str]) -> dict[str, str]:
    return {
        name: base for name, base in islice(clients.items(), MAX_CLIENTS)
        if len(base) <= MAX_STRING_CHARS
    }


def _only_labels(strings: Sequence[str], *, labels: set[int], bound: set[int]) -> tuple[str, ...]:
    """Printed labels, excluding strings also used as source paths. An
    identifier cannot name a URL, document path or SQL query, and may also
    occur as Series/column metadata, so those other uses do not erase its
    known role as a caption."""
    elsewhere = {text for index, text in enumerate(strings) if index not in labels and index not in bound}
    return tuple(dict.fromkeys(
        text for index, text in enumerate(strings)
        if index in labels and (text not in elsewhere or text.strip().isidentifier())
    ))


def _merged(*bindings: NameBinding) -> NameBinding:
    """A binding whose value may be any of ``bindings``'."""
    inputs = dict.fromkeys(name for binding in bindings for name in binding.inputs)
    return NameBinding(
        from_read=any(binding.from_read for binding in bindings), inputs=tuple(inputs)[:MAX_BINDING_INPUTS],
    )


def _set_entry[V](table: dict[str, V], *, key: str, value: V | None) -> None:
    if value is None:
        table.pop(key, None)
    else:
        table[key] = value


def _reads_data(name: str | None, *, writes: bool) -> bool:
    """Whether a call by this name reads data, apart from a request: the data
    helper, a web tool, a call in _READ_CALLS or with a _READ_PREFIXES name,
    or an open() that does not write."""
    if name is None:
        return False
    return (
        name == "query_minds_data"
        or name in WEB_TOOLS
        or name in _READ_CALLS
        or name.startswith(_READ_PREFIXES)
        or (name == "open" and not writes)
    )


def _operands(node: ast.BinOp) -> list[ast.expr]:
    """The operands of a chain of ``node``'s operator, left to right:
    ``a + b + c`` has three. Walked without recursion, so a long chain costs
    no stack."""
    operator = type(node.op)
    operands: list[ast.expr] = []
    pending: list[ast.expr] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.BinOp) and isinstance(current.op, operator):
            pending.extend((current.right, current.left))
        else:
            operands.append(current)
    return operands


def _joins_a_path(node: ast.Call) -> bool:
    """Whether ``node`` is ``os.path.join(...)`` or ``Path(...)`` over
    positional parts the code spells."""
    return (
        bool(node.args)
        and not any(isinstance(argument, ast.Starred) for argument in node.args)
        and (_dotted(node.func) in _PATH_JOINS or _called_name(node.func) in _PATH_TYPES)
    )


def _joins_a_sequence(node: ast.Call) -> bool:
    """Whether ``node`` is ``sep.join([...])`` or ``sep.join((...))`` over
    parts the code spells."""
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and _dotted(node.func) not in _PATH_JOINS
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], (ast.List, ast.Tuple))
        and not any(isinstance(element, ast.Starred) for element in node.args[0].elts)
    )


def _parameters(arguments: ast.arguments) -> list[str]:
    parameters = [*arguments.posonlyargs, *arguments.args, arguments.vararg, *arguments.kwonlyargs, arguments.kwarg]
    return [parameter.arg for parameter in parameters if parameter is not None]


def _root_name(node: ast.expr) -> str | None:
    """The name an item or attribute chain starts from: ``df`` for
    ``df["qty"]`` or ``df.loc[0].qty``."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _ends_like_a_sentence(statement: str) -> bool:
    """A printed label ends in sentence punctuation and a statement does not.
    A statement can end in a bare ``?`` parameter marker, which never follows
    a word the way a question's ``?`` does."""
    if statement.endswith("?"):
        return statement[-2:-1].isalnum()
    return statement.endswith((":", ".", "!"))


def _get_or_post(node: ast.Call) -> str:
    """urllib's method: POST when the call carries data, else GET."""
    return "GET" if _argument(node, position=1, keyword="data") is None else "POST"


def _is_string_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))


def _environment_variable(node: ast.expr) -> str | None:
    """The variable ``os.environ["KEY"]``, ``os.environ.get("KEY")`` or
    ``os.getenv("KEY")`` reads: how a cell reaches a saved connection."""
    if isinstance(node, ast.Subscript):
        key = node.slice
        reads_environment = _dotted(node.value) in ("os.environ", "environ")
    elif isinstance(node, ast.Call) and node.args:
        key = node.args[0]
        reads_environment = _dotted(node.func) in ("os.environ.get", "environ.get", "os.getenv", "getenv")
    else:
        return None
    if reads_environment and isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _names_a_url(template: str) -> bool:
    """Whether a URL template reads as one: it starts with http(s)://, or with
    a placeholder for a base URL such as ``{base}/inventory``,
    ``{DS_ERP_URL}``, ``{api_url}`` or ``{urls[i]}``. A placeholder's name
    counts by its whole words, so ``{database}`` or ``{hostname}`` is not
    one."""
    if URL_START.match(template):
        return True
    placeholder = _LEADING_PLACEHOLDER.match(template)
    if placeholder is None:
        return False
    name = placeholder.group(1)
    return (
        template[placeholder.end():].startswith("/")
        or name.startswith("DS_")
        or any(part.lower() in _URL_WORDS for part in _IDENTIFIER_PART.findall(name))
    )


def _argument(node: ast.Call, *, position: int, keyword: str) -> ast.expr | None:
    for item in node.keywords:
        if item.arg == keyword:
            return item.value
    if position < len(node.args) and not isinstance(node.args[position], ast.Starred):
        return node.args[position]
    return None


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for a chain of names and attributes, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _first_unique[T](items: Iterable[T], *, limit: int) -> list[T]:
    unique: list[T] = []
    for item in items:
        if item not in unique:
            unique.append(item)
            if len(unique) == limit:
                break
    return unique
