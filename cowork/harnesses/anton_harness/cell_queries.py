"""The SQL a scratchpad cell runs, the HTTP requests it makes and the files it
writes, read from its code without running it.

The code is parsed with ``ast`` and nothing in it is evaluated. A string the
code spells is rendered from what it states: an f-string keeps each
``{expr}`` as written, except a bare name bound to a query, which reads as
that query, and a bare name bound to a saved connection's lookup, which reads
``{KEY}``; ``+`` joins its parts, ``text(...)``, ``dedent(...)`` and
``"...".format(...)`` give back the string they wrap, an environment lookup
such as ``os.environ["DS_ERP_URL"]`` renders as ``{DS_ERP_URL}``, and any
other part renders as ``{expr}``. A name bound to a string earlier in the
cell, or in an earlier cell of the turn (a scratchpad keeps one namespace
across cells), stands for that string. Comments never reach the parsed tree,
and a bare string statement (a docstring) is skipped. Code inside an
f-string's ``{...}`` or a ``+`` operand still counts: a call there is a call
the cell makes.

Every string a cell spells or names is a SQL candidate, so the statement a
``query_minds_data`` call runs (its first argument on the desktop, its second
in the cloud), or an ``.execute``, ``read_sql`` or ``.sql`` call, is one
whichever argument carries it. A string bound to a name is a candidate where
the code reads that name on its own, not where it binds it, so a base query
that ``+``, ``+=`` or an f-string extends is not one. A candidate counts when,
past its comments, it starts with SELECT or a WITH ... AS ( common table
expression, reads FROM, and does not end like a sentence: that drops
``df.query('qty > 5')``, DDL, writes and a printed label such as
``"With rows from stock.csv:"``.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

MAX_QUERIES = 3
MAX_REQUESTS = 3
# The characters a cell's `+`, `+=` and f-string query substitutions may
# render in all. Past it they render as nothing, so code that doubles a
# string on every line reads in bounded time and memory.
MAX_RENDERED_CHARS = 1_000_000
# The longest string a name stays bound to. Names carry into the turn's later
# cells, whose budgets start afresh, so this bounds what one name brings in.
MAX_STRING_CHARS = 65_536

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})
# Modules whose request functions make an HTTP call whatever their URL reads.
_HTTP_MODULES = frozenset({"requests", "httpx"})
_WEB_TOOLS = frozenset({"web_search", "web_fetch"})
_UNWRAPPERS = frozenset({"text", "dedent"})
# Calls that write the file their first argument names, with that argument's
# keyword.
_FILE_WRITERS = {
    "to_csv": "path_or_buf",
    "to_json": "path_or_buf",
    "to_excel": "excel_writer",
    "to_parquet": "path",
    "to_markdown": "buf",
    "to_html": "buf",
    "savefig": "fname",
}
# Path methods that write the file their Path(...) receiver names.
_PATH_WRITERS = frozenset({"write_text", "write_bytes"})
# An open() mode holding one of these writes the file.
_WRITE_MODES = frozenset("wax")

# An unclosed /* comments out the rest of the text, as in SQL; ending the
# match there keeps the scan linear.
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?(?:\*/|\Z)", re.DOTALL)
_SQL_SELECT = re.compile(r"select\b", re.IGNORECASE)
# WITH only opens a query as a CTE: a name (bare, quoted or a placeholder),
# optional column list, then AS (.
_SQL_CTE = re.compile(
    r"with\s+(?:recursive\s+)?(?:\w+|\"[^\"]+\"|`[^`]+`|\{[^{}]*\})(?:\s*\([^)]*\))?\s+as\s*\(",
    re.IGNORECASE,
)
_SQL_FROM = re.compile(r"\bfrom\b", re.IGNORECASE)

_URL_START = re.compile(r"https?://", re.IGNORECASE)
# A URL's user and password sit between :// and @, whatever comes before the
# ://, a placeholder such as {proto} included.
_USERINFO = re.compile(r"(://)[^/?#\s]*@")
_LEADING_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
# The words of an identifier, split on _, ., brackets, quotes and camelCase.
_IDENTIFIER_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_URL_WORDS = frozenset({"url", "uri", "endpoint", "base", "host"})
_CONNECTION_PLACEHOLDER = re.compile(r"\{DS_[A-Z0-9_]+\}")
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
    # The strings the code names as a file it writes (a pandas or matplotlib
    # save, a Path's write_text or write_bytes, or an open() in a write,
    # append or create mode), in code order: a file it wrote is not one it read.
    written_files: tuple[str, ...] = ()
    # The names bound to a string once the cell has run, for the next cell.
    names: Mapping[str, str] = field(default_factory=dict)


def read_cell(code: str, *, names: Mapping[str, str]) -> CellReading:
    """Read ``code`` with ``names`` (the strings earlier cells bound) in scope.
    Code that does not parse never ran, so it reads as nothing."""
    try:
        reader = _CellReader(code, names)
        reader.visit(ast.parse(code))
    except (SyntaxError, ValueError, RecursionError):
        return CellReading(names=dict(names))
    candidates = dict.fromkeys(text for index, text in enumerate(reader.strings) if index not in reader.bound)
    queries = (textwrap.dedent(text).strip() for text in candidates if is_sql(text))
    return CellReading(
        queries=tuple(_first_unique(queries, limit=MAX_QUERIES)),
        requests=tuple(_first_unique(reader.requests, limit=MAX_REQUESTS)),
        strings=tuple(dict.fromkeys(reader.strings)),
        calls_query_minds_data=reader.calls_query_minds_data,
        calls_web_tool=reader.calls_web_tool,
        environment_variables=tuple(dict.fromkeys(reader.environment_variables)),
        written_files=tuple(dict.fromkeys(reader.written_files)),
        names=reader.names,
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


class _CellReader(ast.NodeVisitor):
    """Walks a cell in code order, binding each name assigned a string before
    the code after it reads that name."""

    def __init__(self, code: str, names: Mapping[str, str]) -> None:
        # The cell as UTF-8 and the byte offset each line starts at: a node's
        # col_offset counts bytes, and slicing here costs only the node's size.
        self._code = code.encode()
        self._line_starts = [0, *(match.end() for match in _LINE_END.finditer(self._code))]
        self.names: dict[str, str] = dict(names)
        self.strings: list[str] = []
        # Indexes into strings of the ones a name was bound to: such a string
        # runs where the code reads the name on its own, not where it binds it.
        self.bound: set[int] = set()
        self.requests: list[HttpRequest] = []
        self.calls_query_minds_data = False
        self.calls_web_tool = False
        self.environment_variables: list[str] = []
        self.written_files: list[str] = []
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
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind(target.id, self._string(node.value))
            else:
                self.visit(target)
        if all(isinstance(target, ast.Name) and target.id in self.names for target in node.targets):
            self._mark_bound(first, self.names[node.targets[0].id])

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        first = len(self.strings)
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id, self._string(node.value))
            if node.target.id in self.names:
                self._mark_bound(first, self.names[node.target.id])

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if not isinstance(node.target, ast.Name):
            return
        name = node.target.id
        joined = None
        if isinstance(node.op, ast.Add) and name in self.names:
            joined = self._joined(self.names[name], self._text(node.value))
        self._bind(name, joined)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.names:
                self.strings.append(self.names[node.id])
        else:
            # Rebound by a loop, an unpacking or a `with ... as`: no longer
            # the string it was.
            self.names.pop(node.id, None)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.append(node.value)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.strings.append(self._text(node))
        self._visit_placeholders(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        rendered = self._string(node)
        if rendered is None:
            self.generic_visit(node)
        else:
            self.strings.append(rendered)
            self._visit_placeholders(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._environment_variable(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._environment_variable(node)
        name = _called_name(node.func)
        if name == "query_minds_data":
            self.calls_query_minds_data = True
        elif name in _WEB_TOOLS:
            self.calls_web_tool = True
        request = self._request(node)
        if request is not None:
            self.requests.append(request)
        written = self._written_file(node)
        if written is not None:
            self.written_files.append(written)
        self.generic_visit(node)

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
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            self._visit_placeholders(node.left)
            self._visit_placeholders(node.right)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            # "... %s" % value renders only its template; the value is code the cell runs.
            self._visit_placeholders(node.left)
            self.visit(node.right)
        elif self._string(node) is None:
            self.visit(node)

    def _environment_variable(self, node: ast.expr) -> str | None:
        """The variable ``node`` reads from the environment, noted as one the
        cell reads."""
        variable = _environment_variable(node)
        if variable is not None:
            self.environment_variables.append(variable)
        return variable

    def _bind(self, name: str, rendered: str | None) -> None:
        """Bind ``name`` to ``rendered``, or unbind it when the code gave it no
        string or one longer than MAX_STRING_CHARS."""
        if rendered is None or len(rendered) > MAX_STRING_CHARS:
            self.names.pop(name, None)
        else:
            self.names[name] = rendered

    def _mark_bound(self, index: int, rendered: str) -> None:
        """Note ``strings[index]`` as the string a name was just bound to, when
        it is: it runs where the code reads the name on its own. A string from
        code inside the value, such as a call in an f-string's placeholder,
        comes after it and still runs here."""
        if index < len(self.strings) and self.strings[index] == rendered:
            self.bound.add(index)

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
            return self.names.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._string(node.left), self._string(node.right)
            if left is None and right is None:
                return None
            return self._joined(
                left if left is not None else self._placeholder(node.left),
                right if right is not None else self._placeholder(node.right),
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            # "... %s" % value keeps its template.
            return self._string(node.left)
        if isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name in _UNWRAPPERS and node.args:
                return self._string(node.args[0])
            if name == "format" and isinstance(node.func, ast.Attribute):
                return self._string(node.func.value)
        return None

    def _joined(self, left: str, right: str) -> str | None:
        """``left + right``, or None once the cell's renders would pass
        MAX_RENDERED_CHARS in all."""
        return left + right if self._spend(len(left) + len(right)) else None

    def _spend(self, chars: int) -> bool:
        """Count ``chars`` against the cell's MAX_RENDERED_CHARS; False once
        they pass it."""
        self._rendered_chars += chars
        return self._rendered_chars <= MAX_RENDERED_CHARS

    def _fstring_part(self, part: ast.expr) -> str:
        """A literal part as it reads, and a placeholder as written, except an
        environment lookup, or a bare name bound to a saved connection's
        lookup, which reads ``{KEY}`` here as it does elsewhere, and a bare
        name bound to a query, which reads as that query."""
        if isinstance(part, ast.Constant):
            return str(part.value)
        if not isinstance(part, ast.FormattedValue):
            return self._source(part)
        variable = self._environment_variable(part.value)
        if variable is not None:
            return "{" + variable + "}"
        if isinstance(part.value, ast.Name) and part.conversion == -1 and part.format_spec is None:
            bound = self.names.get(part.value.id)
            if bound is not None and (
                _CONNECTION_PLACEHOLDER.fullmatch(bound) or (self._is_query(bound) and self._spend(len(bound)))
            ):
                return bound
        return self._source(part)

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

    def _request(self, node: ast.Call) -> HttpRequest | None:
        """The request ``node`` makes, from its method and URL alone: a call's
        headers, auth, body, params and cookies never enter it."""
        func = node.func
        name = _called_name(func)
        is_method = isinstance(func, ast.Attribute)
        receiver = _dotted(func.value) if is_method else None
        if is_method and name in _HTTP_METHODS:
            url, method = _argument(node, 0, "url"), name.upper()
        elif is_method and name == "request":
            url, method = _argument(node, 1, "url"), self._method(_argument(node, 0, "method"))
        elif name == "urlopen":
            url, method = _argument(node, 0, "url"), _get_or_post(node)
            # urlopen(Request(...)): the Request call is the request.
            if isinstance(url, ast.Call):
                return None
        elif name == "Request" and receiver in (None, "request", "urllib.request"):
            url = _argument(node, 0, "url")
            method_argument = _argument(node, 5, "method")
            method = _get_or_post(node) if method_argument is None else self._method(method_argument)
        else:
            return None
        if url is None:
            return None
        template = strip_url(self._text(url))
        if not template or not (receiver in _HTTP_MODULES or _names_a_url(template)):
            return None
        return HttpRequest(method=method, url=template)

    def _method(self, node: ast.expr | None) -> str:
        """The method a call names, upper-cased, or its source as a
        placeholder when the code doesn't spell it."""
        if node is None:
            return "GET"
        rendered = self._string(node)
        return rendered.upper() if rendered is not None else self._placeholder(node)

    def _written_file(self, node: ast.Call) -> str | None:
        """The file ``node`` writes: a pandas or matplotlib save, a Path's
        write_text or write_bytes, or an open() in a write, append or create
        mode."""
        name = _called_name(node.func)
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        # Path("x").write_text(...) or Path("x").open("w"): the receiver names the file.
        path = (
            receiver.args[0]
            if isinstance(receiver, ast.Call) and _called_name(receiver.func) == "Path" and receiver.args
            else None
        )
        if name in _FILE_WRITERS:
            target = _argument(node, 0, _FILE_WRITERS[name])
        elif name in _PATH_WRITERS:
            target = path
        elif name == "open" and path is not None:
            # A Path's open takes its mode first.
            target = path if self._writes(_argument(node, 0, "mode")) else None
        elif name == "open":
            target = _argument(node, 0, "file") if self._writes(_argument(node, 1, "mode")) else None
        else:
            target = None
        return self._string(target) if target is not None else None

    def _writes(self, mode: ast.expr | None) -> bool:
        """Whether an open() mode writes its file: it holds w, a or x."""
        rendered = self._string(mode) if mode is not None else None
        return rendered is not None and bool(_WRITE_MODES & set(rendered))


def _ends_like_a_sentence(statement: str) -> bool:
    """A printed label ends in sentence punctuation and a statement does not.
    A statement can end in a bare ``?`` parameter marker, which never follows
    a word the way a question's ``?`` does."""
    if statement.endswith("?"):
        return statement[-2:-1].isalnum()
    return statement.endswith((":", ".", "!"))


def _get_or_post(node: ast.Call) -> str:
    """urllib's method: POST when the call carries data, else GET."""
    return "GET" if _argument(node, 1, "data") is None else "POST"


def _is_string_literal(node: ast.expr) -> bool:
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
    ``{DS_ERP_URL}`` or ``{api_url}``. A placeholder's name counts by its
    whole words, so ``{database}`` or ``{hostname}`` is not one."""
    if _URL_START.match(template):
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


def _argument(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
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
