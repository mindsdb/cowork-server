"""Tie each figure in a finished answer to the steps whose output holds it.

The formatter hands this module the final answer, the question and every tool
result the turn produced. Each figure in the answer (a number that isn't part
of an identifier, a date, a list marker, a link or code) gets one origin:

- ``step``: one or more steps' outputs hold it, at the precision the answer
  shows it, so a rounded or percent form of a step's value still matches;
- ``question``: no step holds it, but the question itself states it;
- ``none``: nothing in the turn holds it, so the renderer marks it as a figure
  to check before acting on it.

A step whose own input states the figure (a cell that prints a literal it was
given, or a memorize call) is not a source for it: its output only echoes what
the model wrote. A figure the model works out in its own text, without a step
printing it, has no source.

Each step a figure names is described once: its kind, a short label, the
project file it read, the web page it reached, and the SQL and HTTP requests
its code shows (``cell_queries``). Everything in that description passes
through ``_redacted`` before it leaves this module.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from cowork.coding.redaction import redact_text
from cowork.common.history_scrub import scrub_credentials
from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.harnesses.anton_harness.cell_queries import CellReading, read_cell, strip_url

MAX_QUERY_CHARS = 4_000
MAX_REQUEST_CHARS = 500
# The most document strings of one step looked up on disk.
MAX_FILE_LITERALS = 3


class FigureOrigin(StrEnum):
    step = "step"
    question = "question"
    none = "none"


class SourceKind(StrEnum):
    query = "query"
    file = "file"
    web = "web"
    memory = "memory"
    step = "step"


class FigureStep(BaseModel):
    """A step at least one figure in the answer names."""

    tool_use_id: str
    kind: SourceKind
    label: str
    # The project-relative POSIX path of a file the step read, when that file
    # is inside the conversation's project and exists when the turn ends.
    path: str | None = None
    # The http(s) URL of the web page the step reached, without its user,
    # password, query string or fragment.
    url: str | None = None
    # The SQL statements and "METHOD url-template" requests the step's code
    # shows, in code order, redacted.
    queries: list[str] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)


class FigureSource(BaseModel):
    """A step whose output holds a figure, named by its entry in
    ``FigureSources.steps``."""

    tool_use_id: str
    page: int | None = None


class Figure(BaseModel):
    """A figure in the answer and where it came from.

    ``start`` and ``end`` count UTF-16 code units into the answer text, the
    unit the renderer's JavaScript strings index by, so ``text`` is
    ``content.slice(start, end)`` there.
    """

    start: int
    end: int
    text: str
    origin: FigureOrigin
    sources: list[FigureSource] = Field(default_factory=list)


class FigureSources(BaseModel):
    """The figures in an answer and, once each, the steps they name."""

    steps: list[FigureStep] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)


@dataclass(frozen=True)
class TurnStep:
    """One tool result of the turn, as the formatter streamed it."""

    tool_use_id: str
    tool_name: str
    action: str
    tool_input: str
    content: str


@dataclass(frozen=True)
class _Number:
    value: Decimal
    # Half a unit of the last digit shown, in the number's own scale: what a
    # value may differ by and still round to the number as written.
    tolerance: Decimal
    percent: bool
    start: int
    end: int


# A number as an answer or an output writes it: an optional currency sign,
# digits with optional thousands separators and decimals (or an exponent, as
# pandas prints large floats), then an optional percent or scale word.
_NUMBER = re.compile(
    r"(?P<currency>[$€£])?"
    r"(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>%|\s?(?:k|K|M|B|bn|thousand|million|billion)\b)?"
)

# A step's output can print raw CSV, where "S1,250" holds 250 and not 1,250, so
# outputs are also read without thousands separators.
_PLAIN_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_SCALES = {
    "k": Decimal(1_000), "K": Decimal(1_000), "thousand": Decimal(1_000),
    "M": Decimal(1_000_000), "million": Decimal(1_000_000),
    "B": Decimal(1_000_000_000), "bn": Decimal(1_000_000_000), "billion": Decimal(1_000_000_000),
}

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"

# Spans of the answer whose numbers are not figures.
_ANSWER_SKIPS = (
    re.compile(r"```.*?(?:```|\Z)", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"\]\([^)\s]*\)"),
    re.compile(r"https?://\S+"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?\b", re.IGNORECASE),
    re.compile(r"^[ \t]*(?:#{1,6}[ \t]+)?\d+[.)][ \t]", re.MULTILINE),
)

_PAGE = re.compile(r"\bpage\s*[:#]?\s*(\d+)", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s'\"<>)\]]+")
_HTTP_URL = re.compile(r"https?://", re.IGNORECASE)
# A string that names a document: a path, without a scheme, ending in one of
# the extensions a cell reads data from.
_DOCUMENT = re.compile(
    r"(?!\w+://)[^'\"\n]*\.(?:csv|tsv|xlsx|xls|pdf|txt|json|md|docx|parquet)", re.IGNORECASE,
)
# A saved connection's variable, which a request template reads its base URL
# or credentials from.
_DS_VARIABLE = re.compile(r"\bDS_[A-Z0-9_]+")
# A URL's user and password, whatever comes before the ://. A quote ends the
# match, so it never runs past a string literal in SQL.
_USERINFO = re.compile(r"(://)[^\s/?#'\"]*@")
# A scratchpad action that clears the cell namespace the next cell reads.
_NAMESPACE_RESETS = frozenset({"reset", "remove"})


def find_figure_sources(
    *, answer: str, question: str, steps: Sequence[TurnStep], files: AgentFiles | None = None,
) -> FigureSources:
    """Every figure in ``answer`` with its origin, in answer order, and the
    steps they name in step order. ``files`` finds the project file a step
    read; without it no step carries a path."""
    numbers = _numbers(answer, skips=_skipped_spans(answer))
    if not numbers:
        return FigureSources()
    described = _describe_steps(steps)
    question_values = [n.value for n in _numbers(question)]
    named: set[int] = set()
    figures: list[Figure] = []
    for number in numbers:
        holders = [index for index, step in enumerate(described) if _holds(step, number)]
        named.update(holders)
        if holders:
            origin = FigureOrigin.step
        elif any(_same_value(value, number) for value in question_values):
            origin = FigureOrigin.question
        else:
            origin = FigureOrigin.none
        figures.append(Figure(
            start=_utf16_offset(answer, number.start),
            end=_utf16_offset(answer, number.end),
            text=answer[number.start:number.end],
            origin=origin,
            sources=[_source(described[index], number) for index in holders],
        ))
    return FigureSources(
        steps=[_figure_step(described[index], files=files) for index in sorted(named)],
        figures=figures,
    )


def question_text(content: str | list[Any]) -> str:
    """The text parts of a question as the responses route receives it."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") in ("input_text", "text"):
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


@dataclass(frozen=True)
class _DescribedStep:
    tool_use_id: str
    kind: SourceKind
    label: str
    url: str | None
    # The first MAX_FILE_LITERALS strings that name a document the step read
    # (not one it wrote), in code order; the first that resolves into the
    # project gives the step its path.
    file_literals: tuple[str, ...]
    queries: tuple[str, ...]
    requests: tuple[str, ...]
    output: str
    output_numbers: tuple[_Number, ...]
    input_values: frozenset[Decimal]
    document_has_pages: bool


def _describe_steps(steps: Sequence[TurnStep]) -> list[_DescribedStep]:
    """Each step that can be a source, in step order. A scratchpad action
    other than running a cell is not one: its output repeats earlier cells or
    is empty. Each named scratchpad keeps the strings its cells bind, so a
    cell can run a query an earlier cell spelled."""
    names_by_pad: dict[str, dict[str, str]] = {}
    described: list[_DescribedStep] = []
    for step in steps:
        tool_input = _parse_object(step.tool_input) or {}
        if step.tool_name != "scratchpad":
            described.append(_describe_tool(step, tool_input))
            continue
        pad = str(tool_input.get("name") or "")
        if step.action in _NAMESPACE_RESETS:
            names_by_pad.pop(pad, None)
        if step.action != "exec":
            continue
        code = str(tool_input.get("code") or "")
        reading = read_cell(code, names=names_by_pad.get(pad, {}))
        names_by_pad[pad] = dict(reading.names)
        described.append(_describe_cell(step, code=code, reading=reading))
    return described


def _describe_cell(step: TurnStep, *, code: str, reading: CellReading) -> _DescribedStep:
    """A cell's kind, in order: a web tool call; a query (SQL, the data
    helper, a saved connection's variable, or a request that names one); a
    document it reads (a file it writes is not one); a request to, or a
    string holding, a literal http(s) URL; else a step."""
    cell = _parse_object(step.content)
    output = str(cell.get("stdout") or "") if cell is not None else step.content
    description = str(cell.get("description") or "") if cell is not None else ""
    written = set(reading.written_files)
    documents = tuple(
        text for text in dict.fromkeys(reading.strings) if text not in written and _DOCUMENT.fullmatch(text)
    )[:MAX_FILE_LITERALS]
    web_page = next(
        (url for url in (*(r.url for r in reading.requests), *reading.strings) if _HTTP_URL.match(url)), None,
    )
    url = None
    if reading.calls_web_tool:
        found = _URL.search(output)
        url = _web_url(found.group(0) if found else web_page)
        kind, label = SourceKind.web, _web_label(url) if url else description or "a web search"
    elif (
        reading.queries
        or reading.calls_query_minds_data
        or any(_DS_VARIABLE.fullmatch(name) for name in reading.environment_variables)
        or any(_DS_VARIABLE.search(request.url) for request in reading.requests)
    ):
        kind, label = SourceKind.query, description or "a query"
    elif documents:
        kind, label = SourceKind.file, _basename(documents[0])
    elif web_page is not None:
        url = _web_url(web_page)
        kind, label = SourceKind.web, _web_label(web_page)
    else:
        kind, label = SourceKind.step, description or "a calculation"
    return _described(
        step,
        kind=kind,
        label=label,
        url=url,
        file_literals=documents,
        queries=reading.queries,
        requests=tuple(str(request) for request in reading.requests),
        output=output,
        literals=code,
    )


def _describe_tool(step: TurnStep, tool_input: dict[str, Any]) -> _DescribedStep:
    output = step.content
    url = None
    file_literals: tuple[str, ...] = ()
    if step.tool_name in ("web_search", "web_fetch"):
        found = _URL.search(output)
        url = _web_url(str(tool_input.get("url") or "") or (found.group(0) if found else None))
        kind = SourceKind.web
        label = _web_label(url) if url else str(tool_input.get("query") or step.tool_name)
    elif "recall" in step.tool_name:
        kind, label = SourceKind.memory, "memory"
    else:
        path = tool_input.get("path") or tool_input.get("file_path")
        if isinstance(path, str) and path:
            kind, label, file_literals = SourceKind.file, _basename(path), (path,)
        else:
            kind, label = SourceKind.step, step.tool_name
    return _described(
        step,
        kind=kind,
        label=label,
        url=url,
        file_literals=file_literals,
        queries=(),
        requests=(),
        output=output,
        literals=step.tool_input,
    )


def _described(
    step: TurnStep,
    *,
    kind: SourceKind,
    label: str,
    url: str | None,
    file_literals: tuple[str, ...],
    queries: tuple[str, ...],
    requests: tuple[str, ...],
    output: str,
    literals: str,
) -> _DescribedStep:
    return _DescribedStep(
        tool_use_id=step.tool_use_id,
        kind=kind,
        label=label,
        url=url,
        file_literals=file_literals,
        queries=queries,
        requests=requests,
        output=output,
        output_numbers=tuple(_numbers(output) + _numbers(output, pattern=_PLAIN_NUMBER)),
        input_values=frozenset(n.value for n in _numbers(literals)),
        document_has_pages=kind is SourceKind.file and label.lower().endswith(".pdf"),
    )


def _figure_step(step: _DescribedStep, *, files: AgentFiles | None) -> FigureStep:
    """The step as the event lists it. Its path is looked up only here, once
    a figure names it, since that reads the disk."""
    label, path = step.label, None
    if files is not None:
        for literal in step.file_literals:
            path = files.project_path(literal)
            if path is not None:
                if step.kind is SourceKind.file:
                    label = _basename(literal)
                break
    return FigureStep(
        tool_use_id=step.tool_use_id,
        kind=step.kind,
        label=_redacted(label),
        path=path,
        url=_redacted(step.url) if step.url else None,
        queries=[_capped(_redacted(query), MAX_QUERY_CHARS) for query in step.queries],
        requests=[_capped(_redacted(request), MAX_REQUEST_CHARS) for request in step.requests],
    )


def _redacted(text: str) -> str:
    """``text`` without credentials: a URL's user and password, the values of
    saved connections' secret fields and known key shapes
    (``scrub_credentials``), and the value of any key=value pair whose key
    names a secret (``redact_text``)."""
    return redact_text(scrub_credentials(_USERINFO.sub(r"\1***@", text)))


def _capped(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _web_url(url: str | None) -> str | None:
    """``url`` as a link the event can carry: http(s), with no user, password,
    query string or fragment, and no placeholder left in it."""
    if not url or not _HTTP_URL.match(url):
        return None
    stripped = strip_url(url)
    return None if "{" in stripped else stripped


def _web_label(url: str) -> str:
    """A web address shown as host and path, without its query string."""
    stripped = strip_url(url)
    try:
        parts = urlsplit(stripped)
    except ValueError:
        return stripped
    return (parts.netloc + parts.path).rstrip("/") or stripped


def _basename(path: str) -> str:
    return posixpath.basename(path.replace("\\", "/"))


def _holds(step: _DescribedStep, figure: _Number) -> bool:
    if figure.value in step.input_values:
        return False
    return any(_same_value(number.value, figure) for number in step.output_numbers)


def _source(step: _DescribedStep, figure: _Number) -> FigureSource:
    page = _page_of(step, figure) if step.document_has_pages else None
    return FigureSource(tool_use_id=step.tool_use_id, page=page)


def _page_of(step: _DescribedStep, figure: _Number) -> int | None:
    """The last page marker before the figure's first place in the output."""
    for number in step.output_numbers:
        if _same_value(number.value, figure):
            pages = _PAGE.findall(step.output, 0, number.start)
            return int(pages[-1]) if pages else None
    return None


def _same_value(value: Decimal, figure: _Number) -> bool:
    """Whether a value rounds to the figure as the answer writes it. A percent
    figure also takes the fraction it stands for (0.125 for 12.5%)."""
    if abs(abs(value) - figure.value) <= figure.tolerance:
        return True
    return figure.percent and abs(abs(value) * 100 - figure.value) <= figure.tolerance


def _numbers(
    text: str, *, skips: Iterable[tuple[int, int]] = (), pattern: re.Pattern[str] = _NUMBER,
) -> list[_Number]:
    skipped = sorted(skips)
    numbers = []
    for match in pattern.finditer(text):
        start, end = match.span()
        if not _standalone(text, start, end) or _inside(skipped, start, end):
            continue
        number = _parse(match)
        if number is not None:
            numbers.append(number)
    return numbers


def _standalone(text: str, start: int, end: int) -> bool:
    """A number not glued to an identifier: not A2210 or W3001, not Q-03, not
    32GB, and not one part of a version such as 1.2.3."""
    before = text[start - 1] if start > 0 else ""
    if before.isalnum() or before in ("_", "."):
        return False
    if before == "-" and start > 1 and text[start - 2].isalnum():
        return False
    after = text[end] if end < len(text) else ""
    if after.isalnum() or after == "_":
        return False
    return not (after == "." and end + 1 < len(text) and text[end + 1].isdigit())


def _parse(match: re.Match[str]) -> _Number | None:
    groups = match.groupdict()
    digits = (groups.get("digits") or match.group(0)).replace(",", "")
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    suffix = (groups.get("suffix") or "").strip()
    scale = _SCALES.get(suffix, Decimal(1))
    exponent = value.as_tuple().exponent
    decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return _Number(
        value=abs(value) * scale,
        tolerance=Decimal(5).scaleb(-(decimals + 1)) * scale,
        percent=suffix == "%",
        start=match.start(),
        end=match.end(),
    )


def _skipped_spans(answer: str) -> list[tuple[int, int]]:
    return [match.span() for pattern in _ANSWER_SKIPS for match in pattern.finditer(answer)]


def _inside(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _parse_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _utf16_offset(text: str, index: int) -> int:
    """Python indexes code points and JavaScript UTF-16 code units; they part
    at the first character outside the Basic Multilingual Plane, an emoji."""
    return len(text[:index].encode("utf-16-le")) // 2
