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

from pydantic import BaseModel, Field


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


class FigureSource(BaseModel):
    """One step whose output holds a figure."""

    tool_use_id: str
    kind: SourceKind
    label: str
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
_DOCUMENT = re.compile(
    r"""['"]([^'"\n]*?\.(?:csv|tsv|xlsx|xls|pdf|txt|json|md|docx|parquet))['"]""",
    re.IGNORECASE,
)
# A cell that reaches connected data: anton's data helper, SQL, a saved
# connection's DS_ variables, or a database cursor.
_QUERY = re.compile(
    r"query_minds_data\(|\bselect\b[\s\S]+?\bfrom\b|\bDS_[A-Z0-9_]+|\bread_sql|\.execute\(",
    re.IGNORECASE,
)
_WEB_CALL = re.compile(r"\bweb_(?:search|fetch)\(")


def find_figure_sources(*, answer: str, question: str, steps: Sequence[TurnStep]) -> list[Figure]:
    """Return every figure in ``answer`` with its origin, in answer order."""
    figures = _numbers(answer, skips=_skipped_spans(answer))
    if not figures:
        return []
    described_steps = [(step, _describe(step)) for step in steps]
    described_steps = [(step, described) for step, described in described_steps if described is not None]
    question_values = [n.value for n in _numbers(question)]
    result: list[Figure] = []
    for figure in figures:
        sources = [
            _source(step, described, figure)
            for step, described in described_steps
            if _holds(described, figure)
        ]
        if sources:
            origin = FigureOrigin.step
        elif any(_same_value(value, figure) for value in question_values):
            origin = FigureOrigin.question
        else:
            origin = FigureOrigin.none
        result.append(Figure(
            start=_utf16_offset(answer, figure.start),
            end=_utf16_offset(answer, figure.end),
            text=answer[figure.start:figure.end],
            origin=origin,
            sources=sources,
        ))
    return result


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
    kind: SourceKind
    label: str
    output: str
    output_numbers: tuple[_Number, ...]
    input_values: frozenset[Decimal]
    document_has_pages: bool


def _describe(step: TurnStep) -> _DescribedStep | None:
    """The step's kind, label and output, or None for a scratchpad action other
    than running a cell, whose output repeats earlier cells or none."""
    tool_input = _parse_object(step.tool_input) or {}
    if step.tool_name == "scratchpad":
        if step.action != "exec":
            return None
        code = str(tool_input.get("code") or "")
        cell = _parse_object(step.content)
        output = str(cell.get("stdout") or "") if cell is not None else step.content
        description = str(cell.get("description") or "") if cell is not None else ""
        kind, label = _classify_cell(code, output, description)
        literals = code
    else:
        output = step.content
        kind, label = _classify_tool(step.tool_name, tool_input, output)
        literals = step.tool_input
    return _DescribedStep(
        kind=kind,
        label=label,
        output=output,
        output_numbers=tuple(_numbers(output) + _numbers(output, pattern=_PLAIN_NUMBER)),
        input_values=frozenset(n.value for n in _numbers(literals)),
        document_has_pages=kind is SourceKind.file and label.lower().endswith(".pdf"),
    )


def _classify_cell(code: str, output: str, description: str) -> tuple[SourceKind, str]:
    if _WEB_CALL.search(code):
        url = _URL.search(output) or _URL.search(code)
        return SourceKind.web, url.group(0) if url else description or "a web search"
    if _QUERY.search(code):
        return SourceKind.query, description or "a query"
    document = _DOCUMENT.search(code)
    if document:
        return SourceKind.file, posixpath.basename(document.group(1))
    url = _URL.search(code)
    if url:
        return SourceKind.web, url.group(0)
    return SourceKind.step, description or "a calculation"


def _classify_tool(name: str, tool_input: dict[str, Any], output: str) -> tuple[SourceKind, str]:
    if name in ("web_search", "web_fetch"):
        found = _URL.search(output)
        url = tool_input.get("url") or (found.group(0) if found else "")
        return SourceKind.web, str(url or tool_input.get("query") or name)
    if "recall" in name:
        return SourceKind.memory, "memory"
    path = tool_input.get("path") or tool_input.get("file_path")
    if isinstance(path, str) and path:
        return SourceKind.file, posixpath.basename(path)
    return SourceKind.step, name


def _holds(step: _DescribedStep, figure: _Number) -> bool:
    if figure.value in step.input_values:
        return False
    return any(_same_value(number.value, figure) for number in step.output_numbers)


def _source(step: TurnStep, described: _DescribedStep, figure: _Number) -> FigureSource:
    page = None
    if described.document_has_pages:
        page = _page_of(described, figure)
    return FigureSource(tool_use_id=step.tool_use_id, kind=described.kind, label=described.label, page=page)


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
