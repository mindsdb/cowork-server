"""Tie each figure in a finished answer to the steps whose output holds it.

The formatter hands this module the final answer, the question and the tool
calls the model saw with their results: this turn's, and those of the earlier
turns the model is seeded with. Each figure in the answer (a number that isn't
part of an identifier, a date, a list marker, a citation marker, a link, code,
display math or anton's sentence naming the limit that stopped the turn) gets
one origin:

- ``step``: one or more of this turn's steps' outputs hold it, at the
  precision the answer shows it, so a rounded or percent form of a step's
  value still matches, and a hedged round number ("about 1,200") matches to
  its second significant digit. A cell that holds it also names the cells that
  read the data its value was computed from, traced through the names a
  scratchpad keeps across cells and turns, so such a cell may be an earlier
  turn's. A figure names at most ``MAX_FIGURE_SOURCES`` steps, the ones whose
  output holds it first. A step with no tool_use_id holds a figure but is
  never named, nor are the cells it read data from: the client opens a step
  by that id;
- ``question``: no step of this turn holds it, but the question writes it in
  the same form (value, sign, currency, scale and percent);
- ``earlier_answer``: neither a step of this turn nor the question gives it,
  but one of the conversation's last answers wrote it in the same form with
  an origin other than none (``mark_repeated_figures``). It names no source;
- ``none``: nothing in the turn holds it and no earlier answer wrote it in
  the same form, so the renderer marks it as a figure to check before acting
  on it.

A step of an earlier turn never holds a figure. Its cells are read only for
the names, functions and data their scratchpad passes on to this turn's cells.
A figure that no step of this turn holds and the question doesn't write is
marked earlier_answer when one of the conversation's last
``cowork.services.conversations.MAX_EARLIER_ANSWERS`` answers wrote it in the
same form with an origin other than none. Nothing of that answer is copied,
so the figure names no step. The form is the value (exactly, not to a
rounding), sign, currency sign, scale, percent and a unit written right after
the digits (3x, 40kg, 12h). A unit after a space ("12 hours", "40 kg") and an
ISO currency code ("4,600 EUR") are not part of it, so "12 hours" repeats "12
days"; the same unit spelled two ways ("12h", "12hrs") is not the same form.
An answer whose saved events hold no figure_sources event (one with no
figures, or a turn that stopped or failed) gives no figure and still takes
one of those places. The turn's producer reads the earlier answers when the
turn starts.

Only the last ``MAX_EARLIER_CELLS`` earlier-turn cells are read, and the
formatter reads the history only when the answer has figures
(``has_figures``) and this turn ran a cell (``ran_cell``). These limits never
change a figure's origin. They can only drop an earlier cell a figure names
as the one that read its data, or a string or function a cell of this turn
reuses.

Only the scratchpad's cells and the calls of the tools ``is_evidence_tool``
counts are steps: the web tools and read_text_file
(``_NATIVE_EVIDENCE_TOOLS``), and the connector tools that return the
customer's records. A connector tool is an MCP tool anton registers under its
engine's name, and anton's ``returns_records`` says which ones return
records, such as CRM objects or campaign analytics. A connector tool that
returns docs, schema or account details is no step, nor is a write tool:
some write tools can also view records, but a tool's name can't tell a view
from a write, so none counts. Nor is a tool of an engine anton has not
classified. Every other tool hands back the agent's own text (an earlier
answer, an artifact it wrote, a memory), the user's answer, or a status
line. A read_text_file of a file in anton's memory folder (``.anton/memory``)
is no step either: it reads back what the agent memorized. Nor is a cell
whose only source would be the memory files it, or an earlier cell, read
(``_describe_cell``). The formatter passes on a call of any of these tools
only when its tool_done said ok=True: a failed call's result is the error
the tool or anton wrote, which can quote numbers.

A step whose own input states the figure (a cell that prints a literal it was
given) is not a source for it: its output only echoes what the model wrote.
For the same reason an output value equal to a number its input states is not
read at all. Nor is a number inside an ISO date-time in a connector tool's
records: a record's timestamps are not figures, and the minutes of the
created and last-changed times every record carries would source almost any
small number. A date-time in a connector call's input, such as a filter's,
states no number either, so a record's value equal to its hour, minutes or
seconds is still read. A figure the model works out in its own text, without
a step printing it, has no source.

A figure written in a time unit, glued to its digits (12h), as the word
after it (22 hours, a 13.08-day lead time, 22 business days) or as the
unit of the range it starts (22-25 hours, 22 to 25 hours), is not held by a
step whose output names time units but never the figure's: the 22 in a
lead_time_days column is not the "22 hours" the model worked out. A step
names a unit by its plural, inside identifiers too, or in any form right
after a number (36h, a 30-day window). A singular elsewhere doesn't count,
since it is more often a rate's (cost_per_day) or a date part's (GROUP BY
month). Only the output's units rule a step out, so a window in its code or
label (interval '12 months', "over the last 30 days") doesn't, while a unit
its code or label names keeps the figure. A step whose output names no time
unit may hold the figure. A rate ("$42 per hour", "10 units/day") has no
time unit: the word after its figure names none. The rule misses a step
whose output prints the figure's value while it names only another unit,
such as a cell that turns days into hours (days_left * 24) and prints the
bare hours beside a days column, or prints a plural date part as a header
(months); that figure has no source. It doesn't fire on a days column named
only in the code (SELECT lead_time_days ...) whose output prints a bare row,
nor on a range joined by "and" (between 22 and 25 hours), so such a figure
keeps its step.

Each step a figure names is described once: its kind, a short label, the file
it read, the web page it reached, and the SQL and HTTP requests its code shows
(``cell_queries``) or, for a connector tool, the statement its call ran. Its
label, web page, SQL and requests pass through ``_redacted``, a best-effort
credential sweep, before they leave this module. Its file is named by its
place in the project or by its files-store id, never by the text the code
spelled.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from itertools import accumulate
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

from cowork.common.history_scrub import scrub_credentials
from cowork.common.logger import get_logger
from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.harnesses.anton_harness.cell_queries import (
    URL_START,
    WEB_TOOLS,
    CellReading,
    FunctionReading,
    read_cell,
    strip_url,
)
from cowork.schemas.figure_sources import (
    EarlierAnswer,
    Figure,
    FigureOrigin,
    FigureSource,
    FigureSources,
    FigureStep,
    SourceKind,
)
from cowork.schemas.project_files import CANONICAL_PROJECT_MEMORY_PATHS

if TYPE_CHECKING:
    from anton.core.mcp.names import ToolNameParts

try:
    from anton.core.mcp.access import returns_records
    from anton.core.mcp.names import parse_namespaced_tool_name
except ImportError:
    # An anton older than mindsdb/anton#585 cannot say which connector tools
    # return the customer's records. There no connector tool is evidence
    # (_records_tool).
    returns_records = None
    parse_namespaced_tool_name = None

logger = get_logger(__name__)

MAX_QUERY_CHARS = 4_000
MAX_REQUEST_CHARS = 500
# The most documents a step's label names; it counts the rest.
MAX_LABELED_DOCUMENTS = 3
# The most steps one figure names.
MAX_FIGURE_SOURCES = 3
# The most earlier-turn cells read for the names, functions and data they
# pass to this turn's cells, the latest ones, because each one's code is
# parsed again on every answer.
MAX_EARLIER_CELLS = 50
# The tools anton runs itself whose results are evidence: a web page or
# search result, and a file's text. recall_history and anton's recall hand
# back earlier answers verbatim, the artifact and skill tools the agent's own
# writing, memorize and ask_user what the agent or the user wrote, read_image
# a line about the image rather than what it shows, and the rest a status.
# A read_text_file of a memory file is left out too (reads_memory_file), and
# so is a cell whose only source is memory files it or an earlier cell read
# (_describe_cell). Besides these, a connector tool that returns the
# customer's records is evidence, so a caller asks is_evidence_tool rather
# than reading this set.
_NATIVE_EVIDENCE_TOOLS = WEB_TOOLS | {"read_text_file"}
# The folder anton keeps the project's memory files in, .anton/memory, by its
# path parts in lower case.
_MEMORY_FOLDERS = frozenset(
    tuple(part.casefold() for part in path[:-1]) for path in CANONICAL_PROJECT_MEMORY_PATHS
)


class CellResult(BaseModel):
    """A scratchpad cell's result as anton sends it, json.dumps(asdict(Cell)),
    or as the formatter rebuilds it from a replayed history row. What the cell
    printed is required, so a result without it is logged rather than read as
    a cell that printed nothing."""

    model_config = ConfigDict(extra="ignore")

    stdout: str
    code: str = ""
    description: str = ""


def _text(value: object) -> str:
    """An argument as the matcher reads it: its text, and a null, empty or
    zero value as no text."""
    return str(value or "")


def _path(value: object) -> str | None:
    """A file read's path argument, or None when it is not a non-empty
    string or path object."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return value if isinstance(value, str) and value else None


class ToolArguments(BaseModel):
    """What the matcher reads from a tool call's arguments: the scratchpad
    it runs in and the code it runs, the page a web call fetches or the
    query it searches, the file a read names, and the SQL-shaped statement a
    connector tool runs (HubSpot's query_crm_data takes it as sql). Every
    other argument is left out, and a value of the wrong type reads as text
    (or as no path), so arguments always read."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: Annotated[str, BeforeValidator(_text)] = ""
    code: Annotated[str, BeforeValidator(_text)] = ""
    url: Annotated[str, BeforeValidator(_text)] = ""
    query: Annotated[str, BeforeValidator(_text)] = ""
    path: Annotated[str | None, BeforeValidator(_path)] = None
    sql: Annotated[str, BeforeValidator(_text)] = ""

    @classmethod
    def from_call(cls, *, arguments: Mapping[str, Any]) -> ToolArguments:
        """The arguments a history row holds, already parsed."""
        return cls.model_validate(arguments)

    @classmethod
    def from_text(cls, *, text: str) -> ToolArguments:
        """The arguments as the model streamed them. Arguments with something
        after them still read; arguments that don't read as an object (cut
        short by the cloud pod at its wire limit, or nested too deep to
        parse) keep only their scratchpad's name, and a cell's result gives
        the code that ran."""
        arguments = _leading_object(text)
        if arguments is not None:
            return cls.from_call(arguments=arguments)
        found = _PAD_NAME.search(text)
        name = _json_string(found.group(1)) if found else None
        return cls(name=name or "")


class ToolInput(BaseModel):
    """A tool call's arguments: the text the model wrote, which a step's
    output may only echo, and what the matcher reads from it."""

    model_config = ConfigDict(frozen=True)

    text: str
    arguments: ToolArguments

    @classmethod
    def from_call(cls, *, arguments: Mapping[str, Any]) -> ToolInput:
        """A history row's call, whose arguments are already parsed."""
        return cls(text=json.dumps(arguments, default=str), arguments=ToolArguments.from_call(arguments=arguments))

    @classmethod
    def from_text(cls, *, text: str) -> ToolInput:
        """A streamed call, whose arguments arrive as the model wrote them."""
        return cls(text=text, arguments=ToolArguments.from_text(text=text))


@dataclass(frozen=True)
class TurnStep:
    """One tool call the model made and its result: this turn's, as the
    formatter streamed it or the turn's tool rows hold it, or an earlier
    turn's, as the history the model is seeded with holds it."""

    tool_use_id: str
    tool_name: str
    action: str
    tool_input: ToolInput
    # The result as the model saw it: for a streamed scratchpad cell, anton's
    # JSON of it.
    content: str
    # A scratchpad cell's result already read from a history row, which holds
    # it as labeled text rather than anton's JSON.
    cell: CellResult | None = None
    # A step of an earlier turn of the conversation. It never holds a figure.
    # Its cells are read before this turn's, so the names, functions and data
    # they bound reach this turn's cells, and a cell of this turn computed
    # from data such a cell read names it. Only the last MAX_EARLIER_CELLS
    # such cells are read.
    earlier_turn: bool = False


@dataclass(frozen=True)
class _Number:
    # The number's size; its sign is ``negative``.
    value: Decimal
    # Half a unit of the last digit shown, in the number's own scale: what a
    # value may differ by and still round to the number as written.
    tolerance: Decimal
    percent: bool
    negative: bool
    # The currency sign and scale as written, so a figure can be compared
    # with a number in the question in the same form.
    currency: str
    scale: Decimal
    start: int
    end: int
    # The unit written right after the digits (kg, lbs, hrs, h, d, pcs) or
    # the multiplier x, lowercased, with × read as x; empty for a percent, a
    # scale or no suffix. A unit after a space is not read.
    unit: str = ""


# A minus sign, not a hyphen: nothing word-like and no decimal point just
# before it.
_SIGN = r"(?P<sign>(?<![\w.])[-\u2212])?"

# A number as an answer or an output writes it: an optional sign and currency
# sign (with a country prefix, as in C$), digits with optional thousands
# separators and decimals (or an exponent, as pandas prints large floats),
# then an optional percent, scale, multiplier or unit. A scale letter after a
# space must end the word, so "250 B-grade" is not 250 billion; a lowercase m
# or b counts only after a currency sign (``_parse`` drops "250m").
_NUMBER = re.compile(
    _SIGN
    + r"(?P<currency>[A-Z]{0,3}[$€£])?"
    r"(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>"
    r"[ \u00a0]?(?:%|(?i:percent|pct)(?!\w))"
    r"|[ \u00a0]?(?i:thousand|million|billion)(?!\w)"
    r"|(?:MM|mln|mn|bn|[kKMBmb])(?!\w)"
    r"|[ \u00a0](?:MM|mln|mn|bn|[kKMB])(?![-\w])"
    r"|[x\u00d7](?!\w)"
    r"|(?:kg|lbs?|hrs?|pcs|[hd])(?!\w)"
    r")?"
)

# A step's output can print raw CSV, where "S1,250" holds 250 and not 1,250, so
# outputs are also read without thousands separators (``_output_numbers``).
_PLAIN_NUMBER = re.compile(_SIGN + r"(?P<digits>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

# Scale suffixes, lowercased.
_SCALES = {
    "k": Decimal(1_000), "thousand": Decimal(1_000),
    "m": Decimal(1_000_000), "mm": Decimal(1_000_000), "mn": Decimal(1_000_000), "mln": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "b": Decimal(1_000_000_000), "bn": Decimal(1_000_000_000), "billion": Decimal(1_000_000_000),
}
_PERCENTS = frozenset({"%", "percent", "pct"})

# The largest power of ten a number may have: past any float (1.8e308), and
# far enough inside decimal's range that no arithmetic on it overflows.
_MAX_EXPONENT = 400

# A word or sign just before a whole number that says it is rounded.
_HEDGE = re.compile(
    r"(?:\b(?:about|around|roughly|approximately|approx\.?|nearly|almost|some|circa)|[~≈])[ \t]*\Z",
    re.IGNORECASE,
)


class _TimeUnit(StrEnum):
    """A unit a duration is written in."""

    second = "second"
    minute = "minute"
    hour = "hour"
    day = "day"
    week = "week"
    month = "month"
    year = "year"


# The time units a figure can be written in, by the words and short forms
# that name each one: glued to its digits (12h) or the word after it.
_TIME_UNITS = {
    "sec": _TimeUnit.second, "secs": _TimeUnit.second, "second": _TimeUnit.second, "seconds": _TimeUnit.second,
    "min": _TimeUnit.minute, "mins": _TimeUnit.minute, "minute": _TimeUnit.minute, "minutes": _TimeUnit.minute,
    "h": _TimeUnit.hour, "hr": _TimeUnit.hour, "hrs": _TimeUnit.hour, "hour": _TimeUnit.hour, "hours": _TimeUnit.hour,
    "d": _TimeUnit.day, "day": _TimeUnit.day, "days": _TimeUnit.day,
    "wk": _TimeUnit.week, "wks": _TimeUnit.week, "week": _TimeUnit.week, "weeks": _TimeUnit.week,
    "month": _TimeUnit.month, "months": _TimeUnit.month,
    "yr": _TimeUnit.year, "yrs": _TimeUnit.year, "year": _TimeUnit.year, "years": _TimeUnit.year,
}
# The words that name a time unit anywhere in a step's text: the plurals a
# duration's name takes (lead_time_days, hours=21.98). A singular counts only
# right after a number (a 30-day window); elsewhere it is more often a rate's
# (holding_cost_per_day) or a date part's (GROUP BY month).
_STEP_TIME_WORDS = frozenset({
    "secs", "seconds", "mins", "minutes", "hrs", "hours", "days", "wks", "weeks", "months", "yrs", "years",
})
# The word just after a figure in prose, past emphasis, one space or hyphen
# and a kind of day or hour: "22 hours", "**22** hours", "13.08-day", "22
# business days". A single letter there names no unit ("3 D printers", "22
# H-beams"); a glued one (22h) is read as the number's own unit.
_WORD_AFTER = re.compile(
    r"[*_]{0,2}[ \u00a0-]?[*_]{0,2}(?:(?i:business|working|work|calendar)[ \u00a0-])?(?P<word>[A-Za-z]{2,})"
)
# What joins a range's two numbers when the unit follows only the second:
# "22-25 hours", "22 to 25 hours", "22 or 23 days".
_RANGE_JOIN = re.compile(r"[*_]{0,2}[ \u00a0]*(?:[-\u2013]|to|or)[ \u00a0]*[*_]{0,2}")
# A word of a step's text, with identifiers split at underscores, digits and
# case changes, so lead_time_days, leadTimeDays and LEAD_TIME_DAYS each end
# in the word days.
_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")

# A month name, spelled out or cut short, so "Maybe" and "decreased" are not
# months. May counts only capitalized: "12 may arrive" is not a date.
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|(?-i:May|MAY)|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_YEAR = r"(?:19|20)\d{2}"

# Spans of the answer, or of the question, whose numbers are not figures.
_ANSWER_SKIPS = (
    re.compile(r"```.*?(?:```|\Z)", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    # Display math: $$...$$, \[...\] and \(...\). A single $ is a currency
    # sign, not math. Like a code fence, \[ or \( left open runs to the end.
    re.compile(r"\$\$.+?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?(?:\\\]|\Z)", re.DOTALL),
    re.compile(r"\\\(.*?(?:\\\)|\Z)", re.DOTALL),
    # A Markdown link's target. A ] ends it too, so a run of ]( is read once.
    re.compile(r"\]\([^)\s\]]*+\)"),
    re.compile(r"https?://\S+"),
    # Hyphenated digits that are an identifier, not a range: a part number,
    # a phone number or a SKU, with three or more parts (100-200-300) or a
    # second part with a leading zero (4711-002, 555-0100).
    re.compile(r"(?<![\w.-])\d++(?:(?:-\d++){2,}+|-0\d++)(?![\w-])"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    # 10am, 10:30 pm, 3 p.m.
    re.compile(r"\b\d{1,2}(?::\d{2}){0,2}[ \t]?[ap]\.?m\b\.?", re.IGNORECASE),
    # 10/14/2026, 10-14-2026, 2026/10/14; two parts ("12/15 sites") stay.
    re.compile(r"\b(?:\d{1,2}([/-])\d{1,2}\1\d{2,4}|\d{4}/\d{1,2}/\d{1,2})\b"),
    re.compile(rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTH},?\s+{_YEAR}\b", re.IGNORECASE),
    # Q3 2026, H1 '26, FY 2026, 2026 Q3.
    re.compile(rf"\b(?:[QH][1-4]|FY)[ \t]*(?:{_YEAR}|'\d{{2}})\b|\b{_YEAR}[ \t]+[QH][1-4]\b"),
    # Citation markers, [1] and [^1], but not a link whose text is a number.
    re.compile(r"\[\^?\d{1,3}\](?!\()"),
    re.compile(r"^[ \t]*(?:#{1,6}[ \t]+)?\d+[.)][ \t]", re.MULTILINE),
    # The sentence anton opens a hand-back with when a turn reaches its round
    # cap or spend ceiling ("This task reached its limit of 25 steps per
    # request", anton core.session ChatSession._limit_reached). Its number is
    # anton's setting, not a figure the answer reports. A step count has no
    # thousands separators; a token count has them.
    re.compile(r"\breached its limit of \d[\d,]*+ (?:steps|tokens) per request\b"),
)

# A line that marks a page of a document's text: a header such as
# "--- page 2 ---", or a footer such as "Page 1 of 2", which closes its page.
_PAGE = re.compile(
    r"^[ \t\-=#*\[(]*page[ \t]*[:#]?[ \t]*(?P<page>\d+)"
    r"(?:[ \t]*(?:of|/)[ \t]*(?P<pages>\d+))?[ \t\r\-=*\])\.:]*$",
    re.IGNORECASE | re.MULTILINE,
)

# A printed DataFrame or Series row: an integer row label (or a describe()
# percentile) and two or more spaces before the values. pandas leaves a
# positive value a blank where a minus sign goes, so before a negative first
# value a frame row has one space.
_ROW_LABEL = re.compile(r"\d+%?(?= {2,}\S| -\d)")
# One space and a negative value after a row label. print(year, change)
# writes the same line by hand, so it is a row only under a frame's header.
_NEGATIVE_FIRST = re.compile(r" -\d")
# A frame's column header, indented past the row labels.
_FRAME_HEADER = re.compile(r"[ \t]+\S")
_SERIES_FOOTER = re.compile(r"(?:Name|Length|Freq|dtype): ")
# A MultiIndex row whose outer keys pandas leaves blank, since the row above
# shows them: indented and holding a value.
_INNER_ROW = re.compile(r"[ \t]+\S.*\d")
# Where a field of a printed row starts: a character after a space or a tab.
_FIELD_START = re.compile(r"(?<=[ \t])\S")
# The row pandas prints where it cuts a long frame or Series: ".." under a
# Series' labels, "...  ..  .." across a frame.
_TRUNCATED = re.compile(r"[ \t]*\.\.\.?(?:[ \t]+\.\.\.?)*[ \t]*")
# The gap between a MultiIndex Series' level names.
_LEVEL_GAP = re.compile(r" {2,}")
# Lines and fields that state a frame's or a Series' size.
_SHAPE = re.compile(r"^(?:Shape: .*|\[\d+ rows x \d+ columns\])$|\bLength: \d+", re.MULTILINE)
# A number that is only digits and thousands separators.
_GROUPED = re.compile(r"[-\u2212]?\d{1,3}(?:,\d{3})+")
_LETTER = re.compile(r"[^\W\d_]")
# An underscore between digits, as in the literal 1_200_000.
_DIGIT_UNDERSCORE = re.compile(r"(?<=\d)_(?=\d)")
# A URL in prose. A quote, a backtick or an angle bracket ends it, and the
# punctuation a sentence puts after it is trimmed (``_found_url``).
_URL = re.compile(r"https?://[^\s'\"<>`]+")
# What ends a sentence or markup around a URL rather than the URL itself.
_URL_TRAILERS = frozenset(".,;:!?*_~")
# A string that names a document: a path, without a scheme, ending in one of
# the extensions a cell reads data from.
_DOCUMENT = re.compile(
    r"(?!\w+://)[^'\"\n]*\.(?:csv|tsv|xlsx|xls|pdf|txt|json|md|docx|parquet)", re.IGNORECASE,
)
# A part of a string the code fills in when the cell runs, which the cell
# reader writes as that code in braces (data/{f} for os.path.join("data", f),
# data/{f.name} for Path("data") / f.name) or leaves as empty braces
# ("data/{}".format(name)), or a glob's * (glob.glob("data/*")).
_RUN_TIME_PART = re.compile(r"\{[^{}]*\}|\*")
# A saved connection's variable, which a request template reads its base URL
# or credentials from.
_DS_VARIABLE = re.compile(r"\bDS_[A-Z0-9_]+")
# A scratchpad action that clears the cell namespace the next cell reads.
_NAMESPACE_RESETS = frozenset({"reset", "remove"})
# How the result anton streams for a scratchpad call it refused to run begins
# (anton core.session._discarded_tool_call_events): the call was cut off, or
# went down with its round, and nothing ran. History rows mark such a result
# is_error instead, and the formatter leaves those calls out.
_DISCARDED_CALL = "[error] exec failed"
# A scratchpad call's name argument, found in arguments too long to parse
# whole: the cloud pod cuts them at its wire limit.
_PAD_NAME = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
# Where the cloud pod cut a result past its wire limit
# (anton.cloud_turn.__main__._TRUNCATION_MARKER).
_CLIP_MARKER = "\n[… truncated …]\n"
# How a cell result's printed output begins in anton's JSON.
_STDOUT_KEY = '"stdout": "'
# The line read_text_file puts over the lines it read, "<path> — lines 1-120
# of 900" with notes after it, or "<path> — empty file"; and the mark it
# leaves in a long line it shortened.
_TEXT_RANGE_HEADER = re.compile(r"\A[^\n]* — (?:lines \d+-\d+ of \d+[^\n]*|empty file)(?:\n|\Z)")
_SHORTENED_LINE = re.compile(r"\[line truncated: \d+ chars\]")
# An ISO date-time, as a connector stamps each record with when it was
# created and last changed: 2026-03-04T09:12:44.512Z, 2026-03-04 09:12+05:00.
_ISO_DATETIME = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?")

_REDACTED = "[redacted]"
# The rules below bound how far each part of a match may run, or let a match
# that has started consume all it read, so that text a cell spells (up to
# cell_queries.MAX_STRING_CHARS) is redacted in time that grows with its
# length, whatever it holds. Each rule's comment says why it is linear.
#
# One character of a URL's user name, and of its query string. Without the
# enclosing SQL dialect a quote in a user name may be part of it: doubled as
# SQL writes it (o''brien), escaped with a backslash as MySQL and a Postgres
# E'' string write it (o\'brien), or the quote that does not close the
# literal around it (o"brien inside '...'). A lone quote joins a user name
# only before a letter, a digit or an underscore, which the quote that
# closes a literal rarely has after it. So a user name does not run on into
# the next literal or JSON member ('http://h','token=a@b'), where it would
# take that key's name and leave the part of its value after the @
# unmasked. In a query string a doubled quote counts as one character. In
# both, the branches start with different characters, or a lone quote is
# never followed by the quote a doubled one starts with, so a run of them
# reads one way only.
_USER_CHAR = r"(?:[^\s/?#'\"@:]|''|\"\"|['\"](?=\w))"
_QUERY_CHAR = r"(?:[^\s'\"]|''|\"\")"
# A URL's user and password, whatever comes before the ://. Once a : has ended
# the user name, the password may hold /, ? or #, as a database URI's may; a
# quote may be password data under the enclosing SQL dialect. Keep it in
# the bounded scan; ambiguous detail is safer to redact than to expose. A
# user, with a password or alone, may hold a quote. A user before a password
# may also hold an @ (an email login, or a user@server login), which
# SQLAlchemy reads up to its first : and the password up to the next @. Such
# a user's password runs to an @ before any /, ?, #, ;, comma, & or quote.
# Past one, a user, a host and a port before a path, a query string, a
# parameter or the next literal that holds an @
# (mysql://app@db:3306/s?password=P@ss, 'mysql://app@db:3306/s','pw=P@ss')
# would read as one user and password, and the password's part after its @
# would show; urllib reads the user as app there, and the key rules mask the
# value. A user with no @ is tried first, so its password still runs to the
# last @. Linear: a match starts only at ://, and each attempt reads at most
# 256 characters of user and 1,024 characters of password. The user holds no
# :, so it ends at a : in one place only.
_USERINFO = re.compile(
    rf"(://)(?:{_USER_CHAR}{{0,256}}:\S{{0,1024}}|(?:{_USER_CHAR}|@){{0,256}}:[^\s/?#;,&'\"]{{0,1024}}"
    rf"|{_USER_CHAR}{{0,256}})@"
)
# On overflow, mask the whole non-whitespace span, including its host and
# any quotes. Without the enclosing SQL dialect a quote may be password
# data, even when not doubled or escaped. Omit excess detail rather than
# leave part of that credential visible. Linear: the password run is
# possessive, so an attempt that fails read fewer than 1,025 password
# characters, and one that succeeds consumes all it read.
_LONG_USERINFO = re.compile(rf"(://)(?:{_USER_CHAR}|@){{0,256}}:\S{{1025,}}+")
# A user over 256 characters overflows the same way, once a : or an @ ends it.
# A short password may hold a :// and a long run after it, so this rule runs
# after _USERINFO has masked the user and password around that run. Linear:
# the user run holds no : or /, so the runs after two :// never overlap.
_LONG_USER = re.compile(rf"(://){_USER_CHAR}{{257,}}+[:@]\S*+")
# An Oracle EZConnect string, user/password@host[:port][/service], whose user
# starts where a word starts, so a URL's path is never read as one. Oracle
# lets a user or a password be double-quoted, and then it may hold any
# character but a lone " (a quoted user, as a quoted name, holds at most 128).
# A proxy login names the user it connects as in brackets after the user,
# double-quoted or not (proxy[scott]/pw@host, proxy["Scott Lee"]/pw@host).
# The host may be a name, an address, an IPv6 address in brackets, a
# connect descriptor in parentheses (pw@(DESCRIPTION=...)) or a placeholder
# the code fills in ({host}, %(host)s, $HOST). Linear: an
# unquoted user is at most 128 characters of one word. A quoted user starts
# at a quote no quote comes right before, reads at most 128 units and gives
# none back, so it ends at the first lone quote and at most one reaches each
# /. The bracketed name, quoted or not, reads at most 128 units and gives
# none back, and each [ tries its two forms once. An unquoted password holds
# no /, so each / starts at most one. A quoted password ends at a run of
# quotes of odd length, while one that runs on needs an even run, so of the
# quoted passwords that reach a character, at most one runs on. An unquoted
# password that holds a secret key's name is masked as a key's value
# (``_masked_ezconnect``).
_EZCONNECT = re.compile(
    r"(?<![\w$#./:-])(?:[A-Za-z_][\w$#]{0,127}|(?<!\")\"(?:[^\"\n]|\"\"){1,128}+\")"
    r"(?:\[(?:\"(?:[^\"\n]|\"\"){1,128}+\"|[^\]\s/@]{1,128}+)\])?"
    r"/(?:\"(?:[^\"\n]|\"\")*+\"|(?P<bare>(?:[^\s/@'\"]|''|\"\")++))@(?=[\w./(\[{$%-])"
)
# Hosts whose URL path is itself a credential: Slack (and GovSlack), Discord,
# Microsoft Teams and Zapier webhooks, IFTTT's Maker key, and a Telegram
# bot's API, whose path holds the bot's token. What follows the part that
# names the service is masked.
_SECRET_URL_PATH = re.compile(
    r"(?<![\w.-])(?P<service>hooks\.slack(?:-gov)?\.com/|hooks\.zapier\.com/|maker\.ifttt\.com/"
    r"|(?:[\w-]{1,63}\.)?discord(?:app)?\.com/api/(?:v\d+/)?webhooks/"
    r"|[\w-]{1,63}\.webhook\.office\.com/|outlook\.office(?:365)?\.com/webhook/|api\.telegram\.org/(?:file/)?bot)"
    r"[^\s'\"<>`]++",
    re.IGNORECASE,
)
# A URL's query string and fragment, which can carry a key or a signature (a
# presigned S3 or Azure URL), after a path of any length. Linear: both parts
# are possessive or optional, so a match never fails once it has found a ://,
# and it consumes every character it read.
_QUERY_STRING = re.compile(rf"(://(?:[^\s'\"?#]|''|\"\")*+)(?P<query>[?#]{_QUERY_CHAR}*+)?")
# A key that names a secret, as a word of its own or a part of one, starting
# where a word starts: password, Pwd, s3_secret_access_key, X-Api-Key,
# private_key, passphrase, AccountKey, X-Amz-Signature, SharedAccessSignature.
# "pass" and "sig" count only as the key's last part (pass, db_pass, sig, as
# an Azure SAS writes it), so pass_rate, passenger and sigma are not secret
# keys.
_SENSITIVE_NAME = (
    r"(?<![\w.-])[\w.-]{0,64}(?:password|passwd|passphrase|pwd|secret|token|api[_-]?key|access[_-]?key"
    r"|account[_-]?key|private[_-]?key|authorization|credential|cookie|signature"
    r"|(?<![a-z0-9])(?:pass|sig)(?![\w.-]))[\w.-]{0,64}+"
)
# What joins a key to its value: =, :, or => as Ruby and PHP write it. Once
# => matched, the = is never read alone, which would take the > as a value.
_ASSIGN = r"[ \t]*(?>=>|[:=])[ \t]*"
# A value in quotes. A doubled quote inside the value is a quote, as SQL
# writes it. The quotes stay, so the SQL around the value still reads.
# Multiline values retain one boundary; an unclosed value is redacted through
# the end of the input. Linear: possessive, and it never fails once its
# opening quote matched.
_QUOTED_VALUE = (
    r"(?:(?P<single>')(?:''|\\[\s\S]|[^'\\])*+(?P<single_close>')?"
    r"|(?P<double>\")(?:\"\"|\\[\s\S]|[^\"\\])*+(?P<double_close>\")?)"
)
# A value's opening quote, followed by the value's first character rather
# than a space, a comma or a ). The quote that closes a literal ending in a
# keyword and a space ('Your password ' || name) is followed by one of
# those, so it is not read as the opening quote of a value that would run on
# to the next quote or the end of the text.
_VALUE_START = r"(?=['\"](?:['\"]|[^\s,)]))"
# A string's introducer before a value's opening quote: Postgres's E'', a
# national N'' string or MySQL's _charset'' (_utf8mb4'x'). It stays with the
# key. An unquoted value that starts with _ before the quote that closes its
# literal (pwd=_x') is no introducer when a space, a comma or a ) follows
# that quote.
_INTRODUCER = rf"(?:(?:[EeNn]|_\w{{1,32}}+){_VALUE_START})?"
# A secret key's value in quotes after its key, the key quoted or not, with
# or without a string's introducer: password='a b c', "token": "x",
# 'Authorization': 'Bearer x', PASSWORD=N'a b'.
_QUOTED_SECRET = re.compile(
    rf"(?P<key>(?<![\w.-])[\"']?{_SENSITIVE_NAME}[\"']?{_ASSIGN}{_INTRODUCER}){_QUOTED_VALUE}",
    re.IGNORECASE,
)
# The rules below mask a password a SQL statement sets with a keyword rather
# than a key=value pair, which a split the readings miss can list after a
# query (cell_queries._statements) and a string literal can hold.
#
# A quoted value after a secret key's name and whitespace, the key quoted or
# not (Postgres's CREATE ROLE ... PASSWORD 'x', DuckDB's CREATE SECRET (...,
# SECRET 'x', SESSION_TOKEN 'y'), a foreign server's OPTIONS (password 'x')),
# or after PASSWORD FOR <user> = (MySQL's SET PASSWORD FOR app = 'x').
# Linear: a key starts where a word starts and reads at most 64 characters
# each side of its secret word, every other run in the key is possessive or
# bounded, and the value never fails once its opening quote matched.
_KEYWORD_SECRET = re.compile(
    rf"(?P<key>(?:\bpassword\s++for\s++[^\s=]{{1,256}}+\s*+=\s*+|(?<![\w.-])[\"']?{_SENSITIVE_NAME}[\"']?\s++)"
    rf"{_INTRODUCER}){_VALUE_START}{_QUOTED_VALUE}",
    re.IGNORECASE,
)
# A MySQL authentication plugin's name: bare, quoted or in backticks.
_PLUGIN = r"(?:[\w$]{1,64}+|'[^'\n]{1,64}+'|\"[^\"\n]{1,64}+\"|`[^`\n]{1,64}+`)"
# The value after IDENTIFIED BY, quoted or not: Oracle's IDENTIFIED BY x and
# IDENTIFIED BY VALUES 'hash', and MySQL's IDENTIFIED BY 'x', IDENTIFIED BY
# PASSWORD 'hash' and IDENTIFIED WITH <plugin> BY 'x' or AS 'hash'. PASSWORD
# and VALUES stay with the key, and RANDOM PASSWORD, which sets a password
# the text does not hold, stays as written. An unquoted value starts after
# whitespace. Linear: the keyword starts where a word starts, every run in
# the key is possessive or bounded, and the value never fails once its first
# character matched.
_IDENTIFIED_BY = re.compile(
    rf"(?P<key>\bidentified(?:\s++with\s++{_PLUGIN}\s++(?:by|as)|\s++by)\b(?:\s++(?:password|values)\b)?\s*+"
    rf"{_INTRODUCER})(?!random\s++password\b)(?:{_VALUE_START}{_QUOTED_VALUE}|(?<=\s)(?P<bare>[^\s,;'\"()]++))",
    re.IGNORECASE,
)
# The current password MySQL's ALTER USER and SET PASSWORD and Oracle's ALTER
# USER name after REPLACE, once the new password before it is masked
# (IDENTIFIED BY 'new' REPLACE 'current') or after MySQL's RANDOM PASSWORD.
# REPLACE counts only right after the mask or RANDOM PASSWORD, so a REPLACE
# in prose or in a query keeps its words. Linear: a match starts only at the
# mask or at RANDOM, and the value never fails once its first character
# matched.
_REPLACED_SECRET = re.compile(
    rf"(?P<key>(?:{re.escape(_REDACTED)}['\"]?|\brandom\s++password)\s++replace\s++{_INTRODUCER})"
    rf"(?:{_VALUE_START}{_QUOTED_VALUE}|(?P<bare>[^\s,;'\"()]++))",
    re.IGNORECASE,
)
# A secret key's unquoted value, up to a space, a comma, a semicolon, an
# ampersand, or a quote that closes the literal around it. An HTTP auth
# scheme before the value (Basic, Bearer, Token, any word followed by a
# token-shaped value) is redacted with it. An ODBC value in braces may hold
# any of those, so a value that opens with { reads to the } that closes it
# (}} is a brace inside it), or to the end of the line. Linear: the braced
# part is possessive and never fails once its { matched.
_BARE_SECRET = re.compile(
    rf"(?P<key>{_SENSITIVE_NAME}{_ASSIGN})"
    r"(?:[A-Za-z][\w-]{0,31}[ \t]+(?=[A-Za-z0-9._~+/=-]{8}))?"
    r"(?:\{(?:\}\}|[^}\r\n])*+\}?|[^\s,;&'\"])(?:[^\s,;&'\"]|['\"](?=[^\s,;&)\]}'\"]))*+",
    re.IGNORECASE,
)
# A secret key's name and what joins it to a value, inside an unquoted
# EZConnect password. Linear: it fails at once where no word starts, and a
# word start reads at most 64 characters each side of a secret word, inside
# the word that starts there.
_KEY_IN_PASSWORD = re.compile(rf"{_SENSITIVE_NAME}{_ASSIGN}", re.IGNORECASE)
# A bearer token no key names.
_BEARER = re.compile(r"\b(?P<scheme>bearer[ \t]+)[A-Za-z0-9._~+/=-]{8,}+", re.IGNORECASE)
# Credential shapes no key names: AWS access keys; GitHub, Slack (bot, user
# and app-level), Stripe secret and restricted, Hugging Face and Google OAuth
# access tokens; JWTs and PEM private keys. A JWT starts where a word starts,
# so a long run of its prefix is read in one pass.
_SECRET_SHAPE = re.compile(
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}"
    r"|\bgithub_pat_\w{20,}"
    r"|\bxox[abposre]-[A-Za-z0-9-]{8,}"
    r"|\bxapp-\d+-[A-Za-z0-9-]{8,}"
    r"|\b[rs]k_(?:live|test)_[A-Za-z0-9]{8,}"
    r"|\bhf_[A-Za-z0-9]{20,}"
    r"|\bya29\.[\w-]{20,}"
    r"|(?<![\w-])eyJ[\w-]{4,}+\.eyJ[\w-]{4,}+\.[\w-]*+"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)

_JSON = json.JSONDecoder()


def has_figures(answer: str) -> bool:
    """Whether ``answer`` holds a figure ``find_figure_sources`` would list."""
    return bool(_numbers(answer, skips=_skipped_spans(answer), prose=True))


def ran_cell(step: TurnStep) -> bool:
    """Whether the step is a scratchpad cell anton ran: an exec call it did
    not discard."""
    return step.tool_name == "scratchpad" and step.action == "exec" and not step.content.startswith(_DISCARDED_CALL)


def find_figure_sources(
    *, answer: str, question: str, steps: Sequence[TurnStep], files: AgentFiles | None = None,
) -> FigureSources:
    """Every figure in ``answer`` with its origin, in answer order, and the
    steps they name, earlier turns' first and each in step order. Only a step
    of this turn holds a figure; an earlier turn's cell is named only as one
    that read the data a holder's value was computed from. ``files`` finds
    the file a step read and names the turn's project; without it no step
    carries a path or an attachment."""
    numbers = _numbers(answer, skips=_skipped_spans(answer), prose=True)
    if not numbers:
        return FigureSources()
    described = _describe_steps(steps)
    # An earlier turn's step has no output values (``_described``), so only
    # this turn's steps hold a figure.
    outputs = [_output_values(step.output_numbers) for step in described]
    index_of: dict[str, int] = {}
    for index, step in enumerate(described):
        if step.tool_use_id:
            index_of.setdefault(step.tool_use_id, index)
    asked = _numbers(question, skips=_skipped_spans(question), prose=True)
    named: set[int] = set()
    figures: list[Figure] = []
    for position, number in enumerate(numbers):
        next_number = numbers[position + 1] if position + 1 < len(numbers) else None
        time_unit = _time_unit(answer, number=number, next_number=next_number)
        held = _holders(described, outputs=outputs, figure=number, time_unit=time_unit)
        holders = _named_holders(described, outputs=outputs, figure=number, holders=held)
        readers = _readers(described, holders=holders, index_of=index_of)
        named.update(holders, readers)
        if held:
            origin = FigureOrigin.step
        elif any(_same_form(value, figure=number) for value in asked):
            origin = FigureOrigin.question
        else:
            origin = FigureOrigin.none
        figures.append(Figure(
            start=_utf16_offset(answer, index=number.start),
            end=_utf16_offset(answer, index=number.end),
            text=answer[number.start:number.end],
            origin=origin,
            sources=[
                *(_source(described[index], figure=number) for index in holders),
                *(FigureSource(tool_use_id=described[index].tool_use_id) for index in readers),
            ],
        ))
    return FigureSources(
        project_id=files.project_id if files is not None else None,
        steps=[_figure_step(described[index], files=files) for index in sorted(named)],
        figures=figures,
    )


@dataclass(frozen=True)
class _WrittenForm:
    """How a number is written, apart from how its digits are grouped: its
    value, sign, currency, scale and percent, and the unit written right
    after the digits. An earlier answer's figure must match all of them.
    The question's rule (``_same_form``) leaves the unit out: a question
    match only spares a figure the no-source mark, while an earlier answer's
    match claims the figure was given before, so "3x" must not stand for
    "3"."""

    value: Decimal
    negative: bool
    currency: str
    scale: Decimal
    percent: bool
    unit: str

    @classmethod
    def of(cls, *, number: _Number) -> _WrittenForm:
        """The form ``number`` is written in."""
        return cls(
            value=number.value, negative=number.negative, currency=number.currency, scale=number.scale,
            percent=number.percent, unit=number.unit,
        )


def mark_repeated_figures(*, sources: FigureSources, earlier_answers: Sequence[EarlierAnswer]) -> FigureSources:
    """``sources`` with each figure of origin none that an earlier answer
    wrote in the same form (``_WrittenForm``), with an origin other than
    none, marked earlier_answer. The value must be equal, not only round to
    it. A step or question figure keeps its origin, and nothing of the
    earlier answer is copied: a marked figure keeps its empty sources, and
    the steps stay this turn's."""
    if not any(figure.origin is FigureOrigin.none for figure in sources.figures):
        return sources
    given = {
        form
        for answer in earlier_answers
        for figure in answer.figures
        if figure.origin is not FigureOrigin.none and (form := _written_form(text=figure.text)) is not None
    }
    return sources.model_copy(update={"figures": [
        figure.model_copy(update={"origin": FigureOrigin.earlier_answer})
        if figure.origin is FigureOrigin.none and _written_form(text=figure.text) in given
        else figure
        for figure in sources.figures
    ]})


def _written_form(*, text: str) -> _WrittenForm | None:
    """How a listed figure is written, read from its text alone, or None when
    the text is not one number from end to end."""
    numbers = _numbers(text, prose=True)
    if len(numbers) != 1 or (numbers[0].start, numbers[0].end) != (0, len(text)):
        return None
    return _WrittenForm.of(number=numbers[0])


@dataclass(frozen=True)
class _DescribedStep:
    tool_use_id: str
    kind: SourceKind
    label: str
    url: str | None
    # The documents the step read (not the ones it wrote or only printed), in
    # code order. The step's file is looked up only when there is one.
    documents: tuple[str, ...]
    queries: tuple[str, ...]
    requests: tuple[str, ...]
    output: str
    # The values a figure can match; none for an earlier turn's step.
    output_numbers: tuple[_Number, ...]
    input_values: frozenset[Decimal]
    document_has_pages: bool
    # The time units its output names, the only ones that rule out a figure
    # in another unit, and those its output, input and label name, any of
    # which keeps a figure in that unit (``_time_units_named``); none for an
    # earlier turn's step.
    output_time_units: frozenset[_TimeUnit]
    time_units: frozenset[_TimeUnit]
    # The cells, by tool_use_id, that read the data this cell's code draws on
    # through names earlier cells bound.
    drawn_from: frozenset[str] = frozenset()


@dataclass
class _Pad:
    """What one scratchpad's namespace holds after the cells read so far: the
    strings and functions its names are bound to, and for each name, the
    cells that read the data its value was computed from."""

    names: dict[str, str] = field(default_factory=dict)
    functions: dict[str, FunctionReading] = field(default_factory=dict)
    clients: dict[str, str] = field(default_factory=dict)
    lineage: dict[str, frozenset[str]] = field(default_factory=dict)

    def drawn_from(self, names: Iterable[str]) -> frozenset[str]:
        """The cells the values of ``names`` were computed from."""
        return frozenset().union(*(self.lineage.get(name, frozenset()) for name in names))

    def record(self, reading: CellReading, *, tool_use_id: str) -> None:
        """Take in a cell that ran: each name it bound now comes from the
        cells its inputs came from, and from the cell itself when the value
        comes from data the cell read."""
        lineage = {}
        for name, binding in reading.bindings.items():
            cells = self.drawn_from(binding.inputs)
            lineage[name] = cells | {tool_use_id} if binding.from_read else cells
        for name, cells in lineage.items():
            if cells:
                self.lineage[name] = cells
            else:
                self.lineage.pop(name, None)
        self.names = dict(reading.names)
        self.functions = dict(reading.functions)
        self.clients = dict(reading.clients)


def _describe_steps(steps: Sequence[TurnStep]) -> list[_DescribedStep]:
    """Each step that can be a source, earlier turns' first and then this
    turn's, each in step order. Of the earlier turns' steps, only those from
    the ``MAX_EARLIER_CELLS``-th last cell that ran on are read. A scratchpad
    call is one only when it ran a cell: anton runs no call it discarded, and
    another action's output repeats earlier cells or is empty. A tool call is
    one only when ``is_evidence_call`` says so. Each scratchpad keeps what
    its cells bound (``_Pad``) across cells and turns, until a reset or
    remove that ran. A cell ``_describe_cell`` leaves out because what it
    prints can only be what the agent memorized is a memory cell, and so is
    a later cell whose only source would be memory cells' data."""
    earlier = [step for step in steps if step.earlier_turn]
    ordered = [*_latest_cells(earlier, limit=MAX_EARLIER_CELLS), *(step for step in steps if not step.earlier_turn)]
    pads: dict[str, _Pad] = {}
    described: list[_DescribedStep] = []
    memory_cells: set[str] = set()
    for step in ordered:
        if step.tool_name != "scratchpad":
            if is_evidence_call(tool_name=step.tool_name, arguments=step.tool_input.arguments):
                described.append(_describe_tool(step))
            continue
        if step.content.startswith(_DISCARDED_CALL):
            continue
        call = step.tool_input.arguments
        if step.action in _NAMESPACE_RESETS:
            pads.pop(call.name, None)
            continue
        if not ran_cell(step):
            continue
        cell = step.cell if step.cell is not None else _cell_result(step.content)
        code = call.code or (cell.code if cell is not None else "")
        pad = pads.setdefault(call.name, _Pad())
        reading = read_cell(code, names=pad.names, functions=pad.functions, clients=pad.clients)
        drawn_from = pad.drawn_from(reading.loaded_names) - {step.tool_use_id}
        pad.record(reading, tool_use_id=step.tool_use_id)
        described_cell = _describe_cell(
            step, code=code, reading=reading, cell=cell, drawn_from=drawn_from, memory_cells=memory_cells,
        )
        if described_cell is not None:
            described.append(described_cell)
        elif step.tool_use_id:
            memory_cells.add(step.tool_use_id)
    return described


def is_evidence_call(*, tool_name: str, arguments: ToolArguments) -> bool:
    """Whether a call of ``tool_name`` with these arguments can be a step
    other than a scratchpad cell: a call of a tool ``is_evidence_tool``
    counts that does not read a memory file (``reads_memory_file``)."""
    return is_evidence_tool(name=tool_name) and not reads_memory_file(tool_name=tool_name, path=arguments.path)


def is_evidence_tool(*, name: str) -> bool:
    """Whether a call of the tool ``name`` can be a step other than a
    scratchpad cell: a web tool or read_text_file
    (``_NATIVE_EVIDENCE_TOOLS``), or a connector tool that returns the
    customer's records (``_records_tool``). It is the one answer to which
    tools are evidence."""
    return name in _NATIVE_EVIDENCE_TOOLS or _records_tool(name=name) is not None


def _records_tool(*, name: str) -> ToolNameParts | None:
    """The engine, connection and tool a connector tool's registered name is
    built from ({engine}__{tool}, or {engine}__{connection}__{tool} when the
    turn has two connections of one engine), when anton's
    ``returns_records`` says the tool returns the customer's records, such
    as CRM objects or campaign analytics. None for any other name: a tool
    anton runs itself, a connector tool that returns docs, schema or account
    details, a write tool, a tool of an engine anton has not classified, and
    every name when the installed anton cannot classify them."""
    if parse_namespaced_tool_name is None:
        return None
    parts = parse_namespaced_tool_name(name=name)
    if parts is None or not returns_records(engine=parts.engine, tool=parts.tool_name):
        return None
    return parts


def reads_memory_file(*, tool_name: str, path: str | None) -> bool:
    """Whether a call of ``tool_name`` with this path argument is a
    read_text_file of a file in a memory folder (``_in_memory_folder``).
    Such a read is never a step."""
    return tool_name == "read_text_file" and path is not None and _in_memory_folder(path=path)


def _in_memory_folder(*, path: str) -> bool:
    """Whether ``path`` names a file in a memory folder, at any depth of the
    path. The path is read by its text alone, without the whitespace around
    it, which anton drops too: either slash separates its parts, ``.`` and
    ``..`` collapse, and case is ignored, as macOS and Windows ignore it by
    default. A link from elsewhere into the folder is not followed, so a
    read through one stays a step."""
    parts = posixpath.normpath(path.strip().replace("\\", "/")).casefold().split("/")
    # A folder followed by at least one more part: a file in it, or in a
    # folder below it.
    return any(
        tuple(parts[start:start + len(folder)]) == folder
        for folder in _MEMORY_FOLDERS
        for start in range(len(parts) - len(folder))
    )


def _run_time_path(*, text: str) -> bool:
    """Whether ``text``, a string the cell reader found, is a path outside a
    memory folder that the code fills in when the cell runs: it holds a
    part the reader writes in braces, or a glob's * (``_RUN_TIME_PART``),
    and a slash or a backslash."""
    return (
        _RUN_TIME_PART.search(text) is not None
        and ("/" in text or "\\" in text)
        and not _in_memory_folder(path=text)
    )


def _latest_cells(steps: Sequence[TurnStep], *, limit: int) -> Sequence[TurnStep]:
    """``steps`` from the ``limit``-th last cell that ran on, or all of them
    when fewer cells ran."""
    ran = [index for index, step in enumerate(steps) if ran_cell(step)]
    return steps[ran[-limit]:] if len(ran) > limit else steps


def _cell_result(content: str) -> CellResult | None:
    """The cell result ``content`` holds. A result the cloud pod cut past its
    wire limit is no longer JSON, and only what the cell printed before the
    cut is read from it. JSON without what the cell printed is logged and
    gives None."""
    try:
        return CellResult.model_validate_json(content)
    except ValidationError as error:
        if not any(detail["type"] == "json_invalid" for detail in error.errors()):
            logger.warning("A scratchpad cell's result carries no printed output; no figure can name that cell for it")
            return None
    return CellResult(stdout=_clipped_stdout(content))


def _clipped_stdout(content: str) -> str:
    """What a cut cell result says the cell printed: the whole field when the
    cut missed it, what came before the cut when the cut fell inside it, and
    nothing when the field is gone."""
    key = content.find(_STDOUT_KEY)
    if key < 0:
        return ""
    start = key + len(_STDOUT_KEY)
    try:
        return json.decoder.scanstring(content, start)[0]
    except ValueError:
        pass
    cut = content.find(_CLIP_MARKER, start)
    fragment = content[start:cut] if cut >= 0 else content[start:]
    # The cut may split an escape, at most six characters such as the one for
    # a euro sign; drop what is left of it.
    for end in range(len(fragment), max(len(fragment) - 6, 0) - 1, -1):
        text = _json_string(fragment[:end])
        if text is not None:
            return text
    return ""


def _describe_cell(
    step: TurnStep,
    *,
    code: str,
    reading: CellReading,
    cell: CellResult | None,
    drawn_from: frozenset[str],
    memory_cells: Set[str],
) -> _DescribedStep | None:
    """A cell's kind, in order: a web tool call; a query (SQL, the data
    helper, a saved connection's variable, or a request that names one); a
    document it reads (not one it writes or only prints); a request to, or a
    string holding, a literal http(s) URL; else a step. Without a description
    a query, a web search or a step has an empty label, which the client
    words by the kind alone.

    A memory file (``_in_memory_folder``) holds the agent's own text, as a
    read_text_file of it does (``reads_memory_file``), so it is never one of
    the cell's documents. None for a cell that would otherwise be a bare
    step making no request, whose only sources are the memory files it read
    and ``memory_cells``, the earlier cells this rule left out: what it
    prints is what the agent memorized, read here or in an earlier cell. A
    request whose URL the code builds at run time may fetch live data, and
    so may a file read whose path the code builds at run time outside the
    memory folder (``_run_time_path``: data/{f} for os.path.join("data",
    f)), though that path ends in no document's extension. So a cell that
    makes such a request, or holds such a path that it neither writes to
    nor only prints, stays a step.

    The rule misses what the cell reader does not see: a cell that also
    computes a figure from numbers its code writes, or loads data through a
    call the reader does not classify (a library's download, a pickle, a
    query held in a name) or from a path a name holds alone (pd.read_csv(f)
    for each f in files), for which the reader names no string, is left out
    with its figures; and a memory path the code builds from a name
    (f"{MEM}/lessons.md") or finds by listing the folder
    (os.listdir(".anton/memory")) is read as any other file, so the cell
    names it or stays a step. A string with a part the code fills in and a
    slash that names no file (key = f"{region}/{year}") keeps a cell that
    reads only memory files a step too."""
    output = cell.stdout if cell is not None else ""
    description = cell.description if cell is not None else ""
    # A separately printed caption is not a pandas index name. Keep line
    # boundaries while removing known labels before examining row metadata.
    if reading.labels:
        captions = {label.strip() for label in reading.labels}
        output = "\n".join(" " * len(line) if line.strip() in captions else line for line in output.split("\n"))
    not_read = {*reading.written_files, *reading.labels}
    read = tuple(text for text in reading.strings if text not in not_read and _DOCUMENT.fullmatch(text))
    documents = tuple(text for text in read if not _in_memory_folder(path=text))
    web_page = next(
        (url for url in (*(request.url for request in reading.requests), *reading.strings) if URL_START.match(url)),
        None,
    )
    url = None
    if reading.calls_web_tool:
        url = _web_url(_found_url(output) or web_page)
        kind, label = SourceKind.web, _web_label(url) if url else description
    elif (
        reading.queries
        or reading.calls_query_minds_data
        or any(_DS_VARIABLE.fullmatch(name) for name in reading.environment_variables)
        or any(_DS_VARIABLE.search(request.url) for request in reading.requests)
    ):
        kind, label = SourceKind.query, description
    elif documents:
        kind, label = SourceKind.file, _documents_label(documents)
    elif web_page is not None:
        url = _web_url(web_page)
        kind, label = SourceKind.web, _web_label(web_page)
    elif (
        not reading.requests
        and not any(_run_time_path(text=text) for text in reading.strings if text not in not_read)
        and (read or drawn_from)
        and drawn_from <= memory_cells
    ):
        return None
    else:
        kind, label = SourceKind.step, description
    return _described(
        step,
        kind=kind,
        label=label,
        url=url,
        documents=documents,
        queries=reading.queries,
        requests=tuple(str(request) for request in reading.requests),
        output=output,
        literals=code,
        drawn_from=drawn_from,
    )


def _describe_tool(step: TurnStep) -> _DescribedStep:
    """A call of a tool ``is_evidence_tool`` counts: a web search or fetch;
    a connector tool that returns the customer's records, a query named by
    its engine, connection and tool that lists the statement the call ran
    when its arguments hold one as sql, whose result and input are read
    less their ISO date-times and whose result may only echo its input; or
    read_text_file, a file read by its path, where the line-range header
    and shortened-line marks are not the file's text and the line range it
    asked for is not a figure the model wrote. A read without a path is a
    step named by the tool."""
    arguments = step.tool_input.arguments
    output, literals = step.content, step.tool_input.text
    url = None
    documents: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    connector = _records_tool(name=step.tool_name)
    if step.tool_name in WEB_TOOLS:
        url = _web_url(arguments.url or _found_url(output))
        kind = SourceKind.web
        label = _web_label(url) if url else arguments.query
    elif connector is not None:
        kind, label = SourceKind.query, _connector_label(parts=connector)
        queries = (arguments.sql,) if arguments.sql else ()
        # A record's timestamps are not figures, and their minutes and zone
        # offsets would source almost any small number. A date-time in the
        # call's input, such as a filter's, would make the records' values
        # equal to its hour, minutes or seconds read as echoes. Each is
        # blanked in place, so every other number keeps its offset.
        output = _ISO_DATETIME.sub(_blanked, output)
        literals = _ISO_DATETIME.sub(_blanked, literals)
    else:
        output = _SHORTENED_LINE.sub("", _TEXT_RANGE_HEADER.sub("", output))
        path = arguments.path
        if path is not None:
            kind, label, documents, literals = SourceKind.file, _basename(path), (path,), path
        else:
            kind, label = SourceKind.step, step.tool_name
    return _described(
        step,
        kind=kind,
        label=label,
        url=url,
        documents=documents,
        queries=queries,
        requests=(),
        output=output,
        literals=literals,
    )


def _blanked(match: re.Match[str]) -> str:
    """``match``'s text as spaces, so every other number keeps its offset."""
    return " " * len(match.group(0))


def _described(
    step: TurnStep,
    *,
    kind: SourceKind,
    label: str,
    url: str | None,
    documents: tuple[str, ...],
    queries: tuple[str, ...],
    requests: tuple[str, ...],
    output: str,
    literals: str,
    drawn_from: frozenset[str] = frozenset(),
) -> _DescribedStep:
    # An output value equal to a number the step's input states only echoes
    # it, so it is not read, not even for a figure that rounds to it. An
    # earlier turn's step holds no figure, so neither is read for it, nor the
    # time units it names.
    if step.earlier_turn:
        input_values, output_numbers = frozenset(), ()
        output_time_units = time_units = frozenset()
    else:
        input_values = frozenset(n.value for n in _numbers(_DIGIT_UNDERSCORE.sub("", literals)))
        output_numbers = tuple(n for n in _output_numbers(output) if n.value not in input_values)
        output_time_units = _time_units_named(output)
        time_units = output_time_units | _time_units_named(f"{literals}\n{label}")
    return _DescribedStep(
        tool_use_id=step.tool_use_id,
        kind=kind,
        label=label,
        url=url,
        documents=documents,
        queries=queries,
        requests=requests,
        output=output,
        output_numbers=output_numbers,
        input_values=input_values,
        # With several documents a figure's page could be any one's.
        document_has_pages=kind is SourceKind.file and len(documents) == 1 and documents[0].lower().endswith(".pdf"),
        output_time_units=output_time_units,
        time_units=time_units,
        drawn_from=drawn_from,
    )


def _figure_step(step: _DescribedStep, *, files: AgentFiles | None) -> FigureStep:
    """The step as the event lists it. Its file is looked up only here, once
    a figure names it, since that reads the disk, and only when the step read
    one document: with several, the figure may come from any of them. A web
    page whose URL redaction changes is not linked."""
    named = files.named_file(step.documents[0]) if files is not None and len(step.documents) == 1 else None
    url = step.url if step.url is not None and _redacted(step.url) == step.url else None
    return FigureStep(
        tool_use_id=step.tool_use_id,
        kind=step.kind,
        label=_redacted(step.label),
        path=named.path if named is not None else None,
        attachment_id=named.attachment_id if named is not None else None,
        url=url,
        queries=[_capped(_redacted(query), limit=MAX_QUERY_CHARS) for query in step.queries],
        requests=[_capped(_redacted(request), limit=MAX_REQUEST_CHARS) for request in step.requests],
    )


def _readers(
    steps: Sequence[_DescribedStep], *, holders: Sequence[int], index_of: Mapping[str, int],
) -> list[int]:
    """The cells that read the data the holders computed the figure from, in
    step order, leaving out the holders themselves, in the places
    MAX_FIGURE_SOURCES leaves after the holders."""
    room = MAX_FIGURE_SOURCES - len(holders)
    if room <= 0:
        return []
    cells = {index_of[cell] for holder in holders for cell in steps[holder].drawn_from if cell in index_of}
    return sorted(cells - set(holders))[:room]


def _redacted(text: str) -> str:
    """``text`` with the credentials this sweep knows redacted. It is best
    effort: it finds a secret by its key's name or its own shape, so a secret
    under any other key name, one passed by position (ClickHouse's
    ``s3(url, key_id, secret)``), or a token in the path of a URL on a host
    ``_SECRET_URL_PATH`` does not list goes through as written.

    It redacts:

    - what ``scrub_credentials`` knows: the turn's registered connection
      secrets, the provider keys the server holds, and mdb_, sk-, AIza and
      AQ. keys;
    - ``_SECRET_SHAPE``: AWS access key ids (AKIA, ASIA), GitHub tokens
      (ghp_, gho_, ghu_, ghs_, ghr_, github_pat_), Slack tokens (xox?-,
      xapp-), Stripe secret and restricted keys (sk_, rk_), Hugging Face
      (hf_) and Google OAuth (ya29.) tokens, JWTs and PEM private keys;
    - a URL's user and password, the user holding a doubled or escaped
      quote, a lone quote before a letter, a digit or an underscore, or,
      before a password with no raw /, ?, #, ;, comma, & or quote ahead of
      its @, an @ (jane@corp.com:pw@host) (on overflow, a password of which over 1,024
      characters reach this sweep or a user over 256, its whole
      non-whitespace span, including ambiguous SQL quotes, even when the
      password holds an @), an Oracle EZConnect string's user and
      password, either one double-quoted or not, a proxy login's bracketed
      user, quoted or not, included, before a host name, an IPv6 address, a
      connect descriptor or a placeholder the code fills in ({host},
      %(host)s, $HOST) (and when an unquoted password holds a secret key's
      name and =, : or =>, the part after the @ up to where a key's value
      ends), a URL's query string and fragment after a path of any length,
      and the path of a Slack, Discord, Teams, Zapier, IFTTT or Telegram bot
      URL;
    - a bearer token;
    - a password a SQL statement sets with a keyword (``_KEYWORD_SECRET``,
      ``_IDENTIFIED_BY``, ``_REPLACED_SECRET``): a quoted value after a
      secret key's name and whitespace (PASSWORD 'x', DuckDB's SECRET 'x'
      and SESSION_TOKEN 'x', OPTIONS (sfPassword 'x')) or after PASSWORD FOR
      <user> =; the value after IDENTIFIED BY, IDENTIFIED BY PASSWORD or
      VALUES, or IDENTIFIED WITH <plugin> BY or AS, the plugin and the value
      quoted or not; and the current password after REPLACE. A string's
      introducer (E'x', N'x', _utf8mb4'x') stays before the mask;
    - the value of a key=value, key: value or key => value pair, quoted or
      not, with any auth scheme before it, whose key's name holds password,
      passwd, passphrase, pwd, secret, token, api key, access key, account
      key, private key, authorization, credential, cookie or signature, or
      whose last part is pass or sig, on its own or after _, - or . (pass,
      db_pass, sig; not dbpass) (``_SENSITIVE_NAME``). A quoted value keeps
      its quotes, and a string's introducer before them stays after a
      quoted key ("password": N'a b' becomes "password": N'[redacted]').
      After an unquoted key, the unquoted-value rule masks the introducer
      and the mask again, so only the closing quote stays (PASSWORD=N'a b'
      becomes PASSWORD=[redacted]'). An ODBC value in braces (Pwd={a;b}) is
      masked to its closing brace.

    It does not redact, and these go through as written:

    - the part after the quote of a URL query value that holds a quote
      other than the one around its string literal (sig=ab"cd inside
      '...'), or a backslash-escaped quote as MySQL writes it (sig=ab\\'cd):
      the rule stops at either quote, and telling which quote opened the
      literal would take a parser;
    - a URL's query string after a space, a lone quote or a
      backslash-escaped quote in its path (/Q3 sales.csv?k=v,
      /O\\'Brien.csv?k=v): the query string rule ends the path there, so
      only a value whose key names a secret (sig=, X-Amz-Signature=) is
      masked;
    - a URL with no scheme (//bob:pw@host): every user and password rule
      starts at ://;
    - the part of a URL password over 1,024 characters before a raw # or ?
      in it, in a web step's label, link and request
      (https://bob:Tr0ub4dor#<1,100 characters>@host reads
      https://bob:Tr0ub4dor): ``strip_url`` keeps a user and password too
      long for its window and cuts the URL at that # or ?, so the text that
      reaches this sweep has no @ and a password under 1,025 characters;
    - a password that holds a /, ?, #, ;, comma, & or quote after a user
      that holds an @, and the user's part after its @
      (snowflake://jane@corp.com:Pw/x@acct reads
      snowflake://***@corp.com:Pw/x@acct): such a user's password runs only
      to an @ before any of them, so that a user, a host and a port before a
      path, query string, parameter or next literal that holds an @
      (mysql://app@db:3306/s?password=P@ss) are not read as one user and
      password;
    - the part of a query string's value after its last @, after a user
      with no @ and a password (postgresql://app:pw@db/x?password=P@ss reads
      postgresql://***@ss): that user's password, which may hold /, ? or #,
      runs to the last @;
    - the part of an unquoted EZConnect password after an @ in it
      (scott/pw@x@host): the rule ends the password at its first @;
    - an EZConnect login whose user is over 128 characters, or that comes
      right after a scheme or a colon (oracle://scott/pw@host,
      jdbc:oracle:thin:scott/pw@host): the rule reads a user of at most 128
      characters, only where a word starts, so a URL's path is never read
      as one;
    - the value of a key whose name has more than 64 characters before or
      after the secret word in it: ``_SENSITIVE_NAME`` reads at most 64 on
      each side, which keeps it linear;
    - a quoted value inside another string literal, whose quotes SQL
      doubles (password=''x'' or PASSWORD ''x'' inside '...'): the rule
      reads the first doubled quote as an empty value;
    - a password a statement sets that starts with a space, a comma or a )
      (PASSWORD ' x'): the keyword rules read that quote as the one that
      closes a literal ending in the keyword ('Your password ' || name);
    - a password set after a secret key's name in dollar quotes (PASSWORD
      $$x$$), with no space or with a comment between the name and the
      value (PASSWORD'x', PASSWORD /* c */ 'x'), after TO (SET
      s3_secret_access_key TO 'x'), or through MySQL's PASSWORD() function
      (SET PASSWORD FOR bob = PASSWORD('x')): ``_KEYWORD_SECRET`` reads only
      a quote right after the name's whitespace or the =;
    - a password after IDENTIFIED BY with a comment between the two words
      (IDENTIFIED/**/BY x, IDENTIFIED -- c<newline>BY x):
      ``_IDENTIFIED_BY`` reads only whitespace between its words.

    It masks more than the secret in these cases, on display only:

    - a password over 1,024 characters, a user over 256, a password followed
      by more than 1,024 characters of host and path, or a host:port, with
      or without a user before it, followed by more than 1,024 characters of
      path: everything after the :// up to the next space is masked, quotes
      and the SQL after them included;
    - any :// followed by more than 256 characters with no /, ?, #, @ or :
      in them, then a : or an @ (a long JDBC property list,
      jdbc:sqlserver://h;a=b;...;keyStoreLocation=C:\\k): everything after
      the :// is masked the same way;
    - text after a URL's host with no / before an @ in the same run of
      non-whitespace (s3://bucket,me@corp): it reads as a user name and is
      masked with the URL's user, unless a quote that no letter, digit or
      underscore follows ends the user name first ('s3://bucket','me@corp'
      stays);
    - a URL's host and port with no user before them, when an @ follows in
      the same run of non-whitespace (https://h:8443/u/a@b): the host reads
      as a user and the port and path up to the last @ as its password, and
      both are masked;
    - a value after a column or key name that holds a secret word, or
      whose last part is pass or sig, though it names no secret
      (password_hint = 'pet name', signature_count = 5, email_signature:
      Thanks, p.sig = 0.05): it reads as that key's value;
    - an EZConnect host after a password that holds a secret key's name,
      and what follows the host up to a space, a comma, a semicolon, an
      ampersand or a quote: it is masked as that key's value, and
      password= stands in for the password;
    - a column divided by a double-quoted name followed by an @
      (a/"Col Name"@x), a double-quoted column divided by anything
      followed by an @ ("Col Name"/a@x), or a subscripted column divided
      by anything followed by an @ (a[1]/b@x), reads as an EZConnect string
      and becomes ***@x;
    - a word divided by a word followed by an @ and a letter, a digit, {,
      $ or % (revenue/unit@five, revenue/unit@$5, a/b@{x}, growth/yr@%5):
      it reads as an EZConnect login before a host, or before a
      placeholder the code fills in, and becomes ***@five or ***@$5;
    - an unquoted secret value that starts with _ or is a lone E or N,
      right before the quote that closes its literal, when a character
      other than a space, a comma or a ) follows that quote
      ('Password=_x';...): it reads as a string's introducer, so that quote
      opens a quoted value that runs to the next quote or the end of the
      text, and the mask takes the SQL in between;
    - a secret key's value that opens with a brace and has no closing
      brace: it is masked to the end of its line;
    - the word after "identified by" in prose ("rows identified by
      region"), and a quoted word after a secret key's name and whitespace,
      in prose or as a column's quoted alias ("the secret 'sauce'",
      password_hash "Hash"): they read as a password a statement sets.

    A connector whose credential has a key name or shape not listed here
    adds it to ``_SENSITIVE_NAME`` or ``_SECRET_SHAPE``, with a case in
    tests/test_figure_sources.py."""
    # Whole shapes must be removed before a named-value replacement can
    # erase a delimiter, such as a PEM key's BEGIN line.
    text = _SECRET_SHAPE.sub(_REDACTED, scrub_credentials(text))
    # The long-password rule runs first: a shorter rule would stop at an @
    # inside a long password and leave the rest of it. The long-user rule
    # runs after the short one: run first, it would read a long run after a
    # :// inside a short password as a user and take the @ the short rule
    # ends at, leaving the user and the start of the password.
    text = _LONG_USERINFO.sub(r"\1***", text)
    text = _USERINFO.sub(r"\1***@", text)
    text = _LONG_USER.sub(r"\1***", text)
    text = _EZCONNECT.sub(_masked_ezconnect, text)
    text = _SECRET_URL_PATH.sub(r"\g<service>***", text)
    text = _QUERY_STRING.sub(_masked_query_string, text)
    text = _QUOTED_SECRET.sub(_redacted_quoted_value, text)
    text = _KEYWORD_SECRET.sub(_redacted_quoted_value, text)
    text = _BARE_SECRET.sub(lambda match: match["key"] + _REDACTED, text)
    text = _BEARER.sub(lambda match: match["scheme"] + _REDACTED, text)
    # IDENTIFIED BY runs after the key=value and bearer rules: its unquoted
    # value stops at a quote or a parenthesis, and its mask takes the key's
    # name or the Bearer those rules need, so run first it would leave the
    # rest of their value. REPLACE reads the mask before it, so it runs last.
    text = _IDENTIFIED_BY.sub(_redacted_keyword_value, text)
    text = _REPLACED_SECRET.sub(_redacted_keyword_value, text)
    return text


def _masked_ezconnect(match: re.Match[str]) -> str:
    """An ``_EZCONNECT`` match with its user and password masked. An
    unquoted password that holds a secret key's name and =, : or =>
    (scott/password=Tr0ub@dor) may instead be that key and a value that runs
    past the @. Then a fixed key, password=, stands in for the password, with
    the mask as its value, so ``_BARE_SECRET`` masks that value and what
    follows it up to where a value ends, the part after the @ included. The
    key's own name is not kept: it may be the password itself
    (scott/Kz9_token=@host) or its start (scott/MyPwd:Tr0ub@host)."""
    key = _KEY_IN_PASSWORD.search(match["bare"] or "")
    return "***@" if key is None else "***/password=***@"


def _masked_query_string(match: re.Match[str]) -> str:
    """A ``_QUERY_STRING`` match with its query string and fragment masked,
    or the URL as written when it has neither."""
    return match[1] + ("?..." if match["query"] else "")


def _redacted_quoted_value(match: re.Match[str]) -> str:
    """A ``_QUOTED_SECRET`` match with its value redacted inside its quotes,
    the closing one kept only when the value had one."""
    quote = match["single"] or match["double"]
    close = match["single_close"] or match["double_close"] or ""
    return match["key"] + quote + _REDACTED + close


def _redacted_keyword_value(match: re.Match[str]) -> str:
    """An ``_IDENTIFIED_BY`` or ``_REPLACED_SECRET`` match with its value
    redacted: inside its quotes when it has them, whole when it is bare."""
    return match["key"] + _REDACTED if match["bare"] else _redacted_quoted_value(match)


def _capped(text: str, *, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _found_url(text: str) -> str | None:
    """The first http(s) URL in ``text``, without the punctuation or markup
    that ends the sentence around it: a period, a comma, ** and the like, or
    a ) or ] that closes no ( or [ inside the URL."""
    found = _URL.search(text)
    if found is None:
        return None
    url = found.group(0)
    while url and (
        url[-1] in _URL_TRAILERS
        or (url[-1] == ")" and url.count("(") < url.count(")"))
        or (url[-1] == "]" and url.count("[") < url.count("]"))
    ):
        url = url[:-1]
    return url


def _web_url(url: str | None) -> str | None:
    """``url`` as a link the event can carry: http(s), with no user, password,
    query string or fragment, and no placeholder left in it."""
    if not url or not URL_START.match(url):
        return None
    stripped = strip_url(url)
    return None if "{" in stripped else stripped


def _web_label(url: str) -> str:
    """A web address shown as host, port and path, without its user,
    password or query string. strip_url keeps a user or a password too long
    for its window, so the host is urlsplit's hostname, which leaves out a
    user and password that end at the @. A /, ? or # inside such a user or
    password ends urlsplit's host there instead, and the rest of it reads as
    a port that does not parse. So when the port does not parse, the label
    is built from what follows the last @, or is the stripped URL when no @
    is left; a port out of range reads the same way. ``_redacted`` masks
    that stripped URL as a long user or password only when over 1,024
    characters of the password are left in it: a raw # or ? earlier in a
    long password, where strip_url cuts, leaves the part before it in the
    label. urlsplit lowercases the host and writes the port as a number
    (Example.COM:08080 reads example.com:8080). A long password whose part
    before the / reads as a port (bob:8080/<over 1,024 characters>@host)
    still shows in the label as a host and a path."""
    stripped = strip_url(url)
    try:
        parts = urlsplit(stripped)
    except ValueError:
        return stripped
    try:
        port = parts.port
    except ValueError:
        if "@" in stripped:
            return _web_label("https://" + stripped.rpartition("@")[2])
        return stripped
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host += f":{port}"
    return (host + parts.path).rstrip("/") or stripped


def _connector_label(*, parts: ToolNameParts) -> str:
    """A connector tool as "<engine>: <tool>", with the connection in
    parentheses after the engine when the name carries one, since then the
    turn has two connections of that engine."""
    if parts.connection_name is None:
        return f"{parts.engine}: {parts.tool_name}"
    return f"{parts.engine} ({parts.connection_name}): {parts.tool_name}"


def _documents_label(documents: Sequence[str]) -> str:
    """The documents' file names, the first MAX_LABELED_DOCUMENTS of them,
    and how many more there are."""
    names = ", ".join(_basename(document) for document in documents[:MAX_LABELED_DOCUMENTS])
    more = len(documents) - MAX_LABELED_DOCUMENTS
    return f"{names} and {more} more" if more > 0 else names


def _basename(path: str) -> str:
    return posixpath.basename(path.replace("\\", "/"))


def _leading_object(raw: str) -> dict[str, Any] | None:
    """The JSON object ``raw`` starts with, whatever follows it, or None when
    it doesn't start with one. Nesting too deep for the parser's recursion
    limit reads as no object."""
    try:
        parsed, _ = _JSON.raw_decode(raw.lstrip())
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_string(escaped: str) -> str | None:
    """The text a JSON string's escaped contents stand for, or None when they
    don't read as one."""
    try:
        return json.loads(f'"{escaped}"')
    except ValueError:
        return None



@dataclass(frozen=True)
class _OutputValues:
    """A step's output values sorted by size, so the ones a figure rounds
    from are found by bisection instead of a scan."""

    sizes: tuple[Decimal, ...]
    # The sizes of the negative values, the only ones a negative figure takes.
    negative_sizes: tuple[Decimal, ...]


def _output_values(numbers: Sequence[_Number]) -> _OutputValues:
    return _OutputValues(
        sizes=tuple(sorted(number.value for number in numbers)),
        negative_sizes=tuple(sorted(number.value for number in numbers if number.negative)),
    )


def _holders(
    steps: Sequence[_DescribedStep], *, outputs: Sequence[_OutputValues], figure: _Number, time_unit: _TimeUnit | None,
) -> list[int]:
    """The steps whose output holds the figure, in step order. With the
    figure's ``time_unit``, a step whose output names time units, none of
    them that one, holds none unless its code or label names that one: the
    22 of its lead_time_days is not "22 hours"."""
    return [
        index for index, step in enumerate(steps)
        if _holds(step, values=outputs[index], figure=figure, tolerance=figure.tolerance)
        and (time_unit is None or not step.output_time_units or time_unit in step.time_units)
    ]


def _time_unit(text: str, *, number: _Number, next_number: _Number | None = None) -> _TimeUnit | None:
    """The time unit a number of ``text`` is written in, glued to its digits
    (12h), the word after it (22 hours, a 13.08-day lead time, 22 business
    days) or, for a range's first number, the unit of ``next_number`` (22-25
    hours, 22 to 25 hours), or None. A rate ("$42 per hour", "10 units/day")
    has none, and nor does a number with another unit glued to it (40kg)."""
    if number.unit:
        return _TIME_UNITS.get(number.unit)
    after = _WORD_AFTER.match(text, number.end)
    if after and (unit := _TIME_UNITS.get(after["word"].lower())) is not None:
        return unit
    if next_number is not None and _RANGE_JOIN.fullmatch(text, number.end, next_number.start):
        return _time_unit(text, number=next_number)
    return None


def _time_units_named(text: str) -> frozenset[_TimeUnit]:
    """The time units a step's text names: by their plurals
    (``_STEP_TIME_WORDS``), as words and inside identifiers, and in any form
    right after a number (36h, a 30-day window), read as a figure's is
    (``_time_unit``)."""
    plurals = {
        _TIME_UNITS[word] for match in _WORD.finditer(text) if (word := match.group().lower()) in _STEP_TIME_WORDS
    }
    after_numbers = {unit for number in _numbers(text) if (unit := _time_unit(text, number=number)) is not None}
    return frozenset(plurals | after_numbers)


def _named_holders(
    steps: Sequence[_DescribedStep], *, outputs: Sequence[_OutputValues], figure: _Number, holders: Sequence[int],
) -> list[int]:
    """The holders the figure names: those with a tool_use_id, in step order.
    Past MAX_FIGURE_SOURCES, the steps that print its exact value are kept
    before those that print a value that only rounds to it."""
    named = [index for index in holders if steps[index].tool_use_id]
    if len(named) <= MAX_FIGURE_SOURCES:
        return named
    exact = [
        index for index in named
        if _holds(steps[index], values=outputs[index], figure=figure, tolerance=Decimal(0))
    ]
    rounded = [index for index in named if index not in exact]
    return sorted((exact + rounded)[:MAX_FIGURE_SOURCES])


def _holds(step: _DescribedStep, *, values: _OutputValues, figure: _Number, tolerance: Decimal) -> bool:
    """Whether the step's output has a value ``_same_value`` takes for the
    figure, and its input does not state the figure itself."""
    if figure.value in step.input_values:
        return False
    sizes = values.negative_sizes if figure.negative else values.sizes
    low, high = figure.value - tolerance, figure.value + tolerance
    return _any_between(sizes, low=low, high=high) or (
        figure.percent and _any_between(sizes, low=low, high=high, key=_hundredfold)
    )


def _any_between(
    sizes: Sequence[Decimal], *, low: Decimal, high: Decimal, key: Callable[[Decimal], Decimal] | None = None,
) -> bool:
    """Whether a size in sorted ``sizes`` falls in [low, high], after ``key``."""
    index = bisect_left(sizes, low, key=key)
    if index == len(sizes):
        return False
    size = sizes[index] if key is None else key(sizes[index])
    return size <= high


def _hundredfold(size: Decimal) -> Decimal:
    return size * 100


def _source(step: _DescribedStep, *, figure: _Number) -> FigureSource:
    page = _page_of(step, figure=figure) if step.document_has_pages else None
    return FigureSource(tool_use_id=step.tool_use_id, page=page)


def _page_of(step: _DescribedStep, *, figure: _Number) -> int | None:
    """The page of the figure's first place in the output, where a value equal
    to it comes before one that only rounds to it. The last page marker
    before that place names its page, a footer ("Page 1 of 2") closing the
    page before it; with no marker before it, the first footer after it
    names its page. Headers that count from page 0 are read from 1."""
    place = _first_place(step.output_numbers, figure=figure)
    if place is None:
        return None
    markers = list(_PAGE.finditer(step.output))
    from_zero = any(marker["pages"] is None and int(marker["page"]) == 0 for marker in markers)
    before = [marker for marker in markers if marker.end() <= place.start]
    if before:
        marker = before[-1]
        page = int(marker["page"])
        if marker["pages"] is not None:
            page = page + 1 if page < int(marker["pages"]) else 0
        elif from_zero:
            page += 1
    else:
        footer = next((m for m in markers if m.start() >= place.end and m["pages"] is not None), None)
        page = int(footer["page"]) if footer is not None else 0
    return page if page >= 1 else None


def _first_place(numbers: Sequence[_Number], *, figure: _Number) -> _Number | None:
    for tolerance in (Decimal(0), figure.tolerance):
        for number in numbers:
            if _same_value(number, figure=figure, tolerance=tolerance):
                return number
    return None


def _same_value(number: _Number, *, figure: _Number, tolerance: Decimal) -> bool:
    """Whether an output number is within ``tolerance`` of the figure as the
    answer writes it. A percent figure also takes the fraction it stands for
    (0.125 for 12.5%); a negative figure takes only a negative number, and an
    unsigned one takes either sign."""
    if figure.negative and not number.negative:
        return False
    if abs(number.value - figure.value) <= tolerance:
        return True
    return figure.percent and abs(number.value * 100 - figure.value) <= tolerance


def _same_form(asked: _Number, *, figure: _Number) -> bool:
    """Whether a number in the question writes the figure: the same written
    form (``_WrittenForm``) less the unit, so the same value, sign, currency,
    scale and percent, and "30 days" does not cover "$30"."""
    return replace(_WrittenForm.of(number=asked), unit="") == replace(_WrittenForm.of(number=figure), unit="")


def _output_numbers(output: str) -> list[_Number]:
    """Every number a step's output holds, numbers as written before their
    parts. A number with thousands separators is one value unless its commas
    sit between fields; then each part is its own value. Printed row labels
    and size lines are not read."""
    text = _unlabeled(output)
    csv_rows = _csv_rows(text)
    whole = [number for number in _numbers(text) if not _between_fields(text, number=number, csv_rows=csv_rows)]
    starts = [number.start for number in whole]
    parts = []
    for number in _numbers(text, pattern=_PLAIN_NUMBER):
        index = bisect_right(starts, number.start) - 1
        if index < 0 or number.end > whole[index].end:
            parts.append(number)
    return whole + parts


def _unlabeled(output: str) -> str:
    """``output`` with the row labels of a printed frame or Series, and the
    lines that state its size, blanked; every other character keeps its
    place. A row label counts rows, so it would source every small whole
    number.

    A run of lines that each start with a number and two spaces is a frame's
    or a Series' rows when it has two or more lines, a column header above
    it, or a Series footer below it. Right under a frame's indented column
    header, a number and one space before a negative value is a row too,
    since pandas leaves that space for the minus sign; print(year, change)
    writes the same line by hand, so elsewhere it is not a row. The run goes
    on across the row where pandas cuts a long print, and across a
    MultiIndex's rows that leave the outer keys blank, so one print is one
    run.

    An index name that pandas printed (``_names_index``) makes the index data,
    read in any order: a Series' over its footer, and a frame's under its
    column header. So groupby, value_counts(), set_index(), pivot_table() and
    crosstab() keys printed with print() are read. An index printed with no
    name is pandas' own row numbers, blanked in whatever order a sort or a
    sample left them when the run sits right under a frame's indented column
    header or over a Series' footer. A run with neither is blanked when its
    numbers count up; otherwise it is values printed by hand where labels
    go, and is read.

    Some data reads as row numbers when it counts up: the keys of a named
    Series printed with to_string(), as anton's sample() prints a Series,
    which has no footer; the keys of a frame index whose name is not
    identifier words, such as one with a hyphen; and a frame printed without
    its index whose first column counts up from column 0, whose header looks
    like a named columns header over row numbers. Some data reads as row
    numbers in any order: the keys of a Series whose index has no name, such
    as the value_counts() of a Series with no name or a Series made from a
    dict, and of a Series index whose name is not a Python identifier, such
    as one with a space or a hyphen, whose name line reads as a caption.

    Some row numbers read as data: an unnamed Series under a one-word caption
    with no colon that the cell's code does not write as a literal, such as
    print(title) with a title a function returned, which reads as the
    index's name (``_describe_cell`` blanks a caption the code spells, as in
    print("Results"), before this runs); a frame whose columns have a name,
    after reset_index() and a sort, since its header is not indented or the
    columns' level names sit between it and the rows; a frame printed right
    after a print whose index has a name, when its header is as wide as that
    print's first row and its column name holds a digit, is as wide as one
    of that row's values and starts where that value starts ("          q3"
    under "1         22"), or one column off it where a minus sign stands in
    that value or in the column name ("          q3" under "1        -16",
    "         -12" under "1         13"), since the header reads as an inner
    MultiIndex row and the frame's rows join the named run; and a Series whose
    index has no name that anton's sample() prints after a sort
    (sample(df.qty.nlargest(3))), since sample() prints the rows with
    to_string() under its "Head (10):" or "First 3:" caption, with no
    footer, so rows out of order read as values printed by hand where
    labels go."""
    lines = output.split("\n")
    labels = [_ROW_LABEL.match(line) for line in lines]
    index = 0
    while index < len(lines):
        if labels[index] is None:
            index += 1
            continue
        first = index
        # pandas aligns an inner row with the rows around it and leaves its
        # outer keys blank, so it is as wide as the run's first row and its
        # key starts where one of that row's later fields starts
        # (``_starts_at_a_field``). The next print's indented header joins
        # the run only when it passes both and holds a digit, as a column
        # name as wide as the value above it and starting where it starts
        # can.
        field_starts = _field_starts(lines[first], after=labels[first].end())
        while index < len(lines) and (
            labels[index] is not None
            or _TRUNCATED.fullmatch(lines[index]) is not None
            or (
                len(lines[index]) == len(lines[first])
                and _starts_at_a_field(lines[index], first_row=lines[first], field_starts=field_starts)
                and _INNER_ROW.match(lines[index]) is not None
            )
        ):
            index += 1
        under_header = first > 0 and _FRAME_HEADER.match(lines[first - 1]) is not None
        run = [
            (row, label) for row, label in enumerate(labels[first:index], start=first)
            if label is not None and (under_header or _NEGATIVE_FIRST.match(lines[row], label.end()) is None)
        ]
        over_footer = index < len(lines) and _SERIES_FOOTER.match(lines[index]) is not None
        named_index = first > 0 and _names_index(
            lines[first - 1], header=lines[first - 2] if first > 1 else "", over_footer=over_footer,
        )
        if not (len(run) > 1 or under_header or over_footer):
            continue
        counts = [Decimal(label.group(0).rstrip("%")) for _, label in run]
        if not named_index and (
            under_header or over_footer or all(low < high for low, high in zip(counts, counts[1:]))
        ):
            for row, label in run:
                lines[row] = " " * label.end() + lines[row][label.end():]
    return _SHAPE.sub(_blanked, "\n".join(lines))


def _field_starts(line: str, *, after: int) -> set[int]:
    """Each column where a field of ``line`` past column ``after`` starts."""
    return {match.start() for match in _FIELD_START.finditer(line) if match.start() > after}


def _starts_at_a_field(line: str, *, first_row: str, field_starts: Set[int]) -> bool:
    """Whether ``line`` starts where pandas starts an inner MultiIndex key
    under ``first_row``, whose later fields start at ``field_starts``: at
    one of them, or a column off one where a minus sign stands. pandas
    leaves a column before a key for its minus sign, so a -2 under a 0
    starts a column left of the 0, and a 0 under a -2 a column right of the
    -2's minus sign."""
    indent = len(line) - len(line.lstrip())
    return (
        indent in field_starts
        or (indent + 1 in field_starts and line[indent] == "-")
        or (indent - 1 in field_starts and first_row[indent - 1] == "-")
    )


def _names_index(line: str, *, header: str, over_footer: bool) -> bool:
    """Whether ``line``, right above a run of printed rows, is the name pandas
    printed for their index; ``header`` is the line above it. Each name is an
    identifier.

    A Series prints its index name unindented over its rows, and a
    MultiIndex's names two spaces apart. One space between words is a
    caption's, as print("Top sites") writes it, so a Series index whose name
    is not a Python identifier, such as one with a space or a hyphen, reads
    as a caption. A frame prints its index names a space apart under its
    column header: a header indented past the row labels or, when the
    columns have a name, one as wide as the name line. pandas pads a frame's
    index name line to the frame's width, so a name of several words is read
    only from a padded line: the unpadded line under a header, where a frame
    with two levels of columns prints the columns' level names, is not an
    index name. A caption's line is not padded either. A frame wider than
    the display has the columns pandas cut out marked with "..." on every
    line, its padded index name line included, and that mark is not part of
    the name."""
    name = line.rstrip()
    if not name or name[0].isspace():
        return False
    if over_footer:
        return all(level.isidentifier() for level in _LEVEL_GAP.split(name))
    if not (_FRAME_HEADER.match(header) is not None or len(header) == len(line) > len(name)):
        return False
    if name.isidentifier():
        return True
    words = [word for word in name.split() if word != "..."]
    return len(line) > len(name) and bool(words) and all(word.isidentifier() for word in words)


def _csv_rows(text: str) -> set[int]:
    """Where the lines of raw CSV data begin: lines without a letter that have
    as many commas as the nearest line above them with one, their header.
    Neither has a space after a comma, as prose does."""
    rows = set()
    header_commas = 0
    offset = 0
    for line in text.split("\n"):
        if _LETTER.search(line):
            header_commas = 0 if ", " in line else line.count(",")
        elif header_commas and ", " not in line and line.count(",") == header_commas:
            rows.add(offset)
        offset += len(line) + 1
    return rows


def _between_fields(text: str, *, number: _Number, csv_rows: set[int]) -> bool:
    """Whether a number's thousands separators sit between fields: it is only
    digits and commas, and it is an item of a printed list ("[100,250,300]")
    or of a CSV row ("1,250,S1", or a row under a header with as many
    commas). A currency sign, decimals, a scale, or a space or colon before
    it ("Total units: 1,250") keep it one number."""
    if not _GROUPED.fullmatch(text, number.start, number.end):
        return False
    before = text[number.start - 1:number.start]
    after = text[number.end:number.end + 2]
    return (
        before in (",", "[")
        or after[:1] == "]"
        or (after[:1] == "," and after[1:2].strip() != "")
        or number.start in csv_rows
    )


def _numbers(
    text: str,
    *,
    skips: Iterable[tuple[int, int]] = (),
    pattern: re.Pattern[str] = _NUMBER,
    prose: bool = False,
) -> list[_Number]:
    """The numbers ``text`` writes, outside ``skips``. Read as prose (the
    answer or the question), a hyphen just after a number is a range's: the
    number after it is the range's second end (150-200), not a negative one
    (5%-10%). A hedge word before a round whole number widens its tolerance
    ("about 1,200")."""
    skipped = _Spans.of(skips)
    numbers: list[_Number] = []
    for match in pattern.finditer(text):
        start, end = match.span()
        after_figure = prose and bool(numbers) and numbers[-1].end == start
        if after_figure and match["sign"] == "-":
            start = match.end("sign")
        range_end = prose and bool(numbers) and numbers[-1].end == start - 1 and text[start - 1] == "-"
        if not _standalone(text, start=start, end=end, range_end=range_end) or skipped.hold(start=start, end=end):
            continue
        hedged = prose and _HEDGE.search(text, max(0, start - 20), start) is not None
        number = _parse(match, hedged=hedged, start=start)
        if number is not None:
            numbers.append(number)
    return numbers


def _standalone(text: str, *, start: int, end: int, range_end: bool = False) -> bool:
    """A number not glued to an identifier: not A2210 or W3001, not Q-03, not
    32GB, and not one part of a version such as 1.2.3. Underscores around it
    are emphasis (_250_) unless a word goes on past them (row_250). After a
    hyphen it is a range's second end when ``range_end`` says a figure ends
    just before the hyphen."""
    left, right = start, end
    while left > 0 and text[left - 1] == "_":
        left -= 1
    while right < len(text) and text[right] == "_":
        right += 1
    before = text[left - 1] if left > 0 else ""
    if before.isalnum() or before == ".":
        return False
    if before == "-" and not range_end and left > 1 and text[left - 2].isalnum():
        return False
    after = text[right] if right < len(text) else ""
    if after.isalnum():
        return False
    return not (after == "." and right + 1 < len(text) and text[right + 1].isdigit())


def _parse(match: re.Match[str], *, hedged: bool = False, start: int | None = None) -> _Number | None:
    """The number a match writes, or None for a length such as 250m (a
    lowercase scale without a currency sign) and for an exponent past any
    real size, which would overflow decimal's arithmetic. A ``start`` past
    the match's sign reads the number without it."""
    start = match.start() if start is None else start
    groups = match.groupdict()
    digits = groups["digits"].replace(",", "")
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    if abs(value.adjusted()) > _MAX_EXPONENT:
        return None
    currency = groups.get("currency") or ""
    suffix = (groups.get("suffix") or "").strip(" \u00a0")
    if suffix in ("m", "b") and not currency:
        return None
    scale = _SCALES.get(suffix.lower(), Decimal(1))
    percent = suffix.lower() in _PERCENTS
    unit = "" if percent or suffix.lower() in _SCALES else suffix.lower().replace("\u00d7", "x")
    exponent = value.as_tuple().exponent
    decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    tolerance = Decimal(5).scaleb(-(decimals + 1))
    if hedged and digits.isdigit():
        tolerance = max(tolerance, _rounding(digits))
    return _Number(
        value=value * scale,
        tolerance=tolerance * scale,
        percent=percent,
        negative=bool(groups.get("sign")) and start == match.start(),
        currency=currency,
        scale=scale,
        start=start,
        end=match.end(),
        unit=unit,
    )


def _rounding(digits: str) -> Decimal:
    """Half a unit of a whole number's last nonzero digit when it has two or
    more significant digits: "about 1,200" may be 50 off. One significant
    digit, as in 1,000, says nothing about rounding, so it stays exact."""
    significant = digits.lstrip("0").rstrip("0")
    if len(significant) < 2:
        return Decimal(0)
    zeros = len(digits.lstrip("0")) - len(significant)
    return Decimal(5).scaleb(zeros - 1)


def _skipped_spans(answer: str) -> list[tuple[int, int]]:
    return [match.span() for pattern in _ANSWER_SKIPS for match in pattern.finditer(answer)]


@dataclass(frozen=True)
class _Spans:
    """Spans sorted by start, each with the furthest end it or a span
    before it reaches, so whether one span holds a place is found by
    bisection rather than a scan of every span."""

    starts: tuple[int, ...]
    reach: tuple[int, ...]

    @classmethod
    def of(cls, spans: Iterable[tuple[int, int]]) -> _Spans:
        ordered = sorted(spans)
        return cls(
            starts=tuple(start for start, _ in ordered),
            reach=tuple(accumulate((end for _, end in ordered), max)),
        )

    def hold(self, *, start: int, end: int) -> bool:
        """Whether one span starts at or before ``start`` and ends at or
        after ``end``."""
        index = bisect_right(self.starts, start) - 1
        return index >= 0 and self.reach[index] >= end


def _utf16_offset(text: str, *, index: int) -> int:
    """Python indexes code points and JavaScript UTF-16 code units; they part
    at the first character outside the Basic Multilingual Plane, an emoji."""
    return len(text[:index].encode("utf-16-le")) // 2
