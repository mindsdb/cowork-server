"""Tie each figure in a finished answer to the steps whose output holds it.

The formatter hands this module the final answer, the question and the tool
calls the model saw with their results: this turn's, and those of the earlier
turns the model is seeded with. Each figure in the answer (a number that isn't
part of an identifier, a date, a list marker, a citation marker, a link, code
or display math) gets one origin:

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
- ``none``: nothing in the turn holds it, so the renderer marks it as a figure
  to check before acting on it.

A step of an earlier turn never holds a figure. Its cells are read only for
the names, functions and data their scratchpad passes on to this turn's cells,
so a figure the answer repeats from an earlier turn, with no step of this turn
holding it, has no source. Only the last ``MAX_EARLIER_CELLS`` earlier-turn
cells are read, and the formatter reads the history only when the answer has
figures (``has_figures``) and this turn ran a cell (``ran_cell``). These
limits never change a figure's origin. They can only drop an earlier cell a
figure names as the one that read its data, or a string or function a cell
of this turn reuses.

Only the scratchpad's cells and the tools in ``EVIDENCE_TOOLS`` are steps: the
web tools and read_text_file. Every other tool hands back the agent's own text
(an earlier answer, an artifact it wrote, a memory), the user's answer, or a
status line.

A step whose own input states the figure (a cell that prints a literal it was
given) is not a source for it: its output only echoes what the model wrote.
For the same reason an output value equal to a number its input states is not
read at all. A figure the model works out in its own text, without a step
printing it, has no source.

Each step a figure names is described once: its kind, a short label, the file
it read, the web page it reached, and the SQL and HTTP requests its code shows
(``cell_queries``). Its label, web page, SQL and requests pass through
``_redacted``, a best-effort credential sweep, before they leave this module.
Its file is named by its place in the project or by its files-store id, never
by the text the code spelled.
"""

from __future__ import annotations

import json
import posixpath
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import accumulate
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

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
    Figure,
    FigureOrigin,
    FigureSource,
    FigureSources,
    FigureStep,
    SourceKind,
)

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
# The tools besides the scratchpad whose results are evidence: a web page or
# search result, and a file's text. recall_history and anton's recall hand
# back earlier answers verbatim, the artifact and skill tools the agent's own
# writing, memorize and ask_user what the agent or the user wrote, read_image
# a line about the image rather than what it shows, and the rest a status.
EVIDENCE_TOOLS = WEB_TOOLS | {"read_text_file"}


class CellResult(BaseModel):
    """A scratchpad cell's result as anton sends it, json.dumps(asdict(Cell)),
    or as the formatter rebuilds it from a replayed history row. What the cell
    printed is required, so a result without it is logged rather than read as
    a cell that printed nothing."""

    model_config = ConfigDict(extra="ignore")

    stdout: str
    code: str = ""
    description: str = ""


@dataclass(frozen=True)
class TurnStep:
    """One tool call the model made and its result: this turn's, as the
    formatter streamed it or the turn's tool rows hold it, or an earlier
    turn's, as the history the model is seeded with holds it."""

    tool_use_id: str
    tool_name: str
    action: str
    tool_input: str
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
)

# A line that marks a page of a document's text: a header such as
# "--- page 2 ---", or a footer such as "Page 1 of 2", which closes its page.
_PAGE = re.compile(
    r"^[ \t\-=#*\[(]*page[ \t]*[:#]?[ \t]*(?P<page>\d+)"
    r"(?:[ \t]*(?:of|/)[ \t]*(?P<pages>\d+))?[ \t\r\-=*\])\.:]*$",
    re.IGNORECASE | re.MULTILINE,
)

# A printed DataFrame or Series row: an integer row label (or a describe()
# percentile) and two or more spaces before the values.
_ROW_LABEL = re.compile(r"\d+%?(?= {2,}\S)")
# A frame's column header, indented past the row labels.
_FRAME_HEADER = re.compile(r"[ \t]+\S")
_SERIES_FOOTER = re.compile(r"(?:Name|Length|Freq|dtype): ")
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

_REDACTED = "[redacted]"
# The rules below bound how far each part of a match may run, so that text a
# cell spells (up to cell_queries.MAX_STRING_CHARS) is redacted in time that
# grows with its length, whatever it holds.
#
# A URL's user and password, whatever comes before the ://. Once a : has ended
# the user name, the password may hold /, ? or #, as a database URI's may; a
# quote may be password data under the enclosing SQL dialect. Keep it in
# the bounded scan; ambiguous detail is safer to redact than to expose.
_USERINFO = re.compile(r"(://)(?:[^\s/?#'\"@:]{0,256}:\S{0,1024}|[^\s/?#'\"@]{0,256})@")
# On overflow, mask the whole non-whitespace span, including its host and
# any quotes. Without the enclosing SQL dialect a quote may be password
# data, even when not doubled or escaped. Omit excess detail rather than
# leave part of that credential visible. Possessive matching stays linear.
_LONG_USERINFO = re.compile(r"(://)[^\s/?#'\"@:]{0,256}:\S{1025,}+")
# An Oracle EZConnect string, user/password@host[:port][/service], whose user
# starts where a word starts, so a URL's path is never read as one.
_EZCONNECT = re.compile(r"(?<![\w$#./:-])[A-Za-z_][\w$#]{0,127}/[^\s/@'\"]{1,1024}@(?=[\w./-])")
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
# presigned S3 or Azure URL).
_QUERY_STRING = re.compile(r"(://[^\s'\"?#]{0,2048}+)[?#][^\s'\"]*+")
# A key that names a secret, as a word of its own or a part of one, starting
# where a word starts: password, Pwd, s3_secret_access_key, X-Api-Key,
# private_key, passphrase, AccountKey. "pass" counts only as the key's last
# part (pass, db_pass), so pass_rate and passenger are not secret keys.
_SENSITIVE_NAME = (
    r"(?<![\w.-])[\w.-]{0,64}(?:password|passwd|passphrase|pwd|secret|token|api[_-]?key|access[_-]?key"
    r"|account[_-]?key|private[_-]?key|authorization|credential|cookie|(?<![a-z0-9])pass(?![\w.-]))[\w.-]{0,64}+"
)
# What joins a key to its value: =, :, or => as Ruby and PHP write it. Once
# => matched, the = is never read alone, which would take the > as a value.
_ASSIGN = r"[ \t]*(?>=>|[:=])[ \t]*"
# A secret key's value in quotes after its key, the key quoted or not:
# password='a b c', "token": "x", 'Authorization': 'Bearer x'. A doubled
# quote inside the value is a quote, as SQL writes it. The quotes stay, so the
# SQL around the value still reads. Multiline values retain one boundary;
# an unclosed value is redacted through the end of the input.
_QUOTED_SECRET = re.compile(
    rf"(?P<key>(?<![\w.-])[\"']?{_SENSITIVE_NAME}[\"']?{_ASSIGN})"
    r"(?:(?P<single>')(?:''|\\[\s\S]|[^'\\])*+(?P<single_close>')?"
    r"|(?P<double>\")(?:\"\"|\\[\s\S]|[^\"\\])*+(?P<double_close>\")?)",
    re.IGNORECASE,
)
# A secret key's unquoted value, up to a space, a comma, a semicolon, an
# ampersand, or a quote that closes the literal around it. An HTTP auth
# scheme before the value (Basic, Bearer, Token, any word followed by a
# token-shaped value) is redacted with it.
_BARE_SECRET = re.compile(
    rf"(?P<key>{_SENSITIVE_NAME}{_ASSIGN})"
    r"(?:[A-Za-z][\w-]{0,31}[ \t]+(?=[A-Za-z0-9._~+/=-]{8}))?"
    r"[^\s,;&'\"](?:[^\s,;&'\"]|['\"](?=[^\s,;&)\]}'\"]))*+",
    re.IGNORECASE,
)
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
    for number in numbers:
        held = _holders(described, outputs=outputs, figure=number)
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


@dataclass(frozen=True)
class _ScratchpadCall:
    """What a scratchpad call's arguments name: the scratchpad, and the code
    it runs."""

    pad: str
    code: str


def _describe_steps(steps: Sequence[TurnStep]) -> list[_DescribedStep]:
    """Each step that can be a source, earlier turns' first and then this
    turn's, each in step order. Of the earlier turns' steps, only those from
    the ``MAX_EARLIER_CELLS``-th last cell that ran on are read. A scratchpad
    call is one only when it ran a cell: anton runs no call it discarded, and
    another action's output repeats earlier cells or is empty. A tool call is
    one only when the tool is one of ``EVIDENCE_TOOLS``. Each scratchpad keeps
    what its cells bound (``_Pad``) across cells and turns, until a reset or
    remove that ran."""
    earlier = [step for step in steps if step.earlier_turn]
    ordered = [*_latest_cells(earlier, limit=MAX_EARLIER_CELLS), *(step for step in steps if not step.earlier_turn)]
    pads: dict[str, _Pad] = {}
    described: list[_DescribedStep] = []
    for step in ordered:
        if step.tool_name != "scratchpad":
            if step.tool_name in EVIDENCE_TOOLS:
                described.append(_describe_tool(step))
            continue
        if step.content.startswith(_DISCARDED_CALL):
            continue
        call = _scratchpad_call(step)
        if step.action in _NAMESPACE_RESETS:
            pads.pop(call.pad, None)
            continue
        if not ran_cell(step):
            continue
        cell = step.cell if step.cell is not None else _cell_result(step.content)
        code = call.code or (cell.code if cell is not None else "")
        pad = pads.setdefault(call.pad, _Pad())
        reading = read_cell(code, names=pad.names, functions=pad.functions, clients=pad.clients)
        drawn_from = pad.drawn_from(reading.loaded_names) - {step.tool_use_id}
        pad.record(reading, tool_use_id=step.tool_use_id)
        described.append(_describe_cell(step, code=code, reading=reading, cell=cell, drawn_from=drawn_from))
    return described


def _latest_cells(steps: Sequence[TurnStep], *, limit: int) -> Sequence[TurnStep]:
    """``steps`` from the ``limit``-th last cell that ran on, or all of them
    when fewer cells ran."""
    ran = [index for index, step in enumerate(steps) if ran_cell(step)]
    return steps[ran[-limit]:] if len(ran) > limit else steps


def _scratchpad_call(step: TurnStep) -> _ScratchpadCall:
    """The call's scratchpad and code. Arguments with something after them
    still read; arguments cut short keep only their scratchpad's name, and
    the cell's result gives the code that ran."""
    arguments = _leading_object(step.tool_input)
    if arguments is not None:
        return _ScratchpadCall(pad=str(arguments.get("name") or ""), code=str(arguments.get("code") or ""))
    found = _PAD_NAME.search(step.tool_input)
    return _ScratchpadCall(pad=_json_string(found.group(1)) if found else "", code="")


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
    step: TurnStep, *, code: str, reading: CellReading, cell: CellResult | None, drawn_from: frozenset[str],
) -> _DescribedStep:
    """A cell's kind, in order: a web tool call; a query (SQL, the data
    helper, a saved connection's variable, or a request that names one); a
    document it reads (not one it writes or only prints); a request to, or a
    string holding, a literal http(s) URL; else a step. Without a description
    a query, a web search or a step has an empty label, which the client
    words by the kind alone."""
    output = cell.stdout if cell is not None else ""
    description = cell.description if cell is not None else ""
    # A separately printed caption is not a pandas index name. Keep line
    # boundaries while removing known labels before examining row metadata.
    if reading.labels:
        captions = {label.strip() for label in reading.labels}
        output = "\n".join(" " * len(line) if line.strip() in captions else line for line in output.split("\n"))
    not_read = {*reading.written_files, *reading.labels}
    documents = tuple(text for text in reading.strings if text not in not_read and _DOCUMENT.fullmatch(text))
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
    """A call of one of ``EVIDENCE_TOOLS``: a web search or fetch, or
    read_text_file, a file read by its path, where the line-range header and
    shortened-line marks are not the file's text and the line range it asked
    for is not a figure the model wrote. A read without a path is a step
    named by the tool."""
    tool_input = _leading_object(step.tool_input) or {}
    output, literals = step.content, step.tool_input
    url = None
    documents: tuple[str, ...] = ()
    if step.tool_name in WEB_TOOLS:
        url = _web_url(str(tool_input.get("url") or "") or _found_url(output))
        kind = SourceKind.web
        label = _web_label(url) if url else str(tool_input.get("query") or "")
    else:
        output = _SHORTENED_LINE.sub("", _TEXT_RANGE_HEADER.sub("", output))
        path = tool_input.get("path")
        if isinstance(path, str) and path:
            kind, label, documents, literals = SourceKind.file, _basename(path), (path,), path
        else:
            kind, label = SourceKind.step, step.tool_name
    return _described(
        step,
        kind=kind,
        label=label,
        url=url,
        documents=documents,
        queries=(),
        requests=(),
        output=output,
        literals=literals,
    )


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
    # earlier turn's step holds no figure, so neither is read for it.
    if step.earlier_turn:
        input_values, output_numbers = frozenset(), ()
    else:
        input_values = frozenset(n.value for n in _numbers(_DIGIT_UNDERSCORE.sub("", literals)))
        output_numbers = tuple(n for n in _output_numbers(output) if n.value not in input_values)
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
    - a URL's user and password (on overflow, its whole non-whitespace
      span, including ambiguous SQL quotes),
      an Oracle EZConnect string's password, a URL's query string and
      fragment, and the path of a Slack, Discord, Teams, Zapier, IFTTT or
      Telegram bot URL;
    - a bearer token;
    - the value of a key=value, key: value or key => value pair, quoted or
      not, with any auth scheme before it, whose key's name holds password,
      passwd, passphrase, pwd, secret, token, api key, access key, account
      key, private key, authorization, credential or cookie, or whose last
      part is pass, on its own or after _, - or . (pass, db_pass; not
      dbpass) (``_SENSITIVE_NAME``). A quoted value keeps its quotes.

    A connector whose credential has a key name or shape not listed here
    adds it to ``_SENSITIVE_NAME`` or ``_SECRET_SHAPE``, with a case in
    tests/test_figure_sources.py."""
    # Whole shapes must be removed before a named-value replacement can
    # erase a delimiter, such as a PEM key's BEGIN line.
    text = _SECRET_SHAPE.sub(_REDACTED, scrub_credentials(text))
    text = _USERINFO.sub(r"\1***@", text)
    text = _LONG_USERINFO.sub(r"\1***", text)
    text = _EZCONNECT.sub("***@", text)
    text = _SECRET_URL_PATH.sub(r"\g<service>***", text)
    text = _QUERY_STRING.sub(r"\1?...", text)
    text = _QUOTED_SECRET.sub(_redacted_quoted_value, text)
    text = _BARE_SECRET.sub(lambda match: match["key"] + _REDACTED, text)
    text = _BEARER.sub(lambda match: match["scheme"] + _REDACTED, text)
    return text


def _redacted_quoted_value(match: re.Match[str]) -> str:
    """A ``_QUOTED_SECRET`` match with its value redacted inside its quotes,
    the closing one kept only when the value had one."""
    quote = match["single"] or match["double"]
    close = match["single_close"] or match["double_close"] or ""
    return match["key"] + quote + _REDACTED + close


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
    """A web address shown as host and path, without its query string."""
    stripped = strip_url(url)
    try:
        parts = urlsplit(stripped)
    except ValueError:
        return stripped
    return (parts.netloc + parts.path).rstrip("/") or stripped


def _documents_label(documents: Sequence[str]) -> str:
    """The documents' file names, the first MAX_LABELED_DOCUMENTS of them,
    and how many more there are."""
    names = ", ".join(_basename(document) for document in documents[:MAX_LABELED_DOCUMENTS])
    more = len(documents) - MAX_LABELED_DOCUMENTS
    return f"{names} and {more} more" if more > 0 else names


def _basename(path: str) -> str:
    return posixpath.basename(path.replace("\\", "/"))


def _leading_object(raw: str) -> dict[str, Any] | None:
    """The JSON object ``raw`` starts with, whatever follows it."""
    try:
        parsed, _ = _JSON.raw_decode(raw.lstrip())
    except ValueError:
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


def _holders(steps: Sequence[_DescribedStep], *, outputs: Sequence[_OutputValues], figure: _Number) -> list[int]:
    """The steps whose output holds the figure, in step order."""
    return [
        index for index, step in enumerate(steps)
        if _holds(step, values=outputs[index], figure=figure, tolerance=figure.tolerance)
    ]


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
    """Whether a number in the question writes the figure: the same value,
    sign, currency, scale and percent, so "30 days" does not cover "$30"."""
    return (asked.value, asked.negative, asked.currency, asked.scale, asked.percent) == (
        figure.value, figure.negative, figure.currency, figure.scale, figure.percent,
    )


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
    it, or a Series footer below it. Its numbers are row labels when they
    count up without a named index, as a frame's own row numbers do, kept or
    filtered; or when the column header is right above the run, so they are the frame's row numbers
    in whatever order a sort or a sample left them. Otherwise they are values
    printed where labels go, as value_counts() or set_index() print them, and
    are read."""
    lines = output.split("\n")
    labels = [_ROW_LABEL.match(line) for line in lines]
    index = 0
    while index < len(lines):
        if labels[index] is None:
            index += 1
            continue
        first = index
        while index < len(lines) and labels[index] is not None:
            index += 1
        run = [(row, label) for row, label in enumerate(labels[first:index], start=first) if label is not None]
        under_header = first > 0 and _FRAME_HEADER.match(lines[first - 1]) is not None
        over_footer = index < len(lines) and _SERIES_FOOTER.match(lines[index]) is not None
        # pandas prints an index's name on its own, unindented line above
        # Series rows or below a frame's column header. Those labels are
        # actual data even when sorted or reduced to a single value. Require
        # an identifier; an arbitrary caption is not reliable index metadata.
        # Non-identifier index names remain ambiguous and keep the fallback.
        named_index = (
            first > 0 and lines[first - 1].rstrip().isidentifier() and not lines[first - 1][0].isspace()
            and (over_footer or (first > 1 and _FRAME_HEADER.match(lines[first - 2]) is not None))
        )
        if not (len(run) > 1 or under_header or over_footer):
            continue
        counts = [Decimal(label.group(0).rstrip("%")) for _, label in run]
        if not named_index and (under_header or all(low < high for low, high in zip(counts, counts[1:]))):
            for row, label in run:
                lines[row] = " " * label.end() + lines[row][label.end():]
    return _SHAPE.sub(lambda match: " " * len(match.group(0)), "\n".join(lines))


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
    exponent = value.as_tuple().exponent
    decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    tolerance = Decimal(5).scaleb(-(decimals + 1))
    if hedged and digits.isdigit():
        tolerance = max(tolerance, _rounding(digits))
    return _Number(
        value=value * scale,
        tolerance=tolerance * scale,
        percent=suffix.lower() in _PERCENTS,
        negative=bool(groups.get("sign")) and start == match.start(),
        currency=currency,
        scale=scale,
        start=start,
        end=match.end(),
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
