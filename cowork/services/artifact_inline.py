"""Make an artifact's HTML self-contained, with no network and no file outside it.

Every asset the document keeps becomes a ``data:`` URI read from the pinned
artifact folder; every other reference is dropped and listed in ``omitted``.

The document is parsed from bytes by ``lxml.html`` with the network off.
libxml2's HTML parser reads no DTD internal subset, so a declared external
entity stays literal text, and it has no XInclude step, so ``xi:include`` is
only an unknown element. Nothing here parses artifact HTML with
``lxml.etree``, hands a parser a filename or URL, or calls ``xinclude()``.

Elements and attributes are denied by default: only the allowlists below
survive. An unknown element is unwrapped, so its text stays, while the
elements in ``_DROPPED_ELEMENTS`` go with everything inside them. CSS goes
through tinycss2, which decodes escapes before it recognizes ``url(``, so an
escaped ``u\\72 l(`` is found too.

When the rest of the document is lost, past the node cap, past the deadline
or where libxml2 stopped reading the source, the output ends with
``TRUNCATION_NOTE``.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re
import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import unquote, unquote_to_bytes

import lxml.etree
import lxml.html
from lxml.etree import ParserError

from cowork.common.paths import PinnedDir
from cowork.schemas.artifact_export import (
    InlinedDocument,
    InlineLimits,
    InlineMode,
    OmittedReason,
    OmittedReference,
)
from cowork.services.artifact_files import (
    EXPORT_SCANNED_ENTRIES,
    ArtifactFileContent,
    ArtifactFileRefused,
    ScanBudget,
    read_artifact_file_entry,
    split_relative_path,
)

if TYPE_CHECKING:
    from cowork.services.artifact_css import CssRewriter

# Reads one file of the artifact by its path inside the folder, or raises
# ``ArtifactFileRefused`` with the reason the export records.
AssetReader = Callable[[tuple[str, ...]], ArtifactFileContent]

# The paragraph that ends a document whose rest was left out.
TRUNCATION_NOTE = "The rest of this document was left out because it was too large to convert."
# How long one document may take. Past it, the walk and the stylesheet being
# rewritten stop, and the output ends with ``TRUNCATION_NOTE``.
DEADLINE_SECONDS = 10.0
# Everything XML 1.0 forbids: C0 controls other than tab, newline and
# carriage return, lone surrogates, U+FFFE and U+FFFF. libxml2's HTML parser
# keeps these in names, values and text, but lxml refuses to set or delete a
# string holding one.
XML_UNSAFE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")

# Element names as libxml2's HTML parser reports them: lowercased, with no
# namespace, so SVG's ``linearGradient`` arrives as ``lineargradient``.
_HTML_ELEMENTS = frozenset(
    {
        "html", "head", "body", "title", "style",
        "address", "article", "aside", "footer", "header", "hgroup", "main",
        "nav", "section", "h1", "h2", "h3", "h4", "h5", "h6",
        "blockquote", "dd", "div", "dl", "dt", "figcaption", "figure", "hr",
        "li", "menu", "ol", "p", "pre", "ul",
        "a", "abbr", "b", "bdi", "bdo", "br", "cite", "code", "data", "dfn",
        "em", "i", "kbd", "mark", "q", "rp", "rt", "ruby", "s", "samp",
        "small", "span", "strong", "sub", "sup", "time", "u", "var", "wbr",
        "del", "ins", "img", "canvas",
        "caption", "col", "colgroup", "table", "tbody", "td", "tfoot", "th",
        "thead", "tr",
        "details", "summary", "fieldset", "legend", "label", "meter",
        "progress", "output",
    }
)
_SVG_ELEMENTS = frozenset(
    {
        "svg", "a", "circle", "clippath", "defs", "desc", "ellipse", "filter",
        "g", "line", "lineargradient", "marker", "mask", "path", "pattern",
        "polygon", "polyline", "radialgradient", "rect", "stop", "switch",
        "symbol", "text", "textpath", "tspan", "use",
        "feblend", "fecolormatrix", "fecomponenttransfer", "fecomposite",
        "feconvolvematrix", "fediffuselighting", "fedisplacementmap",
        "fedistantlight", "fedropshadow", "feflood", "fefunca", "fefuncb",
        "fefuncg", "fefuncr", "fegaussianblur", "femerge", "femergenode",
        "femorphology", "feoffset", "fepointlight", "fespecularlighting",
        "fespotlight", "fetile", "feturbulence",
    }
)
_MATH_ELEMENTS = frozenset(
    {
        "math", "annotation", "menclose", "merror", "mfenced", "mfrac", "mi",
        "mlabeledtr", "mmultiscripts", "mn", "mo", "mover", "mpadded",
        "mphantom", "mprescripts", "mroot", "mrow", "ms", "mspace", "msqrt",
        "mstyle", "msub", "msubsup", "msup", "mtable", "mtd", "mtext", "mtr",
        "munder", "munderover", "none", "semantics",
    }
)
# Removed with everything inside them, in any namespace. ``meta`` is among
# them because the output declares its own charset; ``script``, ``style`` and
# ``title`` have rules of their own in ``_Inliner._visit``, and so does a
# ``link`` to the artifact's own stylesheet, which becomes a ``style``.
_DROPPED_ELEMENTS = frozenset(
    {
        # Content a browser reads as raw text or keeps inert, where libxml2
        # may build elements, or the reverse.
        "noscript", "template", "xmp", "noembed", "noframes", "plaintext",
        "textarea", "iframe",
        # Elements that load, run or point somewhere.
        "frame", "frameset", "object", "embed", "applet", "portal", "param",
        "base", "meta", "link", "video", "audio", "source", "track", "image",
        # Form controls that load or submit.
        "input", "button", "select", "option", "optgroup", "datalist",
        "keygen",
        # SVG animation, and SVG elements that load or embed.
        "animate", "animatemotion", "animatetransform", "animatecolor", "set",
        "discard", "foreignobject", "feimage", "cursor", "font-face-uri",
        "handler", "listener", "tref", "altglyph",
        # MathML elements that switch a browser back to HTML parsing.
        "annotation-xml", "mglyph", "malignmark",
    }
)
# A browser parses these as void, but libxml2 nests what follows some of them
# (``source``, ``track``, ``embed``, ``keygen``) inside. That content is
# really their following siblings, so refusing one of these unwraps it.
_VOID_ELEMENTS = frozenset(
    {
        "area", "base", "basefont", "bgsound", "br", "col", "embed", "frame",
        "hr", "image", "img", "input", "keygen", "link", "meta", "param",
        "source", "track", "wbr",
    }
)

_HTML_GLOBAL_ATTRIBUTES = frozenset(
    {
        "id", "class", "title", "lang", "dir", "role", "hidden", "translate",
        "align", "valign", "width", "height", "bgcolor", "border", "nowrap",
    }
)
_TABLE_CELL_ATTRIBUTES = frozenset({"colspan", "rowspan", "headers", "scope", "abbr"})
_EDIT_ATTRIBUTES = frozenset({"datetime"})
# Per element, beyond the global ones. ``href`` and ``src`` have rules of
# their own, and ``style`` goes through the CSS rewrite.
_HTML_ATTRIBUTES: dict[str, frozenset[str]] = {
    "a": frozenset({"name", "rel", "hreflang"}),
    "img": frozenset({"alt"}),
    "ol": frozenset({"start", "reversed", "type"}),
    "ul": frozenset({"type"}),
    "li": frozenset({"value", "type"}),
    "table": frozenset({"cellpadding", "cellspacing", "frame", "rules", "summary"}),
    "td": _TABLE_CELL_ATTRIBUTES,
    "th": _TABLE_CELL_ATTRIBUTES,
    "col": frozenset({"span"}),
    "colgroup": frozenset({"span"}),
    "time": _EDIT_ATTRIBUTES,
    "del": _EDIT_ATTRIBUTES,
    "ins": _EDIT_ATTRIBUTES,
    "data": frozenset({"value"}),
    "meter": frozenset({"value", "min", "max", "low", "high", "optimum"}),
    "progress": frozenset({"value", "max"}),
    "details": frozenset({"open"}),
    "label": frozenset({"for"}),
    "output": frozenset({"for", "name"}),
    "style": frozenset({"media"}),
    "script": frozenset({"type", "async", "defer"}),
}
_SVG_ATTRIBUTES = frozenset(
    {
        "id", "class", "lang", "xml:space", "viewbox", "preserveaspectratio",
        "version", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx",
        "ry", "fx", "fy", "fr", "z", "width", "height", "d", "points",
        "pathlength", "transform", "opacity", "display", "visibility",
        "overflow", "color", "cursor", "fill", "fill-opacity", "fill-rule",
        "stroke", "stroke-width", "stroke-opacity", "stroke-linecap",
        "stroke-linejoin", "stroke-dasharray", "stroke-dashoffset",
        "stroke-miterlimit", "clip-path", "clip-rule", "clippathunits",
        "mask", "maskunits", "maskcontentunits", "filter", "filterunits",
        "primitiveunits", "marker-start", "marker-mid", "marker-end",
        "markerwidth", "markerheight", "markerunits", "refx", "refy",
        "orient", "gradientunits", "gradienttransform", "spreadmethod",
        "offset", "stop-color", "stop-opacity", "patternunits",
        "patterncontentunits", "patterntransform", "font-family", "font-size",
        "font-size-adjust", "font-stretch", "font-style", "font-variant",
        "font-weight", "text-anchor", "text-decoration", "dominant-baseline",
        "alignment-baseline", "baseline-shift", "letter-spacing",
        "word-spacing", "writing-mode", "dx", "dy", "rotate", "textlength",
        "lengthadjust", "startoffset", "method", "spacing", "side",
        "stddeviation", "in", "in2", "result", "mode", "operator", "k1", "k2",
        "k3", "k4", "values", "type", "tablevalues", "slope", "intercept",
        "amplitude", "exponent", "radius", "scale", "xchannelselector",
        "ychannelselector", "basefrequency", "numoctaves", "seed",
        "stitchtiles", "flood-color", "flood-opacity", "lighting-color",
        "surfacescale", "diffuseconstant", "specularconstant",
        "specularexponent", "kernelmatrix", "kernelunitlength", "order",
        "divisor", "bias", "targetx", "targety", "edgemode", "preservealpha",
        "azimuth", "elevation", "pointsatx", "pointsaty", "pointsatz",
        "limitingconeangle", "vector-effect", "shape-rendering",
        "text-rendering", "color-interpolation",
        "color-interpolation-filters", "paint-order", "role", "focusable",
    }
)
_MATH_ATTRIBUTES = frozenset(
    {
        "id", "class", "dir", "display", "displaystyle", "mathvariant",
        "mathsize", "mathcolor", "mathbackground", "scriptlevel", "fence",
        "separator", "separators", "open", "close", "stretchy", "symmetric",
        "largeop", "movablelimits", "accent", "accentunder", "lspace",
        "rspace", "linethickness", "columnalign", "rowalign", "columnspan",
        "rowspan", "columnlines", "rowlines", "frame", "form", "notation",
        "width", "height", "depth", "voffset", "minsize", "maxsize",
        "subscriptshift", "superscriptshift",
    }
)
_KEPT_ATTRIBUTE_PREFIXES = ("aria-", "data-")
# Attributes that make a browser load something. On a kept ``img`` the
# ``src`` and ``srcset`` are inlined instead; anywhere else the value is
# dropped and listed in ``omitted``.
_LOAD_ATTRIBUTES = frozenset(
    {
        "src", "srcset", "data", "poster", "background", "codebase",
        "archive", "classid", "lowsrc", "dynsrc", "manifest",
    }
)
_HREF_ATTRIBUTES = frozenset({"href", "xlink:href"})

# A script a render service may fetch: one file of one npm package at one
# exact version on jsDelivr, with no range, tag, query or dot segment.
_PINNED_CDN_SCRIPT = re.compile(
    r"https://cdn\.jsdelivr\.net/npm/"
    r"(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*"
    r"@[0-9]+\.[0-9]+\.[0-9]+"
    r"(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)+"
)
# What a browser's script tokenizer treats specially inside a script; a
# script holding any of them could end at a different place on a reparse.
_UNSTABLE_SCRIPT_MARKERS = ("<!--", "<script", "</script")

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:")
_QUERY_OR_FRAGMENT = re.compile(r"[?#]")
_NAVIGATION_SCHEME = re.compile(r"(?:https?|mailto):", re.IGNORECASE)
_DATA_URI = re.compile(r"data:([^,]*),(.*)", re.IGNORECASE | re.DOTALL)
# A ``srcset`` candidate's URL runs to the next whitespace; commas before it
# separate candidates. Its descriptors run to a comma outside parentheses.
_SRCSET_URL = re.compile(r"[ \t\n\r\f,]*([^ \t\n\r\f,][^ \t\n\r\f]*)")
_SRCSET_DESCRIPTORS = re.compile(r"(?:[^,(]|\([^)]*\)?)*,?")
_HTML_TOKEN_SEPARATORS = re.compile(r"[ \t\n\r\f]+")
_BASE64_WHITESPACE = re.compile(rb"[ \t\n\r\f]")

# The signatures that open each raster type the export keeps.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
_JPEG_FRAME_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
# The JPEG markers with no length after them: TEM, SOI and RST0-7.
JPEG_STANDALONE_MARKERS = frozenset({0x01, 0xD8, *range(0xD0, 0xD8)})
_FONT_SIGNATURES = (
    (b"wOF2", "font/woff2"),
    (b"wOFF", "font/woff"),
    (b"\x00\x01\x00\x00", "font/ttf"),
    (b"true", "font/ttf"),
    (b"OTTO", "font/otf"),
)

# The omitted list keeps at most this many entries, so a hostile document
# can't grow it without bound. The drops past it still happen.
_MAX_OMITTED = 1000


class _Namespace(StrEnum):
    HTML = "html"
    SVG = "svg"
    MATH = "math"


class _Action(Enum):
    KEEP = "keep"
    UNWRAP = "unwrap"
    DROP = "drop"


class _Visit(NamedTuple):
    action: _Action
    namespace: _Namespace  # the namespace the element's children are in


class AssetKind(Enum):
    IMAGE = "image"
    FONT = "font"


@dataclass(frozen=True)
class _Asset:
    uri: str
    size: int


class _ImageSources(NamedTuple):
    """An image's references: those tried in order, then those never tried."""

    tried: list[str]
    untried: list[str]


class _ImageHeader(NamedTuple):
    mime: str
    width: int
    height: int


class _ParsedHtml(NamedTuple):
    root: lxml.html.HtmlElement
    stopped: bool  # libxml2 hit a resource limit and dropped the rest of the input


@dataclass
class _ElementFrame:
    """A kept element whose children ``_Inliner.sanitize`` is still visiting.

    ``pending`` holds what is left, last first: a child to visit, or the tail
    of an unwrapped child, which comes after that child's own children.
    ``texts[0]`` becomes the element's text and ``texts[i]`` the tail of
    ``children[i - 1]``.
    """

    element: lxml.html.HtmlElement
    namespace: _Namespace  # the namespace the element's children are in
    pending: list[lxml.html.HtmlElement | str]
    texts: list[list[str]]
    children: list[lxml.html.HtmlElement] = field(default_factory=list)
    changed: bool = False

    def rebuild(self) -> None:
        """Give the element its kept children and their merged text in one pass.

        lxml's ``drop_tag`` and ``drop_tree`` search the parent and copy the
        text merged so far on every call, which is quadratic in the number of
        siblings. Text set here loses the characters lxml refuses.
        """
        if not self.changed:
            return
        self.element[:] = self.children
        self.element.text = _joined_text(pieces=self.texts[0])
        for child, pieces in zip(self.children, self.texts[1:], strict=True):
            child.tail = _joined_text(pieces=pieces)


@dataclass
class UriPlaceholders:
    """Long ``data:`` URIs held out of an lxml parse, each behind a short placeholder.

    libxml2 with huge_tree off stops reading after about 10 MB of large
    attribute or text tokens and drops the rest of the document, so until
    the parse is done the text holds a placeholder instead, and ``restore``
    swaps each one back. The nonce is random, so a document can't spell a
    placeholder of its own. The inliner and the Word writer both use it.
    """

    uris: list[str] = field(default_factory=list)
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))

    def hold(self, *, uri: str) -> str:
        """The placeholder for *uri*: letters, digits and a hyphen."""
        self.uris.append(uri)
        return f"{self.nonce}{len(self.uris) - 1}-"

    def restore(self, *, text: str) -> str:
        """*text* with each placeholder swapped back for its URI."""
        if not self.uris:
            return text
        placeholder = re.compile(re.escape(self.nonce) + r"([0-9]+)-")
        return placeholder.sub(lambda match: self.uris[int(match.group(1))], text)


def folder_reader(*, folder: PinnedDir, max_bytes: int) -> AssetReader:
    """Read files below *folder* through the pinned walk, all reads sharing one scan budget.

    A read that fails for any other reason than a refusal, such as an I/O
    error, is a missing file: the export leaves that reference out.
    """
    scan_budget = ScanBudget(remaining=EXPORT_SCANNED_ENTRIES)

    def read(parts: tuple[str, ...]) -> ArtifactFileContent:
        try:
            return read_artifact_file_entry(
                folder=folder, parts=parts, max_bytes=max_bytes, scan_budget=scan_budget
            )
        except (OSError, ValueError) as exc:
            raise ArtifactFileRefused(OmittedReason.MISSING, "read failed") from exc

    return read


def inline_artifact_html(
    *,
    html: bytes,
    folder: PinnedDir,
    base_parts: tuple[str, ...],
    mode: InlineMode,
    limits: InlineLimits,
) -> InlinedDocument:
    """Return *html* with every kept asset inlined and everything else dropped.

    *base_parts* is the folder of the source document inside *folder*, and
    references resolve against it. Going over a limit drops an asset, a
    declaration or the rest of the tree, and never raises.
    """
    return inline_html(
        html=html,
        read_asset=folder_reader(folder=folder, max_bytes=limits.max_asset_bytes),
        base_parts=base_parts,
        mode=mode,
        limits=limits,
    )


def inline_html(
    *,
    html: bytes,
    read_asset: AssetReader,
    base_parts: tuple[str, ...],
    mode: InlineMode,
    limits: InlineLimits,
) -> InlinedDocument:
    """``inline_artifact_html``, reading each file through *read_asset*.

    The export's conversion process reads through the server this way, since
    it gets no path and no folder of its own.
    """
    # Imported here: the CSS rewrite works on this module's InlineState.
    from cowork.services.artifact_css import CssRewriter

    state = InlineState(
        read_asset=read_asset,
        base_parts=base_parts,
        mode=mode,
        limits=limits,
        deadline=time.monotonic() + DEADLINE_SECONDS,
    )
    inliner = _Inliner(state=state, css=CssRewriter(state=state))
    # Removing an unknown wrapper can leave a nesting the parser builds
    # differently (a block inside a paragraph, say), so the output is what
    # one more parse makes of it: the structure every later parse rebuilds.
    reparsed = _parse_html(data=inliner.sanitized_html(html=html))
    if state.truncated or reparsed.stopped:
        _append_truncation_note(root=reparsed.root)
    return InlinedDocument(
        html=state.placeholders.restore(text=_serialize_html(root=reparsed.root)),
        omitted=state.omitted,
        inlined_bytes=state.total_bytes,
        scripts_dropped=inliner.scripts_dropped,
    )


@dataclass
class InlineState:
    """What the HTML walk and the CSS rewrite share while one document inlines.

    Both spend the one inline budget and reference count, read through the
    one reader, record into the one ``omitted`` list, hold their ``data:``
    URIs as the one set of placeholders, and stop at the one deadline. The
    asset loading that both use lives here too.
    """

    read_asset: AssetReader
    base_parts: tuple[str, ...]
    mode: InlineMode
    limits: InlineLimits
    deadline: float  # a time.monotonic() reading
    omitted: list[OmittedReference] = field(default_factory=list)
    references: int = 0
    total_bytes: int = 0
    cached_bytes: int = 0
    truncated: bool = False
    assets: dict[tuple[tuple[str, ...], AssetKind], _Asset] = field(default_factory=dict)
    # Why each file that failed to read or type was refused, so a repeated
    # reference costs no second read. Only the reason is kept: a refusal's
    # traceback would hold the bytes it refused.
    refusals: dict[tuple[tuple[str, ...], AssetKind], OmittedReason] = field(
        default_factory=dict
    )
    # Each inlined ``data:`` URI, which the tree holds as a placeholder until
    # the reparse is done.
    placeholders: UriPlaceholders = field(default_factory=UriPlaceholders)

    def out_of_time(self) -> bool:
        """Whether the deadline has passed; once it has, the output is cut short."""
        if time.monotonic() <= self.deadline:
            return False
        self.truncated = True
        return True

    def omit_unloaded(self, *, reference: str) -> None:
        """Record a reference the document loses without any attempt to read it."""
        cleaned = clean_reference(value=reference)
        if cleaned:
            self.omit(reference=cleaned, reason=_unloaded_reason(reference=cleaned))

    def omit(self, *, reference: str, reason: OmittedReason) -> None:
        if len(self.omitted) < _MAX_OMITTED:
            self.omitted.append(OmittedReference(reference=reference, reason=reason))

    def inline_asset(
        self, *, reference: str, base_parts: tuple[str, ...], kind: AssetKind
    ) -> _Asset | None:
        """The asset as a ``data:`` URI, or None after recording why it was left out."""
        cleaned = clean_reference(value=reference)
        if not cleaned:
            return None
        try:
            asset = self._load_asset(reference=cleaned, base_parts=base_parts, kind=kind)
            self.spend(size=asset.size)
        except ArtifactFileRefused as refusal:
            self.omit(reference=cleaned, reason=refusal.reason)
            return None
        return asset

    def _load_asset(
        self, *, reference: str, base_parts: tuple[str, ...], kind: AssetKind
    ) -> _Asset:
        if is_data_uri(reference=reference):
            self.count_reference()
            data = _decode_data_uri(reference=reference, max_bytes=self.limits.max_asset_bytes)
            return _typed_asset(data=data, kind=kind, limits=self.limits)
        parts = base_parts + reference_parts(reference=reference)
        self.count_reference()
        key = (parts, kind)
        reason = self.refusals.get(key)
        if reason is not None:
            raise ArtifactFileRefused(reason, "refused before")
        asset = self.assets.get(key)
        if asset is None:
            try:
                data = self.read(parts=parts).data
                asset = _typed_asset(data=data, kind=kind, limits=self.limits)
            except ArtifactFileRefused as refusal:
                self.refusals[key] = refusal.reason
                raise
            # Only what the budget could still hold is kept for reuse, so a
            # folder of large files can't fill memory with assets left out.
            if self.cached_bytes + asset.size <= self.limits.max_total_bytes:
                self.assets[key] = asset
                self.cached_bytes += asset.size
        return asset

    def count_reference(self) -> None:
        self.references += 1
        if self.references > self.limits.max_references:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "too many references")

    def spend(self, *, size: int) -> None:
        if self.total_bytes + size > self.limits.max_total_bytes:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "over the inline budget")
        self.total_bytes += size

    def read(self, *, parts: tuple[str, ...]) -> ArtifactFileContent:
        return self.read_asset(parts)


@dataclass
class _Inliner:
    """The walk over the HTML tree. The CSS it meets goes to ``css``."""

    state: InlineState
    css: CssRewriter
    scripts_dropped: bool = False
    title_kept: bool = False
    # Elements inside dropped subtrees scanned for what they would have loaded.
    dropped_scanned: int = 0

    def sanitized_html(self, *, html: bytes) -> bytes:
        """The sanitized document as UTF-8, with placeholders for its ``data:`` URIs.

        The parse tree is freed when this returns, before the caller parses
        the result again.
        """
        source = _parse_html(data=html)
        self.state.truncated = source.stopped
        self.sanitize(root=source.root)
        _declare_utf8(root=source.root)
        return _serialize_html(root=source.root).encode("utf-8")

    def sanitize(self, *, root: lxml.html.HtmlElement) -> None:
        """Walk the tree in document order with an explicit stack of kept elements.

        Each kept element's children are rebuilt once, after the walk has
        visited all of them. Past the node cap or the deadline the rest of the
        tree goes, tails and all, and ``truncated`` is set.
        """
        visit = self._visit(element=root, namespace=_Namespace.HTML, parent_tag="")
        frame = _element_frame(element=root, namespace=visit.namespace)
        if visit.action is not _Action.KEEP:
            # The root is always the kept ``html``; any other root keeps nothing.
            frame.pending.clear()
            frame.texts = [[]]
            frame.changed = True
        visited = 1
        stack = [frame]
        while stack:
            frame = stack[-1]
            if not frame.pending:
                frame.rebuild()
                stack.pop()
                continue
            item = frame.pending.pop()
            if isinstance(item, str):
                frame.texts[-1].append(item)
                continue
            visited += 1
            if visited > self.state.limits.max_nodes or self.state.out_of_time():
                self.state.truncated = True
                for open_frame in stack:
                    open_frame.pending.clear()
                    open_frame.changed = True
                continue
            visit = self._visit(element=item, namespace=frame.namespace, parent_tag=frame.element.tag)
            if visit.action is _Action.KEEP:
                frame.children.append(item)
                frame.texts.append([item.tail or ""])
                stack.append(_element_frame(element=item, namespace=visit.namespace))
                continue
            frame.changed = True
            if visit.action is _Action.UNWRAP:
                frame.texts[-1].append(item.text or "")
                frame.pending.append(item.tail or "")
                frame.pending.extend(reversed(item))
            else:
                frame.texts[-1].append(item.tail or "")
                self._omit_dropped_loads(element=item, namespace=visit.namespace)

    def _omit_dropped_loads(
        self, *, element: lxml.html.HtmlElement, namespace: _Namespace
    ) -> None:
        """List what the elements inside a dropped element would have loaded.

        The walk never visits them, so a ``<source>`` inside a ``<video>`` or
        an ``<img>`` inside a ``<noscript>`` is found here. The scan stops
        once ``omitted`` is full, past the node cap across every dropped
        subtree, or past the deadline; the subtree goes either way.
        """
        stack = [(child, namespace) for child in reversed(element)]
        while stack and len(self.state.omitted) < _MAX_OMITTED:
            if self.dropped_scanned >= self.state.limits.max_nodes or self.state.out_of_time():
                return
            self.dropped_scanned += 1
            node, node_namespace = stack.pop()
            if not isinstance(node.tag, str):
                continue
            self._omit_loads(element=node, namespace=node_namespace)
            inner = _child_namespace(tag=node.tag, namespace=node_namespace) or node_namespace
            stack.extend((child, inner) for child in reversed(node))

    def _visit(
        self, *, element: lxml.html.HtmlElement, namespace: _Namespace, parent_tag: str
    ) -> _Visit:
        """Decide *element*'s fate; *parent_tag* is the kept element it will end up in."""
        tag = element.tag
        if not isinstance(tag, str):
            return _Visit(_Action.DROP, namespace)
        if tag == "script":
            return self._visit_script(element=element, namespace=namespace)
        if tag == "title":
            if self.title_kept or parent_tag != "head":
                return _Visit(_Action.DROP, namespace)
            self.title_kept = True
        if tag == "link" and namespace is _Namespace.HTML and _links_a_stylesheet(element=element):
            return self._visit_stylesheet_link(element=element)
        if tag in _DROPPED_ELEMENTS or (tag == "style" and namespace is not _Namespace.HTML):
            self._omit_loads(element=element, namespace=namespace)
            if tag in _VOID_ELEMENTS:
                return _Visit(_Action.UNWRAP, namespace)
            return _Visit(_Action.DROP, namespace)
        child_namespace = _child_namespace(tag=tag, namespace=namespace)
        if child_namespace is None:
            self._omit_loads(element=element, namespace=namespace)
            return _Visit(_Action.UNWRAP, namespace)
        if tag == "img":
            sources = _pop_image_sources(element=element)
            self._sanitize_attributes(element=element, namespace=child_namespace)
            self._inline_image(element=element, sources=sources)
        else:
            self._sanitize_attributes(element=element, namespace=child_namespace)
        if tag == "style":
            element.text = self.css.rewrite_stylesheet(text=element.text or "") or None
        return _Visit(_Action.KEEP, child_namespace)

    def _visit_script(self, *, element: lxml.html.HtmlElement, namespace: _Namespace) -> _Visit:
        """PDF mode keeps inline scripts and pinned CDN scripts in HTML; other modes keep none.

        libxml2 reads every script as raw text, but a browser reads a script
        inside SVG or MathML as markup, where it can build elements, such as
        an image with an outside source, that this walk never sees. Every
        script that goes lists its ``src`` in ``omitted``, and an SVG script
        its ``href`` too.
        """
        src = element.attrib.pop("src", None)
        if self.state.mode is InlineMode.PDF and namespace is _Namespace.HTML:
            if src is None and _inline_script_is_stable(text=element.text or ""):
                self._sanitize_attributes(element=element, namespace=namespace)
                return _Visit(_Action.KEEP, namespace)
            if src is not None and _PINNED_CDN_SCRIPT.fullmatch(src):
                element.text = None
                self._sanitize_attributes(element=element, namespace=namespace)
                element.set("src", src)
                return _Visit(_Action.KEEP, namespace)
        if src is not None:
            self.state.omit_unloaded(reference=src)
        self._omit_loads(element=element, namespace=namespace)
        self.scripts_dropped = True
        return _Visit(_Action.DROP, namespace)

    def _visit_stylesheet_link(self, *, element: lxml.html.HtmlElement) -> _Visit:
        """Turn a ``link`` to the artifact's own stylesheet into a ``style`` holding it.

        The stylesheet splices as an ``@import`` of it would, under the same
        caps. When it can't, the ``link`` goes and its reference is recorded.
        """
        reference = clean_reference(value=element.attrib.pop("href", ""))
        css = None
        if reference and not reference.startswith("#"):
            css = self.css.linked_stylesheet(reference=reference)
        if css is None:
            self._omit_loads(element=element, namespace=_Namespace.HTML)
            return _Visit(_Action.UNWRAP, _Namespace.HTML)
        element.tag = "style"
        self._sanitize_attributes(element=element, namespace=_Namespace.HTML)
        element.text = css or None
        return _Visit(_Action.KEEP, _Namespace.HTML)

    def _sanitize_attributes(
        self, *, element: lxml.html.HtmlElement, namespace: _Namespace
    ) -> None:
        """Keep only the allowed attributes, with their rewritten values.

        An attribute whose name or kept value holds a character lxml refuses
        goes too. lxml can't delete such a name, so the attributes are cleared
        and the kept ones set again.
        """
        tag = element.tag
        attributes = list(element.attrib.items())
        kept = []
        for name, value in attributes:
            kept_value = self._kept_attribute(tag=tag, namespace=namespace, name=name, value=value)
            if kept_value is not None and not XML_UNSAFE.search(name + kept_value):
                kept.append((name, kept_value))
        if kept == attributes:
            return
        element.attrib.clear()
        for name, value in kept:
            element.set(name, value)

    def _kept_attribute(
        self, *, tag: str, namespace: _Namespace, name: str, value: str
    ) -> str | None:
        """The value *name* keeps on a kept element, or None to drop it."""
        if name.startswith("on"):
            return None
        if name == "style":
            return self.css.rewrite_style_attribute(value=value) or None
        if name in _HREF_ATTRIBUTES:
            return self._kept_href(tag=tag, namespace=namespace, value=value)
        if name in _LOAD_ATTRIBUTES:
            references = _srcset_candidates(value=value) if name == "srcset" else [value]
            for reference in references:
                self.state.omit_unloaded(reference=reference)
            return None
        if not _attribute_allowed(tag=tag, namespace=namespace, name=name):
            return None
        return value if self.css.attribute_is_inert(value=value) else None

    def _kept_href(self, *, tag: str, namespace: _Namespace, value: str) -> str | None:
        """A link keeps http, https, mailto or a fragment; SVG keeps fragments only."""
        cleaned = clean_reference(value=value)
        if tag == "a":
            if not (cleaned.startswith("#") or _NAVIGATION_SCHEME.match(cleaned)):
                return None
        elif namespace is not _Namespace.SVG:
            return None
        elif not cleaned.startswith("#"):
            self.state.omit_unloaded(reference=cleaned)
            return None
        return cleaned if self.css.attribute_is_inert(value=cleaned) else None

    def _omit_loads(self, *, element: lxml.html.HtmlElement, namespace: _Namespace) -> None:
        """List what a dropped or unwrapped element would have loaded."""
        for name, value in element.attrib.items():
            if name == "srcset":
                for candidate in _srcset_candidates(value=value):
                    self.state.omit_unloaded(reference=candidate)
            elif name in _LOAD_ATTRIBUTES:
                self.state.omit_unloaded(reference=value)
            elif name in _HREF_ATTRIBUTES and _href_loads(tag=element.tag, namespace=namespace):
                cleaned = clean_reference(value=value)
                if not cleaned.startswith("#"):
                    self.state.omit_unloaded(reference=cleaned)

    def _inline_image(self, *, element: lxml.html.HtmlElement, sources: _ImageSources) -> None:
        """Set ``src`` to the first source tried that inlines; with none, the image keeps no source.

        Every source the image doesn't use is listed: one tried and refused
        with its reason, and one never tried as a load nothing read. A
        candidate that names the source the image uses isn't listed.
        """
        unused = sources.untried
        used = None
        for index, candidate in enumerate(sources.tried):
            asset = self.state.inline_asset(
                reference=candidate, base_parts=self.state.base_parts, kind=AssetKind.IMAGE
            )
            if asset is not None:
                element.set("src", self.state.placeholders.hold(uri=asset.uri))
                used = clean_reference(value=candidate)
                unused = [*sources.tried[index + 1:], *sources.untried]
                break
        for reference in unused:
            if clean_reference(value=reference) != used:
                self.state.omit_unloaded(reference=reference)


def _parse_html(*, data: bytes) -> _ParsedHtml:
    """Parse *data* as an HTML document, never as XML.

    Bytes that are valid UTF-8 parse as UTF-8, since libxml2 reads a document
    with no charset declaration as windows-1252. Other bytes keep libxml2's
    own detection from a byte order mark or a ``meta`` charset. Past a
    resource limit, such as nesting deeper than 256 or about 10 MB of large
    tokens, libxml2 stops and the tree holds only what came before.
    """
    encoding = None
    if data.startswith(codecs.BOM_UTF8):
        data = data[len(codecs.BOM_UTF8):]
        encoding = "utf-8"
    elif _is_utf8(data=data):
        encoding = "utf-8"
    parser = lxml.html.HTMLParser(
        encoding=encoding,
        no_network=True,
        huge_tree=False,
        remove_comments=True,
        remove_pis=True,
    )
    try:
        root = lxml.html.document_fromstring(data, parser=parser)
    except ParserError:
        root = None
    except lxml.etree.XMLSyntaxError as failed:
        # Out of memory, libxml2 fails the parse rather than stopping it.
        parse_stopped(error_log=failed.error_log)
        raise
    stopped = parse_stopped(error_log=parser.error_log)
    if root is None:
        # Nothing but whitespace, or nothing before a limit: an empty document.
        root = lxml.html.document_fromstring(b"<html></html>", parser=parser)
    return _ParsedHtml(root=root, stopped=stopped)


# libxml2 2.12 and later report a parse stopped at a resource limit as
# ERR_RESOURCE_LIMIT. 2.11, which the Windows lxml wheel bundles, reports
# nesting past 256 as an internal error naming XML_PARSE_HUGE, a huge text
# node as out of memory and a huge attribute value as unfinished, so those
# messages count too.
_STOPPED_PARSE_MESSAGES = ("XML_PARSE_HUGE", "huge text node", "attribute value too long")


def parse_stopped(*, error_log: Iterable[lxml.etree._LogEntry]) -> bool:
    """Whether libxml2 stopped reading at a resource limit and dropped the rest.

    Raises ``MemoryError`` when libxml2 ran out of memory instead: the rest
    went for want of memory, not for size, so the document is refused rather
    than written short. Under the export's memory limit this is how an
    oversized parse ends. libxml2 2.11 logs its huge-text-node limit as
    ``ERR_NO_MEMORY`` too, so a known limit message decides first.
    """
    stopped = False
    for entry in error_log:
        if entry.type_name == "ERR_RESOURCE_LIMIT" or any(
            message in entry.message for message in _STOPPED_PARSE_MESSAGES
        ):
            stopped = True
        elif entry.type_name == "ERR_NO_MEMORY":
            raise MemoryError("libxml2 ran out of memory")
    return stopped


def _is_utf8(*, data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _serialize_html(*, root: lxml.html.HtmlElement) -> str:
    return lxml.html.tostring(root, encoding="unicode", method="html", doctype="<!DOCTYPE html>")


def _declare_utf8(*, root: lxml.html.HtmlElement) -> None:
    """Open the head with ``<meta charset="utf-8">``, since the output is text."""
    head = root.find("head")
    if head is None:
        head = root.makeelement("head", {})
        root.insert(0, head)
    head.insert(0, head.makeelement("meta", {"charset": "utf-8"}))


def _append_truncation_note(*, root: lxml.html.HtmlElement) -> None:
    """End the body with a paragraph saying the rest of the document was left out."""
    body = root.find("body")
    if body is None:
        body = root.makeelement("body", {})
        root.append(body)
    note = body.makeelement("p", {})
    note.text = TRUNCATION_NOTE
    body.append(note)


def _element_frame(*, element: lxml.html.HtmlElement, namespace: _Namespace) -> _ElementFrame:
    return _ElementFrame(
        element=element,
        namespace=namespace,
        pending=list(reversed(element)),
        texts=[[element.text or ""]],
    )


def _joined_text(*, pieces: list[str]) -> str | None:
    return XML_UNSAFE.sub("", "".join(pieces)) or None


def _child_namespace(*, tag: str, namespace: _Namespace) -> _Namespace | None:
    """The namespace a kept element's children are in, or None when it isn't kept."""
    if namespace is _Namespace.HTML:
        if tag == "svg":
            return _Namespace.SVG
        if tag == "math":
            return _Namespace.MATH
        return namespace if tag in _HTML_ELEMENTS else None
    allowed = _SVG_ELEMENTS if namespace is _Namespace.SVG else _MATH_ELEMENTS
    return namespace if tag in allowed else None


def _attribute_allowed(*, tag: str, namespace: _Namespace, name: str) -> bool:
    if name.startswith(_KEPT_ATTRIBUTE_PREFIXES):
        return True
    if namespace is _Namespace.HTML:
        return name in _HTML_GLOBAL_ATTRIBUTES or name in _HTML_ATTRIBUTES.get(tag, frozenset())
    if namespace is _Namespace.SVG:
        return name in _SVG_ATTRIBUTES
    return name in _MATH_ATTRIBUTES


def _href_loads(*, tag: str, namespace: _Namespace) -> bool:
    """Whether an ``href`` on this element loads a resource, rather than navigating."""
    if namespace is _Namespace.SVG:
        return tag != "a"
    return namespace is _Namespace.HTML and tag == "link"


def _links_a_stylesheet(*, element: lxml.html.HtmlElement) -> bool:
    """Whether a browser applies the stylesheet this ``link`` names.

    An alternate or disabled stylesheet isn't applied. libxml2 parses ``link``
    as empty, so it never has children to keep inside the ``style`` it becomes.
    """
    rel = _HTML_TOKEN_SEPARATORS.split(element.get("rel", "").lower())
    return (
        "stylesheet" in rel
        and "alternate" not in rel
        and element.get("disabled") is None
        and len(element) == 0
    )


def _pop_image_sources(*, element: lxml.html.HtmlElement) -> _ImageSources:
    """Take ``src`` and the ``srcset`` candidates off an image.

    ``src`` and then the first candidate are tried; the others never are.
    """
    src = element.attrib.pop("src", None)
    srcset = element.attrib.pop("srcset", None)
    element.attrib.pop("sizes", None)
    candidates = _srcset_candidates(value=srcset) if srcset is not None else []
    tried = [src] if src is not None else []
    return _ImageSources(tried=[*tried, *candidates[:1]], untried=candidates[1:])


def _srcset_candidates(*, value: str) -> list[str]:
    """The URL of each ``srcset`` candidate, as the HTML parsing rules split them.

    At most ``_MAX_OMITTED``, as many as ``omitted`` can hold.
    """
    urls: list[str] = []
    position = 0
    while len(urls) < _MAX_OMITTED and (match := _SRCSET_URL.match(value, position)):
        url = match.group(1)
        position = match.end()
        if url.endswith(","):
            # A URL that ends with commas has no descriptors.
            url = url.rstrip(",")
        else:
            position = _SRCSET_DESCRIPTORS.match(value, position).end()
        urls.append(url)
    return urls


def _inline_script_is_stable(*, text: str) -> bool:
    lowered = text.lower()
    return not any(marker in lowered for marker in _UNSTABLE_SCRIPT_MARKERS)


def clean_reference(*, value: str) -> str:
    """Strip control characters (TAB, LF and CR among them) and outer spaces."""
    return _CONTROL_CHARACTERS.sub("", value).strip(" ")


def is_data_uri(*, reference: str) -> bool:
    return reference[:5].lower() == "data:"


def reference_parts(*, reference: str) -> tuple[str, ...]:
    """The path segments a cleaned relative reference names, in the refusal order.

    A scheme, then a leading slash or backslash, are refused first. The query
    and fragment are cut, and the rest is percent-decoded once.
    ``split_relative_path`` then splits it by the rules every relative path
    follows, skipping ``.`` segments, which name the current folder, and a
    segment holding ``:`` is refused. Reserved device names, a trailing dot or
    space and housekeeping names are refused by the pinned file walk, after
    these.
    """
    if _SCHEME.match(reference):
        raise ArtifactFileRefused(OmittedReason.SCHEME, "has a scheme")
    if reference.startswith(("/", "\\")):
        raise ArtifactFileRefused(OmittedReason.ABSOLUTE, "leading separator")
    path = _QUERY_OR_FRAGMENT.split(reference, maxsplit=1)[0]
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeDecodeError as exc:
        raise ArtifactFileRefused(OmittedReason.MISSING, "not UTF-8") from exc
    parts = split_relative_path(text=decoded, skip_dot_segments=True)
    if any(":" in segment for segment in parts):
        raise ArtifactFileRefused(OmittedReason.TRAVERSAL, "segment is not a plain name")
    return parts


def _unloaded_reason(*, reference: str) -> OmittedReason:
    """Why a reference nothing reads was left out: its path's refusal, else its type."""
    if is_data_uri(reference=reference):
        return OmittedReason.UNSUPPORTED_TYPE
    try:
        reference_parts(reference=reference)
    except ArtifactFileRefused as refusal:
        return refusal.reason
    return OmittedReason.UNSUPPORTED_TYPE


def _decode_data_uri(*, reference: str, max_bytes: int) -> bytes:
    match = _DATA_URI.fullmatch(reference)
    if match is None:
        raise ArtifactFileRefused(OmittedReason.UNSUPPORTED_TYPE, "malformed data: URI")
    header, payload = match.groups()
    if len(payload) > 4 * max_bytes + 4:
        raise ArtifactFileRefused(OmittedReason.TOO_LARGE, "data: URI over the asset cap")
    data = unquote_to_bytes(payload)
    if header.lower().endswith(";base64"):
        compact = _BASE64_WHITESPACE.sub(b"", data).rstrip(b"=")
        try:
            data = base64.b64decode(compact + b"=" * (-len(compact) % 4), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ArtifactFileRefused(OmittedReason.UNSUPPORTED_TYPE, "bad base64") from exc
    if len(data) > max_bytes:
        raise ArtifactFileRefused(OmittedReason.TOO_LARGE, "data: URI over the asset cap")
    return data


def _typed_asset(*, data: bytes, kind: AssetKind, limits: InlineLimits) -> _Asset:
    """Name the bytes' type from their signature, never from a file name."""
    if kind is AssetKind.FONT:
        mime = _font_type(data=data)
        if mime is None:
            raise ArtifactFileRefused(OmittedReason.UNSUPPORTED_TYPE, "not a font")
    else:
        image = _image_header(data=data)
        if image is None:
            raise ArtifactFileRefused(OmittedReason.UNSUPPORTED_TYPE, "not a raster image")
        if image.width * image.height > limits.max_image_pixels:
            raise ArtifactFileRefused(OmittedReason.TOO_LARGE, "too many pixels")
        mime = image.mime
    encoded = base64.b64encode(data).decode("ascii")
    return _Asset(uri=f"data:{mime};base64,{encoded}", size=len(data))


def _font_type(*, data: bytes) -> str | None:
    for signature, mime in _FONT_SIGNATURES:
        if data.startswith(signature):
            return mime
    return None


def _image_header(*, data: bytes) -> _ImageHeader | None:
    """The type and size a PNG, GIF, JPEG or WebP header declares, with no decoding."""
    header = None
    if data.startswith(PNG_SIGNATURE) and data[12:16] == b"IHDR" and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        header = _ImageHeader("image/png", width, height)
    elif data.startswith(GIF_SIGNATURES) and len(data) >= 10:
        width = int.from_bytes(data[6:8], "little")
        height = int.from_bytes(data[8:10], "little")
        header = _ImageHeader("image/gif", width, height)
    elif data.startswith(JPEG_SIGNATURE):
        header = _jpeg_header(data=data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        header = _webp_header(data=data)
    if header is None or header.width <= 0 or header.height <= 0:
        return None
    return header


def _jpeg_header(*, data: bytes) -> _ImageHeader | None:
    """Walk the JPEG segments to the first frame header."""
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            return None
        marker = data[position + 1]
        if marker == 0xFF:
            position += 1
            continue
        if marker in JPEG_STANDALONE_MARKERS:
            position += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        if marker in _JPEG_FRAME_MARKERS:
            if position + 9 > len(data):
                return None
            height = int.from_bytes(data[position + 5:position + 7], "big")
            width = int.from_bytes(data[position + 7:position + 9], "big")
            return _ImageHeader("image/jpeg", width, height)
        length = int.from_bytes(data[position + 2:position + 4], "big")
        if length < 2:
            return None
        position += 2 + length
    return None


def _webp_header(*, data: bytes) -> _ImageHeader | None:
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return _ImageHeader("image/webp", width, height)
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return _ImageHeader("image/webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return _ImageHeader("image/webp", width, height)
    return None
