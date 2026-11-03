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
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import NamedTuple
from urllib.parse import unquote, unquote_to_bytes

import lxml.etree
import lxml.html
import tinycss2
from lxml.etree import ParserError
from tinycss2.ast import LiteralToken, Node, URLToken, WhitespaceToken
from tinycss2.serializer import serialize_identifier

from cowork.common.paths import PinnedDir
from cowork.schemas.artifact_export import (
    InlinedDocument,
    InlineLimits,
    InlineMode,
    OmittedReason,
    OmittedReference,
)
from cowork.services.artifact_files import (
    ArtifactFileContent,
    ArtifactFileRefused,
    read_artifact_file_entry,
    split_relative_path,
)

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

# CSS functions whose argument names a resource: ``url()`` and ``src()``
# take one string; in ``image-set()`` and ``image()`` a bare string is a URL.
_URL_FUNCTIONS = frozenset({"url", "src"})
_STRING_URL_FUNCTIONS = frozenset({"image-set", "-webkit-image-set", "image"})
_LOADING_FUNCTIONS = (
    _URL_FUNCTIONS | _STRING_URL_FUNCTIONS | frozenset({"cross-fade", "-webkit-cross-fade"})
)
# Inside a loading function these yield a value only a browser computes, so
# the URL they make can't be checked here.
_SUBSTITUTION_FUNCTIONS = frozenset({"var", "env", "attr", "if", "inherit"})
_CSS_BLOCK_AT_RULES = frozenset(
    {
        "media", "supports", "container", "layer", "page", "font-face",
        "keyframes", "-webkit-keyframes",
        "top-left-corner", "top-left", "top-center", "top-right",
        "top-right-corner", "bottom-left-corner", "bottom-left",
        "bottom-center", "bottom-right", "bottom-right-corner", "left-top",
        "left-middle", "left-bottom", "right-top", "right-middle",
        "right-bottom",
    }
)
_CSS_BLOCKS = frozenset({"() block", "[] block", "{} block"})
# tinycss2 serializes nested blocks recursively, so nesting stays shallow.
_MAX_CSS_NESTING = 32
# A ``>`` delimiter token (a child combinator or a media range) stands in as
# NUL while the CSS is serialized: tinycss2 turns every NUL in its input into
# U+FFFD, so a NUL in the output can only be this marker.
_GREATER_THAN_DELIM = "\x00"
_CSS_UNSAFE = re.compile(r"\\(.)|([<>\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff])", re.DOTALL)
_CSS_HEX_ESCAPED = re.compile(r"[<>\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")

# The omitted list keeps at most this many entries, so a hostile document
# can't grow it without bound. The drops past it still happen.
_MAX_OMITTED = 1000
# tinycss2 keeps every token as a Python object, which takes about 130 times
# the text's size in memory. So a document parses at most this much CSS in
# all: its style elements, its style attributes, every other kept attribute
# whose value holds "(", which could hide an escaped url(, and the
# stylesheets it links or imports. Past it, inline CSS and each such
# attribute are dropped, and a stylesheet is left out as BUDGET.
_MAX_CSS_BYTES = 1024 * 1024


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


class _AssetKind(Enum):
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


class _CssReference(NamedTuple):
    """One resource a CSS value names, and where its token sits."""

    items: list[Node]
    index: int
    url: str


class _CssScan(NamedTuple):
    references: list[_CssReference]
    checkable: bool


class _ScanFrame(NamedTuple):
    """One list of nodes on the explicit stack of ``_scan_css``."""

    items: list[Node]
    function: str  # the lowercased name of the function directly around the list, or ""
    depth: int
    loading: bool  # whether a loading function encloses the list


class _LoadedStylesheet(NamedTuple):
    parts: tuple[str, ...]
    content: ArtifactFileContent


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
class _CssFrame:
    """One rule list being rewritten, on the explicit stack of ``_run_css``."""

    nodes: list[Node]
    base_parts: tuple[str, ...]
    import_depth: int
    chain: frozenset[tuple[int, int]]
    nesting: int
    top_level: bool
    font_face: bool
    pieces: list[str]
    closing: str
    index: int = 0


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
    inliner = _Inliner(
        folder=folder,
        base_parts=base_parts,
        mode=mode,
        limits=limits,
        deadline=time.monotonic() + DEADLINE_SECONDS,
    )
    # Removing an unknown wrapper can leave a nesting the parser builds
    # differently (a block inside a paragraph, say), so the output is what
    # one more parse makes of it: the structure every later parse rebuilds.
    reparsed = _parse_html(data=inliner.sanitized_html(html=html))
    if inliner.truncated or reparsed.stopped:
        _append_truncation_note(root=reparsed.root)
    return InlinedDocument(
        html=inliner.placeholders.restore(text=_serialize_html(root=reparsed.root)),
        omitted=inliner.omitted,
        inlined_bytes=inliner.total_bytes,
        scripts_dropped=inliner.scripts_dropped,
    )


@dataclass
class _Inliner:
    folder: PinnedDir
    base_parts: tuple[str, ...]
    mode: InlineMode
    limits: InlineLimits
    deadline: float  # a time.monotonic() reading
    omitted: list[OmittedReference] = field(default_factory=list)
    references: int = 0
    total_bytes: int = 0
    cached_bytes: int = 0
    css_bytes: int = 0
    scripts_dropped: bool = False
    title_kept: bool = False
    truncated: bool = False
    # Elements inside dropped subtrees scanned for what they would have loaded.
    dropped_scanned: int = 0
    assets: dict[tuple[tuple[str, ...], _AssetKind], _Asset] = field(default_factory=dict)
    # Why each file that failed to read or type was refused, so a repeated
    # reference costs no second read. Only the reason is kept: a refusal's
    # traceback would hold the bytes it refused.
    refusals: dict[tuple[tuple[str, ...], _AssetKind], OmittedReason] = field(
        default_factory=dict
    )
    # Each inlined ``data:`` URI, which the tree holds as a placeholder until
    # the reparse is done.
    placeholders: UriPlaceholders = field(default_factory=UriPlaceholders)

    def sanitized_html(self, *, html: bytes) -> bytes:
        """The sanitized document as UTF-8, with placeholders for its ``data:`` URIs.

        The parse tree is freed when this returns, before the caller parses
        the result again.
        """
        source = _parse_html(data=html)
        self.truncated = source.stopped
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
            if visited > self.limits.max_nodes or self._out_of_time():
                self.truncated = True
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
        while stack and len(self.omitted) < _MAX_OMITTED:
            if self.dropped_scanned >= self.limits.max_nodes or self._out_of_time():
                return
            self.dropped_scanned += 1
            node, node_namespace = stack.pop()
            if not isinstance(node.tag, str):
                continue
            self._omit_loads(element=node, namespace=node_namespace)
            inner = _child_namespace(tag=node.tag, namespace=node_namespace) or node_namespace
            stack.extend((child, inner) for child in reversed(node))

    def _out_of_time(self) -> bool:
        """Whether the deadline has passed; once it has, the output is cut short."""
        if time.monotonic() <= self.deadline:
            return False
        self.truncated = True
        return True

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
            element.text = self._rewrite_stylesheet(text=element.text or "") or None
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
        if self.mode is InlineMode.PDF and namespace is _Namespace.HTML:
            if src is None and _inline_script_is_stable(text=element.text or ""):
                self._sanitize_attributes(element=element, namespace=namespace)
                return _Visit(_Action.KEEP, namespace)
            if src is not None and _PINNED_CDN_SCRIPT.fullmatch(src):
                element.text = None
                self._sanitize_attributes(element=element, namespace=namespace)
                element.set("src", src)
                return _Visit(_Action.KEEP, namespace)
        if src is not None:
            self._omit_unloaded(reference=src)
        self._omit_loads(element=element, namespace=namespace)
        self.scripts_dropped = True
        return _Visit(_Action.DROP, namespace)

    def _visit_stylesheet_link(self, *, element: lxml.html.HtmlElement) -> _Visit:
        """Turn a ``link`` to the artifact's own stylesheet into a ``style`` holding it.

        The stylesheet splices as an ``@import`` of it would, under the same
        caps. When it can't, the ``link`` goes and its reference is recorded.
        """
        reference = _clean_reference(value=element.attrib.pop("href", ""))
        css = None
        if reference and not reference.startswith("#"):
            css = self._linked_stylesheet(reference=reference)
        if css is None:
            self._omit_loads(element=element, namespace=_Namespace.HTML)
            return _Visit(_Action.UNWRAP, _Namespace.HTML)
        element.tag = "style"
        self._sanitize_attributes(element=element, namespace=_Namespace.HTML)
        element.text = css or None
        return _Visit(_Action.KEEP, _Namespace.HTML)

    def _linked_stylesheet(self, *, reference: str) -> str | None:
        """The rewritten stylesheet *reference* names, or None after recording why not."""
        frame = self._spliced_frame(
            reference=reference, frame=self._root_css_frame(nodes=[]), media=[]
        )
        if frame is None:
            return None
        return _escape_css(text=self._run_css(root=frame))

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
            return self._rewrite_style_attribute(value=value) or None
        if name in _HREF_ATTRIBUTES:
            return self._kept_href(tag=tag, namespace=namespace, value=value)
        if name in _LOAD_ATTRIBUTES:
            references = _srcset_candidates(value=value) if name == "srcset" else [value]
            for reference in references:
                self._omit_unloaded(reference=reference)
            return None
        if not _attribute_allowed(tag=tag, namespace=namespace, name=name):
            return None
        return value if self._attribute_is_inert(value=value) else None

    def _kept_href(self, *, tag: str, namespace: _Namespace, value: str) -> str | None:
        """A link keeps http, https, mailto or a fragment; SVG keeps fragments only."""
        cleaned = _clean_reference(value=value)
        if tag == "a":
            if not (cleaned.startswith("#") or _NAVIGATION_SCHEME.match(cleaned)):
                return None
        elif namespace is not _Namespace.SVG:
            return None
        elif not cleaned.startswith("#"):
            self._omit_unloaded(reference=cleaned)
            return None
        return cleaned if self._attribute_is_inert(value=cleaned) else None

    def _attribute_is_inert(self, *, value: str) -> bool:
        """Whether *value* names no resource through a CSS ``url(``.

        Only ``url(#fragment)`` may stay, which names something in this
        document, as SVG's ``fill`` or ``filter`` do.
        """
        if "(" not in value:
            return True
        if not self._take_css(size=len(value)):
            return False
        nodes = tinycss2.parse_component_value_list(value, skip_comments=True)
        scan = _scan_css(nodes=nodes, strict=False)
        if not scan.checkable:
            return False
        for reference in scan.references:
            cleaned = _clean_reference(value=reference.url)
            if not cleaned.startswith("#"):
                self._omit_unloaded(reference=cleaned)
                return False
        return True

    def _omit_loads(self, *, element: lxml.html.HtmlElement, namespace: _Namespace) -> None:
        """List what a dropped or unwrapped element would have loaded."""
        for name, value in element.attrib.items():
            if name == "srcset":
                for candidate in _srcset_candidates(value=value):
                    self._omit_unloaded(reference=candidate)
            elif name in _LOAD_ATTRIBUTES:
                self._omit_unloaded(reference=value)
            elif name in _HREF_ATTRIBUTES and _href_loads(tag=element.tag, namespace=namespace):
                cleaned = _clean_reference(value=value)
                if not cleaned.startswith("#"):
                    self._omit_unloaded(reference=cleaned)

    def _omit_unloaded(self, *, reference: str) -> None:
        """Record a reference the document loses without any attempt to read it."""
        cleaned = _clean_reference(value=reference)
        if cleaned:
            self._omit(reference=cleaned, reason=_unloaded_reason(reference=cleaned))

    def _omit(self, *, reference: str, reason: OmittedReason) -> None:
        if len(self.omitted) < _MAX_OMITTED:
            self.omitted.append(OmittedReference(reference=reference, reason=reason))

    def _inline_image(self, *, element: lxml.html.HtmlElement, sources: _ImageSources) -> None:
        """Set ``src`` to the first source tried that inlines; with none, the image keeps no source.

        Every source the image doesn't use is listed: one tried and refused
        with its reason, and one never tried as a load nothing read. A
        candidate that names the source the image uses isn't listed.
        """
        unused = sources.untried
        used = None
        for index, candidate in enumerate(sources.tried):
            asset = self._inline_asset(
                reference=candidate, base_parts=self.base_parts, kind=_AssetKind.IMAGE
            )
            if asset is not None:
                element.set("src", self.placeholders.hold(uri=asset.uri))
                used = _clean_reference(value=candidate)
                unused = [*sources.tried[index + 1:], *sources.untried]
                break
        for reference in unused:
            if _clean_reference(value=reference) != used:
                self._omit_unloaded(reference=reference)

    def _inline_asset(
        self, *, reference: str, base_parts: tuple[str, ...], kind: _AssetKind
    ) -> _Asset | None:
        """The asset as a ``data:`` URI, or None after recording why it was left out."""
        cleaned = _clean_reference(value=reference)
        if not cleaned:
            return None
        try:
            asset = self._load_asset(reference=cleaned, base_parts=base_parts, kind=kind)
            self._spend(size=asset.size)
        except ArtifactFileRefused as refusal:
            self._omit(reference=cleaned, reason=refusal.reason)
            return None
        return asset

    def _load_asset(
        self, *, reference: str, base_parts: tuple[str, ...], kind: _AssetKind
    ) -> _Asset:
        if _is_data_uri(reference=reference):
            self._count_reference()
            data = _decode_data_uri(reference=reference, max_bytes=self.limits.max_asset_bytes)
            return _typed_asset(data=data, kind=kind, limits=self.limits)
        parts = base_parts + _reference_parts(reference=reference)
        self._count_reference()
        key = (parts, kind)
        reason = self.refusals.get(key)
        if reason is not None:
            raise ArtifactFileRefused(reason, "refused before")
        asset = self.assets.get(key)
        if asset is None:
            try:
                data = self._read(parts=parts).data
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

    def _count_reference(self) -> None:
        self.references += 1
        if self.references > self.limits.max_references:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "too many references")

    def _spend(self, *, size: int) -> None:
        if self.total_bytes + size > self.limits.max_total_bytes:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "over the inline budget")
        self.total_bytes += size

    def _take_css(self, *, size: int) -> bool:
        """Count *size* more units of CSS to parse; False once the document's CSS cap is spent."""
        if self.css_bytes + size > _MAX_CSS_BYTES:
            return False
        self.css_bytes += size
        return True

    def _read(self, *, parts: tuple[str, ...]) -> ArtifactFileContent:
        try:
            return read_artifact_file_entry(
                folder=self.folder, parts=parts, max_bytes=self.limits.max_asset_bytes
            )
        except (OSError, ValueError) as exc:
            raise ArtifactFileRefused(OmittedReason.MISSING, "read failed") from exc

    def _rewrite_style_attribute(self, *, value: str) -> str:
        if not self._take_css(size=len(value)):
            return ""
        pieces = []
        nodes = tinycss2.parse_blocks_contents(value, skip_comments=True, skip_whitespace=True)
        for node in nodes:
            if self._out_of_time():
                break
            if node.type == "declaration" and self._rewrite_declaration(
                declaration=node, base_parts=self.base_parts, kind=_AssetKind.IMAGE
            ):
                pieces.append(_serialize_css(nodes=[node]))
        return _escape_css(text="".join(pieces))

    def _rewrite_stylesheet(self, *, text: str) -> str:
        if not self._take_css(size=len(text)):
            return ""
        nodes = tinycss2.parse_stylesheet(text, skip_comments=True, skip_whitespace=True)
        return _escape_css(text=self._run_css(root=self._root_css_frame(nodes=nodes)))

    def _root_css_frame(self, *, nodes: list[Node]) -> _CssFrame:
        """The frame for a stylesheet in the document itself."""
        return _CssFrame(
            nodes=nodes,
            base_parts=self.base_parts,
            import_depth=0,
            chain=frozenset(),
            nesting=0,
            top_level=True,
            font_face=False,
            pieces=[],
            closing="",
        )

    def _run_css(self, *, root: _CssFrame) -> str:
        """Rewrite a rule list, splicing imports, with an explicit stack.

        Past the deadline the rest of every open rule list goes, and each
        block still closes.
        """
        stack = [root]
        while True:
            frame = stack[-1]
            if frame.index == len(frame.nodes):
                stack.pop()
                text = "".join(frame.pieces) + frame.closing
                if not stack:
                    return text
                stack[-1].pieces.append(text)
                continue
            if self._out_of_time():
                for open_frame in stack:
                    open_frame.index = len(open_frame.nodes)
                continue
            node = frame.nodes[frame.index]
            frame.index += 1
            child = self._css_node(node=node, frame=frame)
            if child is not None:
                stack.append(child)

    def _css_node(self, *, node: Node, frame: _CssFrame) -> _CssFrame | None:
        """Write *node* into *frame*, or return the frame that rewrites its contents."""
        if node.type == "declaration":
            font = frame.font_face and node.lower_name == "src"
            kind = _AssetKind.FONT if font else _AssetKind.IMAGE
            if self._rewrite_declaration(declaration=node, base_parts=frame.base_parts, kind=kind):
                frame.pieces.append(_serialize_css(nodes=[node]))
            return None
        if node.type == "qualified-rule":
            if frame.nesting >= _MAX_CSS_NESTING or not _css_is_inert(nodes=node.prelude):
                return None
            return _css_block(
                frame=frame,
                opening=_serialize_css(nodes=node.prelude) + "{",
                content=node.content,
                font_face=False,
            )
        if node.type != "at-rule":
            return None
        keyword = node.lower_at_keyword
        if keyword == "import":
            return self._css_import(rule=node, frame=frame) if frame.top_level else None
        opening = "@" + serialize_identifier(node.at_keyword)
        if node.content is None:
            if keyword == "layer" and _css_is_inert(nodes=node.prelude):
                frame.pieces.append(opening + _serialize_css(nodes=node.prelude) + ";")
            return None
        if (
            keyword not in _CSS_BLOCK_AT_RULES
            or frame.nesting >= _MAX_CSS_NESTING
            or not _css_is_inert(nodes=node.prelude)
        ):
            return None
        media = keyword in ("media", "container")
        return _css_block(
            frame=frame,
            opening=opening + _serialize_css(nodes=node.prelude, media=media) + "{",
            content=node.content,
            font_face=keyword == "font-face",
        )

    def _rewrite_declaration(
        self, *, declaration: Node, base_parts: tuple[str, ...], kind: _AssetKind
    ) -> bool:
        """Inline every resource the declaration names; False drops the declaration."""
        scan = _scan_css(nodes=declaration.value, strict=True)
        if not scan.checkable:
            return False
        spent = 0
        kept_uris = len(self.placeholders.uris)
        for reference in scan.references:
            cleaned = _clean_reference(value=reference.url)
            if cleaned.startswith("#"):
                continue
            asset = self._inline_asset(reference=cleaned, base_parts=base_parts, kind=kind)
            if asset is None:
                # The declaration goes, so the URIs it placed go with it.
                self.total_bytes -= spent
                del self.placeholders.uris[kept_uris:]
                return False
            spent += asset.size
            # The placeholder is letters, digits and a hyphen, so it needs no
            # escape in an unquoted url().
            placeholder = self.placeholders.hold(uri=asset.uri)
            original = reference.items[reference.index]
            reference.items[reference.index] = URLToken(
                original.source_line, original.source_column, placeholder, f"url({placeholder})"
            )
        return True

    def _css_import(self, *, rule: Node, frame: _CssFrame) -> _CssFrame | None:
        """Splice the imported stylesheet in place of its ``@import``."""
        prelude = _strip_css_whitespace(nodes=rule.prelude)
        url = _import_url(prelude=prelude)
        if url is None:
            return None
        condition = _strip_css_whitespace(nodes=prelude[1:])
        wrapped = bool(condition) and _is_media_condition(condition=condition)
        return self._spliced_frame(reference=url, frame=frame, media=condition if wrapped else [])

    def _spliced_frame(
        self, *, reference: str, frame: _CssFrame, media: list[Node]
    ) -> _CssFrame | None:
        """The frame that rewrites the stylesheet *reference* names, inside *frame*.

        A non-empty *media* query wraps it in an ``@media`` block. None means
        the stylesheet was left out, and why is recorded.
        """
        cleaned = _clean_reference(value=reference)
        if not cleaned:
            return None
        try:
            stylesheet = self._load_stylesheet(reference=cleaned, frame=frame)
        except ArtifactFileRefused as refusal:
            self._omit(reference=cleaned, reason=refusal.reason)
            return None
        content = stylesheet.content
        rules, _encoding = tinycss2.parse_stylesheet_bytes(
            content.data, skip_comments=True, skip_whitespace=True
        )
        return _CssFrame(
            nodes=rules,
            base_parts=stylesheet.parts[:-1],
            import_depth=frame.import_depth + 1,
            chain=frame.chain | {(content.device, content.inode)},
            nesting=frame.nesting + 1 if media else frame.nesting,
            top_level=True,
            font_face=False,
            pieces=["@media " + _serialize_css(nodes=media, media=True) + "{" if media else ""],
            closing="}" if media else "",
        )

    def _load_stylesheet(self, *, reference: str, frame: _CssFrame) -> _LoadedStylesheet:
        if _is_data_uri(reference=reference):
            raise ArtifactFileRefused(OmittedReason.UNSUPPORTED_TYPE, "data: stylesheet")
        parts = frame.base_parts + _reference_parts(reference=reference)
        self._count_reference()
        if frame.import_depth >= self.limits.max_import_depth:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "imports nest too deep")
        content = self._read(parts=parts)
        if (content.device, content.inode) in frame.chain:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "import cycle")
        if not self._take_css(size=len(content.data)):
            raise ArtifactFileRefused(OmittedReason.BUDGET, "over the CSS cap")
        self._spend(size=len(content.data))
        return _LoadedStylesheet(parts=parts, content=content)


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
    """Whether libxml2 stopped reading at a resource limit and dropped the rest."""
    return any(
        entry.type_name == "ERR_RESOURCE_LIMIT"
        or any(message in entry.message for message in _STOPPED_PARSE_MESSAGES)
        for entry in error_log
    )


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


def _clean_reference(*, value: str) -> str:
    """Strip control characters (TAB, LF and CR among them) and outer spaces."""
    return _CONTROL_CHARACTERS.sub("", value).strip(" ")


def _is_data_uri(*, reference: str) -> bool:
    return reference[:5].lower() == "data:"


def _reference_parts(*, reference: str) -> tuple[str, ...]:
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
    if _is_data_uri(reference=reference):
        return OmittedReason.UNSUPPORTED_TYPE
    try:
        _reference_parts(reference=reference)
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


def _typed_asset(*, data: bytes, kind: _AssetKind, limits: InlineLimits) -> _Asset:
    """Name the bytes' type from their signature, never from a file name."""
    if kind is _AssetKind.FONT:
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


def _scan_css(*, nodes: list[Node], strict: bool) -> _CssScan:
    """Find every resource *nodes* name, in source order, with an explicit stack.

    The scan is not checkable when nesting runs past the cap, when a loading
    function takes a value from ``var()``, ``env()`` or ``attr()``, or, when
    *strict*, when tinycss2 reported a parse error in the value.
    """
    unchecked = _CssScan(references=[], checkable=False)
    references: list[_CssReference] = []
    stack = [_ScanFrame(items=nodes, function="", depth=0, loading=False)]
    while stack:
        frame = stack.pop()
        if frame.depth > _MAX_CSS_NESTING:
            return unchecked
        items = frame.items
        for index, node in enumerate(items):
            if node.type == "error":
                if strict:
                    return unchecked
            elif node.type == "url":
                references.append(_CssReference(items, index, node.value))
            elif node.type == "string":
                if frame.function in _STRING_URL_FUNCTIONS:
                    references.append(_CssReference(items, index, node.value))
            elif node.type == "function":
                name = node.lower_name
                if frame.loading and name in _SUBSTITUTION_FUNCTIONS:
                    return unchecked
                if name in _URL_FUNCTIONS:
                    url = _first_string(nodes=node.arguments)
                    references.append(_CssReference(items, index, url))
                stack.append(
                    _ScanFrame(
                        items=node.arguments,
                        function=name,
                        depth=frame.depth + 1,
                        loading=frame.loading or name in _LOADING_FUNCTIONS,
                    )
                )
            elif node.type in _CSS_BLOCKS:
                stack.append(
                    _ScanFrame(
                        items=node.content, function="", depth=frame.depth + 1, loading=frame.loading
                    )
                )
    references.sort(
        key=lambda ref: (ref.items[ref.index].source_line, ref.items[ref.index].source_column)
    )
    return _CssScan(references=references, checkable=True)


def _css_is_inert(*, nodes: list[Node]) -> bool:
    scan = _scan_css(nodes=nodes, strict=True)
    return scan.checkable and not scan.references


def _first_string(*, nodes: list[Node]) -> str:
    return next((node.value for node in nodes if node.type == "string"), "")


def _strip_css_whitespace(*, nodes: list[Node]) -> list[Node]:
    start = 0
    end = len(nodes)
    while start < end and nodes[start].type == "whitespace":
        start += 1
    while end > start and nodes[end - 1].type == "whitespace":
        end -= 1
    return nodes[start:end]


def _import_url(*, prelude: list[Node]) -> str | None:
    """The URL an ``@import`` prelude names first, or None when it names none."""
    if not prelude:
        return None
    target = prelude[0]
    if target.type in ("url", "string"):
        return target.value
    if target.type == "function" and target.lower_name in _URL_FUNCTIONS:
        return _first_string(nodes=target.arguments)
    return None


def _is_media_condition(*, condition: list[Node]) -> bool:
    """Whether an import's condition is a media query an ``@media`` block can carry.

    A ``layer`` or ``supports()`` condition isn't, so that import splices in
    unconditionally.
    """
    if not _css_is_inert(nodes=condition):
        return False
    return not any(
        (node.type == "ident" and node.lower_value == "layer")
        or (node.type == "function" and node.lower_name in ("layer", "supports"))
        for node in condition
    )


def _css_block(
    *, frame: _CssFrame, opening: str, content: list[Node], font_face: bool
) -> _CssFrame:
    return _CssFrame(
        nodes=tinycss2.parse_blocks_contents(content, skip_whitespace=True),
        base_parts=frame.base_parts,
        import_depth=frame.import_depth,
        chain=frame.chain,
        nesting=frame.nesting + 1,
        top_level=False,
        font_face=font_face,
        pieces=[opening],
        closing="}",
    )


def _serialize_css(*, nodes: list[Node], media: bool = False) -> str:
    """Serialize *nodes*, marking each ``>`` delimiter for ``_escape_css`` to keep.

    With *media*, *nodes* are a media query, and each range in it written with
    ``<`` is turned round first, so that it survives the escape.
    """
    stack = [nodes]
    while stack:
        items = stack.pop()
        for index, node in enumerate(items):
            if node.type == "literal" and node.value == ">":
                items[index] = LiteralToken(
                    node.source_line, node.source_column, _GREATER_THAN_DELIM
                )
            elif node.type == "declaration":
                stack.append(node.value)
            elif node.type == "function":
                stack.append(node.arguments)
            elif node.type in _CSS_BLOCKS:
                if media and node.type == "() block":
                    node.content = _turned_range(content=node.content)
                stack.append(node.content)
    return tinycss2.serialize(nodes)


def _turned_range(*, content: list[Node]) -> list[Node]:
    """A media feature range written with ``<``, turned round to use ``>``.

    ``600px <= width < 900px`` becomes ``900px > width >= 600px``, which
    means the same. ``_escape_css`` escapes every ``<``, and an escaped one
    leaves a query no browser matches. Content with no ``<``, or with a
    ``>`` too, is no range this can turn, and comes back as it was.
    """
    delimiters = {node.value for node in content if node.type == "literal"}
    if "<" not in delimiters or ">" in delimiters:
        return content
    terms: list[list[Node]] = [[]]
    operators: list[list[Node]] = []
    for node in content:
        if node.type == "literal" and node.value == "<":
            operators.append([LiteralToken(node.source_line, node.source_column, ">")])
            terms.append([])
        elif node.type == "literal" and node.value == "=" and operators and not terms[-1]:
            operators[-1].append(node)  # the "=" of "<="
        else:
            terms[-1].append(node)
    turned: list[Node] = []
    for term, operator in zip(reversed(terms), [*reversed(operators), []], strict=True):
        turned.extend(_strip_css_whitespace(nodes=term))
        if operator:
            space = WhitespaceToken(operator[0].source_line, operator[0].source_column, " ")
            turned.extend([space, *operator, space])
    return turned


def _escape_css(*, text: str) -> str:
    """Rewrite every ``<`` and ``>`` in serialized CSS as ``\\3c `` and ``\\3e ``.

    So the CSS can't end its ``style`` element or be read again as markup. A
    ``>`` delimiter (a child combinator or a media range) stays: no escape can
    stand for it, and it can't open markup on its own. A ``<`` delimiter has
    no escape either, so ``_serialize_css`` turns a media range written with
    one round to use ``>``; any other ``<`` delimiter is escaped, and a rule
    holding one no longer applies. Control characters an
    HTML document can't hold become hex escapes too. An escape already in the
    text is read as a pair, so ``\\>`` becomes ``\\3e `` and ``\\\\`` stays.
    """
    return _CSS_UNSAFE.sub(_escape_css_match, text)


def _escape_css_match(match: re.Match[str]) -> str:
    """``re.sub`` callback for ``_escape_css``, so it takes the match positionally."""
    escaped, bare = match.groups()
    if bare == _GREATER_THAN_DELIM:
        return ">"
    char = escaped if bare is None else bare
    if bare is None and not _CSS_HEX_ESCAPED.fullmatch(char):
        return match.group(0)
    return f"\\{ord(char):x} "
