"""Rewrite the CSS of a document the inliner makes self-contained.

The inliner (``cowork.services.artifact_inline``) hands this module every
style element, style attribute, kept attribute that holds "(", and linked
stylesheet it meets. Each resource the CSS names inlines as a ``data:`` URI,
or the declaration or rule that names it goes; an ``@import`` of the
artifact's own stylesheet is spliced in its place. Everything else that could
load something is dropped. The rewrite shares the inliner's ``InlineState``:
one inline budget, one reference count, one list of what was left out, one set
of placeholders and one deadline for the whole document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

import tinycss2
from tinycss2.ast import LiteralToken, Node, URLToken, WhitespaceToken
from tinycss2.serializer import serialize_identifier

from cowork.schemas.artifact_export import OmittedReason
from cowork.services.artifact_files import ArtifactFileContent, ArtifactFileRefused
from cowork.services.artifact_inline import (
    AssetKind,
    InlineState,
    clean_reference,
    is_data_uri,
    reference_parts,
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
# tinycss2 keeps every token as a Python object, which takes about 130 times
# the text's size in memory. So a document's style elements and the
# stylesheets it links or imports parse at most this much CSS in all. Past
# it, a style element's CSS is dropped and a stylesheet is left out as BUDGET.
_MAX_CSS_BYTES = 1024 * 1024
# Each attribute's CSS is bounded on its own: a style attribute, or another
# kept attribute whose value holds "(", which could hide an escaped url(,
# parses at most this much, and a longer one is dropped and listed as
# TOO_LARGE. Attributes have no total, so a chart's thousandth transform keeps
# working: the inliner's deadline, checked before each one, bounds them all.
# The most a page can send through tinycss2 this way, nine 1 MiB transforms in
# a 10 MiB page (scripts/measure_export_capacity.py, html-attribute-css-heavy,
# in the server's image at docker --cpus=1 --memory=1g), converts in 6.6 s and
# peaks at 136 MiB in the conversion process, under its 640 MiB org limit:
# tinycss2 holds one attribute's tokens at a time.
_MAX_ATTRIBUTE_CSS_BYTES = _MAX_CSS_BYTES


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
class CssRewriter:
    """Rewrites one document's CSS against the inliner's shared state."""

    state: InlineState
    # The CSS parsed so far, against ``_MAX_CSS_BYTES``.
    css_bytes: int = 0

    def take_css(self, *, size: int) -> bool:
        """Count *size* more units of CSS to parse; False once the document's CSS cap is spent."""
        if self.css_bytes + size > _MAX_CSS_BYTES:
            return False
        self.css_bytes += size
        return True

    def linked_stylesheet(self, *, reference: str) -> str | None:
        """The rewritten stylesheet *reference* names, or None after recording why not."""
        frame = self._spliced_frame(
            reference=reference, frame=self._root_css_frame(nodes=[]), media=[]
        )
        if frame is None:
            return None
        return _escape_css(text=self._run_css(root=frame))

    def attribute_is_inert(self, *, value: str) -> bool:
        """Whether *value* names no resource through a CSS ``url(``.

        Only ``url(#fragment)`` may stay, which names something in this
        document, as SVG's ``fill`` or ``filter`` do.
        """
        if "(" not in value:
            return True
        if not self._attribute_parses(value=value):
            return False
        nodes = tinycss2.parse_component_value_list(value, skip_comments=True)
        scan = _scan_css(nodes=nodes, strict=False)
        if not scan.checkable:
            return False
        for reference in scan.references:
            cleaned = clean_reference(value=reference.url)
            if not cleaned.startswith("#"):
                self.state.omit_unloaded(reference=cleaned)
                return False
        return True

    def rewrite_style_attribute(self, *, value: str) -> str:
        if not self._attribute_parses(value=value):
            return ""
        pieces = []
        nodes = tinycss2.parse_blocks_contents(value, skip_comments=True, skip_whitespace=True)
        for node in nodes:
            if self.state.out_of_time():
                break
            if node.type == "declaration" and self._rewrite_declaration(
                declaration=node, base_parts=self.state.base_parts, kind=AssetKind.IMAGE
            ):
                pieces.append(_serialize_css(nodes=[node]))
        return _escape_css(text="".join(pieces))

    def _attribute_parses(self, *, value: str) -> bool:
        """Whether one attribute's CSS may parse: before the deadline, and within its own bound.

        One over the bound is listed as left out, so the export counts it.
        """
        if self.state.out_of_time():
            return False
        if len(value) > _MAX_ATTRIBUTE_CSS_BYTES:
            self.state.omit(reference=value, reason=OmittedReason.TOO_LARGE)
            return False
        return True

    def rewrite_stylesheet(self, *, text: str) -> str:
        if not self.take_css(size=len(text)):
            return ""
        nodes = tinycss2.parse_stylesheet(text, skip_comments=True, skip_whitespace=True)
        return _escape_css(text=self._run_css(root=self._root_css_frame(nodes=nodes)))

    def _root_css_frame(self, *, nodes: list[Node]) -> _CssFrame:
        """The frame for a stylesheet in the document itself."""
        return _CssFrame(
            nodes=nodes,
            base_parts=self.state.base_parts,
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
            if self.state.out_of_time():
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
            kind = AssetKind.FONT if font else AssetKind.IMAGE
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
        self, *, declaration: Node, base_parts: tuple[str, ...], kind: AssetKind
    ) -> bool:
        """Inline every resource the declaration names; False drops the declaration."""
        scan = _scan_css(nodes=declaration.value, strict=True)
        if not scan.checkable:
            return False
        spent = 0
        kept_uris = len(self.state.placeholders.uris)
        for reference in scan.references:
            cleaned = clean_reference(value=reference.url)
            if cleaned.startswith("#"):
                continue
            asset = self.state.inline_asset(reference=cleaned, base_parts=base_parts, kind=kind)
            if asset is None:
                # The declaration goes, so the URIs it placed go with it.
                self.state.total_bytes -= spent
                del self.state.placeholders.uris[kept_uris:]
                return False
            spent += asset.size
            # The placeholder is letters, digits and a hyphen, so it needs no
            # escape in an unquoted url().
            placeholder = self.state.placeholders.hold(uri=asset.uri)
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
        cleaned = clean_reference(value=reference)
        if not cleaned:
            return None
        try:
            stylesheet = self._load_stylesheet(reference=cleaned, frame=frame)
        except ArtifactFileRefused as refusal:
            self.state.omit(reference=cleaned, reason=refusal.reason)
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
        if is_data_uri(reference=reference):
            raise ArtifactFileRefused(OmittedReason.UNSUPPORTED_TYPE, "data: stylesheet")
        parts = frame.base_parts + reference_parts(reference=reference)
        self.state.count_reference()
        if frame.import_depth >= self.state.limits.max_import_depth:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "imports nest too deep")
        content = self.state.read(parts=parts)
        if (content.device, content.inode) in frame.chain:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "import cycle")
        if not self.take_css(size=len(content.data)):
            raise ArtifactFileRefused(OmittedReason.BUDGET, "over the CSS cap")
        self.state.spend(size=len(content.data))
        return _LoadedStylesheet(parts=parts, content=content)


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
