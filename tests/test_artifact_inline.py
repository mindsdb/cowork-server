"""The inliner keeps only an artifact's own assets, as data: URIs, and drops the rest.

Every document case runs inside the ``no_fetch()`` guard with a 127.0.0.1
recording server as the outside target, and checks that neither saw a call,
that another partition's sentinel bytes never reach the output, and that the
output parses again to the same structure. The output is also read by
html5lib, which parses as a browser does, so a document libxml2 and a browser
read differently can't carry a load or a handler past the checks.
"""
from __future__ import annotations

import ast
import base64
import html as html_lib
import json
import os
import re
import struct
import subprocess
import sys
import time
import tracemalloc
import types
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import Element

import html5lib
import lxml.etree
import lxml.html
import pytest
import yaml
from tinycss2.serializer import serialize_string_value

from cowork.common.paths import pinned_dir
from cowork.schemas.artifact_export import (
    InlinedDocument,
    InlineLimits,
    InlineMode,
    OmittedReason,
)
from cowork.services import artifact_files, artifact_inline
from cowork.services.artifact_inline import TRUNCATION_NOTE, inline_artifact_html
from cowork.services.markdown_render import render_markdown
from tests._no_fetch import RecordingServer, no_fetch, recording_server

FIXTURES = Path(__file__).parent / "fixtures" / "artifact_inline"
SENTINEL = b"SENTINEL-B"
BUDGET = 25 * 1024 * 1024

NEVER_KEPT = frozenset(
    yaml.safe_load((FIXTURES / "invariants.yaml").read_text(encoding="utf-8"))["never_kept"]
)
DATA_URI = re.compile(
    r"data:(image/(png|jpeg|gif|webp)|font/(woff2|woff|ttf|otf));base64,[A-Za-z0-9+/]*={0,2}"
)
NAVIGATION = re.compile(r"(https?|mailto):", re.IGNORECASE)
PINNED_CDN = "https://cdn.jsdelivr.net/npm/"
# html5lib names an SVG or MathML element ``{namespace}tag``; HTML elements
# carry no namespace.
FOREIGN_TAG = re.compile(r"\{http://www\.w3\.org/(2000/svg|1998/Math/MathML)\}(.+)")


def _png(*, width: int, height: int, text: bytes = b"") -> bytes:
    """A decodable PNG, optionally carrying *text* in a tEXt chunk."""
    rows = zlib.compress(b"".join(b"\x00" + b"\x00" * width for _ in range(height)))
    extra = _png_chunk(kind=b"tEXt", data=b"note\x00" + text) if text else b""
    return (
        _png_header(width=width, height=height)
        + extra
        + _png_chunk(kind=b"IDAT", data=rows)
        + _png_chunk(kind=b"IEND", data=b"")
    )


def _png_header(*, width: int, height: int) -> bytes:
    """Only the signature and IHDR: enough for a header check, nothing to decode."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(kind=b"IHDR", data=ihdr)


def _png_chunk(*, kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _jpeg(*, width: int, height: int, frame: bool = True) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + (sof0 if frame else b"") + b"\xff\xd9"


def _gif(*, width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00;"


def _webp(*, chunk: bytes, payload: bytes) -> bytes:
    body = b"WEBP" + chunk + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


# 3x2 images: a VP8 frame header, a VP8L header and a VP8X canvas size.
WEBP_LOSSY = _webp(
    chunk=b"VP8 ", payload=b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", 3, 2) + b"\x00" * 4
)
WEBP_LOSSLESS = _webp(
    chunk=b"VP8L", payload=b"\x2f" + (2 | (1 << 14)).to_bytes(4, "little") + b"\x00" * 4
)
WEBP_EXTENDED = _webp(
    chunk=b"VP8X", payload=b"\x00" * 4 + (2).to_bytes(3, "little") + (1).to_bytes(3, "little")
)
CHART_PNG = _png(width=1, height=1)

ARTIFACT_FILES = {
    "chart.png": CHART_PNG,
    "photo.jpg": _jpeg(width=2, height=3),
    "headless.jpg": _jpeg(width=2, height=3, frame=False),
    "anim.gif": _gif(width=4, height=5),
    "lossy.webp": WEBP_LOSSY,
    "lossless.webp": WEBP_LOSSLESS,
    "extended.webp": WEBP_EXTENDED,
    "png-named.jpg": CHART_PNG,
    "chart-named.svg": CHART_PNG,
    "empty-size.png": _png_header(width=0, height=1),
    "huge.png": _png_header(width=10_000, height=10_000),
    "font.woff2": b"wOF2" + b"\x00" * 44,
    "font.ttf": b"\x00\x01\x00\x00" + b"\x00" * 12,
    "notes.txt": b"plain text",
    "vector.svg": b'<svg xmlns="http://www.w3.org/2000/svg"/>',
    "metadata.json": b"{}",
    "sub/inner.png": _png(width=2, height=1),
    "sub/Metadata.JSON": b"{}",
    ".revisions/old.png": CHART_PNG,
    "css/more.css": b".more{background:url(inner.png)}",
    "css/inner.png": _png(width=1, height=2),
    "css/d1.css": b'@import "d2.css"; .d1{color:red}',
    "css/d2.css": b'@import "d3.css"; .d2{color:red}',
    "css/d3.css": b'@import "d4.css"; .d3{color:red}',
    "css/d4.css": b".d4{color:red}",
    "css/loop-a.css": b'@import "loop-b.css"; .loop-a{color:red}',
    "css/loop-b.css": b'@import "loop-a.css"; .loop-b{color:blue}',
    "css/angles.css": b'p::after{content:"</style><script>alert(1)</script>"}',
}
# Another person's partition, beside the artifact folder.
OTHER_FILES = {
    "secret.txt": SENTINEL,
    "secret.png": _png(width=1, height=1, text=SENTINEL),
    "secret.css": b"body{background:url(secret.png)}/*" + SENTINEL + b"*/",
}


@dataclass(frozen=True)
class Artifact:
    folder: Path
    other: Path


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Artifact:
    root = tmp_path_factory.mktemp("partitions")
    folder = root / "art"
    other = root / "other"
    for base, files in ((folder, ARTIFACT_FILES), (other, OTHER_FILES)):
        for name, data in files.items():
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return Artifact(folder=folder, other=other)


@pytest.fixture(scope="module")
def outside():
    with recording_server() as server:
        yield server


def _placeholders(*, server: RecordingServer) -> dict[str, str]:
    encoded = {
        "png_b64": ARTIFACT_FILES["chart.png"],
        "jpeg_b64": ARTIFACT_FILES["photo.jpg"],
        "gif_b64": ARTIFACT_FILES["anim.gif"],
        "webp_vp8_b64": WEBP_LOSSY,
        "webp_vp8l_b64": WEBP_LOSSLESS,
        "webp_vp8x_b64": WEBP_EXTENDED,
        "woff2_b64": ARTIFACT_FILES["font.woff2"],
        "ttf_b64": ARTIFACT_FILES["font.ttf"],
        "inner_b64": ARTIFACT_FILES["sub/inner.png"],
    }
    values = {name: base64.b64encode(data).decode("ascii") for name, data in encoded.items()}
    values["outside"] = server.url
    return values


def _fill(*, text: str, server: RecordingServer) -> str:
    for name, value in _placeholders(server=server).items():
        text = text.replace("{" + name + "}", value)
    return text


def _limits(**overrides: int) -> InlineLimits:
    return InlineLimits(**{"max_total_bytes": BUDGET, **overrides})


def _inline(
    *,
    artifact: Artifact,
    html: bytes,
    mode: InlineMode = InlineMode.WORD,
    limits: InlineLimits | None = None,
    base_parts: tuple[str, ...] = (),
) -> InlinedDocument:
    with pinned_dir(artifact.folder) as folder:
        return inline_artifact_html(
            html=html,
            folder=folder,
            base_parts=base_parts,
            mode=mode,
            limits=limits or _limits(),
        )


def _guarded_inline(
    *, artifact: Artifact, server: RecordingServer, html: bytes, **kwargs
) -> InlinedDocument:
    """Inline twice under the guard: nothing fetched, the same answer both times."""
    with no_fetch() as recorder:
        document = _inline(artifact=artifact, html=html, **kwargs)
        again = _inline(artifact=artifact, html=html, **kwargs)
    assert recorder.calls == []
    assert server.hits() == []
    assert again == document
    _assert_no_sentinel(document=document)
    _assert_output_invariants(text=document.html, mode=kwargs.get("mode", InlineMode.WORD))
    return document


def _assert_no_sentinel(*, document: InlinedDocument) -> None:
    assert SENTINEL.decode() not in document.html
    for encoded in re.findall(r"base64,([A-Za-z0-9+/=]+)", document.html):
        assert SENTINEL not in base64.b64decode(encoded)


def _parse(*, text: str) -> lxml.html.HtmlElement:
    parser = lxml.html.HTMLParser(encoding="utf-8", no_network=True, huge_tree=False)
    return lxml.html.document_fromstring(text.encode("utf-8"), parser=parser)


def _assert_output_invariants(*, text: str, mode: InlineMode) -> None:
    """What holds for every output: a stable parse and nothing that loads or runs."""
    root = _parse(text=text)
    again = lxml.html.tostring(root, encoding="unicode", method="html", doctype="<!DOCTYPE html>")
    assert again == text
    for element in root.iter():
        assert isinstance(element.tag, str), "a comment or processing instruction survived"
        assert element.tag not in NEVER_KEPT
        if element.tag == "meta":
            assert dict(element.attrib) == {"charset": "utf-8"}
        if element.tag == "title":
            assert element.getparent().tag == "head"
        if element.tag in ("script", "style"):
            assert not any(parent.tag in ("svg", "math") for parent in element.iterancestors())
        if element.tag == "script":
            assert mode is InlineMode.PDF
        if element.tag == "style":
            assert "<" not in (element.text or "")
        _assert_attributes_load_nothing(tag=element.tag, attributes=dict(element.attrib))
    _assert_browser_parse_loads_nothing(text=text, mode=mode)


def _assert_attributes_load_nothing(*, tag: str, attributes: dict[str, str]) -> None:
    for name, value in attributes.items():
        assert not name.startswith("on")
        if name == "src" and tag == "script":
            assert value.startswith(PINNED_CDN)
        elif name == "src":
            assert DATA_URI.fullmatch(value), value[:80]
        elif name in ("href", "xlink:href"):
            assert value.startswith("#") or (tag == "a" and NAVIGATION.match(value))
        elif name != "style" and "url(" in value.lower():
            assert re.fullmatch(r"url\(\s*['\"]?#[^)]*\)", value.strip()), value


def _assert_browser_parse_loads_nothing(*, text: str, mode: InlineMode) -> None:
    """Read the output as a browser does and apply the same element and attribute rules."""
    root = html5lib.parse(text, treebuilder="etree", namespaceHTMLElements=False)
    for element in _html5lib_elements(root=root):
        if not isinstance(element.tag, str):
            continue
        foreign = FOREIGN_TAG.fullmatch(element.tag)
        tag = (foreign.group(2) if foreign else element.tag).lower()
        assert tag not in NEVER_KEPT
        if foreign is not None:
            assert tag not in ("script", "style"), f"{tag} in foreign content"
        if tag == "meta":
            assert dict(element.attrib) == {"charset": "utf-8"}
        if tag == "script":
            assert mode is InlineMode.PDF
        # A foreign attribute such as xlink:href arrives as {namespace}href.
        attributes = {name.rpartition("}")[2].lower(): value for name, value in element.attrib.items()}
        _assert_attributes_load_nothing(tag=tag, attributes=attributes)


def _html5lib_elements(*, root: Element) -> Iterator[Element]:
    stack = [root]
    while stack:
        element = stack.pop()
        yield element
        stack.extend(element)


def _cases(*, name: str) -> list[dict]:
    return yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8"))


def _document_params() -> list:
    params = []
    files = ("elements.yaml", "attributes.yaml", "scripts.yaml", "images.yaml", "css.yaml")
    for file_name in files:
        for case in _cases(name=file_name):
            for mode in case.get("modes", [mode.value for mode in InlineMode]):
                params.append(pytest.param(case, InlineMode(mode), id=f"{case['id']}-{mode}"))
    return params


@pytest.mark.parametrize(("case", "mode"), _document_params())
def test_document_case(case, mode, artifact, outside):
    source = case["html"] if "html" in case else f"<style>{case['css']}</style>"
    document = _guarded_inline(
        artifact=artifact,
        server=outside,
        html=_fill(text=source, server=outside).encode("utf-8"),
        mode=mode,
        limits=_limits(**case.get("limits", {})),
        base_parts=tuple(case.get("base_parts", ())),
    )
    for text in case.get("present", []):
        assert _fill(text=text, server=outside) in document.html
    for text in case.get("absent", []):
        assert _fill(text=text, server=outside) not in document.html
    omitted = case.get("omitted", [])
    if lxml.etree.LIBXML_VERSION < (2, 14) and "omitted_before_libxml2_2_14" in case:
        omitted = case["omitted_before_libxml2_2_14"]
    expected = [(reason, _fill(text=reference, server=outside)) for reason, reference in omitted]
    assert [(entry.reason.value, entry.reference) for entry in document.omitted] == expected
    if "scripts_dropped" in case:
        assert document.scripts_dropped is case["scripts_dropped"]
    if "inlined" in case:
        assert document.inlined_bytes == sum(len(ARTIFACT_FILES[name]) for name in case["inlined"])
    assert TRUNCATION_NOTE not in document.html


REFERENCE_CASES = _cases(name="references.yaml")


def _assert_reference_outcome(*, document: InlinedDocument, case: dict, reference: str) -> None:
    if case["reason"] is None:
        assert "data:image/png;base64," in document.html
        assert document.omitted == []
    else:
        recorded = case.get("recorded", reference)
        assert [(entry.reason.value, entry.reference) for entry in document.omitted] == [
            (case["reason"], recorded)
        ]
        assert "data:" not in document.html


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=[case["id"] for case in REFERENCE_CASES])
def test_reference_in_an_image(case, artifact, outside):
    reference = _fill(text=case["reference"], server=outside)
    html = f'<img src="{html_lib.escape(reference, quote=True)}">'
    document = _guarded_inline(artifact=artifact, server=outside, html=html.encode("utf-8"))
    _assert_reference_outcome(document=document, case=case, reference=reference)


@pytest.mark.parametrize(
    "case",
    [case for case in REFERENCE_CASES if not case.get("img_only")],
    ids=[case["id"] for case in REFERENCE_CASES if not case.get("img_only")],
)
def test_reference_in_css(case, artifact, outside):
    reference = _fill(text=case["reference"], server=outside)
    html = f'<style>.a{{background:url("{serialize_string_value(reference)}")}}</style>'
    document = _guarded_inline(artifact=artifact, server=outside, html=html.encode("utf-8"))
    _assert_reference_outcome(document=document, case=case, reference=reference)


@pytest.mark.skipif(os.name == "nt", reason="symlinks and hard links need privileges on Windows")
def test_links_into_another_partition_are_refused(tmp_path, outside):
    folder = tmp_path / "art"
    other = tmp_path / "other"
    folder.mkdir()
    other.mkdir()
    for name, data in OTHER_FILES.items():
        (other / name).write_bytes(data)
    (folder / "link.png").symlink_to(other / "secret.png")
    (folder / "linkdir").symlink_to(other, target_is_directory=True)
    (folder / "link.css").symlink_to(other / "secret.css")
    os.link(other / "secret.png", folder / "hard.png")
    html = (
        b'<img src="link.png"><img src="hard.png"><img src="linkdir/secret.png">'
        b'<style>@import "link.css";</style>'
    )
    links = Artifact(folder=folder, other=other)
    document = _guarded_inline(artifact=links, server=outside, html=html)
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        (OmittedReason.SYMLINK, "link.png"),
        (OmittedReason.SYMLINK, "hard.png"),
        (OmittedReason.SYMLINK, "linkdir/secret.png"),
        (OmittedReason.SYMLINK, "link.css"),
    ]


@pytest.mark.parametrize("reference", ["sub/a:b.png", "a_b:stream.png", "sub\\x:y.png"])
def test_a_segment_holding_a_colon_is_refused_before_the_file_walk(reference):
    """The file walk refuses ':' too, so only a direct call shows this step holds."""
    with pytest.raises(artifact_inline.ArtifactFileRefused) as refused:
        artifact_inline._reference_parts(reference=reference)

    assert refused.value.reason is OmittedReason.TRAVERSAL


@pytest.mark.parametrize("path", ["sub/C:y.png", "C:x/y.png", "sub/../y.png", "sub//y.png", "a\x00b.png"])
def test_a_request_path_and_a_reference_refuse_the_same_segments(path):
    """The drafts route's request paths and a document's references split by one rule."""
    with pytest.raises(ValueError):
        artifact_files.relative_file_parts(path)
    with pytest.raises(artifact_files.ArtifactFileRefused):
        artifact_inline._reference_parts(reference=path)


def test_the_drafts_path_a_reference_and_the_read_share_one_segment_rule(monkeypatch):
    monkeypatch.setattr(artifact_files, "_is_plain_segment", lambda *, segment: segment != "blocked")

    with pytest.raises(ValueError):
        artifact_files.relative_file_parts("sub/blocked")
    with pytest.raises(artifact_files.ArtifactFileRefused):
        artifact_inline._reference_parts(reference="sub/blocked")
    with pytest.raises(artifact_files.ArtifactFileRefused):
        artifact_files.refuse_unsafe_parts(parts=("sub", "blocked"))


def _is_xml_character(code: int) -> bool:
    """XML 1.0's Char: tab, LF, CR, U+0020 to U+D7FF, U+E000 to U+FFFD, U+10000 and up."""
    return code in (0x9, 0xA, 0xD) or 0x20 <= code <= 0xD7FF or 0xE000 <= code <= 0xFFFD or code >= 0x10000


def test_the_xml_filter_removes_exactly_what_xml_forbids():
    every_character = "".join(map(chr, range(0x110000)))
    expected = "".join(chr(code) for code in range(0x110000) if _is_xml_character(code))

    assert artifact_inline.XML_UNSAFE.sub("", every_character) == expected


def test_a_placeholder_holds_its_uri_until_restored():
    placeholders = artifact_inline.UriPlaceholders()
    first = placeholders.hold(uri="data:image/png;base64,AAAA")
    second = placeholders.hold(uri="data:image/gif;base64,R0lG")

    assert re.fullmatch(r"[0-9a-f]+[0-9]+-", first)
    assert placeholders.restore(text=f'<img src="{first}"><b>{second}</b>') == (
        '<img src="data:image/png;base64,AAAA"><b>data:image/gif;base64,R0lG</b>'
    )


@pytest.mark.parametrize(
    ("tag", "namespace", "name"),
    [
        ("p", artifact_inline._Namespace.HTML, "onclick"),
        ("img", artifact_inline._Namespace.HTML, "onerror"),
        ("rect", artifact_inline._Namespace.SVG, "onbegin"),
        ("mi", artifact_inline._Namespace.MATH, "onload"),
    ],
)
def test_an_event_handler_goes_even_where_an_allowlist_would_keep_it(
    tag, namespace, name, monkeypatch
):
    """The allowlists drop every on* name today; this check must not lean on them."""
    monkeypatch.setattr(artifact_inline, "_attribute_allowed", lambda **_: True)
    inliner = artifact_inline._Inliner(
        folder=None,
        base_parts=(),
        mode=InlineMode.PDF,
        limits=_limits(),
        deadline=time.monotonic() + artifact_inline.DEADLINE_SECONDS,
    )

    kept = inliner._kept_attribute(tag=tag, namespace=namespace, name=name, value="alert(1)")

    assert kept is None
    assert inliner._kept_attribute(tag=tag, namespace=namespace, name="title", value="t") == "t"


def test_doctype_entities_and_dtd_load_nothing(artifact, outside):
    """libxml2's HTML parser reads no internal subset, so no entity resolves."""
    secret = (artifact.other / "secret.txt").as_uri()
    html = (
        f'<!DOCTYPE html PUBLIC "-//X//DTD X//EN" "{outside.url}/dtd" [\n'
        f'<!ENTITY remote SYSTEM "{outside.url}/entity">\n'
        f'<!ENTITY local SYSTEM "{secret}">\n'
        f'<!ENTITY partition SYSTEM "../other/secret.txt">\n'
        f'<!ENTITY % param SYSTEM "{outside.url}/param"> %param;\n'
        "]>\n"
        "<html><body><p>&remote;</p><p>&local;</p><p>&partition;</p></body></html>"
    )
    for mode in InlineMode:
        document = _guarded_inline(
            artifact=artifact, server=outside, html=html.encode("utf-8"), mode=mode
        )
        assert "<p>&amp;remote;</p><p>&amp;local;</p><p>&amp;partition;</p>" in document.html
        assert "<!ENTITY" not in document.html
        assert document.omitted == []


def test_xinclude_loads_nothing(artifact, outside):
    """There is no XInclude step: ``xi:include`` is an unknown element and goes."""
    secret = (artifact.other / "secret.txt").as_uri()
    html = (
        '<html xmlns:xi="http://www.w3.org/2001/XInclude"><body>'
        f'<xi:include href="{outside.url}/xi"/>'
        f'<xi:include href="{secret}" parse="text"/>'
        '<xi:include href="../other/secret.txt" parse="text">'
        "<xi:fallback>fallback</xi:fallback></xi:include>"
        "<p>kept</p></body></html>"
    )
    for mode in InlineMode:
        document = _guarded_inline(
            artifact=artifact, server=outside, html=html.encode("utf-8"), mode=mode
        )
        assert "<p>kept</p>" in document.html
        assert "xi:" not in document.html
        assert "XInclude" not in document.html
        assert document.omitted == []


def _dotted(*, node: ast.expr) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_name(*, node: ast.Call) -> str:
    return _dotted(node=node.func)


SERVICES = Path(artifact_inline.__file__).parent
LXML_IMPORT = re.compile(r"^\s*(?:import|from)\s+lxml\b", re.MULTILINE)
# Every module of the export services that uses lxml, since each parses artifact HTML.
LXML_MODULES = sorted(
    path for path in SERVICES.glob("*.py") if LXML_IMPORT.search(path.read_text(encoding="utf-8"))
)
# lxml entry points that parse XML, read a file or URL, or expand XInclude.
FORBIDDEN_LXML_NAMES = frozenset(
    {
        "lxml.etree.parse", "lxml.etree.fromstring", "lxml.etree.fromstringlist",
        "lxml.etree.XML", "lxml.etree.HTML", "lxml.etree.XMLID", "lxml.etree.XMLDTDID",
        "lxml.etree.iterparse", "lxml.etree.XInclude", "lxml.etree.ElementTree",
        "lxml.etree.XSLT", "lxml.etree.XMLSchema", "lxml.etree.RelaxNG", "lxml.etree.DTD",
        "lxml.html.parse", "lxml.html.html_parser", "lxml.html.xhtml_parser",
        "lxml.ElementInclude",
    }
)
SAFE_XML_PARSER_OPTIONS = {
    "resolve_entities": False, "no_network": True, "load_dtd": False, "huge_tree": False
}
HTML_STRING_PARSERS = frozenset(
    {
        "lxml.html.document_fromstring", "lxml.html.fromstring",
        "lxml.html.fragment_fromstring", "lxml.html.fragments_fromstring",
    }
)


def _lxml_aliases(*, tree: ast.Module) -> dict[str, str]:
    """Map each name a module binds by importing lxml to the dotted name it stands for."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("lxml"):
                    bound = alias.asname or alias.name.split(".")[0]
                    aliases[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lxml"):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _resolved(*, node: ast.expr, aliases: dict[str, str]) -> str:
    head, _, rest = _dotted(node=node).partition(".")
    if head not in aliases:
        return ""
    return aliases[head] + ("." + rest if rest else "")


def _constant_options(*, call: ast.Call) -> dict[str, object]:
    return {
        keyword.arg: keyword.value.value
        for keyword in call.keywords
        if isinstance(keyword.value, ast.Constant)
    }


def test_every_module_parsing_artifact_html_is_checked():
    assert {"artifact_inline.py", "docx_writer.py"} <= {path.name for path in LXML_MODULES}


@pytest.mark.parametrize("path", LXML_MODULES, ids=[path.name for path in LXML_MODULES])
def test_artifact_html_is_parsed_only_by_lxml_html_with_the_network_off(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _lxml_aliases(tree=tree)
    used = {
        _resolved(node=node, aliases=aliases)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    }
    assert not {name for name in used | set(aliases.values()) if name in FORBIDDEN_LXML_NAMES}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    # A dynamic import would hide an lxml name from the checks here.
    hidden = {"xinclude", "ElementInclude", "XInclude", "__import__", "import_module", "importlib"}
    assert not (attributes | names) & hidden
    html_parsers = 0
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = _resolved(node=call.func, aliases=aliases) or _call_name(node=call)
        options = _constant_options(call=call)
        if name.endswith("HTMLParser"):
            assert name == "lxml.html.HTMLParser"
            assert options.get("no_network") is True and options.get("huge_tree") is False
            html_parsers += 1
        elif name.endswith("XMLParser"):
            assert all(options.get(key) is value for key, value in SAFE_XML_PARSER_OPTIONS.items())
        elif name in HTML_STRING_PARSERS:
            assert any(keyword.arg == "parser" for keyword in call.keywords)
    assert html_parsers >= 1


def test_inlining_loads_no_word_library():
    """HTML and PDF exports inline too, and must not pay for loading python-docx."""
    program = (
        "import json, sys\n"
        "import cowork.services.artifact_inline\n"
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] == 'docx')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []


ENCODING_CASES = _cases(name="encodings.yaml")


@pytest.mark.parametrize("case", ENCODING_CASES, ids=[case["id"] for case in ENCODING_CASES])
def test_source_encoding_is_read_and_the_output_declares_utf8(case, artifact, outside):
    source = case["html"].encode(case["encoding"])
    document = _guarded_inline(artifact=artifact, server=outside, html=source)
    assert document.html.startswith('<!DOCTYPE html>\n<html><head><meta charset="utf-8">')
    assert f"<p>{case['text']}</p>" in document.html


def test_output_keeps_the_title_in_the_head_for_the_word_writer(artifact, outside):
    html = b"<html><head><title>Quarterly report</title></head><body><h1>Q3</h1></body></html>"
    document = _guarded_inline(artifact=artifact, server=outside, html=html)
    root = _parse(text=document.html)
    assert root.find("head/title").text == "Quarterly report"


def test_asset_over_the_per_asset_cap(artifact, outside):
    data_uri = "data:image/png;base64," + base64.b64encode(CHART_PNG).decode()
    html = f'<img src="chart.png"><img src="{data_uri}">'.encode()
    limits = _limits(max_asset_bytes=len(CHART_PNG) - 1)
    document = _guarded_inline(artifact=artifact, server=outside, html=html, limits=limits)
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        (OmittedReason.TOO_LARGE, "chart.png"),
        (OmittedReason.TOO_LARGE, data_uri),
    ]
    assert document.inlined_bytes == 0


def test_reference_cap_counts_html_and_css(artifact, outside):
    html = b'<img src="chart.png"><p style="background:url(chart.png)">x</p><img src="chart.png">'
    limits = _limits(max_references=2)
    document = _guarded_inline(artifact=artifact, server=outside, html=html, limits=limits)
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        (OmittedReason.BUDGET, "chart.png")
    ]
    assert document.inlined_bytes == 2 * len(CHART_PNG)


def test_reference_cap_skips_references_refused_before_any_read(artifact, outside):
    html = f'<img src="{outside.url}/a.png"><img src="../x.png"><img src="chart.png">'.encode()
    limits = _limits(max_references=1)
    document = _guarded_inline(artifact=artifact, server=outside, html=html, limits=limits)
    reasons = [entry.reason for entry in document.omitted]
    assert reasons == [OmittedReason.SCHEME, OmittedReason.TRAVERSAL]
    assert document.inlined_bytes == len(CHART_PNG)


@pytest.mark.parametrize(("max_pixels", "inlined"), [(6, True), (5, False)])
def test_pixel_cap_reads_the_header(max_pixels, inlined, artifact, outside):
    document = _guarded_inline(
        artifact=artifact,
        server=outside,
        html=b'<img src="photo.jpg">',
        limits=_limits(max_image_pixels=max_pixels),
    )
    assert ("data:image/jpeg" in document.html) is inlined
    reasons = [entry.reason for entry in document.omitted]
    assert reasons == ([] if inlined else [OmittedReason.TOO_LARGE])


def test_total_budget_counts_every_occurrence(artifact, outside):
    html = b'<img src="chart.png"><img src="chart.png"><img src="chart.png">'
    limits = _limits(max_total_bytes=2 * len(CHART_PNG))
    document = _guarded_inline(artifact=artifact, server=outside, html=html, limits=limits)
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        (OmittedReason.BUDGET, "chart.png")
    ]
    assert document.inlined_bytes == 2 * len(CHART_PNG)


def test_a_refused_asset_is_read_once_per_kind(artifact, monkeypatch):
    """Repeated references to a file that can't inline cost one read, not one each."""
    reads = []
    read = artifact_inline.read_artifact_file_entry

    def counting_read(*, folder, parts, max_bytes):
        reads.append(parts)
        return read(folder=folder, parts=parts, max_bytes=max_bytes)

    monkeypatch.setattr(artifact_inline, "read_artifact_file_entry", counting_read)
    html = (
        '<img src="notes.txt">' * 3
        + '<img src="absent.png">' * 3
        + '<p style="background:url(notes.txt)">x</p>'
        + '<img src="font.woff2">'
        + "<style>@font-face{font-family:f;src:url(font.woff2)}</style>"
    ).encode()
    document = _inline(artifact=artifact, html=html)
    assert sorted(reads) == [("absent.png",), ("font.woff2",), ("font.woff2",), ("notes.txt",)]
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        *[(OmittedReason.UNSUPPORTED_TYPE, "notes.txt")] * 3,
        *[(OmittedReason.MISSING, "absent.png")] * 3,
        (OmittedReason.UNSUPPORTED_TYPE, "notes.txt"),
        (OmittedReason.UNSUPPORTED_TYPE, "font.woff2"),
    ]
    assert "data:font/woff2;base64," in document.html


def test_total_budget_counts_spliced_css(artifact, outside):
    budget = len(ARTIFACT_FILES["css/more.css"]) - 1
    document = _guarded_inline(
        artifact=artifact,
        server=outside,
        html=b'<style>@import "css/more.css";</style>',
        limits=_limits(max_total_bytes=budget),
    )
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        (OmittedReason.BUDGET, "css/more.css")
    ]
    assert ".more" not in document.html


def test_the_budget_counts_a_spliced_stylesheet_as_read_not_as_escaped(tmp_path):
    """Each "<" becomes four characters in the output, but costs one byte of the budget."""
    folder = tmp_path / "art"
    folder.mkdir()
    sheet = b'p::after{content:"' + b"<" * 1000 + b'"}'
    (folder / "angles.css").write_bytes(sheet)
    document = _inline(
        artifact=Artifact(folder=folder, other=folder),
        html=b'<style>@import "angles.css";</style>',
        limits=_limits(max_total_bytes=len(sheet)),
    )
    assert document.omitted == []
    assert document.inlined_bytes == len(sheet)
    assert document.html.count("\\3c ") == 1000


def test_a_file_past_the_spent_budget_is_still_read_then_left_out(artifact, monkeypatch):
    """The budget counts a file once it is read, so it never bounds the reading."""
    reads = []
    read = artifact_inline.read_artifact_file_entry

    def counting_read(*, folder, parts, max_bytes):
        reads.append(parts)
        return read(folder=folder, parts=parts, max_bytes=max_bytes)

    monkeypatch.setattr(artifact_inline, "read_artifact_file_entry", counting_read)
    html = b'<img src="chart.png"><img src="photo.jpg"><img src="anim.gif">'
    document = _inline(
        artifact=artifact, html=html, limits=_limits(max_total_bytes=len(CHART_PNG))
    )
    assert reads == [("chart.png",), ("photo.jpg",), ("anim.gif",)]
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        (OmittedReason.BUDGET, "photo.jpg"),
        (OmittedReason.BUDGET, "anim.gif"),
    ]


def test_node_cap_drops_the_rest_of_the_tree_and_says_so(artifact, outside):
    html = "".join(f"<p>para {index}</p>" for index in range(100)).encode()
    limits = _limits(max_nodes=20)
    document = _guarded_inline(artifact=artifact, server=outside, html=html, limits=limits)
    assert "<p>para 0</p>" in document.html
    assert "para 99" not in document.html
    assert document.html.count("<p>") < 20
    assert document.html.endswith(f"<p>{TRUNCATION_NOTE}</p></body></html>")
    assert document.omitted == []


def test_node_cap_reached_exactly_keeps_everything_without_a_note(artifact, outside):
    html = b"<p>a</p><p>b</p>"
    # html, head, body and the two paragraphs.
    document = _guarded_inline(artifact=artifact, server=outside, html=html, limits=_limits(max_nodes=5))
    assert "<p>a</p><p>b</p>" in document.html
    assert TRUNCATION_NOTE not in document.html


def _clock_past_the_deadline_after(*, readings: int) -> types.SimpleNamespace:
    """A stand-in for the inliner's ``time`` whose clock passes the deadline after *readings*."""
    taken = []

    def monotonic() -> float:
        # The deadline is set from the first reading.
        taken.append(None)
        return 0.0 if len(taken) <= readings else artifact_inline.DEADLINE_SECONDS + 1

    return types.SimpleNamespace(monotonic=monotonic)


def test_deadline_drops_the_rest_of_the_tree_and_says_so(artifact, monkeypatch):
    monkeypatch.setattr(artifact_inline, "time", _clock_past_the_deadline_after(readings=5))
    html = "".join(f"<p>para {index}</p>" for index in range(100)).encode()
    document = _inline(artifact=artifact, html=html)
    _assert_output_invariants(text=document.html, mode=InlineMode.WORD)
    assert "<p>para 0</p>" in document.html
    assert "para 99" not in document.html
    assert document.html.endswith(f"<p>{TRUNCATION_NOTE}</p></body></html>")


def test_deadline_stops_a_stylesheet_with_its_blocks_closed(artifact, monkeypatch):
    monkeypatch.setattr(artifact_inline, "time", _clock_past_the_deadline_after(readings=20))
    rules = "".join(f".r{index}{{color:red}}" for index in range(200))
    html = f"<style>@media print{{{rules}}}</style><p>after</p>".encode()
    document = _inline(artifact=artifact, html=html)
    _assert_output_invariants(text=document.html, mode=InlineMode.WORD)
    style = _parse(text=document.html).find("head/style").text
    assert style.startswith("@media print{.r0{color:red;}")
    assert ".r199" not in style
    assert style.endswith("}}")
    assert style.count("{") == style.count("}")
    assert "after" not in document.html
    assert document.html.endswith(f"<p>{TRUNCATION_NOTE}</p></body></html>")


def test_deadline_stops_a_style_attribute(artifact, monkeypatch):
    monkeypatch.setattr(artifact_inline, "time", _clock_past_the_deadline_after(readings=4))
    declarations = ";".join(f"--d{index}:1" for index in range(200))
    document = _inline(artifact=artifact, html=f'<p style="{declarations}">x</p>'.encode())
    _assert_output_invariants(text=document.html, mode=InlineMode.WORD)
    assert '<p style="--d0:1;">x</p>' in document.html
    assert document.html.endswith(f"<p>{TRUNCATION_NOTE}</p></body></html>")


STOPPED_PARSES = _cases(name="stopped_parses.yaml")


@pytest.mark.parametrize("case", STOPPED_PARSES, ids=[case["id"] for case in STOPPED_PARSES])
def test_a_stopped_parse_is_recognized_in_every_libxml2_version(case):
    entry = types.SimpleNamespace(type_name=case["type_name"], message=case["message"])

    assert artifact_inline.parse_stopped(error_log=[entry]) is case["stopped"]


def test_a_source_parse_cut_short_by_libxml2_ends_with_the_note(artifact, outside):
    """Past 256 levels of nesting libxml2 stops reading and keeps only what came before."""
    html = b"<p>start</p>" + b"<div>" * 300 + b"<p>LOST</p>"
    document = _guarded_inline(artifact=artifact, server=outside, html=html)
    assert "<p>start</p>" in document.html
    assert "LOST" not in document.html
    assert document.html.endswith(f"<p>{TRUNCATION_NOTE}</p></body></html>")


def test_loads_inside_dropped_elements_are_listed_up_to_the_node_cap(artifact, outside):
    images = "".join(f'<img src="{outside.url}/{index}.png">' for index in range(10))
    html = f"<template>{images}</template><noscript>{images}</noscript><p>kept</p>".encode()
    document = _guarded_inline(
        artifact=artifact, server=outside, html=html, limits=_limits(max_nodes=12)
    )
    assert [entry.reference for entry in document.omitted] == [
        f"{outside.url}/{index}.png" for index in (*range(10), 0, 1)
    ]
    assert "<p>kept</p>" in document.html
    assert TRUNCATION_NOTE not in document.html


def test_omitted_entries_cap_the_reference_at_200_characters(artifact, outside):
    reference = f"{outside.url}/{'a' * 500}.png"
    html = f'<img src="{reference}">'.encode()
    document = _guarded_inline(artifact=artifact, server=outside, html=html)
    assert [entry.reference for entry in document.omitted] == [reference[:200]]


def test_omitted_list_stops_growing_but_drops_continue(artifact, outside):
    rules = "".join(
        f".r{index}{{background:url({outside.url}/{index}.png)}}" for index in range(1500)
    )
    html = f"<style>{rules}</style>".encode()
    document = _guarded_inline(artifact=artifact, server=outside, html=html)
    assert len(document.omitted) == artifact_inline._MAX_OMITTED
    assert outside.url not in document.html


def test_css_nesting_past_the_cap_is_dropped(artifact, outside):
    depth = artifact_inline._MAX_CSS_NESTING
    nested = "a{" * (depth + 5) + "color:red" + "}" * (depth + 5)
    deep_value = ".b{width:" + "calc(" * 500 + "1px" + ")" * 500 + ";color:blue}"
    html = f"<style>{nested}{deep_value}</style>".encode()
    document = _guarded_inline(artifact=artifact, server=outside, html=html)
    assert "a{" * depth in document.html
    assert "a{" * (depth + 1) not in document.html
    assert ".b{color:blue;}" in document.html


HOSTILE_SHAPES = _cases(name="hostile_shapes.yaml")


@pytest.mark.parametrize("case", HOSTILE_SHAPES, ids=[case["id"] for case in HOSTILE_SHAPES])
def test_hostile_shapes_never_raise(case, artifact, outside):
    encoding = case.get("encoding", "utf-8")
    source = b"".join(text.encode(encoding) * count for text, count in case["parts"])
    for mode in InlineMode:
        document = _guarded_inline(artifact=artifact, server=outside, html=source, mode=mode)
        assert document.html.startswith("<!DOCTYPE html>")


@pytest.mark.parametrize("foreign", ["svg", "math"])
def test_markdown_raw_html_script_in_foreign_content_goes_in_pdf(foreign, artifact, outside):
    text = f'Intro\n\n<{foreign}><script>1;<img src="{outside.url}/md-{foreign}"></script></{foreign}>\n\nEnd\n'
    html = render_markdown(text=text).encode("utf-8")
    document = _guarded_inline(artifact=artifact, server=outside, html=html, mode=InlineMode.PDF)
    assert "<script" not in document.html
    assert outside.url not in document.html
    assert document.scripts_dropped is True


SIBLING_CASES = _cases(name="many_siblings.yaml")


@pytest.mark.parametrize("case", SIBLING_CASES, ids=[case["id"] for case in SIBLING_CASES])
def test_unwrapping_or_dropping_many_siblings_stays_fast(case, artifact):
    unit = case["unit"].replace("{text}", "x" * 400)
    html = ("<p>" + unit * case["count"] + "</p>").encode()
    started = time.monotonic()
    document = _inline(artifact=artifact, html=html)
    elapsed = time.monotonic() - started
    assert document.html.count("x") == unit.count("x") * case["count"]
    assert "<font" not in document.html and "<noscript" not in document.html
    # Rebuilding the parent once per element took 3 to 30 seconds here.
    assert elapsed < 2.5


LARGE_IMAGES = 12
LARGE_IMAGE_BYTES = 1_900_000


def _large_png(*, index: int) -> bytes:
    """A distinct PNG just under the per-asset cap; its padding sits in a tEXt chunk."""
    return _png(width=1, height=1, text=f"{index:02d}".encode() * (LARGE_IMAGE_BYTES // 2))


@pytest.fixture(scope="module")
def large_artifact(tmp_path_factory) -> Artifact:
    folder = tmp_path_factory.mktemp("large") / "art"
    folder.mkdir()
    for index in range(LARGE_IMAGES):
        (folder / f"big{index}.png").write_bytes(_large_png(index=index))
    return Artifact(folder=folder, other=folder)


def test_large_inlined_images_keep_the_rest_of_the_document(large_artifact):
    """libxml2 with huge_tree off stops after about 10 MB of large tokens; nothing may be lost."""
    images = "".join(f'<p><img src="big{index}.png"></p>' for index in range(LARGE_IMAGES))
    html = f"<html><body><p>start</p>{images}<p>END</p></body></html>".encode()
    with no_fetch() as recorder:
        document = _inline(artifact=large_artifact, html=html)
    assert recorder.calls == []
    assert document.omitted == []
    assert document.inlined_bytes == sum(len(_large_png(index=i)) for i in range(LARGE_IMAGES))
    uris = re.findall(r'<img src="data:image/png;base64,([A-Za-z0-9+/=]+)">', document.html)
    assert [base64.b64decode(uri) for uri in uris] == [
        _large_png(index=index) for index in range(LARGE_IMAGES)
    ]
    assert document.html.endswith("<p>END</p></body></html>")
    assert TRUNCATION_NOTE not in document.html


def test_inlining_holds_at_most_two_copies_of_the_base64(large_artifact):
    """The URIs and the finished output; no tree, serialization or byte copy holds them too."""
    images = "".join(f'<p><img src="big{index}.png"></p>' for index in range(LARGE_IMAGES))
    tracemalloc.start()
    try:
        document = _inline(artifact=large_artifact, html=images.encode())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert document.html.count("data:image/png;base64,") == LARGE_IMAGES
    assert peak < 2.5 * len(document.html)


def test_large_css_backgrounds_keep_the_body_and_stay_fast(large_artifact):
    rules = "".join(
        f".b{index}{{background:url(big{index % 4}.png)}}" for index in range(8)
    )
    html = f"<html><head><style>{rules}</style></head><body><p>END</p></body></html>".encode()
    started = time.monotonic()
    with no_fetch() as recorder:
        document = _inline(artifact=large_artifact, html=html)
    elapsed = time.monotonic() - started
    assert recorder.calls == []
    assert document.omitted == []
    uris = re.findall(r"url\(data:image/png;base64,([A-Za-z0-9+/=]+)\)", document.html)
    assert [base64.b64decode(uri) for uri in uris] == [
        _large_png(index=index % 4) for index in range(8)
    ]
    assert document.html.endswith("<p>END</p></body></html>")
    # Escaping each URI character by character in Python took about 1 s per 2 MB.
    assert elapsed < 3


def _traced_peak(*, run) -> tuple[InlinedDocument, int]:
    tracemalloc.start()
    try:
        document = run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return document, peak


def test_assets_the_budget_refuses_are_not_kept_in_memory(large_artifact):
    """Only two images fit; the other ten are read, refused and let go."""
    limits = _limits(max_total_bytes=2 * LARGE_IMAGE_BYTES + 1024)
    images = "".join(f'<p><img src="big{index}.png"></p>' for index in range(LARGE_IMAGES))
    document, peak = _traced_peak(
        run=lambda: _inline(artifact=large_artifact, html=images.encode(), limits=limits)
    )
    assert document.html.count("data:image/png;base64,") == 2
    assert [entry.reason for entry in document.omitted] == [OmittedReason.BUDGET] * 10
    assert peak < 4 * limits.max_total_bytes


def test_a_dropped_declaration_lets_go_of_the_uris_it_placed(large_artifact):
    """Each rule inlines one image, then fails on a missing one, so every rule goes."""
    limits = _limits(max_total_bytes=2 * LARGE_IMAGE_BYTES + 1024)
    rules = "".join(
        f".b{index}{{background:url(big{index}.png),url(missing.png)}}"
        for index in range(LARGE_IMAGES)
    )
    html = f"<html><head><style>{rules}</style></head><body><p>END</p></body></html>"
    document, peak = _traced_peak(
        run=lambda: _inline(artifact=large_artifact, html=html.encode(), limits=limits)
    )
    assert "data:" not in document.html
    assert peak < 4 * limits.max_total_bytes


def _css(*, rules: int) -> str:
    return "".join(f".r{index}{{color:red}}" for index in range(rules))


def test_css_past_the_document_cap_is_dropped_and_the_text_stays(artifact):
    """A style that fits is kept; once too little of the cap is left, later CSS goes."""
    kept = _css(rules=40_000).ljust(artifact_inline._MAX_CSS_BYTES - 8)
    html = (
        f"<html><head><style>{kept}</style><style>.late{{color:blue}}</style></head>"
        f'<body><p style="color:green">END</p></body></html>'
    ).encode()
    document = _inline(artifact=artifact, html=html)
    assert ".r39999{color:red;}" in document.html
    assert ".late" not in document.html
    assert "color:green" not in document.html
    assert ">END</p>" in document.html
    assert TRUNCATION_NOTE not in document.html


def test_a_style_over_the_cap_alone_leaves_room_for_the_rest(artifact):
    """A refused style costs nothing, so smaller CSS after it still fits."""
    huge = _css(rules=80_000)
    assert len(huge) > artifact_inline._MAX_CSS_BYTES
    html = (
        f"<html><head><style>{huge}</style></head>"
        f'<body><p style="color:green">END</p></body></html>'
    ).encode()
    document = _inline(artifact=artifact, html=html)
    assert ".r0{" not in document.html
    assert "color:green" in document.html


def test_a_stylesheet_past_the_css_cap_is_left_out_as_budget(tmp_path):
    folder = tmp_path / "art"
    folder.mkdir()
    (folder / "big.css").write_text(_css(rules=80_000), encoding="utf-8")
    html = b'<html><head><link rel="stylesheet" href="big.css"></head><body><p>END</p></body></html>'
    document = _inline(artifact=Artifact(folder=folder, other=folder), html=html)
    assert [(entry.reason, entry.reference) for entry in document.omitted] == [
        (OmittedReason.BUDGET, "big.css")
    ]
    assert ".r0{" not in document.html


def test_inline_limits_hold_their_specified_defaults():
    limits = InlineLimits(max_total_bytes=1)
    assert (
        limits.max_asset_bytes,
        limits.max_references,
        limits.max_image_pixels,
        limits.max_import_depth,
        limits.max_nodes,
    ) == (2 * 1024 * 1024, 256, 40_000_000, 3, 50_000)
