"""The shared page (PAGE_CSS) keeps the editor's column, font and block typography.

``fixtures/docs_typography/rules.json`` pins the numbers the editor (cowork,
components/doc/doc.css) and the page share; cowork checks its copy against
doc.css. See the fixture's README.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import pytest

from cowork.docs.model import SCHEMA, normalize_doc
from cowork.docs.render import FONT_CSS, PAGE_CSS, PDF_CSS, doc_html

_RULES_BYTES = (Path(__file__).parent / "fixtures" / "docs_typography" / "rules.json").read_bytes()
# The same digest is pinned in cowork's docTypography.test.js.
_RULES_SHA256 = "6591c429ba06e88e8d6b0a3582a22ab56b59f9254e1df2b365b0298fb45092fc"
_RULES = json.loads(_RULES_BYTES)
_SIDES = ("top", "right", "bottom", "left")


def _split_top_level(text: str, sep: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _selector(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"\s*([>+,])\s*", r"\1", text)


def _parse(css: str, prefix: str = "", rules: dict[str, list[tuple[str, str]]] | None = None):
    """Each selector's declarations in source order; a rule in @media is keyed ``@media (...) sel``."""
    rules = {} if rules is None else rules
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    i = 0
    while (brace := css.find("{", i)) != -1:
        head = css[i:brace].strip()
        depth, j = 1, brace + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        body = css[brace + 1:j - 1]
        if head.startswith("@media"):
            _parse(body, "@media " + re.sub(r"\s+", "", head[len("@media"):]) + " ", rules)
        elif not head.startswith("@"):
            decls = []
            for d in _split_top_level(body, ";"):
                name, _, value = d.partition(":")
                decls.append((name.strip().lower(), value.strip()))
            for sel in _split_top_level(head, ","):
                rules.setdefault(prefix + _selector(sel), []).extend(decls)
        i = j
    return rules


_PAGE = _parse(PAGE_CSS)


def _side(shorthand: str, side: str) -> str:
    p = shorthand.split()
    p += p[:1] if len(p) == 1 else []  # top right bottom left: a missing one repeats its opposite
    p += p[:1] if len(p) == 2 else []
    p += p[1:2] if len(p) == 3 else []
    return p[_SIDES.index(side)]


def _value(selectors: list[str], prop: str) -> str | None:
    """The last value ``prop`` gets across ``selectors``, from a longhand or its padding/margin shorthand."""
    found = None
    for sel in selectors:
        for name, value in _PAGE.get(sel, []):
            if name == prop:
                found = value
            elif name in ("padding", "margin") and prop.startswith(name + "-") and prop.split("-")[-1] in _SIDES:
                found = _side(value, prop.split("-")[-1])
    if found is None:
        return None
    # A var() resolves from the rule's own selectors (the page sets --doc-gutter on .page).
    return re.sub(r"var\((--[\w-]+)\)", lambda m: _value(selectors, m.group(1)) or m.group(0), found)


def _norm(value: str) -> str:
    value = re.sub(r"\s*,\s*", ",", value.strip().lower())
    return re.sub(r"(?<![\d.])0\.(\d)", r".\1", value)


def test_the_fixture_is_the_copy_cowork_pins():
    assert hashlib.sha256(_RULES_BYTES).hexdigest() == _RULES_SHA256


@pytest.mark.parametrize("rule", _RULES["rules"], ids=lambda r: r["page"][0])
def test_the_page_matches_the_editors_rule(rule):
    for prop, want in rule["css"].items():
        assert _norm(_value(rule["page"], prop) or "") == _norm(want), (rule["page"], prop)


def test_the_page_narrows_where_the_editor_does():
    narrow = _RULES["narrow_below_px"]
    assert f"@media (max-width:{narrow - 0.02:g}px) .page" in _PAGE


def test_lists_start_their_text_where_the_editor_does():
    px = {k: float(v.removesuffix("px")) for k, v in _RULES["list"].items()}
    mark = px["mark_min_width"] + px["mark_gap"]
    task = px["task_box"] + px["mark_gap"]
    assert _value([".tab ul"], "padding-left") == _value([".tab ol"], "padding-left") == f"{mark:g}px"
    assert _value([".tab li>ul"], "padding-left") == _value([".tab li>ol"], "padding-left") == f"{px['indent_step']:g}px"
    assert _value([".tab ul.tasks"], "padding-left") == f"{task:g}px"
    box = [".task-box"]
    assert (_value(box, "width"), _value(box, "margin-right"), _value(box, "margin-left")) == (
        f"{px['task_box']:g}px", f"{px['mark_gap']:g}px", f"-{task:g}px")


def test_the_page_carries_inter_and_the_pdf_export_the_same_font():
    faces = re.findall(r"@font-face\{font-family:Inter;font-style:(\w+);[^}]*base64,([^)]+)\)", FONT_CSS)
    assert [style for style, _ in faces] == ["normal", "italic"]
    assert all(base64.b64decode(data).startswith(b"wOF2") for _, data in faces)
    assert re.search(r"body\{[^}]*font:15px/1\.65 Inter,", PAGE_CSS)
    assert PAGE_CSS.startswith(FONT_CSS) and PDF_CSS.startswith(FONT_CSS)
    assert PDF_CSS.endswith("body{font-family:" + PAGE_CSS.split("font:15px/1.65 ", 1)[1].split(";", 1)[0] + "}")


def _page(*blocks: dict) -> str:
    doc = normalize_doc({"schema": SCHEMA, "title": "T", "rev": 1,
                         "tabs": [{"id": "t_1", "title": "T", "blocks": list(blocks)}]})
    page = doc_html(doc)
    return page[page.index("<main"):]


def test_a_task_shows_the_editors_box_on_the_page():
    page = _page({"type": "task", "text": "Ship", "done": True}, {"type": "task", "text": "Test"})
    assert ("<ul class=\"tasks\"><li class='task done'><span class='task-box'></span><span class='task-text'>Ship</span>"
            "</li><li class='task'><span class='task-box'></span><span class='task-text'>Test</span></li></ul>") in page
    assert "☑" not in page and "☐" not in page


def test_a_callout_shows_its_icon_and_its_title_on_a_line_of_its_own():
    page = _page({"type": "callout", "tone": "warning", "title": "Heads up", "text": "Body."})
    assert "<div class='callout callout-warning'><span class='callout-icon'><svg width='16'" in page
    assert "<div class='callout-body'><div class='callout-title'>Heads up</div><p>Body.</p></div></div>" in page
