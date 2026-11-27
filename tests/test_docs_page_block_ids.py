"""Each block on a doc's page carries the id the comment layer anchors to."""

from __future__ import annotations

import pytest

from cowork.docs.markdown import parse_markdown
from cowork.docs.model import SCHEMA, normalize_doc
from cowork.docs.render import doc_html, export_body_html

MARKDOWN = (
    "# Title\n\nA paragraph.\n\n> A quote.\n\n```\ncode\n```\n\n---\n\n"
    "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n![logo](https://example.com/a.png)\n"
)
EXTRA = [
    {"type": "callout", "tone": "info", "text": "Note."},
    {"type": "chart", "chart": "bar", "title": "Revenue", "x": ["N", "S"],
     "series": [{"name": "2025", "values": [1, 2]}]},
    {"type": "metrics", "items": [{"label": "Users", "value": "12"}]},
]
ROOT_TAG = {
    "heading": "h1", "paragraph": "p", "quote": "blockquote", "code": "pre", "divider": "hr",
    "table": "div", "image": "figure", "callout": "div", "chart": "figure", "metrics": "div",
}


def _doc(blocks: list[dict]) -> dict:
    return normalize_doc({"schema": SCHEMA, "title": "T", "rev": 1,
                          "tabs": [{"id": "t_1", "title": "T", "blocks": blocks}]})


def _numbered(blocks: list[dict]) -> list[dict]:
    return [{**b, "id": f"b_{i:08d}"} for i, b in enumerate(blocks)]


def test_every_block_root_carries_its_page_id():
    blocks = _numbered(parse_markdown(MARKDOWN) + EXTRA)
    page = doc_html(_doc(blocks))
    for b in blocks:
        assert f'<{ROOT_TAG[b["type"]]} id="blk-{b["id"]}"' in page, b["type"]
        assert page.count(f'id="blk-{b["id"]}"') == 1, b["type"]


def test_each_list_item_carries_its_page_id_nested_and_tasks_too():
    blocks = _numbered(parse_markdown("- one\n  - nested\n- [x] done\n1. first\n"))
    page = doc_html(_doc(blocks))
    one, nested, done, first = (b["id"] for b in blocks)
    assert f'<li id="blk-{one}">one' in page
    assert f'<li id="blk-{nested}">nested' in page
    assert f"<li id=\"blk-{done}\" class='task done'>" in page
    assert f'<li id="blk-{first}">first' in page


def test_word_and_pdf_exports_carry_no_page_ids():
    blocks = _numbered(parse_markdown(MARKDOWN + "\n- one\n- [ ] todo\n") + EXTRA)
    # Word's markup has no code spans and the PDF's has them; neither has page ids.
    for highlight in (False, True):
        assert 'id="blk-' not in export_body_html(_doc(blocks), highlight=highlight)


@pytest.mark.parametrize("bad", [7, "b 1", 'b"1', "<b>", "x" * 65])
def test_a_block_id_the_layer_could_not_use_gets_no_page_id(bad):
    page = doc_html(_doc([{"id": bad, "type": "paragraph", "text": "Hi"},
                          {"id": bad, "type": "bullet", "text": "Item"}]))
    assert "<p>Hi</p>" in page and "<li>Item</li>" in page
    assert 'id="blk-' not in page


def test_a_pending_block_still_shows_nothing():
    page = doc_html(_doc([{"id": "b_00000001", "type": "pending"}]))
    assert 'id="blk-' not in page
