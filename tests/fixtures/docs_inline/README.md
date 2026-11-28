# Shared inline Markdown cases

`cases.json` lists a doc block's inline Markdown and the HTML it renders to. The same text must show the same marks on the shared page and in the exports (`cowork/docs/render.py`, `inline_html`) as in the editor (cowork, `src/renderer/cowork/components/doc/docModel.js`, `inlineToHtml`).

cowork keeps a byte-identical copy at `src/renderer/cowork/__fixtures__/docs-inline/cases.json`. Change both files together, in the PRs that change either renderer, and pin the new sha256 in both test files (a test in each repo checks it):

- here, `tests/test_docs_inline.py` checks that `inline_html` returns exactly each case's `html`;
- in cowork, `docModel.inline.test.js` checks `inlineToHtml` after setting aside the editor's own attributes (a link's `target` and `rel`, chip classes and data, `&quot;` in text). A `{{user:me}}` chip differs on purpose: the editor shows it in the app's language, so the cases leave it out.
