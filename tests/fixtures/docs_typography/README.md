# Shared doc typography

`rules.json` pins the numbers a doc's shared page (`cowork/docs/render.py`, `PAGE_CSS`) takes from the editor (cowork, `src/renderer/cowork/components/doc/doc.css`), so a paragraph wraps at the same words in both:

- `rules`: each entry names the editor's selectors and the page's selectors (later ones override earlier ones, as in CSS) and the values both must declare. `padding-*`/`margin-*` may come from the shorthand, and `var()` resolves from the rule itself (and, in the editor, `.doc-root` and the app's tokens);
- `narrow_below_px`: the pane width under which the editor narrows its gutter (`DocPane.jsx`); the page does the same in a media query;
- `list`: where list text starts (the editor's mark width and gap, the per-level indent from `TextBlock.jsx`, the to-do box).

cowork keeps a byte-identical copy at `src/renderer/cowork/__fixtures__/docs-typography/rules.json`. Change both files together, in the PRs that change either side's look, and pin the new sha256 in both test files:

- here, `tests/test_docs_page_typography.py` checks `PAGE_CSS`;
- in cowork, `docTypography.test.js` checks `doc.css` and `docPane.css`.

The page also embeds Inter (`cowork/docs/page_fonts`, built by `scripts/build_doc_page_fonts.py` from the app's own font files), so a reader without Inter installed sees the editor's font.
