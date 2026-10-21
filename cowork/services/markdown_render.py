"""Render a Markdown artifact to HTML for export.

The output is untrusted HTML: Python-Markdown passes raw HTML through and turns
``![x](url)`` into ``<img>``, so ``artifact_inline.inline_artifact_html`` runs
after it.
"""


def render_markdown(*, text: str) -> str:
    raise NotImplementedError
