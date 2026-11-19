"""The lead of a UI turn's system-prompt suffix.

It must stay byte-identical: it is part of anton's cache-stable prompt prefix.
"""

from cowork.harnesses.anton_harness.harness import _TURN_STYLE_CONTEXT

UI_LEAD = (
    "The Cowork UI displays progress, tool usage, and actions "
    "as separate structured activity rows, so do not narrate each step "
    "(\"let me check\", \"now I'll query\"). When a request needs tool work, "
    "open your reply with one short sentence telling the user what you are "
    "about to do; after that, keep assistant text focused on the user-facing "
    "answer. "
    "Files you create as artifacts appear automatically in the Live Artifacts "
    "panel beside the chat, where the user previews them and uses the Download "
    "control (and Open, on desktop). When a file is ready, tell the user it is "
    "in the Live Artifacts panel and can be downloaded there — do NOT hand them "
    "its location on disk. Never put a file's local path (for example "
    "C:\\Users\\... or /Users/...) into your reply as a markdown link or as "
    "text: such a link does nothing when clicked in chat, and the bare path "
    "only exposes the user's machine layout. Never invent a download URL such "
    "as sandbox:/mnt/data/...; no link of that form exists. If the user says "
    "they cannot find or download the file, point them again at the Live "
    "Artifacts panel's Download control (and Open, on desktop) — never repeat "
    "the path."
)


def test_ui_lead_is_byte_stable():
    assert _TURN_STYLE_CONTEXT == UI_LEAD


def test_desktop_lead_asks_for_an_opening_sentence_not_step_narration():
    """The user sees text within seconds of asking, without a running commentary:
    one sentence of intent before tool work, no "let me check" narration after it."""
    text = _TURN_STYLE_CONTEXT
    assert "open your reply with one short sentence" in text
    assert "do not narrate each step" in text
    assert "Keep assistant text focused" not in text  # the old line forbade the opening sentence too


def test_desktop_lead_names_artifact_retrieval_path():
    """The desktop lead must point users at the Live Artifacts panel
    and forbid the inert-local-path / fabricated-sandbox-link reflex."""
    text = _TURN_STYLE_CONTEXT
    assert "Live Artifacts" in text
    # Download is named unconditionally; Open is qualified as desktop-only,
    # since hosted web has no Open control.
    assert "Download control" in text
    assert "Open, on desktop" in text
    assert "Open and Download controls" not in text
    assert "sandbox:/mnt/data" in text
    assert "does nothing when clicked" in text
