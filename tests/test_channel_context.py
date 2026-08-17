"""ENG-591: channel turns get support-mode prompt guidance.

The desktop suffix lead must stay byte-identical (it participates in anton's
cache-stable prompt prefix); channel turns swap in chat/support guidance plus
optional per-binding operator instructions.
"""

import asyncio
import inspect
from types import SimpleNamespace

import cowork.services.artifact_autopublish as autopublish
import cowork.services.task_objects as task_objects
from cowork.harnesses.anton_harness.harness import AntonHarness, _turn_style_context
from cowork.harnesses.base import ChannelContext

DESKTOP_LEAD = (
    "The Anton CoWork desktop UI displays progress, tool usage, and actions "
    "as separate structured activity rows. Keep assistant text focused on the "
    "user-facing answer; do not narrate internal work with status phrases like "
    "\"I'll check\", \"let me query\", or \"I have access\" unless that wording "
    "is itself the final answer the user needs. "
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


def test_desktop_lead_is_byte_stable():
    assert _turn_style_context(None) == DESKTOP_LEAD


def test_desktop_lead_names_artifact_retrieval_path():
    """ENG-1636: the desktop lead must point users at the Live Artifacts panel
    and forbid the inert-local-path / fabricated-sandbox-link reflex."""
    text = _turn_style_context(None)
    assert "Live Artifacts" in text
    # Download is named unconditionally; Open is qualified as desktop-only,
    # since hosted web has no Open control (ENG-1636 review).
    assert "Download control" in text
    assert "Open, on desktop" in text
    assert "Open and Download controls" not in text
    assert "sandbox:/mnt/data" in text
    assert "does nothing when clicked" in text


def test_channel_variant_swaps_desktop_guidance():
    text = _turn_style_context(ChannelContext(channel_type="telegram", is_group=True))
    assert "telegram" in text
    assert "group chat" in text
    assert "plain-text" in text
    assert "sent into this chat automatically" in text
    assert "prefixed with the sender's name" in text
    assert "desktop UI displays" not in text and "activity rows" not in text


def test_dm_variant_includes_display_name():
    text = _turn_style_context(
        ChannelContext(channel_type="whatsapp", is_group=False, display_name="Lead chat")
    )
    assert "one-on-one direct chat" in text and "(Lead chat)" in text
    assert "prefixed with the sender's name" not in text


def test_operator_instructions_appended_only_when_set():
    with_instructions = _turn_style_context(
        ChannelContext(channel_type="telegram", is_group=True, instructions="Speak Spanish.")
    )
    assert "Operator instructions for this chat:" in with_instructions
    assert "Speak Spanish." in with_instructions
    without = _turn_style_context(ChannelContext(channel_type="telegram", is_group=True))
    assert "Operator instructions" not in without


class _FakeSession:
    async def turn_stream(self, user_input, *, turn_id=None):
        return
        yield  # noqa: marks this an async generator


def test_stream_response_forwards_channel_context(monkeypatch):
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(task_objects, "index_turn_artifacts", lambda *_a, **_k: ([], set(), None))
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)
    received = {}

    async def _fake_build(self, conversation, disabled_connections, channel_context=None):
        received["channel_context"] = channel_context
        return _FakeSession(), None, None

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)
    conversation = SimpleNamespace(
        id="conv-1", project_id="proj-1", project=SimpleNamespace(path="/tmp", name="tmp")
    )
    ctx = ChannelContext(channel_type="telegram", is_group=True)

    async def _drain():
        return [
            event
            async for event in AntonHarness().stream_response(
                conversation=conversation,
                input=[{"type": "text", "text": "hi"}],
                channel_context=ctx,
            )
        ]

    asyncio.run(_drain())
    assert received["channel_context"] is ctx


def test_harness_signatures_accept_channel_context():
    from cowork.harnesses.hermes_harness.harness import HermesHarness

    assert "channel_context" in inspect.signature(AntonHarness.stream_response).parameters
    assert "channel_context" in inspect.signature(HermesHarness.stream_response).parameters
