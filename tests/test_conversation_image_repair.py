"""Repairing a conversation's stored history after ContentValidationError
(ENG-1992): a provider permanently rejected an image content block, and
retrying the identical request fails identically forever, since the same
translation runs fresh from stored history on every call. Fixing the DATA
once — stripping the offending image blocks — means every future turn just
replays clean, with no ongoing flag or per-turn filtering needed.
"""
from __future__ import annotations

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.message import Message
from cowork.services.conversations import ConversationService, _strip_image_blocks
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture
def svc(session):
    return ConversationService(ScopedSession(session, LOCAL_SCOPE))


@pytest.fixture
def conv(svc):
    return svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)


# ─── _strip_image_blocks — the pure block-level transform ───────────────────


def test_top_level_image_block_is_replaced_with_a_placeholder():
    content = [
        {"type": "text", "text": "here's a screenshot"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
    ]
    new_content, changed = _strip_image_blocks(content)
    assert changed is True
    assert new_content[0] == content[0]
    assert new_content[1]["type"] == "text"
    assert "removed" in new_content[1]["text"]


def test_image_nested_in_a_tool_result_is_also_replaced():
    # A tool (e.g. a screenshot tool) can return an image inside its own
    # tool_result content list — the block-shape the ENG-1992 translators
    # (anton's core/llm/openai.py) walk when building a provider request.
    content = [{
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": [
            {"type": "text", "text": "here you go"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ],
    }]
    new_content, changed = _strip_image_blocks(content)
    assert changed is True
    nested = new_content[0]["content"]
    assert nested[0] == {"type": "text", "text": "here you go"}
    assert nested[1]["type"] == "text"
    assert "removed" in nested[1]["text"]
    # tool_use_id and every other field on the tool_result block survive.
    assert new_content[0]["tool_use_id"] == "call_1"


def test_tool_result_with_string_content_is_left_alone():
    content = [{"type": "tool_result", "tool_use_id": "call_1", "content": "plain text result"}]
    new_content, changed = _strip_image_blocks(content)
    assert changed is False
    assert new_content is content


def test_content_with_no_images_is_returned_unchanged():
    content = [{"type": "text", "text": "hello"}]
    new_content, changed = _strip_image_blocks(content)
    assert changed is False
    assert new_content is content  # same object — callers use this to skip a write


def test_string_content_passes_through_unchanged():
    new_content, changed = _strip_image_blocks("plain string message")
    assert changed is False
    assert new_content == "plain string message"


def test_non_dict_blocks_in_content_are_preserved():
    content = ["not a dict", {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]
    new_content, changed = _strip_image_blocks(content)
    assert changed is True
    assert new_content[0] == "not a dict"


# ─── ConversationService.repair_image_content — the persisted-data repair ───


def test_repair_strips_images_and_returns_changed_message_ids(svc, conv, session):
    poisoned = Message(
        conversation_id=conv.id, role="user",
        content=[{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}],
    )
    clean = Message(conversation_id=conv.id, role="assistant", content=[{"type": "text", "text": "hi"}])
    session.add(poisoned)
    session.add(clean)
    session.commit()
    session.refresh(poisoned)
    session.refresh(clean)

    repaired = svc.repair_image_content(conv.id)

    assert repaired == [poisoned.id]
    session.refresh(poisoned)
    assert poisoned.content[0]["type"] == "text"
    session.refresh(clean)
    assert clean.content == [{"type": "text", "text": "hi"}]  # untouched


def test_repair_persists_across_a_fresh_session_read(svc, conv, session):
    # The point of the repair is that a LATER turn's replay sees clean data —
    # prove the write actually committed, not just mutated the in-memory row.
    session.add(Message(
        conversation_id=conv.id, role="user",
        content=[{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}],
    ))
    session.commit()

    svc.repair_image_content(conv.id)

    reread = ConversationService(ScopedSession(session, LOCAL_SCOPE)).get_ordered_messages(conv.id)
    assert reread[0].content[0]["type"] == "text"


def test_repair_is_a_noop_when_nothing_needs_it(svc, conv, session):
    session.add(Message(conversation_id=conv.id, role="user", content=[{"type": "text", "text": "hi"}]))
    session.commit()

    assert svc.repair_image_content(conv.id) == []


def test_repair_scans_pending_rows_too(svc, conv, session):
    # A poisoned image could be in the in-flight turn that just failed, not
    # only settled history — pending rows must be included in the scan.
    session.add(Message(
        conversation_id=conv.id, role="user", pending=True,
        content=[{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}],
    ))
    session.commit()

    assert len(svc.repair_image_content(conv.id)) == 1
