"""GET /conversations/{id}/items grows an opt-in, cursor-based
`limit`/`before` pagination on ConversationService.get_messages_page.
Omitting both params keeps get_messages's existing unbounded bare-list
behavior untouched (asserted at the route level in
test_conversation_model_persistence.py / test_org_isolation_e2e.py, which
this must not break).
"""
from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.schemas.responses import Role
from cowork.services.conversations import ConversationService, InvalidPaginationParams


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    with Session(engine) as raw:
        yield ScopedSession(raw, LOCAL_SCOPE)


@pytest.fixture
def conversation(tmp_path, session):
    project = Project(name="P", path=str(tmp_path / "p"))
    session.add(project)
    session.commit()
    session.refresh(project)
    conv = Conversation(project_id=project.id, topic="t")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
_TOOL_CONTENT = [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]


def _add_message(
    session, conversation, *, role=Role.assistant, content="m", seq, minute=0,
    pending=False, content_is_tool_row=False,
):
    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=_TOOL_CONTENT if content_is_tool_row else content,
        seq=seq,
        created_at=_BASE_TIME + timedelta(minutes=minute),
        pending=pending,
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    return message


def _add_message_with_server_default_created_at(session, conversation, *, role, content, seq):
    """Like _add_message, but omits created_at so the DB's own
    `server_default=sa.func.now()` populates it — the way save_assistant_turn
    writes. save_user_message takes the opposite path and binds an explicit
    Python datetime (handlers/responses.py, channels/runtime.py), so a single
    turn stores the two formats side by side: SQLite keeps a bare
    second-precision string for one and a microsecond-formatted value for the
    other. Ordering must not depend on the two being comparable."""
    message = Message(conversation_id=conversation.id, role=role, content=content, seq=seq)
    session.add(message)
    session.commit()
    session.refresh(message)
    return message


def _add_visible_run(session, conversation, count, *, start_seq=0):
    """`count` plain visible messages, oldest to newest, alternating role."""
    messages = []
    for i in range(count):
        role = Role.user if i % 2 == 0 else Role.assistant
        messages.append(
            _add_message(session, conversation, role=role, content=f"m{i}", seq=start_seq + i, minute=i)
        )
    return messages


def _walk_all_pages(svc, conversation_id, *, limit):
    """Follow next_before until has_more is False; return (all_items, page_count)."""
    items: list[dict] = []
    cursor = None
    pages = 0
    seen_cursors = set()
    while True:
        pages += 1
        assert pages < 1000, "runaway pagination loop"
        page = svc.get_messages_page(conversation_id, limit=limit, before=cursor)
        items = page.items + items  # pages arrive newest-first; prepend to rebuild ascending order
        if not page.has_more:
            break
        assert page.next_before not in seen_cursors, "cursor did not advance"
        seen_cursors.add(page.next_before)
        cursor = page.next_before
    return items, pages


class TestPageBoundariesAndCursor:
    def test_first_page_is_the_most_recent_items_in_ascending_order(self, session, conversation):
        _add_visible_run(session, conversation, 5)
        svc = ConversationService(session)
        page = svc.get_messages_page(conversation.id, limit=2)
        assert [i["content"] for i in page.items] == ["m3", "m4"]
        assert page.has_more is True
        assert page.next_before is not None

    def test_cursor_walks_back_without_duplicate_or_skipped_messages(self, session, conversation):
        _add_visible_run(session, conversation, 11)
        svc = ConversationService(session)
        items, pages = _walk_all_pages(svc, conversation.id, limit=3)
        assert [i["content"] for i in items] == [f"m{i}" for i in range(11)]
        assert pages == 4  # 3+3+3+2

    def test_has_more_false_when_limit_covers_the_whole_conversation(self, session, conversation):
        _add_visible_run(session, conversation, 3)
        svc = ConversationService(session)
        page = svc.get_messages_page(conversation.id, limit=50)
        assert len(page.items) == 3
        assert page.has_more is False
        assert page.next_before is None

    def test_second_page_continues_strictly_before_the_first(self, session, conversation):
        _add_visible_run(session, conversation, 4)
        svc = ConversationService(session)
        first = svc.get_messages_page(conversation.id, limit=2)
        second = svc.get_messages_page(conversation.id, limit=2, before=first.next_before)
        assert [i["content"] for i in second.items] == ["m0", "m1"]
        assert second.has_more is False

    def test_cursor_advances_when_created_at_comes_from_the_db_server_default(self, session, conversation):
        # The real write path never passes an explicit created_at (see
        # _add_message_with_server_default_created_at) — SQLite's own
        # CURRENT_TIMESTAMP stores a bare second-precision string, while a
        # cursor built from a Python datetime binds with forced microsecond
        # formatting. A keyset comparison that leads on created_at compares
        # those two representations as permanently unequal-length strings,
        # which SQLite orders lexicographically: the boundary row's
        # created_at then never satisfies `<`, so every "before" query
        # matches every row again and every page repeats the same content
        # forever. Keying the cursor on `seq` instead (monotonic, assigned
        # at insert, independent of created_at's storage format) is immune
        # to this regardless of which write path populated the timestamp.
        for i in range(4):
            role = Role.user if i % 2 == 0 else Role.assistant
            _add_message_with_server_default_created_at(
                session, conversation, role=role, content=f"m{i}", seq=i,
            )
        svc = ConversationService(session)
        items, pages = _walk_all_pages(svc, conversation.id, limit=2)
        assert [i["content"] for i in items] == ["m0", "m1", "m2", "m3"]
        assert pages == 2


class TestToolRowFiltering:
    def test_tool_rows_beyond_scan_cap_yield_an_empty_page_with_a_usable_cursor(self, session, conversation):
        # limit=1 -> scan_cap=10 raw rows; 12 consecutive tool rows means the
        # first 11 scanned (scan_cap + 1 lookahead) are all hidden, so this
        # page comes back with nothing to show but has_more True and a
        # cursor the client can immediately continue from.
        for i in range(12):
            _add_message(session, conversation, seq=i, minute=i, content_is_tool_row=True)
        svc = ConversationService(session)
        page = svc.get_messages_page(conversation.id, limit=1)
        assert page.items == []
        assert page.has_more is True
        assert page.next_before is not None
        # Continuing from that cursor reaches the one real message underneath.
        _add_message(session, conversation, seq=-1, minute=-1, content="oldest")
        page2 = svc.get_messages_page(conversation.id, limit=1, before=page.next_before)
        assert [i["content"] for i in page2.items] == ["oldest"]

    def test_tool_rows_interleaved_with_visible_messages_are_skipped_not_counted(self, session, conversation):
        _add_message(session, conversation, seq=0, minute=0, content="visible-1")
        _add_message(session, conversation, seq=1, minute=1, content_is_tool_row=True)
        _add_message(session, conversation, seq=2, minute=2, content="visible-2")
        svc = ConversationService(session)
        page = svc.get_messages_page(conversation.id, limit=2)
        assert [i["content"] for i in page.items] == ["visible-1", "visible-2"]
        assert page.has_more is False


class TestPendingRowInclusion:
    def test_pending_row_is_included_matching_get_messages(self, session, conversation):
        _add_message(session, conversation, seq=0, minute=0, content="q", role=Role.user, pending=True)
        svc = ConversationService(session)
        page = svc.get_messages_page(conversation.id, limit=10)
        assert [i["content"] for i in page.items] == ["q"]


class TestPagedAndUnboundedReadsAgree:
    """The two branches of GET /items must render the same history, and both
    must render it the right way round.

    An earlier version of this class built every row through `_add_message`,
    which always passes an explicit created_at. Uniform timestamp formatting
    is exactly the shape where the two orders cannot disagree, so the tests
    could not fail for the reason the property exists. The real write path
    does not look like that: save_user_message binds a Python datetime
    (microsecond precision) and save_assistant_turn lets the column default
    fire (second precision on SQLite), and SQLite compares those strings
    lexicographically.
    """

    def _conversation_via_the_real_write_path(self, session, conversation):
        """One turn persisted exactly the way a live turn persists it.

        Asserts the two rows really did land in different timestamp formats.
        Without that check these tests would quietly stop guarding anything the
        moment the writes straddled a second boundary — the old created_at-led
        order sorts them correctly then, so a revert would pass.
        """
        svc = ConversationService(session)
        svc.save_user_message(
            conversation.id, "question", created_at=datetime.now(timezone.utc)
        )
        svc.save_assistant_turn(conversation.id, "answer", [{"type": "response.completed"}])

        rows = {
            m.role: m.created_at
            for m in session.exec(
                session.select(Message).where(Message.conversation_id == conversation.id)
            ).all()
        }
        user_at, assistant_at = str(rows[Role.user]), str(rows[Role.assistant])
        assert user_at[:19] == assistant_at[:19], (
            f"fixture straddled a second boundary ({user_at} vs {assistant_at}); "
            "the format divergence these tests guard is not exercised"
        )
        assert "." in user_at and "." not in assistant_at, (
            f"expected the two write paths to store different precisions, got "
            f"user={user_at!r} assistant={assistant_at!r}"
        )
        return svc

    def test_a_turn_reads_question_before_answer_on_both_paths(self, session, conversation):
        svc = self._conversation_via_the_real_write_path(session, conversation)

        unbounded = [m["content"] for m in svc.get_messages(conversation.id)]
        paged = [m["content"] for m in svc.get_messages_page(conversation.id, limit=50).items]

        assert unbounded == ["question", "answer"]
        assert paged == unbounded

    def test_replayed_llm_history_matches_what_the_ui_renders(self, session, conversation):
        # get_ordered_messages feeds the harness. If it disagrees with the UI
        # the model is replayed a conversation the user never had.
        svc = self._conversation_via_the_real_write_path(session, conversation)

        replayed = [m.content for m in svc.get_ordered_messages(conversation.id)]
        rendered = [m["content"] for m in svc.get_messages(conversation.id)]

        assert replayed == ["question", "answer"]
        assert replayed == rendered

    def test_page_walk_reproduces_the_unbounded_list(self, session, conversation):
        svc = ConversationService(session)
        for i in range(6):
            role = Role.user if i % 2 == 0 else Role.assistant
            _add_message(session, conversation, role=role, content=f"m{i}", seq=i, minute=i)

        items, _ = _walk_all_pages(svc, conversation.id, limit=2)

        unbounded = svc.get_messages(conversation.id)
        assert [i["id"] for i in items] == [m["id"] for m in unbounded]
        assert [i["content"] for i in items] == [f"m{i}" for i in range(6)]

    def test_rows_sharing_one_created_at_stay_in_seq_order(self, session, conversation):
        # One turn's block-messages commit together, so they share a
        # second-precision created_at and only seq separates them.
        svc = ConversationService(session)
        for i in range(6):
            role = Role.user if i % 2 == 0 else Role.assistant
            _add_message(session, conversation, role=role, content=f"m{i}", seq=i, minute=0)

        items, _ = _walk_all_pages(svc, conversation.id, limit=2)

        assert [i["content"] for i in items] == [f"m{i}" for i in range(6)]
        assert [i["id"] for i in items] == [m["id"] for m in svc.get_messages(conversation.id)]


class TestValidation:
    def test_limit_zero_is_rejected(self, session, conversation):
        with pytest.raises(InvalidPaginationParams):
            ConversationService(session).get_messages_page(conversation.id, limit=0)

    def test_limit_over_the_cap_is_rejected(self, session, conversation):
        with pytest.raises(InvalidPaginationParams):
            ConversationService(session).get_messages_page(conversation.id, limit=10_000)

    def test_malformed_cursor_is_rejected(self, session, conversation):
        with pytest.raises(InvalidPaginationParams):
            ConversationService(session).get_messages_page(conversation.id, before="not-a-real-cursor")

    @pytest.mark.parametrize("payload", [
        [10**30, 1, "00000000-0000-0000-0000-000000000001"],   # seq wider than the column
        [-1, 1, "00000000-0000-0000-0000-000000000001"],       # seq below any real row
        [0, 7, "00000000-0000-0000-0000-000000000001"],        # role rank outside {0, 1}
    ])
    def test_out_of_range_cursor_is_rejected_not_raised_from_the_driver(
        self, session, conversation, payload
    ):
        # `before` is unsigned client input. A decodable but out-of-range
        # value used to reach the driver and surface as a 500 with a
        # SQLAlchemy traceback instead of this service's own 400.
        cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        with pytest.raises(InvalidPaginationParams):
            ConversationService(session).get_messages_page(conversation.id, before=cursor)


class TestOrgIsolation404Parity:
    def test_unknown_and_foreign_conversation_ids_fail_the_same_way(self, tmp_path, engine):
        org_a = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id="org-a", user_id="user-a"))
        org_b = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id="org-b", user_id="user-b"))

        project = Project(name="P", path=str(tmp_path / "p"))
        org_a.add(project)
        org_a.commit()
        org_a.refresh(project)
        conv = Conversation(project_id=project.id, topic="t")
        org_a.add(conv)
        org_a.commit()
        org_a.refresh(conv)

        with pytest.raises(ValueError) as foreign_exc:
            ConversationService(org_b).get_messages_page(conv.id, limit=10)
        with pytest.raises(ValueError) as unknown_exc:
            ConversationService(org_b).get_messages_page(uuid.uuid4(), limit=10)
        assert not isinstance(foreign_exc.value, InvalidPaginationParams)
        assert not isinstance(unknown_exc.value, InvalidPaginationParams)
        assert str(foreign_exc.value) == str(unknown_exc.value)
