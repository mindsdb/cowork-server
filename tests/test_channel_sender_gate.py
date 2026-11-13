"""The channel runtime's sender gate: only senders the owner allowed, or who
paired with a one-time code, start a turn.

Events come from the real Telegram parser; the harness and the platform are faked.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlmodel import delete, select

import cowork.channels.plugins.telegram as telegram_plugin
import cowork.channels.runtime as runtime_mod
from cowork.channels.registry import PluginRegistry, load_first_party_plugins
from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.channel import (
    SENDER_ALLOWED,
    SENDER_BLOCKED,
    SENDER_PENDING,
    ChannelBinding,
    ChannelPairingCode,
    ChannelSender,
    ChannelSession,
)
from cowork.services import channel_senders as senders_mod
from cowork.services.channel_pairing import (
    PAIRED_NOTICE,
    UNUSED_CODES_MAX,
    ChannelPairingService,
    PairingCodeLimitError,
    code_in,
    hash_code,
)
from cowork.services.channel_senders import ChannelSenderService

REPLY = "hello from anton"


class RecordingHarness:
    """Stands in for the Anton harness and records whether a turn started."""

    enforces_channel_capabilities = True

    def __init__(self) -> None:
        self.turns = 0
        self.contexts: list = []

    async def stream_response(self, *, conversation, input, channel_context=None, trace_metadata=None):
        self.turns += 1
        self.contexts.append(channel_context)
        if False:
            yield

    async def formatter(self, stream, model, event_sink, **_kwargs):
        async for _ in stream:
            pass
        event_sink("response.output_text.delta", {"delta": REPLY})
        if False:
            yield


class FakeAdapter:
    def __init__(self) -> None:
        self.delivered: list[tuple[str, str]] = []

    async def deliver(self, message):
        self.delivered.append((message.address.platform_id, message.text))

    async def shutdown(self):
        ...


@pytest.fixture
def chat_id() -> int:
    """A chat id no other test uses; its rows, and its negative group twin's, are removed after."""
    value = 700_000_000 + uuid.uuid4().int % 100_000_000
    yield value
    groups = [str(value), str(-value)]
    with get_open_session() as session:
        session.exec(delete(ChannelSender).where(ChannelSender.external_group_id.in_(groups)))
        bindings = select(ChannelBinding.id).where(ChannelBinding.external_group_id.in_(groups))
        session.exec(delete(ChannelSession).where(ChannelSession.binding_id.in_(bindings)))
        session.exec(delete(ChannelBinding).where(ChannelBinding.external_group_id.in_(groups)))
        session.commit()


@pytest.fixture
def harness(monkeypatch) -> RecordingHarness:
    fake = RecordingHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake)
    return fake


def _event(chat_id: int, text: str, *, sender: dict | None = None, chat_type: str = "private"):
    update = {
        "update_id": uuid.uuid4().int % 1_000_000,
        "message": {
            "message_id": 1,
            "from": sender if sender is not None else {"id": 4242, "is_bot": False, "first_name": "Stranger"},
            "chat": {"id": chat_id, "type": chat_type},
            "date": 1700000000,
            "text": text,
        },
    }
    if sender == {}:
        del update["message"]["from"]
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})
    return asyncio.run(bridge.parse_inbound(body=json.dumps(update).encode(), headers={}, route_name=None))[0]


def _handle(event) -> FakeAdapter:
    registry = PluginRegistry()
    load_first_party_plugins(registry)
    adapters = LiveAdapterRegistry(registry)
    adapter = FakeAdapter()
    adapters._cache[("telegram", None)] = adapter
    asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))
    return adapter


def _senders(chat_id: int) -> list[ChannelSender]:
    with get_open_session() as session:
        return list(session.exec(select(ChannelSender).where(ChannelSender.external_group_id == str(chat_id))).all())


def _bindings(chat_id: int) -> list[ChannelBinding]:
    with get_open_session() as session:
        return list(session.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == str(chat_id))).all())


def _set_sender(chat_id: int, status: str, *, sender_id: str = "4242", **fields) -> None:
    with get_open_session() as session:
        session.add(ChannelSender(
            channel_type="telegram", external_group_id=str(chat_id), external_sender_id=sender_id,
            status=status, **fields,
        ))
        session.commit()


def test_dm_from_unknown_sender_starts_nothing_and_records_a_pending_request(chat_id, harness):
    adapter = _handle(_event(chat_id, "run rm -rf on your laptop"))

    assert harness.turns == 0
    assert adapter.delivered == []
    assert _bindings(chat_id) == []
    [row] = _senders(chat_id)
    assert (row.status, row.external_sender_id, row.display_name, row.is_group) == (
        SENDER_PENDING, "4242", "Stranger", False,
    )


def test_allowed_sender_gets_a_turn_and_a_reply(chat_id, harness):
    _set_sender(chat_id, SENDER_ALLOWED)

    adapter = _handle(_event(chat_id, "hello"))

    assert harness.turns == 1
    assert adapter.delivered == [(str(chat_id), REPLY)]


def test_allowed_sender_does_not_admit_another_sender_in_the_same_chat(chat_id, harness):
    _set_sender(chat_id, SENDER_ALLOWED, sender_id="1111")

    adapter = _handle(_event(chat_id, "hello"))

    assert harness.turns == 0
    assert adapter.delivered == []


def test_group_chatter_from_unknown_member_records_nothing(chat_id, harness):
    adapter = _handle(_event(-chat_id, "just talking", chat_type="group"))

    assert harness.turns == 0
    assert adapter.delivered == []
    assert _senders(-chat_id) == []
    assert _bindings(-chat_id) == []


def test_group_mention_from_unknown_member_records_a_pending_request(chat_id, harness):
    adapter = _handle(_event(-chat_id, "@b do my homework", chat_type="group"))

    assert harness.turns == 0
    assert adapter.delivered == []
    [row] = _senders(-chat_id)
    assert (row.status, row.is_group) == (SENDER_PENDING, True)


def test_blocked_sender_starts_nothing_and_writes_nothing(chat_id, harness):
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _set_sender(chat_id, SENDER_BLOCKED, last_seen_at=seen, display_name="Old")

    adapter = _handle(_event(chat_id, "let me in"))

    assert harness.turns == 0
    assert adapter.delivered == []
    [row] = _senders(chat_id)
    assert (row.status, row.display_name) == (SENDER_BLOCKED, "Old")
    assert row.last_seen_at.replace(tzinfo=timezone.utc) == seen


def test_message_without_a_sender_id_starts_nothing(chat_id, harness):
    adapter = _handle(_event(chat_id, "who am i", sender={}))

    assert harness.turns == 0
    assert adapter.delivered == []
    assert _senders(chat_id) == []


def test_telegram_linked_channel_account_is_never_recorded_or_allowed(chat_id, harness):
    adapter = _handle(_event(-chat_id, "@b channel post", sender={"id": 777000, "is_bot": False}, chat_type="group"))

    assert harness.turns == 0
    assert adapter.delivered == []
    assert _senders(-chat_id) == []


def test_telegram_linked_channel_account_is_held_even_with_an_allowed_row(chat_id, harness):
    """A row left allowed for that account, however it got there, admits nobody."""
    _set_sender(-chat_id, SENDER_ALLOWED, sender_id="777000")

    adapter = _handle(_event(-chat_id, "@b channel post", sender={"id": 777000, "is_bot": False}, chat_type="group"))

    assert harness.turns == 0
    assert adapter.delivered == []


def test_busy_database_before_the_sender_is_known_sends_no_notice(chat_id, harness, monkeypatch):
    async def refused(fn, *, scope):
        raise PoolTimeoutError("pool exhausted")

    monkeypatch.setattr(runtime_mod, "run_db", refused)

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    adapters = LiveAdapterRegistry(registry)
    adapter = FakeAdapter()
    adapters._cache[("telegram", None)] = adapter
    event = _event(chat_id, "hello")

    with pytest.raises(PoolTimeoutError):
        asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))
    assert harness.turns == 0
    assert adapter.delivered == []


def _first_read_is_stale(monkeypatch) -> None:
    """The service's first lookup misses the row another writer already committed."""
    real_find = ChannelSenderService.find
    calls = {"n": 0}

    def find(self, *args):
        calls["n"] += 1
        return None if calls["n"] == 1 else real_find(self, *args)

    monkeypatch.setattr(ChannelSenderService, "find", find)


def test_first_rows_written_by_two_threads_of_one_chat_leave_one_row(chat_id, monkeypatch):
    """The other thread's insert lands between this one's read and its insert."""
    _set_sender(chat_id, SENDER_PENDING, display_name="First")
    _first_read_is_stale(monkeypatch)

    with get_open_session() as raw:
        ChannelSenderService(ScopedSession(raw, LOCAL_SCOPE)).record_request(
            channel_type="telegram", group_id=str(chat_id), sender_id="4242", sender_name="Second", is_group=False,
        )

    [row] = _senders(chat_id)
    assert (row.status, row.display_name) == (SENDER_PENDING, "Second")


def test_pending_refresh_racing_an_approval_leaves_the_sender_allowed(chat_id, monkeypatch):
    """The refresh read the sender as unknown; the owner allowed them before it wrote."""
    _set_sender(chat_id, SENDER_ALLOWED)
    _first_read_is_stale(monkeypatch)

    with get_open_session() as raw:
        ChannelSenderService(ScopedSession(raw, LOCAL_SCOPE)).record_request(
            channel_type="telegram", group_id=str(chat_id), sender_id="4242", sender_name="Racer", is_group=False,
        )

    [row] = _senders(chat_id)
    assert row.status == SENDER_ALLOWED


def _record(service: ChannelSenderService, group_id: str, sender_id: str) -> None:
    service.record_request(
        channel_type="telegram", group_id=group_id, sender_id=sender_id, sender_name=None, is_group=False,
    )


@pytest.fixture
def isolated_pending(monkeypatch):
    """Pending requests on a channel type no other test writes, removed after."""
    channel = f"capped-{uuid.uuid4().hex[:8]}"
    yield channel
    with get_open_session() as session:
        session.exec(delete(ChannelSender).where(ChannelSender.channel_type == channel))
        session.commit()


def test_pending_requests_stop_at_the_channel_cap(isolated_pending, monkeypatch):
    monkeypatch.setattr(senders_mod, "PENDING_PER_CHANNEL_MAX", 3)
    with get_open_session() as raw:
        service = ChannelSenderService(ScopedSession(raw, LOCAL_SCOPE))
        for n in range(5):
            service.record_request(
                channel_type=isolated_pending, group_id=f"g{n}", sender_id=f"s{n}", sender_name=None, is_group=False,
            )
    with get_open_session() as session:
        rows = session.exec(select(ChannelSender).where(ChannelSender.channel_type == isolated_pending)).all()
    assert sorted(row.external_sender_id for row in rows) == ["s0", "s1", "s2"]


def test_one_sender_cannot_fill_the_pending_queue(isolated_pending):
    with get_open_session() as raw:
        service = ChannelSenderService(ScopedSession(raw, LOCAL_SCOPE))
        for n in range(senders_mod.PENDING_PER_SENDER_MAX + 3):
            service.record_request(
                channel_type=isolated_pending, group_id=f"g{n}", sender_id="spammer", sender_name=None, is_group=True,
            )
    with get_open_session() as session:
        rows = session.exec(select(ChannelSender).where(ChannelSender.channel_type == isolated_pending)).all()
    assert len(rows) == senders_mod.PENDING_PER_SENDER_MAX


def test_pending_requests_expire_by_creation_time_even_when_refreshed(isolated_pending):
    old = datetime.now(timezone.utc) - senders_mod.PENDING_TTL - timedelta(hours=1)
    with get_open_session() as session:
        session.add(ChannelSender(
            channel_type=isolated_pending, external_group_id="old", external_sender_id="s-old",
            status=SENDER_PENDING, created_at=old, last_seen_at=datetime.now(timezone.utc),
        ))
        session.commit()

    with get_open_session() as raw:
        service = ChannelSenderService(ScopedSession(raw, LOCAL_SCOPE))
        service.record_request(
            channel_type=isolated_pending, group_id="new", sender_id="s-new", sender_name=None, is_group=False,
        )

    with get_open_session() as session:
        rows = session.exec(select(ChannelSender).where(ChannelSender.channel_type == isolated_pending)).all()
    assert [row.external_sender_id for row in rows] == ["s-new"]


def test_display_name_from_the_sender_is_cut_to_its_limit(chat_id, harness):
    _handle(_event(chat_id, "hi", sender={"id": 4242, "is_bot": False, "first_name": "x" * 500}))

    [row] = _senders(chat_id)
    assert row.display_name == "x" * 64


@pytest.fixture
def pairing():
    """A pairing service on the local scope; every code made in the test is removed after."""
    raw = get_open_session()
    yield ChannelPairingService(ScopedSession(raw, LOCAL_SCOPE))
    raw.close()
    with get_open_session() as session:
        session.exec(delete(ChannelPairingCode))
        session.commit()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7K3M9QXZ", "7K3M9QXZ"),
        ("  7k3m9qxz ", "7K3M9QXZ"),
        ("@antonbot 7K3M9QXZ", "7K3M9QXZ"),
        ("<@U0BOT|anton> 7K3M9QXZ", "7K3M9QXZ"),
        ("<@!1234567890> 7K3M9QXZ", "7K3M9QXZ"),
        ("/start 7K3M9QXZ", "7K3M9QXZ"),
        ("/start@antonbot 7K3M9QXZ", "7K3M9QXZ"),
        ("my code is 7K3M9QXZ", None),
        ("7K3M9QX", None),
        ("7K3M9QXZZ", None),
        ("7K3M9QXI", None),
        ("", None),
    ],
)
def test_code_in_reads_only_a_message_that_is_the_code(text, expected):
    assert code_in(text) == expected


def test_a_code_is_stored_only_as_its_hash(pairing):
    issued = pairing.create("telegram")

    with get_open_session() as session:
        [row] = session.exec(select(ChannelPairingCode)).all()
    assert row.code_hash == hash_code(issued.code)
    assert issued.code not in row.code_hash


def test_live_codes_per_channel_are_capped(pairing):
    for _ in range(UNUSED_CODES_MAX):
        pairing.create("telegram")

    with pytest.raises(PairingCodeLimitError):
        pairing.create("telegram")
    pairing.create("slack")


def test_dm_with_a_valid_code_pairs_that_sender_and_runs_no_turn(chat_id, harness, pairing):
    issued = pairing.create("telegram")

    adapter = _handle(_event(chat_id, issued.code.lower()))

    assert harness.turns == 0
    assert adapter.delivered == [(str(chat_id), PAIRED_NOTICE)]
    [row] = _senders(chat_id)
    assert (row.status, row.external_sender_id) == (SENDER_ALLOWED, "4242")

    adapter = _handle(_event(chat_id, "now answer me"))
    assert harness.turns == 1
    assert adapter.delivered == [(str(chat_id), REPLY)]


def test_a_code_works_once(chat_id, harness, pairing):
    issued = pairing.create("telegram")
    _handle(_event(chat_id, issued.code))

    adapter = _handle(_event(chat_id, issued.code, sender={"id": 5151, "is_bot": False}))

    assert adapter.delivered == []
    statuses = {row.external_sender_id: row.status for row in _senders(chat_id)}
    assert statuses == {"4242": SENDER_ALLOWED, "5151": SENDER_PENDING}


def test_an_expired_code_does_not_pair(chat_id, harness, pairing):
    issued = pairing.create("telegram")
    with get_open_session() as session:
        row = session.exec(select(ChannelPairingCode)).one()
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(row)
        session.commit()

    adapter = _handle(_event(chat_id, issued.code))

    assert adapter.delivered == []
    assert [row.status for row in _senders(chat_id)] == [SENDER_PENDING]


def test_a_code_made_for_another_channel_does_not_pair(chat_id, harness, pairing):
    issued = pairing.create("slack")

    adapter = _handle(_event(chat_id, issued.code))

    assert adapter.delivered == []
    assert [row.status for row in _senders(chat_id)] == [SENDER_PENDING]


def test_a_blocked_sender_stays_blocked_and_leaves_the_code_unused(chat_id, harness, pairing):
    _set_sender(chat_id, SENDER_BLOCKED)
    issued = pairing.create("telegram")

    adapter = _handle(_event(chat_id, issued.code))

    assert adapter.delivered == []
    assert [row.status for row in _senders(chat_id)] == [SENDER_BLOCKED]
    with get_open_session() as session:
        assert session.exec(select(ChannelPairingCode)).one().used_at is None


def test_the_telegram_linked_channel_account_cannot_pair_and_leaves_the_code_unused(chat_id, harness, pairing):
    issued = pairing.create("telegram")

    redeemed = pairing.redeem(
        channel_type="telegram", code=issued.code, group_id=str(-chat_id),
        sender_id="777000", sender_name=None, is_group=True,
    )
    adapter = _handle(_event(-chat_id, issued.code, sender={"id": 777000, "is_bot": False}, chat_type="group"))

    assert redeemed is False
    assert adapter.delivered == []
    assert _senders(-chat_id) == []
    with get_open_session() as session:
        assert session.exec(select(ChannelPairingCode)).one().used_at is None


def test_a_code_in_a_group_pairs_only_its_sender_even_without_a_mention(chat_id, harness, pairing):
    issued = pairing.create("telegram")

    adapter = _handle(_event(-chat_id, issued.code, chat_type="group"))

    assert adapter.delivered == [(str(-chat_id), PAIRED_NOTICE)]
    [row] = _senders(-chat_id)
    assert (row.status, row.external_sender_id, row.is_group) == (SENDER_ALLOWED, "4242", True)
    other = _handle(_event(-chat_id, "@b me too", sender={"id": 5151, "is_bot": False}, chat_type="group"))
    assert other.delivered == []
    assert harness.turns == 0


def test_two_chats_redeeming_one_code_at_once_let_in_exactly_one(chat_id, pairing, monkeypatch):
    """Both callers find the unused code before either claims it."""
    issued = pairing.create("telegram")
    both_found = threading.Barrier(2, timeout=10)
    real_claim = ChannelPairingService._claim

    def claim_after_both_found(self, code_id):
        both_found.wait()
        return real_claim(self, code_id)

    monkeypatch.setattr(ChannelPairingService, "_claim", claim_after_both_found)
    results: dict[str, bool] = {}

    def redeem(group_id: str) -> None:
        with get_open_session() as raw:
            results[group_id] = ChannelPairingService(ScopedSession(raw, LOCAL_SCOPE)).redeem(
                channel_type="telegram", code=issued.code, group_id=group_id,
                sender_id="4242", sender_name=None, is_group=False,
            )

    threads = [threading.Thread(target=redeem, args=(str(group),)) for group in (chat_id, -chat_id)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(results.values()) == [False, True]
    allowed = [row for row in _senders(chat_id) + _senders(-chat_id) if row.status == SENDER_ALLOWED]
    assert len(allowed) == 1


def test_a_harness_that_does_not_enforce_the_switches_runs_no_channel_turn(chat_id, monkeypatch):
    class UnawareHarness(RecordingHarness):
        enforces_channel_capabilities = False

    unaware = UnawareHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: unaware)
    _set_sender(chat_id, SENDER_ALLOWED)

    adapter = _handle(_event(chat_id, "hello"))

    assert unaware.turns == 0
    assert adapter.delivered == []


def test_a_turn_gets_its_bindings_switches(chat_id, harness):
    _set_sender(chat_id, SENDER_ALLOWED)
    _handle(_event(chat_id, "first"))
    with get_open_session() as session:
        binding = session.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == str(chat_id))).one()
        binding.allow_scratchpad = True
        binding.allow_file_replies = True
        session.add(binding)
        session.commit()

    _handle(_event(chat_id, "second"))

    first, second = harness.contexts
    switches = ("allow_scratchpad", "allow_connectors", "allow_memory_writes", "allow_file_replies")
    assert [getattr(first, name) for name in switches] == [False, False, False, False]
    assert [getattr(second, name) for name in switches] == [True, False, False, True]
