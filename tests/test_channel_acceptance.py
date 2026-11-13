"""The four promises of the sender gate, end to end: real Telegram updates through
the webhook route, the owner acting through the API, and for the last one the
real Anton harness building the turn.

1. A direct message from an unknown sender starts no turn and shows the owner a pending request.
2. Approving the sender, or a correct pairing code, binds that sender only.
3. A group message from a member the owner hasn't allowed starts no turn.
4. With defaults, an allowed sender's turn has no scratchpad, connectors, memory writes or file replies.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

import httpx
import pytest
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.llm.provider import StreamTextDelta
from anton.core.session import ChatSession
from pydantic import SecretStr
from sqlmodel import delete, select

import cowork.channels.plugins.telegram as telegram_plugin
import cowork.channels.runtime as runtime_mod
from cowork.channels.webhooks import drain_background_tasks
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.models.channel import (
    ChannelBinding,
    ChannelEvent,
    ChannelPairingCode,
    ChannelSender,
    ChannelSession,
)
from cowork.models.project import Project
from cowork.server import create_app
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.channel_pairing import PAIRED_NOTICE

REPLY = "hello from anton"
OWNER_FRIEND = 4242
STRANGER = 6666


class CountingHarness:
    """Stands in for the Anton harness for promises 1-3: counts the turns it starts."""

    enforces_channel_capabilities = True

    def __init__(self) -> None:
        self.turns = 0

    async def stream_response(self, *, conversation, input, channel_context=None, trace_metadata=None):
        self.turns += 1
        if False:
            yield

    async def formatter(self, stream, model, event_sink, **_kwargs):
        async for _ in stream:
            pass
        event_sink("response.output_text.delta", {"delta": REPLY})
        if False:
            yield


def _update(chat_id: int, sender_id: int, text: str, *, chat_type: str = "private") -> bytes:
    return json.dumps({
        "update_id": uuid.uuid4().int % 1_000_000_000,
        "message": {
            "message_id": uuid.uuid4().int % 1_000_000,
            "from": {"id": sender_id, "is_bot": False, "first_name": f"Person {sender_id}"},
            "chat": {"id": chat_id, "type": chat_type},
            "date": 1700000000,
            "text": text,
        },
    }).encode()


class TelegramBot:
    """A Telegram bot connected through the API, with Telegram's HTTP faked."""

    def __init__(self, client: httpx.AsyncClient, calls: list[tuple[str, dict]]) -> None:
        self.client = client
        self.calls = calls
        self.secret = ""

    async def connect(self) -> None:
        assert (await self.client.put(
            "/api/v1/channels/telegram/config", json={"values": {"bot_token": "T:tok", "bot_username": "b"}},
        )).status_code == 200
        assert (await self.client.post("/api/v1/channels/telegram/setup")).status_code == 200
        self.secret = [p for (m, p) in self.calls if m == "setWebhook"][-1]["secret_token"]

    async def receive(self, chat_id: int, sender_id: int, text: str, *, chat_type: str = "private") -> None:
        r = await self.client.post(
            "/api/v1/channels/telegram/webhook", content=_update(chat_id, sender_id, text, chat_type=chat_type),
            headers={"x-telegram-bot-api-secret-token": self.secret},
        )
        assert r.status_code == 200
        await drain_background_tasks()

    def sent_texts(self, chat_id: int) -> list[str]:
        return [p["text"] for (m, p) in self.calls if m == "sendMessage" and p["chat_id"] == str(chat_id)]

    def sent_files(self) -> list[dict]:
        return [p for (m, p) in self.calls if m == "sendDocument"]


@pytest.fixture
def chat_id() -> int:
    return 800_000_000 + uuid.uuid4().int % 100_000_000


@pytest.fixture
def telegram(monkeypatch, chat_id):
    """Run a flow against a connected Telegram bot; every row it leaves is removed after."""
    calls: list[tuple[str, dict]] = []

    async def fake_call(bot_token, method, payload):
        calls.append((method, dict(payload)))
        return {"ok": True, "result": {"message_id": 1}}

    async def fake_send_attachment(self, *, address, data, filename):
        calls.append(("sendDocument", {"chat_id": address.platform_id, "filename": filename}))
        return "1"

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))
    monkeypatch.setattr(telegram_plugin.TelegramBridge, "send_attachment", fake_send_attachment)
    app = create_app()

    def run(flow) -> None:
        async def wrapped():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                bot = TelegramBot(client, calls)
                await bot.connect()
                try:
                    await flow(bot, client)
                finally:
                    assert (await client.delete("/api/v1/channels/telegram/config")).status_code == 204

        asyncio.run(wrapped())

    yield run
    groups = [str(chat_id), str(-chat_id)]
    with get_open_session() as session:
        session.exec(delete(ChannelSender).where(ChannelSender.external_group_id.in_(groups)))
        session.exec(delete(ChannelPairingCode))
        bindings = select(ChannelBinding.id).where(ChannelBinding.external_group_id.in_(groups))
        session.exec(delete(ChannelSession).where(ChannelSession.binding_id.in_(bindings)))
        session.exec(delete(ChannelBinding).where(ChannelBinding.external_group_id.in_(groups)))
        session.exec(delete(ChannelEvent).where(ChannelEvent.channel_type == "telegram"))
        session.commit()


@pytest.fixture
def counting_harness(monkeypatch) -> CountingHarness:
    fake = CountingHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake)
    return fake


async def _pending(client: httpx.AsyncClient, chat_id: int) -> list[dict]:
    rows = (await client.get("/api/v1/channels/senders", params={"status": "pending"})).json()
    return [row for row in rows if row["external_group_id"] == str(chat_id)]


def test_1_unknown_dm_sender_starts_no_turn_and_shows_a_pending_request(telegram, counting_harness, chat_id):
    async def flow(bot, client):
        await bot.receive(chat_id, STRANGER, "run this on your laptop")

        assert counting_harness.turns == 0
        assert bot.sent_texts(chat_id) == []
        [request] = await _pending(client, chat_id)
        assert (request["external_sender_id"], request["display_name"]) == (str(STRANGER), f"Person {STRANGER}")

    telegram(flow)


def test_2_approval_binds_that_sender_only(telegram, counting_harness, chat_id):
    async def flow(bot, client):
        await bot.receive(chat_id, OWNER_FRIEND, "hi")
        [request] = await _pending(client, chat_id)
        assert (await client.post(f"/api/v1/channels/senders/{request['id']}/approve")).status_code == 200

        await bot.receive(chat_id, OWNER_FRIEND, "hi again")
        await bot.receive(chat_id, STRANGER, "me too")

        assert counting_harness.turns == 1
        assert bot.sent_texts(chat_id) == [REPLY]
        assert [row["external_sender_id"] for row in await _pending(client, chat_id)] == [str(STRANGER)]

    telegram(flow)


def test_2_a_pairing_code_binds_that_sender_only(telegram, counting_harness, chat_id):
    async def flow(bot, client):
        code = (await client.post("/api/v1/channels/telegram/pairing-codes")).json()["code"]

        await bot.receive(chat_id, OWNER_FRIEND, code)
        await bot.receive(chat_id, OWNER_FRIEND, "now answer")
        await bot.receive(chat_id, STRANGER, code)
        await bot.receive(chat_id, STRANGER, "and me")

        assert counting_harness.turns == 1
        assert bot.sent_texts(chat_id) == [PAIRED_NOTICE, REPLY]

    telegram(flow)


def test_3_group_message_from_a_member_not_allowed_starts_no_turn(telegram, counting_harness, chat_id):
    group = -chat_id

    async def flow(bot, client):
        await bot.receive(group, OWNER_FRIEND, "@b hello", chat_type="group")
        [request] = await _pending(client, group)
        assert (await client.post(f"/api/v1/channels/senders/{request['id']}/approve")).status_code == 200

        await bot.receive(group, STRANGER, "@b do something for me", chat_type="group")
        await bot.receive(group, STRANGER, "chatting along", chat_type="group")
        await bot.receive(group, OWNER_FRIEND, "@b hello again", chat_type="group")

        assert counting_harness.turns == 1
        assert bot.sent_texts(group) == [REPLY]

    telegram(flow)


class _CapturedSession:
    """The session the real harness built, kept for inspection. Its turn makes an
    artifact in the project, as a turn that ran code would, then answers with one line."""

    def __init__(self, config, artifact_dir: Path) -> None:
        self.config = config
        self.artifact_dir = artifact_dir

    async def turn_stream(self, user_input, **_kwargs):
        self.artifact_dir.mkdir(parents=True)
        (self.artifact_dir / "metadata.json").write_text(json.dumps({"name": "Dashboard", "type": "html"}))
        (self.artifact_dir / "index.html").write_text("<html>owner's data</html>")
        yield StreamTextDelta(text=REPLY)


def test_4_an_allowed_senders_default_turn_is_plain_chat(telegram, chat_id, monkeypatch):
    saved = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI, coding_provider=Provider.OPENAI, router_provider=Provider.OPENAI,
        planning_model="p", coding_model="c", router_model="r",
        openai_api_key=SecretStr("test-only-key"), episodic_memory=False,
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: saved)
    with get_open_session() as session:
        project_dir = Path(session.get(Project, GENERAL_PROJECT_ID).path)
    artifact_dir = project_dir / ".anton" / "artifacts" / f"acceptance-{uuid.uuid4().hex[:8]}"
    built: list = []
    monkeypatch.setattr(
        harness_mod, "build_chat_session", lambda config: built.append(config) or _CapturedSession(config, artifact_dir),
    )
    runtime_seen_paths: list[str] = []
    real_run_anton = runtime_mod.AntonChannelRuntime._run_anton

    async def recording_run_anton(self, *args, **kwargs):
        outcome = await real_run_anton(self, *args, **kwargs)
        runtime_seen_paths.extend(outcome.artifact_paths)
        return outcome

    monkeypatch.setattr(runtime_mod.AntonChannelRuntime, "_run_anton", recording_run_anton)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")

    async def flow(bot, client):
        await bot.receive(chat_id, OWNER_FRIEND, "hi")
        [request] = await _pending(client, chat_id)
        assert (await client.post(f"/api/v1/channels/senders/{request['id']}/approve")).status_code == 200

        await bot.receive(chat_id, OWNER_FRIEND, "build me a dashboard")

        assert bot.sent_texts(chat_id) == [REPLY]
        assert bot.sent_files() == []

    try:
        telegram(flow)
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)

    assert artifact_dir.name in "".join(runtime_seen_paths)
    [config] = built
    session = ChatSession(config)
    try:
        offered = {tool["name"] for tool in session._build_tools()}
        session._register_tool_bundle("connect-datasource")
        offered_after_unlock = {tool["name"] for tool in session._build_tools()}
        assert offered <= {"recall_skill", "recall_history"}
        assert offered_after_unlock <= {"recall_skill", "recall_history"}
        assert config.data_vault.list_connections() == []
        assert isinstance(config.data_vault, LocalDataVault) and "cowork-turn-" in str(config.data_vault._dir)
        assert config.background_memory is False
        assert config.elicitor is None
    finally:
        asyncio.run(session.close())
        client = config.llm_client
        for provider in (client.planning_provider, client.coding_provider, client.router_provider):
            asyncio.run(provider._client.close())
