"""End-to-end smoke for the Telegram channel through the full server stack.

Drives the real ASGI app over httpx on a single event loop so the runtime's
background task is awaitable (``drain_background_tasks``). Telegram HTTP goes
through ``TelegramBridge._call`` (mocked) and the Anton harness is faked — no
real credentials, network, or LLM.
"""
import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from sqlmodel import select

import cowork.channels.plugins.telegram as telegram_plugin
import cowork.channels.runtime as runtime_mod
from cowork.channels.registry import PluginRegistry, load_first_party_plugins
from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
from cowork.channels.webhooks import drain_background_tasks
from cowork.db.session import get_open_session
from cowork.harnesses.base import ChannelContext
from cowork.models.channel import ChannelBinding, ChannelEvent, ChannelSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.server import create_app
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.services.conversations import ConversationService, _is_tool_row

pytestmark = pytest.mark.usefixtures("allow_channel_senders")

REPLY = "hello from anton"
LINK_PREFIX = "https://app.example.com/c/"


class FakeHarness:
    """Stands in for the Anton harness — one assistant delta, no LLM."""

    enforces_channel_capabilities = True

    def __init__(
        self, tool_event: bool = False, delay: float = 0.0, turn_history: list | None = None,
        artifacts: list[dict] | None = None,
    ):
        self.tool_event = tool_event
        self.delay = delay
        self.turn_history = turn_history
        # Artifact cards the turn emits, as the real formatter does for each
        # ArtifactCreated.
        self.artifacts = artifacts or []
        self.inputs: list[list[dict]] = []
        self.channel_contexts: list = []
        self.trace_metadata: list = []

    async def stream_response(self, *, conversation, input, channel_context=None,
                              trace_metadata=None):
        self.inputs.append(input)
        self.channel_contexts.append(channel_context)
        self.trace_metadata.append(trace_metadata)
        if False:
            yield

    async def formatter(self, stream, model, event_sink, **_kwargs):
        async for _ in stream:
            pass
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.tool_event:
            event_sink("response.in_progress", {
                "type": "response.in_progress",
                "thought_role": "thought_scratchpad_start",
                "tool_use_id": "t1",
            })
        if self.turn_history is not None:
            event_sink("response.turn_history", {
                "type": "response.turn_history",
                "rows": self.turn_history,
            })
        for card in self.artifacts:
            event_sink("response.artifact_created", {"type": "response.artifact_created", "artifact": card})
        event_sink("response.output_text.delta", {"delta": REPLY})
        if False:
            yield


class FakeAdapter:
    def __init__(self):
        self.delivered = []

    async def deliver(self, message):
        self.delivered.append((message.address.platform_id, message.text))

    async def shutdown(self):
        ...


def telegram_update(update_id: int, chat_id: int, message_id: int, text: str, chat_type: str = "private") -> bytes:
    return json.dumps({
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": chat_id, "type": chat_type},
            "date": 1700000000,
            "text": text,
        },
    }).encode()


def inbound_events(session):
    return session.exec(select(ChannelEvent).where(ChannelEvent.direction == "inbound")).all()


def test_telegram_end_to_end(monkeypatch):
    calls: list[tuple[str, dict]] = []

    async def fake_call(bot_token, method, payload):
        calls.append((method, dict(payload)))
        return {"ok": True, "result": {"message_id": 999}}

    fake_harness = FakeHarness()
    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake_harness)

    app = create_app()
    adapters = app.state.channel_adapters

    async def flow():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.put("/api/v1/channels/telegram/config", json={"values": {"bot_token": "T:tok"}})
            assert r.status_code == 200

            r = await client.post("/api/v1/channels/telegram/setup")
            assert r.status_code == 200 and r.json()["active"] is True

            set_hooks = [p for (m, p) in calls if m == "setWebhook"]
            assert len(set_hooks) == 1
            assert set_hooks[0]["url"] == "https://hooks.example.com/api/v1/channels/telegram/webhook"
            secret = set_hooks[0]["secret_token"]
            assert secret and adapters.get("telegram") is not None

            cfg = (await client.get("/api/v1/channels/telegram/config")).json()
            assert cfg["fields"]["secret_token"]["is_set"] is True
            assert cfg["fields"]["secret_token"]["value"] is None
            assert secret not in json.dumps(cfg) and "T:tok" not in json.dumps(cfg)

            body = telegram_update(1, 7, 5, "hi anton")
            headers = {"x-telegram-bot-api-secret-token": secret}
            r = await client.post("/api/v1/channels/telegram/webhook", content=body, headers=headers)
            assert r.status_code == 200
            s = get_open_session()
            assert len(inbound_events(s)) == 1
            s.close()
            await drain_background_tasks()

            s = get_open_session()
            binding = s.exec(select(ChannelBinding)).one()
            assert binding.channel_type == "telegram" and binding.anton_conversation_id is not None
            sessions = s.exec(select(ChannelSession)).all()
            assert len(sessions) == 1 and sessions[0].binding_id == binding.id
            conversation_id = binding.anton_conversation_id
            msgs = s.exec(select(Message).where(Message.conversation_id == conversation_id)).all()
            assert sorted(m.role for m in msgs) == ["assistant", "user"]
            assistant = next(m for m in msgs if m.role == "assistant")
            assert assistant.content == REPLY and assistant.harness == "anton"
            assert inbound_events(s)[0].status == "routed"
            s.close()

            # A channel turn never passes through ResponsesHandler, so it has
            # to carry its own build stamp (ENG-1279) — otherwise every bot
            # turn is unattributable to the release that produced it.
            assert fake_harness.trace_metadata[0]["cowork_server_version"]
            assert fake_harness.trace_metadata[0]["install_channel"]

            # No tool events in this turn → reply delivered verbatim, no link.
            sends = [p for (m, p) in calls if m == "sendMessage"]
            assert len(sends) == 1 and sends[0]["chat_id"] == "7" and sends[0]["text"] == REPLY

            # duplicate webhook → dropped, Anton not run twice
            r = await client.post("/api/v1/channels/telegram/webhook", content=body, headers=headers)
            assert r.status_code == 200
            await drain_background_tasks()
            s = get_open_session()
            assert len(inbound_events(s)) == 1
            # Scoped to THIS binding's conversation: the test DB is shared by
            # the whole session, so a global Message count is really counting
            # every other test's turns too (it only passed by file ordering).
            replies = s.exec(select(Message).where(Message.conversation_id == conversation_id)).all()
            assert len([m for m in replies if m.role == "assistant"]) == 1
            s.close()
            assert len([p for (m, p) in calls if m == "sendMessage"]) == 1

            r = await client.post("/api/v1/channels/telegram/teardown")
            assert r.status_code == 200 and r.json()["active"] is False
            assert any(m == "deleteWebhook" for (m, p) in calls)
            assert adapters.get("telegram") is None

    asyncio.run(flow())


def test_rich_turn_appends_conversation_link(monkeypatch):
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(tool_event=True))

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})
    event = asyncio.run(bridge.parse_inbound(
        body=telegram_update(50, 99, 1, "build me a dashboard"), headers={}, route_name=None,
    ))[0]

    adapters = LiveAdapterRegistry(registry)
    adapter = FakeAdapter()
    adapters._cache[("telegram", None)] = adapter
    asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))

    chat_id, delivered = adapter.delivered[0]
    assert chat_id == "99"
    s = get_open_session()
    binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == "99")).one()
    assert delivered == f"{REPLY}\n\n{LINK_PREFIX}{binding.anton_conversation_id}"
    # The stored assistant message stays canonical — link is channel-only.
    assistant = next(
        m for m in s.exec(select(Message).where(Message.conversation_id == binding.anton_conversation_id)).all()
        if m.role == "assistant"
    )
    assert assistant.content == REPLY
    s.close()


@pytest.mark.parametrize(
    ("chat_type", "chat_id", "text", "expected"),
    [
        # A shared group: marked with the route's own channel.
        ("group", -98, "@b hello", "telegram"),
        # A one-to-one chat with the bot: never marked, so no admin can read it.
        ("private", 97, "hello", None),
    ],
)
def test_runtime_marks_only_shared_channel_conversations(monkeypatch, chat_type, chat_id, text, expected):
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness())

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})
    event = asyncio.run(bridge.parse_inbound(
        body=telegram_update(51, chat_id, 1, text, chat_type=chat_type), headers={}, route_name=None,
    ))[0]

    adapters = LiveAdapterRegistry(registry)
    adapters._cache[("telegram", None)] = FakeAdapter()
    asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))

    s = get_open_session()
    try:
        binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == str(chat_id))).one()
        conversation = s.get(Conversation, binding.anton_conversation_id)
        assert conversation is not None, "the message must have started a conversation"
        assert conversation.channel_type == expected
    finally:
        s.close()


def test_channel_turn_persists_tool_rows_and_hides_history_event(monkeypatch):
    """A channel turn must persist the harness's tool rows for LLM replay, and
    the synthetic turn_history event must NOT leak into stored MessageEvents."""
    rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "B"}]},
    ]
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(turn_history=rows))

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})
    event = asyncio.run(bridge.parse_inbound(
        body=telegram_update(60, 88, 1, "use a tool"), headers={}, route_name=None,
    ))[0]

    adapters = LiveAdapterRegistry(registry)
    adapters._cache[("telegram", None)] = FakeAdapter()
    asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))

    s = get_open_session()
    binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == "88")).one()
    cid = binding.anton_conversation_id
    svc = ConversationService(ScopedSession(s, LOCAL_SCOPE))
    # (a) tool rows persisted for LLM replay.
    assert sum(1 for m in svc.get_ordered_messages(cid) if _is_tool_row(m.content)) == 2
    # (b) the turn_history blob is not stored as a MessageEvent / shown to UI,
    #     and tool rows stay hidden from the chat.
    ui = svc.get_messages(cid)
    assert all(
        e.get("type") != "response.turn_history"
        for m in ui for e in m.get("events", [])
    )
    assert all(not _is_tool_row(m["content"]) for m in ui)
    s.close()


def test_typing_indicator_runs_during_turn(monkeypatch):
    monkeypatch.setattr(runtime_mod, "TYPING_REFRESH_S", 0.01)
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(delay=0.05))

    class TypingAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.typing = []

        async def set_typing(self, *, address):
            self.typing.append(address.platform_id)

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})
    event = asyncio.run(bridge.parse_inbound(
        body=telegram_update(60, 123, 1, "ping"), headers={}, route_name=None,
    ))[0]

    adapters = LiveAdapterRegistry(registry)
    adapter = TypingAdapter()
    adapters._cache[("telegram", None)] = adapter
    asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))

    # Indicator refreshed while the turn ran, on the right chat, then stopped.
    assert len(adapter.typing) >= 2 and set(adapter.typing) == {"123"}
    assert adapter.delivered and adapter.delivered[0][0] == "123"


def test_telegram_set_typing_calls_send_chat_action(monkeypatch):
    calls = []

    async def fake_call(bot_token, method, payload):
        calls.append((method, dict(payload)))
        return {"ok": True}

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s"})
    from anton.core.dispatch import PlatformAddress
    asyncio.run(bridge.set_typing(address=PlatformAddress("telegram", "7", None)))
    assert calls == [("sendChatAction", {"chat_id": "7", "action": "typing"})]


def media_update(update_id: int, chat_id: int, *, caption: str = "", photo=None, document=None) -> bytes:
    msg = {"message_id": update_id, "from": {"id": 42, "is_bot": False},
           "chat": {"id": chat_id, "type": "private"}, "date": 1700000000}
    if caption:
        msg["caption"] = caption
    if photo is not None:
        msg["photo"] = photo
    if document is not None:
        msg["document"] = document
    return json.dumps({"update_id": update_id, "message": msg}).encode()


def test_telegram_parses_media_messages():
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s"})

    def parse(body):
        return asyncio.run(bridge.parse_inbound(body=body, headers={}, route_name=None))

    photo = [{"file_id": "small", "file_unique_id": "u1", "file_size": 100},
             {"file_id": "big", "file_unique_id": "u1", "file_size": 5000}]
    events = parse(media_update(70, 5, caption="look", photo=photo))
    assert len(events) == 1 and events[0].message.content == "look"
    atts = events[0].message.attachments
    assert len(atts) == 1 and atts[0].mime_type == "image/jpeg"
    assert atts[0].telegram_file_id == "big"  # largest rendition

    doc = {"file_id": "d1", "file_unique_id": "u2", "file_name": "report.pdf",
           "mime_type": "application/pdf", "file_size": 1000}
    events = parse(media_update(71, 5, document=doc))
    assert len(events) == 1 and events[0].message.content == ""
    assert events[0].message.attachments[0].filename == "report.pdf"

    # over the getFile limit → attachment dropped, caption still produces an event
    big = dict(doc, file_size=30 * 1024 * 1024)
    events = parse(media_update(72, 5, caption="too big", document=big))
    assert len(events) == 1 and events[0].message.attachments == []

    # media-only over the limit and no caption → nothing to route
    assert parse(media_update(73, 5, document=big)) == []


def test_telegram_fetch_attachment(monkeypatch):
    async def fake_call(bot_token, method, payload):
        assert method == "getFile" and payload == {"file_id": "f1"}
        return {"ok": True, "result": {"file_path": "photos/x.jpg"}}

    async def fake_download(bot_token, file_path):
        assert file_path == "photos/x.jpg"
        return b"BYTES"

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))
    monkeypatch.setattr(telegram_plugin.TelegramBridge, "download_file", staticmethod(fake_download))
    bridge = telegram_plugin.TelegramBridge({"bot_token": "tok", "secret_token": "s"})
    attachment = telegram_plugin.Attachment(filename="x.jpg", mime_type="image/jpeg")
    attachment.telegram_file_id = "f1"
    assert asyncio.run(bridge.fetch_attachment(attachment)) == b"BYTES"

    async def failing_call(bot_token, method, payload):
        return {"ok": False}

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(failing_call))
    assert asyncio.run(bridge.fetch_attachment(attachment)) is None


@pytest.mark.parametrize(("name", "stored_as"), [("..", "file"), (" . ", "file"), ("../../notes.txt", "notes.txt")])
def test_a_channel_attachments_name_keeps_its_file_in_its_own_folder(monkeypatch, tmp_path, name, stored_as):
    """A platform's file name is untrusted. Only its last component is kept,
    and a name that would point at the folder itself or the files root ("."
    or "..") is stored as "file", so the turn goes on rather than failing on
    a write to a directory."""
    from pathlib import Path

    from anton.core.dispatch import Attachment
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.scoped import LOCAL_SCOPE

    monkeypatch.setenv("COWORK_FILES_DIR", str(tmp_path / "files"))
    get_app_settings.cache_clear()
    event = SimpleNamespace(message=SimpleNamespace(
        attachments=[Attachment(filename=name, mime_type="text/plain", data=b"notes")],
    ))
    try:
        runtime = AntonChannelRuntime(LiveAdapterRegistry(PluginRegistry()))
        blocks = asyncio.run(runtime.build_input_blocks(LOCAL_SCOPE, None, event, "read this"))
    finally:
        get_app_settings.cache_clear()

    [file_block, text_block] = blocks
    stored = Path(file_block["path"])
    # The test database is shared: other tests count the channel's files.
    from cowork.models.file import File

    s = get_open_session()
    try:
        [row] = s.exec(select(File).where(File.path == str(stored))).all()
        s.delete(row)
        s.commit()
    finally:
        s.close()
    assert (file_block["filename"], stored.name, stored.read_bytes()) == (stored_as, stored_as, b"notes")
    assert (tmp_path / "files").resolve() in stored.resolve().parents
    assert text_block == {"type": "text", "text": "read this"}


def test_inbound_media_becomes_harness_blocks(monkeypatch, tmp_path):
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_FILES_DIR", str(tmp_path / "files"))
    get_app_settings.cache_clear()

    harness = FakeHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: harness)

    class MediaAdapter(FakeAdapter):
        async def fetch_attachment(self, attachment):
            return b"IMGDATA"

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s"})
    photo = [{"file_id": "big", "file_unique_id": "u9", "file_size": 5000}]
    event = asyncio.run(bridge.parse_inbound(
        body=media_update(80, 321, caption="what is this", photo=photo), headers={}, route_name=None,
    ))[0]

    adapters = LiveAdapterRegistry(registry)
    adapters._cache[("telegram", None)] = MediaAdapter()
    asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))

    blocks = harness.inputs[0]
    assert [b["type"] for b in blocks] == ["image", "text"]
    import base64
    assert blocks[0]["source"]["data"] == base64.standard_b64encode(b"IMGDATA").decode("ascii")
    assert blocks[1]["text"] == "what is this"

    from cowork.models.file import File
    from pathlib import Path
    s = get_open_session()
    stored = [f for f in s.exec(select(File)).all() if f.purpose == "channel"]
    assert len(stored) == 1 and Path(stored[0].path).read_bytes() == b"IMGDATA"
    binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == "321")).one()
    user_msg = next(
        m for m in s.exec(select(Message).where(Message.conversation_id == binding.anton_conversation_id)).all()
        if m.role == "user"
    )
    assert user_msg.content == "what is this"
    s.close()


def test_telegram_send_attachment(monkeypatch):
    captured = {}

    async def fake_post(self, url, data=None, files=None, **kw):
        captured.update(url=url, data=data, files=files)

        class R:
            def json(self):
                return {"ok": True, "result": {"message_id": 321}}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "tok", "secret_token": "s"})
    from anton.core.dispatch import PlatformAddress

    def send():
        return asyncio.run(bridge.send_attachment(
            address=PlatformAddress("telegram", "7", None), data=b"PDFDATA", filename="report.pdf",
        ))

    mid = send()
    assert mid == "321"
    assert captured["url"].endswith("/sendDocument") and captured["data"] == {"chat_id": "7"}
    assert captured["files"]["document"] == ("report.pdf", b"PDFDATA")

    monkeypatch.setattr(telegram_plugin, "TELEGRAM_MAX_UPLOAD_BYTES", 3)
    try:
        send()
        raise SystemExit("oversize must raise")
    except RuntimeError:
        pass


def _general_project_dir() -> str:
    from cowork.models.project import Project
    from cowork.services.projects import GENERAL_PROJECT_ID

    s = get_open_session()
    try:
        return s.get(Project, GENERAL_PROJECT_ID).path
    finally:
        s.close()


def _telegram_event(update_id: int, chat_id: int, text: str, chat_type: str = "private"):
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})
    return asyncio.run(bridge.parse_inbound(
        body=telegram_update(update_id, chat_id, 1, text, chat_type=chat_type), headers={}, route_name=None,
    ))[0]


def _runtime_with(adapter) -> AntonChannelRuntime:
    registry = PluginRegistry()
    load_first_party_plugins(registry)
    adapters = LiveAdapterRegistry(registry)
    adapters._cache[("telegram", None)] = adapter
    return AntonChannelRuntime(adapters)


class ArtifactAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.sent = []

    async def send_attachment(self, *, address, data, filename):
        self.sent.append((address.platform_id, data, filename))
        return "1"


def _allow_file_replies(chat_id: int) -> None:
    """Turn on the file replies switch of the chat's binding, making the binding if needed."""
    with get_open_session() as session:
        binding = session.exec(select(ChannelBinding).where(
            ChannelBinding.channel_type == "telegram", ChannelBinding.external_group_id == str(chat_id),
        )).first() or ChannelBinding(channel_type="telegram", external_group_id=str(chat_id))
        binding.allow_file_replies = True
        session.add(binding)
        session.commit()


def test_turn_artifacts_are_not_sent_without_the_file_replies_switch(monkeypatch):
    """With the binding's defaults the reply goes out and no file follows it."""
    import os
    import shutil

    made = os.path.join(_general_project_dir(), ".anton", "artifacts", "demo-withheld")
    os.makedirs(made, exist_ok=True)
    with open(os.path.join(made, "dashboard.html"), "w") as f:
        f.write("<html/>")
    card = {"path": os.path.join(made, "dashboard.html"), "folder": made}
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(tool_event=True, artifacts=[card]))
    adapter = ArtifactAdapter()
    try:
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(93, 558, "make a dashboard")))
    finally:
        shutil.rmtree(made, ignore_errors=True)

    assert [chat for chat, _text in adapter.delivered] == ["558"]
    assert adapter.sent == []


def test_turn_artifacts_delivered(monkeypatch):
    """The turn sends the files of the artifact cards it emitted, and no other
    artifact in the project, however recent."""
    import os
    import shutil

    project_dir = _general_project_dir()
    artifacts_dir = os.path.join(project_dir, ".anton", "artifacts")
    made = os.path.join(artifacts_dir, "demo-delivered")
    os.makedirs(made, exist_ok=True)
    with open(os.path.join(made, "metadata.json"), "w") as f:
        f.write(json.dumps({"name": "Demo", "type": "html"}))
    with open(os.path.join(made, "dashboard.html"), "w") as f:
        f.write("<html/>")
    # Written just now by another conversation: no card of this turn names it.
    other = os.path.join(artifacts_dir, "someone-elses")
    os.makedirs(other, exist_ok=True)
    with open(os.path.join(other, "page.html"), "w") as f:
        f.write("<html>theirs</html>")

    card = {"path": os.path.join(made, "dashboard.html"), "folder": made}
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(tool_event=True, artifacts=[card]))
    adapter = ArtifactAdapter()
    _allow_file_replies(555)
    try:
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(90, 555, "make a dashboard")))
    finally:
        shutil.rmtree(made, ignore_errors=True)
        shutil.rmtree(other, ignore_errors=True)

    # Text reply (with link) lands first, then exactly the turn's own file.
    assert adapter.delivered and adapter.delivered[0][0] == "555"
    assert adapter.sent == [("555", b"<html/>", "dashboard.html")]


def test_reading_a_turns_files_keeps_regular_files_under_the_artifacts_base(tmp_path):
    """A card's path is a file the agent wrote, so only a regular file inside
    the project's artifacts base is read: never a link it planted, a folder,
    a path outside the base, or a file past the size cap."""
    from cowork.channels.runtime import TurnFile, read_turn_files

    project = tmp_path / "proj"
    folder = project / ".anton" / "artifacts" / "report"
    folder.mkdir(parents=True)
    (folder / "report.csv").write_bytes(b"a,b\n")
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"not yours")
    (folder / "leak.txt").symlink_to(outside)
    linked = project / ".anton" / "artifacts" / "linked"
    linked.symlink_to(tmp_path)
    (folder / "huge.bin").write_bytes(b"x" * 11)

    paths = [
        str(folder / "leak.txt"),
        str(linked / "secret.txt"),
        str(folder),
        str(outside),
        str(project / "notes.txt"),
        str(folder / "huge.bin"),
        str(folder / "report.csv"),
        str(folder / "report.csv"),
    ]

    files = read_turn_files(project_path=str(project), paths=paths, max_bytes=10)

    assert files == [TurnFile(filename="report.csv", data=b"a,b\n")]


def test_a_turn_never_reads_a_file_past_its_adapters_upload_cap(monkeypatch):
    """Each bridge states its platform's upload cap. A turn's file past the
    adapter's cap, though far under MAX_TURN_ATTACHMENT_BYTES, is skipped on
    its size: its bytes are never read off the file store, and it is never
    sent. A file within the cap still goes."""
    import os
    import shutil

    project_dir = _general_project_dir()
    made = os.path.join(project_dir, ".anton", "artifacts", "demo-capped")
    os.makedirs(made, exist_ok=True)
    with open(os.path.join(made, "big.bin"), "wb") as f:
        f.write(b"x" * 16)
    with open(os.path.join(made, "small.txt"), "wb") as f:
        f.write(b"ok")
    cards = [
        {"path": os.path.join(made, "big.bin"), "folder": made},
        {"path": os.path.join(made, "small.txt"), "folder": made},
    ]
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(tool_event=True, artifacts=cards))
    read_sizes: list[int] = []
    real_read_bounded = runtime_mod.read_bounded

    def recording_read_bounded(fd, size):
        read_sizes.append(size)
        return real_read_bounded(fd, size)

    monkeypatch.setattr(runtime_mod, "read_bounded", recording_read_bounded)

    class CappedAdapter(ArtifactAdapter):
        max_attachment_bytes = 10

    adapter = CappedAdapter()
    _allow_file_replies(557)
    try:
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(92, 557, "make a file")))
    finally:
        shutil.rmtree(made, ignore_errors=True)

    assert read_sizes == [2]
    assert adapter.sent == [("557", b"ok", "small.txt")]


def test_each_first_party_bridge_states_its_platforms_upload_cap():
    """The cap a turn reads files under is the one send_attachment enforces."""
    from cowork.channels.plugins import discord, slack, telegram, whatsapp

    assert discord.DiscordBridge.max_attachment_bytes == discord.DISCORD_MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    assert slack.SlackBridge.max_attachment_bytes == slack.SLACK_MAX_UPLOAD_BYTES == 50 * 1024 * 1024
    assert telegram.TelegramBridge.max_attachment_bytes == telegram.TELEGRAM_MAX_UPLOAD_BYTES == 50 * 1024 * 1024
    assert whatsapp.WhatsAppBridge.max_attachment_bytes == whatsapp.WHATSAPP_MAX_UPLOAD_BYTES == 100 * 1024 * 1024


def test_a_file_store_that_refuses_the_turns_files_still_ends_the_turn_well(monkeypatch, caplog):
    """The reply is already delivered when the turn's files are read, so a
    busy file store sends no files, logs one WARNING, and sends no busy
    notice: the turn ended well."""
    import logging
    from cowork.db.units import FilesystemBusy

    card = {"path": "/nowhere/.anton/artifacts/x/index.html"}
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(tool_event=True, artifacts=[card]))

    async def refused(fn, *, scope, deadline=True):
        raise FilesystemBusy("no shared file store slot freed within POOL_TIMEOUT")

    monkeypatch.setattr(runtime_mod, "run_efs", refused)
    adapter = ArtifactAdapter()
    _allow_file_replies(556)
    with caplog.at_level(logging.WARNING, logger=runtime_mod.log.name):
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(91, 556, "make a dashboard")))

    assert [chat for chat, _text in adapter.delivered] == ["556"]
    assert REPLY in adapter.delivered[0][1]
    assert adapter.sent == []
    [record] = [r for r in caplog.records if r.name == runtime_mod.log.name and r.levelname == "WARNING"]
    assert record.getMessage() == "channel turn: no file store slot freed in time to send the turn's files"
    assert record.conversation_id


@pytest.mark.parametrize(
    ("chat_type", "chat_id", "text", "told"),
    [
        ("private", 601, "hello", True),
        ("group", -602, "@b hello", True),
        # A group message not addressed to the bot: the trigger rule, unread,
        # may be set to ignore it.
        ("group", -603, "hello all", False),
    ],
)
def test_a_message_whose_binding_unit_finds_no_connection_hears_the_busy_notice(
    monkeypatch, chat_type, chat_id, text, told,
):
    from cowork.channels.runtime import BUSY_NOTICE
    from cowork.db.units import DatabaseBusy

    def refused(self, scoped, channel_type, event):
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")

    monkeypatch.setattr(AntonChannelRuntime, "_resolve_or_create_binding", refused)
    adapter = FakeAdapter()

    with pytest.raises(DatabaseBusy):
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(600, chat_id, text, chat_type)))

    assert adapter.delivered == ([(str(chat_id), BUSY_NOTICE)] if told else [])


@pytest.mark.parametrize(("step", "chat_id"), [("read", 607), ("make", 608)])
def test_a_refusal_at_either_step_of_a_binding_tells_a_direct_sender(monkeypatch, step, chat_id):
    """A thread's binding is read as a row in a database unit, and only a new
    one is made on the file store's threads (run_efs_unit), since it may
    start in the general project. The pool refusing the read, or the file
    store refusing the make, tells a sender who wrote to the bot directly."""
    from cowork.channels.runtime import BUSY_NOTICE
    from cowork.db.units import DatabaseBusy, FilesystemBusy

    if step == "read":
        refusal = DatabaseBusy("no database connection freed within POOL_TIMEOUT")

        def refused_read(scoped, *, channel_type, event):
            raise refusal

        monkeypatch.setattr(AntonChannelRuntime, "_find_binding", staticmethod(refused_read))
    else:
        refusal = FilesystemBusy("no shared file store slot freed within POOL_TIMEOUT")

        async def refused_make(fn, *, scope, deadline=True):
            raise refusal

        monkeypatch.setattr(runtime_mod, "run_efs_unit", refused_make)
    adapter = FakeAdapter()

    with pytest.raises(type(refusal)):
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(606, chat_id, "hello")))

    assert adapter.delivered == [(str(chat_id), BUSY_NOTICE)]


def test_a_new_command_whose_detach_finds_no_connection_hears_the_busy_notice(monkeypatch):
    """/new runs no turn: it detaches the route's conversation in a unit of
    its own. A pool that refuses that unit tells the sender, who would
    otherwise hear nothing, and the refusal still propagates so the event
    records failed."""
    from cowork.channels.runtime import BUSY_NOTICE
    from cowork.db.units import DatabaseBusy

    def refused(self, scoped, *, binding):
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")

    monkeypatch.setattr(AntonChannelRuntime, "_detach_conversation", refused)
    adapter = FakeAdapter()

    with pytest.raises(DatabaseBusy):
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(604, 605, "/new")))

    assert adapter.delivered == [("605", BUSY_NOTICE)]


def _assistant_rows(chat_id: int) -> list:
    s = get_open_session()
    try:
        binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == str(chat_id))).one()
        return [
            m for m in s.exec(select(Message).where(Message.conversation_id == binding.anton_conversation_id)).all()
            if m.role == "assistant"
        ]
    finally:
        s.close()


def test_an_answer_whose_save_is_refused_every_time_is_still_delivered(monkeypatch, caplog):
    """The pod already produced the reply: a pool that refuses the answer's
    save on every try costs the conversation its row, not the sender the
    reply, and logs one WARNING naming the conversation."""
    import logging
    from cowork.db.units import DatabaseBusy

    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness())
    monkeypatch.setattr(runtime_mod, "_SAVE_BACKOFF_S", 0.0)
    tries = []

    def refused(self, conversation_id, text, events, harness=None, tool_rows=None):
        tries.append(conversation_id)
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")

    monkeypatch.setattr(ConversationService, "save_assistant_turn", refused)
    adapter = FakeAdapter()
    with caplog.at_level(logging.WARNING, logger=runtime_mod.log.name):
        asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(610, 611, "hi")))

    assert adapter.delivered == [("611", REPLY)]
    assert len(tries) == runtime_mod._SAVE_TRIES
    [record] = [r for r in caplog.records if r.name == runtime_mod.log.name and r.levelname == "WARNING"]
    assert record.getMessage() == (
        "channel turn: no database connection freed to save a message; the conversation lacks it"
    )
    assert (record.conversation_id, record.step, record.attempts) == (
        str(tries[0]), "save_answer", runtime_mod._SAVE_TRIES,
    )
    assert _assistant_rows(611) == []


def test_an_answer_whose_save_is_refused_once_is_saved_on_the_retry(monkeypatch):
    from cowork.db.units import DatabaseBusy

    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness())
    monkeypatch.setattr(runtime_mod, "_SAVE_BACKOFF_S", 0.0)
    real_save = ConversationService.save_assistant_turn
    tries = []

    def refused_once(self, *args, **kwargs):
        tries.append(1)
        if len(tries) == 1:
            raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")
        return real_save(self, *args, **kwargs)

    monkeypatch.setattr(ConversationService, "save_assistant_turn", refused_once)
    adapter = FakeAdapter()
    asyncio.run(_runtime_with(adapter).handle("telegram", _telegram_event(620, 621, "hi")))

    assert adapter.delivered == [("621", REPLY)]
    assert len(tries) == 2
    assert [m.content for m in _assistant_rows(621)] == [REPLY]


def test_slack_parses_shared_files(monkeypatch):
    from cowork.channels.plugins.slack import SlackBridge
    bridge = SlackBridge({"signing_secret": "ss", "bot_token": "xoxb"})
    body = json.dumps({"type": "event_callback", "event_id": "E2", "event": {
        "type": "message", "subtype": "file_share", "text": "", "channel": "C9", "ts": "2.2", "user": "U1",
        "files": [
            {"name": "notes.txt", "mimetype": "text/plain", "size": 10, "url_private": "https://files.slack/x"},
            {"name": "huge.bin", "mimetype": "application/octet-stream", "size": 30 * 1024 * 1024,
             "url_private": "https://files.slack/y"},
        ],
    }}).encode()
    events = asyncio.run(bridge.parse_inbound(body=body, headers={}, route_name="events"))
    assert len(events) == 1 and events[0].message.content == ""
    atts = events[0].message.attachments
    assert len(atts) == 1 and atts[0].filename == "notes.txt"  # oversize dropped
    assert atts[0].slack_url == "https://files.slack/x"

    async def fake_download(token, url):
        assert token == "xoxb" and url == "https://files.slack/x"
        return b"NOTES"

    monkeypatch.setattr(SlackBridge, "download_url", staticmethod(fake_download))
    assert asyncio.run(bridge.fetch_attachment(atts[0])) == b"NOTES"


def test_discord_parses_interaction_attachments(monkeypatch):
    from cowork.channels.plugins.discord import DiscordBridge
    bridge = DiscordBridge({"public_key": "00", "bot_token": "Bot x"})
    cmd = {"type": 2, "id": "I9", "channel_id": "CH2", "data": {
        "name": "ask", "options": [{"value": "analyse this"}],
        "resolved": {"attachments": {
            "1": {"filename": "data.csv", "content_type": "text/csv", "size": 5, "url": "https://cdn/x"},
            "2": {"filename": "big.bin", "size": 30 * 1024 * 1024, "url": "https://cdn/y"},
        }},
    }}
    events = asyncio.run(bridge.parse_inbound(body=json.dumps(cmd).encode(), headers={}, route_name="interactions"))
    atts = events[0].message.attachments
    assert len(atts) == 1 and atts[0].filename == "data.csv" and atts[0].discord_url == "https://cdn/x"

    async def fake_download(url):
        assert url == "https://cdn/x"
        return b"CSV"

    monkeypatch.setattr(DiscordBridge, "download_url", staticmethod(fake_download))
    assert asyncio.run(bridge.fetch_attachment(atts[0])) == b"CSV"


def test_whatsapp_parses_media_messages(monkeypatch):
    from cowork.channels.plugins.whatsapp import WhatsAppBridge
    bridge = WhatsAppBridge({"phone_number_id": "p", "access_token": "tok", "app_secret": "k", "verify_token": "v"})
    body = json.dumps({"entry": [{"changes": [{"value": {"messages": [
        {"type": "image", "from": "155", "id": "wamid.I", "timestamp": "1700000000",
         "image": {"id": "m1", "mime_type": "image/jpeg", "caption": "see"}},
        {"type": "document", "from": "155", "id": "wamid.D", "timestamp": "1700000000",
         "document": {"id": "m2", "mime_type": "application/pdf", "filename": "r.pdf"}},
    ]}}]}]}).encode()
    events = asyncio.run(bridge.parse_inbound(body=body, headers={}, route_name=None))
    assert len(events) == 2
    image, document = events
    assert image.message.content == "see" and image.message.attachments[0].mime_type == "image/jpeg"
    assert image.message.attachments[0].whatsapp_media_id == "m1"
    assert document.message.attachments[0].filename == "r.pdf"

    async def fake_info(token, media_id):
        assert token == "tok" and media_id == "m1"
        return {"url": "https://lookaside/x", "file_size": 10}

    async def fake_download(token, url):
        assert url == "https://lookaside/x"
        return b"IMG"

    monkeypatch.setattr(WhatsAppBridge, "media_info", staticmethod(fake_info))
    monkeypatch.setattr(WhatsAppBridge, "download_url", staticmethod(fake_download))
    assert asyncio.run(bridge.fetch_attachment(image.message.attachments[0])) == b"IMG"

    async def oversize_info(token, media_id):
        return {"url": "https://lookaside/x", "file_size": 30 * 1024 * 1024}

    monkeypatch.setattr(WhatsAppBridge, "media_info", staticmethod(oversize_info))
    assert asyncio.run(bridge.fetch_attachment(image.message.attachments[0])) is None


def test_slack_send_attachment(monkeypatch):
    from cowork.channels.plugins.slack import SlackBridge
    from anton.core.dispatch import PlatformAddress
    calls = []

    async def fake_web_api(bot_token, method, *, data=None, json_body=None):
        calls.append((method, data, json_body))
        if method == "files.getUploadURLExternal":
            return {"ok": True, "upload_url": "https://up.slack/x", "file_id": "F1"}
        return {"ok": True}

    async def fake_upload(upload_url, data):
        calls.append(("upload", upload_url, data))
        return True

    monkeypatch.setattr(SlackBridge, "web_api", staticmethod(fake_web_api))
    monkeypatch.setattr(SlackBridge, "upload_bytes", staticmethod(fake_upload))
    bridge = SlackBridge({"signing_secret": "ss", "bot_token": "xoxb"})

    mid = asyncio.run(bridge.send_attachment(
        address=PlatformAddress("slack", "C9", "111.222"), data=b"HI", filename="notes.txt"))
    assert mid == "F1"
    assert calls[0] == ("files.getUploadURLExternal", {"filename": "notes.txt", "length": 2}, None)
    assert calls[1] == ("upload", "https://up.slack/x", b"HI")
    method, _, payload = calls[2]
    assert method == "files.completeUploadExternal"
    assert payload["channel_id"] == "C9" and payload["thread_ts"] == "111.222"
    assert payload["files"] == [{"id": "F1", "title": "notes.txt"}]


def test_discord_send_attachment(monkeypatch):
    from cowork.channels.plugins.discord import DiscordBridge
    from anton.core.dispatch import PlatformAddress
    captured = {}

    async def fake_post(self, url, headers=None, files=None, **kw):
        captured.update(url=url, files=files)

        class R:
            status_code = 200

            def json(self):
                return {"id": "M77"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    bridge = DiscordBridge({"public_key": "00", "bot_token": "tok"})

    mid = asyncio.run(bridge.send_attachment(
        address=PlatformAddress("discord", "CH2", None), data=b"PNG", filename="chart.png",
    ))
    assert mid == "M77"
    assert captured["url"].endswith("/channels/CH2/messages")
    assert captured["files"]["files[0]"] == ("chart.png", b"PNG")


def test_whatsapp_send_attachment(monkeypatch):
    import datetime as dt
    from cowork.channels.plugins.whatsapp import WhatsAppBridge
    from anton.core.dispatch import PlatformAddress
    calls = []

    async def fake_upload(token, phone_id, name, mime, data):
        calls.append(("upload", name, mime, data))
        return "MEDIA9"

    async def fake_send(token, phone_id, recipient, media_id, name):
        calls.append(("send", recipient, media_id, name))
        return {"messages": [{"id": "wamid.OUT"}]}

    monkeypatch.setattr(WhatsAppBridge, "upload_media", staticmethod(fake_upload))
    monkeypatch.setattr(WhatsAppBridge, "send_media_message", staticmethod(fake_send))
    bridge = WhatsAppBridge({"phone_number_id": "p", "access_token": "tok", "app_secret": "k", "verify_token": "v"})

    def send():
        return asyncio.run(bridge.send_attachment(
            address=PlatformAddress("whatsapp", "155", None), data=b"PDF", filename="report.pdf",
        ))

    # outside the 24h window → refused before any upload
    try:
        send()
        raise SystemExit("window must be enforced")
    except RuntimeError:
        pass
    assert calls == []

    bridge._last_inbound["155"] = dt.datetime.now(dt.timezone.utc)
    mid = send()
    assert mid == "wamid.OUT"
    assert calls[0] == ("upload", "report.pdf", "application/pdf", b"PDF")
    assert calls[1] == ("send", "155", "MEDIA9", "report.pdf")


def test_channels_harness_selection_and_pinning(monkeypatch):
    import cowork.common.settings.user_settings as user_settings_mod

    anton_harness = FakeHarness()
    other_harness = FakeHarness()
    registered = {"anton", "other"}

    def fake_get_harness(name):
        if name == "anton":
            return anton_harness
        if name == "other" and "other" in registered:
            return other_harness
        raise ValueError(name)

    monkeypatch.setattr(runtime_mod, "get_harness", fake_get_harness)
    monkeypatch.setattr(user_settings_mod, "_harness_options", lambda: ["anton", "other"])

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s"})
    adapters = LiveAdapterRegistry(registry)
    adapters._cache[("telegram", None)] = FakeAdapter()
    runtime = AntonChannelRuntime(adapters)

    def turn(chat_id, update_id):
        event = asyncio.run(bridge.parse_inbound(
            body=telegram_update(update_id, chat_id, 1, "hi"), headers={}, route_name=None))[0]
        asyncio.run(runtime.handle("telegram", event))

    def harnesses_of(chat_id):
        s = get_open_session()
        binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == str(chat_id))).one()
        msgs = s.exec(select(Message).where(Message.conversation_id == binding.anton_conversation_id)).all()
        s.close()
        return sorted({m.harness for m in msgs if m.role == "assistant"})

    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    def set_channels_harness(name):
        # Persist to the DB — get_user_settings() loads fresh (no cache), so the
        # runtime must read the stored value, not a mutated in-memory object.
        s = get_open_session()
        try:
            SettingService(s).upsert_setting("channels_harness", name)
        finally:
            s.close()

    set_channels_harness("other")
    turn(700, 200)
    assert harnesses_of(700) == ["other"] and other_harness.inputs

    # Flipping the setting must never switch an existing conversation: pinned.
    set_channels_harness("anton")
    turn(700, 201)
    assert harnesses_of(700) == ["other"]

    # New conversations follow the current setting.
    turn(701, 202)
    assert harnesses_of(701) == ["anton"]

    # A stored row naming a harness that no longer exists (the API rejects
    # writing one, so seed it directly) resolves to the default rather than
    # failing the turn.
    from cowork.models.setting import Setting

    s = get_open_session()
    try:
        row = s.exec(select(Setting).where(Setting.key == "channels_harness", Setting.scope.is_(None))).one()
        row.value = "ghost"
        s.add(row)
        s.commit()
    finally:
        s.close()
    turn(702, 203)
    assert harnesses_of(702) == ["anton"]

    # A conversation pinned to a harness that has since been removed keeps
    # working on the default; its history keeps the old tag.
    registered.discard("other")
    turn(700, 204)
    assert harnesses_of(700) == ["anton", "other"]


def test_channel_agent_endpoint_validates_and_persists(monkeypatch):
    import pytest
    from fastapi import HTTPException

    import cowork.api.v1.endpoints.channels as channels_ep
    import cowork.common.settings.user_settings as user_settings_mod
    from cowork.api.v1.endpoints.channels import get_channel_agent, set_channel_agent
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.schemas.channels import ChannelAgentUpdateRequest
    from cowork.services.settings import SettingService

    # A second registered harness, so the switch is observable.
    monkeypatch.setattr(channels_ep, "available_harness_ids", lambda: ["anton", "other"])
    monkeypatch.setattr(user_settings_mod, "_harness_options", lambda: ["anton", "other"])

    session = get_open_session()
    scoped = ScopedSession(session, LOCAL_SCOPE)
    try:
        # Unknown harness is rejected, not persisted.
        with pytest.raises(HTTPException) as exc:
            set_channel_agent(ChannelAgentUpdateRequest(harness="ghost"), session, scoped)
        assert exc.value.status_code == 400

        resp = set_channel_agent(ChannelAgentUpdateRequest(harness="other"), session, scoped)
        assert resp.harness == "other"
        assert resp.options == ["anton", "other"]
        assert get_channel_agent(scoped).harness == "other"
        assert get_user_settings().channels_harness == "other"
    finally:
        session.close()

    # Reset so the stored setting doesn't leak into other tests.
    session = get_open_session()
    try:
        SettingService(session).delete_setting("channels_harness")
    finally:
        session.close()


def test_channel_agent_switch_resets_bound_conversations():
    from cowork.models.channel import ChannelBinding
    from cowork.services.channel_bindings import ChannelBindingService
    from cowork.services.conversations import ConversationService
    from cowork.services.projects import GENERAL_PROJECT_ID

    session = get_open_session()
    bid = None
    try:
        conv = ConversationService(session).create_conversation(topic="chan", project_id=GENERAL_PROJECT_ID)
        binding = ChannelBinding(
            channel_type="telegram",
            external_group_id="reset-test",
            external_thread_key="__default__",
            anton_conversation_id=conv.id,
            anton_project_id=GENERAL_PROJECT_ID,
            trigger_rule="always",
        )
        session.add(binding)
        session.commit()
        session.refresh(binding)
        bid = binding.id

        reset = ChannelBindingService(ScopedSession(session, LOCAL_SCOPE)).reset_conversations(channel_type="telegram")
        assert reset >= 1
        session.expire_all()
        assert session.get(ChannelBinding, bid).anton_conversation_id is None
    finally:
        if bid is not None:
            row = session.get(ChannelBinding, bid)
            if row is not None:
                session.delete(row)
                session.commit()
        session.close()


def test_is_new_command_matching():
    from cowork.channels.runtime import is_new_command

    assert is_new_command("/new")
    assert is_new_command("  /New ")
    assert is_new_command("/new@MyBot")
    assert is_new_command("/new@MyBot", is_mention=True)
    assert is_new_command("@mybot /new")
    assert is_new_command("<@U123ABC> /new")
    assert is_new_command("/new", is_mention=False)
    # A suffixed command the platform says isn't addressed to us belongs to another bot.
    assert not is_new_command("/new@OtherBot", is_mention=False)
    assert not is_new_command("/new@")
    assert not is_new_command("how do I use /new")
    assert not is_new_command("/new please")
    assert not is_new_command("/newer")
    assert not is_new_command("new")
    assert not is_new_command("")


def test_new_command_starts_fresh_conversation(monkeypatch):
    fake_harness = FakeHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake_harness)

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})

    def event(update_id, text):
        return asyncio.run(bridge.parse_inbound(
            body=telegram_update(update_id, 777, 1, text), headers={}, route_name=None,
        ))[0]

    adapters = LiveAdapterRegistry(registry)
    adapter = FakeAdapter()
    adapters._cache[("telegram", None)] = adapter
    runtime = AntonChannelRuntime(adapters)

    asyncio.run(runtime.handle("telegram", event(100, "hi")))
    s = get_open_session()
    stmt = select(ChannelBinding).where(ChannelBinding.external_group_id == "777")
    binding = s.exec(stmt).one()
    first_conv = binding.anton_conversation_id
    assert first_conv is not None

    asyncio.run(runtime.handle("telegram", event(101, "/new")))
    s.expire_all()
    binding = s.exec(stmt).one()
    assert binding.anton_conversation_id is None
    assert not s.exec(select(ChannelSession).where(ChannelSession.binding_id == binding.id)).all()
    # Deterministic confirmation naming the project; the harness never ran for this turn.
    assert adapter.delivered[-1] == ("777", 'Starting fresh — your next message begins a new conversation in the "general" project.')
    assert len(fake_harness.inputs) == 1

    asyncio.run(runtime.handle("telegram", event(102, "hi again")))
    s.expire_all()
    binding = s.exec(stmt).one()
    assert binding.anton_conversation_id is not None
    assert binding.anton_conversation_id != first_conv
    assert adapter.delivered[-1] == ("777", REPLY)
    s.close()


def test_binding_project_change_detaches_conversation():
    from cowork.models.channel import ChannelBinding
    from cowork.models.project import Project
    from cowork.schemas.channels import BindingUpdateRequest
    from cowork.services.channel_bindings import ChannelBindingService
    from cowork.services.conversations import ConversationService
    from cowork.services.projects import GENERAL_PROJECT_ID

    session = get_open_session()
    bid = None
    other = None
    try:
        other = Project(name="reroute-target", path="/tmp/reroute-target")
        session.add(other)
        conv = ConversationService(session).create_conversation(topic="chan", project_id=GENERAL_PROJECT_ID)
        binding = ChannelBinding(
            channel_type="telegram",
            external_group_id="reroute-test",
            external_thread_key="__default__",
            anton_conversation_id=conv.id,
            anton_project_id=GENERAL_PROJECT_ID,
            trigger_rule="always",
        )
        session.add(binding)
        session.commit()
        session.refresh(binding)
        session.refresh(other)
        bid = binding.id
        session.add(ChannelSession(binding_id=bid, external_session_key="__default__"))
        session.commit()

        svc = ChannelBindingService(ScopedSession(session, LOCAL_SCOPE))

        # Same project: conversation stays pinned.
        svc.update(bid, BindingUpdateRequest(anton_project_id=GENERAL_PROJECT_ID))
        session.expire_all()
        assert session.get(ChannelBinding, bid).anton_conversation_id == conv.id

        # New project: conversation and session rows are detached.
        svc.update(bid, BindingUpdateRequest(anton_project_id=other.id))
        session.expire_all()
        row = session.get(ChannelBinding, bid)
        assert row.anton_project_id == other.id
        assert row.anton_conversation_id is None
        assert not session.exec(select(ChannelSession).where(ChannelSession.binding_id == bid)).all()

        # Explicit conversation in the same request wins over the detach.
        svc.update(bid, BindingUpdateRequest(anton_project_id=GENERAL_PROJECT_ID, anton_conversation_id=conv.id))
        session.expire_all()
        row = session.get(ChannelBinding, bid)
        assert row.anton_project_id == GENERAL_PROJECT_ID
        assert row.anton_conversation_id == conv.id
    finally:
        if bid is not None:
            row = session.get(ChannelBinding, bid)
            if row is not None:
                session.delete(row)
        if other is not None and other.id is not None:
            row = session.get(Project, other.id)
            if row is not None:
                session.delete(row)
        session.commit()
        session.close()


def test_plugin_capabilities_match_declared_hooks():
    registry = PluginRegistry()
    load_first_party_plugins(registry)
    assert registry.channel_types(), "expected first-party plugins to be discovered"
    for plugin in registry.all():
        caps = plugin.capabilities
        if caps.supports_oauth:
            assert plugin.oauth is not None, f"{plugin.channel_type}: oauth capability without OAuthSpec"
        if caps.supports_webhook_setup or caps.supports_teardown:
            assert plugin.lifecycle is not None, f"{plugin.channel_type}: lifecycle capability without lifecycle"
        if caps.supports_webhook_ingress:
            assert plugin.webhooks, f"{plugin.channel_type}: webhook capability without webhook routes"


# --- ENG-591: group mention gating + channel context ------------------------

def group_telegram_update(
    update_id: int, chat_id: int, message_id: int, text: str,
    *, entities=None, reply_to=None,
) -> bytes:
    msg = {
        "message_id": message_id,
        "from": {"id": 42, "is_bot": False, "first_name": "Alice", "last_name": "Realtor"},
        "chat": {"id": chat_id, "type": "supergroup"},
        "date": 1700000000,
        "text": text,
    }
    if entities is not None:
        msg["entities"] = entities
    if reply_to is not None:
        msg["reply_to_message"] = reply_to
    return json.dumps({"update_id": update_id, "message": msg}).encode()


def test_telegram_group_mention_only_flow(monkeypatch):
    calls: list[tuple[str, dict]] = []

    async def fake_call(bot_token, method, payload):
        calls.append((method, dict(payload)))
        if method == "getMe":
            return {"ok": True, "result": {"id": 999, "username": "AntonBot"}}
        return {"ok": True, "result": {"message_id": 1}}

    fake_harness = FakeHarness()
    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake_harness)

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    # No bot_username credential: identity must come from getMe.
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x"})
    adapters = LiveAdapterRegistry(registry)
    adapter = FakeAdapter()
    adapters._cache[("telegram", None)] = adapter
    runtime = AntonChannelRuntime(adapters)

    async def inbound(body):
        for event in await bridge.parse_inbound(body=body, headers={}, route_name=None):
            await runtime.handle("telegram", event)

    # Plain group message: binding auto-created as mention_only, turn skipped.
    asyncio.run(inbound(group_telegram_update(70, -100123, 1, "what listings are there?")))
    s = get_open_session()
    binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == "-100123")).one()
    assert binding.trigger_rule == "mention_only"
    assert binding.anton_conversation_id is None
    s.close()
    assert adapter.delivered == []

    # @mention (case differs from getMe's username) → served, with the group
    # channel context handed to the harness.
    asyncio.run(inbound(group_telegram_update(71, -100123, 2, "@antonbot show me listings")))
    assert len(adapter.delivered) == 1
    assert fake_harness.channel_contexts == [
        ChannelContext(channel_type="telegram", is_group=True, display_name=None, instructions=None)
    ]
    # Group turns carry speaker attribution: harness input and stored history
    # are prefixed with the sender's name.
    assert fake_harness.inputs[0] == [
        {"type": "text", "text": "Alice Realtor: @antonbot show me listings"}
    ]
    s = get_open_session()
    binding = s.exec(select(ChannelBinding).where(ChannelBinding.external_group_id == "-100123")).one()
    user_msgs = [
        m for m in s.exec(select(Message).where(Message.conversation_id == binding.anton_conversation_id)).all()
        if m.role == "user"
    ]
    assert user_msgs and user_msgs[0].content == "Alice Realtor: @antonbot show me listings"
    s.close()

    # Replying to one of the bot's messages addresses it too.
    asyncio.run(inbound(group_telegram_update(
        72, -100123, 3, "and the price?", reply_to={"from": {"id": 999, "is_bot": True}},
    )))
    assert len(adapter.delivered) == 2

    # Identity fetched once, then cached.
    assert [m for (m, _p) in calls if m == "getMe"] == ["getMe"]


def test_telegram_group_mention_detection(monkeypatch):
    getme_methods: list[str] = []

    async def fake_call(bot_token, method, payload):
        getme_methods.append(method)
        return {"ok": True, "result": {"id": 999, "username": "AntonBot"}}

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))

    async def parse(bridge, body):
        return (await bridge.parse_inbound(body=body, headers={}, route_name=None))[0]

    bridge = telegram_plugin.TelegramBridge({"bot_token": "x"})

    # text_mention entities carry the target user, matched by bot id.
    ev = asyncio.run(parse(bridge, group_telegram_update(
        80, -200, 1, "Anton what's new?",
        entities=[{"type": "text_mention", "offset": 0, "length": 5, "user": {"id": 999}}],
    )))
    assert ev.message.is_mention is True

    # Entity offsets are UTF-16 code units: non-BMP text before the mention
    # must not shift the matched window.
    ev = asyncio.run(parse(bridge, group_telegram_update(
        81, -200, 2, "\U0001F44D\U0001F44D @AntonBot hi",
        entities=[{"type": "mention", "offset": 5, "length": 9}],
    )))
    assert ev.message.is_mention is True

    # Plain group text with no mention → not a mention; sender name captured.
    ev = asyncio.run(parse(bridge, group_telegram_update(82, -200, 3, "hello all")))
    assert ev.message.is_mention is False
    assert ev.message.sender_name == "Alice Realtor"

    # Private chats are always mentions and never trigger getMe.
    fresh = telegram_plugin.TelegramBridge({"bot_token": "x"})
    before = len(getme_methods)
    ev = asyncio.run(parse(fresh, telegram_update(83, 5, 4, "hi")))
    assert ev.message.is_mention is True
    assert len(getme_methods) == before


def test_telegram_mention_falls_back_to_credential_when_getme_fails(monkeypatch):
    async def failing_call(bot_token, method, payload):
        raise ConnectionError("api down")

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(failing_call))

    async def parse(bridge, body):
        return (await bridge.parse_inbound(body=body, headers={}, route_name=None))[0]

    with_cred = telegram_plugin.TelegramBridge({"bot_token": "x", "bot_username": "AntonBot"})
    ev = asyncio.run(parse(with_cred, group_telegram_update(90, -300, 1, "hey @antonbot")))
    assert ev.message.is_mention is True

    without = telegram_plugin.TelegramBridge({"bot_token": "x"})
    ev = asyncio.run(parse(without, group_telegram_update(91, -300, 2, "hey @antonbot")))
    assert ev.message.is_mention is False


def test_should_respond_matrix():
    def event(text="hi", is_mention=False):
        return SimpleNamespace(message=SimpleNamespace(content=text, is_mention=is_mention))

    def binding(rule, pattern=None):
        return SimpleNamespace(trigger_rule=rule, trigger_pattern=pattern)

    should = AntonChannelRuntime._should_respond
    assert asyncio.run(should(binding("always"), event())) is True
    assert asyncio.run(should(binding("mention_only"), event(is_mention=True))) is True
    assert asyncio.run(should(binding("mention_only"), event(is_mention=False))) is False
    assert asyncio.run(should(binding("regex", r"listing"), event("any listings?"))) is True
    assert asyncio.run(should(binding("regex", r"listing"), event("hello"))) is False
    assert asyncio.run(should(binding("regex", None), event("hello"))) is False
    assert asyncio.run(should(binding("regex", "("), event("hello"))) is False


def test_should_respond_regex_catastrophic_backtracking_is_bounded():
    # A syntactically-valid-but-pathological trigger_pattern (ChannelBindingService
    # only checks it parses, not that it's cheap to run) must not hang. Under a
    # backtracking engine (stdlib `re`) this input takes exponential time; re2's
    # linear-time automaton has no such failure mode, so no timeout is needed —
    # this just proves the match still completes fast and correctly.
    def event(text):
        return SimpleNamespace(message=SimpleNamespace(content=text, is_mention=False))

    def binding(pattern):
        return SimpleNamespace(trigger_rule="regex", trigger_pattern=pattern)

    evil_pattern = r"(a|aa)+$"
    evil_input = "a" * 5000 + "!"  # no trailing match -> pathological for a backtracking engine

    started = time.monotonic()
    result = asyncio.run(AntonChannelRuntime._should_respond(binding(evil_pattern), event(evil_input)))
    elapsed = time.monotonic() - started

    assert result is False
    assert elapsed < 1.0  # re2 resolves this in microseconds; generous margin for CI jitter


def test_should_respond_regex_re2_incompatible_syntax_is_caught():
    # re2 isn't syntax-compatible with `re`/`regex` — no backreferences, no
    # lookahead/lookbehind. ChannelBindingService._validate_trigger rejects these
    # at binding-creation time, but this proves the runtime match path is also
    # safe (returns no-match rather than raising) for any pattern that reaches
    # it regardless, e.g. one created before this constraint existed.
    def event(text):
        return SimpleNamespace(message=SimpleNamespace(content=text, is_mention=False))

    def binding(pattern):
        return SimpleNamespace(trigger_rule="regex", trigger_pattern=pattern)

    backreference_pattern = r"(a)\1"  # valid in `re`/`regex`, unsupported by re2

    result = asyncio.run(AntonChannelRuntime._should_respond(binding(backreference_pattern), event("aa")))

    assert result is False


def test_binding_instructions_roundtrip():
    app = create_app()

    async def flow():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/api/v1/channels/bindings", json={
                "channel_type": "telegram",
                "external_group_id": "-400500",
                "instructions": "You are the listings concierge.",
            })
            assert r.status_code == 201
            body = r.json()
            assert body["instructions"] == "You are the listings concierge."

            r = await client.patch(
                f"/api/v1/channels/bindings/{body['id']}", json={"instructions": "Be brief."}
            )
            assert r.status_code == 200 and r.json()["instructions"] == "Be brief."

            r = await client.get("/api/v1/channels/bindings", params={"channel_type": "telegram"})
            row = next(b for b in r.json() if b["id"] == body["id"])
            assert row["instructions"] == "Be brief."

    asyncio.run(flow())


def test_telegram_identity_fetch_runs_on_fresh_boot(monkeypatch):
    # Regression (PR #177 review): a 0.0 cooldown sentinel with time.monotonic
    # (which counts from boot) skipped the very first getMe whenever machine
    # uptime < IDENTITY_RETRY_S — fresh CI VMs and just-provisioned agent
    # boxes ignored group mentions for their first ~5 minutes.
    methods: list[str] = []

    async def fake_call(bot_token, method, payload):
        methods.append(method)
        return {"ok": True, "result": {"id": 999, "username": "AntonBot"}}

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))
    monkeypatch.setattr(telegram_plugin, "time", SimpleNamespace(monotonic=lambda: 120.0))

    bridge = telegram_plugin.TelegramBridge({"bot_token": "x"})
    ev = asyncio.run(bridge.parse_inbound(
        body=group_telegram_update(95, -500, 1, "hey @antonbot"), headers={}, route_name=None,
    ))[0]
    assert "getMe" in methods
    assert ev.message.is_mention is True


def test_telegram_mention_fallback_requires_username_boundary(monkeypatch):
    # "@antonbot" must not match a mention aimed at "@antonbotdev".
    async def fake_call(bot_token, method, payload):
        return {"ok": True, "result": {"id": 999, "username": "AntonBot"}}

    monkeypatch.setattr(telegram_plugin.TelegramBridge, "_call", staticmethod(fake_call))

    async def parse(bridge, body):
        return (await bridge.parse_inbound(body=body, headers={}, route_name=None))[0]

    bridge = telegram_plugin.TelegramBridge({"bot_token": "x"})
    ev = asyncio.run(parse(bridge, group_telegram_update(96, -600, 1, "@antonbotdev please help")))
    assert ev.message.is_mention is False
    ev = asyncio.run(parse(bridge, group_telegram_update(97, -600, 2, "hey @antonbot, listings?")))
    assert ev.message.is_mention is True
