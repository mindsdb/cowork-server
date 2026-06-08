"""End-to-end smoke for the Telegram channel through the full server stack.

Drives the real ASGI app over httpx on a single event loop so the runtime's
background task is awaitable (``drain_background_tasks``). Telegram HTTP goes
through ``TelegramBridge._call`` (mocked) and the Anton harness is faked — no
real credentials, network, or LLM.
"""
import asyncio
import json

import httpx
from sqlmodel import select

import cowork.channels.plugins.telegram as telegram_plugin
import cowork.channels.runtime as runtime_mod
from cowork.channels.registry import PluginRegistry, load_first_party_plugins
from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
from cowork.channels.webhooks import drain_background_tasks
from cowork.db.session import get_open_session
from cowork.models.channel import ChannelBinding, ChannelEvent, ChannelSession
from cowork.models.message import Message
from cowork.server import create_app

REPLY = "hello from anton"
LINK_PREFIX = "https://app.example.com/c/"


class FakeHarness:
    """Stands in for the Anton harness — one assistant delta, no LLM."""

    def __init__(self, tool_event: bool = False, delay: float = 0.0):
        self.tool_event = tool_event
        self.delay = delay
        self.synced: list[int] = []
        self.inputs: list[list[dict]] = []

    async def sync_skills(self, skills):
        self.synced.append(len(skills))

    async def stream_response(self, *, conversation, input):
        self.inputs.append(input)
        if False:
            yield

    async def formatter(self, stream, model, event_sink):
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


def telegram_update(update_id: int, chat_id: int, message_id: int, text: str) -> bytes:
    return json.dumps({
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": chat_id, "type": "private"},
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

            assert fake_harness.synced, "sync_skills must run before the turn"

            s = get_open_session()
            binding = s.exec(select(ChannelBinding)).one()
            assert binding.channel_type == "telegram" and binding.anton_conversation_id is not None
            sessions = s.exec(select(ChannelSession)).all()
            assert len(sessions) == 1 and sessions[0].binding_id == binding.id
            msgs = s.exec(select(Message).where(Message.conversation_id == binding.anton_conversation_id)).all()
            assert sorted(m.role for m in msgs) == ["assistant", "user"]
            assistant = next(m for m in msgs if m.role == "assistant")
            assert assistant.content == REPLY and assistant.harness == "anton"
            assert inbound_events(s)[0].status == "routed"
            s.close()

            # No tool events in this turn → reply delivered verbatim, no link.
            sends = [p for (m, p) in calls if m == "sendMessage"]
            assert len(sends) == 1 and sends[0]["chat_id"] == "7" and sends[0]["text"] == REPLY

            # duplicate webhook → dropped, Anton not run twice
            r = await client.post("/api/v1/channels/telegram/webhook", content=body, headers=headers)
            assert r.status_code == 200
            await drain_background_tasks()
            s = get_open_session()
            assert len(inbound_events(s)) == 1
            assert len([m for m in s.exec(select(Message)).all() if m.role == "assistant"]) == 1
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
    adapters._cache["telegram"] = adapter
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
    adapters._cache["telegram"] = adapter
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


def test_inbound_media_becomes_harness_blocks(monkeypatch):
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
    adapters._cache["telegram"] = MediaAdapter()
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


def test_telegram_send_attachment(monkeypatch, tmp_path):
    captured = {}

    async def fake_post(self, url, data=None, files=None, **kw):
        captured.update(url=url, data=data, files=files)

        class R:
            def json(self):
                return {"ok": True, "result": {"message_id": 321}}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    source = tmp_path / "report.pdf"
    source.write_bytes(b"PDFDATA")
    bridge = telegram_plugin.TelegramBridge({"bot_token": "tok", "secret_token": "s"})
    from anton.core.dispatch import PlatformAddress

    mid = asyncio.run(bridge.send_attachment(address=PlatformAddress("telegram", "7", None), path=str(source)))
    assert mid == "321"
    assert captured["url"].endswith("/sendDocument") and captured["data"] == {"chat_id": "7"}
    assert captured["files"]["document"] == ("report.pdf", b"PDFDATA")

    monkeypatch.setattr(telegram_plugin, "TELEGRAM_MAX_UPLOAD_BYTES", 3)
    try:
        asyncio.run(bridge.send_attachment(address=PlatformAddress("telegram", "7", None), path=str(source)))
        raise SystemExit("oversize must raise")
    except RuntimeError:
        pass


def test_turn_artifacts_delivered(monkeypatch):
    import os
    import time as time_mod

    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: FakeHarness(tool_event=True))

    class ArtifactAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.sent = []

        async def send_attachment(self, *, address, path, filename=None):
            self.sent.append((address.platform_id, path, filename))
            return "1"

    from cowork.models.project import Project
    from cowork.services.projects import GENERAL_PROJECT_ID
    s = get_open_session()
    project_dir = s.get(Project, GENERAL_PROJECT_ID).path
    s.close()

    fresh = os.path.join(project_dir, ".anton", "artifacts", "demo")
    os.makedirs(fresh, exist_ok=True)
    with open(os.path.join(fresh, "metadata.json"), "w") as f:
        f.write(json.dumps({"name": "Demo", "type": "html"}))
    with open(os.path.join(fresh, "dashboard.html"), "w") as f:
        f.write("<html/>")
    stale = os.path.join(project_dir, ".anton", "artifacts", "old")
    os.makedirs(stale, exist_ok=True)
    with open(os.path.join(stale, "metadata.json"), "w") as f:
        f.write(json.dumps({"name": "Old", "type": "html"}))
    with open(os.path.join(stale, "page.html"), "w") as f:
        f.write("<html/>")
    past = time_mod.time() - 3600
    os.utime(os.path.join(stale, "metadata.json"), (past, past))

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s"})
    event = asyncio.run(bridge.parse_inbound(
        body=telegram_update(90, 555, 1, "make a dashboard"), headers={}, route_name=None,
    ))[0]

    adapters = LiveAdapterRegistry(registry)
    adapter = ArtifactAdapter()
    adapters._cache["telegram"] = adapter
    asyncio.run(AntonChannelRuntime(adapters).handle("telegram", event))

    # Text reply (with link) lands first, then exactly the fresh artifact.
    assert adapter.delivered and adapter.delivered[0][0] == "555"
    assert len(adapter.sent) == 1
    chat_id, path, filename = adapter.sent[0]
    assert chat_id == "555" and path.endswith("dashboard.html") and filename == "dashboard.html"


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


def test_slack_send_attachment(monkeypatch, tmp_path):
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
    source = tmp_path / "notes.txt"
    source.write_bytes(b"HI")
    bridge = SlackBridge({"signing_secret": "ss", "bot_token": "xoxb"})

    mid = asyncio.run(bridge.send_attachment(
        address=PlatformAddress("slack", "C9", "111.222"), path=str(source)))
    assert mid == "F1"
    assert calls[0] == ("files.getUploadURLExternal", {"filename": "notes.txt", "length": 2}, None)
    assert calls[1] == ("upload", "https://up.slack/x", b"HI")
    method, _, payload = calls[2]
    assert method == "files.completeUploadExternal"
    assert payload["channel_id"] == "C9" and payload["thread_ts"] == "111.222"
    assert payload["files"] == [{"id": "F1", "title": "notes.txt"}]


def test_discord_send_attachment(monkeypatch, tmp_path):
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
    source = tmp_path / "chart.png"
    source.write_bytes(b"PNG")
    bridge = DiscordBridge({"public_key": "00", "bot_token": "tok"})

    mid = asyncio.run(bridge.send_attachment(address=PlatformAddress("discord", "CH2", None), path=str(source)))
    assert mid == "M77"
    assert captured["url"].endswith("/channels/CH2/messages")
    assert captured["files"]["files[0]"] == ("chart.png", b"PNG")


def test_whatsapp_send_attachment(monkeypatch, tmp_path):
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
    source = tmp_path / "report.pdf"
    source.write_bytes(b"PDF")
    bridge = WhatsAppBridge({"phone_number_id": "p", "access_token": "tok", "app_secret": "k", "verify_token": "v"})

    # outside the 24h window → refused before any upload
    try:
        asyncio.run(bridge.send_attachment(address=PlatformAddress("whatsapp", "155", None), path=str(source)))
        raise SystemExit("window must be enforced")
    except RuntimeError:
        pass
    assert calls == []

    bridge._last_inbound["155"] = dt.datetime.now(dt.timezone.utc)
    mid = asyncio.run(bridge.send_attachment(address=PlatformAddress("whatsapp", "155", None), path=str(source)))
    assert mid == "wamid.OUT"
    assert calls[0] == ("upload", "report.pdf", "application/pdf", b"PDF")
    assert calls[1] == ("send", "155", "MEDIA9", "report.pdf")


def test_channels_harness_selection_and_pinning(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings

    anton_harness = FakeHarness()
    hermes_harness = FakeHarness()

    def fake_get_harness(name):
        if name == "anton":
            return anton_harness
        if name == "hermes":
            return hermes_harness
        raise ValueError(name)

    monkeypatch.setattr(runtime_mod, "get_harness", fake_get_harness)

    registry = PluginRegistry()
    load_first_party_plugins(registry)
    bridge = telegram_plugin.TelegramBridge({"bot_token": "x", "secret_token": "s"})
    adapters = LiveAdapterRegistry(registry)
    adapters._cache["telegram"] = FakeAdapter()
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

    settings = get_app_settings()
    monkeypatch.setattr(settings, "channels_harness", "hermes")
    turn(700, 200)
    assert harnesses_of(700) == ["hermes"] and hermes_harness.inputs

    # Flipping the setting must never switch an existing conversation: pinned.
    monkeypatch.setattr(settings, "channels_harness", "anton")
    turn(700, 201)
    assert harnesses_of(700) == ["hermes"]

    # New conversations follow the current setting.
    turn(701, 202)
    assert harnesses_of(701) == ["anton"]

    # Unregistered name falls back to the default rather than failing the turn.
    monkeypatch.setattr(settings, "channels_harness", "ghost")
    turn(702, 203)
    assert harnesses_of(702) == ["anton"]


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
