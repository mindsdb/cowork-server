"""Browser Control M1 `open_url` — user-directed URL = implicit grant.

Covers the deterministic user-anchor guard (`url_user_anchored`), the
BridgeClient `open_url` path (denial creates NO grant and enqueues NOTHING;
success grants the new host, revokes prior grants, moves `active_domain`,
and enqueues an `open_url` command with the href), the scenario-5 regression
(`follow_link` cross-domain stays blocked after an open_url retarget), and
the tool-schema/prompt contract.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.browser import BrowserAction, BrowserSession, BrowserTabGrant
from cowork.models.conversation import Conversation
from cowork.schemas.browser import (
    BridgeCommandResult,
    BrowserActionClass,
    BrowserActionType,
    PermissionDecision,
    ResultCode,
)
from cowork.services.browser.bridge import BridgeCommandService
from cowork.services.browser.client import (
    OPEN_URL_NOT_ANCHORED_DETAIL,
    BridgeClient,
    _session_locks,
    url_user_anchored,
)
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _make_session(session: Session, domain="example.com", grant=True) -> BrowserSession:
    conv = Conversation(topic="t", project_id=GENERAL_PROJECT_ID)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    bs = BrowserSession(
        conversation_id=conv.id, project_id=GENERAL_PROJECT_ID,
        active_domain=domain, available=True,
    )
    session.add(bs)
    session.commit()
    session.refresh(bs)
    if grant:
        session.add(
            BrowserTabGrant(
                session_id=bs.id, domain=domain,
                action_class=BrowserActionClass.navigate.value,
                decision=PermissionDecision.granted.value,
                granted_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
    return bs


def _grants(session: Session, bs: BrowserSession) -> list[BrowserTabGrant]:
    return list(
        session.exec(
            select(BrowserTabGrant).where(BrowserTabGrant.session_id == bs.id)
        ).all()
    )


# ── url_user_anchored (pure guard) ───────────────────────────────────
class TestUrlUserAnchored:
    def test_host_in_user_message(self):
        assert url_user_anchored("https://bbc.co.uk/news", ["go to bbc.co.uk"])

    def test_case_insensitive(self):
        assert url_user_anchored("https://BBC.co.uk", ["open BBC.CO.UK please"])

    def test_www_and_path_reduce_to_registrable_host(self):
        # Both sides are reduced through registrable_host, so www/path/
        # subdomain variants on either side anchor the same site.
        assert url_user_anchored(
            "https://www.bbc.co.uk/news/live", ["please open https://bbc.co.uk"]
        )
        assert url_user_anchored(
            "https://bbc.co.uk", ["go to www.bbc.co.uk for me"]
        )
        # Subdomain target: news.bbc.co.uk reduces to bbc.co.uk — the user
        # naming the site covers its subdomains (same registrable site).
        assert url_user_anchored(
            "https://news.bbc.co.uk/today", ["go to bbc.co.uk"]
        )

    def test_bare_host_mid_sentence_with_punctuation(self):
        assert url_user_anchored("https://example.com", ["try example.com, please"])
        assert url_user_anchored("http://example.com/x", ["(see example.com!)"])

    def test_full_url_in_user_text(self):
        assert url_user_anchored(
            "https://bbc.co.uk", ["open https://www.bbc.co.uk/news"]
        )

    def test_host_absent(self):
        assert not url_user_anchored(
            "https://evil.com/x", ["go to bbc.co.uk", "what's the weather?"]
        )

    def test_superstring_host_does_not_anchor(self):
        # HIGH 2 regression: raw substring matching allowed these.
        # "abbc.co.uk" must NOT anchor bbc.co.uk (different registrable host).
        assert not url_user_anchored("https://bbc.co.uk", ["go to abbc.co.uk"])
        # "bbc.co.uk.evil.com" must NOT anchor bbc.co.uk — it anchors
        # evil.com (its own registrable host) instead.
        assert not url_user_anchored(
            "https://bbc.co.uk", ["go to bbc.co.uk.evil.com"]
        )
        assert url_user_anchored(
            "https://evil.com", ["go to bbc.co.uk.evil.com"]
        )

    def test_ip_prefix_does_not_anchor(self):
        # "127.0.0.10" must NOT anchor 127.0.0.1 (boundary-anchored tokens).
        assert not url_user_anchored("http://127.0.0.1:8080", ["ping 127.0.0.10"])
        assert url_user_anchored("http://127.0.0.1:8080", ["ping 127.0.0.1"])

    def test_empty_inputs_never_match(self):
        assert not url_user_anchored("", ["bbc.co.uk"])
        assert not url_user_anchored("https://bbc.co.uk", [])
        assert not url_user_anchored("https://bbc.co.uk", ["", None])  # type: ignore[list-item]


# ── tools.py user-text extraction ────────────────────────────────────
class TestExtractUserTexts:
    def _extract(self, history):
        from cowork.harnesses.anton_harness import tools as tools_mod

        class _S:
            _history = history

        return tools_mod._extract_user_texts(_S())

    def test_string_content(self):
        assert self._extract(
            [
                {"role": "user", "content": "go to bbc.co.uk"},
                {"role": "assistant", "content": "sure"},
            ]
        ) == ["go to bbc.co.uk"]

    def test_block_content(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "open example.org"},
                    {"type": "image", "source": {}},
                ],
            }
        ]
        assert self._extract(history) == ["open example.org"]

    def test_assistant_text_excluded(self):
        history = [
            {"role": "assistant", "content": "I suggest evil.com"},
            {"role": "assistant", "content": [{"type": "text", "text": "evil.com"}]},
        ]
        assert self._extract(history) == []

    def test_tool_result_blocks_under_user_role_excluded(self):
        # tool_result blocks ride under the user role in Anthropic-style
        # history but are NOT the user's words — page text from a previous
        # inspect must never anchor an open_url.
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "text": "evil.com"},
                ],
            }
        ]
        assert self._extract(history) == []

    def test_defensive_on_malformed(self):
        assert self._extract(None) == []
        assert self._extract([{"role": "user"}, "junk", {"content": "x"}]) == []


# ── BridgeClient.open_url: guard denial ──────────────────────────────
def test_open_url_not_anchored_denied_no_grant_no_command(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(
            bs.conversation_id,
            "open_url",
            href="https://bbc.co.uk/news",
            user_texts=["what's in the news today?"],  # never names the site
        )
    )
    assert verdict.result_code == ResultCode.permission_denied
    assert verdict.action_type == BrowserActionType.open_url
    assert verdict.detail == OPEN_URL_NOT_ANCHORED_DETAIL
    # NO grant was created for the new host, NO command enqueued.
    assert broker.pending_count(str(bs.id)) == 0
    grants = _grants(session, bs)
    assert [g.domain for g in grants] == ["example.com"]
    session.refresh(bs)
    assert bs.active_domain == "example.com"


def test_open_url_missing_href_is_error_without_dispatch(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(bs.conversation_id, "open_url", user_texts=["go to bbc.co.uk"])
    )
    assert verdict.result_code == ResultCode.error
    assert broker.pending_count(str(bs.id)) == 0
    assert [g.domain for g in _grants(session, bs)] == ["example.com"]


# ── helpers ──────────────────────────────────────────────────────────
async def _open_url_ok(bc, broker, bs, *, href, user_texts):
    """Run an open_url with a fake poller that pulls the command and posts
    an ok observed result — the full happy path. Returns (cmd, verdict)."""
    task = asyncio.create_task(
        bc.send(bs.conversation_id, "open_url", href=href, user_texts=user_texts)
    )
    cmd = await broker.next(str(bs.id), wait_s=2.0)
    assert cmd is not None
    await broker.resolve(
        cmd.command_id,
        BridgeCommandResult(
            command_id=cmd.command_id,
            result_code=ResultCode.ok,
            observed={"http_status": 200, "settled": True},
        ),
    )
    return cmd, await task


# ── BridgeClient.open_url: success (grant + retarget + enqueue) ──────
def test_open_url_anchored_grants_retargets_and_enqueues(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=2.0)
    bc = BridgeClient(session, broker=broker)

    cmd, verdict = asyncio.run(
        _open_url_ok(
            bc,
            broker,
            bs,
            href="https://www.bbc.co.uk/news",
            user_texts=["please go to bbc.co.uk"],
        )
    )
    # The command reached the bridge with the open_url action + full href.
    assert cmd.action_type == BrowserActionType.open_url
    assert cmd.href == "https://www.bbc.co.uk/news"
    assert cmd.domain == "www.bbc.co.uk"
    assert verdict.result_code == ResultCode.ok
    assert verdict.action_type == BrowserActionType.open_url

    # Grant created for the new host — stored as the PSL-registrable host
    # (the same function Electron's tab approval uses, so `www.` reduces to
    # the registrable site); the prior grant revoked (single active
    # domain); active_domain moved; no reapproval needed after a completed
    # command (both sides saw the retarget).
    grants = {g.domain: g for g in _grants(session, bs)}
    assert grants["bbc.co.uk"].decision == PermissionDecision.granted.value
    assert grants["example.com"].decision == PermissionDecision.revoked.value
    assert grants["example.com"].expires_at is not None
    session.refresh(bs)
    assert bs.active_domain == "bbc.co.uk"
    assert bs.requires_reapproval is False

    # The persisted action row records action_type open_url.
    rows = session.exec(
        select(BrowserAction).where(BrowserAction.session_id == bs.id)
    ).all()
    assert [r.action_type for r in rows] == [BrowserActionType.open_url.value]


# ── HIGH 1 regressions: retarget is atomic with the command handoff ──
def test_open_url_busy_session_does_not_mutate_grants(session):
    # Another action in flight (the per-session lock is held) → open_url
    # refuses busy BEFORE the retarget: grants and active_domain unchanged,
    # nothing enqueued. Server policy state cannot diverge from Electron
    # on a refused call.
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.5)
    bc = BridgeClient(session, broker=broker)

    async def go():
        lock = _session_locks[str(bs.id)]
        async with lock:  # simulate an in-flight command
            return await bc.send(
                bs.conversation_id,
                "open_url",
                href="https://bbc.co.uk",
                user_texts=["go to bbc.co.uk"],
            )

    verdict = asyncio.run(go())
    assert verdict.result_code == ResultCode.permission_denied
    assert "in flight" in (verdict.detail or "")
    assert broker.pending_count(str(bs.id)) == 0
    grants = {g.domain: g for g in _grants(session, bs)}
    assert set(grants) == {"example.com"}
    assert grants["example.com"].decision == PermissionDecision.granted.value
    session.refresh(bs)
    assert bs.active_domain == "example.com"
    assert bs.requires_reapproval is False


def test_open_url_timeout_forces_reapproval(session):
    # The command timed out before any poller pulled it: the server has
    # retargeted but Electron never saw the command — the documented
    # recovery is forcing `requires_reapproval`, so NOTHING dispatches
    # until a fresh tab approval re-syncs both sides.
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(
            bs.conversation_id,
            "open_url",
            href="https://bbc.co.uk",
            user_texts=["go to bbc.co.uk"],
        )
    )
    assert verdict.result_code == ResultCode.timeout
    session.refresh(bs)
    assert bs.requires_reapproval is True
    # No divergent policy window: every subsequent action is refused until
    # a fresh approval, including same-site-on-new-domain follow_link.
    blocked = asyncio.run(
        bc.send(bs.conversation_id, "follow_link", href="https://bbc.co.uk/x")
    )
    assert blocked.result_code == ResultCode.unapproved_tab
    assert "re-approval" in (blocked.detail or "")
    assert broker.pending_count(str(bs.id)) == 0


def test_open_url_blocked_session_creates_no_grant(session):
    # A stopped session refuses open_url via the control gate WITHOUT
    # retargeting — a blocked session must not accumulate grants.
    from cowork.services.browser.control import BrowserControlService

    bs = _make_session(session, domain="example.com")
    BrowserControlService(session).stop(bs.id)
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(
            bs.conversation_id,
            "open_url",
            href="https://bbc.co.uk",
            user_texts=["go to bbc.co.uk"],
        )
    )
    assert verdict.result_code == ResultCode.permission_denied
    assert broker.pending_count(str(bs.id)) == 0
    assert [g.domain for g in _grants(session, bs)] == ["example.com"]


# ── scenario 5 regression: follow_link cross-domain stays blocked ────
def test_follow_link_cross_domain_still_denied_after_open_url_retarget(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=2.0)
    bc = BridgeClient(session, broker=broker)

    async def go():
        await _open_url_ok(
            bc, broker, bs, href="https://bbc.co.uk", user_texts=["go to bbc.co.uk"]
        )
        # Now: follow_link back to the OLD domain must be denied (its grant
        # was revoked by the retarget)...
        old = await bc.send(
            bs.conversation_id, "follow_link", href="https://example.com/x"
        )
        # ...and follow_link to any other cross-domain host stays denied.
        other = await bc.send(
            bs.conversation_id, "follow_link", href="https://evil.com/x"
        )
        return old, other

    old, other = asyncio.run(go())
    assert old.result_code == ResultCode.unapproved_tab
    assert other.result_code == ResultCode.unapproved_tab
    assert broker.pending_count(str(bs.id)) == 0


def test_follow_link_same_site_as_new_domain_allowed_after_retarget(session):
    # After a COMPLETED user-directed retarget, same-site follow_link on
    # the NEW domain passes the grant (reaches the broker; times out with
    # no poller for the second command).
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.5)
    bc = BridgeClient(session, broker=broker)

    async def go():
        await _open_url_ok(
            bc, broker, bs, href="https://bbc.co.uk", user_texts=["go to bbc.co.uk"]
        )
        return await bc.send(
            bs.conversation_id, "follow_link", href="https://bbc.co.uk/news"
        )

    verdict = asyncio.run(go())
    assert verdict.result_code == ResultCode.timeout  # passed the grant


# ── re-approving the old domain re-enables it ────────────────────────
def test_reapproving_old_domain_after_retarget_reenables_it(session):
    from cowork.services.browser.approval import BrowserApprovalService

    bs = _make_session(session, domain="example.com")
    svc = BrowserApprovalService(session)
    svc.retarget_domain(bs.id, "bbc.co.uk")
    grants = {g.domain: g for g in _grants(session, bs)}
    assert grants["example.com"].decision == PermissionDecision.revoked.value
    # A fresh approval of the old domain refreshes the revoked grant.
    svc.grant_domain(bs.id, "example.com")
    grants = {g.domain: g for g in _grants(session, bs)}
    assert grants["example.com"].decision == PermissionDecision.granted.value
    assert grants["example.com"].expires_at is None
