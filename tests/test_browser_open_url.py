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
from cowork.models.browser import BrowserSession, BrowserTabGrant
from cowork.models.conversation import Conversation
from cowork.schemas.browser import (
    BrowserActionClass,
    BrowserActionType,
    PermissionDecision,
    ResultCode,
)
from cowork.services.browser.bridge import BridgeCommandService
from cowork.services.browser.client import (
    OPEN_URL_NOT_ANCHORED_DETAIL,
    BridgeClient,
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
        # www.bbc.co.uk/news → registrable host bbc.co.uk, which the user's
        # words contain even when they typed the full URL.
        assert url_user_anchored(
            "https://www.bbc.co.uk/news/live", ["please open https://bbc.co.uk"]
        )
        # And the reverse: user typed www., target is the bare host — the
        # needle bbc.co.uk is a substring of www.bbc.co.uk.
        assert url_user_anchored(
            "https://bbc.co.uk", ["go to www.bbc.co.uk for me"]
        )

    def test_host_absent(self):
        assert not url_user_anchored(
            "https://evil.com/x", ["go to bbc.co.uk", "what's the weather?"]
        )

    def test_empty_inputs_never_match(self):
        assert not url_user_anchored("", ["bbc.co.uk"])
        assert not url_user_anchored("https://bbc.co.uk", [])
        assert not url_user_anchored("https://bbc.co.uk", ["", None])  # type: ignore[list-item]

    def test_subdomain_target_anchored_by_registrable_host(self):
        # news.bbc.co.uk reduces to bbc.co.uk — the user naming the site
        # covers its subdomains (same registrable site).
        assert url_user_anchored(
            "https://news.bbc.co.uk/today", ["go to bbc.co.uk"]
        )


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


def test_open_url_anchored_only_in_assistant_text_denied(session):
    # The guard input is the USER texts — the caller must not pass assistant
    # text, and when it doesn't, an assistant-only mention denies.
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(
            bs.conversation_id,
            "open_url",
            href="https://evil.com",
            user_texts=["hi"],
        )
    )
    assert verdict.result_code == ResultCode.permission_denied
    assert broker.pending_count(str(bs.id)) == 0
    assert [g.domain for g in _grants(session, bs)] == ["example.com"]


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


# ── BridgeClient.open_url: success (grant + retarget + enqueue) ──────
def test_open_url_anchored_grants_retargets_and_enqueues(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.5)
    bc = BridgeClient(session, broker=broker)

    async def go():
        # No poller: capture the enqueued command via broker.next, then let
        # the producer time out (dispatch reached the bridge — that's what
        # this test asserts; the result path is covered elsewhere).
        task = asyncio.create_task(
            bc.send(
                bs.conversation_id,
                "open_url",
                href="https://www.bbc.co.uk/news",
                user_texts=["please go to bbc.co.uk"],
            )
        )
        cmd = await broker.next(str(bs.id), wait_s=2.0)
        verdict = await task
        return cmd, verdict

    cmd, verdict = asyncio.run(go())
    # The command reached the bridge with the open_url action + full href.
    assert cmd is not None
    assert cmd.action_type == BrowserActionType.open_url
    assert cmd.href == "https://www.bbc.co.uk/news"
    assert cmd.domain == "www.bbc.co.uk"
    # (no poller posted a result, so the producer timed out — NOT a
    # permission failure: the grant existed at dispatch time)
    assert verdict.result_code == ResultCode.timeout
    assert verdict.action_type == BrowserActionType.open_url

    # Grant created for the new host — stored as the PSL-registrable host
    # (the same function Electron's tab approval uses, so `www.` reduces to
    # the registrable site); the prior grant revoked (single active
    # domain); active_domain moved.
    grants = {g.domain: g for g in _grants(session, bs)}
    assert grants["bbc.co.uk"].decision == PermissionDecision.granted.value
    assert grants["example.com"].decision == PermissionDecision.revoked.value
    assert grants["example.com"].expires_at is not None
    session.refresh(bs)
    assert bs.active_domain == "bbc.co.uk"

    # The persisted action row records action_type open_url.
    from cowork.models.browser import BrowserAction

    rows = session.exec(
        select(BrowserAction).where(BrowserAction.session_id == bs.id)
    ).all()
    assert [r.action_type for r in rows] == [BrowserActionType.open_url.value]


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
    broker = BridgeCommandService(default_timeout_s=0.5)
    bc = BridgeClient(session, broker=broker)

    async def go():
        task = asyncio.create_task(
            bc.send(
                bs.conversation_id,
                "open_url",
                href="https://bbc.co.uk",
                user_texts=["go to bbc.co.uk"],
            )
        )
        await broker.next(str(bs.id), wait_s=2.0)  # drain the open_url cmd
        await task  # times out (no poller result) — retarget already done
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
    # After the user-directed retarget, same-site follow_link on the NEW
    # domain passes the grant (reaches the broker; times out with no poller).
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.5)
    bc = BridgeClient(session, broker=broker)

    async def go():
        task = asyncio.create_task(
            bc.send(
                bs.conversation_id,
                "open_url",
                href="https://bbc.co.uk",
                user_texts=["go to bbc.co.uk"],
            )
        )
        await broker.next(str(bs.id), wait_s=2.0)
        await task
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
