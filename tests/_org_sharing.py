"""Sharing test doubles: the people in a test org, a member's request,
an artifact's rule as MindsHub stores it, and the MindsHub reads an access
check makes, stubbed at get_mindshub_reply."""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from cowork.services import mindshub_people, mindshub_teams
from cowork.services.artifact_access import ArtifactRule
from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

ALICE = "a1111111-1111-4111-8111-111111111111"
BOB = "b2222222-2222-4222-8222-222222222222"
CAROL = "c3333333-3333-4333-8333-333333333333"
DAVE = "d4444444-4444-4444-8444-444444444444"
ADMIN = "e5555555-5555-4555-8555-555555555555"
EVE = "f6666666-6666-4666-8666-666666666666"  # not in the org
MAL = "f7777777-7777-4777-8777-777777777777"  # not in the org
EMAILS = {ALICE: "alice@x.com", BOB: "bob@x.com", CAROL: "carol@x.com", DAVE: "dave@x.com", ADMIN: "admin@x.com"}
# MindsHub keys every team by UUID.
TEAM_ENG = "7ea30000-0000-4000-8000-000000000e01"
TEAM_OPS = "7ea30000-0000-4000-8000-000000000e02"
TEAM_QA = "7ea30000-0000-4000-8000-000000000e03"
# The org's default team, whose members are everyone in it.
TEAM_ALL = "7ea30000-0000-4000-8000-000000000a11"
# A team of another organization.
TEAM_ELSEWHERE = "7ea30000-0000-4000-8000-0000000000f0"

# The credential the gateway validated, in a member's Authorization header.
MEMBER_BEARER = "member-session"


def member_request(principal=None, *, bearer: str = MEMBER_BEARER) -> SimpleNamespace:
    """A member's request to a sharing route: the Authorization the gateway
    validated, carrying ``bearer``, and a client-set X-MindsHub-Authorization
    from another org, which sharing never reads."""
    return SimpleNamespace(
        headers={"Authorization": f"Bearer {bearer}", "X-MindsHub-Authorization": "Bearer other-org-credential"},
        state=SimpleNamespace(principal=principal),
    )


def artifact_rule(*, org=False, teams=(), people=(), emails=()) -> ArtifactRule:
    """An artifact's rule as MindsHub stores it."""
    return ArtifactRule(
        org_allowed=org, allowed_team_ids=list(teams), allowed_user_ids=list(people), allowed_emails=list(emails),
    )


def auth_replies(read):
    """``read``, a fake MindsHub that answers a path with its JSON body, or
    None when it couldn't be asked, as get_mindshub_reply answers."""

    async def reply(path, bearer):
        payload = await read(path, bearer)
        if payload is None:
            return MindsHubReply(MindsHubAnswer.UNREACHABLE)
        return MindsHubReply(MindsHubAnswer.OK, payload=payload, status_code=200)

    return reply


def mindshub_answers(status_code: int):
    """A fake MindsHub that answers every read with ``status_code`` and no
    body, as get_mindshub_reply answers a 4xx or 5xx."""
    answer = (
        MindsHubAnswer.THROTTLED if status_code == 429
        else MindsHubAnswer.REFUSED if 400 <= status_code < 500
        else MindsHubAnswer.UNREACHABLE
    )

    async def reply(path, bearer):
        return MindsHubReply(answer, status_code=status_code)

    return reply


def mindshub_sharing_answers(
    monkeypatch, *, teams=(TEAM_ENG,), default=None, permissions=(), members=tuple(EMAILS), members_down=False,
    asked=None, bearer=MEMBER_BEARER, org_teams=None, org_teams_down=False,
):
    """MindsHub's reads for an access check: the caller's teams
    (``default`` adds the org's default team) and permissions, every team of
    the organization (``org_teams``, else the caller's own teams; the default
    one either way), and which of some ids are org ``members`` (every test
    user by default; the whole list for a read that names no ids). None for
    ``teams`` or ``permissions``, ``org_teams_down`` or ``members_down`` is a
    read that fails. Each read is recorded in ``asked`` as ``(path,
    bearer)``, and must carry ``bearer``."""
    mindshub_teams.reset_caches_for_tests()

    def record(path, sent):
        if asked is not None:
            asked.append((path, sent))
        assert sent == bearer, f"{path} was read with {sent!r}, not {bearer!r}"

    def team_rows(ids):
        return [{"id": t} for t in ids] + ([{"id": default, "is_default": True}] if default else [])

    async def teams_read(path, sent):
        record(path, sent)
        if path == mindshub_teams.TEAMS_PATH:
            return None if teams is None else {"results": team_rows(teams or ())}
        if path == mindshub_teams.ORG_TEAMS_PATH:
            if org_teams_down:
                return None
            return {"results": team_rows(org_teams if org_teams is not None else teams or ())}
        if path == mindshub_teams.ENTITLEMENTS_PATH:
            return None if permissions is None else {"permissions": list(permissions)}
        raise AssertionError(path)

    async def members_read(path, sent):
        record(path, sent)
        if members_down:
            return None
        named = parse_qs(urlsplit(path).query).get("ids")
        ids = named[0].split(",") if named else list(members)
        return {"results": [{"id": i} for i in ids if i in members]}

    monkeypatch.setattr(mindshub_teams, "get_mindshub_reply", auth_replies(teams_read))
    monkeypatch.setattr(mindshub_people, "get_mindshub_reply", auth_replies(members_read))
