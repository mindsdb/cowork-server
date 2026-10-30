"""Org sharing test doubles (ENG-3787): the people in a test org, a
member's request, an artifact's rule as the auth service stores it, and the
auth service's reads an access check makes, stubbed at get_auth_json."""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from cowork.services import hub_people, hub_teams
from cowork.services.artifact_access import ArtifactRule

ALICE = "a1111111-1111-4111-8111-111111111111"
BOB = "b2222222-2222-4222-8222-222222222222"
CAROL = "c3333333-3333-4333-8333-333333333333"
DAVE = "d4444444-4444-4444-8444-444444444444"
ADMIN = "e5555555-5555-4555-8555-555555555555"
EVE = "f6666666-6666-4666-8666-666666666666"  # not in the org
MAL = "f7777777-7777-4777-8777-777777777777"  # not in the org
EMAILS = {ALICE: "alice@x.com", BOB: "bob@x.com", CAROL: "carol@x.com", DAVE: "dave@x.com", ADMIN: "admin@x.com"}
TEAM_ENG = "team-eng"
TEAM_OPS = "team-ops"

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
    """An artifact's rule as the auth service stores it."""
    return ArtifactRule(
        org_allowed=org, allowed_team_ids=list(teams), allowed_user_ids=list(people), allowed_emails=list(emails),
    )


def access_service_answers(
    monkeypatch, *, teams=(TEAM_ENG,), default=None, permissions=(), members=tuple(EMAILS), members_down=False,
    asked=None, bearer=MEMBER_BEARER,
):
    """The auth service's reads for an access check: the caller's teams
    (``default`` adds the org's default team) and permissions, and which of
    some ids are org ``members`` (every test user by default; the whole list
    for a read that names no ids). None for ``teams`` or ``permissions``, or
    ``members_down``, is a read that fails. Each read is recorded in
    ``asked`` as ``(path, bearer)``, and must carry ``bearer``."""
    hub_teams.reset_caches_for_tests()

    def record(path, sent):
        if asked is not None:
            asked.append((path, sent))
        assert sent == bearer, f"{path} was read with {sent!r}, not {bearer!r}"

    async def teams_read(path, sent):
        record(path, sent)
        if path == hub_teams.TEAMS_PATH:
            if teams is None:
                return None
            rows = [{"id": t} for t in teams] + ([{"id": default, "is_default": True}] if default else [])
            return {"results": rows}
        if path == hub_teams.ENTITLEMENTS_PATH:
            return None if permissions is None else {"permissions": list(permissions)}
        raise AssertionError(path)

    async def members_read(path, sent):
        record(path, sent)
        if members_down:
            return None
        named = parse_qs(urlsplit(path).query).get("ids")
        ids = named[0].split(",") if named else list(members)
        return {"results": [{"id": i} for i in ids if i in members]}

    monkeypatch.setattr(hub_teams, "get_auth_json", teams_read)
    monkeypatch.setattr(hub_people, "get_auth_json", members_read)
