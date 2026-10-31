"""Owner-side share controls for org (Cloud) artifacts — ENG-2316.

Handlers are called directly, which is this suite's convention for the
workspace routes (the identity gate has its own over-HTTP file).

What matters here is the boundary: setting an audience is a re-publish, so it
must go through the org-mode publish path autopublish already proves works, and
it must be refused for anyone but the owner. The GET exists because the artifact
CARD deliberately withholds `accessEmails`/`accessPassword` in org mode — one
artifacts root is shared by the whole org, so a card cannot tell owner from
co-member. This route can.
"""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

# Imported before any fixture patches publish_artifact: artifact_autopublish
# binds publish_artifact by name when it is first imported, and
# publish_context's patch of it by path would otherwise first import it while
# publish_calls holds the fake, for every later test in the run.
import cowork.services.artifact_autopublish  # noqa: F401
from cowork.api.v1.endpoints import artifact_workspace as aw
from cowork.db.scoped import TenantScope, get_scoped_session
from cowork.schemas.artifact_access import ArtifactAccessRequest
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.services import artifact_locks
from tests._org_sharing import (
    BOB,
    CAROL,
    DAVE,
    EVE,
    MEMBER_BEARER,
    TEAM_ELSEWHERE,
    TEAM_ENG,
    TEAM_OPS,
    mindshub_answers,
    mindshub_sharing_answers,
    artifact_rule,
    member_request,
)

# The gateway forwards every org and person id as a UUID.
ORG_SCOPE = TenantScope(
    org_mode=True, org_id="0a0a0a0a-0000-4000-8000-000000000001", user_id="0b0b0b0b-0000-4000-8000-000000000001"
)
ARTIFACT_ID = "11111111-1111-1111-1111-111111111111"


class _Session:
    scope = ORG_SCOPE


# The route's request: the caller's credential, read only to check teams.
_REQUEST = SimpleNamespace(headers={}, state=SimpleNamespace(principal=None))
_MEMBER_REQUEST = member_request()


@pytest.fixture
def folder(tmp_path):
    base = tmp_path / ORG_SCOPE.org_id / "proj" / ".anton" / "artifacts"
    f = base / "rep"
    f.mkdir(parents=True)
    (f / "report.html").write_text("<html></html>")
    (f / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    return f


@pytest.fixture
def as_owner(monkeypatch, folder):
    """Owner resolution succeeds and yields this folder."""
    monkeypatch.setattr(
        aw, "_owner_workspace",
        lambda session, project_ref, artifact_id: (
            SimpleNamespace(project_id="proj-1"), folder, {"slug": "rep"}, {},
        ),
    )
    return folder


@pytest.fixture
def publish_calls(monkeypatch):
    calls = []

    def fake_publish(artifact, *, artifacts_base, api_key, publish_url,
                     password=None, access=None, scope=None, project_id=None,
                     notify=None, stored=None, owner_save=False):
        # publish_artifact takes the model; what it hands anton is its
        # publisher_access.
        assert isinstance(access, ArtifactAccessRequest)
        calls.append({
            "folder": artifact, "artifacts_base": artifacts_base, "api_key": api_key,
            "publish_url": publish_url, "access": access.publisher_access(), "scope": scope,
            "project_id": project_id, "notify": notify, "stored": stored,
            "owner_save": owner_save,
        })
        return {"status": "ok", "url": "https://view.example/r/1"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fake_publish)
    return calls


@pytest.fixture
def publish_context(monkeypatch):
    """Org publish endpoint and a turn key; yields the list of revoke calls."""
    monkeypatch.setattr(
        "cowork.services.artifact_autopublish._publish_url",
        lambda scope: "https://api.staging.mindshub.ai",
    )
    revoked = []

    class FakeKey:
        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return "turnkey-1"

        async def revoke(self):
            revoked.append(True)

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey", FakeKey)
    return revoked


pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("granted_product_permissions")]


def _upload_service_answers(monkeypatch, status: int, reason: str, body: bytes) -> None:
    """The real publish path runs, and the upload service answers `status`."""
    import io
    from urllib.error import HTTPError

    def reject(*args, **kwargs):
        raise HTTPError("https://publish.example/upload", status, reason, {}, io.BytesIO(body))

    monkeypatch.setattr("anton.publisher.publish", reject)
    monkeypatch.setattr("cowork.services.publish.vault_for_scope", lambda scope: object())
    monkeypatch.setattr(
        "cowork.services.artifact_authorization_identity.publish_authorization_key",
        lambda *args: "owner/artifact",
    )


def _lock_is_free(folder) -> bool:
    free = artifact_locks.acquire(folder.parent, folder.name, ttl_s=60)
    if free:
        artifact_locks.release(folder.parent, folder.name)
    return free


async def test_setting_access_republishes_with_the_chosen_audience(
    as_owner, publish_calls, publish_context,
):
    access = {"mode": "restricted", "allowed_emails": ["a@example.com"], "org_allowed": False}

    out = await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access=access), _Session(), _REQUEST,
    )

    assert out["url"] == "https://view.example/r/1"
    # MindsHub's allowed_emails reaches anton's publisher as its emails.
    assert publish_calls[0]["access"] == {
        "mode": "restricted", "emails": ["a@example.com"], "org_allowed": False,
    }
    # The org-mode credential and endpoint, not the desktop provider settings.
    assert publish_calls[0]["api_key"] == "turnkey-1"
    assert publish_calls[0]["publish_url"] == "https://api.staging.mindshub.ai"
    # `vault_for_scope` RAISES on an org deployment without a scope, so dropping
    # this kwarg would fail every share rather than read the wrong vault.
    assert publish_calls[0]["scope"] is ORG_SCOPE
    assert publish_calls[0]["artifacts_base"] == as_owner.parent
    assert publish_calls[0]["project_id"] == "proj-1"


async def test_a_save_that_chose_notify_passes_the_people_apart_from_the_access(
    as_owner, publish_calls, publish_context,
):
    """The app sends ``notify: {user_ids, emails}`` inside the
    access object. It is this save's alone, so it is lifted out and passed on
    its own, never left in the access that gets stored and re-published."""
    access = {"mode": "restricted", "allowed_emails": ["a@example.com"], "org_allowed": False}
    published = {"mode": "restricted", "emails": ["a@example.com"], "org_allowed": False}
    person = "AAAAAAAA-1111-1111-1111-111111111111"

    await aw.set_artifact_access(
        "proj", ARTIFACT_ID,
        aw._AccessBody(access={**access, "notify": {"user_ids": [person], "emails": [" A@Example.com "]}}),
        _Session(), _REQUEST,
    )
    notify = publish_calls[-1]["notify"]
    assert (notify.user_ids, notify.emails) == ((person.lower(),), ("a@example.com",))
    assert publish_calls[-1]["access"] == published

    # Absent, or empty: nobody to notify.
    for extra in ({}, {"notify": {}}, {"notify": {"user_ids": [], "emails": []}}):
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={**access, **extra}), _Session(), _REQUEST,
        )
        assert publish_calls[-1]["notify"] is None
        assert publish_calls[-1]["access"] == published


async def _put_access(body) -> httpx.Response:
    """PUT .../access over HTTP, through the route as the app mounts it, so
    FastAPI reads the body: the session is the org scope's, and the owner and
    the publish are the tests' fixtures."""
    app = FastAPI()
    app.include_router(aw.router)
    app.dependency_overrides[get_scoped_session] = _Session
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.put(f"/workspace/proj/{ARTIFACT_ID}/access", json=body)


async def test_a_top_level_emails_is_a_400(as_owner, publish_calls, publish_context):
    """The app sends MindsHub's four names (org_allowed, allowed_team_ids,
    allowed_user_ids, allowed_emails). A top-level emails, which an app tab
    loaded from an older build sends, is refused rather than dropped, so
    it can't shrink an audience unnoticed."""
    answer = await _put_access({"access": {"mode": "restricted", "emails": []}})
    assert (answer.status_code, answer.json()) == (
        400, {"detail": "emails is not an access field; send allowed_emails."},
    )
    assert publish_calls == []


async def test_a_malformed_save_is_a_400_with_a_sentence_never_a_422(as_owner, publish_calls, publish_context):
    """The Share dialog shows a refusal's detail as it is, so a save FastAPI
    can't read answers 400 with one sentence naming what is wrong, never
    FastAPI's 422 with a list of errors. Fields are named from inside the
    access object, as the dialog sends them."""
    refusals = [
        ({"access": {"mode": "restricted", "allowed_user_ids": "nope"}}, "allowed_user_ids must be a list."),
        ({"access": {"mode": "everyone"}}, "mode: Input should be 'restricted', 'password' or 'public'."),
        ({"access": {"mode": "restricted", "org_allowed": "maybe"}}, "org_allowed must be true or false."),
        ({"access": {"mode": "restricted", "notify": "everyone"}}, "notify must be an object."),
        ({"access": {"mode": "restricted", "notify": {"emails": "a@example.com"}}}, "notify.emails must be a list."),
        ({"access": "public"}, "access must be an object."),
        ({}, "access: Field required."),
        (["public"], aw.ACCESS_BODY_REFUSAL),
    ]
    for body, detail in refusals:
        answer = await _put_access(body)
        assert (answer.status_code, answer.json()) == (400, {"detail": detail}), body
    assert publish_calls == []


async def test_a_save_ignores_fields_it_does_not_know(as_owner, publish_calls, publish_context):
    """A field the save doesn't define, inside access or beside it, is
    ignored rather than refused: team_ids and org_wide among them."""
    answer = await _put_access({
        "access": {"mode": "restricted", "allowed_emails": ["a@example.com"], "team_ids": ["t"], "org_wide": True},
        "extra": 1,
    })
    assert answer.status_code == 200, answer.text
    assert publish_calls[-1]["access"] == {"mode": "restricted", "emails": ["a@example.com"]}


@pytest.fixture
def stored_rule(monkeypatch):
    """The artifact's rule as MindsHub stores it, which the route
    reads with cowork.services.publish.stored_artifact_rule: nobody but the
    owner until a test sets ``rule``. Each read is counted in ``reads``.
    For an entry that isn't restricted the route reads
    cowork.services.publish.rule_held_over_entry instead, which answers
    ``held`` (no rule until a test sets one), counted in ``held_reads``."""
    from cowork.services.artifact_access import ArtifactRule

    state = SimpleNamespace(rule=ArtifactRule.nobody(), reads=[], held=None, held_reads=[])

    def read(artifact, *, artifacts_base, scope, project_id):
        state.reads.append((artifact, artifacts_base, scope, project_id))
        if isinstance(state.rule, Exception):
            raise state.rule
        return state.rule

    def read_held(artifact, *, artifacts_base, scope, project_id):
        state.held_reads.append((artifact, artifacts_base, scope, project_id))
        if isinstance(state.held, Exception):
            raise state.held
        return state.held

    monkeypatch.setattr("cowork.services.publish.stored_artifact_rule", read)
    monkeypatch.setattr("cowork.services.publish.rule_held_over_entry", read_held)
    return state


async def test_sharing_with_everyone_takes_share_org_before_anything_is_published(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """"Everyone at the org" on an artifact takes artifacts.share_org, read
    with the gateway's credential. A member without it, the skill capability
    alone included, gets a 403 and nothing is published."""
    asked: list = []
    mindshub_sharing_answers(monkeypatch, asked=asked)
    everyone = {"mode": "restricted", "org_allowed": True, "allowed_emails": []}
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=everyone), _Session(), _MEMBER_REQUEST)
    assert (caught.value.status_code, caught.value.detail) == (403, aw.ARTIFACT_ORG_REFUSAL)
    assert publish_calls == []
    assert {bearer for _path, bearer in asked} == {MEMBER_BEARER}
    # Read for the owner's own artifact, in the publish's artifacts root.
    assert stored_rule.reads == [(as_owner, as_owner.parent, ORG_SCOPE, "proj-1")]
    assert _lock_is_free(as_owner)

    mindshub_sharing_answers(monkeypatch, permissions=("skills.share_org",))
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=everyone), _Session(), _MEMBER_REQUEST)
    assert (caught.value.status_code, caught.value.detail) == (403, aw.ARTIFACT_ORG_REFUSAL)
    assert publish_calls == []

    mindshub_sharing_answers(monkeypatch, permissions=("artifacts.share_org",))
    await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=everyone), _Session(), _MEMBER_REQUEST)
    assert publish_calls[-1]["access"]["org_allowed"] is True


def _planted(folder, **entry) -> None:
    """What an agent in the pod can write: the artifact folder is mounted
    writable, .published.json included."""
    (folder / ".published.json").write_text(json.dumps({"report.html": {
        "report_id": "r1", "url": "https://view.example/r/1", "mode": "restricted", "emails": [],
        "org_allowed": False, "owner_only": False, "access_version": 2, **entry,
    }}))


async def test_an_entry_an_agent_planted_never_lets_a_member_share_with_everyone(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """.published.json says the rule is shared with everyone already, but an
    agent can write that file, and the rule MindsHub stores says
    otherwise. So a member without skills.share_org who asks for everyone
    still gets the 403, and nothing is published."""
    _planted(as_owner, org_allowed=True)
    mindshub_sharing_answers(monkeypatch)
    everyone = {"mode": "restricted", "org_allowed": True, "allowed_emails": []}
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=everyone), _Session(), _MEMBER_REQUEST)
    assert (caught.value.status_code, caught.value.detail) == (403, aw.ARTIFACT_ORG_REFUSAL)
    assert publish_calls == []


async def test_a_member_keeps_an_artifact_already_shared_with_everyone(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """Only sharing an artifact with everyone takes artifacts.share_org. A member who
    owns one the rule already shares with everyone saves it again, say to
    add an address, without it."""
    stored_rule.rule = artifact_rule(org=True)
    mindshub_sharing_answers(monkeypatch)
    access = {"mode": "restricted", "org_allowed": True, "allowed_emails": ["e@else.com"]}
    await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=access), _Session(), _MEMBER_REQUEST)
    assert publish_calls[-1]["access"] == {"mode": "restricted", "org_allowed": True, "emails": ["e@else.com"]}


async def test_an_admins_team_from_outside_the_org_is_a_400_before_anything_is_published(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """An admin may name any team of the organization, so a team they aren't
    in is checked against the organization's teams, read now, before the
    publish: MindsHub refuses a team from elsewhere only after the lambda
    has made the save's new emails live. While those teams can't be read the
    save is a 503. Nothing is published either way, and a team of the
    organization the admin isn't in is published."""
    from cowork.services.artifact_access import FOREIGN_TEAM_REFUSAL

    admin = member_request(Principal(
        user_id=ORG_SCOPE.user_id, org_id=ORG_SCOPE.org_id, email="admin@x.com",
        roles=frozenset({ORG_MANAGE_ROLE}),
    ))
    save = {"mode": "restricted", "allowed_emails": ["new@else.com"], "allowed_team_ids": [TEAM_ELSEWHERE]}
    mindshub_sharing_answers(monkeypatch, teams=(TEAM_ENG,), org_teams=(TEAM_ENG, TEAM_OPS))
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=save), _Session(), admin)
    assert (caught.value.status_code, caught.value.detail) == (400, f"{FOREIGN_TEAM_REFUSAL}: {TEAM_ELSEWHERE}")

    mindshub_sharing_answers(monkeypatch, teams=(TEAM_ENG,), org_teams_down=True)
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=save), _Session(), admin)
    assert caught.value.status_code == 503
    assert publish_calls == []
    assert _lock_is_free(as_owner)

    mindshub_sharing_answers(monkeypatch, teams=(TEAM_ENG,), org_teams=(TEAM_ENG, TEAM_OPS))
    ops = {**save, "allowed_team_ids": [TEAM_OPS]}
    await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=ops), _Session(), admin)
    assert publish_calls[-1]["access"]["allowed_team_ids"] == [TEAM_OPS]


def _cache_the_callers_teams(*team_ids: str) -> None:
    """The caller's teams as the per-caller cache holds them from a read
    MindsHub answered a moment ago."""
    from cowork.services import mindshub_teams
    from cowork.services.mindshub_client import CallerCacheKey

    key = CallerCacheKey.for_caller(org_id=ORG_SCOPE.org_id, user_id=ORG_SCOPE.user_id, bearer_token=MEMBER_BEARER)
    listing = mindshub_teams.MindsHubTeamListing(
        teams=[mindshub_teams.MindsHubTeam(id=team_id) for team_id in team_ids], reachable=True,
    )
    mindshub_teams._teams_cache.put(key, listing)


async def test_a_team_deleted_since_the_cached_read_is_refused_before_anything_is_published(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """The caller's teams are cached for a minute, so a team deleted within
    it still reads as theirs there. A save that adds a team reads the
    caller's teams from MindsHub now, so the deleted team is the 400 a team
    they aren't in gets, before anything is published, never MindsHub's
    unknown_team after the upload has made the save's emails live. The
    cache keeps that answer. A team MindsHub still lists is published."""
    from cowork.services import mindshub_teams

    asked: list = []
    mindshub_sharing_answers(monkeypatch, teams=(TEAM_ENG,), asked=asked)
    _cache_the_callers_teams(TEAM_ENG, TEAM_OPS)
    save = {"mode": "restricted", "allowed_emails": ["new@else.com"], "allowed_team_ids": [TEAM_OPS]}

    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=save), _Session(), _MEMBER_REQUEST)

    assert (caught.value.status_code, caught.value.detail) == (
        400, f"You can only share with teams you belong to: {TEAM_OPS}.",
    )
    assert publish_calls == []
    assert _lock_is_free(as_owner)
    assert (mindshub_teams.TEAMS_PATH, MEMBER_BEARER) in asked
    cached = await mindshub_teams.fetch_mindshub_teams(
        bearer_token=MEMBER_BEARER, org_id=ORG_SCOPE.org_id, user_id=ORG_SCOPE.user_id,
    )
    assert cached.team_ids == {TEAM_ENG}

    eng = {**save, "allowed_team_ids": [TEAM_ENG]}
    await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=eng), _Session(), _MEMBER_REQUEST)
    assert publish_calls[-1]["access"]["allowed_team_ids"] == [TEAM_ENG]


@pytest.mark.parametrize(("mindshub_status", "answer", "throttle_logged"), [(503, 503, 0), (429, 429, 1)])
async def test_a_save_adding_a_team_waits_while_mindshub_cant_list_the_callers_teams_now(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch, caplog,
    mindshub_status, answer, throttle_logged,
):
    """The cached teams never stand in for the read a save that adds a team
    makes: while MindsHub can't answer it the save is a 503, and while it
    throttles the caller a 429, with nothing published. The throttle is
    logged, as every throttled teams read is."""
    from cowork.services import mindshub_teams

    mindshub_sharing_answers(monkeypatch, teams=(TEAM_ENG,))
    _cache_the_callers_teams(TEAM_ENG)
    monkeypatch.setattr(mindshub_teams, "get_mindshub_reply", mindshub_answers(mindshub_status))
    save = {"mode": "restricted", "allowed_emails": [], "allowed_team_ids": [TEAM_ENG]}

    with caplog.at_level(logging.WARNING, logger="cowork.services.mindshub_teams"), \
            pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=save), _Session(), _MEMBER_REQUEST)

    assert caught.value.status_code == answer
    assert publish_calls == []
    assert _lock_is_free(as_owner)
    refused = [
        record for record in caplog.records
        if record.name == "cowork.services.mindshub_teams" and record.levelno == logging.WARNING
        and getattr(record, "event", None) == "mindshub_throttled"
        and getattr(record, "outcome", None) == "refused" and record.route == mindshub_teams.TEAMS_PATH
    ]
    assert len(refused) == throttle_logged


async def test_a_team_an_agent_planted_in_the_entry_is_checked_like_any_other(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """A team in .published.json the member isn't in, sent back beside a
    team of theirs, is refused as one they don't belong to, not kept: the
    rule MindsHub stores doesn't have it, and the entry can't vouch
    for it."""
    _planted(as_owner, owner_only=True, allowed_team_ids=[TEAM_OPS])
    mindshub_sharing_answers(monkeypatch, teams=(TEAM_ENG,))
    access = {"mode": "restricted", "allowed_team_ids": [TEAM_OPS, TEAM_ENG], "allowed_emails": []}
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=access), _Session(), _MEMBER_REQUEST)
    assert (caught.value.status_code, caught.value.detail) == (
        400, f"You can only share with teams you belong to: {TEAM_OPS}.",
    )
    assert publish_calls == []


async def test_a_person_who_left_the_org_does_not_block_adding_someone_else(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """Carol left the org. MindsHub keeps her id on the
    rule and would refuse it on the next write, and the share dialog sends
    her back. The owner adds Dave: the save goes through without Carol."""
    stored_rule.rule = artifact_rule(people=[BOB, CAROL])
    mindshub_sharing_answers(monkeypatch, members=(BOB, DAVE))
    access = {"mode": "restricted", "allowed_user_ids": [BOB, CAROL, DAVE], "allowed_emails": []}
    await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=access), _Session(), _MEMBER_REQUEST)
    assert publish_calls[-1]["access"]["allowed_user_ids"] == [BOB, DAVE]


async def test_a_save_hands_the_rule_its_check_read_to_the_publish(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """A save that names teams, people or the org flag reads the
    artifact's rule once, under the publish lock, to check it, and passes
    that rule to publish_artifact as ``stored``, so the publish plans against
    the rule the check judged without reading it again. A save the check
    reads nothing for leaves it None, for the publish to read. Either way it
    is the owner's own save (``owner_save``), never refused as a mode an
    agent changed."""
    stored_rule.rule = artifact_rule(teams=[TEAM_ENG])
    mindshub_sharing_answers(monkeypatch, teams=(TEAM_ENG,))
    team = {"mode": "restricted", "allowed_team_ids": [TEAM_ENG], "allowed_emails": []}
    await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=team), _Session(), _MEMBER_REQUEST)
    assert len(stored_rule.reads) == 1
    assert publish_calls[-1]["stored"] is stored_rule.rule
    assert publish_calls[-1]["owner_save"] is True

    for access in ({"mode": "restricted", "allowed_emails": ["e@else.com"]}, {"mode": "public"}):
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=access), _Session(), _MEMBER_REQUEST)
        assert len(stored_rule.reads) == 1
        assert (publish_calls[-1]["stored"], publish_calls[-1]["owner_save"]) == (None, True)


async def test_adding_someone_outside_the_org_is_a_400_before_anything_is_published(
    as_owner, publish_calls, publish_context, stored_rule, monkeypatch,
):
    """The people a save adds are checked before the publish, so the email
    and org part of the change never goes live beside a refusal."""
    from cowork.services.artifact_access import NOT_A_MEMBER_REFUSAL

    mindshub_sharing_answers(monkeypatch, members=(BOB,))
    access = {"mode": "restricted", "allowed_user_ids": [BOB, EVE], "allowed_emails": ["e@else.com"]}
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access("proj", ARTIFACT_ID, aw._AccessBody(access=access), _Session(), _MEMBER_REQUEST)
    assert (caught.value.status_code, caught.value.detail) == (400, f"{NOT_A_MEMBER_REFUSAL}: {EVE}")
    assert publish_calls == []


async def test_notify_is_bounded_at_mindshubs_limits_before_anything_is_published(
    as_owner, publish_calls, publish_context,
):
    """MindsHub emails at most 100 addresses and 100 people from one
    save, and refuses more. A save naming more is a 400 here, before the
    publish, rather than a share that lands with nobody told."""
    from cowork.services.sharing_contract import MAX_NOTIFY_EMAILS, MAX_PEOPLE_PER_RULE

    emails = [f"p{n}@else.com" for n in range(MAX_NOTIFY_EMAILS + 1)]
    people = [f"{n:08d}-1111-4111-8111-111111111111" for n in range(MAX_PEOPLE_PER_RULE + 1)]
    base = {"mode": "restricted", "allowed_emails": emails}
    for notify, detail in (
        ({"emails": emails}, f"notify can name at most {MAX_NOTIFY_EMAILS} email addresses."),
        ({"user_ids": people}, f"notify can name at most {MAX_PEOPLE_PER_RULE} people."),
    ):
        with pytest.raises(HTTPException) as caught:
            await aw.set_artifact_access(
                "proj", ARTIFACT_ID, aw._AccessBody(access={**base, "notify": notify}), _Session(), _REQUEST,
            )
        assert (caught.value.status_code, caught.value.detail) == (400, detail)
    assert publish_calls == []
    await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={**base, "notify": {"emails": emails[:-1]}}), _Session(), _REQUEST,
    )
    assert len(publish_calls[-1]["notify"].emails) == MAX_NOTIFY_EMAILS


async def test_mindshub_refusing_the_share_is_a_409_never_a_retryable_503(
    as_owner, publish_context, stored_rule, monkeypatch,
):
    """The artifact's recorded owner or organization doesn't match, or the
    owner's role no longer allows artifact changes: the same answer however
    often the person tries, so a 409 with the reason, on the save and on the
    read of who it is shared with."""
    from cowork.services.artifact_access import ArtifactAccessRefused

    def refuse(*_args, **_kwargs):
        raise ArtifactAccessRefused("MindsHub refused the change: Your role does not allow it.")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", refuse)
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "restricted", "allowed_emails": ["e@else.com"]}),
            _Session(), _REQUEST,
        )
    assert (caught.value.status_code, caught.value.detail) == (
        409, "MindsHub refused the change: Your role does not allow it.",
    )
    assert _lock_is_free(as_owner)

    _planted(as_owner)
    stored_rule.rule = ArtifactAccessRefused("MindsHub refused the read: no.")
    with pytest.raises(HTTPException) as caught:
        await aw.artifact_access("proj", ARTIFACT_ID, _Session())
    assert caught.value.status_code == 409


async def test_too_many_people_is_a_400_before_anything_is_published(as_owner, publish_calls, publish_context):
    """MindsHub takes at most 100 people on a rule; more is refused
    here, about the artifact, before the bundle is published."""
    people = [f"{n:08d}-1111-4111-8111-111111111111" for n in range(101)]
    with pytest.raises(HTTPException) as caught:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID,
            aw._AccessBody(access={"mode": "restricted", "allowed_user_ids": people, "allowed_emails": []}),
            _Session(), _REQUEST,
        )
    assert (caught.value.status_code, caught.value.detail) == (
        400, "An artifact can be shared with at most 100 people.",
    )
    assert publish_calls == []


async def test_sharing_publicly_is_passed_through_verbatim(
    as_owner, publish_calls, publish_context,
):
    """The publisher owns the access schema; this route must not reinterpret it."""
    await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
    )

    assert publish_calls[0]["access"] == {"mode": "public"}


async def test_audience_change_waits_for_an_inflight_live_sync(
    as_owner, publish_calls, publish_context,
):
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    task = asyncio.create_task(
        aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "restricted"}), _Session(), _REQUEST,
        )
    )

    try:
        await asyncio.sleep(0.05)
        assert publish_calls == []
    finally:
        artifact_locks.release(as_owner.parent, as_owner.name)

    await task
    assert publish_calls[0]["access"] == {"mode": "restricted"}
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    artifact_locks.release(as_owner.parent, as_owner.name)


@pytest.mark.parametrize("status", [403, 503])
async def test_share_preserves_mint_failure_after_waiting_for_publish_lock(
    as_owner, publish_calls, monkeypatch, status,
):
    from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable

    error_type = ProductPermissionDenied if status == 403 else ProductPermissionUnavailable
    admitted = asyncio.Event()

    async def allow_initial_check(scope, permission):
        admitted.set()

    async def reject_mint(**kwargs):
        raise error_type()

    monkeypatch.setattr(aw, "require_product_permission", allow_initial_check)
    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", reject_mint)
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60)
    task = asyncio.create_task(aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
    ))
    try:
        await admitted.wait()
    finally:
        artifact_locks.release(as_owner.parent, as_owner.name)
    with pytest.raises(error_type) as error:
        await task
    assert error.value.status_code == status
    assert publish_calls == []
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60)
    artifact_locks.release(as_owner.parent, as_owner.name)


@pytest.mark.parametrize("status", [403, 503])
async def test_share_preserves_authority_failure_from_the_artifact_consumer(
    as_owner, publish_context, monkeypatch, status,
):
    body = (
        {"code": "permission_denied"} if status == 403
        else {"error": "Auth unavailable", "code": "auth_unavailable"}
    )
    _upload_service_answers(monkeypatch, status, "Rejected", json.dumps(body).encode())
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert error.value.status_code == status
    assert error.value.detail["code"] == ("permission_denied" if status == 403 else "permission_unavailable")
    assert not (as_owner / ".published.json").exists()
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60)
    artifact_locks.release(as_owner.parent, as_owner.name)


async def test_a_non_owner_cannot_change_access(monkeypatch, publish_calls, publish_context):
    """`_owner_workspace` is the boundary — a reviewer holds a read grant, and
    must not be able to widen the audience with it."""
    def refuse(*_a, **_kw):
        raise HTTPException(status_code=403, detail="Not the owner")

    monkeypatch.setattr(aw, "_owner_workspace", refuse)

    with pytest.raises(HTTPException) as excinfo:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )

    assert excinfo.value.status_code == 403
    assert publish_calls == []


async def test_access_is_not_changed_when_no_publish_credential_is_available(
    as_owner, publish_calls, monkeypatch,
):
    """A share that cannot be performed must fail loudly, not report success."""
    monkeypatch.setattr(
        "cowork.services.artifact_autopublish._publish_url",
        lambda scope: "https://api.staging.mindshub.ai",
    )

    class NoKey:
        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return ""

        async def revoke(self):
            pass

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey", NoKey)

    with pytest.raises(HTTPException) as excinfo:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )

    assert excinfo.value.status_code == 503
    assert publish_calls == []


async def test_owner_reads_back_the_email_list_a_card_would_withhold(as_owner, stored_rule):
    """Pre-filling "selected users" needs the emails the org card strips. In
    MindsHub mode they, and everyone else the artifact is shared with, come from
    the rule MindsHub stores, never from .published.json."""
    (as_owner / ".published.json").write_text(json.dumps({
        "report.html": {
            "report_id": "rid", "url": "u", "published": True,
            "mode": "restricted", "emails": ["planted@else.com"],
            "org_allowed": True, "owner_only": False, "allowed_team_ids": [TEAM_ELSEWHERE],
        },
    }))
    stored_rule.rule = artifact_rule(teams=[TEAM_ENG], people=[BOB], emails=["a@example.com"])

    out = (await aw.artifact_access("proj", ARTIFACT_ID, _Session())).model_dump(by_alias=True, mode="json")

    assert out["accessMode"] == "restricted"
    assert (out["accessEmails"], out["orgAllowed"], out["ownerOnly"], out["allowedTeamIds"], out["allowedUserIds"]) == (
        ["a@example.com"], False, False, [TEAM_ENG], [BOB],
    )
    assert stored_rule.reads == [(as_owner, as_owner.parent, ORG_SCOPE, "proj-1")]


async def test_the_access_read_is_a_503_while_the_rule_cannot_be_read(as_owner, stored_rule):
    """The share dialog never shows a guess: no rule read, no audience."""
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    _planted(as_owner)
    stored_rule.rule = ArtifactAccessUnavailable("Could not read who the artifact is shared with")
    with pytest.raises(HTTPException) as caught:
        await aw.artifact_access("proj", ARTIFACT_ID, _Session())
    assert caught.value.status_code == 503
    # A public artifact MindsHub holds no rule for stays public.
    (as_owner / ".published.json").write_text(json.dumps({"report.html": {"report_id": "r", "mode": "public"}}))
    assert (await aw.artifact_access("proj", ARTIFACT_ID, _Session())).access_mode == "public"
    assert len(stored_rule.reads) == 1
    assert stored_rule.held_reads == [(as_owner, as_owner.parent, ORG_SCOPE, "proj-1")]
    # Nor a guess that it is public while the check for a rule can't be made.
    stored_rule.held = ArtifactAccessUnavailable("Could not read who the artifact is shared with")
    with pytest.raises(HTTPException) as caught:
        await aw.artifact_access("proj", ARTIFACT_ID, _Session())
    assert caught.value.status_code == 503


async def test_identity_service_failure_does_not_report_a_successful_share(
    as_owner, publish_context, monkeypatch,
):
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    def unavailable(*_args, **_kwargs):
        raise ArtifactAccessUnavailable("Could not establish artifact authorization")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", unavailable)
    with pytest.raises(HTTPException) as excinfo:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert excinfo.value.status_code == 503
    assert not (as_owner / ".published.json").exists()
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    artifact_locks.release(as_owner.parent, as_owner.name)


async def test_owner_reads_back_an_owner_only_artifact(as_owner, stored_rule):
    (as_owner / ".published.json").write_text(json.dumps({
        "report.html": {
            "report_id": "rid", "url": "u", "published": True,
            "mode": "restricted", "access_version": 1,
        },
    }))

    out = (await aw.artifact_access("proj", ARTIFACT_ID, _Session())).model_dump(by_alias=True, mode="json")

    assert out["ownerOnly"] is True
    assert out["accessEmails"] == []


async def test_share_of_a_report_owned_by_another_account_is_409(as_owner, publish_context, monkeypatch):
    """The org Share path says so instead of publishing a copy under a new URL."""
    from anton.publisher import ArtifactOwnedByOtherUserError

    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs)
        raise ArtifactOwnedByOtherUserError()

    monkeypatch.setattr("anton.publisher.publish", reject)
    monkeypatch.setattr("cowork.services.publish.vault_for_scope", lambda scope: object())
    monkeypatch.setattr(
        "cowork.services.artifact_authorization_identity.publish_authorization_key",
        lambda *args: "owner/artifact",
    )
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert error.value.status_code == 409
    assert error.value.detail == {
        "code": "artifact_owned_by_other_user",
        "message": "This artifact was published by another owner.",
    }
    assert len(calls) == 1
    assert not (as_owner / ".published.json").exists()
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    artifact_locks.release(as_owner.parent, as_owner.name)


async def test_share_reports_an_unmarked_upload_503_as_the_service_being_unavailable(
    as_owner, publish_context, monkeypatch,
):
    _upload_service_answers(monkeypatch, 503, "Service Unavailable", b'{"error": "Service Unavailable"}')
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert error.value.status_code == 503
    assert error.value.detail == {
        "code": "publish_unavailable",
        "message": "The publishing service is temporarily unavailable. Try again in a moment.",
    }
    assert not (as_owner / ".published.json").exists()
    assert _lock_is_free(as_owner)


async def test_a_too_large_artifact_is_413_with_its_code(as_owner, publish_context, monkeypatch):
    _upload_service_answers(monkeypatch, 413, "Payload Too Large", b'{"message": "Request Too Long"}')
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert error.value.status_code == 413
    assert error.value.detail == {
        "code": "artifact_too_large",
        "message": "This artifact is too large to publish. Remove or shrink large files in it, such as PDFs or images.",
    }
    assert not (as_owner / ".published.json").exists()
    assert _lock_is_free(as_owner)


async def test_a_rejected_publish_shows_the_upload_service_text(as_owner, publish_context, monkeypatch):
    from cowork.services.publish import PublishRejected

    def reject(*args, **kwargs):
        raise PublishRejected("Publishing failed: This organization is archived or still being provisioned.")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", reject)
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert error.value.status_code == 502
    assert error.value.detail == "Publishing failed: This organization is archived or still being provisioned."
    assert _lock_is_free(as_owner)


@pytest.mark.parametrize("local_error", [
    RuntimeError("Publishing failed: [Errno 2] No such file: '/data/org-1/proj/x.pdf'"),
    PermissionError("[Errno 13] Permission denied: '/data/org-1/proj/.published.json'"),
])
async def test_a_local_publish_failure_hides_its_text_but_logs_it(
    as_owner, publish_context, monkeypatch, caplog, local_error,
):
    import logging

    def fail(*args, **kwargs):
        raise local_error

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fail)
    with caplog.at_level(logging.WARNING, logger=aw.logger.name):
        with pytest.raises(HTTPException) as error:
            await aw.set_artifact_access(
                "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
            )
    assert error.value.status_code == 502
    assert error.value.detail == "Publishing failed. Try again in a moment."
    record = next(r for r in caplog.records if "Could not change access for artifact" in r.getMessage())
    assert record.exc_info is not None
    assert _lock_is_free(as_owner)


async def test_a_missing_publisher_is_503(as_owner, publish_context, monkeypatch):
    from cowork.services.publish import PublisherUnavailable

    def fail(*args, **kwargs):
        raise PublisherUnavailable("Anton publisher is unavailable")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fail)
    with pytest.raises(HTTPException) as error:
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert error.value.status_code == 503
    assert error.value.detail == "Anton publisher is unavailable"


async def test_the_turn_key_is_revoked_after_a_share(as_owner, publish_calls, publish_context):
    await aw.set_artifact_access(
        "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
    )
    assert publish_context == [True]


async def test_the_turn_key_is_revoked_after_a_failed_share(as_owner, publish_context, monkeypatch):
    from cowork.services.publish import PublishRejected

    def reject(*args, **kwargs):
        raise PublishRejected("Publishing failed: nope")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", reject)
    with pytest.raises(HTTPException):
        await aw.set_artifact_access(
            "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
        )
    assert publish_context == [True]


async def test_the_turn_key_is_kept_while_a_timed_out_publish_still_runs(
    as_owner, publish_context, monkeypatch,
):
    import threading

    finish = threading.Event()

    def slow(*args, **kwargs):
        finish.wait(5)
        return {"status": "ok", "url": "https://view.example/r/1"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", slow)
    monkeypatch.setattr(aw, "_LIVE_PUBLISH_TIMEOUT_S", 0.05)
    try:
        with pytest.raises(HTTPException) as error:
            await aw.set_artifact_access(
                "proj", ARTIFACT_ID, aw._AccessBody(access={"mode": "public"}), _Session(), _REQUEST,
            )
    finally:
        finish.set()
    assert error.value.detail == "Publishing timed out. Try again in a moment."
    assert publish_context == []
