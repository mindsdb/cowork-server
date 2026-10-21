"""What a turn tells the agent, and hands anton, with live apps on: the live-app
rule in place of the sandbox's refusal, the person's running apps and the
stops they haven't heard of, the copies in the project, and a host that
records each launch for the person whose turn it is."""

from __future__ import annotations

import sys
from uuid import uuid4

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.handlers.responses import ResponsesHandler
from cowork.harnesses.anton_harness import harness
from cowork.models.live_app import LiveAppCopy
from cowork.models.project import Project
from cowork.principal import Principal
from cowork.services.live_apps import (
    CoworkLiveAppHost,
    current_live_app_launcher,
    live_app_person,
    use_live_app_launcher,
)
from tests._live_apps import (
    ALICE,
    BEARER_TOKEN,
    EMAIL_A,
    EMAIL_B,
    LAUNCH_TOKEN,
    MODULE,
    PARTITION_A,
    SUB_A,
    FakeServices,
    answer_over_http,
    clear_live_app_rows,
    person_scope,
    turn_on_live_apps,
)
from tests._turns import built_turn

ALICE_SCOPE = person_scope(PARTITION_A, SUB_A)
PRINCIPAL_A = Principal(user_id=SUB_A, org_id=PARTITION_A, email=EMAIL_A)
BOOT_1 = "1" * 32


@pytest.fixture
def stack(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", BEARER_TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    yield monkeypatch
    clear_live_app_rows()
    get_app_settings.cache_clear()


@pytest.fixture
def live(stack):
    turn_on_live_apps(stack)
    services = FakeServices()
    answer_over_http(stack, services)
    return services


async def _launch(services: FakeServices, slot: int, *, slug: str, project_id) -> None:
    host = CoworkLiveAppHost(ALICE, project_id=project_id, launch_token=LAUNCH_TOKEN)
    await host.claim(slot)
    await host.launched(
        sys.modules[MODULE].LiveAppLaunch(
            slot=slot,
            slug=slug,
            artifact_dir=f"/p/{slug}",
            address=f"http://app{slot}.view.localhost:8080/",
            boot_id=BOOT_1,
        )
    )
    services.run(slot, boot_id=BOOT_1, artifact=slug, email=EMAIL_A)


def _suffix(turn) -> str:
    return turn.config.system_prompt_context.suffix


# ── the launcher ─────────────────────────────────────────────────────────────


def test_a_principal_without_an_email_names_nobody():
    assert live_app_person(Principal(user_id=SUB_A, org_id=PARTITION_A)) is None
    assert live_app_person(None) is None
    assert live_app_person(Principal(user_id=SUB_A, org_id=PARTITION_A, email=" Alice@Example.com ")) == ALICE


async def test_a_turn_runs_with_its_principals_launcher(stack):
    """The in-process turn's producer names the person whose turn it is."""
    seen = []

    async def run_turn(**_kwargs):
        seen.append(current_live_app_launcher())

    handler = ResponsesHandler(principal=PRINCIPAL_A)
    handler._run_turn = run_turn
    await handler._produce()

    assert seen == [ALICE]
    assert current_live_app_launcher() is None


# ── the turn ─────────────────────────────────────────────────────────────────


async def test_a_turn_with_live_apps_off_keeps_the_sandboxs_refusal(stack, monkeypatch, tmp_path):
    with use_live_app_launcher(PRINCIPAL_A):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE_SCOPE) as turn:
            suffix = _suffix(turn)
            config = turn.config

    assert harness._SANDBOXED_BACKENDS_RULE in suffix
    assert harness._LIVE_APPS_RULE not in suffix
    assert "fullstack apps" not in suffix
    assert getattr(config, "live_app_host", None) is None


async def test_a_turn_with_live_apps_on_offers_full_stack_apps(live, monkeypatch, tmp_path):
    with use_live_app_launcher(PRINCIPAL_A):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE_SCOPE) as turn:
            suffix = _suffix(turn)
            host = turn.config.live_app_host

    assert harness._LIVE_APPS_RULE in suffix
    assert harness._SANDBOXED_BACKENDS_RULE not in suffix
    assert "datasets, fullstack apps, etc." in suffix
    assert isinstance(host, CoworkLiveAppHost)
    assert (host.launcher.subject, host.launcher.email) == (SUB_A, EMAIL_A)
    assert host.launch_token == LAUNCH_TOKEN


async def test_a_turn_that_names_nobody_gets_no_host(live, monkeypatch, tmp_path):
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE_SCOPE) as turn:
        suffix = _suffix(turn)
        config = turn.config

    assert harness._LIVE_APPS_RULE in suffix
    assert getattr(config, "live_app_host", None) is None


async def test_the_turn_names_running_apps_copies_and_each_stop_once(live, monkeypatch, tmp_path):
    folder = tmp_path / "projects" / "alices"
    folder.mkdir(parents=True)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        project = Project(name="alices", path=str(folder), org_id=PARTITION_A, created_by=SUB_A)
        raw.add(project)
        raw.add(
            LiveAppCopy(
                partition_id=PARTITION_A, project_id=project.id, slug="orders-copy", owner_email=EMAIL_B, share_id=uuid4()
            )
        )
        raw.commit()
        raw.refresh(project)
        raw.expunge(project)
    await _launch(live, 1, slug="dash", project_id=project.id)
    await _launch(live, 2, slug="report", project_id=project.id)
    # report's slot restarts with no report, so the turn finds it lost.
    live.slots[2].boot_id = "2" * 32
    live.slots[2].state = "empty"

    async def turn_suffix() -> str:
        with use_live_app_launcher(PRINCIPAL_A):
            async with built_turn(
                monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE_SCOPE, project=project
            ) as turn:
                return _suffix(turn)

    once = await turn_suffix()
    twice = await turn_suffix()

    assert "The person's live app `dash` is running at `http://app1.view.localhost:8080/`." in once
    assert "The person's live app `report` stopped at" in once
    assert "its app slot restarted. Tell them once, and offer to relaunch it." in once
    assert f"copied into this project: `orders-copy`, from {EMAIL_B}." in once
    assert "`report` stopped" not in twice
    assert "`dash` is running" in twice
