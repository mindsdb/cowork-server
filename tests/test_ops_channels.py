"""``python -m cowork.ops.channels``: the operator sets a local deployment's
channels, which no person's request may change where identity headers are
trusted (tests/test_channels_trusted_identity.py).

``set`` reads the fields on stdin and never prints a secret, ``delete``
removes them, and ``agent`` points channel conversations at a harness. Each
acts on the deployment's one installation, and org tenancy is refused.
"""
from __future__ import annotations

import io

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import get_user_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.ops.channels import main
from cowork.services.channels import ChannelConfigService


@pytest.fixture
def deployment(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    get_app_settings.cache_clear()
    yield
    main(["delete", "telegram"])
    get_app_settings.cache_clear()


def _run(argv: list[str], capsys, stdin: str = "") -> tuple[int, str, str]:
    status = main(argv, stdin=io.StringIO(stdin))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _stored(channel: str) -> dict[str, str]:
    with get_open_session() as raw:
        return ChannelConfigService(ScopedSession(raw, LOCAL_SCOPE)).load_credentials(channel)


def test_set_reads_the_fields_on_stdin_and_never_prints_them(deployment, capsys):
    status, out, err = _run(["set", "telegram"], capsys, " bot_token = operators-bot \n\n")

    assert status == 0
    assert out.startswith("Saved bot_token for telegram, which is configured.")
    assert "next starts" in out
    assert "operators-bot" not in out + err
    assert _stored("telegram") == {"bot_token": "operators-bot"}


def test_delete_removes_the_channels_credentials(deployment, capsys):
    _run(["set", "telegram"], capsys, "bot_token=operators-bot")

    assert _run(["delete", "telegram"], capsys)[0] == 0
    assert _stored("telegram") == {}
    status, _out, err = _run(["delete", "telegram"], capsys)
    assert (status, err) == (1, "error: telegram has no stored config.\n")


@pytest.mark.parametrize(
    ("argv", "stdin", "refusal"),
    [
        (["set", "telegram"], "", "error: Give the channel's fields as name=value lines on stdin.\n"),
        (["set", "telegram"], "bot_token\n", "error: Line 1 of stdin is not name=value.\n"),
        (["set", "telegram"], "bot_token= \n", "error: Line 1 of stdin gives bot_token no value.\n"),
        (["set", "telegram"], "nope=1", "error: unknown credential field(s) for 'telegram': nope\n"),
        (["set", "carrier-pigeon"], "bot_token=x", "error: No channel is named carrier-pigeon.\n"),
        (["agent", "nobody"], "", "error: unknown harness 'nobody'. Available: anton\n"),
    ],
)
def test_a_command_that_cant_do_what_it_was_asked_changes_nothing(deployment, capsys, argv, stdin, refusal):
    status, out, err = _run(argv, capsys, stdin)

    assert (status, out, err) == (1, "", refusal)
    assert _stored("telegram") == {}


def test_agent_sets_the_harness_that_serves_channel_conversations(deployment, capsys):
    status, out, _err = _run(["agent", "anton"], capsys)

    assert status == 0
    assert out.startswith("Channel conversations use anton;")
    assert get_user_settings(LOCAL_SCOPE).channels_harness == "anton"


def test_org_tenancy_is_refused(org_deployment, capsys):
    status, out, err = _run(["set", "slack"], capsys, "signing_secret=x")

    assert (status, out) == (1, "")
    assert "in org tenancy each org's admin sets its own" in err


def test_set_refuses_a_terminal_rather_than_wait_on_it(deployment, capsys):
    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    status = main(["set", "telegram"], stdin=Terminal())

    assert (status, capsys.readouterr().err) == (
        1, "error: Pipe the channel's fields as name=value lines on stdin.\n"
    )
