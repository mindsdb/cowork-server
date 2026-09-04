"""A stored harness the install cannot build must not brick every settings read.

``validate_harness`` checks the stored value against the harnesses this process
knows, so once an optional harness package is gone, ``SettingService._load``
raises for every read of that scope and the UI cannot offer a way out, because
rendering settings is itself a read. ``reset_unbuildable_harness`` runs at boot
and points the row back at the default.

Own in-memory engine per test, following tests/test_settings_tenancy.py: the
session-scoped DB in conftest is shared with every other test file, and a
leftover global ``harness`` row would change what they resolve.
"""

from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import cowork.harnesses  # noqa: F401 — populates the harness registry
from cowork.harnesses import base
from cowork.common.settings.user_settings import UserSettings
from cowork.migrations import reset_unbuildable_harness
from cowork.models.setting import Setting

_DEFAULT = UserSettings.model_fields["harness"].default


@pytest.fixture()
def engine():
    import cowork.models.project, cowork.models.conversation  # noqa: F401
    import cowork.models.message, cowork.models.message_event  # noqa: F401
    import cowork.models.file, cowork.models.channel, cowork.models.setting  # noqa: F401
    import cowork.models.task_object, cowork.models.schedule, cowork.models.pin  # noqa: F401
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    return eng


def _seed(engine, value: str) -> None:
    """A global (scope NULL) harness row, the shape desktop and legacy rows have."""
    with Session(engine) as session:
        session.add(Setting(key="harness", value=value))
        session.commit()


def _stored(engine) -> str | None:
    with Session(engine) as session:
        row = session.exec(select(Setting).where(Setting.key == "harness")).first()
        return row.value if row else None


def _registry(monkeypatch, *ids: str, no_org: tuple[str, ...] = ()) -> None:
    """Stand in for an install that ships only ``ids``.

    Entries carry ``supports_org_mode`` because that is what
    ``available_harness_ids`` filters on; a bare stub would read as org-capable
    through its ``getattr`` default and hide the distinction under test.

    A copy, never the live dict: ``base._registry`` is process-global and other
    test modules import the harness modules directly, so mutating it in place
    would leak across the session.
    """
    entries = {
        i: type("StubHarness", (), {"id": i, "supports_org_mode": i not in no_org})
        for i in ids
    }
    monkeypatch.setattr(base, "_registry", entries)


def test_resets_a_harness_that_is_not_installed(engine, monkeypatch) -> None:
    _seed(engine, "hermes")
    _registry(monkeypatch, _DEFAULT)

    with Session(engine) as session:
        assert reset_unbuildable_harness(session) is True

    assert _stored(engine) == _DEFAULT


def test_leaves_an_installed_harness_alone(engine, monkeypatch) -> None:
    _seed(engine, "hermes")
    _registry(monkeypatch, _DEFAULT, "hermes")

    with Session(engine) as session:
        assert reset_unbuildable_harness(session) is False

    assert _stored(engine) == "hermes"


def test_a_second_boot_changes_nothing(engine, monkeypatch) -> None:
    _seed(engine, "hermes")
    _registry(monkeypatch, _DEFAULT)

    with Session(engine) as session:
        assert reset_unbuildable_harness(session) is True
    with Session(engine) as session:
        assert reset_unbuildable_harness(session) is False

    assert _stored(engine) == _DEFAULT


def test_writes_nothing_when_the_default_itself_is_missing(engine, monkeypatch) -> None:
    """Otherwise one unvalidatable value replaces another, with nothing to undo it."""
    _seed(engine, "hermes")
    _registry(monkeypatch, "something-else")

    with Session(engine) as session:
        assert reset_unbuildable_harness(session) is False

    assert _stored(engine) == "hermes"


def test_writes_nothing_when_no_harness_is_registered(engine, monkeypatch) -> None:
    _seed(engine, "hermes")
    _registry(monkeypatch)

    with Session(engine) as session:
        assert reset_unbuildable_harness(session) is False

    assert _stored(engine) == "hermes"


def test_a_fresh_install_with_no_row_is_not_an_error(engine, monkeypatch) -> None:
    _registry(monkeypatch, _DEFAULT)

    with Session(engine) as session:
        assert reset_unbuildable_harness(session) is False

    assert _stored(engine) is None


def test_the_repaired_value_is_one_the_model_accepts(engine, monkeypatch) -> None:
    """The point of the repair: the row loads again afterwards.

    Both halves matter. The stored value raised before, and the value written in
    its place validates, so a settings read stops failing rather than failing
    differently.
    """
    _seed(engine, "hermes")
    _registry(monkeypatch, _DEFAULT)

    with pytest.raises(Exception):
        UserSettings(harness="hermes")

    with Session(engine) as session:
        reset_unbuildable_harness(session)

    assert UserSettings(harness=_stored(engine)).harness == _DEFAULT


def test_an_installed_harness_hidden_by_org_mode_is_left_alone(engine, monkeypatch) -> None:
    """Pins the choice of registered_harness_ids over available_harness_ids.

    In org mode a harness with supports_org_mode False is filtered out of the
    offered list while remaining installed and buildable. Rewriting on that
    basis would discard a valid preference during a rolling deploy, where an
    older replica that does not offer a harness would reset every row naming it,
    with no sentinel and no history to recover the value.
    """
    from cowork.common.settings.app_settings import get_app_settings

    _seed(engine, "hermes")
    _registry(monkeypatch, _DEFAULT, "hermes", no_org=("hermes",))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        from cowork.harnesses.base import available_harness_ids, registered_harness_ids

        assert "hermes" in registered_harness_ids()
        assert "hermes" not in available_harness_ids()

        with Session(engine) as session:
            assert reset_unbuildable_harness(session) is False
    finally:
        get_app_settings.cache_clear()

    assert _stored(engine) == "hermes"
