"""Turn-boundary artifact state: what appeared and what changed.

The pre-turn snapshot carries content mtimes, not just folder names, so an
artifact the agent EDITED (rather than created) is reported as touched. The
autopublish reconciler's first phase depends on that distinction.
"""
from __future__ import annotations

import json
import os

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.services import task_objects as t
from cowork.services.conversations import ConversationService


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture
def conv(session):
    return ConversationService(ScopedSession(session, LOCAL_SCOPE)).create_conversation(topic="t")


def _make_artifact(base, slug, *, files: dict[str, str], meta: dict) -> None:
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(json.dumps(meta))


def _bump_mtime(path, delta_s: int) -> None:
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + delta_s))


def test_snapshot_carries_slugs_and_content_mtimes(tmp_path):
    base = tmp_path / "artifacts"
    _make_artifact(base, "one", files={"a.md": "x"}, meta={"slug": "one", "type": "document"})

    slugs, mtimes = t.snapshot_artifact_state(base)

    assert slugs == {"one"}
    assert mtimes["one"] > 0


def test_snapshot_of_missing_dir_is_empty(tmp_path):
    slugs, mtimes = t.snapshot_artifact_state(tmp_path / "nope")
    assert slugs == set()
    assert mtimes == {}


def test_index_reports_new_slug_as_new_and_touched(conv, tmp_path):
    base = tmp_path / "artifacts"
    base.mkdir()
    before, before_mtimes = t.snapshot_artifact_state(base)

    _make_artifact(base, "fresh", files={"r.md": "hi"}, meta={"slug": "fresh", "type": "document"})

    new, touched, _scope = t.index_turn_artifacts(
        conv, conv.id, conv.project_id, base, before, before_mtimes,
    )
    assert new == ["fresh"]
    assert touched == {"fresh"}


def test_index_reports_edited_existing_slug_as_touched_not_new(conv, tmp_path):
    base = tmp_path / "artifacts"
    _make_artifact(base, "old", files={"a.md": "v1"}, meta={"slug": "old", "type": "document"})
    before, before_mtimes = t.snapshot_artifact_state(base)

    (base / "old" / "a.md").write_text("v2")
    _bump_mtime(base / "old" / "a.md", 120)

    new, touched, _scope = t.index_turn_artifacts(
        conv, conv.id, conv.project_id, base, before, before_mtimes,
    )
    assert new == []
    assert touched == {"old"}


def test_index_leaves_untouched_slug_out_of_touched(conv, tmp_path):
    base = tmp_path / "artifacts"
    _make_artifact(base, "old", files={"a.md": "v1"}, meta={"slug": "old", "type": "document"})
    before, before_mtimes = t.snapshot_artifact_state(base)

    new, touched, _scope = t.index_turn_artifacts(
        conv, conv.id, conv.project_id, base, before, before_mtimes,
    )
    assert new == []
    assert touched == set()


def test_index_never_raises_and_degrades_to_empty(conv):
    # A non-path artifacts_base makes the very first operation blow up; the
    # caller runs this right after a turn's finally and must not get an
    # exception that masks the turn's real outcome.
    new, touched, scope = t.index_turn_artifacts(
        conv, conv.id, conv.project_id, object(), set(), {},
    )
    assert (new, touched, scope) == ([], set(), None)


def test_cards_for_slugs_builds_one_card_per_slug(tmp_path):
    base = tmp_path / "artifacts"
    _make_artifact(base, "dash", files={"index.html": "<html></html>"},
                   meta={"slug": "dash", "name": "Dash", "type": "html-app"})

    cards = t.cards_for_slugs(base, ["dash"])

    assert [c["slug"] for c in cards] == ["dash"]
    assert cards[0]["title"] == "Dash"


def test_cards_for_slugs_skips_unreadable_metadata(tmp_path):
    base = tmp_path / "artifacts"
    folder = base / "broken"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text("{ not json")

    assert t.cards_for_slugs(base, ["broken"]) == []


def test_cards_carry_project_identity_when_given(tmp_path):
    # Inline chat cards must be addressable the same way the artifacts panel
    # addresses them (project id + slug), otherwise Delete from a chat card
    # would fall back to the path-based endpoint, which org mode fails closed.
    base = tmp_path / "artifacts"
    _make_artifact(base, "dash", files={"index.html": "<html></html>"},
                   meta={"slug": "dash", "name": "Dash", "type": "html-app"})

    card = t.cards_for_slugs(base, ["dash"], project_id="p-1", project_name="Alpha")[0]

    assert card["projectId"] == "p-1"
    assert card["projectName"] == "Alpha"


def test_scope_falls_back_to_the_ambient_turn_scope(conv, tmp_path, monkeypatch):
    # A detached or expired session yields no scope. The turn boundary binds an
    # ambient one (use_settings_scope in handlers.responses), and it survives
    # that — otherwise a whole class of "artifact never published" would depend
    # on session lifetime.
    from cowork.db.scoped import TenantScope

    ambient = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
    monkeypatch.setattr(t, "scope_of_session", lambda _s: None)
    monkeypatch.setattr(
        "cowork.common.settings.user_settings.current_settings_scope", lambda: ambient
    )

    base = tmp_path / "artifacts"
    base.mkdir()
    _, _, scope = t.index_turn_artifacts(conv, conv.id, conv.project_id, base, set(), {})

    assert scope is ambient


def test_scope_is_none_when_neither_source_has_one(conv, tmp_path, monkeypatch):
    monkeypatch.setattr(t, "scope_of_session", lambda _s: None)
    monkeypatch.setattr(
        "cowork.common.settings.user_settings.current_settings_scope", lambda: None
    )

    base = tmp_path / "artifacts"
    base.mkdir()
    _, _, scope = t.index_turn_artifacts(conv, conv.id, conv.project_id, base, set(), {})

    assert scope is None
