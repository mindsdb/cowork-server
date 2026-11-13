"""An agent turn's artifact revision says what changed (ENG-4227): the
editor's own summary, else the reply's first sentence, else the request,
else the generic label."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.services import task_objects as t
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_revisions import capture_agent_revision, current_source, list_revisions
from cowork.services.conversations import ConversationService
from cowork.services.revision_summaries import (
    MAX_SUMMARY,
    TurnRevisionSummaries,
    editor_summaries,
    reply_summary,
    summaries_from_messages,
)


def edit_result(slug: str, name: str, reply: str) -> dict:
    """A tool row as anton's edit_artifact leaves it for an applied edit."""
    text = (
        f"Edited “{name}” (`{slug}`): index.html, in 1 model call.\n"
        f"What changed: {reply}\n"
        "The change is applied, but no browser verified it: your reply MUST say it was not verified."
    )
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}]}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


# ── the sources ───────────────────────────────────────────────────────────


def test_the_editor_s_summary_is_read_per_artifact():
    rows = [
        edit_result("sales-1a2b", "Sales", "Added a pricing section under the chart."),
        edit_result("costs-3c4d", "Costs", "Made the totals bold."),
        # A later edit of the first artifact in the same turn.
        {"role": "user", "content": [{"type": "tool_result", "content": [
            {"type": "text", "text": "Edited “Sales” (`sales-1a2b`): app.js, in 2 model calls.\nWhat changed: Sorted the table by revenue."},
        ]}]},
        # A failed edit says nothing changed: ignored.
        {"role": "user", "content": [{"type": "tool_result", "content": "Nothing was changed. You may call edit_artifact once more."}]},
    ]
    assert editor_summaries(rows) == {
        "sales-1a2b": "Added a pricing section under the chart.; Sorted the table by revenue.",
        "costs-3c4d": "Made the totals bold.",
    }


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("**Done!** I added a pricing section with three tiers. It is not verified.", "I added a pricing section with three tiers."),
        ("I added a pricing section with three tiers. It is not verified in a browser.", "I added a pricing section with three tiers."),
        ("## Updated\nThe header is now navy and the chart uses a log scale.", "The header is now navy and the chart uses a log scale."),
        ("Would you like me to also change the footer?", None),
        ("Here is what I changed:", None),
        ("", None),
    ],
)
def test_the_reply_s_first_sentence(reply, expected):
    assert reply_summary(reply) == expected


def test_a_long_label_is_capped():
    summaries = TurnRevisionSummaries(editor={"s": "word " * 100})
    label = summaries.summary_for("s")
    assert label.startswith("Agent: word word")
    assert len(label) <= MAX_SUMMARY and label.endswith("…")


# ── the order ─────────────────────────────────────────────────────────────


def test_the_fallback_order():
    full = TurnRevisionSummaries(
        editor={"sales": "Added a pricing section."},
        reply="I made the table sortable by revenue. Not verified.",
        request="make the table sortable",
    )
    assert full.summary_for("sales") == "Agent: Added a pricing section."
    # Another artifact of the same turn, which the editor did not touch.
    assert full.summary_for("costs") == "Agent: I made the table sortable by revenue."
    no_reply = TurnRevisionSummaries(reply="Should I also sort the chart?", request="make the table sortable")
    assert no_reply.summary_for("costs") == "Agent: Asked: make the table sortable"
    assert TurnRevisionSummaries().summary_for("costs") is None


def test_summaries_from_a_turn_s_messages():
    messages = [
        {"role": "user", "content": "make the heading navy"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "edit_artifact", "input": {"slug": "Sales"}}]},
        edit_result("sales-1a2b", "Sales", "The heading is navy now."),
        assistant("I changed the heading to navy. It was not verified in a browser."),
    ]
    summaries = summaries_from_messages(messages, request="make the heading navy")
    assert summaries.summary_for("sales-1a2b") == "Agent: The heading is navy now."
    assert summaries.summary_for("other") == "Agent: I changed the heading to navy."


def test_the_harness_reads_the_turn_s_part_of_the_history():
    from cowork.harnesses.anton_harness.harness import _turn_revision_summaries

    earlier = [assistant("An earlier turn's reply about something else entirely.")]
    turn = [edit_result("sales-1a2b", "Sales", "Added a legend."), assistant("I added a legend to the chart.")]
    session = SimpleNamespace(history=earlier + turn)
    summaries = _turn_revision_summaries(session, len(earlier), "add a legend")
    assert summaries.summary_for("sales-1a2b") == "Agent: Added a legend."
    assert summaries.reply == "I added a legend to the chart."
    assert _turn_revision_summaries(SimpleNamespace(history=None), 0, "x").summary_for("a") == "Agent: Asked: x"


# ── the revisions ─────────────────────────────────────────────────────────


@pytest.fixture
def artifact(tmp_path):
    folder = tmp_path / "artifacts" / "sales"
    folder.mkdir(parents=True)
    meta = {"id": "a1b2c3d4", "slug": "sales", "createdAt": "2026-11-12T12:00:00+00:00",
            "name": "Sales", "type": "html-app", "primary": "index.html"}
    (folder / "metadata.json").write_text(json.dumps(meta))
    (folder / "index.html").write_text("<h1>v1</h1>")
    artifact_id, meta = ensure_full_id(folder, meta)
    return folder, meta, artifact_id


def test_the_capture_uses_the_summary_and_keeps_the_actor(artifact):
    folder, meta, artifact_id = artifact
    current_source(folder, meta, artifact_id)
    (folder / "index.html").write_text("<h1>v2</h1>")
    revision = capture_agent_revision(folder, conversation_id="c1", summary="Agent: Added a pricing section.")
    assert revision["summary"] == "Agent: Added a pricing section."
    assert revision["actor"]["kind"] == "agent"


def test_without_a_summary_the_label_stays_generic(artifact):
    folder, meta, artifact_id = artifact
    current_source(folder, meta, artifact_id)
    (folder / "index.html").write_text("<h1>v2</h1>")
    assert capture_agent_revision(folder, conversation_id="c1")["summary"] == "Agent updated artifact"


def test_a_viewer_capture_of_the_turn_s_write_takes_the_summary(artifact):
    folder, meta, artifact_id = artifact
    current_source(folder, meta, artifact_id)
    (folder / "index.html").write_text("<h1>v2</h1>")
    captured = current_source(folder, meta, artifact_id)["revision"]  # the viewer reloaded mid-turn
    revision = capture_agent_revision(folder, conversation_id="c1", summary="Agent: Added a legend.")
    assert revision["id"] == captured["id"]
    assert (revision["summary"], revision["actor"]["kind"]) == ("Agent: Added a legend.", "agent")
    assert len(list_revisions(folder)) == 2


def test_old_revisions_are_left_alone(artifact):
    folder, meta, artifact_id = artifact
    current_source(folder, meta, artifact_id)
    (folder / "index.html").write_text("<h1>v2</h1>")
    capture_agent_revision(folder, conversation_id="c1")
    (folder / "index.html").write_text("<h1>v3</h1>")
    capture_agent_revision(folder, conversation_id="c2", summary="Agent: Made it v3.")
    assert [r["summary"] for r in list_revisions(folder)] == [
        "Agent: Made it v3.", "Agent updated artifact", "Captured current draft",
    ]


# ── the turn end ──────────────────────────────────────────────────────────


@pytest.fixture
def conv():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield ConversationService(ScopedSession(s, LOCAL_SCOPE)).create_conversation(topic="t")


def _bump(path, seconds=120):
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + seconds))


def test_each_artifact_a_turn_changed_gets_its_own_label(conv, tmp_path):
    base = tmp_path / "artifacts"
    for slug in ("sales", "costs"):
        folder = base / slug
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(json.dumps({"slug": slug, "type": "document", "primary": "a.md"}))
        (folder / "a.md").write_text("v1")
        meta = json.loads((folder / "metadata.json").read_text())
        artifact_id, meta = ensure_full_id(folder, meta)
        current_source(folder, meta, artifact_id)
    before, before_mtimes = t.snapshot_artifact_state(base)
    for slug in ("sales", "costs"):
        (base / slug / "a.md").write_text("v2")
        _bump(base / slug / "a.md")

    summaries = TurnRevisionSummaries(
        editor={"sales": "Added a pricing section."}, reply="I updated both reports.", request="update both",
    )
    changes = t.turn_artifact_changes(
        conversation_id=conv.id, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        summaries=summaries,
    )

    assert changes.touched == {"sales", "costs"}
    assert list_revisions(base / "sales")[0]["summary"] == "Agent: Added a pricing section."
    assert list_revisions(base / "costs")[0]["summary"] == "Agent: I updated both reports."
