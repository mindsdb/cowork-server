"""Docs on disk and their announcements: the write path, kept versions, listing, the hub."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from cowork.docs import store
from cowork.docs.hub import DocHub
from cowork.docs.model import DocError
from cowork.docs.ops import apply_ops
from cowork.docs.service import apply, create_doc
from cowork.docs.store import DOC_FILENAME, doc_lock, list_doc_folders, list_history, load_doc, with_doc
from cowork.services.artifact_lock import artifact_lock


@pytest.fixture
def made(tmp_path):
    return create_doc(tmp_path / "artifacts", title="Plan", markdown="Lead.\n\n- [ ] One", by="agent")


def _edit(folder: Path, text: str, *, by: str, turn: str | None = None):
    lead = load_doc(folder)["tabs"][0]["blocks"][0]
    return apply(folder, [{"op": "update", "id": lead["id"], "set": {"text": text}}], by=by, turn=turn)


@pytest.fixture
def clock(monkeypatch):
    """The store's clock, moved on by hand: ``clock.minutes(6)``."""
    now = [time.time()]

    class Clock:
        @staticmethod
        def minutes(n: float) -> None:
            now[0] += n * 60

    monkeypatch.setattr(store.time, "time", lambda: now[0])
    return Clock


def test_a_new_doc_is_returned_as_it_was_written(made):
    on_disk = json.loads((made.folder / DOC_FILENAME).read_text())
    assert made.doc == on_disk and made.doc["created_at"] and made.doc["updated_at"]
    assert made.doc["id"] == made.artifact_id
    assert (made.folder / store.DOC_HTML).is_file()


def test_one_agent_reply_is_one_version_kept_when_the_next_session_starts(made):
    for n in range(3):
        _edit(made.folder, f"Agent {n}.", by="agent")  # the reply that created it (no turn named)
    assert list_history(made.folder) == []
    _edit(made.folder, "Typed.", by="user")
    (kept,) = list_history(made.folder)
    assert kept["rev"] == 4 and kept["by"] == "agent" and kept["first"] is True
    assert load_doc(made.folder)["rev"] == 5


def test_each_agent_reply_is_its_own_version(made):
    _edit(made.folder, "First reply.", by="agent", turn="reply-1")
    _edit(made.folder, "First reply, again.", by="agent", turn="reply-1")
    _edit(made.folder, "Second reply.", by="agent", turn="reply-2")
    # The doc as created (rev 1), then as reply 1 left it (rev 3); reply 2 is the doc now.
    assert [(v["rev"], v["by"]) for v in list_history(made.folder)] == [(3, "agent"), (1, "agent")]


def test_a_persons_typing_is_one_version_until_they_pause(made, clock):
    _edit(made.folder, "Typed.", by="user")
    for n in range(3):
        clock.minutes(2)
        _edit(made.folder, f"Typed {n}.", by="user")
    assert [v["rev"] for v in list_history(made.folder)] == [1]  # only the agent's draft, before the typing
    clock.minutes(store.SESSION_PAUSE_S / 60)
    _edit(made.folder, "After a pause.", by="user")
    newest = list_history(made.folder)[0]
    assert newest["rev"] == 5 and newest["by"] == "user"
    assert store._seconds_since(newest["at"], store.time.time()) == pytest.approx(store.SESSION_PAUSE_S, abs=1)


def test_a_long_stretch_of_typing_is_cut_every_half_hour(made, clock):
    for n in range(10):  # every four minutes for forty: never a pause
        _edit(made.folder, f"Typed {n}.", by="user")
        clock.minutes(4)
    users = [v for v in list_history(made.folder) if v["by"] == "user"]
    assert len(users) == 1  # one cut, at the thirty-minute mark
    assert store._seconds_since(users[0]["started"], store.time.time()) == pytest.approx(40 * 60, abs=1)


def test_the_agent_stepping_in_ends_the_persons_session_and_back(made):
    _edit(made.folder, "Typed.", by="user")
    _edit(made.folder, "Agent.", by="agent", turn="reply-2")
    _edit(made.folder, "Typed again.", by="user")
    assert [v["by"] for v in list_history(made.folder)] == ["agent", "user", "agent"]


def test_a_doc_written_before_sessions_were_recorded_keeps_its_version_too(made):
    (made.folder.joinpath(*store.HISTORY_DIR) / "open.json").unlink()
    _edit(made.folder, "Typed.", by="user")
    (kept,) = list_history(made.folder)
    assert kept["rev"] == 1 and kept["by"] is None and kept["at"] == made.doc["updated_at"]


def test_only_the_newest_versions_are_kept(made, monkeypatch):
    monkeypatch.setattr(store, "HISTORY_KEEP", 3)
    for n in range(7):
        _edit(made.folder, f"Agent {n}.", by="agent", turn=f"reply-{n}")
    assert [v["rev"] for v in list_history(made.folder)] == [7, 6, 5]
    kept = sorted(p.name for p in made.folder.joinpath(*store.HISTORY_DIR).glob("0*.json"))
    assert kept == ["0000005.json", "0000006.json", "0000007.json"]


def _kept_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.joinpath(*store.HISTORY_DIR).glob("0*.json"))


def test_the_newest_versions_are_kept_within_a_total_size(made, monkeypatch):
    """Up to a hundred versions of up to 8 MB each would be 800 MB of one doc."""
    _edit(made.folder, "Agent 0.", by="agent", turn="reply-0")
    one_version = (made.folder.joinpath(*store.HISTORY_DIR) / "0000001.json").stat().st_size
    monkeypatch.setattr(store, "HISTORY_MAX_BYTES", int(one_version * 2.5))
    _edit(made.folder, "Agent 1.", by="agent", turn="reply-1")
    # As a version kept before the index recorded sizes: read off the file instead.
    index = made.folder.joinpath(*store.HISTORY_DIR) / "index.json"
    index.write_text(json.dumps([{k: v for k, v in e.items() if k != "bytes"} for e in json.loads(index.read_text())]))
    for n in range(2, 5):
        _edit(made.folder, f"Agent {n}.", by="agent", turn=f"reply-{n}")
    assert [v["rev"] for v in list_history(made.folder)] == [5, 4]
    assert _kept_files(made.folder) == ["0000004.json", "0000005.json"]


def test_the_newest_version_is_kept_even_when_it_alone_is_past_the_size(made, monkeypatch):
    monkeypatch.setattr(store, "HISTORY_MAX_BYTES", 1)
    for n in range(3):
        _edit(made.folder, f"Agent {n}.", by="agent", turn=f"reply-{n}")
    assert [v["rev"] for v in list_history(made.folder)] == [3]
    assert _kept_files(made.folder) == ["0000003.json"]


def _texts(folder: Path) -> list[str]:
    return [b.get("text") for b in load_doc(folder)["tabs"][0]["blocks"]]


def test_a_write_waits_for_another_process_writing_the_doc_and_keeps_its_edit(made):
    """Hosted, two replicas write one shared tree: each read-change-write holds
    the artifact's lock file, so neither writes its rev over the other's."""
    read, done = threading.Event(), threading.Event()

    def other_replica():  # its own mutate_doc, midway: it has read the doc and not yet written it
        with artifact_lock(made.folder):
            doc = load_doc(made.folder)
            read.set()
            time.sleep(0.3)
            doc, _ = apply_ops(doc, [{"op": "insert", "position": "end", "markdown": "From the other replica."}],
                               by="user")
            store.atomic_write(store.doc_file(made.folder), json.dumps(doc).encode("utf-8"))
        done.set()

    threading.Thread(target=other_replica).start()
    assert read.wait(1)
    _edit(made.folder, "From here.", by="user")
    assert done.is_set()  # it waited for the other write to finish
    assert load_doc(made.folder)["rev"] == 3
    assert {"From here.", "From the other replica."} <= set(_texts(made.folder))


def test_a_write_made_while_holding_the_doc_takes_its_lock_file_once(made):
    """cloud.apply_edit holds the doc's lock around a write; the lock file is not re-entrant."""
    lock_file = made.folder / ".revisions" / ".lock"
    with doc_lock(made.folder):
        assert lock_file.is_file()
        with_doc(made.folder, lambda doc: None)
        _edit(made.folder, "Nested.", by="user")
        assert lock_file.is_file()
    assert not lock_file.exists()
    assert _texts(made.folder)[0] == "Nested."


def test_a_write_kept_waiting_past_the_lock_files_wait_is_refused_as_busy(made, monkeypatch):
    from functools import partial

    monkeypatch.setattr(store, "artifact_lock", partial(artifact_lock, timeout=0.1))
    with artifact_lock(made.folder):  # another process, holding the doc too long
        with pytest.raises(DocError) as err:
            _edit(made.folder, "Refused.", by="user")
    assert err.value.code == "busy"
    _edit(made.folder, "Later.", by="user")  # nothing in this process was left holding it
    assert _texts(made.folder)[0] == "Later."


def test_a_write_to_a_doc_deleted_meanwhile_is_refused_and_leaves_no_folder(made):
    import shutil

    shutil.rmtree(made.folder)
    with pytest.raises(DocError) as err:
        apply(made.folder, [{"op": "insert", "position": "end", "markdown": "x"}], by="agent")
    assert err.value.code == "no_doc"
    assert not made.folder.exists()


def test_a_failed_change_writes_nothing(made):
    before = (made.folder / DOC_FILENAME).read_bytes()
    with pytest.raises(DocError):
        apply(made.folder, [{"op": "delete", "target": "b_nothere1"}], by="user")
    assert (made.folder / DOC_FILENAME).read_bytes() == before


def test_a_change_that_would_write_an_unloadable_doc_is_refused(made):
    before = (made.folder / DOC_FILENAME).read_bytes()

    def breaks_it(doc):
        doc = {**doc, "tabs": [{**doc["tabs"][0], "title": "t" * 20_001}]}
        return doc, None

    with pytest.raises(DocError) as err:
        store.mutate_doc(made.folder, breaks_it, by="user")
    assert err.value.code == "too_long"
    assert (made.folder / DOC_FILENAME).read_bytes() == before
    load_doc(made.folder)


def test_listing_skips_a_doc_file_that_wont_parse(made, tmp_path):
    broken = create_doc(tmp_path / "artifacts", title="Broken", markdown="x").folder
    (broken / DOC_FILENAME).write_text("{not json")
    assert [d["title"] for d in list_doc_folders(tmp_path / "artifacts")] == ["Plan"]
    with pytest.raises(DocError) as err:
        load_doc(broken)
    assert err.value.code == "bad_doc"


def test_a_late_subscriber_hears_a_kept_event_and_a_slow_one_learns_it_missed_some():
    async def run():
        hub = DocHub()
        hub.publish("conv:1", {"type": "doc.created", "rev": 1}, keep=True)
        late = hub.subscribe("conv:1", replay=True)
        assert (await late.get()) == {"type": "doc.created", "rev": 1, "replayed": True}

        slow = hub.subscribe("doc:a")
        for rev in range(80):  # more than its queue holds
            hub.publish("doc:a", {"rev": rev})
        await asyncio.sleep(0)
        events = [slow.get_nowait() for _ in range(slow.qsize())]
        assert events[-1] == {"rev": 79, "gap": True}  # the client rereads the doc

    asyncio.run(run())


def test_a_conversation_with_nothing_left_to_replay_is_forgotten(monkeypatch):
    from cowork.docs import hub as hub_module

    hub = DocHub()
    clock = [1000.0]
    monkeypatch.setattr(hub_module.time, "monotonic", lambda: clock[0])
    hub.publish("conv:old", {"rev": 1}, keep=True)
    clock[0] += hub_module._REPLAY_S + 1
    hub.publish("conv:new", {"rev": 1}, keep=True)
    assert list(hub._recent) == ["conv:new"]
