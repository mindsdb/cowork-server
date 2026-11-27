"""A memory slot changed from the memory editor takes the lock anton's writers
take, so a background compaction can neither replace the edit nor undo a delete."""

from __future__ import annotations

import sys
import threading

import pytest
from anton.core.memory.cortex import plan_compaction, read_compaction_input
from fastapi import HTTPException
from anton.core.memory.hippocampus import Hippocampus

from cowork.harnesses.memory.registry import MemorySlot
from cowork.api.v1.endpoints import memory as memory_endpoints
from cowork.db.scoped import LOCAL_SCOPE
from cowork.harnesses.memory import store as store_mod
from cowork.harnesses.memory.store import GlobalMemoryStore, MemorySlotBusyError, ProjectMemoryStore
from cowork.schemas.memory import MemoryDeleteRequest, MemoryUpdateRequest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the writer lock is a no-op without fcntl"
)


class _KeepFirst:
    """Stands in for the coding model: keeps entry 1 and drops the rest."""

    async def generate_object_code(self, schema, *, system, messages, max_tokens):
        return schema(keep=[1])


def _lessons(store: GlobalMemoryStore, count: int) -> Hippocampus:
    hc = Hippocampus(store.root)
    for i in range(count):
        hc.encode_lesson(f"Fact number {i}", topic=f"t{i}")
    return hc


@pytest.mark.parametrize("change", ["write", "delete", "restore_exact"])
def test_a_slot_change_waits_for_anton_s_writer_lock(tmp_path, change):
    store = GlobalMemoryStore(root=tmp_path)
    store.write(MemorySlot.RULES, "# Rules\n")
    hc = Hippocampus(store.root)
    reached = threading.Event()
    done = threading.Event()

    def run():
        reached.set()
        if change == "write":
            store.write(MemorySlot.RULES, "# Rules\n\n## Always\n- edited\n")
        elif change == "delete":
            store.delete(MemorySlot.RULES)
        else:
            store.restore_exact(MemorySlot.RULES, "# Rules\n", existed=True)
        done.set()

    with hc._file_lock(hc._rules_path):
        threading.Thread(target=run, daemon=True).start()
        assert reached.wait(5)
        assert not done.wait(0.3), f"{change} went through while anton held the lock"
    assert done.wait(5)


async def test_an_editor_delete_during_compaction_is_not_undone(tmp_path):
    store = GlobalMemoryStore(root=tmp_path)
    hc = _lessons(store, 10)
    plan = await plan_compaction(_KeepFirst(), read_compaction_input(hc._lessons_path, "lesson"))

    store.delete(MemorySlot.LESSONS)

    assert plan.apply(hc) is False
    assert not hc._lessons_path.exists()


async def test_an_editor_save_during_compaction_is_not_replaced(tmp_path):
    store = GlobalMemoryStore(root=tmp_path)
    hc = _lessons(store, 10)
    plan = await plan_compaction(_KeepFirst(), read_compaction_input(hc._lessons_path, "lesson"))

    store.write(MemorySlot.LESSONS, "# Lessons\n- Typed in the editor")

    assert plan.apply(hc) is False
    assert store.read(MemorySlot.LESSONS) == "# Lessons\n- Typed in the editor\n"


@pytest.mark.parametrize("change", ["write", "delete"])
def test_a_link_at_the_lock_name_is_refused(tmp_path, change):
    project = tmp_path / "project"
    memory = project / ".anton" / "memory"
    memory.mkdir(parents=True)
    (memory / "rules.md").write_text("# Rules\n- Kept\n")
    victim = tmp_path / "victim"
    (memory / "rules.md.lock").symlink_to(victim)
    store = ProjectMemoryStore(project)

    with pytest.raises(OSError):
        if change == "write":
            store.write(MemorySlot.RULES, "# Rules\n")
        else:
            store.delete(MemorySlot.RULES)

    assert not victim.exists()
    assert (memory / "rules.md").read_text() == "# Rules\n- Kept\n"


@pytest.mark.parametrize("change", ["write", "delete"])
def test_a_lock_held_too_long_fails_the_change(tmp_path, monkeypatch, change):
    """A stuck holder must not hold the editor's worker, and its database lock, forever."""
    monkeypatch.setattr(store_mod, "MEMORY_SLOT_LOCK_TIMEOUT", 0.1)
    store = GlobalMemoryStore(root=tmp_path)
    store.write(MemorySlot.RULES, "# Rules\n- Kept\n")
    hc = Hippocampus(store.root)

    with hc._file_lock(hc._rules_path), pytest.raises(MemorySlotBusyError):
        if change == "write":
            store.write(MemorySlot.RULES, "# Rules\n")
        else:
            store.delete(MemorySlot.RULES)

    assert store.read(MemorySlot.RULES) == "# Rules\n- Kept\n"


async def _busy(use, *, scope, principal):
    raise MemorySlotBusyError("busy")


@pytest.mark.parametrize("call", ["update", "delete"])
async def test_a_busy_slot_answers_503(monkeypatch, call):
    monkeypatch.setattr(memory_endpoints, "_with_memory_service", _busy)
    with pytest.raises(HTTPException) as raised:
        if call == "update":
            await memory_endpoints.update_memory(
                MemoryUpdateRequest(scope="global", category="rules", content="# Rules\n"), LOCAL_SCOPE, None
            )
        else:
            await memory_endpoints.delete_memory(
                MemoryDeleteRequest(scope="global", category="rules"), LOCAL_SCOPE, None
            )

    assert raised.value.status_code == 503


@pytest.mark.parametrize("change", ["write", "delete"])
def test_a_lock_file_open_failure_answers_busy_not_500(tmp_path, monkeypatch, change):
    """ENOSPC/EMFILE etc. from creating the lock file must not reach the caller
    as a bare, unhandled OSError."""
    store = GlobalMemoryStore(root=tmp_path)
    store.write(MemorySlot.RULES, "# Rules\n- Kept\n")

    from cowork.common import paths as paths_mod

    real_dir_open = paths_mod.dir_open

    def failing_dir_open(root, name, flags, mode=0o600):
        if name.endswith(".lock"):
            raise OSError(24, "Too many open files")
        return real_dir_open(root, name, flags, mode)

    monkeypatch.setattr(store_mod, "dir_open", failing_dir_open)

    with pytest.raises(MemorySlotBusyError):
        if change == "write":
            store.write(MemorySlot.RULES, "# Rules\n")
        else:
            store.delete(MemorySlot.RULES)

    assert store.read(MemorySlot.RULES) == "# Rules\n- Kept\n"
