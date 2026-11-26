"""find_doc_folder: which doc a tool's ref names, and how much of the project it reads to find it."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from anton.core.artifacts import ArtifactStore

from cowork.docs import store
from cowork.docs.model import DocError
from cowork.docs.service import create_doc
from cowork.docs.store import find_doc_folder


@pytest.fixture
def base(tmp_path):
    return tmp_path / "artifacts"


@pytest.fixture
def project(base):
    """Two docs among plain artifacts."""
    plain = ArtifactStore(base)
    for i in range(5):
        made = plain.create(name=f"Chart {i}", description="", type="html-app", primary="index.html")
        (plain.folder_for(made.slug) / "index.html").write_text("<p>chart</p>")
    return create_doc(base, title="Plan", markdown="x"), create_doc(base, title="Budget", markdown="x")


@pytest.fixture
def reads(monkeypatch):
    """The folders whose metadata.json is read, in order."""
    seen: list[str] = []
    read = store._meta_in

    def counting(d):
        seen.append(d.path.name)
        return read(d)

    monkeypatch.setattr(store, "_meta_in", counting)
    return seen


def _refused(base: Path, ref: str) -> DocError:
    with pytest.raises(DocError) as refused:
        find_doc_folder(base, ref)
    return refused.value


def test_a_slug_reads_only_its_own_folder(project, base, reads):
    plan, _ = project
    assert find_doc_folder(base, plan.slug) == plan.folder
    assert reads == [plan.slug]


@pytest.mark.parametrize("spell", [str, lambda i: str(uuid.UUID(i)), str.upper, lambda i: str(uuid.UUID(i)).upper()])
def test_an_id_reads_only_the_folder_its_slug_names(project, base, reads, spell):
    plan, _ = project  # listed after the charts, so a scan would read them first
    assert find_doc_folder(base, spell(plan.artifact_id)) == plan.folder
    assert reads == [plan.slug]


def test_folders_sharing_an_id_prefix_are_told_apart_by_the_whole_id(project, base, reads):
    plan, budget = project
    prefix = plan.artifact_id[:8]
    twin_id = prefix + "0" * 24  # a uuid4's version digit keeps it from being the plan's
    meta_path = budget.folder / "metadata.json"
    meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), "id": twin_id}))
    twin = budget.folder.rename(base / f"aaa-{prefix}")  # listed before the plan's folder

    assert find_doc_folder(base, plan.artifact_id) == plan.folder
    assert find_doc_folder(base, twin_id) == twin
    assert reads == [twin.name, plan.slug, twin.name]


@pytest.mark.parametrize("ref", ["-", "---"])
def test_a_ref_of_dashes_is_no_id_even_for_a_doc_without_metadata(project, base, ref):
    plan, _ = project
    (plan.folder / "metadata.json").unlink()
    assert find_doc_folder(base, plan.slug) == plan.folder
    assert _refused(base, ref).code == "no_doc"


@pytest.mark.parametrize("renamed", ["legacy-budget", "budget-{prefix}-2"])
def test_an_id_its_folder_name_doesnt_end_in_is_still_found(project, base, renamed):
    """Older artifacts' slugs carry no id, and a slug made unique carries a counter after it."""
    _, budget = project
    folder = budget.folder.rename(base / renamed.format(prefix=budget.artifact_id[:8]))
    assert find_doc_folder(base, budget.artifact_id) == folder


def test_a_title_is_matched_whatever_its_case(project, base):
    plan, _ = project
    assert find_doc_folder(base, "  pLAN ") == plan.folder


def test_a_title_that_looks_like_a_path_is_still_a_title(base):
    made = create_doc(base, title="Q1/Q2 ../plan", markdown="x")
    assert find_doc_folder(base, "q1/q2 ../PLAN") == made.folder


def test_two_docs_with_one_title_are_ambiguous(project, base):
    create_doc(base, title="plan", markdown="x")
    refused = _refused(base, "Plan")
    assert refused.code == "ambiguous"
    assert str(refused) == "2 docs are titled 'Plan'; pass the id from list_docs."


def test_a_slug_wins_over_another_docs_title(project, base):
    plan, _ = project
    create_doc(base, title=plan.slug, markdown="x")
    assert find_doc_folder(base, plan.slug) == plan.folder


def test_a_slug_wins_over_another_docs_id(project, base):
    plan, budget = project
    named_as_id = plan.folder.rename(base / budget.artifact_id)
    assert find_doc_folder(base, budget.artifact_id) == named_as_id


def test_an_unknown_ref_and_an_empty_project_say_so(base):
    assert str(_refused(base, "Plan")) == "no doc 'Plan': this project has no artifacts yet."
    create_doc(base, title="Plan", markdown="x")
    refused = _refused(base, "Nope")
    assert refused.code == "no_doc"
    assert str(refused) == "no doc 'Nope' in this project; list_docs shows the docs that exist."
    assert str(_refused(base, "  ")) == "name the doc: its id, slug or title (list_docs shows them)."


def test_an_artifact_that_isnt_a_doc_is_not_found(project, base):
    chart = ArtifactStore(base).create(name="Sales", description="", type="html-app", primary="index.html")
    for ref in (chart.slug, chart.id, chart.name):
        assert _refused(base, ref).code == "no_doc"


def test_a_doc_in_a_dot_folder_is_not_found(project, base):
    plan, _ = project
    hidden = plan.folder.rename(base / f".hidden-{plan.artifact_id[:8]}")
    for ref in (hidden.name, plan.artifact_id, "Plan"):
        assert _refused(base, ref).code == "no_doc"


def test_a_linked_folder_is_not_a_doc(base, tmp_path):
    elsewhere = create_doc(tmp_path / "elsewhere", title="Elsewhere", markdown="x")
    base.mkdir()
    link = base / f"linked-{elsewhere.artifact_id[:8]}"
    link.symlink_to(elsewhere.folder, target_is_directory=True)
    for ref in (link.name, elsewhere.artifact_id, "Elsewhere"):
        assert _refused(base, ref).code == "no_doc"


def test_a_ref_never_reaches_outside_the_artifacts_folder(project, base, tmp_path):
    plan, _ = project
    outside = create_doc(tmp_path / "outside", title="Outside", markdown="x")
    hostile = [
        "..", ".", "/", str(outside.folder), f"../outside/{outside.slug}", outside.slug,
        f"{plan.slug}/", f"./{plan.slug}", f"{plan.slug}/../{plan.slug}", f"{plan.slug}\x00", "\x00",
    ]
    for ref in hostile:
        assert _refused(base, ref).code == "no_doc", ref
