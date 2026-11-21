"""Pick the final anton that replaces a locked release candidate."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("anton_final_lock_under_test", ROOT / "scripts/anton_final_lock.py")
lock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lock)

FILE = [{"yanked": False}]


@pytest.mark.parametrize(
    "lower, higher",
    [
        ("2.26.11.13.1rc3", "2.26.11.13.1"),
        ("2.26.11.13.1rc3", "2.26.11.15.1"),
        ("2.26.11.13.1rc2", "2.26.11.13.1rc3"),
        ("1.0a2", "1.0b1"),
        ("1.0b1", "1.0rc1"),
        ("2.26.9.2.1", "2.26.10.1.1"),
        ("1.0.0rc1", "1.0"),
        ("1!1.0rc1", "1!1.0"),
        ("1.0.dev1", "1.0"),
        ("1.0", "1.0.post1"),
    ],
)
def test_versions_sort_in_pep440_order(lower, higher):
    assert lock.sort_key(lower) < lock.sort_key(higher)


def test_a_newer_final_replaces_the_locked_candidate():
    releases = {
        "2.26.11.9.1": FILE,
        "2.26.11.13.1rc3": FILE,
        "2.26.11.15.1": FILE,
        "2.26.11.16.1rc1": FILE,
    }
    assert lock.replacement("2.26.11.13.1rc3", releases) == "2.26.11.15.1"


def test_the_final_of_the_same_release_counts():
    assert lock.replacement("2.26.11.13.1rc3", {"2.26.11.13.1": FILE}) == "2.26.11.13.1"


@pytest.mark.parametrize("candidate, final", [
    ("1.0.0rc1", "1.0"), ("1!1.0rc1", "1!1.0"), ("1.0.dev1", "1.0"),
])
def test_pep440_final_replaces_candidate(candidate, final):
    assert lock.replacement(candidate, {final: FILE}) == final


def test_post_release_ordering_selects_newest_final():
    assert lock.replacement("1.0rc1", {"1.0.post2": FILE, "1.0.post1": FILE}) == "1.0.post2"


def test_no_replacement_while_only_older_finals_exist():
    releases = {"2.26.11.9.1": FILE, "2.26.11.13.1rc3": FILE, "2.26.11.16.1rc1": FILE}
    assert lock.replacement("2.26.11.13.1rc3", releases) is None


def test_a_locked_final_is_left_alone():
    assert lock.replacement("2.26.11.15.1", {"2.26.11.16.1": FILE}) is None


@pytest.mark.parametrize("files", [[], [{"yanked": True}]])
def test_a_final_with_nothing_installable_is_skipped(files):
    assert lock.replacement("2.26.11.13.1rc3", {"2.26.11.15.1": files}) is None


def test_the_locked_version_is_read_from_the_export():
    export = "aiohttp==3.14.0 \\\n    --hash=sha256:a\nanton-agent==2.26.11.13.1rc3 \\\n    --hash=sha256:b\n"
    assert lock.locked_version(export) == "2.26.11.13.1rc3"


PYPROJECT = (
    '[project]\ndependencies = [\n'
    '    # 2.26.11.20.1rc3 is the first with the feature.\n'
    '    "anton-agent>={floor},<3",\n'
    '    "other>=1",\n]\n'
)


def test_a_candidate_floor_is_raised_to_the_final():
    raised = lock.raise_floor(PYPROJECT.format(floor="2.26.11.20.1rc3"), "2.26.11.20.1")
    assert raised == PYPROJECT.format(floor="2.26.11.20.1")


def test_a_final_floor_is_left_alone():
    text = PYPROJECT.format(floor="2.26.11.15.1")
    assert lock.raise_floor(text, "2.26.11.20.1") == text


@pytest.mark.parametrize("text", [
    '[project]\ndependencies = ["other>=1"]\n',
    '[project]\ndependencies = ["anton-agent>=1.0rc1"]\n[project.optional-dependencies]\nx = ["anton-agent>=1.0rc1"]\n',
])
def test_the_floor_must_be_found_exactly_once(text):
    with pytest.raises(ValueError, match="Expected one anton-agent>= floor"):
        lock.raise_floor(text, "1.0")


ALLOWANCE = (
    '[project]\ndependencies = ["anton-agent>=1.0rc1,<3"]\n\n'
    '[tool.cowork.release.allow-prerelease]\n'
    '# Removed by the relock.\n'
    'other = "2.0b1"\n'
    '{entry}'
    'later = "3.0rc1"\n\n'
    '[tool.other]\n'
    'anton-agent = "kept"\n'
)


def test_the_relock_drops_only_anton_agents_allowance():
    dropped = lock.drop_allowance(ALLOWANCE.format(entry='anton-agent = "1.0rc1"\n'))
    assert dropped == ALLOWANCE.format(entry="")


def test_no_allowance_is_left_alone():
    text = ALLOWANCE.format(entry="")
    assert lock.drop_allowance(text) == text


def test_an_allowance_the_relock_cannot_remove_is_refused():
    with pytest.raises(ValueError, match="Could not remove anton-agent's entry"):
        lock.drop_allowance(ALLOWANCE.format(entry='"anton-agent" = "1.0rc1"\n'))


def test_a_final_below_the_candidate_floor_is_refused():
    with pytest.raises(ValueError, match="not newer than the floor"):
        lock.raise_floor(PYPROJECT.format(floor="2.0rc1"), "1.9")


@pytest.mark.parametrize("value, expected", [
    ("2.26.11.15.1", "2.26.11.15.1"), ("", None), ("2.26.11.16.1rc1", None), ("not-a-version", None),
])
def test_only_a_final_announcement_is_waited_for(value, expected):
    assert lock.announced_final(value) == expected


def test_waits_until_pypi_lists_the_announced_final():
    listings = iter([{"2.26.11.9.1": FILE}, {"2.26.11.9.1": FILE}, {"2.26.11.9.1": FILE, "2.26.11.15.1": FILE}])
    sleeps = []
    releases = lock.releases_listing("2.26.11.15.1", fetch=lambda: next(listings), sleep=sleeps.append,
                                     attempts=5, interval=1)
    assert "2.26.11.15.1" in releases
    assert sleeps == [1, 1]


def test_no_announcement_reads_pypi_once():
    calls = []
    def fetch():
        calls.append(1)
        return {"2.26.11.9.1": FILE}
    lock.releases_listing(None, fetch=fetch, sleep=lambda _: None)
    assert len(calls) == 1


def test_gives_up_when_pypi_never_lists_the_announced_final():
    with pytest.raises(RuntimeError, match="periodic run will retry"):
        lock.releases_listing("2.26.11.15.1", fetch=lambda: {"2.26.11.15.1": [{"yanked": True}]},
                              sleep=lambda _: None, attempts=3, interval=0)
