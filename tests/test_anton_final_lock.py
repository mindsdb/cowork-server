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
