"""Publication status - not the shape of the key - picks the comments backend.

Since the stable-key publish flow, `artifact/<uuid>` addresses BOTH the local
journal and the cloud service, so the router has to read the publication record
to tell them apart.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cowork.api.v1.endpoints import comments as router

UUID = "e9267de0-3fa4-4c17-b964-dc63216311cf"
STABLE_KEY = f"artifact/{UUID}"


@pytest.fixture
def artifact(tmp_path: Path, monkeypatch):
    """An artifact folder whose `.published.json` the test can rewrite."""
    folder = tmp_path / "ascii-e9267de0"
    folder.mkdir()

    state: dict = {"entry": {}}
    monkeypatch.setattr(router, "artifacts_sources_for_scan", lambda: [])
    monkeypatch.setattr(
        router, "resolve_artifact_folder", lambda _sources, _aid: (object(), folder, {})
    )
    monkeypatch.setattr(router, "published_owner_state", lambda _path: state["entry"])
    monkeypatch.setattr(router, "_org_mode", lambda: False)
    return state


def _restricted(**overrides) -> dict:
    entry = {
        "report_id": "c675003f",
        "url": "https://view.dev.mindshub.ai/view/b9996ebec/c675003f",
        "published": True,
        "mode": "restricted",
    }
    entry.update(overrides)
    return entry


def test_unpublished_artifact_routes_to_the_local_journal(artifact):
    assert router.resolve_comments_route("artifact", UUID) is None


def test_public_publication_routes_to_the_local_journal(artifact):
    artifact["entry"] = _restricted(mode="public")
    assert router.resolve_comments_route("artifact", UUID) is None


def test_restricted_publication_routes_to_the_composite_scope(artifact):
    artifact["entry"] = _restricted()
    assert router.resolve_comments_route("artifact", UUID) == ("b9996ebec", "c675003f")


def test_restricted_publication_prefers_the_lambda_echo(artifact):
    artifact["entry"] = _restricted(lambda_artifact_key=STABLE_KEY)
    assert router.resolve_comments_route("artifact", UUID) == ("artifact", UUID)


def test_unresolvable_artifact_routes_to_the_local_journal(artifact, monkeypatch):
    def boom(_sources, _aid):
        raise FileNotFoundError("Artifact not found")

    monkeypatch.setattr(router, "resolve_artifact_folder", boom)
    # The local handler raises the user-facing 404; the router must not turn an
    # unknown id into an upstream call carrying the caller's credential.
    assert router.resolve_comments_route("artifact", UUID) is None


def test_composite_key_is_forwarded_unchanged(artifact):
    # An OTA renderer can lag a release behind and still hold a composite key
    # from an older card. It names a real published artifact, so it keeps being
    # proxied rather than resolved against the local roots (where it is not an
    # artifact id at all and would 404).
    assert router.resolve_comments_route("alice", "rep123") == ("alice", "rep123")


def test_org_mode_always_proxies_under_the_incoming_key(artifact, monkeypatch):
    monkeypatch.setattr(router, "_org_mode", lambda: True)
    assert router.resolve_comments_route("artifact", UUID) == ("artifact", UUID)


def test_org_mode_ignores_the_publication_record(artifact, monkeypatch):
    # Drafts and publications share one scope in org mode; that must not change.
    monkeypatch.setattr(router, "_org_mode", lambda: True)
    artifact["entry"] = _restricted()
    assert router.resolve_comments_route("artifact", UUID) == ("artifact", UUID)
