"""The website access mode: anton's contract this server relies on, the
mode's breadth, and the access a website publish may carry."""

from __future__ import annotations


def test_the_installed_anton_carries_the_website_mode():
    """What publish_artifact relies on: anton keeps `website` through
    resolve_access and a kept entry, never stores confirmed_by_owner, and
    sends confirmed_by_owner only when the caller's access carries it, never
    from the previous entry, which the pod can write."""
    from anton.publish_access import WEBSITE_MODE, access_from_owner_side, resolve_access
    from anton.publisher import build_access_payload

    assert WEBSITE_MODE == "website"
    effective, _pwd_version, _access_version, owner_side = resolve_access(None, {"mode": "website"}, None)
    assert effective == {"mode": "website"}
    assert owner_side["mode"] == "website"
    assert "confirmed_by_owner" not in owner_side
    assert access_from_owner_side({"mode": "website", "confirmed_by_owner": True}) == {"mode": "website"}
    assert build_access_payload({"mode": "website"}) == {"mode": "website"}
    assert build_access_payload({"mode": "website", "confirmed_by_owner": True}) == {
        "mode": "website", "confirmed_by_owner": True,
    }
    assert build_access_payload(
        {"mode": "website"}, previous={"mode": "website", "confirmed_by_owner": True},
    ) == {"mode": "website"}
