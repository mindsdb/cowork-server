"""Conformance fixtures for the server/Electron "same host" contract.

The grant domain is the PSL-registrable-or-exact host Electron computes
(tldts `getDomain`, allowPrivateDomains: true) and reports in bridge hello;
the server reduces candidate hosts through the SAME PSL function
(`registrable_host`, tldextract with private domains, offline) and matches
by equality (`host_matches_grant`). This table is mirrored VERBATIM in the
TypeScript repo — both sides must accept/refuse exactly the same set. Do
not change the expected values here without changing them there.
"""
from __future__ import annotations

import pytest

from cowork.schemas.browser import host_matches_grant, host_only, registrable_host

# (url, grant, expected_match) — mirrored verbatim in the TypeScript repo.
CONFORMANCE_TABLE = [
    ("https://shop.example.com/a?x=1", "example.com", True),
    ("https://example.com", "example.com", True),
    ("https://app.bank.co.uk/login", "bank.co.uk", True),
    ("https://other.co.uk/", "bank.co.uk", False),
    ("https://bank.co.uk.evil.com/", "bank.co.uk", False),
    ("https://foo.github.io/docs", "foo.github.io", True),
    ("https://bar.github.io", "foo.github.io", False),
    ("http://[::1]:8080/x", "::1", True),
    ("http://192.168.0.1/x", "192.168.0.1", True),
    ("http://localhost:3000/x", "localhost", True),
    ("https://user:pass@sub.example.com/", "example.com", True),
    ("HTTPS://WWW.EXAMPLE.COM", "example.com", True),
    ("https://notexample.com", "example.com", False),
    # A grant that is ITSELF a public/private suffix (only arises when the
    # approved tab is literally at that host — getDomain returns null and
    # Electron falls back to the exact host) matches only that exact host:
    # foo.github.io's registrable host is foo.github.io ≠ github.io.
    ("https://foo.github.io/docs", "github.io", False),
    ("https://bank.co.uk/login", "co.uk", False),
]


@pytest.mark.parametrize("url,grant,expected", CONFORMANCE_TABLE)
def test_host_matches_grant_conformance(url, grant, expected):
    assert host_matches_grant(url, grant) is expected


@pytest.mark.parametrize("url,grant,expected", CONFORMANCE_TABLE)
def test_host_matches_grant_conformance_prenormalized(url, grant, expected):
    # Matching an already-normalized host gives the same verdict — the
    # matcher normalizes both sides itself.
    assert host_matches_grant(host_only(url), grant) is expected


# ── host_only cases ────────────────────────────────────────────────────
HOST_ONLY_TABLE = [
    ("http://[::1]:8080/x", "::1"),
    ("https://user:pass@Sub.Example.COM:8443/p?q#f", "sub.example.com"),
    # host_only returns the bare hostname — NOT a PSL registrable domain.
    ("sub.example.co.uk", "sub.example.co.uk"),
]


@pytest.mark.parametrize("value,expected", HOST_ONLY_TABLE)
def test_host_only(value, expected):
    assert host_only(value) == expected


@pytest.mark.parametrize(
    "value",
    [v for v, _ in HOST_ONLY_TABLE]
    + [u for u, _, _ in CONFORMANCE_TABLE]
    + [g for _, g, _ in CONFORMANCE_TABLE],
)
def test_host_only_is_idempotent(value):
    # Digest validation and telemetry rely on `host_only(v) != v` as an
    # "already normalized" check, so a second pass must be a no-op — in
    # particular a bare IPv6 literal (`::1`) must not be re-mangled.
    once = host_only(value)
    assert host_only(once) == once
    assert host_only(host_only(once)) == once


# ── registrable_host vs host_only ─────────────────────────────────────
@pytest.mark.parametrize(
    "value,registrable,bare",
    [
        # registrable_host reduces to the PSL registrable domain; host_only
        # keeps the full hostname — persisted values always use host_only,
        # matching only ever uses registrable_host.
        ("shop.example.com", "example.com", "shop.example.com"),
        ("sub.example.co.uk", "example.co.uk", "sub.example.co.uk"),
        ("https://foo.github.io/docs", "foo.github.io", "foo.github.io"),
        # PSL yields nothing → fall back to the bare host (exactly like
        # Electron's getDomain-null fallback).
        ("localhost", "localhost", "localhost"),
        ("192.168.0.1", "192.168.0.1", "192.168.0.1"),
        ("http://[::1]:8080/x", "::1", "::1"),
        ("github.io", "github.io", "github.io"),
        ("co.uk", "co.uk", "co.uk"),
    ],
)
def test_registrable_host_vs_host_only(value, registrable, bare):
    assert registrable_host(value) == registrable
    assert host_only(value) == bare


def test_registrable_host_empty():
    assert registrable_host("") == ""


def test_host_matches_grant_empty_never_matches():
    # An empty grant must never match everything; an empty host never
    # matches anything.
    assert host_matches_grant("https://example.com", "") is False
    assert host_matches_grant("", "example.com") is False
    assert host_matches_grant("", "") is False
