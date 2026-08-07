"""fetch_minds_models: the picker's grouping metadata (`provider` / `family`).

MindsHub's `/v1/models` publishes who makes each model and which moving alias it
belongs to. Those two fields are what let the app group the picker into vendor
sections and mark the aliases whose version moves, so this is the seam where a
gateway change silently empties a section — hence direct coverage.

Embedding filtering and force_refresh have their own files; the failure-path
tests here exist because this change turned the return value into a NamedTuple,
and a failure path returning the wrong arity would otherwise only surface as an
attribute error inside the endpoint.

No network: httpx.AsyncClient is replaced with a fake that serves a canned body.
"""
import asyncio

import pytest

import cowork.services.providers as providers
from cowork.services.providers import fetch_minds_models

_URL = "https://api.mindshub.example"
_KEY = "mdb_test"


def _row(model_id, **extra):
    row = {"id": model_id, "object": "model", "created": 0}
    row.update(extra)
    return row


class _Resp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


def _serve(monkeypatch, body, status_code=200):
    """Point fetch_minds_models at a canned /models body."""

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return _Resp(body, status_code)

    monkeypatch.setattr(providers.httpx, "AsyncClient", _Client)


@pytest.fixture(autouse=True)
def _clear_cache():
    # The listing is cached per base URL for 5 minutes; each test needs a fetch.
    providers._minds_models_cache.clear()
    yield
    providers._minds_models_cache.clear()


def test_extracts_provider_and_family(monkeypatch):
    _serve(
        monkeypatch,
        {
            "object": "list",
            "data": [
                _row("sonnet", label="Claude Sonnet 5", provider="anthropic", family="sonnet"),
                _row("kimi", label="Kimi K3", provider="moonshot", family="kimi"),
            ],
        },
    )

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    assert listing.providers == {"sonnet": "anthropic", "kimi": "moonshot"}
    assert listing.families == {"sonnet": "sonnet", "kimi": "kimi"}


def test_family_defaults_to_the_id_so_the_map_is_dense(monkeypatch):
    """A row with a provider but no family is its own head, and says so.

    Densifying here is what lets the app read `families[id] == id` as "this alias
    moves" for every model, with no missing-key branch at the render site. Every
    alias in the catalog is such a head today, so this is the common path.
    """
    _serve(monkeypatch, {"data": [_row("grok", provider="xai")]})

    assert asyncio.run(fetch_minds_models(_URL, _KEY)).families == {"grok": "grok"}


def test_a_pinned_row_keeps_the_family_it_names(monkeypatch):
    _serve(
        monkeypatch,
        {
            "data": [
                _row("sonnet", provider="anthropic"),
                _row("sonnet-4-5", provider="anthropic", family="sonnet"),
            ]
        },
    )

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    # The head names itself, the pin names the head: the app tags only the former
    # "latest" and lists the pin underneath it.
    assert listing.families == {"sonnet": "sonnet", "sonnet-4-5": "sonnet"}


def test_a_gateway_that_publishes_no_metadata_yields_empty_maps(monkeypatch):
    """A plain OpenAI-compatible endpoint, or a MindsHub older than these fields.

    Empty (not None, not partially filled) is the signal the app reads as "no
    grouping metadata available", which is what makes it fall back to a single
    ungrouped list instead of inventing an "unknown vendor" section.
    """
    _serve(monkeypatch, {"data": [_row("model-a"), _row("model-b")]})

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    assert listing.ids == ["model-a", "model-b"]
    assert listing.providers == {}
    assert listing.families == {}


# Junk is applied to ONE field at a time on purpose. Giving both the same value
# collapses every case to "both absent" and the assertions become {} == {}, which
# leaves the two interesting branches uncovered: a valid provider with a junk
# family (the only case where the `or model_id` fallback fires on junk input), and
# a junk provider with a valid family.
_JUNK = [None, 42, "", "   ", [], {}, True]


@pytest.mark.parametrize("junk", _JUNK)
def test_junk_provider_is_treated_as_absent(monkeypatch, junk):
    """Anything that isn't a non-empty string reads as "not published".

    Otherwise an unexpected gateway turns `42` into a section heading.
    """
    _serve(monkeypatch, {"data": [_row("model-a", provider=junk)]})

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    assert listing.ids == ["model-a"]
    assert listing.providers == {}
    assert listing.families == {}


@pytest.mark.parametrize("junk", _JUNK)
def test_junk_family_falls_back_to_the_id_when_the_provider_is_valid(monkeypatch, junk):
    """The dense default has to fire on junk input, not just on an absent key.

    A blank `family` is the shape a console edit produces, and it must read as "this
    alias is its own head" rather than as a pin pointing at "".
    """
    _serve(monkeypatch, {"data": [_row("sonnet", provider="anthropic", family=junk)]})

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    assert listing.providers == {"sonnet": "anthropic"}
    assert listing.families == {"sonnet": "sonnet"}


@pytest.mark.parametrize("junk", _JUNK)
def test_a_valid_family_survives_a_junk_provider(monkeypatch, junk):
    """The row is unplaceable in a section, but the family must still be recorded.

    Dropping it makes the row ABSENT from the map, and a consumer reading absent as
    "is its own head" would tag a frozen version "latest" — the one claim it must
    never make.
    """
    _serve(monkeypatch, {"data": [_row("sonnet-4-5", provider=junk, family="sonnet")]})

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    assert listing.providers == {}
    assert listing.families == {"sonnet-4-5": "sonnet"}


def test_a_family_without_a_provider_is_still_recorded(monkeypatch):
    _serve(monkeypatch, {"data": [_row("orphan", family="something")]})

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    # Not dropped: absent would be read as "is its own head" and tagged latest.
    assert listing.families == {"orphan": "something"}
    assert listing.providers == {}


def test_the_pre_existing_fields_survive_the_named_tuple(monkeypatch):
    """Guard on the refactor: ids/enabled/efforts/labels still land where they did."""
    _serve(
        monkeypatch,
        {
            "data": [
                _row(
                    "sonnet",
                    label="Claude Sonnet 5",
                    provider="anthropic",
                    enabled=False,
                    reasoning_efforts=["low", "high"],
                    default_reasoning_effort="high",
                ),
                _row("gpt", provider="openai", enabled=True),
            ]
        },
    )

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    assert listing.ids == ["sonnet", "gpt"]
    assert listing.enabled == {"sonnet": False, "gpt": True}
    assert listing.efforts == {"sonnet": {"efforts": ["low", "high"], "default": "high"}}
    assert listing.labels == {"sonnet": "Claude Sonnet 5"}


@pytest.mark.parametrize(
    "body,status",
    [
        ({"error": "nope"}, 503),          # HTTP error
        ({"object": "list", "data": "x"}, 200),  # body that isn't a list
    ],
)
def test_failure_paths_return_the_empty_listing(monkeypatch, body, status):
    """Every failure path returns one shared value of the right arity.

    ids is None rather than [] — that is what tells the caller to keep the list it
    already holds instead of emptying the picker.
    """
    _serve(monkeypatch, body, status)

    listing = asyncio.run(fetch_minds_models(_URL, _KEY))

    assert listing == providers._EMPTY_LISTING
    assert listing.ids is None


def test_missing_url_or_key_short_circuits(monkeypatch):
    def _explode(*a, **k):
        raise AssertionError("no fetch should be attempted")

    monkeypatch.setattr(providers.httpx, "AsyncClient", _explode)

    assert asyncio.run(fetch_minds_models("", _KEY)) == providers._EMPTY_LISTING
    assert asyncio.run(fetch_minds_models(_URL, "")) == providers._EMPTY_LISTING
