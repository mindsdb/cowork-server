"""cowork.common.ui_locale: the app's language header -> a supported locale (ENG-4149)."""
import pytest

from cowork.common.ui_locale import normalize_ui_locale, reply_language_line, with_reply_language


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("en", "en"),
        ("es", "es"),
        ("pt-BR", "pt-BR"),
        ("pt-PT", "pt-BR"),
        ("pt", "pt-BR"),
        ("es-MX", "es"),
        ("EN_us", "en"),
        ("nl-NL, es;q=0.8", "es"),  # nl unsupported: the best supported one wins
        ("es;q=0.5, pt-BR;q=0.9", "pt-BR"),
        ("*, es;q=0.4", "es"),
        ("ja", "ja"),
        ("ja-JP", "ja"),
        ("zh-TW, ja;q=0.9", "ja"),  # Traditional Chinese unsupported
        ("zh-CN", "zh-CN"),
        ("zh", "zh-CN"),
        ("zh_SG", "zh-CN"),
        ("zh-Hans-HK", "zh-CN"),
        ("it-IT", "it"),
        ("it-CH", "it"),
        ("de-CH", "de"),
        ("fr-CA", "fr"),
        ("ko-KR", "ko"),
    ],
)
def test_normalize_supported(header, expected):
    assert normalize_ui_locale(header) == expected


@pytest.mark.parametrize(
    "header",
    [None, "", "nl", "ru-RU", "zh-TW", "zh-HK", "zh-Hant", "*", "es;q=abc", "es;q=2", "x" * 40, 123],
)
def test_normalize_unsupported_or_malformed(header):
    assert normalize_ui_locale(header) is None


def test_q_zero_means_not_acceptable():
    assert normalize_ui_locale("es;q=0, nl") is None


def test_header_length_is_bounded():
    assert normalize_ui_locale("es" + "," * 5000 + "pt") == "es"


def test_reply_language_line():
    assert "Spanish (es)" in reply_language_line("es")
    assert "Reply in the language the user writes in" in reply_language_line("pt-BR")
    assert reply_language_line(None) == ""
    assert reply_language_line("nl") == ""
    assert "Simplified Chinese (zh-CN)" in reply_language_line("zh-CN")
    assert "Japanese (ja)" in reply_language_line("ja")


def test_with_reply_language_appends_last_and_is_noop_without_locale():
    assert with_reply_language("ROLE", None) == "ROLE"
    out = with_reply_language("ROLE", "pt-BR")
    assert out.startswith("ROLE\n\n") and out.endswith("reply in Brazilian Portuguese.")


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"X-Cowork-UI-Locale": "es"}, "es"),
        ({"x-cowork-ui-locale": "zh-CN", "Accept-Language": "de-DE,de;q=0.9"}, "zh-CN"),
        # A browser's own language is not the app's setting: an older build or
        # an API caller that only sends Accept-Language gets no language line.
        ({"Accept-Language": "de-DE,de;q=0.9,en;q=0.8"}, None),
        ({}, None),
    ],
)
def test_the_turn_reads_only_the_apps_own_header(monkeypatch, headers, expected):
    import asyncio
    from types import SimpleNamespace

    from starlette.requests import Request

    from cowork.api.v1.endpoints import responses as responses_ep

    built: list[str | None] = []

    class _Handler:
        def __init__(self, *, principal, mindshub_bearer, ui_locale=None):
            built.append(ui_locale)
            raise RuntimeError("stop here")

    async def _admit(**_kwargs) -> None:
        return None

    monkeypatch.setattr(responses_ep, "ResponsesHandler", _Handler)
    monkeypatch.setattr(responses_ep, "_refuse_while_answering", _admit)
    request = Request({
        "type": "http", "method": "POST", "path": "/v1/responses",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    })
    with pytest.raises(RuntimeError):
        asyncio.run(responses_ep.responses(SimpleNamespace(conversation=None), request, None))
    assert built == [expected]
