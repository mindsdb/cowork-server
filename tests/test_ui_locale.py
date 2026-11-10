"""cowork.common.ui_locale: Accept-Language -> the app's language (ENG-4149)."""
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
        ("fr-FR, es;q=0.8", "es"),  # fr unsupported: the best supported one wins
        ("es;q=0.5, pt-BR;q=0.9", "pt-BR"),
        ("*, es;q=0.4", "es"),
    ],
)
def test_normalize_supported(header, expected):
    assert normalize_ui_locale(header) == expected


@pytest.mark.parametrize(
    "header",
    [None, "", "fr", "de-DE", "*", "es;q=abc", "es;q=2", "x" * 40, 123],
)
def test_normalize_unsupported_or_malformed(header):
    assert normalize_ui_locale(header) is None


def test_q_zero_means_not_acceptable():
    assert normalize_ui_locale("es;q=0, fr") is None


def test_header_length_is_bounded():
    assert normalize_ui_locale("es" + "," * 5000 + "pt") == "es"


def test_reply_language_line():
    assert "Spanish (es)" in reply_language_line("es")
    assert "Reply in the language the user writes in" in reply_language_line("pt-BR")
    assert reply_language_line(None) == ""
    assert reply_language_line("fr") == ""


def test_with_reply_language_appends_last_and_is_noop_without_locale():
    assert with_reply_language("ROLE", None) == "ROLE"
    out = with_reply_language("ROLE", "pt-BR")
    assert out.startswith("ROLE\n\n") and out.endswith("reply in Brazilian Portuguese.")
