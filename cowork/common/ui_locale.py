"""The user's UI language, read from the app's own request header (ENG-4149).

The Cowork app sends the language it is showing as ``X-Cowork-UI-Locale``
(one of ``en``, ``es``, ``pt-BR``, ``de``, ``fr``, ``it``, ``ja``, ``ko``,
``zh-CN``), and only while its language picker is on. Not ``Accept-Language``:
browsers and Electron send that on their own from the OS or browser language,
so older app builds, web bundles and API callers would all look as if they had
picked a language. A request without the header leaves every prompt as it was.

A turn passes the normalized tag to anton as ``ChatSessionConfig.ui_locale``
(the in-process path) or as the queued job's ``ui_locale`` param (the remote
path, relayed to the pod by scratchpad-controller), and anton adds one
language line to its system prompt. The fast responders that answer before
anton (the routing gate, the direct answer and the first take) add the rule
through :func:`reply_language_line`.

The line never forces a language: the user's own message wins, and the UI
language only decides when the message does not say.
"""
from __future__ import annotations

#: The header the app sets; read by the responses endpoint.
UI_LOCALE_HEADER = "X-Cowork-UI-Locale"

#: Supported UI locales and the English name the prompts use for each. A copy
#: of anton's ``anton.core.llm.ui_locale.UI_LOCALE_NAMES`` until the anton floor
#: in pyproject.toml includes that module (mindsdb/anton#776); then this file
#: imports it and keeps only the header parsing. A tag anton does not know is
#: dropped there, so a mismatch only loses the line; tests/test_ui_locale.py
#: compares the two whenever the installed anton has the module.
UI_LOCALE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt-BR": "Brazilian Portuguese",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Simplified Chinese",
}

#: Primary language subtag -> supported tag. Every Portuguese variant maps to
#: pt-BR, the only Portuguese the app ships. Chinese is handled in
#: :func:`_supported_tag`, since only Simplified is shipped.
_BY_PRIMARY = {"en": "en", "es": "es", "pt": "pt-BR", "de": "de", "fr": "fr", "it": "it", "ja": "ja", "ko": "ko"}

#: Subtags that mark a Chinese tag as Traditional, which the app does not ship.
_TRADITIONAL_ZH = {"hant", "tw", "hk", "mo"}


def _supported_tag(tag: str) -> str | None:
    """The supported locale a single language tag maps to, else None.

    Chinese maps to ``zh-CN`` unless it asks for Traditional script or a
    Traditional region (``zh-TW``, ``zh-HK``, ``zh-Hant``); same rule as the
    app's ``normalizeLocale``."""
    parts = tag.replace("_", "-").lower().split("-")
    if parts[0] == "zh":
        if "hans" in parts or not _TRADITIONAL_ZH.intersection(parts):
            return "zh-CN"
        return None
    return _BY_PRIMARY.get(parts[0])

#: Bounds on what is parsed: a real header is a few short tags.
_MAX_HEADER_CHARS = 1024
_MAX_TAG_CHARS = 35


def _quality(params: list[str]) -> float | None:
    """The ``q`` weight among a tag's parameters: 1.0 when absent, None when malformed."""
    for param in params:
        name, _, value = param.partition("=")
        if name.strip().lower() != "q":
            continue
        try:
            q = float(value.strip())
        except ValueError:
            return None
        return q if 0.0 <= q <= 1.0 else None
    return 1.0


def normalize_ui_locale(accept_language: str | None) -> str | None:
    """The supported UI locale a header value names, else None.

    The app sends one tag; an ``Accept-Language``-style list with q-weights is
    read too.

    Takes the supported tag with the highest q-weight (the first one on a tie),
    mapped by its primary subtag: ``es-*`` -> ``es``, ``pt-*`` -> ``pt-BR``,
    ``en-*`` -> ``en``. Unsupported languages and ``*`` are skipped, so
    ``fr, es;q=0.8`` gives ``es``. No supported language, a malformed or a
    missing header gives None. Never raises: the header is client input.
    """
    if not accept_language or not isinstance(accept_language, str):
        return None
    best: str | None = None
    best_q = 0.0
    for item in accept_language[:_MAX_HEADER_CHARS].split(","):
        tag, *params = item.split(";")
        tag = tag.strip()
        if not tag or len(tag) > _MAX_TAG_CHARS:
            continue
        q = _quality(params)
        if q is None or q <= best_q:
            continue
        # "*" and languages the app does not ship never shadow one it does.
        locale = _supported_tag(tag)
        if locale is None:
            continue
        best, best_q = locale, q
    return best


def reply_language_line(ui_locale: str | None) -> str:
    """The language rule for a prompt, or "" when there is no supported locale.

    For the fast responders that run without an anton session. Same rule as
    anton's ``ui_locale_line``, which adds a ``## Language`` heading and a
    sentence about tool step descriptions: the agent writes those, the fast
    responders do not."""
    name = UI_LOCALE_NAMES.get(ui_locale or "")
    if name is None:
        return ""
    return (
        f"The user's app is set to {name} ({ui_locale}). Reply in the language the "
        f"user writes in; if that is unclear (e.g. a very short message, or only data "
        f"or code), reply in {name}."
    )


def with_reply_language(system: str, ui_locale: str | None) -> str:
    """``system`` with the language rule at its end; unchanged without a locale.

    At the end so the role prompt before it stays one cached prefix."""
    line = reply_language_line(ui_locale)
    return f"{system}\n\n{line}" if line else system


__all__ = ["UI_LOCALE_HEADER", "UI_LOCALE_NAMES", "normalize_ui_locale", "reply_language_line", "with_reply_language"]
