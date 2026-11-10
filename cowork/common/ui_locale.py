"""The user's UI language, read from the request's Accept-Language (ENG-4149).

The Cowork app sends its active language on every request as
``Accept-Language`` (one of ``en``, ``es``, ``pt-BR``). A turn passes the
normalized tag to anton as ``ChatSessionConfig.ui_locale`` (the in-process
path) or as the queued job's ``ui_locale`` param (the remote path), and anton
adds one language line to its system prompt. The fast responders that answer
before anton (the routing gate, the direct answer and the first take) add the
same rule through :func:`reply_language_line`.

The line never forces a language: the user's own message wins, and the UI
language only decides when the message does not say.
"""
from __future__ import annotations

#: Supported UI locales and the English name the prompts use for each. Kept in
#: step with anton's ``anton.core.llm.ui_locale.UI_LOCALE_NAMES``; a tag anton
#: does not know is dropped there, so a mismatch only loses the line.
UI_LOCALE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt-BR": "Brazilian Portuguese",
}

#: Primary language subtag -> supported tag. Every Portuguese variant maps to
#: pt-BR, the only Portuguese the app ships.
_BY_PRIMARY = {"en": "en", "es": "es", "pt": "pt-BR"}

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
    """The supported UI locale an ``Accept-Language`` value asks for, else None.

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
        locale = _BY_PRIMARY.get(tag.replace("_", "-").split("-", 1)[0].lower())
        if locale is None:
            continue
        best, best_q = locale, q
    return best


def reply_language_line(ui_locale: str | None) -> str:
    """The language rule for a prompt, or "" when there is no supported locale.

    Same wording as anton's ``ui_locale_line``, for the fast responders that
    run without an anton session."""
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


__all__ = ["UI_LOCALE_NAMES", "normalize_ui_locale", "reply_language_line", "with_reply_language"]
