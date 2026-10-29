"""The prompt section for skills an organization applies on every turn.

A mirror of anton's ``anton.core.memory.always_on_skills`` (ENG-3787): the
cloud pod renders this section from the request's ``always_on`` skills, and an
in-process turn renders it here, so both paths put the same text in the prompt.
The format is a contract with anton; change the two together. Kept local
because the anton-agent this server pins predates the helper.
"""

from __future__ import annotations

ALWAYS_ON_SKILL_MAX_CHARS = 6000
ALWAYS_ON_TOTAL_MAX_CHARS = 24000

_HEADER = (
    "\n\n## Organization and team skills (always apply)\n"
    "These are how this organization works. Follow them on every task where "
    "they apply; they override general defaults but never the user's explicit "
    "instructions in this conversation."
)
_TRUNCATED = "\n[…truncated; the full skill is available on demand]"


def _truncate(body: str) -> str:
    """``body`` cut to the per-skill cap at the last line break, unmarked. A
    single line longer than the cap is cut at the cap."""
    if len(body) <= ALWAYS_ON_SKILL_MAX_CHARS:
        return body
    head = body[:ALWAYS_ON_SKILL_MAX_CHARS]
    cut = head.rfind("\n")
    return (head[:cut] if cut > 0 else head).rstrip()


def render_always_on_skills(skills: list[dict]) -> str:
    """The always-apply section for ``skills``, or "" when there are none.

    ``skills`` is ``[{"name": str, "body": str}]`` in priority order. Each body
    counts against the total after its cut and before the truncation marker.
    The first skill that no longer fits, and every one after it however small,
    is left out and named on a closing line instead: it is still there to
    recall.
    """
    blocks: list[str] = []
    omitted: list[str] = []
    used = 0
    for skill in skills:
        name, body = skill["name"], skill["body"]
        text = _truncate(body)
        if omitted or used + len(text) > ALWAYS_ON_TOTAL_MAX_CHARS:
            omitted.append(name)
            continue
        used += len(text)
        if text != body:
            text += _TRUNCATED
        blocks.append(f"### {name}\n{text}")
    if not blocks and not omitted:
        return ""
    section = _HEADER + "".join(f"\n\n{block}" for block in blocks)
    if omitted:
        section += f"\n\n(Also available on demand: {', '.join(omitted)}.)"
    return section
