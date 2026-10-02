"""Check the direct-OpenAI model catalog against the live API.

    OPENAI_API_KEY=sk-... uv run python scripts/check_openai_catalog.py

For every model the picker offers: is it in GET /v1/models, does each effort
level in DIRECT_EFFORT_CATALOG get accepted on both /v1/chat/completions and
/v1/responses, is "minimal" rejected, and does the effort the API reports when
none is sent match the catalog's default. One tiny call per (model, level,
endpoint); a full run costs a few cents.
"""
import os
import sys

from openai import BadRequestError, NotFoundError, OpenAI

from cowork.common.settings.app_settings import (
    DIRECT_EFFORT_CATALOG,
    MODEL_ROLE_DEFAULTS,
    RECOMMENDED_MODELS,
)

PROMPT = "Reply with the single word: ok"


def _chat(client: OpenAI, model: str, effort: str | None):
    kw = {"reasoning_effort": effort} if effort else {}
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=256, **kw,
    )


def _responses(client: OpenAI, model: str, effort: str | None):
    kw = {"reasoning": {"effort": effort}} if effort else {}
    return client.responses.create(model=model, input=PROMPT, max_output_tokens=256, **kw)


def _try(call) -> str:
    try:
        call()
        return "ok"
    except (BadRequestError, NotFoundError) as e:
        return f"{e.status_code}: {getattr(e, 'message', e)}"


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    client = OpenAI()
    failures: list[str] = []

    served = {m.id for m in client.models.list()}
    offered = RECOMMENDED_MODELS["openai"]
    for model in [*offered, *MODEL_ROLE_DEFAULTS["openai"].values()]:
        if model not in served:
            failures.append(f"{model}: not in /v1/models")
    print("served:", ", ".join(sorted(m for m in served if m.startswith(("gpt-5", "o3", "o4")))))

    for model in offered:
        entry = DIRECT_EFFORT_CATALOG.get(model)
        if entry is None:
            continue
        for effort in entry["efforts"]:
            for name, call in (("chat", _chat), ("responses", _responses)):
                result = _try(lambda: call(client, model, effort))
                print(f"{model:14} {effort:8} {name:10} {result}")
                if result != "ok":
                    failures.append(f"{model} {effort} {name}: {result}")
        for name, call in (("chat", _chat), ("responses", _responses)):
            result = _try(lambda: call(client, model, "minimal"))
            print(f"{model:14} {'minimal':8} {name:10} {result}")
            if result == "ok":
                failures.append(f"{model} minimal {name}: accepted, catalog may be missing it")
        try:
            reported = _responses(client, model, None).reasoning.effort
        except (BadRequestError, NotFoundError) as e:
            failures.append(f"{model} default: {e.status_code}")
            continue
        print(f"{model:14} default reported by API: {reported}")
        if reported != entry["default"]:
            failures.append(f"{model}: API default {reported!r}, catalog says {entry['default']!r}")

    print()
    print("\n".join(failures) or "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
