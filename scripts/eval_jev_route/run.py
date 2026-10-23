"""Compare the current LLM route gate with Jev on the hand-written cases.

Three strategies, each run on every case in `cases.py`:

- ``llm_gate``: today's gate as shipped. `decide_route` with its code checks
  and the router model (mindshub_air by default), on the real budgets.
- ``jev_current``: Jev swapped into today's gate unchanged. Today's code checks
  (including `prior_turn_ran_tools`) and the shadow probe's question and state.
- ``jev_proposed``: the shipped Jev gate. The code checks without
  `prior_turn_ran_tools`, and `jev_route.ROUTE_QUESTION`, which routes on
  whether the reply's facts are in visible text.

Jev is asked on every case even when a code check settles the route, so the
per-case table shows its raw answer too. The headline number is
``false_direct``: a case that needs the agent but was routed to a direct answer.

Usage (from cowork-server/):

    MINDS_API_KEY=... uv run python -m scripts.eval_jev_route.run
    uv run python -m scripts.eval_jev_route.run --only jev_proposed --repeat 3 --out /tmp/jev.json

The key comes from MINDS_API_KEY, else MINDSHUB_API_KEY or ANTON_MINDS_API_KEY
in --env-file (default ~/.anton/.env); the URL from MINDS_URL, else
ANTON_MINDS_URL there, else api.mindshub.ai.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

from cowork.handlers import jev_route
from cowork.handlers.response_routing import (
    DIRECT_CONTEXT,
    RouterBinding,
    _text_history,
    decide_route,
    explicit_file_context,
    prior_turn_ran_tools,
)
from cowork.services.providers import minds_chat_base_url
from scripts.eval_jev_route.cases import CASES

STRATEGIES = ("llm_gate", "jev_current", "jev_proposed")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

# The question the retired shadow probe asked (cowork/handlers/jev_shadow.py
# before the Jev gate shipped), kept here so `jev_current` stays comparable.
SHADOW_QUESTION: dict = {
    "type": "choice",
    "instructions": (
        "Classify what is required to produce the assistant's next reply. "
        "Return exactly one of: answer_directly or needs_agent. Classify as "
        "needs_agent if any condition listed for it applies; otherwise "
        "classify as answer_directly. When uncertain, choose needs_agent."
    ),
    "criteria": {
        "answer_directly": (
            "The reply can be produced solely from text already visible in "
            "the conversation and stable general knowledge. It requires no "
            "tool use, external retrieval, access to an attachment, file, "
            "account, or dataset, calculation, fact verification, or "
            "multi-step planning. It is not about the assistant's runtime "
            "identity, provider, model, permissions, or available "
            "resources, and it is not about using, installing, updating, "
            "pricing, supporting platforms, or configuring Cowork."
        ),
        "needs_agent": (
            "The reply requires any tool use; browsing or external "
            "retrieval; access to an attachment, file, account, or "
            "dataset; executing or testing code; verifying potentially "
            "time-sensitive information; a calculation; or a multi-step "
            "plan. Also choose this for questions about the assistant's "
            "runtime identity, provider, model, permissions, or available "
            "resources, and for questions about using, installing, "
            "updating, pricing, supporting platforms, or configuring "
            "Cowork. Choose this option when uncertain."
        ),
    },
}


def _credentials(env_file: Path) -> tuple[str, str]:
    env = dotenv_values(env_file) if env_file.is_file() else {}
    key = (os.environ.get("MINDS_API_KEY") or env.get("MINDSHUB_API_KEY")
           or env.get("ANTON_MINDS_API_KEY") or "")
    url = os.environ.get("MINDS_URL") or env.get("ANTON_MINDS_URL") or "https://api.mindshub.ai"
    return key, minds_chat_base_url(url)


async def _ask_jev(client: httpx.AsyncClient, *, base_url: str, key: str, model: str,
                   state, question: dict) -> dict:
    started = time.monotonic()
    try:
        response = await client.post(
            f"{base_url}/decisions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "state": state, "questions": {"route": question}},
        )
    except httpx.HTTPError as exc:
        return {"ms": round((time.monotonic() - started) * 1000), "error": type(exc).__name__}
    ms = round((time.monotonic() - started) * 1000)
    if response.status_code != 200:
        return {"ms": ms, "error": f"http_{response.status_code}"}
    try:
        answer = response.json()["answers"]["route"]
        choice = answer["choice"]
        probabilities = answer.get("probabilities") or {}
        # Older replies carried no probabilities; fall back to the choice itself.
        p_direct = probabilities.get("answer_directly")
        if p_direct is None:
            p_direct = 1.0 if choice == "answer_directly" else 0.0
        return {"ms": ms, "choice": choice, "p_direct": float(p_direct)}
    except Exception as exc:
        return {"ms": ms, "error": f"malformed:{type(exc).__name__}"}


def _check_route(history: list[dict], *, with_prior_tools: bool) -> str | None:
    """The code checks' verdict, or None when it is left to the model."""
    if explicit_file_context(history):
        return "explicit_file_context"
    if with_prior_tools and prior_turn_ran_tools(history):
        return "prior_turn_ran_tools"
    return None


def _jev_route(check: str | None, jev: dict, threshold: float) -> str:
    if check or "error" in jev:
        return "agent"
    return "direct" if jev["p_direct"] >= threshold else "agent"


async def _run_case(case: dict, *, args, key: str, base_url: str, client: httpx.AsyncClient,
                    binding: RouterBinding | None) -> dict:
    history = case["history"]
    messages = _text_history(history)
    result: dict = {"id": case["id"], "category": case["category"], "expect": case["expect"]}

    if "llm_gate" in args.only:
        started = time.monotonic()
        decision = await decide_route(
            history=history, has_non_text_input=False, has_attachments=False,
            has_disabled_connections=False, binding=binding,
        )
        result["llm_gate"] = {
            "route": "direct" if decision.route == DIRECT_CONTEXT else "agent",
            "reason": decision.reason,
            "ms": round((time.monotonic() - started) * 1000),
        }

    if "jev_current" in args.only:
        check = _check_route(history, with_prior_tools=True)
        jev = await _ask_jev(client, base_url=base_url, key=key, model=args.jev_model,
                             state=messages, question=SHADOW_QUESTION)
        result["jev_current"] = {"check": check, **jev}

    if "jev_proposed" in args.only:
        check = _check_route(history, with_prior_tools=False)
        jev = await _ask_jev(client, base_url=base_url, key=key, model=args.jev_model,
                             state=jev_route.route_state(messages), question=jev_route.ROUTE_QUESTION)
        result["jev_proposed"] = {"check": check, **jev}
    return result


def _route_of(result: dict, strategy: str, threshold: float) -> str | None:
    entry = result.get(strategy)
    if entry is None:
        return None
    if strategy == "llm_gate":
        return entry["route"]
    return _jev_route(entry["check"], entry, threshold)


def _score(results: list[dict], strategy: str, threshold: float) -> dict:
    false_direct, missed_direct, correct, scored = [], [], 0, 0
    for r in results:
        route = _route_of(r, strategy, threshold)
        if route is None or r["expect"] == "either":
            continue
        scored += 1
        if route == r["expect"]:
            correct += 1
        elif route == "direct":
            false_direct.append(r["id"])
        else:
            missed_direct.append(r["id"])
    return {"scored": scored, "correct": correct, "false_direct": false_direct, "missed_direct": missed_direct}


def _latency(results: list[dict], strategy: str) -> str:
    ms = sorted(r[strategy]["ms"] for r in results if strategy in r and "ms" in r[strategy])
    if not ms:
        return "-"
    p95 = ms[min(len(ms) - 1, round(0.95 * (len(ms) - 1)))]
    return f"p50 {statistics.median(ms):.0f}ms  p95 {p95}ms"


def _cell(result: dict, strategy: str, threshold: float) -> str:
    entry = result.get(strategy)
    if entry is None:
        return "-"
    route = _route_of(result, strategy, threshold)
    if strategy == "llm_gate":
        detail = entry["reason"]
    elif "error" in entry:
        detail = entry["error"]
    else:
        detail = f"p={entry['p_direct']:.2f}" + (f" ({entry['check']})" if entry["check"] else "")
    mark = "" if result["expect"] in ("either", route) else " ✗"
    return f"{route}{mark} {detail}"


def _report(results: list[dict], args) -> None:
    strategies = [s for s in STRATEGIES if s in args.only]
    print(f"\n{'case':<22}{'expect':<8}" + "".join(f"{s:<44}" for s in strategies))
    for r in results:
        print(f"{r['id']:<22}{r['expect']:<8}"
              + "".join(f"{_cell(r, s, args.threshold):<44}" for s in strategies))

    print("\nSummary (false_direct = needed the agent, got a direct answer)")
    for s in strategies:
        sc = _score(results, s, args.threshold)
        print(f"  {s:<14} {sc['correct']}/{sc['scored']} correct   "
              f"false_direct={len(sc['false_direct'])} {sc['false_direct']}   "
              f"missed_direct={len(sc['missed_direct'])}   {_latency(results, s)}")
        errors = [r["id"] for r in results if "error" in r.get(s, {})]
        if errors:
            print(f"  {'':<14} errors on {len(errors)} cases: {errors[:5]}")

    for s in (s for s in strategies if s != "llm_gate"):
        print(f"\n{s}: threshold on p(answer_directly)")
        for t in THRESHOLDS:
            sc = _score(results, s, t)
            print(f"  >= {t:<5} {sc['correct']}/{sc['scored']} correct   "
                  f"false_direct={len(sc['false_direct'])}   missed_direct={len(sc['missed_direct'])}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--jev-model", default="jev")
    parser.add_argument("--gate-model", default="mindshub_air")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="p(answer_directly) Jev needs for a direct route in the per-case table")
    parser.add_argument("--repeat", type=int, default=1, help="run every case this many times")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, help="write the raw per-case results here as JSON")
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".anton" / ".env",
                        help="dotenv file holding MINDSHUB_API_KEY or ANTON_MINDS_API_KEY")
    args = parser.parse_args()

    key, base_url = _credentials(args.env_file)
    if len(key) < 20:
        raise SystemExit(f"No usable MindsHub key: set MINDS_API_KEY, or MINDSHUB_API_KEY in {args.env_file}.")

    binding = None
    if "llm_gate" in args.only:
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(api_key=key, base_url=base_url, flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH)
        binding = RouterBinding(provider=provider, model=args.gate_model, label="minds_cloud")

    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=30) as client:
        async def one(case: dict) -> dict:
            async with semaphore:
                return await _run_case(case, args=args, key=key, base_url=base_url, client=client, binding=binding)

        results = await asyncio.gather(*(one(c) for _ in range(args.repeat) for c in CASES))

    _report(results, args)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nRaw results: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
