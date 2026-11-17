"""Ask Jev's `addon_route` question on the hand-written mid-turn messages.

Runs the production request and parse (`jev_route.decide_addon`) on every case
in `addon_cases.py` at several thresholds. The headline number is
``false_add``: a message that should run next but was added to the running
task.

Usage (from cowork-server/), with the same credentials as `run.py`:

    MINDS_API_KEY=... uv run python -m scripts.eval_jev_route.run_addon
    uv run python -m scripts.eval_jev_route.run_addon --repeat 3 --out /tmp/addon.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.handlers import jev_route
from scripts.eval_jev_route.addon_cases import ADDON_CASES
from scripts.eval_jev_route.run import _credentials

THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


async def _run_case(case: dict, *, target: jev_route.JevTarget, model: str, timeout: float) -> dict:
    state = jev_route.addon_state(
        task=case["task"], progress=case["progress"], added=case["added"], message=case["message"],
    )
    verdict = await jev_route.decide_addon(
        target=target, state=state, model=model, timeout_seconds=timeout, min_p_add=0.0,
    )
    return {
        "id": case["id"], "category": case["category"], "expect": case["expect"],
        "chose_add": verdict.chose_add, "p_add": verdict.p_add, "ms": verdict.ms, "error": verdict.error,
    }


def _report(results: list[dict]) -> None:
    for threshold in THRESHOLDS:
        scored = [r for r in results if r["expect"] != "either" and not r["error"]]
        added = [r for r in scored if r["chose_add"] and (r["p_add"] or 0) >= threshold]
        false_add = sum(1 for r in added if r["expect"] == "queue")
        missed_add = sum(1 for r in scored if r["expect"] == "add" and r not in added)
        print(f"threshold {threshold:.2f}: false_add={false_add} missed_add={missed_add} of {len(scored)}")
    print()
    for r in results:
        p = "-" if r["p_add"] is None else f"{r['p_add']:.2f}"
        print(f"{r['id']:<18} {r['expect']:<6} chose_add={r['chose_add']!s:<5} p_add={p:<5} "
              f"{r['ms']:>5}ms {r['error']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".anton" / ".env")
    parser.add_argument("--model", default=TurnQueueSettings().jev_route_model)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    key, base_url = _credentials(args.env_file)
    if len(key) < 20:
        raise SystemExit(f"No usable MindsHub key: set MINDS_API_KEY, or MINDSHUB_API_KEY in {args.env_file}.")
    target = jev_route.JevTarget.mindshub(base_url, key)
    results = []
    for _ in range(args.repeat):
        results += await asyncio.gather(*(
            _run_case(case, target=target, model=args.model, timeout=args.timeout) for case in ADDON_CASES
        ))
    await jev_route.close_client()
    _report(results)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
