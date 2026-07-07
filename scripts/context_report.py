#!/usr/bin/env python3
"""Baseline context-consumption report (ENG-642).

Aggregates the three telemetry log streams emitted by cowork-server
(cowork/services/llm_telemetry.py) into the numbers the context-optimization
initiative is judged against:

    [llm_usage]      per LLM call
    [turn_summary]   per turn
    [prompt_anatomy] per turn

Usage:
    python scripts/context_report.py server.log [more.log ...]
    ... | python scripts/context_report.py -

Stdlib only. Reads plain log files; any line containing a tag is parsed from
the first '{' after it.
"""

from __future__ import annotations

import json
import sys

TAGS = ("[llm_usage]", "[turn_summary]", "[prompt_anatomy]")

# The static floor of one LLM call: anton's always-on prompt blocks (the
# HTML-viz and ASK_FIRST variants are off by default and excluded), plus the
# cowork suffix and the tool schemas measured per turn.
FLOOR_SECTIONS = (
    "CHAT_SYSTEM_PROMPT",
    "BACKEND_GENERATION_PROMPT",
    "ARTIFACTS_PROMPT",
    "BASE_VISUALIZATIONS_PROMPT",
    "VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT",
    "CONVERSATION_DISCIPLINE_ACT_FIRST",
)
CHARS_PER_TOKEN = 4


def parse(lines) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {t: [] for t in TAGS}
    for line in lines:
        for tag in TAGS:
            idx = line.find(tag)
            if idx == -1:
                continue
            start = line.find("{", idx)
            if start == -1:
                continue
            try:
                records[tag].append(json.loads(line[start:].strip()))
            except ValueError:
                pass
            break
    return records


def pct(values: list, p: float):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
    return vals[k]


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}"


def stat_line(label: str, values: list) -> str:
    return (f"  {label:<28} n={len([v for v in values if v is not None]):<6}"
            f" p50={fmt(pct(values, 50)):<12} p95={fmt(pct(values, 95)):<12}"
            f" max={fmt(pct(values, 100))}")


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    lines: list[str] = []
    for path in paths:
        if path == "-":
            lines.extend(sys.stdin)
        else:
            with open(path, errors="replace") as fh:
                lines.extend(fh)
    rec = parse(lines)
    usage, turns, anatomy = rec["[llm_usage]"], rec["[turn_summary]"], rec["[prompt_anatomy]"]

    print("=== Context consumption baseline ===")
    print(f"log lines: {len(lines):,} | llm calls: {len(usage):,} | "
          f"turns: {len(turns):,} | anatomy lines: {len(anatomy):,}")

    if usage:
        print("\n-- Per LLM call --")
        print(stat_line("input_tokens", [u.get("input_tokens") for u in usage]))
        print(stat_line("output_tokens", [u.get("output_tokens") for u in usage]))
        cache_reads = [u.get("cache_read_input_tokens") for u in usage]
        if any(v for v in cache_reads):
            print(stat_line("cache_read_input_tokens", cache_reads))
        pressures = [u.get("context_pressure") for u in usage]
        print(stat_line("context_pressure", pressures))
        for threshold in (0.5, 0.7, 0.9):
            n = sum(1 for p in pressures if (p or 0) > threshold)
            print(f"  calls over {threshold:.0%} of window: {n} "
                  f"({n / len(usage):.1%})")
        models = sorted({u.get("model") for u in usage if u.get("model")})
        if models:
            print(f"  models: {', '.join(str(m) for m in models)}")

    if turns:
        print("\n-- Per turn --")
        print(stat_line("calls/turn", [t.get("calls") for t in turns]))
        print(stat_line("input_tokens/turn", [t.get("input_tokens") for t in turns]))
        print(stat_line("output_tokens/turn", [t.get("output_tokens") for t in turns]))
        print(stat_line("ttft_ms", [t.get("ttft_ms") for t in turns]))
        print(stat_line("duration_ms", [t.get("duration_ms") for t in turns]))

    if anatomy:
        print("\n-- Prompt anatomy (per turn) --")
        print(stat_line("history_chars", [a.get("history_chars") for a in anatomy]))
        print(stat_line("history_messages", [a.get("history_messages") for a in anatomy]))
        tools_totals = [a.get("tools_total_chars") for a in anatomy]
        print(stat_line("tools_total_chars", tools_totals))
        # A rising p50→max spread here exposes the scratchpad-description
        # accumulation bug (anton mutates the shared ToolDef every session).
        suffix_totals = [sum((a.get("suffix_chars") or {}).values()) for a in anatomy]
        print(stat_line("cowork_suffix_chars", suffix_totals))
        static = anatomy[-1].get("anton_static_chars") or {}
        if static:
            print("  anton static blocks (chars):")
            for name, size in sorted(static.items(), key=lambda kv: -kv[1]):
                print(f"    {name:<48} {size:>8,}")
        floor_chars = (
            sum(static.get(s, 0) for s in FLOOR_SECTIONS)
            + (pct(suffix_totals, 50) or 0)
            + (pct(tools_totals, 50) or 0)
        )
        floor_tokens = floor_chars // CHARS_PER_TOKEN
        print(f"\n  estimated static floor: {floor_chars:,} chars"
              f" (~{floor_tokens:,} tokens per call)")
        if usage:
            median_input = pct([u.get("input_tokens") for u in usage], 50)
            if median_input:
                print(f"  floor share of median call input: "
                      f"{floor_tokens / median_input:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
