#!/usr/bin/env python3
"""Summarize GLM-5.3 width-2 MTP timing logs and the S55 cycle budget.

The timing log is diagnostic: its per-cycle fprintf perturbs throughput.  Pass
the matched profiler-OFF decode rate with --decode-tps to derive the canonical
cycle time; never treat the logged timing mean as S55 progress.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MARKER = "ds4: glm mtp utility:"
NUMBER = r"[0-9]+(?:\.[0-9]+)?"
LINE_RE = re.compile(
    rf"{re.escape(MARKER)} width=(?P<width>[0-9]+) "
    rf"pos=(?P<pos>[0-9]+) setup=(?P<setup>{NUMBER}) ms "
    rf"verify\[(?P<path>[^]]+)\]=(?P<verify>{NUMBER}) ms "
    rf"rollback=(?P<rollback>{NUMBER}) ms draft=(?P<draft>{NUMBER}) ms "
    rf"other=(?P<other>{NUMBER}) ms result=(?P<result>ACCEPT|reject) "
    rf"committed=(?P<committed>[0-9]+) total=(?P<total>{NUMBER}) ms"
)


@dataclass(frozen=True)
class Cycle:
    pos: int
    path: str
    committed: int
    setup_ms: float
    verify_ms: float
    rollback_ms: float
    draft_ms: float
    other_ms: float
    total_ms: float


def parse_cycles(lines: Iterable[str]) -> list[Cycle]:
    cycles: list[Cycle] = []
    for line_number, line in enumerate(lines, 1):
        if MARKER not in line:
            continue
        match = LINE_RE.search(line)
        if not match:
            raise ValueError(f"malformed MTP timing line {line_number}")
        width = int(match["width"])
        committed = int(match["committed"])
        result = match["result"]
        if width != 2:
            raise ValueError(f"line {line_number}: expected width 2, got {width}")
        if committed not in (1, 2):
            raise ValueError(
                f"line {line_number}: expected 1 or 2 committed tokens, got {committed}"
            )
        if (result == "ACCEPT") != (committed == 2):
            raise ValueError(
                f"line {line_number}: result={result} contradicts committed={committed}"
            )
        cycles.append(
            Cycle(
                pos=int(match["pos"]),
                path=match["path"],
                committed=committed,
                setup_ms=float(match["setup"]),
                verify_ms=float(match["verify"]),
                rollback_ms=float(match["rollback"]),
                draft_ms=float(match["draft"]),
                other_ms=float(match["other"]),
                total_ms=float(match["total"]),
            )
        )
    if not cycles:
        raise ValueError("no GLM width-2 MTP timing cycles found")
    return cycles


def contiguous_runs(cycles: list[Cycle]) -> list[list[Cycle]]:
    runs: list[list[Cycle]] = []
    for cycle in cycles:
        if not runs or cycle.pos != runs[-1][-1].pos + runs[-1][-1].committed:
            runs.append([])
        runs[-1].append(cycle)
    return runs


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    z2_over_n = z * z / total
    denominator = 1.0 + z2_over_n
    center = (p + z2_over_n / 2.0) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def summarize(
    cycles: list[Cycle], decode_tps: float | None, target_tps: float
) -> dict[str, object]:
    if not cycles:
        raise ValueError("cannot summarize an empty run")
    if decode_tps is not None and decode_tps <= 0.0:
        raise ValueError("decode TPS must be positive")
    if target_tps <= 0.0:
        raise ValueError("target TPS must be positive")

    accepted = sum(c.committed == 2 for c in cycles)
    committed = sum(c.committed for c in cycles)
    acceptance = accepted / len(cycles)
    tokens_per_cycle = committed / len(cycles)
    ci_low, ci_high = _wilson_interval(accepted, len(cycles))
    path_counts: dict[str, int] = {}
    for cycle in cycles:
        path_counts[cycle.path] = path_counts.get(cycle.path, 0) + 1

    result: dict[str, object] = {
        "position": {"first": cycles[0].pos, "last": cycles[-1].pos},
        "cycles": len(cycles),
        "accepted": accepted,
        "rejected": len(cycles) - accepted,
        "committed_tokens": committed,
        "acceptance": acceptance,
        "acceptance_ci95": [ci_low, ci_high],
        "tokens_per_cycle": tokens_per_cycle,
        "path_counts": path_counts,
        "diagnostic_timing_ms": {
            "setup": _stats([c.setup_ms for c in cycles]),
            "verify": _stats([c.verify_ms for c in cycles]),
            "rollback": _stats([c.rollback_ms for c in cycles]),
            "draft": _stats([c.draft_ms for c in cycles]),
            "other": _stats([c.other_ms for c in cycles]),
            "total": _stats([c.total_ms for c in cycles]),
        },
    }
    if decode_tps is not None:
        current_cycle_ms = 1000.0 * tokens_per_cycle / decode_tps
        target_cycle_ms = 1000.0 * tokens_per_cycle / target_tps
        margin_ms = current_cycle_ms - target_cycle_ms
        deficit_ms = max(0.0, margin_ms)
        result["cycle_budget"] = {
            "matched_profiler_off_decode_tps": decode_tps,
            "target_tps": target_tps,
            "current_cycle_ms": current_cycle_ms,
            "target_cycle_ms": target_cycle_ms,
            "remaining_deficit_ms": deficit_ms,
            "cycle_margin_ms": margin_ms,
            "required_cycle_reduction_pct": 100.0 * deficit_ms / current_cycle_ms,
        }
    return result


def render_text(summary: dict[str, object]) -> str:
    position = summary["position"]
    assert isinstance(position, dict)
    ci = summary["acceptance_ci95"]
    assert isinstance(ci, list)
    paths = summary["path_counts"]
    assert isinstance(paths, dict)
    timing = summary["diagnostic_timing_ms"]
    assert isinstance(timing, dict)
    total = timing["total"]
    assert isinstance(total, dict)
    lines = [
        "S55-200 CYCLE BUDGET",
        "",
        f"positions: {position['first']}..{position['last']}",
        f"cycles: {summary['cycles']}",
        f"accepted/rejected: {summary['accepted']}/{summary['rejected']}",
        f"acceptance: {100.0 * float(summary['acceptance']):.4f}% "
        f"(95% Wilson {100.0 * float(ci[0]):.4f}%..{100.0 * float(ci[1]):.4f}%)",
        f"tokens/cycle: {float(summary['tokens_per_cycle']):.6f}",
        "paths: " + ", ".join(f"{name}={count}" for name, count in sorted(paths.items())),
        "",
        "diagnostic timing only (instrumented; zero S55 credit):",
        f"cycle total median/mean/range: {float(total['median']):.3f} / "
        f"{float(total['mean']):.3f} / {float(total['min']):.3f}.."
        f"{float(total['max']):.3f} ms",
    ]
    budget = summary.get("cycle_budget")
    if isinstance(budget, dict):
        lines.extend(
            [
                "",
                "canonical budget (matched profiler-OFF decode):",
                f"current cycle: {float(budget['current_cycle_ms']):.3f} ms",
                f"target cycle for {float(budget['target_tps']):.2f} t/s: "
                f"{float(budget['target_cycle_ms']):.3f} ms",
                f"remaining deficit: {float(budget['remaining_deficit_ms']):.3f} ms",
                f"required cycle reduction: "
                f"{float(budget['required_cycle_reduction_pct']):.3f}%",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-pos", type=int)
    parser.add_argument("--max-pos", type=int)
    parser.add_argument("--run", type=int, help="zero-based contiguous run index")
    parser.add_argument("--expect-cycles", type=int)
    parser.add_argument("--expect-committed", type=int)
    parser.add_argument("--decode-tps", type=float)
    parser.add_argument("--target-tps", type=float, default=55.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with args.log.open(encoding="utf-8", errors="replace") as source:
        cycles = parse_cycles(source)
    if args.min_pos is not None:
        cycles = [cycle for cycle in cycles if cycle.pos >= args.min_pos]
    if args.max_pos is not None:
        cycles = [cycle for cycle in cycles if cycle.pos <= args.max_pos]
    if not cycles:
        raise SystemExit("no timing cycles remain after position filtering")

    runs = contiguous_runs(cycles)
    if args.run is None:
        if len(runs) != 1:
            spans = ", ".join(
                f"{index}:{run[0].pos}..{run[-1].pos} ({len(run)})"
                for index, run in enumerate(runs)
            )
            raise SystemExit(f"found {len(runs)} contiguous runs; select --run: {spans}")
        selected = runs[0]
    else:
        if args.run < 0 or args.run >= len(runs):
            raise SystemExit(f"--run must be between 0 and {len(runs) - 1}")
        selected = runs[args.run]
    if args.expect_cycles is not None and len(selected) != args.expect_cycles:
        raise SystemExit(
            f"expected {args.expect_cycles} cycles, found {len(selected)}"
        )
    committed = sum(cycle.committed for cycle in selected)
    if args.expect_committed is not None and committed != args.expect_committed:
        raise SystemExit(
            f"expected {args.expect_committed} committed tokens, found {committed}"
        )

    report = summarize(selected, args.decode_tps, args.target_tps)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
