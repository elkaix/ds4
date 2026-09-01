#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_glm53_mtp import contiguous_runs, parse_cycles, summarize


def timing_line(pos: int, committed: int, total: float) -> str:
    result = "ACCEPT" if committed == 2 else "reject"
    return (
        "ds4: glm mtp utility: width=2 "
        f"pos={pos} setup=0.8 ms verify[batch]=50.0 ms rollback=0.0 ms "
        f"draft=4.5 ms other=1.5 ms result={result} committed={committed} "
        f"total={total:.1f} ms utility=1.0 tok/s-cycle\n"
    )


class AnalyzeGlm53MtpTest(unittest.TestCase):
    def test_summary_and_cycle_budget(self) -> None:
        cycles = parse_cycles(
            [timing_line(200_000, 2, 57.0), timing_line(200_002, 1, 58.0)]
        )
        report = summarize(cycles, decode_tps=25.0, target_tps=55.0)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(report["tokens_per_cycle"], 1.5)
        budget = report["cycle_budget"]
        self.assertAlmostEqual(budget["current_cycle_ms"], 60.0)
        self.assertAlmostEqual(budget["target_cycle_ms"], 300.0 / 11.0)

    def test_contiguous_run_split(self) -> None:
        cycles = parse_cycles(
            [timing_line(2_000, 1, 50.0), timing_line(2_001, 2, 51.0),
             timing_line(200_000, 2, 70.0)]
        )
        runs = contiguous_runs(cycles)
        self.assertEqual([len(run) for run in runs], [2, 1])

    def test_passing_budget_has_no_remaining_deficit(self) -> None:
        cycles = parse_cycles([timing_line(200_000, 2, 30.0)])
        budget = summarize(cycles, decode_tps=60.0, target_tps=55.0)[
            "cycle_budget"
        ]
        self.assertEqual(budget["remaining_deficit_ms"], 0.0)
        self.assertLess(budget["cycle_margin_ms"], 0.0)

    def test_rejects_inconsistent_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "contradicts"):
            parse_cycles([timing_line(2_000, 2, 50.0).replace("ACCEPT", "reject")])


if __name__ == "__main__":
    unittest.main()
