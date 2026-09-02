#!/usr/bin/env python3

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_s55_200_phase2 import (
    EXPECTED_PROMPT_SHA256,
    SAMPLING,
    atomic_publish_bytes,
    combine_reports,
    file_record,
    log_identity,
    parse_decode_summary,
    prompt_manifest,
    read_log_range,
    validate_arm_report,
    validate_combine_evidence,
    validate_current_instance,
    validate_file_record,
    validate_response,
    validate_stats,
    validate_timing_segment,
)


PROMPT_TOKENS = 197_395
CYCLES = 300
ACCEPTED = 211
REJECTED = 89


def timing_line(pos: int, committed: int) -> str:
    result = "ACCEPT" if committed == 2 else "reject"
    return (
        "ds4: glm mtp utility: width=2 "
        f"pos={pos} setup=1.0 ms verify[batch]=40.0 ms rollback=0.0 ms "
        f"draft=4.0 ms other=1.0 ms result={result} committed={committed} "
        "total=46.0 ms utility=1.0 tok/s-cycle\n"
    )


def timing_segment() -> str:
    pos = PROMPT_TOKENS + 1
    lines = []
    for committed in [2] * ACCEPTED + [1] * REJECTED:
        lines.append(timing_line(pos, committed))
        pos += committed
    return "".join(lines)


def summary_line(request_id: str = "chatcmpl-1", prompt: int = PROMPT_TOKENS) -> str:
    return (
        f"ds4-server: decode-summary req={request_id} prompt={prompt} gen=512 "
        "seconds=20.480000000 tps=25.000000000 seed_gen=1 steady_gen=511 "
        "steady_seconds=20.440000000 mtp_active=1 mtp_width=2 "
        f"mtp_cycles={CYCLES} mtp_accepted={ACCEPTED} "
        f"mtp_rejected={REJECTED} mtp_committed=511\n"
    )


def usage(request_id: str = "chatcmpl-1") -> dict:
    return {
        "request_id": request_id,
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": 512,
        "finish_reason": "length",
        "decoded_output_sha256": "d" * 64,
    }


def valid_arm(mode: str) -> dict:
    decoded = parse_decode_summary(summary_line(), usage())
    runtime = {
        "provenance_sha256": "a" * 64,
        "binary_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "driver_sha256": "1" * 64,
        "launcher_sha256": "2" * 64,
        "models_sha256": "e" * 64,
        "provenance_path": "/tmp/phase2-provenance.json",
        "normalized_argv": ["/tmp/ds4-server", "--metal", "--mtp"],
        "instance": "1" * 16 if mode == "timing" else "2" * 16,
        "pid": 10 if mode == "timing" else 11,
        "model": "glm",
        "model_path": "/tmp/model.gguf",
        "ctx_size": 262_144,
        "slot_count": 1,
        "mtp": {
            "active": True,
            "counters": True,
            "timing": mode == "timing",
            "width": 2,
        },
    }
    mtp = None
    if mode == "timing":
        mtp = {
            "cycles": CYCLES,
            "accepted": ACCEPTED,
            "rejected": REJECTED,
            "committed_tokens": 511,
            "tokens_per_cycle": 511 / CYCLES,
        }
    return {
        "schema": "s55-200-phase2-arm-v2",
        "contract_locked": True,
        "scope": "phase2-cycle-budget-only",
        "mode": mode,
        "endpoint": "http://127.0.0.1:8000",
        "prompt": {
            "prompt_sha256": EXPECTED_PROMPT_SHA256,
            "prompt_bytes": 769_966,
        },
        "request": {
            "model": "glm-5.2-chat",
            "generated_tokens": 512,
            "sampling": SAMPLING,
            "request_id": "chatcmpl-1",
            "requests_before": 0,
            "requests_after": 1,
            "cache_before": {"hits": 0, "cold": 0},
            "cache_after": {"hits": 1, "cold": 0},
        },
        "usage": usage(),
        "decode": decoded,
        "runtime": runtime,
        "log": {"prefix_sha256": "f" * 64, "segment_sha256": "0" * 64},
        "mtp_timing": mtp,
    }


class Phase2DriverTest(unittest.TestCase):
    def test_atomic_publish_is_complete_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            atomic_publish_bytes(path, b"first\n")
            self.assertEqual(path.read_bytes(), b"first\n")
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                atomic_publish_bytes(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")
            self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_prompt_manifest_is_frozen(self) -> None:
        _, manifest = prompt_manifest(Path(__file__).resolve().parents[1])
        self.assertEqual(manifest["prompt_sha256"], EXPECTED_PROMPT_SHA256)
        self.assertEqual(manifest["prompt_bytes"], 769_966)

    def test_validate_response_rejects_early_stop(self) -> None:
        response = {
            "id": "chatcmpl-1",
            "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": 511},
            "choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
        }
        with self.assertRaisesRegex(RuntimeError, "early/short"):
            validate_response(response)

    def test_validate_response_rejects_falsey_non_text(self) -> None:
        response = {
            "id": "chatcmpl-1",
            "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": 512},
            "choices": [{"finish_reason": "length", "message": {"content": 0}}],
        }
        with self.assertRaisesRegex(RuntimeError, "not text"):
            validate_response(response)

    def test_summary_is_request_bound_and_exact(self) -> None:
        decoded = parse_decode_summary(summary_line(), usage())
        self.assertEqual(decoded["committed"], 511)
        self.assertEqual(decoded["tokens_per_cycle"], 511 / CYCLES)
        with self.assertRaisesRegex(RuntimeError, "HTTP response"):
            parse_decode_summary(summary_line("chatcmpl-2"), usage())
        with self.assertRaisesRegex(RuntimeError, "actual context"):
            parse_decode_summary(summary_line(prompt=PROMPT_TOKENS + 1), usage())

    def test_summary_rejects_fake_clean_mtp(self) -> None:
        segment = summary_line().replace("mtp_active=1", "mtp_active=0")
        with self.assertRaisesRegex(RuntimeError, "active width-2"):
            parse_decode_summary(segment, usage())

    def test_timing_covers_same_511_token_window(self) -> None:
        decoded = parse_decode_summary(summary_line(), usage())
        report = validate_timing_segment(timing_segment(), PROMPT_TOKENS, decoded)
        self.assertEqual(report["cycles"], CYCLES)
        self.assertEqual(report["accepted"], ACCEPTED)
        self.assertEqual(report["committed_tokens"], 511)

    def test_current_process_marker_rejects_stale_log(self) -> None:
        stats = {
            "instance": "1" * 16,
            "pid": 42,
        }
        prefix = (
            "  MTP:        enabled (--mtp, width 2) "
            "+ counters (--mtp-counters)\n"
            "  S55:        Phase 2 contract-locked "
            "(no DS4/Metal/DYLD overrides)\n"
            f"ds4-server: runtime instance={'1' * 16} pid=42 schema=1\n"
            "Fans at maximum: fan0=1/1RPM fan1=1/1RPM\n"
        )
        self.assertIn("enabled", validate_current_instance(prefix, stats, "clean"))
        with self.assertRaisesRegex(RuntimeError, "one runtime marker"):
            validate_current_instance(prefix + prefix, stats, "clean")
        with self.assertRaisesRegex(RuntimeError, "prior decode summary"):
            validate_current_instance(prefix + summary_line(), stats, "clean")
        with self.assertRaisesRegex(RuntimeError, "fans fell below"):
            validate_current_instance(
                prefix + "warning=fans-below-max\n", stats, "clean"
            )

    def test_stats_requires_current_active_mode(self) -> None:
        stats = {
            "runtime_schema": 1,
            "instance": "1" * 16,
            "pid": 42,
            "slot_count": 1,
            "busy": False,
            "queue_depth": 0,
            "requests": 0,
            "generated_tokens": 0,
            "live_tokens": 0,
            "cache": {"hits": 0, "cold": 0},
            "ctx_size": 262_144,
            "mtp": {
                "active": True,
                "counters": True,
                "width": 2,
                "timing": False,
            },
        }
        validate_stats(stats, "clean")
        stats["mtp"]["counters"] = False
        validate_stats(stats, "m0")
        stats["mtp"]["counters"] = True
        stats["mtp"]["active"] = False
        with self.assertRaisesRegex(RuntimeError, "no active width-2"):
            validate_stats(stats, "clean")

    def test_log_replacement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_bytes(b"old")
            identity = log_identity(path)
            replacement = Path(directory) / "replacement.log"
            replacement.write_bytes(b"newer")
            os.replace(replacement, path)
            with self.assertRaisesRegex(RuntimeError, "replaced or rotated"):
                read_log_range(path, 0, None, identity)

    def test_full_hash_rejects_same_metadata_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.write_bytes(b"original")
            record = file_record(path, hash_contents=True)
            path.write_bytes(b"modified")
            os.utime(path, ns=(record["mtime_ns"], record["mtime_ns"]))
            validate_file_record(record, rehash=False)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
                validate_file_record(record, rehash=True)

    def test_combine_uses_matched_post_seed_window(self) -> None:
        result = combine_reports(valid_arm("timing"), valid_arm("clean"))
        self.assertAlmostEqual(result["current_cycle_ms"], 20_440 / CYCLES)
        self.assertAlmostEqual(result["target_cycle_ms"], 1000 * (511 / CYCLES) / 55)
        self.assertEqual(result["steady_window_tokens"], 511)
        self.assertFalse(result["target_200k_reached"])
        self.assertEqual(result["full_s55_status"], "NOT_EVALUATED")
        self.assertEqual(result["timing_overhead_status"], "NOT_EVALUATED")
        self.assertNotIn("timing_within_3pct", result)

    def test_combine_rejects_mismatched_provenance(self) -> None:
        timing = valid_arm("timing")
        clean = valid_arm("clean")
        clean["runtime"]["binary_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "binary_sha256"):
            combine_reports(timing, clean)

    def test_combine_rejects_skeletal_or_weakened_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "contract-locked"):
            validate_arm_report({"mode": "timing"}, "timing")
        clean = valid_arm("clean")
        clean["request"]["generated_tokens"] = 513
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_arm_report(clean, "clean")
        clean = valid_arm("clean")
        clean["request"]["requests_before"] = 1
        clean["request"]["requests_after"] = 2
        with self.assertRaisesRegex(ValueError, "fresh isolated"):
            validate_arm_report(clean, "clean")

    def test_combine_reopens_evidence_instead_of_trusting_synthetic_arms(self) -> None:
        with self.assertRaises(FileNotFoundError):
            validate_combine_evidence(valid_arm("timing"), valid_arm("clean"))

    def test_combine_rejects_changed_acceptance_counts(self) -> None:
        timing = valid_arm("timing")
        clean = copy.deepcopy(valid_arm("clean"))
        clean["decode"]["accepted"] -= 1
        clean["decode"]["rejected"] += 1
        clean["decode"]["committed"] -= 1
        with self.assertRaises(ValueError):
            combine_reports(timing, clean)


if __name__ == "__main__":
    unittest.main()
