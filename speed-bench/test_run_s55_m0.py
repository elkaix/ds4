#!/usr/bin/env python3

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_s55_m0 import (
    analyze_segment_log,
    classify_verdict,
    max_completion_tokens,
    monitor_timestamp,
    parse_lsof_records,
    parse_summary,
    parse_telemetry,
    parse_token_ids,
    planned_segments,
    validate_loaded_files,
    validate_artifact_paths,
    validate_m0_argv,
    validate_m0_stats,
    validate_open_provenance,
    validate_response,
)


def synthetic_log() -> str:
    lines = []
    measured = 0
    position = 200_000
    for sequence, kind in enumerate(planned_segments(2)):
        is_measured = kind in {"mtp", "nomtp"}
        measured_index = measured if is_measured else -1
        if is_measured:
            measured += 1
        if kind == "mtp":
            milliseconds, cycles, accepted, committed, calls = 3200.0, 40, 24, 64, 40
        elif kind == "nomtp":
            milliseconds, cycles, accepted, committed, calls = 2560.0, 0, 0, 0, 64
        else:
            milliseconds, cycles, accepted, committed, calls = 3200.0, 39, 24, 63, 40
        position += 64
        lines.append(
            "ds4-server: s55-m0-segment req=chatcmpl-7 "
            f"seq={sequence} kind={kind} measured={measured_index} "
            f"calls={calls} tokens=64 ms={milliseconds:.6f} "
            f"mtp_cycles={cycles} mtp_accepted={accepted} "
            f"mtp_committed={committed} pos={position} invalid=0"
        )
    return "\n".join(lines) + "\n"


def synthetic_summary() -> str:
    return (
        "ds4-server: s55-m0-summary req=chatcmpl-7 complete=1 invalid=0 "
        "prompt=200000 gen=1422 expected_gen=1422 final_pos=201422 "
        "mtp_segments=8 mtp_tokens=512 mtp_ms=25600.000000 "
        "mtp_cycles=320 mtp_accepted=192 mtp_committed=512 "
        "nomtp_segments=8 nomtp_tokens=512 nomtp_ms=20480.000000 "
        f"wash_segments=5 tail_nomtp_tokens=14 output_sha1={'a' * 40} "
        f"token_sha1={'b' * 40} "
        f"trajectory_sha1={'c' * 40} "
        "finish=length\n"
    )


def synthetic_h25_log() -> str:
    lines = []
    measured = 0
    position = 200_000
    for sequence, kind in enumerate(planned_segments(2, "h25")):
        is_measured = kind in {"baseline", "candidate"}
        measured_index = measured if is_measured else -1
        if is_measured:
            measured += 1
        if kind == "baseline":
            milliseconds, cycles, accepted, committed, calls = 3200.0, 40, 24, 64, 40
        elif kind == "candidate":
            milliseconds, cycles, accepted, committed, calls = 2560.0, 40, 24, 64, 40
        elif kind == "warm":
            milliseconds, cycles, accepted, committed, calls = 3200.0, 39, 24, 63, 40
        else:
            milliseconds, cycles, accepted, committed, calls = 3200.0, 40, 24, 64, 40
        position += 64
        lines.append(
            "ds4-server: s55-m0-segment req=chatcmpl-7 "
            f"seq={sequence} experiment=h25-pair-ab kind={kind} "
            f"measured={measured_index} calls={calls} tokens=64 "
            f"ms={milliseconds:.6f} mtp_cycles={cycles} "
            f"mtp_accepted={accepted} mtp_committed={committed} "
            f"pos={position} invalid=0"
        )
    return "\n".join(lines) + "\n"


def synthetic_h25_summary() -> str:
    return synthetic_summary().replace(
        "invalid=0 prompt=200000 gen=1422 expected_gen=1422 final_pos=201422",
        "invalid=0 experiment=h25-pair-ab prompt=200000 gen=1430 "
        "expected_gen=1430 final_pos=201430",
    ).replace("tail_nomtp_tokens=14", "tail_nomtp_tokens=22")


class S55M0ScheduleTest(unittest.TestCase):
    def test_repeat_two_is_balanced_and_wash_separated(self) -> None:
        segments = planned_segments(2)

        self.assertEqual(segments[0], "warm")
        self.assertEqual(
            [kind for kind in segments if kind in {"mtp", "nomtp"}],
            list("ABBABAABABBABAAB".replace("A", "mtp,")
                 .replace("B", "nomtp,").strip(",").split(",")),
        )
        self.assertEqual(segments.count("mtp"), 8)
        self.assertEqual(segments.count("nomtp"), 8)
        self.assertEqual(segments.count("wash"), 5)

        for index, kind in enumerate(segments):
            if kind == "mtp" and index > 0 and segments[index - 1] == "nomtp":
                self.fail("NOMTP to scored-MTP transition lacks a wash")

    def test_completion_budget_covers_every_possible_mtp_overshoot(self) -> None:
        segments = planned_segments(2)
        mtp_like = sum(kind != "nomtp" for kind in segments)

        self.assertEqual(max_completion_tokens(64, 2), len(segments) * 64 + mtp_like)
        self.assertEqual(max_completion_tokens(64, 2), 1422)

    def test_h25_schedule_has_two_speculative_arms_and_full_overshoot_budget(self) -> None:
        segments = planned_segments(2, "h25")

        self.assertEqual(segments.count("baseline"), 8)
        self.assertEqual(segments.count("candidate"), 8)
        self.assertEqual(segments.count("wash"), 5)
        self.assertEqual(max_completion_tokens(64, 2, "h25"), 1430)

        result = analyze_segment_log(
            synthetic_h25_log(), "chatcmpl-7", 64, 2, "h25"
        )
        self.assertEqual(result["baseline"]["committed"], 512)
        self.assertEqual(result["candidate"]["committed"], 512)
        self.assertAlmostEqual(result["speedup_percent"], 25.0)
        self.assertAlmostEqual(result["paired"]["mean_delta_ms_per_token"], 10.0)
        self.assertEqual(classify_verdict(result, "h25"), "CONFIRMED")
        summary = parse_summary(
            synthetic_h25_summary(), "chatcmpl-7", result, 200_000, 1430
        )
        self.assertEqual(summary["tail_nomtp_tokens"], 22)

    def test_invalid_schedule_inputs_are_rejected(self) -> None:
        for repeats in (0, 1, 65):
            with self.subTest(repeats=repeats):
                with self.assertRaises(ValueError):
                    planned_segments(repeats)
        with self.assertRaises(ValueError):
            max_completion_tokens(31, 2)
        with self.assertRaisesRegex(ValueError, "512"):
            max_completion_tokens(32, 2)

    def test_segment_log_scores_only_balanced_measured_arms(self) -> None:
        result = analyze_segment_log(synthetic_log(), "chatcmpl-7", 64, 2)

        self.assertEqual(result["mtp"]["tokens"], 512)
        self.assertEqual(result["nomtp"]["tokens"], 512)
        self.assertAlmostEqual(result["mtp"]["ms_per_token"], 50.0)
        self.assertAlmostEqual(result["nomtp"]["ms_per_token"], 40.0)
        self.assertAlmostEqual(result["mtp"]["tokens_per_cycle"], 1.6)
        self.assertAlmostEqual(result["mtp"]["ms_per_cycle"], 80.0)
        self.assertAlmostEqual(result["mtp_penalty_percent"], 25.0)
        self.assertEqual(result["paired"]["n"], 8)
        self.assertEqual(result["paired"]["mean_delta_ms_per_token"], 10.0)
        self.assertEqual(result["paired"]["mean_ci95"], [10.0, 10.0])
        self.assertAlmostEqual(result["paired"]["two_sided_sign_p"], 0.0078125)

    def test_segment_log_rejects_missing_wash_or_counter_mismatch(self) -> None:
        log = synthetic_log()
        without_wash = "\n".join(
            line for line in log.splitlines()
            if not ("seq=4 " in line and "kind=wash" in line)
        )
        with self.assertRaisesRegex(RuntimeError, "schedule"):
            analyze_segment_log(without_wash, "chatcmpl-7", 64, 2)
        with self.assertRaisesRegex(RuntimeError, "committed"):
            analyze_segment_log(
                log.replace("mtp_committed=64", "mtp_committed=63", 1),
                "chatcmpl-7", 64, 2,
            )

    def test_segment_log_preserves_unstable_control_evidence(self) -> None:
        analysis = analyze_segment_log(
            synthetic_log().replace("kind=nomtp measured=1 calls=64 "
                                    "tokens=64 ms=2560.000000",
                                    "kind=nomtp measured=1 calls=64 "
                                    "tokens=64 ms=3200.000000"),
            "chatcmpl-7", 64, 2,
        )

        self.assertFalse(analysis["control_stability"]["passed"])
        self.assertGreater(analysis["control_stability"]["spread_percent"], 5.0)
        self.assertEqual(classify_verdict(analysis), "REJECTED_UNSTABLE")

    def test_summary_binds_output_state_and_segment_totals(self) -> None:
        analysis = analyze_segment_log(synthetic_log(), "chatcmpl-7", 64, 2)
        summary = parse_summary(
            synthetic_summary(), "chatcmpl-7", analysis, 200_000, 1422
        )
        self.assertEqual(summary["final_position"], 201_422)
        self.assertEqual(summary["tail_nomtp_tokens"], 14)
        self.assertEqual(summary["output_sha1"], "a" * 40)

        with self.assertRaisesRegex(RuntimeError, "invalid"):
            parse_summary(
                synthetic_summary().replace("invalid=0", "invalid=1"),
                "chatcmpl-7", analysis, 200_000, 1422,
            )
        with self.assertRaisesRegex(RuntimeError, "tail"):
            parse_summary(
                synthetic_summary().replace("tail_nomtp_tokens=14", "tail_nomtp_tokens=13"),
                "chatcmpl-7", analysis, 200_000, 1422,
            )

    def test_token_id_artifact_is_exact_and_hash_bound(self) -> None:
        encoded = ",".join(str(token) for token in range(1422))
        digest = hashlib.sha1(encoded.encode()).hexdigest()
        log = (
            "ds4-server: s55-m0-token-ids req=chatcmpl-7 count=1422 "
            f"sha1={digest} ids={encoded}\n"
        )
        record = parse_token_ids(log, "chatcmpl-7", 1422, digest)
        self.assertEqual(record["count"], 1422)
        self.assertEqual(record["first"], 0)
        self.assertEqual(record["last"], 1421)

        with self.assertRaisesRegex(RuntimeError, "hash"):
            parse_token_ids(log.replace("ids=0,", "ids=9,"),
                            "chatcmpl-7", 1422, digest)

    def test_real_timestamped_logs_and_thermal_samples_parse(self) -> None:
        timestamped = "".join(f"0901 12:00:00 {line}\n"
                              for line in synthetic_log().splitlines())
        result = analyze_segment_log(timestamped, "chatcmpl-7", 64, 2)
        self.assertEqual(result["mtp"]["tokens"], 512)

        telemetry = "\n".join(
            f"[monitor 2026-09-01 12:00:{index * 15:02d}] health=ok "
            "fans=fan0=5300/5349,fan1=5700/5777RPM "
            f"gpu_power={100 + index}.0W sys_power=140.0W "
            f"gpu_temp={70 + index}.0C cpu_temp=65.0C "
            "ram_used=120.0GiB swap_used=0.0GiB"
            for index in range(3)
        )
        start = monitor_timestamp("2026-09-01 12:00:00")
        thermal = parse_telemetry(telemetry, start, start + 20.0)
        self.assertEqual(thermal["samples"], 3)
        self.assertEqual(thermal["gpu_temp_c"], [70.0, 72.0])

        with self.assertRaisesRegex(RuntimeError, "fans"):
            parse_telemetry(telemetry + "\nwarning=fans-below-max",
                            start, start + 20.0)

        with self.assertRaisesRegex(RuntimeError, "fan telemetry"):
            parse_telemetry(telemetry.replace("fan0=", "left="),
                            start, start + 20.0)
        with self.assertRaisesRegex(RuntimeError, "fan telemetry"):
            parse_telemetry(telemetry.replace("5300/5349", "4000/5349"),
                            start, start + 20.0)
        with self.assertRaisesRegex(RuntimeError, "sensor telemetry"):
            parse_telemetry(telemetry.replace("gpu_power=100.0W", "gpu_power=0.0W"),
                            start, start + 20.0)

        with self.assertRaisesRegex(RuntimeError, "gap"):
            parse_telemetry(telemetry.replace("12:00:15", "12:00:25"),
                            start, start + 20.0)

        with self.assertRaisesRegex(RuntimeError, "request start"):
            parse_telemetry(telemetry, start - 10.0, start + 20.0)

        dropped_fan = telemetry.splitlines()
        dropped_fan[1] = dropped_fan[1].replace(
            "fan0=5300/5349,fan1=5700/5777RPM",
            "fan0=5300/5349RPM",
        )
        with self.assertRaisesRegex(RuntimeError, "topology"):
            parse_telemetry("\n".join(dropped_fan), start, start + 20.0)

    def test_response_requires_text_and_binds_generated_bytes(self) -> None:
        response = {
            "id": "chatcmpl-7",
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "exact output"},
            }],
            "usage": {"prompt_tokens": 200_000, "completion_tokens": 1422},
        }
        raw = json.dumps(response).encode()
        result = validate_response(response, raw, 1422)
        self.assertEqual(
            result["generated_text_sha1"],
            hashlib.sha1(b"exact output").hexdigest(),
        )

        response["choices"][0]["message"]["content"] = 0
        with self.assertRaisesRegex(RuntimeError, "text"):
            validate_response(response, json.dumps(response).encode(), 1422)

    def test_verdict_requires_material_significant_effect(self) -> None:
        analysis = {
            "control_stability": {"passed": True},
            "nomtp": {"ms_per_token": 40.0},
            "paired": {"mean_ci95": [2.0, 3.0], "two_sided_sign_p": 0.01},
        }
        self.assertEqual(classify_verdict(analysis), "CONFIRMED")
        analysis["paired"] = {"mean_ci95": [0.1, 0.2], "two_sided_sign_p": 0.01}
        self.assertEqual(classify_verdict(analysis), "NEEDS_MORE_DATA")
        analysis["paired"] = {"mean_ci95": [2.0, 3.0], "two_sided_sign_p": 0.2}
        self.assertEqual(classify_verdict(analysis), "NEEDS_MORE_DATA")

    def test_lsof_parser_preserves_fd_device_inode_and_path(self) -> None:
        records = parse_lsof_records(
            "p42\nftxt\nD0x100000e\ni123\nn/tmp/ds4-server\n"
            "f2\nD0x100000e\ni456\nn/tmp/server.log\n"
        )
        self.assertEqual(records[0], {
            "fd": "txt", "device": 0x100000E, "inode": 123,
            "path": "/tmp/ds4-server",
        })
        self.assertEqual(records[1]["fd"], "2")

    def test_loaded_model_may_be_a_numeric_lsof_fd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model, log = (root / name for name in
                                  ("ds4-server", "model.gguf", "server.log"))
            for path in (binary, model, log):
                path.write_bytes(b"x")
            records = []
            for fd, path in (
                ("txt", binary), ("txt", model), ("4r", model),
                ("1", log), ("2", log),
            ):
                stat = path.stat()
                records.append(
                    f"f{fd}\nD{stat.st_dev}\ni{stat.st_ino}\nn{path}\n"
                )
            completed = SimpleNamespace(stdout="".join(records))
            provenance = {
                "binary": {"path": str(binary), "device": binary.stat().st_dev,
                           "inode": binary.stat().st_ino},
                "model": {"path": str(model), "device": model.stat().st_dev,
                          "inode": model.stat().st_ino},
            }
            identity = {"device": log.stat().st_dev, "inode": log.stat().st_ino}
            with mock.patch("run_s55_m0.subprocess.run", return_value=completed):
                loaded = validate_loaded_files(42, provenance, identity)
            self.assertEqual(loaded["model"]["fd"], "4r")
            self.assertEqual(
                [record["fd"] for record in loaded["model_mappings"]],
                ["txt"],
            )

    def test_m0_artifacts_must_stay_outside_the_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            with self.assertRaisesRegex(RuntimeError, "outside"):
                validate_artifact_paths(repo, repo / "m0-result.json")
            validate_artifact_paths(repo, root / "m0-result.json")

    def test_open_provenance_rejects_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            raw = b'{"schema":"test"}\n'
            path.write_bytes(raw)
            with path.open("rb") as source:
                frozen = os.fstat(source.fileno())
                validate_open_provenance(source, path, raw, frozen)
                replacement = path.with_suffix(".new")
                replacement.write_bytes(raw)
                replacement.replace(path)
                with self.assertRaisesRegex(RuntimeError, "replaced"):
                    validate_open_provenance(source, path, raw, frozen)

    def test_m0_requires_canonical_argv_and_no_disk_kv(self) -> None:
        binary = "/tmp/ds4-server"
        model = "/tmp/model.gguf"
        args = SimpleNamespace(
            url="http://127.0.0.1:8000", segment_tokens=64, repeats=2,
            experiment="m0",
        )
        stats = {
            "ctx_size": 262_144,
            "slot_count": 1,
            "mtp": {"active": True, "width": 2, "timing": False,
                    "counters": False},
            "s55_m0": {"enabled": True, "segment_tokens": 64, "repeats": 2,
                       "experiment": "matched-nomtp",
                       "claimed": False},
            "kv_disk": {"enabled": False},
        }
        provenance = {
            "binary": {"path": binary},
            "model": {"path": model},
        }
        argv = [
            str(Path(binary).resolve()), "--metal", "--model",
            str(Path(model).resolve()), "--mtp",
            "--s55-m0-segment-tokens", "64", "--s55-m0-repeats", "2",
            "--ctx", "262144", "--tokens", "32768", "--power", "100",
            "--host", "127.0.0.1", "--port", "8000",
        ]
        validate_m0_stats(stats, 64, 2)
        validate_m0_argv(argv, args, stats, provenance)

        args.experiment = "h25"
        stats["s55_m0"]["experiment"] = "h25-pair-ab"
        h25_argv = argv[:9] + ["--s55-h25-pair-ab"] + argv[9:]
        validate_m0_stats(stats, 64, 2, "h25")
        validate_m0_argv(h25_argv, args, stats, provenance)
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            validate_m0_argv(h25_argv + ["--quality"], args, stats, provenance)
        stats["kv_disk"]["enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "disk KV"):
            validate_m0_stats(stats, 64, 2, "h25")


if __name__ == "__main__":
    unittest.main()
