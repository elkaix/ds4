#!/usr/bin/env python3
"""Run and validate one-request S55 M0 or H25 balanced diagnostics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from run_s55_200_phase2 import (
    INSTANCE_RE as RUNTIME_INSTANCE_RE,
    MAX_CONTEXT,
    MIN_CONTEXT,
    MODEL_ALIAS,
    SAMPLING,
    atomic_publish_bytes,
    file_record,
    freeze_provenance,
    load_provenance_bytes,
    log_identity,
    prompt_manifest,
    read_log_range,
    request_json,
    runtime_provenance,
    sha256_bytes,
    sha256_json,
    validate_file_record,
    wait_for_idle,
    write_or_print,
)


MEASURED_PATTERN = (
    "mtp",
    "nomtp",
    "nomtp",
    "mtp",
    "nomtp",
    "mtp",
    "mtp",
    "nomtp",
)

SEGMENT_RE = re.compile(
    r"^(?:\d{4} \d{2}:\d{2}:\d{2} )?"
    r"ds4-server: s55-m0-segment req=(?P<request>\S+) "
    r"seq=(?P<sequence>\d+) "
    r"(?:experiment=(?P<experiment>matched-nomtp|h25-pair-ab) )?"
    r"kind=(?P<kind>warm|mtp|nomtp|baseline|candidate|wash) "
    r"measured=(?P<measured>-?\d+) calls=(?P<calls>\d+) "
    r"tokens=(?P<tokens>\d+) ms=(?P<milliseconds>[0-9.]+) "
    r"mtp_cycles=(?P<cycles>\d+) mtp_accepted=(?P<accepted>\d+) "
    r"mtp_committed=(?P<committed>\d+) pos=(?P<position>\d+) "
    r"invalid=(?P<invalid>[01])$",
    re.MULTILINE,
)
SUMMARY_RE = re.compile(
    r"^(?:\d{4} \d{2}:\d{2}:\d{2} )?"
    r"ds4-server: s55-m0-summary req=(?P<request>\S+) "
    r"complete=(?P<complete>[01]) invalid=(?P<invalid>[01]) "
    r"(?:experiment=(?P<experiment>matched-nomtp|h25-pair-ab) )?"
    r"prompt=(?P<prompt>\d+) gen=(?P<generated>\d+) "
    r"expected_gen=(?P<expected>\d+) final_pos=(?P<position>\d+) "
    r"mtp_segments=(?P<mtp_segments>\d+) mtp_tokens=(?P<mtp_tokens>\d+) "
    r"mtp_ms=(?P<mtp_ms>[0-9.]+) mtp_cycles=(?P<cycles>\d+) "
    r"mtp_accepted=(?P<accepted>\d+) mtp_committed=(?P<committed>\d+) "
    r"nomtp_segments=(?P<nomtp_segments>\d+) "
    r"nomtp_tokens=(?P<nomtp_tokens>\d+) nomtp_ms=(?P<nomtp_ms>[0-9.]+) "
    r"wash_segments=(?P<washes>\d+) tail_nomtp_tokens=(?P<tail>\d+) "
    r"output_sha1=(?P<output>[0-9a-f]{40}) "
    r"token_sha1=(?P<token_sha1>[0-9a-f]{40}) "
    r"trajectory_sha1=(?P<trajectory_sha1>[0-9a-f]{40}) "
    r"finish=(?P<finish>\S+)$",
    re.MULTILINE,
)
TOKEN_RE = re.compile(
    r"^(?:\d{4} \d{2}:\d{2}:\d{2} )?"
    r"ds4-server: s55-m0-token-ids req=(?P<request>\S+) "
    r"count=(?P<count>\d+) sha1=(?P<sha1>[0-9a-f]{40}) "
    r"ids=(?P<ids>[0-9,-]*)$",
    re.MULTILINE,
)
START_RE = re.compile(
    r"^(?:\d{4} \d{2}:\d{2}:\d{2} )?"
    r"ds4-server: s55-m0-start req=(?P<request>\S+) "
    r"prompt=(?P<prompt>\d+) segment_tokens=(?P<segment>\d+) "
    r"repeats=(?P<repeats>\d+) "
    r"(?:experiment=(?P<experiment>matched-nomtp|h25-pair-ab) )?"
    r"expected_gen=(?P<expected>\d+) "
    r"valid=(?P<valid>[01])$",
    re.MULTILINE,
)
TELEMETRY_RE = re.compile(
    r"^\[monitor (?P<timestamp>[^]]+)] .*?fans=(?P<fans>\S+) "
    r"gpu_power=(?P<gpu_power>[0-9.]+)W "
    r"sys_power=(?P<sys_power>[0-9.]+)W "
    r"gpu_temp=(?P<gpu_temp>[0-9.]+)C "
    r"cpu_temp=(?P<cpu_temp>[0-9.]+)C "
    r"ram_used=(?P<ram>[0-9.]+)GiB "
    r"swap_used=(?P<swap>[0-9.]+)GiB",
    re.MULTILINE,
)
FANS_RE = re.compile(r"fan\d+=\d+/\d+(?:,fan\d+=\d+/\d+)*RPM")


def experiment_arms(experiment: str) -> tuple[str, str, str]:
    if experiment == "m0":
        return "mtp", "nomtp", "matched-nomtp"
    if experiment == "h25":
        return "baseline", "candidate", "h25-pair-ab"
    raise ValueError(f"unknown S55 M0 experiment: {experiment}")


def planned_segments(repeats: int, experiment: str = "m0") -> list[str]:
    if not 2 <= repeats <= 64:
        raise ValueError("M0 repeats must be between 2 and 64")
    arm_a, arm_b, _ = experiment_arms(experiment)
    measured = [arm_a if kind == "mtp" else arm_b
                for kind in MEASURED_PATTERN] * repeats
    segments = ["warm"]
    previous = ""
    for kind in measured:
        if previous == arm_b and kind == arm_a:
            segments.append("wash")
        segments.append(kind)
        previous = kind
    return segments


def max_completion_tokens(
    segment_tokens: int, repeats: int, experiment: str = "m0"
) -> int:
    if not 32 <= segment_tokens <= 4096:
        raise ValueError("M0 segment size must be between 32 and 4096 tokens")
    if segment_tokens * 4 * repeats < 512:
        raise ValueError("M0 needs at least 512 scored tokens per arm")
    _, arm_b, _ = experiment_arms(experiment)
    segments = planned_segments(repeats, experiment)
    mtp_like = len(segments) if experiment == "h25" else sum(
        kind != arm_b for kind in segments
    )
    return len(segments) * segment_tokens + mtp_like


def analyze_segment_log(
    log: str, request_id: str, segment_tokens: int, repeats: int,
    experiment: str = "m0",
) -> dict[str, Any]:
    arm_a, arm_b, log_experiment = experiment_arms(experiment)
    expected = planned_segments(repeats, experiment)
    matches = [match for match in SEGMENT_RE.finditer(log)
               if match["request"] == request_id]
    if len(matches) != len(expected):
        raise RuntimeError(
            f"M0 schedule has {len(matches)} segments, expected {len(expected)}"
        )

    totals = {
        arm: {"segments": 0, "tokens": 0, "milliseconds": 0.0,
              "cycles": 0, "accepted": 0, "committed": 0}
        for arm in (arm_a, arm_b)
    }
    measured_blocks: list[tuple[str, float]] = []
    measured_index = 0
    last_position = -1
    scheduled_tokens = 0
    for sequence, (match, expected_kind) in enumerate(zip(matches, expected)):
        values = {
            key: int(match[key])
            for key in (
                "sequence", "measured", "calls", "tokens", "cycles",
                "accepted", "committed", "position", "invalid",
            )
        }
        values["milliseconds"] = float(match["milliseconds"])
        kind = match["kind"]
        expected_measured = measured_index if kind in {arm_a, arm_b} else -1
        if kind in {arm_a, arm_b}:
            measured_index += 1
        if (
            values["sequence"] != sequence
            or kind != expected_kind
            or values["measured"] != expected_measured
            or values["invalid"] != 0
            or values["calls"] <= 0
            or values["milliseconds"] <= 0.0
            or values["position"] <= last_position
            or (match["experiment"] is not None and
                match["experiment"] != log_experiment)
            or (experiment == "h25" and match["experiment"] is None)
        ):
            raise RuntimeError(f"M0 schedule mismatch at segment {sequence}")
        last_position = values["position"]
        scheduled_tokens += values["tokens"]

        maximum = (
            segment_tokens
            if experiment == "m0" and kind == arm_b
            else segment_tokens + 1
        )
        if not segment_tokens <= values["tokens"] <= maximum:
            raise RuntimeError(f"M0 token boundary mismatch at segment {sequence}")
        if values["accepted"] > values["cycles"] or (
            values["committed"] != values["cycles"] + values["accepted"]
        ):
            raise RuntimeError(f"M0 committed counter mismatch at segment {sequence}")
        if kind == arm_a and values["committed"] != values["tokens"]:
            raise RuntimeError(f"M0 committed tokens do not cover segment {sequence}")
        if experiment == "m0" and kind == arm_b and (
            values["calls"] != values["tokens"]
            or values["cycles"] != 0
            or values["committed"] != 0
        ):
            raise RuntimeError(f"M0 NOMTP counter mismatch at segment {sequence}")
        if experiment == "h25" and kind == arm_b and (
            values["committed"] != values["tokens"]
        ):
            raise RuntimeError(f"H25 candidate counters do not cover segment {sequence}")
        if kind == "warm" and (
            values["committed"] + 1 != values["tokens"]
        ):
            raise RuntimeError(f"M0 warmup did not contain one seed at segment {sequence}")
        if kind == "wash" and (
            values["committed"] + (0 if experiment == "h25" else 1)
            != values["tokens"]
        ):
            raise RuntimeError(f"M0 wash counter mismatch at segment {sequence}")

        if kind in totals:
            arm = totals[kind]
            arm["segments"] += 1
            arm["tokens"] += values["tokens"]
            arm["milliseconds"] += values["milliseconds"]
            arm["cycles"] += values["cycles"]
            arm["accepted"] += values["accepted"]
            arm["committed"] += values["committed"]
            measured_blocks.append(
                (kind, values["milliseconds"] / values["tokens"])
            )

    for kind in (arm_a, arm_b):
        arm = totals[kind]
        if arm["segments"] != repeats * 4 or arm["tokens"] < 512:
            raise RuntimeError(f"M0 {kind} arm lacks eight blocks and 512 tokens")
        arm["ms_per_token"] = arm["milliseconds"] / arm["tokens"]
        arm["tokens_per_second"] = 1000.0 / arm["ms_per_token"]
    speculative_arms = (arm_a, arm_b) if experiment == "h25" else (arm_a,)
    for kind in speculative_arms:
        arm = totals[kind]
        if arm["cycles"] <= 0 or arm["committed"] != arm["tokens"]:
            raise RuntimeError(f"M0 {kind} counters do not cover the scored tokens")
        arm["tokens_per_cycle"] = arm["committed"] / arm["cycles"]
        arm["ms_per_cycle"] = arm["milliseconds"] / arm["cycles"]
    control_arms = (arm_a, arm_b) if experiment == "h25" else (arm_b,)
    by_arm = {}
    for kind in control_arms:
        costs = [cost for block_kind, cost in measured_blocks
                 if block_kind == kind]
        by_arm[kind] = {
            "spread_percent": (max(costs) / min(costs) - 1.0) * 100.0,
            "worst_adjacent_percent": max(
                (abs(right / left - 1.0) * 100.0
                 for left, right in zip(costs, costs[1:])),
                default=0.0,
            ),
        }
    control_spread = max(value["spread_percent"] for value in by_arm.values())
    worst_adjacent = max(value["worst_adjacent_percent"]
                         for value in by_arm.values())
    totals["control_stability"] = {
        "spread_percent": control_spread,
        "worst_adjacent_percent": worst_adjacent,
        "limit_percent": 5.0,
        "passed": control_spread <= 5.0 and worst_adjacent <= 5.0,
        "by_arm": by_arm,
    }
    if experiment == "m0":
        totals["mtp_penalty_percent"] = (
            totals[arm_a]["ms_per_token"] /
            totals[arm_b]["ms_per_token"] - 1.0
        ) * 100.0
    else:
        totals["speedup_percent"] = (
            totals[arm_a]["ms_per_token"] /
            totals[arm_b]["ms_per_token"] - 1.0
        ) * 100.0
    totals["wash_segments"] = expected.count("wash")
    totals["scheduled_tokens"] = scheduled_tokens
    deltas = []
    for index in range(0, len(measured_blocks), 2):
        pair = measured_blocks[index:index + 2]
        if len(pair) != 2 or {kind for kind, _ in pair} != {arm_a, arm_b}:
            raise RuntimeError(f"M0 schedule pair {index // 2} is not balanced")
        costs = dict(pair)
        deltas.append(costs[arm_a] - costs[arm_b])
    rng = random.Random(0)
    bootstrap = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas)
        for _ in range(20_000)
    )
    nonzero = [value for value in deltas if value != 0.0]
    minority = min(sum(value > 0.0 for value in nonzero),
                   sum(value < 0.0 for value in nonzero))
    sign_p = min(
        1.0,
        2.0 * sum(math.comb(len(nonzero), k) for k in range(minority + 1)) /
        (2 ** len(nonzero)),
    ) if nonzero else 1.0
    totals["paired"] = {
        "n": len(deltas),
        "mean_delta_ms_per_token": statistics.fmean(deltas),
        "median_delta_ms_per_token": statistics.median(deltas),
        "range_delta_ms_per_token": [min(deltas), max(deltas)],
        "mean_ci95": [bootstrap[499], bootstrap[19_499]],
        "two_sided_sign_p": sign_p,
    }
    totals["experiment"] = experiment
    return totals


def parse_summary(
    log: str,
    request_id: str,
    analysis: dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    matches = [match for match in SUMMARY_RE.finditer(log)
               if match["request"] == request_id]
    if len(matches) != 1:
        raise RuntimeError(f"M0 log contains {len(matches)} request summaries")
    match = matches[0]
    experiment = analysis.get("experiment", "m0")
    arm_a, arm_b, log_experiment = experiment_arms(experiment)
    if (
        (match["experiment"] is not None and
         match["experiment"] != log_experiment)
        or (experiment == "h25" and match["experiment"] is None)
    ):
        raise RuntimeError("M0 summary identifies the wrong experiment")
    ints = {
        key: int(match[key])
        for key in (
            "complete", "invalid", "prompt", "generated", "expected",
            "position", "mtp_segments", "mtp_tokens", "cycles", "accepted",
            "committed", "nomtp_segments", "nomtp_tokens", "washes",
            "tail",
        )
    }
    if ints["invalid"] or not ints["complete"] or match["finish"] != "length":
        raise RuntimeError("M0 summary is incomplete or invalid")
    if (
        ints["prompt"] != prompt_tokens
        or ints["generated"] != completion_tokens
        or ints["expected"] != completion_tokens
        or ints["position"] != prompt_tokens + completion_tokens
    ):
        raise RuntimeError("M0 summary does not preserve the exact final state")
    expected_ints = {
        "mtp_segments": analysis[arm_a]["segments"],
        "mtp_tokens": analysis[arm_a]["tokens"],
        "cycles": analysis[arm_a]["cycles"],
        "accepted": analysis[arm_a]["accepted"],
        "committed": analysis[arm_a]["committed"],
        "nomtp_segments": analysis[arm_b]["segments"],
        "nomtp_tokens": analysis[arm_b]["tokens"],
        "washes": analysis["wash_segments"],
        "tail": completion_tokens - analysis["scheduled_tokens"],
    }
    for key, expected in expected_ints.items():
        if ints[key] != expected:
            raise RuntimeError(f"M0 summary disagrees with segments at {key}")
    for key, expected in (
        ("mtp_ms", analysis[arm_a]["milliseconds"]),
        ("nomtp_ms", analysis[arm_b]["milliseconds"]),
    ):
        if not math.isclose(float(match[key]), expected, abs_tol=1e-4):
            raise RuntimeError(f"M0 summary disagrees with segments at {key}")
    return {
        "prompt_tokens": ints["prompt"],
        "generated_tokens": ints["generated"],
        "final_position": ints["position"],
        "tail_nomtp_tokens": ints["tail"],
        "output_sha1": match["output"],
        "token_sha1": match["token_sha1"],
        "trajectory_sha1": match["trajectory_sha1"],
        "finish_reason": match["finish"],
    }


def parse_token_ids(
    log: str, request_id: str, expected_count: int, expected_sha1: str
) -> dict[str, Any]:
    matches = [match for match in TOKEN_RE.finditer(log)
               if match["request"] == request_id]
    if len(matches) != 1:
        raise RuntimeError(f"M0 log contains {len(matches)} token-ID artifacts")
    match = matches[0]
    encoded = match["ids"]
    digest = hashlib.sha1(encoded.encode("ascii")).hexdigest()
    if match["sha1"] != expected_sha1 or digest != expected_sha1:
        raise RuntimeError("M0 token-ID artifact hash mismatch")
    tokens = [] if not encoded else [int(value) for value in encoded.split(",")]
    if int(match["count"]) != expected_count or len(tokens) != expected_count:
        raise RuntimeError("M0 token-ID artifact count mismatch")
    if any(token < 0 for token in tokens):
        raise RuntimeError("M0 token-ID artifact contains an invalid token")
    return {
        "count": len(tokens),
        "sha1": digest,
        "first": tokens[0] if tokens else None,
        "last": tokens[-1] if tokens else None,
        "_encoded": encoded,
    }


def validate_m0_stats(
    stats: dict[str, Any], segment_tokens: int, repeats: int,
    experiment: str = "m0",
) -> None:
    mtp = stats.get("mtp")
    m0 = stats.get("s55_m0")
    _, _, log_experiment = experiment_arms(experiment)
    if mtp != {"active": True, "width": 2, "timing": False, "counters": False}:
        raise RuntimeError(f"server is not clean width-2 MTP: {mtp!r}")
    if not isinstance(m0, dict) or {
        key: m0.get(key)
        for key in ("enabled", "segment_tokens", "repeats", "experiment")
    } != {
        "enabled": True, "segment_tokens": segment_tokens, "repeats": repeats,
        "experiment": log_experiment,
    } or not isinstance(m0.get("claimed"), bool):
        raise RuntimeError(f"server is not configured for this M0 schedule: {m0!r}")
    if stats.get("kv_disk", {}).get("enabled") is not False:
        raise RuntimeError("M0 requires disk KV disabled")


def validate_current_m0_instance(
    prefix: str, stats: dict[str, Any], segment_tokens: int, repeats: int,
    experiment: str = "m0",
) -> None:
    instances = list(RUNTIME_INSTANCE_RE.finditer(prefix))
    if len(instances) != 1:
        raise RuntimeError("fresh M0 log must contain one runtime marker")
    instance = instances[0]
    if instance["instance"] != stats["instance"] or int(instance["pid"]) != stats["pid"]:
        raise RuntimeError("M0 log and /stats identify different server processes")
    _, _, log_experiment = experiment_arms(experiment)
    marker = (
        f"ds4-server: S55 M0 enabled segment_tokens={segment_tokens} "
        f"repeats={repeats} experiment={log_experiment} "
        f"expected_gen={max_completion_tokens(segment_tokens, repeats, experiment)} "
        "schedule=ABBA-BAAB wash=arm-b-to-arm-a "
        "tail=matched-nomtp counters=derived"
    )
    if prefix.count(marker) != 1:
        raise RuntimeError("server log lacks the exact M0 startup marker")
    launcher_marker = (
        "S55:        H25 contract-locked"
        if experiment == "h25" else "S55:        M0 contract-locked"
    )
    if prefix.count(launcher_marker) != 1:
        raise RuntimeError("launcher did not prove the clean M0 environment")
    if prefix.count("Fans at maximum:") != 1:
        raise RuntimeError("launcher did not prove one maximum-fan gate")
    if "warning=fans-below-max" in prefix.split("Fans at maximum:", 1)[1]:
        raise RuntimeError("fans fell below maximum before M0")
    if "s55-m0-start" in prefix or "s55-m0-summary" in prefix:
        raise RuntimeError("fresh M0 process already contains a diagnostic request")


def monitor_timestamp(value: str) -> float:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError as exc:
        raise RuntimeError(f"invalid M0 monitor timestamp: {value}") from exc


def telemetry_sample(match: re.Match[str]) -> dict[str, Any]:
    fans = match["fans"]
    if not FANS_RE.fullmatch(fans):
        raise RuntimeError("invalid M0 fan telemetry")
    topology = []
    for index, value in enumerate(fans.removesuffix("RPM").split(",")):
        name, rpms = value.split("=", 1)
        actual_text, maximum_text = rpms.split("/", 1)
        actual, maximum = int(actual_text), int(maximum_text)
        if name != f"fan{index}" or maximum <= 0 or actual < maximum * 0.98:
            raise RuntimeError("invalid M0 fan telemetry")
        topology.append((name, maximum))
    sample = {
        "timestamp": match["timestamp"],
        "wall_time": monitor_timestamp(match["timestamp"]),
        "fans": fans,
        "fan_topology": topology,
        **{
            key: float(match[key])
            for key in ("gpu_power", "sys_power", "gpu_temp", "cpu_temp", "ram", "swap")
        },
    }
    required_positive = ("gpu_power", "sys_power", "gpu_temp", "cpu_temp", "ram")
    if (
        any(not math.isfinite(sample[key]) or sample[key] <= 0.0
            for key in required_positive)
        or not math.isfinite(sample["swap"])
        or sample["swap"] < 0.0
    ):
        raise RuntimeError("invalid M0 sensor telemetry")
    return sample


def parse_telemetry(
    segment: str, request_started_wall: float, request_finished_wall: float
) -> dict[str, Any]:
    samples = [telemetry_sample(match) for match in TELEMETRY_RE.finditer(segment)]
    monitor_samples = sum(
        line.startswith("[monitor ") and " health=" in line
        for line in segment.splitlines()
    )
    if monitor_samples != len(samples):
        raise RuntimeError("invalid M0 sensor telemetry")
    if len(samples) < 3:
        raise RuntimeError(f"M0 needs at least three thermal samples, got {len(samples)}")
    if "warning=fans-below-max" in segment:
        raise RuntimeError("fans fell below maximum during M0")
    times = [sample["wall_time"] for sample in samples]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise RuntimeError("M0 monitor timestamps are not strictly increasing")
    if times[0] > request_started_wall or request_started_wall - times[0] > 20.0:
        raise RuntimeError("M0 telemetry does not cover request start")
    if times[-1] < request_finished_wall or times[-1] - request_finished_wall > 20.0:
        raise RuntimeError("M0 telemetry does not cover request completion")
    if any(right - left > 20.0 for left, right in zip(times, times[1:])):
        raise RuntimeError("M0 telemetry has a gap over 20 seconds")
    topology = samples[0]["fan_topology"]
    if any(sample["fan_topology"] != topology for sample in samples[1:]):
        raise RuntimeError("M0 fan topology changed during the request")
    return {
        "samples": len(samples),
        "fan_topology": [
            {"name": name, "maximum_rpm": maximum}
            for name, maximum in topology
        ],
        "gpu_power_w": [min(s["gpu_power"] for s in samples),
                        max(s["gpu_power"] for s in samples)],
        "sys_power_w": [min(s["sys_power"] for s in samples),
                        max(s["sys_power"] for s in samples)],
        "gpu_temp_c": [min(s["gpu_temp"] for s in samples),
                       max(s["gpu_temp"] for s in samples)],
        "cpu_temp_c": [min(s["cpu_temp"] for s in samples),
                       max(s["cpu_temp"] for s in samples)],
        "ram_used_gib": [min(s["ram"] for s in samples), max(s["ram"] for s in samples)],
        "swap_used_gib": [min(s["swap"] for s in samples), max(s["swap"] for s in samples)],
        "max_sample_gap_seconds": max(
            (right - left for left, right in zip(times, times[1:])), default=0.0
        ),
    }


def wait_for_pre_request_telemetry(
    server_log: Path,
    identity: dict[str, int],
    timeout: float = 25.0,
) -> tuple[bytes, str]:
    deadline = time.monotonic() + timeout
    while True:
        prefix = read_log_range(server_log, 0, None, identity)
        text = prefix.decode("utf-8", errors="replace")
        matches = list(TELEMETRY_RE.finditer(text))
        if matches:
            sample = telemetry_sample(matches[-1])
            age = time.time() - sample["wall_time"]
            if 0.0 <= age <= 20.0:
                if "warning=fans-below-max" in text[matches[-1].end():]:
                    raise RuntimeError("fans fell below maximum before M0")
                return prefix, matches[-1].group(0)
        if time.monotonic() >= deadline:
            raise RuntimeError("M0 has no valid pre-request telemetry sample")
        time.sleep(0.25)


def wait_for_post_request_telemetry(
    server_log: Path,
    identity: dict[str, int],
    request_finished_wall: float,
    timeout: float = 25.0,
) -> bytes:
    deadline = time.monotonic() + timeout
    while True:
        segment = read_log_range(server_log, identity["size"], None, identity)
        timestamps = [monitor_timestamp(match["timestamp"])
                      for match in TELEMETRY_RE.finditer(
                          segment.decode("utf-8", errors="replace"))]
        if timestamps and timestamps[-1] >= request_finished_wall:
            return segment
        if time.monotonic() >= deadline:
            raise RuntimeError("M0 monitor did not sample request completion")
        time.sleep(0.25)


def request_json_raw(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], bytes]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"request failed: {url}: {exc}") from exc
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("error"):
        raise RuntimeError(f"invalid M0 response: {value!r}")
    return value, raw


def validate_response(
    response: dict[str, Any], raw: bytes, expected_tokens: int
) -> dict[str, Any]:
    request_id = response.get("id")
    usage = response.get("usage")
    choices = response.get("choices")
    if not isinstance(request_id, str) or not re.fullmatch(r"chatcmpl-\d+", request_id):
        raise RuntimeError("M0 response lacks a request ID")
    if not isinstance(usage, dict) or not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("M0 response lacks one choice and usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not MIN_CONTEXT <= prompt_tokens <= MAX_CONTEXT:
        raise RuntimeError(f"M0 actual context is not real 200K: {prompt_tokens}")
    if completion_tokens != expected_tokens:
        raise RuntimeError(
            f"M0 generated {completion_tokens} tokens, expected {expected_tokens}"
        )
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "length":
        raise RuntimeError("M0 terminated before its fixed diagnostic window")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("M0 response lacks the assistant message")
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if content is None:
        content = ""
    if reasoning is None:
        reasoning = ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise RuntimeError("M0 response output is not text")
    if reasoning or message.get("tool_calls"):
        raise RuntimeError("M0 response changed the no-thinking/no-tools contract")
    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": choice["finish_reason"],
        "decoded_output_sha256": sha256_json(
            {"content": content, "reasoning_content": reasoning}
        ),
        "generated_text_sha1": hashlib.sha1(content.encode("utf-8")).hexdigest(),
        "_generated_text": content,
        "raw_response_sha256": sha256_bytes(raw),
    }


def classify_verdict(
    analysis: dict[str, Any], experiment: str = "m0"
) -> str:
    if not analysis["control_stability"]["passed"]:
        return "REJECTED_UNSTABLE"
    paired = analysis["paired"]
    lower, upper = paired["mean_ci95"]
    arm_a, arm_b, _ = experiment_arms(experiment)
    reference = arm_b if experiment == "m0" else arm_a
    material = analysis[reference]["ms_per_token"] * 0.03
    significant = paired["two_sided_sign_p"] <= 0.05
    if significant and lower > material:
        return "CONFIRMED"
    if significant and upper < -material:
        return "REJECTED"
    return "NEEDS_MORE_DATA"


def freeze_m0_provenance(
    repo: Path | None, binary: Path, model: Path
) -> dict[str, Any]:
    provenance = freeze_provenance(
        repo, binary, model, driver_path=Path(__file__)
    )
    provenance["harness"]["dependencies"] = [
        file_record(Path(__file__).with_name("run_s55_200_phase2.py"),
                    hash_contents=True)
    ]
    return provenance


def validate_m0_dependencies(provenance: dict[str, Any]) -> list[dict[str, str]]:
    dependencies = provenance.get("harness", {}).get("dependencies")
    expected = Path(__file__).with_name("run_s55_200_phase2.py").resolve()
    if not isinstance(dependencies, list) or len(dependencies) != 1:
        raise RuntimeError("M0 provenance lacks its Phase 2 helper dependency")
    path = validate_file_record(dependencies[0], rehash=True)
    if path != expected:
        raise RuntimeError("M0 provenance binds the wrong helper dependency")
    return [{"path": str(path), "sha256": dependencies[0]["sha256"]}]


def validate_open_provenance(
    source: Any,
    path: Path,
    frozen_raw: bytes,
    frozen_stat: os.stat_result,
) -> None:
    source.seek(0)
    if source.read() != frozen_raw:
        raise RuntimeError("open M0 provenance changed during the request")
    open_stat = os.fstat(source.fileno())
    path_stat = path.resolve(strict=True).stat()
    frozen_identity = (
        frozen_stat.st_dev, frozen_stat.st_ino,
        frozen_stat.st_size, frozen_stat.st_mtime_ns,
    )
    if (
        (open_stat.st_dev, open_stat.st_ino, open_stat.st_size,
         open_stat.st_mtime_ns) != frozen_identity
        or (path_stat.st_dev, path_stat.st_ino, path_stat.st_size,
            path_stat.st_mtime_ns) != frozen_identity
    ):
        raise RuntimeError("M0 provenance path was replaced during the request")


def parse_lsof_records(output: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in output.splitlines():
        if not line:
            continue
        tag, value = line[0], line[1:]
        if tag == "f":
            if current:
                records.append(current)
            current = {"fd": value}
        elif tag == "D":
            current["device"] = int(value, 0)
        elif tag == "i":
            current["inode"] = int(value)
        elif tag == "n":
            current["path"] = value
    if current:
        records.append(current)
    return [record for record in records
            if {"fd", "device", "inode", "path"} <= record.keys()]


def validate_loaded_files(
    pid: int,
    provenance: dict[str, Any],
    log_record: dict[str, int],
) -> dict[str, Any]:
    result = subprocess.run(
        ["lsof", "-a", "-p", str(pid), "-FfDin"],
        check=True,
        capture_output=True,
        text=True,
    )
    records = parse_lsof_records(result.stdout)

    def matches(record: dict[str, Any], frozen: dict[str, Any]) -> bool:
        return (
            (record["device"], record["inode"]) ==
            (frozen["device"], frozen["inode"])
            and ("path" not in frozen or
                 Path(record["path"]).resolve() == Path(frozen["path"]).resolve())
        )

    binary = [record for record in records
              if record["fd"] == "txt" and matches(record, provenance["binary"])]
    model = [record for record in records if matches(record, provenance["model"])]
    model_fds = [record for record in model
                 if re.fullmatch(r"[0-9]+[rwu]?", record["fd"])]
    model_mappings = [record for record in model if record["fd"] == "txt"]
    logs = [record for record in records
            if record["fd"] in {"1", "2"} and matches(record, log_record)]
    if (
        len(binary) != 1
        or len(model_fds) != 1
        or len(model) != len(model_fds) + len(model_mappings)
        or {record["fd"] for record in logs} != {"1", "2"}
    ):
        raise RuntimeError("server loaded files do not match frozen binary/model/log")
    return {
        "binary": binary[0],
        "model": model_fds[0],
        "model_mappings": model_mappings,
        "log_fds": sorted(record["fd"] for record in logs),
    }


def validate_artifact_paths(repo: Path, *paths: Path) -> None:
    source_root = repo.resolve(strict=True)
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved == source_root or source_root in resolved.parents:
            raise RuntimeError(f"M0 artifact must be outside the source tree: {resolved}")


def validate_m0_argv(
    argv: list[str], args: argparse.Namespace, stats: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    if args.url.rstrip("/") != "http://127.0.0.1:8000":
        raise RuntimeError("M0 requires the local production endpoint on port 8000")
    experiment = getattr(args, "experiment", "m0")
    experiment_arms(experiment)
    expected = [
        str(Path(provenance["binary"]["path"]).resolve()),
        "--metal", "--model", str(Path(provenance["model"]["path"]).resolve()),
        "--mtp",
        "--s55-m0-segment-tokens", str(args.segment_tokens),
        "--s55-m0-repeats", str(args.repeats),
    ]
    if experiment == "h25":
        expected.append("--s55-h25-pair-ab")
    expected.extend([
        "--ctx", "262144", "--tokens", "32768",
        "--power", "100", "--host", "127.0.0.1", "--port", "8000",
    ])
    if argv != expected:
        raise RuntimeError(f"server argv is not the canonical M0 command: {argv!r}")
    if stats["ctx_size"] != 262_144 or stats["slot_count"] != 1:
        raise RuntimeError("M0 requires one 262144-token Metal session")


def run_m0(args: argparse.Namespace) -> dict[str, Any]:
    provenance_path = args.provenance.expanduser().resolve(strict=True)
    with provenance_path.open("rb") as provenance_file:
        provenance_stat = os.fstat(provenance_file.fileno())
        provenance_raw = provenance_file.read()
        return _run_m0(
            args, provenance_file, provenance_raw, provenance_stat
        )


def _run_m0(
    args: argparse.Namespace,
    provenance_file: Any,
    provenance_raw: bytes,
    provenance_stat: os.stat_result,
) -> dict[str, Any]:
    output = args.output
    log_output = output.with_suffix(output.suffix + ".server.log")
    response_output = output.with_suffix(output.suffix + ".response.json")
    for path in (output, log_output, response_output):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite M0 artifact: {path}")

    repo = Path(__file__).resolve().parents[1]
    validate_artifact_paths(
        repo, args.server_log, args.provenance,
        output, log_output, response_output,
    )
    prompt, prompt_record = prompt_manifest(repo)
    provenance, provenance_sha256 = load_provenance_bytes(provenance_raw)
    dependencies_before = validate_m0_dependencies(provenance)
    frozen_driver = Path(str(provenance["harness"]["driver"]["path"])).resolve()
    if frozen_driver != Path(__file__).resolve():
        raise RuntimeError("M0 provenance does not bind this driver")

    base_url = args.url.rstrip("/")
    before_hash = wait_for_idle(base_url, "m0", args.idle_timeout, 0.0)
    validate_m0_stats(before_hash, args.segment_tokens, args.repeats,
                      args.experiment)
    runtime_before = runtime_provenance(
        before_hash, provenance, provenance_sha256, "m0", rehash_model=False
    )
    argv = runtime_before["argv"]
    validate_m0_argv(argv, args, before_hash, provenance)

    before = wait_for_idle(base_url, "m0", args.idle_timeout, args.idle_settle)
    validate_m0_stats(before, args.segment_tokens, args.repeats,
                      args.experiment)
    if before["instance"] != before_hash["instance"] or before["pid"] != before_hash["pid"]:
        raise RuntimeError("server restarted during M0 provenance validation")
    if before["requests"] != 0 or before["generated_tokens"] != 0 or before["live_tokens"] != 0:
        raise RuntimeError("M0 requires a fresh one-request server process")
    if before["s55_m0"]["claimed"]:
        raise RuntimeError("M0 process already claimed its one inference request")

    identity = log_identity(args.server_log)
    loaded_before = validate_loaded_files(before["pid"], provenance, identity)
    models = request_json(base_url + "/v1/models", timeout=30.0)
    prefix_bytes, initial_telemetry = wait_for_pre_request_telemetry(
        args.server_log, identity
    )
    identity["size"] = len(prefix_bytes)
    prefix_hash = sha256_bytes(prefix_bytes)
    validate_current_m0_instance(
        prefix_bytes.decode("utf-8", errors="replace"), before,
        args.segment_tokens, args.repeats, args.experiment,
    )
    expected_tokens = max_completion_tokens(
        args.segment_tokens, args.repeats, args.experiment
    )
    payload = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": expected_tokens,
        **SAMPLING,
    }
    request_started_wall = time.time()
    started = time.monotonic()
    response, response_bytes = request_json_raw(
        base_url + "/v1/chat/completions", payload, args.request_timeout
    )
    wall_seconds = time.monotonic() - started
    request_finished_wall = time.time()
    after = wait_for_idle(base_url, "m0", args.idle_timeout, 0.0)
    validate_m0_stats(after, args.segment_tokens, args.repeats,
                      args.experiment)
    if after["instance"] != before["instance"] or after["pid"] != before["pid"]:
        raise RuntimeError("server restarted during M0")
    if after["requests"] - before["requests"] != 1:
        raise RuntimeError("another inference request overlapped M0")
    if after["s55_m0"]["claimed"] is not True:
        raise RuntimeError("M0 server did not seal its one inference request")
    cache_delta = {
        key: after["cache"][key] - before["cache"][key]
        for key in ("hits", "cold")
    }
    if cache_delta != {"hits": 0, "cold": 1}:
        raise RuntimeError("M0 must perform one cold checkpoint-bound prefill")

    if sha256_bytes(read_log_range(args.server_log, 0, identity["size"], identity)) != prefix_hash:
        raise RuntimeError("server log prefix changed during M0")
    segment_bytes = wait_for_post_request_telemetry(
        args.server_log, identity, request_finished_wall
    )
    segment = segment_bytes.decode("utf-8", errors="replace")
    if RUNTIME_INSTANCE_RE.search(segment):
        raise RuntimeError("server restarted inside M0 log segment")
    usage = validate_response(response, response_bytes, expected_tokens)
    starts = [match for match in START_RE.finditer(segment)
              if match["request"] == usage["request_id"]]
    if len(starts) != 1 or any((
        int(starts[0]["prompt"]) != usage["prompt_tokens"],
        int(starts[0]["segment"]) != args.segment_tokens,
        int(starts[0]["repeats"]) != args.repeats,
        starts[0]["experiment"] != experiment_arms(args.experiment)[2],
        int(starts[0]["expected"]) != expected_tokens,
        int(starts[0]["valid"]) != 1,
    )):
        raise RuntimeError("M0 start marker does not bind the HTTP request")
    analysis = analyze_segment_log(
        segment, usage["request_id"], args.segment_tokens, args.repeats,
        args.experiment,
    )
    summary = parse_summary(
        segment, usage["request_id"], analysis,
        usage["prompt_tokens"], expected_tokens,
    )
    token_ids = parse_token_ids(
        segment, usage["request_id"], expected_tokens,
        summary["token_sha1"],
    )
    encoded_token_ids = token_ids.pop("_encoded")
    summary["token_ids"] = token_ids
    generated_text = usage.pop("_generated_text")
    if usage["generated_text_sha1"] != summary["output_sha1"]:
        raise RuntimeError("HTTP response text does not match raw generated output")
    trajectory_sha1 = hashlib.sha1(
        encoded_token_ids.encode("ascii") + b"\0" + generated_text.encode("utf-8")
    ).hexdigest()
    if trajectory_sha1 != summary["trajectory_sha1"]:
        raise RuntimeError("HTTP output and token IDs are not one exact trajectory")
    if after["live_tokens"] != summary["final_position"]:
        raise RuntimeError("/stats does not preserve the M0 final model position")
    telemetry = parse_telemetry(
        initial_telemetry + "\n" + segment,
        request_started_wall, request_finished_wall,
    )

    runtime = dict(runtime_before)
    runtime.update({
        "instance": before["instance"],
        "pid": before["pid"],
        "model": before.get("model"),
        "model_path": before.get("model_path"),
        "ctx_size": before["ctx_size"],
        "slot_count": before["slot_count"],
        "mtp": before["mtp"],
        "s55_m0": before["s55_m0"],
        "models_sha256": sha256_json(models),
        "provenance_path": str(args.provenance.resolve(strict=True)),
        "dependencies": dependencies_before,
        "loaded_files": loaded_before,
    })
    verdict = classify_verdict(analysis, args.experiment)
    report = {
        "schema": "s55-200-h25-v1" if args.experiment == "h25" else "s55-200-m0-v1",
        "scope": ("exact-h25-real-decode-ab"
                  if args.experiment == "h25"
                  else "matched-state-nomtp-diagnostic-only"),
        "full_s55_status": "NOT_EVALUATED",
        "s55_credit_tps": 0.0,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "prompt": prompt_record,
        "request": {
            "model": MODEL_ALIAS,
            "sampling": SAMPLING,
            "segment_tokens": args.segment_tokens,
            "repeats": args.repeats,
            "experiment": args.experiment,
            "generated_tokens": expected_tokens,
            "cache_before": before["cache"],
            "cache_after": after["cache"],
        },
        "usage": usage,
        "state": summary,
        "analysis": analysis,
        "thermal": telemetry,
        "runtime": runtime,
        "wall_seconds": wall_seconds,
        "artifacts": {
            "server_log": str(log_output.resolve()),
            "server_log_sha256": sha256_bytes(segment_bytes),
            "raw_response": str(response_output.resolve()),
            "raw_response_sha256": sha256_bytes(response_bytes),
        },
    }
    stable_keys = ("instance", "pid", "requests", "generated_tokens", "live_tokens", "cache")
    published: list[Path] = []
    try:
        for path, data in (
            (log_output, segment_bytes),
            (response_output, response_bytes),
        ):
            atomic_publish_bytes(path, data)
            published.append(path)
        raw_artifacts = {
            "server_log": file_record(log_output, hash_contents=True),
            "raw_response": file_record(response_output, hash_contents=True),
        }
        if (
            raw_artifacts["server_log"]["sha256"] != sha256_bytes(segment_bytes)
            or raw_artifacts["raw_response"]["sha256"] != sha256_bytes(response_bytes)
        ):
            raise RuntimeError("published M0 raw artifact changed on reopen")

        final_stats = wait_for_idle(base_url, "m0", args.idle_timeout, 0.0)
        validate_m0_stats(final_stats, args.segment_tokens, args.repeats,
                          args.experiment)
        if any(final_stats[key] != after[key] for key in stable_keys):
            raise RuntimeError("server state changed before M0 provenance revalidation")
        runtime_after = runtime_provenance(
            final_stats, provenance, provenance_sha256, "m0", rehash_model=True
        )
        sealed_stats = wait_for_idle(base_url, "m0", args.idle_timeout, 0.0)
        validate_m0_stats(sealed_stats, args.segment_tokens, args.repeats,
                          args.experiment)
        runtime_sealed = runtime_provenance(
            sealed_stats, provenance, provenance_sha256, "m0", rehash_model=False
        )
        dependencies_after = validate_m0_dependencies(provenance)
        loaded_after = validate_loaded_files(sealed_stats["pid"], provenance, identity)
        validate_open_provenance(
            provenance_file, args.provenance.expanduser(),
            provenance_raw, provenance_stat,
        )
        for record in raw_artifacts.values():
            validate_file_record(record, rehash=True)
        if (
            any(sealed_stats[key] != after[key] for key in stable_keys)
            or {**runtime_after, "model_rehashed_now": False} != runtime_before
            or runtime_sealed != runtime_before
            or dependencies_after != dependencies_before
            or loaded_after != loaded_before
        ):
            raise RuntimeError("M0 evidence changed during final sealing")
        report["artifacts"]["server_log_record"] = raw_artifacts["server_log"]
        report["artifacts"]["raw_response_record"] = raw_artifacts["raw_response"]
        report["runtime"]["post_run_validation"] = {
            "stats_stable_after_rehash": True,
            "source_binary_model_rehashed": True,
            "source_binary_sealed_after_model_hash": True,
            "raw_artifacts_reopened_and_rehashed": True,
            "provenance_fd_stable": True,
            "dependencies_rehashed": dependencies_after,
            "loaded_files": loaded_after,
        }
        rendered = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        atomic_publish_bytes(output, rendered)
        published.append(output)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help="verify the frozen natural 200K prompt")

    freeze = sub.add_parser("freeze", help="freeze exact M0 provenance")
    freeze.add_argument("--repo", type=Path)
    freeze.add_argument("--server-binary", type=Path, required=True)
    freeze.add_argument("--model-file", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="run one wash-separated M0 request")
    run.add_argument("--server-log", type=Path, required=True)
    run.add_argument("--provenance", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--segment-tokens", type=int, default=64)
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--experiment", choices=("m0", "h25"), default="m0")
    run.add_argument("--url", default="http://127.0.0.1:8000")
    run.add_argument("--idle-timeout", type=float, default=600.0)
    run.add_argument("--idle-settle", type=float, default=8.0)
    run.add_argument("--request-timeout", type=float, default=3600.0)

    args = parser.parse_args()
    if args.command == "manifest":
        _, manifest = prompt_manifest(Path(__file__).resolve().parents[1])
        write_or_print(manifest, None)
        return 0
    if args.command == "freeze":
        validate_artifact_paths(
            args.server_binary.expanduser().resolve(strict=True).parent,
            args.output,
        )
        write_or_print(
            freeze_m0_provenance(
                args.repo, args.server_binary, args.model_file,
            ),
            args.output,
        )
        return 0
    planned_segments(args.repeats, args.experiment)
    max_completion_tokens(args.segment_tokens, args.repeats, args.experiment)
    if args.idle_timeout <= 0 or args.idle_settle < 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive and idle settle non-negative")
    write_or_print(run_m0(args), None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
