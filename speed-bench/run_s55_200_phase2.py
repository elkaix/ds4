#!/usr/bin/env python3
"""Freeze provenance, run, and combine S55-200 Phase 2 diagnostic arms.

This drives the existing ds4-server over HTTP; it is not an inference harness.
The diagnostic contract is immutable: a frozen natural prompt, one GLM model
alias, greedy width-2 MTP, 512 generated tokens, and an actual 190K-205K prompt.
It derives a cycle budget only; it does not certify the complete S55 ladder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from analyze_glm53_mtp import MARKER, contiguous_runs, parse_cycles, summarize
from build_s55_corpus import build_corpus


INSTRUCTION = (
    "Read the reference material that follows and then write a detailed "
    "technical summary of it.\n\n"
)
CORPUS_CHARS = 769_585
EXPECTED_PROMPT_BYTES = 769_966
EXPECTED_PROMPT_SHA256 = (
    "9b5b32d34afe5cd67c55805a67aadcfcbc0b26b407c0dc13215d0b92f48f0be2"
)
MODEL_ALIAS = "glm-5.2-chat"
GENERATED_TOKENS = 512
MIN_CONTEXT = 190_000
MAX_CONTEXT = 205_000
TARGET_TPS = 55.0
EXPECTED_MODEL_BYTES = 96_505_818_432
EXPECTED_MODEL_NAME = "GLM-5.3-Flash-UNCEN-Q2.gguf"
SAMPLING = {
    "temperature": 0,
    "top_p": 1.0,
    "ignore_eos": True,
    "reasoning_effort": "none",
    "seed": 1,
    "stream": False,
}
SOURCE_STATUS_PATHS = (
    "*.c",
    "*.h",
    "*.m",
    "*.metal",
    "Makefile",
    "run-glm-ds4.sh",
    "speed-bench/*.py",
)

INSTANCE_RE = re.compile(
    r"ds4-server: runtime instance=(?P<instance>[0-9a-f]{16}) "
    r"pid=(?P<pid>[0-9]+) schema=(?P<schema>[0-9]+)"
)
SUMMARY_RE = re.compile(
    r"ds4-server: decode-summary req=(?P<request_id>chatcmpl-[0-9]+) "
    r"prompt=(?P<prompt>[0-9]+) gen=(?P<generated>[0-9]+) "
    r"seconds=(?P<seconds>[0-9.]+) tps=(?P<tps>[0-9.]+) "
    r"seed_gen=(?P<seed_generated>[0-9]+) "
    r"steady_gen=(?P<steady_generated>[0-9]+) "
    r"steady_seconds=(?P<steady_seconds>[0-9.]+) "
    r"mtp_active=(?P<mtp_active>[01]) mtp_width=(?P<mtp_width>[0-9]+) "
    r"mtp_cycles=(?P<cycles>[0-9]+) "
    r"mtp_accepted=(?P<accepted>[0-9]+) "
    r"mtp_rejected=(?P<rejected>[0-9]+) "
    r"mtp_committed=(?P<committed>[0-9]+)"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(data)


def atomic_publish_bytes(path: Path, data: bytes) -> None:
    """Publish one immutable artifact without exposing a partial final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    linked = False
    temporary_stat = None
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        temporary_stat = temporary.stat()
        os.link(temporary, path)
        linked = True
        final_stat = path.stat()
        if (final_stat.st_dev, final_stat.st_ino, final_stat.st_size) != (
            temporary_stat.st_dev, temporary_stat.st_ino, len(data)
        ):
            raise RuntimeError(f"published artifact identity mismatch: {path}")
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite: {path}") from exc
    except BaseException:
        if linked and temporary_stat is not None:
            final_stat = path.stat()
            if (final_stat.st_dev, final_stat.st_ino) == (
                temporary_stat.st_dev, temporary_stat.st_ino
            ):
                path.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_manifest(repo: Path) -> tuple[str, dict[str, Any]]:
    corpus = build_corpus(repo).decode("utf-8", errors="replace")
    prompt = INSTRUCTION + corpus[:CORPUS_CHARS]
    encoded = prompt.encode("utf-8")
    digest = sha256_bytes(encoded)
    if len(encoded) != EXPECTED_PROMPT_BYTES or digest != EXPECTED_PROMPT_SHA256:
        raise RuntimeError(
            f"canonical prompt mismatch: bytes={len(encoded)} sha256={digest}"
        )
    return prompt, {
        "corpus_chars": CORPUS_CHARS,
        "prompt_chars": len(prompt),
        "prompt_bytes": len(encoded),
        "prompt_sha256": digest,
    }


def file_record(path: Path, *, hash_contents: bool) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"not a regular file: {resolved}")
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    if hash_contents:
        record["sha256"] = sha256_file(resolved)
        after = resolved.stat()
        stable = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        frozen = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stable != frozen:
            raise RuntimeError(f"file changed while hashing: {resolved}")
    return record


def tracked_tree_sha256(repo: Path) -> str:
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True
    ).stdout
    names = sorted(name for name in listing.split(b"\0") if name)
    digest = hashlib.sha256()
    for raw_name in names:
        path = repo / os.fsdecode(raw_name)
        digest.update(len(raw_name).to_bytes(8, "big"))
        digest.update(raw_name)
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
            digest.update(b"L" + len(payload).to_bytes(8, "big") + payload)
            continue
        digest.update(b"F")
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def source_status(repo: Path) -> str:
    return git_text(
        repo,
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        *SOURCE_STATUS_PATHS,
    )


def freeze_provenance(
    repo: Path | None, binary: Path, model: Path,
    driver_path: Path | None = None,
) -> dict[str, Any]:
    binary_record = file_record(binary, hash_contents=True)
    repo_hint = repo if repo is not None else Path(binary_record["path"]).parent
    repo = Path(git_text(repo_hint, "rev-parse", "--show-toplevel").strip())
    if Path(binary_record["path"]).parent != repo.resolve():
        raise RuntimeError("server binary must live at the frozen source root")
    driver = (driver_path or Path(__file__)).resolve(strict=True)
    if driver.parents[1] != repo.resolve():
        raise RuntimeError("S55 driver and server binary must use one checkout")
    model_identity = file_record(model, hash_contents=False)
    if Path(model_identity["path"]).name != EXPECTED_MODEL_NAME:
        raise RuntimeError(f"wrong GLM-5.3 artifact name: {model_identity['path']}")
    if model_identity["size"] != EXPECTED_MODEL_BYTES:
        raise RuntimeError(
            f"wrong GLM-5.3 Q2 artifact size: {model_identity['size']} != "
            f"{EXPECTED_MODEL_BYTES}"
        )
    print(f"Hashing full model: {model}", file=os.sys.stderr, flush=True)
    model_record = file_record(model, hash_contents=True)
    head = git_text(repo, "rev-parse", "HEAD").strip()
    status = source_status(repo)
    tree_sha256 = tracked_tree_sha256(repo)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if head != git_text(repo, "rev-parse", "HEAD").strip() or status != source_status(repo):
        raise RuntimeError("source tree changed while freezing provenance")
    return {
        "schema": "s55-200-phase0-provenance-v1",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "repo": str(repo.resolve()),
            "head": head,
            "tracked_tree_sha256": tree_sha256,
            "head_diff_sha256": sha256_bytes(diff),
            "status": status.splitlines(),
        },
        "binary": binary_record,
        "model": model_record,
        "harness": {
            "driver": file_record(driver, hash_contents=True),
            "launcher": file_record(repo / "run-glm-ds4.sh", hash_contents=True),
        },
        "machine": {
            "node": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
    }


def load_provenance_bytes(raw: bytes) -> tuple[dict[str, Any], str]:
    provenance = json.loads(raw)
    if not isinstance(provenance, dict) or provenance.get("schema") != (
        "s55-200-phase0-provenance-v1"
    ):
        raise RuntimeError("invalid Phase 0 provenance schema")
    for key in ("source", "binary", "model", "harness", "machine"):
        if not isinstance(provenance.get(key), dict):
            raise RuntimeError(f"Phase 0 provenance is missing {key}")
    for key in ("binary", "model"):
        digest = str(provenance[key].get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Phase 0 provenance has no full {key} SHA-256")
    for key in ("driver", "launcher"):
        digest = str(provenance["harness"].get(key, {}).get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Phase 0 provenance has no full {key} SHA-256")
    return provenance, sha256_bytes(raw)


def load_provenance(path: Path) -> tuple[dict[str, Any], str]:
    return load_provenance_bytes(path.read_bytes())


def validate_file_record(record: dict[str, Any], *, rehash: bool) -> Path:
    path = Path(str(record["path"])).resolve(strict=True)
    stat = path.stat()
    actual = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    for key, value in actual.items():
        if value != record.get(key):
            raise RuntimeError(f"frozen file changed at {key}: {path}")
    if rehash:
        digest = sha256_file(path)
        after = path.stat()
        stable = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        frozen = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stable != frozen:
            raise RuntimeError(f"file changed while rehashing: {path}")
        if digest != record.get("sha256"):
            raise RuntimeError(f"frozen file SHA-256 changed: {path}")
    return path


def validate_source_record(record: dict[str, Any]) -> Path:
    repo = Path(str(record["repo"])).resolve(strict=True)
    if git_text(repo, "rev-parse", "HEAD").strip() != record.get("head"):
        raise RuntimeError("frozen source HEAD changed")
    if tracked_tree_sha256(repo) != record.get("tracked_tree_sha256"):
        raise RuntimeError("frozen tracked source tree changed")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if sha256_bytes(diff) != record.get("head_diff_sha256"):
        raise RuntimeError("frozen source diff changed")
    status = source_status(repo)
    if status.splitlines() != record.get("status"):
        raise RuntimeError("frozen source status changed")
    return repo


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(
        payload, separators=(",", ":")
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"request failed: {url}: {exc}") from exc
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError("expected a JSON object response")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result


def validate_stats(stats: dict[str, Any], mode: str) -> None:
    if stats.get("runtime_schema") != 1:
        raise RuntimeError("/stats lacks S55 runtime schema 1")
    if not re.fullmatch(r"[0-9a-f]{16}", str(stats.get("instance", ""))):
        raise RuntimeError("/stats lacks a valid server instance")
    if not isinstance(stats.get("pid"), int) or stats["pid"] <= 1:
        raise RuntimeError("/stats lacks a valid server PID")
    if stats.get("slot_count") != 1:
        raise RuntimeError(f"S55 requires one server slot, got {stats.get('slot_count')}")
    if not isinstance(stats.get("busy"), bool):
        raise RuntimeError("/stats is missing boolean busy")
    if not isinstance(stats.get("queue_depth"), int):
        raise RuntimeError("/stats is missing integer queue_depth")
    if not isinstance(stats.get("requests"), int):
        raise RuntimeError("/stats is missing integer requests")
    if not isinstance(stats.get("generated_tokens"), int):
        raise RuntimeError("/stats is missing integer generated_tokens")
    if not isinstance(stats.get("live_tokens"), int):
        raise RuntimeError("/stats is missing integer live_tokens")
    cache = stats.get("cache")
    if (
        not isinstance(cache, dict)
        or not isinstance(cache.get("hits"), int)
        or not isinstance(cache.get("cold"), int)
    ):
        raise RuntimeError("/stats is missing cache counters")
    if not isinstance(stats.get("ctx_size"), int) or stats["ctx_size"] < (
        MAX_CONTEXT + GENERATED_TOKENS
    ):
        raise RuntimeError("server context capacity cannot hold the S55 request")
    mtp = stats.get("mtp")
    if not isinstance(mtp, dict):
        raise RuntimeError("/stats is missing current-process MTP state")
    if mtp.get("active") is not True or mtp.get("width") != 2:
        raise RuntimeError(f"current process has no active width-2 MTP: {mtp!r}")
    expected_counters = mode != "m0"
    if mtp.get("counters") is not expected_counters:
        raise RuntimeError(
            f"current process does not match the {mode} counter mode"
        )
    if mtp.get("timing") is not (mode == "timing"):
        raise RuntimeError(f"current process does not match the {mode} arm")


def wait_for_idle(
    base_url: str,
    mode: str,
    timeout: float,
    settle: float,
    poll: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    idle_since: float | None = None
    while time.monotonic() < deadline:
        stats = request_json(base_url.rstrip("/") + "/stats", timeout=30.0)
        validate_stats(stats, mode)
        if stats["busy"] or stats["queue_depth"]:
            idle_since = None
        else:
            if idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= settle:
                return stats
        time.sleep(poll)
    raise RuntimeError("server did not remain idle before the measurement")


def log_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    if not path.is_file():
        raise RuntimeError(f"server log is not a regular file: {path}")
    return {"device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size}


def read_log_range(
    path: Path, start: int, end: int | None, identity: dict[str, int]
) -> bytes:
    stat = path.stat()
    if stat.st_dev != identity["device"] or stat.st_ino != identity["inode"]:
        raise RuntimeError("server log was replaced or rotated")
    required = start if end is None else end
    if stat.st_size < required:
        raise RuntimeError("server log was truncated")
    with path.open("rb") as source:
        source.seek(start)
        return source.read() if end is None else source.read(end - start)


def validate_current_instance(prefix: str, stats: dict[str, Any], mode: str) -> str:
    instances = list(INSTANCE_RE.finditer(prefix))
    if len(instances) != 1:
        raise RuntimeError(
            f"fresh server log must contain one runtime marker, got {len(instances)}"
        )
    instance = instances[0]
    if (
        instance["instance"] != stats["instance"]
        or int(instance["pid"]) != stats["pid"]
        or int(instance["schema"]) != 1
    ):
        raise RuntimeError("server log is not bound to the current /stats process")
    markers = [line.strip() for line in prefix.splitlines() if "MTP:" in line]
    if len(markers) != 1:
        raise RuntimeError(f"fresh server log must contain one MTP marker, got {len(markers)}")
    marker = markers[0]
    timing = "+ timing (--mtp-timing)" in marker
    if (
        "enabled (--mtp, width 2)" not in marker
        or "+ counters (--mtp-counters)" not in marker
        or timing != (mode == "timing")
    ):
        raise RuntimeError(f"launcher marker does not match the current {mode} arm")
    if (
        "S55:        Phase 2 contract-locked "
        "(no DS4/Metal/DYLD overrides)" not in prefix
    ):
        raise RuntimeError("launcher did not prove a clean S55 environment")
    fan_markers = [
        line for line in prefix.splitlines() if line.startswith("Fans at maximum:")
    ]
    if len(fan_markers) != 1:
        raise RuntimeError("launcher did not prove one maximum-fan gate")
    if "warning=fans-below-max" in prefix.split(fan_markers[0], 1)[1]:
        raise RuntimeError("fans fell below maximum before the measurement")
    if SUMMARY_RE.search(prefix):
        raise RuntimeError("fresh server log contains a prior decode summary")
    return marker


def option_value(argv: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError(f"server command must contain exactly one {name}")
    return argv[positions[0] + 1]


def runtime_provenance(
    stats: dict[str, Any], provenance: dict[str, Any], provenance_sha256: str,
    mode: str, *, rehash_model: bool = True,
) -> dict[str, Any]:
    validate_source_record(provenance["source"])
    binary = validate_file_record(provenance["binary"], rehash=True)
    model = validate_file_record(provenance["model"], rehash=rehash_model)
    validate_file_record(provenance["harness"]["driver"], rehash=True)
    validate_file_record(provenance["harness"]["launcher"], rehash=True)
    server_model = Path(str(stats.get("model_path", ""))).expanduser().resolve(strict=True)
    if server_model != model:
        raise RuntimeError(f"server model does not match Phase 0: {server_model} != {model}")
    result = subprocess.run(
        ["ps", "-ww", "-p", str(stats["pid"]), "-o", "command="],
        check=True,
        capture_output=True,
        text=True,
    )
    argv = shlex.split(result.stdout.strip())
    if not argv or Path(argv[0]).resolve(strict=True) != binary:
        raise RuntimeError("/stats PID does not execute the frozen binary")
    if Path(option_value(argv, "--model")).expanduser().resolve(strict=True) != model:
        raise RuntimeError("server argv model does not match Phase 0")
    if int(option_value(argv, "--ctx")) != stats["ctx_size"]:
        raise RuntimeError("server argv and /stats disagree on context capacity")
    if "--mtp" not in argv or ("--mtp-timing" in argv) != (mode == "timing"):
        raise RuntimeError("server argv does not prove the requested MTP mode")
    return {
        "provenance_sha256": provenance_sha256,
        "binary_sha256": provenance["binary"]["sha256"],
        "model_sha256": provenance["model"]["sha256"],
        "driver_sha256": provenance["harness"]["driver"]["sha256"],
        "launcher_sha256": provenance["harness"]["launcher"]["sha256"],
        "model_rehashed_now": rehash_model,
        "argv": argv,
        "normalized_argv": [value for value in argv if value != "--mtp-timing"],
    }


def validate_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    choices = response.get("choices")
    request_id = response.get("id")
    if not isinstance(request_id, str) or not re.fullmatch(r"chatcmpl-[0-9]+", request_id):
        raise RuntimeError("chat response is missing its server request ID")
    if not isinstance(usage, dict) or not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("chat response is missing one choice and usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not MIN_CONTEXT <= prompt_tokens <= MAX_CONTEXT:
        raise RuntimeError(
            f"actual context {prompt_tokens} is outside {MIN_CONTEXT}..{MAX_CONTEXT}"
        )
    if completion_tokens != GENERATED_TOKENS:
        raise RuntimeError(
            f"early/short completion: got {completion_tokens}, expected {GENERATED_TOKENS}"
        )
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "length":
        raise RuntimeError(f"finish_reason is not length: {choice!r}")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("chat response is missing the assistant message")
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if content is None:
        content = ""
    if reasoning is None:
        reasoning = ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise RuntimeError("assistant output is not text")
    decoded = {"content": content, "reasoning_content": reasoning}
    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": choice["finish_reason"],
        "decoded_output_sha256": sha256_json(decoded),
    }


def parse_decode_summary(segment: str, usage: dict[str, Any]) -> dict[str, Any]:
    matches = list(SUMMARY_RE.finditer(segment))
    if len(matches) != 1:
        raise RuntimeError(f"request log contains {len(matches)} decode summaries")
    match = matches[0]
    summary: dict[str, Any] = {
        "request_id": match["request_id"],
        "prompt_tokens": int(match["prompt"]),
        "generated_tokens": int(match["generated"]),
        "decode_seconds": float(match["seconds"]),
        "decode_tps": float(match["tps"]),
        "seed_generated_tokens": int(match["seed_generated"]),
        "steady_generated_tokens": int(match["steady_generated"]),
        "steady_decode_seconds": float(match["steady_seconds"]),
        "mtp_active": bool(int(match["mtp_active"])),
        "mtp_width": int(match["mtp_width"]),
        "cycles": int(match["cycles"]),
        "accepted": int(match["accepted"]),
        "rejected": int(match["rejected"]),
        "committed": int(match["committed"]),
    }
    if summary["request_id"] != usage["request_id"]:
        raise RuntimeError("decode summary is not for the HTTP response")
    if summary["prompt_tokens"] != usage["prompt_tokens"]:
        raise RuntimeError("decode summary and response disagree on actual context")
    if summary["generated_tokens"] != GENERATED_TOKENS:
        raise RuntimeError("decode summary is not the exact 512-token request")
    if summary["decode_seconds"] <= 0.0 or not math.isclose(
        summary["decode_tps"],
        GENERATED_TOKENS / summary["decode_seconds"],
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise RuntimeError("decode summary TPS does not match tokens/time")
    if (
        summary["seed_generated_tokens"] != 1
        or summary["steady_generated_tokens"] != GENERATED_TOKENS - 1
        or summary["steady_decode_seconds"] <= 0.0
        or summary["steady_decode_seconds"] >= summary["decode_seconds"]
    ):
        raise RuntimeError("decode summary has no exact 511-token post-seed window")
    if not summary["mtp_active"] or summary["mtp_width"] != 2:
        raise RuntimeError("request did not execute active width-2 MTP")
    if summary["cycles"] <= 0 or summary["accepted"] + summary["rejected"] != summary["cycles"]:
        raise RuntimeError("request MTP counters are inconsistent")
    if summary["committed"] != summary["steady_generated_tokens"]:
        raise RuntimeError("MTP counters do not cover the exact steady window")
    if summary["committed"] != summary["cycles"] + summary["accepted"]:
        raise RuntimeError("greedy width-2 acceptance counters are inconsistent")
    summary["steady_decode_tps"] = (
        summary["steady_generated_tokens"] / summary["steady_decode_seconds"]
    )
    summary["acceptance"] = summary["accepted"] / summary["cycles"]
    summary["tokens_per_cycle"] = summary["committed"] / summary["cycles"]
    return summary


def validate_timing_segment(
    segment: str, prompt_tokens: int, decode: dict[str, Any]
) -> dict[str, Any]:
    cycles = parse_cycles(segment.splitlines())
    runs = contiguous_runs(cycles)
    if len(runs) != 1:
        raise RuntimeError(f"timing segment contains {len(runs)} MTP runs")
    selected = runs[0]
    committed = sum(cycle.committed for cycle in selected)
    if selected[0].pos != prompt_tokens + 1:
        raise RuntimeError("timing run does not begin after the seed token")
    if selected[-1].pos + selected[-1].committed != prompt_tokens + GENERATED_TOKENS:
        raise RuntimeError("timing run does not end at the 512-token frontier")
    if (
        len(selected) != decode["cycles"]
        or committed != decode["committed"]
        or sum(cycle.committed == 2 for cycle in selected) != decode["accepted"]
    ):
        raise RuntimeError("per-cycle timing and request MTP counters disagree")
    return summarize(selected, decode_tps=None, target_tps=TARGET_TPS)


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    output: Path = args.output
    segment_output = output.with_suffix(output.suffix + ".server.log")
    if output.exists() or segment_output.exists():
        raise RuntimeError("refusing to overwrite an existing Phase 2 artifact")
    if not args.server_log.is_file():
        raise RuntimeError(f"server log does not exist: {args.server_log}")

    repo = Path(__file__).resolve().parents[1]
    prompt, manifest = prompt_manifest(repo)
    provenance, provenance_sha256 = load_provenance(args.provenance)
    base_url = args.url.rstrip("/")
    before_hash = wait_for_idle(base_url, args.mode, args.idle_timeout, 0.0)
    runtime = runtime_provenance(
        before_hash, provenance, provenance_sha256, args.mode
    )
    before = wait_for_idle(base_url, args.mode, args.idle_timeout, args.idle_settle)
    if (
        before["instance"] != before_hash["instance"]
        or before["pid"] != before_hash["pid"]
    ):
        raise RuntimeError("server restarted while validating provenance")
    if (
        before["requests"] != 0
        or before["generated_tokens"] != 0
        or before["live_tokens"] != 0
    ):
        raise RuntimeError("Phase 2 arm requires a fresh server with no prior inference")
    identity = log_identity(args.server_log)
    prefix_bytes = read_log_range(args.server_log, 0, identity["size"], identity)
    prefix_sha256 = sha256_bytes(prefix_bytes)
    launch_marker = validate_current_instance(
        prefix_bytes.decode("utf-8", errors="replace"), before, args.mode
    )
    models = request_json(base_url + "/v1/models", timeout=30.0)

    payload = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": GENERATED_TOKENS,
        **SAMPLING,
    }
    started = time.monotonic()
    response = request_json(
        base_url + "/v1/chat/completions",
        payload=payload,
        timeout=args.request_timeout,
    )
    wall_seconds = time.monotonic() - started
    after = wait_for_idle(base_url, args.mode, args.idle_timeout, 0.0)
    if after["instance"] != before["instance"] or after["pid"] != before["pid"]:
        raise RuntimeError("server restarted during the Phase 2 request")
    if after["requests"] - before["requests"] != 1:
        raise RuntimeError("another inference request overlapped the Phase 2 measurement")

    prefix_after = read_log_range(args.server_log, 0, identity["size"], identity)
    if sha256_bytes(prefix_after) != prefix_sha256:
        raise RuntimeError("server log prefix changed during the request")
    segment_bytes = read_log_range(args.server_log, identity["size"], None, identity)
    segment = segment_bytes.decode("utf-8", errors="replace")
    if INSTANCE_RE.search(segment):
        raise RuntimeError("server restarted inside the request log segment")
    if "warning=fans-below-max" in segment:
        raise RuntimeError("fans fell below maximum during the measurement")

    usage = validate_response(response)
    decode = parse_decode_summary(segment, usage)
    if abs(decode["decode_tps"] - float(after.get("last_decode_tps", -1.0))) > 0.011:
        raise RuntimeError("/stats and request summary disagree on decode throughput")
    if args.mode == "timing":
        mtp = validate_timing_segment(segment, usage["prompt_tokens"], decode)
    else:
        if MARKER in segment:
            raise RuntimeError("clean arm contains --mtp-timing output")
        mtp = None

    runtime.update(
        {
            "instance": before["instance"],
            "pid": before["pid"],
            "model": before.get("model"),
            "model_path": str(Path(before["model_path"]).expanduser().resolve()),
            "ctx_size": before["ctx_size"],
            "slot_count": before["slot_count"],
            "mtp": before["mtp"],
            "models_sha256": sha256_json(models),
            "provenance_path": str(
                args.provenance.expanduser().resolve(strict=True)
            ),
        }
    )
    report = {
        "schema": "s55-200-phase2-arm-v2",
        "contract_locked": True,
        "scope": "phase2-cycle-budget-only",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "endpoint": base_url,
        "launch_marker": launch_marker,
        "runtime": runtime,
        "prompt": manifest,
        "request": {
            "model": MODEL_ALIAS,
            "generated_tokens": GENERATED_TOKENS,
            "sampling": SAMPLING,
            "request_id": usage["request_id"],
            "requests_before": before["requests"],
            "requests_after": after["requests"],
            "cache_before": before["cache"],
            "cache_after": after["cache"],
        },
        "usage": usage,
        "decode": decode,
        "wall_seconds": wall_seconds,
        "mtp_timing": mtp,
        "log": {
            "path": str(args.server_log.resolve()),
            "device": identity["device"],
            "inode": identity["inode"],
            "start": identity["size"],
            "end": identity["size"] + len(segment_bytes),
            "prefix_sha256": prefix_sha256,
            "segment_sha256": sha256_bytes(segment_bytes),
        },
    }
    atomic_publish_bytes(segment_output, segment_bytes)
    atomic_publish_bytes(
        output, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    )
    return report


def validate_arm_report(report: dict[str, Any], mode: str) -> None:
    if (
        report.get("schema") != "s55-200-phase2-arm-v2"
        or report.get("contract_locked") is not True
        or report.get("scope") != "phase2-cycle-budget-only"
    ):
        raise ValueError("combine requires contract-locked Phase 2 v2 artifacts")
    if report.get("mode") != mode:
        raise ValueError(f"expected {mode} arm")
    records = tuple(
        report.get(key) for key in ("prompt", "request", "usage", "decode", "runtime", "log")
    )
    if not all(isinstance(value, dict) for value in records):
        raise ValueError("arm artifact is missing required records")
    prompt, request, usage, decode, runtime, log = records
    if prompt.get("prompt_sha256") != EXPECTED_PROMPT_SHA256 or prompt.get("prompt_bytes") != EXPECTED_PROMPT_BYTES:
        raise ValueError("arm does not use the frozen S55 prompt")
    if request.get("model") != MODEL_ALIAS or request.get("generated_tokens") != GENERATED_TOKENS:
        raise ValueError("arm request contract is not immutable")
    if request.get("sampling") != SAMPLING:
        raise ValueError("arm sampling contract is not immutable")
    before = request.get("requests_before")
    after = request.get("requests_after")
    if before != 0 or after != 1:
        raise ValueError("arm does not prove one fresh isolated inference request")
    cache_before = request.get("cache_before")
    cache_after = request.get("cache_after")
    if not isinstance(cache_before, dict) or not isinstance(cache_after, dict):
        raise ValueError("arm lacks cache-state evidence")
    cache_values = [
        cache.get(key)
        for cache in (cache_before, cache_after)
        for key in ("hits", "cold")
    ]
    if not all(isinstance(value, int) for value in cache_values):
        raise ValueError("arm cache-state counters are not integers")
    cache_delta = {
        key: cache_after[key] - cache_before[key]
        for key in ("hits", "cold")
    }
    if sorted(cache_delta.values()) != [0, 1]:
        raise ValueError("arm did not record exactly one cold-or-hit cache outcome")
    if usage.get("request_id") != request.get("request_id"):
        raise ValueError("arm request IDs disagree")
    if usage.get("completion_tokens") != GENERATED_TOKENS or usage.get("finish_reason") != "length":
        raise ValueError("arm is not an exact 512-token length completion")
    if not MIN_CONTEXT <= usage.get("prompt_tokens", 0) <= MAX_CONTEXT:
        raise ValueError("arm is not at real approximately-200K context")
    if decode.get("request_id") != usage["request_id"] or decode.get("prompt_tokens") != usage["prompt_tokens"]:
        raise ValueError("arm decode summary is not request-bound")
    if decode.get("generated_tokens") != GENERATED_TOKENS or decode.get("seed_generated_tokens") != 1:
        raise ValueError("arm decode window is not fixed")
    if decode.get("steady_generated_tokens") != GENERATED_TOKENS - 1 or decode.get("committed") != GENERATED_TOKENS - 1:
        raise ValueError("arm steady MTP window is not exactly 511 tokens")
    cycles = decode.get("cycles")
    accepted = decode.get("accepted")
    rejected = decode.get("rejected")
    if not all(isinstance(value, int) for value in (cycles, accepted, rejected)) or cycles <= 0:
        raise ValueError("arm MTP counters are invalid")
    if accepted + rejected != cycles or decode["committed"] != cycles + accepted:
        raise ValueError("arm MTP counter arithmetic is invalid")
    if decode.get("mtp_active") is not True or decode.get("mtp_width") != 2:
        raise ValueError("arm has no request-level width-2 proof")
    for key in ("decode_seconds", "steady_decode_seconds"):
        if not isinstance(decode.get(key), (int, float)) or decode[key] <= 0:
            raise ValueError(f"arm has no {key}")
    expected_decode_tps = GENERATED_TOKENS / decode["decode_seconds"]
    expected_steady_tps = (GENERATED_TOKENS - 1) / decode["steady_decode_seconds"]
    if not math.isclose(decode.get("decode_tps", -1.0), expected_decode_tps, rel_tol=1e-8):
        raise ValueError("arm decode TPS does not match its exact window")
    if not math.isclose(decode.get("steady_decode_tps", -1.0), expected_steady_tps, rel_tol=1e-8):
        raise ValueError("arm steady TPS does not match its exact window")
    if runtime.get("mtp") != {
        "active": True,
        "counters": True,
        "timing": mode == "timing",
        "width": 2,
    }:
        raise ValueError("arm current-process MTP provenance is invalid")
    for key in (
        "provenance_sha256",
        "binary_sha256",
        "model_sha256",
        "driver_sha256",
        "launcher_sha256",
        "models_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(runtime.get(key, ""))):
            raise ValueError(f"arm runtime lacks {key}")
    if not isinstance(runtime.get("normalized_argv"), list) or not runtime["normalized_argv"]:
        raise ValueError("arm lacks normalized server argv")
    if not Path(str(runtime.get("provenance_path", ""))).is_absolute():
        raise ValueError("arm lacks an absolute provenance path")
    for key in ("prefix_sha256", "segment_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(log.get(key, ""))):
            raise ValueError(f"arm log lacks {key}")
    if mode == "timing" and not isinstance(report.get("mtp_timing"), dict):
        raise ValueError("timing arm lacks per-cycle evidence")
    if mode == "timing":
        mtp_timing = report["mtp_timing"]
        if (
            mtp_timing.get("cycles") != cycles
            or mtp_timing.get("accepted") != accepted
            or mtp_timing.get("rejected") != rejected
            or mtp_timing.get("committed_tokens") != decode["committed"]
            or not math.isclose(
                mtp_timing.get("tokens_per_cycle", -1.0),
                decode["committed"] / cycles,
                rel_tol=1e-12,
            )
        ):
            raise ValueError("timing arm per-cycle evidence disagrees with counters")
    if mode == "clean" and report.get("mtp_timing") is not None:
        raise ValueError("clean arm contains timing evidence")


def validate_combine_evidence(timing: dict[str, Any], clean: dict[str, Any]) -> None:
    validate_arm_report(timing, "timing")
    validate_arm_report(clean, "clean")
    provenance_paths = {
        timing["runtime"]["provenance_path"],
        clean["runtime"]["provenance_path"],
    }
    if len(provenance_paths) != 1:
        raise ValueError("arms reference different provenance files")
    provenance, provenance_sha256 = load_provenance(Path(provenance_paths.pop()))
    if timing["runtime"]["provenance_sha256"] != provenance_sha256:
        raise ValueError("arm provenance hash does not match the reopened artifact")

    validate_source_record(provenance["source"])
    binary = validate_file_record(provenance["binary"], rehash=True)
    model = validate_file_record(provenance["model"], rehash=True)
    driver = validate_file_record(provenance["harness"]["driver"], rehash=True)
    launcher = validate_file_record(provenance["harness"]["launcher"], rehash=True)
    if driver != Path(__file__).resolve():
        raise ValueError("combine is not running the frozen Phase 2 driver")

    expected_runtime = {
        "binary_sha256": provenance["binary"]["sha256"],
        "model_sha256": provenance["model"]["sha256"],
        "driver_sha256": provenance["harness"]["driver"]["sha256"],
        "launcher_sha256": provenance["harness"]["launcher"]["sha256"],
        "model_path": str(model),
    }
    source_root = Path(provenance["source"]["repo"]).resolve(strict=True)
    if binary.parent != source_root:
        raise ValueError("frozen binary no longer belongs to the frozen source root")
    if launcher != source_root / "run-glm-ds4.sh":
        raise ValueError("frozen launcher no longer belongs to the frozen source root")

    for report in (timing, clean):
        mode = report["mode"]
        runtime = report["runtime"]
        for key, expected in expected_runtime.items():
            if runtime.get(key) != expected:
                raise ValueError(f"{mode} arm disagrees with provenance at {key}")
        log = report["log"]
        identity = {"device": log["device"], "inode": log["inode"]}
        path = Path(log["path"]).resolve(strict=True)
        prefix = read_log_range(path, 0, log["start"], identity)
        segment_bytes = read_log_range(
            path, log["start"], log["end"], identity
        )
        if sha256_bytes(prefix) != log["prefix_sha256"]:
            raise ValueError(f"{mode} arm log prefix hash changed")
        if sha256_bytes(segment_bytes) != log["segment_sha256"]:
            raise ValueError(f"{mode} arm log segment hash changed")
        prefix_text = prefix.decode("utf-8", errors="replace")
        segment = segment_bytes.decode("utf-8", errors="replace")
        validate_current_instance(prefix_text, runtime, mode)
        if INSTANCE_RE.search(segment) or "warning=fans-below-max" in segment:
            raise ValueError(f"{mode} arm log has a restart or fan failure")
        decoded = parse_decode_summary(segment, report["usage"])
        if decoded != report["decode"]:
            raise ValueError(f"{mode} arm decode summary changed")
        if mode == "timing":
            mtp = validate_timing_segment(
                segment, report["usage"]["prompt_tokens"], decoded
            )
            if mtp != report["mtp_timing"]:
                raise ValueError("timing arm per-cycle evidence changed")
        elif MARKER in segment:
            raise ValueError("clean arm log contains timing output")


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    z2n = z * z / total
    denominator = 1.0 + z2n
    center = (p + z2n / 2.0) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - half, center + half]


def combine_reports(timing: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
    validate_arm_report(timing, "timing")
    validate_arm_report(clean, "clean")
    pairs = (
        ("endpoint", timing["endpoint"], clean["endpoint"]),
        ("prompt", timing["prompt"], clean["prompt"]),
        ("request.model", timing["request"]["model"], clean["request"]["model"]),
        ("request.sampling", timing["request"]["sampling"], clean["request"]["sampling"]),
        ("request.cache_before", timing["request"]["cache_before"], clean["request"]["cache_before"]),
        ("request.cache_after", timing["request"]["cache_after"], clean["request"]["cache_after"]),
        ("usage.prompt_tokens", timing["usage"]["prompt_tokens"], clean["usage"]["prompt_tokens"]),
        ("usage.completion_tokens", timing["usage"]["completion_tokens"], clean["usage"]["completion_tokens"]),
        ("usage.decoded_output_sha256", timing["usage"]["decoded_output_sha256"], clean["usage"]["decoded_output_sha256"]),
        ("runtime.provenance_sha256", timing["runtime"]["provenance_sha256"], clean["runtime"]["provenance_sha256"]),
        ("runtime.binary_sha256", timing["runtime"]["binary_sha256"], clean["runtime"]["binary_sha256"]),
        ("runtime.model_sha256", timing["runtime"]["model_sha256"], clean["runtime"]["model_sha256"]),
        ("runtime.driver_sha256", timing["runtime"]["driver_sha256"], clean["runtime"]["driver_sha256"]),
        ("runtime.launcher_sha256", timing["runtime"]["launcher_sha256"], clean["runtime"]["launcher_sha256"]),
        ("runtime.models_sha256", timing["runtime"]["models_sha256"], clean["runtime"]["models_sha256"]),
        ("runtime.normalized_argv", timing["runtime"]["normalized_argv"], clean["runtime"]["normalized_argv"]),
        ("runtime.model", timing["runtime"].get("model"), clean["runtime"].get("model")),
        ("runtime.model_path", timing["runtime"].get("model_path"), clean["runtime"].get("model_path")),
        ("runtime.ctx_size", timing["runtime"].get("ctx_size"), clean["runtime"].get("ctx_size")),
        ("runtime.slot_count", timing["runtime"].get("slot_count"), clean["runtime"].get("slot_count")),
    )
    for path, left, right in pairs:
        if left != right:
            raise ValueError(f"arms differ at {path}: {left!r} != {right!r}")
    for key in ("cycles", "accepted", "rejected", "committed"):
        if timing["decode"][key] != clean["decode"][key]:
            raise ValueError(f"arms differ at decode.{key}")
    if timing["runtime"]["instance"] == clean["runtime"]["instance"]:
        raise ValueError("timing and clean arms must use separate server processes")

    cycles = clean["decode"]["cycles"]
    accepted = clean["decode"]["accepted"]
    committed = clean["decode"]["committed"]
    acceptance = accepted / cycles
    tokens_per_cycle = committed / cycles
    current_cycle_ms = 1000.0 * clean["decode"]["steady_decode_seconds"] / cycles
    target_cycle_ms = 1000.0 * tokens_per_cycle / TARGET_TPS
    remaining = max(0.0, current_cycle_ms - target_cycle_ms)
    timing_tps = timing["decode"]["steady_decode_tps"]
    clean_tps = clean["decode"]["steady_decode_tps"]
    timing_delta_pct = 100.0 * (timing_tps - clean_tps) / clean_tps
    return {
        "schema": "s55-200-phase2-combined-v2",
        "scope": "phase2-cycle-budget-only",
        "full_s55_status": "NOT_EVALUATED",
        "target_200k_reached": clean_tps >= TARGET_TPS,
        "prompt_tokens": clean["usage"]["prompt_tokens"],
        "completion_tokens": GENERATED_TOKENS,
        "decoded_output_sha256": clean["usage"]["decoded_output_sha256"],
        "cycles": cycles,
        "accepted": accepted,
        "rejected": clean["decode"]["rejected"],
        "acceptance": acceptance,
        "acceptance_ci95": wilson_interval(accepted, cycles),
        "tokens_per_cycle": tokens_per_cycle,
        "clean_decode_tps": clean["decode"]["decode_tps"],
        "clean_steady_decode_tps": clean["decode"]["steady_decode_tps"],
        "timing_steady_decode_tps": timing_tps,
        "uncontrolled_timing_delta_pct": timing_delta_pct,
        "timing_overhead_status": "NOT_EVALUATED",
        "counter_overhead_status": "NOT_EVALUATED",
        "steady_window_tokens": committed,
        "current_cycle_ms": current_cycle_ms,
        "target_cycle_ms": target_cycle_ms,
        "remaining_deficit_ms": remaining,
        "required_cycle_reduction_pct": 100.0 * remaining / current_cycle_ms,
        "provenance_sha256": clean["runtime"]["provenance_sha256"],
    }


def write_or_print(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    atomic_publish_bytes(output, rendered.encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser(
        "manifest", help="verify and print the frozen prompt manifest"
    )
    manifest.add_argument(
        "--prompt-output", type=Path,
        help="also publish the exact rendered prompt for native model tests",
    )

    freeze = sub.add_parser("freeze", help="hash exact source, binary, model, and machine")
    freeze.add_argument("--repo", type=Path)
    freeze.add_argument("--server-binary", type=Path, required=True)
    freeze.add_argument("--model-file", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="issue one immutable Phase 2 diagnostic request")
    run.add_argument("--mode", choices=("timing", "clean"), required=True)
    run.add_argument("--server-log", type=Path, required=True)
    run.add_argument("--provenance", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--url", default="http://127.0.0.1:8000")
    run.add_argument("--idle-timeout", type=float, default=600.0)
    run.add_argument("--idle-settle", type=float, default=8.0)
    run.add_argument("--request-timeout", type=float, default=1800.0)

    combine = sub.add_parser("combine", help="strictly validate and combine both arms")
    combine.add_argument("timing", type=Path)
    combine.add_argument("clean", type=Path)
    combine.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "manifest":
        prompt, record = prompt_manifest(Path(__file__).resolve().parents[1])
        if args.prompt_output is not None:
            atomic_publish_bytes(args.prompt_output, prompt.encode("utf-8"))
        write_or_print(record, None)
        return 0
    if args.command == "freeze":
        write_or_print(
            freeze_provenance(args.repo, args.server_binary, args.model_file),
            args.output,
        )
        return 0
    if args.command == "run":
        if args.idle_timeout <= 0 or args.idle_settle < 0 or args.request_timeout <= 0:
            parser.error("timeouts must be positive and idle settle cannot be negative")
        write_or_print(run_arm(args), None)
        return 0
    timing = json.loads(args.timing.read_text(encoding="utf-8"))
    clean = json.loads(args.clean.read_text(encoding="utf-8"))
    validate_combine_evidence(timing, clean)
    write_or_print(combine_reports(timing, clean), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
