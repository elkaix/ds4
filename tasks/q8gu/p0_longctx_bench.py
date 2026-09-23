#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tokenizers>=0.22",
# ]
# ///
"""8K/32K/64K x 3-rep baseline against a live ds4-server. Does not start/stop it."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

SIZES = (8192, 32768, 65536)
REPS = 3
GEN_TOKENS = 64
BASE = "http://127.0.0.1:8000"
PID = 62368
HF_TOK = Path.home() / "models/hf/Qwen3.8-Flash-Next-Uncensored/tokenizer.json"
CORPUS = Path(__file__).resolve().parents[2] / "speed-bench/promessi_sposi.txt"
OUT = Path(__file__).resolve().parent


def swap() -> str:
    return subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()


def rss_bytes(pid: int) -> int:
    out = subprocess.check_output(["ps", "-p", str(pid), "-o", "rss="], text=True).strip()
    return int(out) * 1024


def get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post_completion(prompt: str) -> dict:
    body = json.dumps({
        "model": "qwen3.8-flash-next",
        "prompt": prompt,
        "max_tokens": GEN_TOKENS,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode())


def make_prompt(tok: Tokenizer, n: int, salt: str) -> tuple[str, int]:
    corpus = CORPUS.read_text(encoding="utf-8", errors="replace")
    pivot = (n * 104729 + hash(salt)) % max(1, len(corpus))
    rotated = corpus[pivot:] + corpus[:pivot]
    prefix = f"{salt}\n"
    prefix_ids = tok.encode(prefix, add_special_tokens=False).ids
    need = n - len(prefix_ids)
    if need < 1:
        raise SystemExit(f"salt too long for n={n}")
    ids = tok.encode(rotated, add_special_tokens=False).ids
    if len(ids) < need:
        ids = (ids * ((need // len(ids)) + 2))[:need]
    else:
        ids = ids[:need]
    prompt = prefix + tok.decode(ids, skip_special_tokens=False)
    got = len(tok.encode(prompt, add_special_tokens=False).ids)
    # trim/pad by tokens if round-trip drifted
    if got > n:
        all_ids = tok.encode(prompt, add_special_tokens=False).ids[:n]
        prompt = tok.decode(all_ids, skip_special_tokens=False)
        got = len(tok.encode(prompt, add_special_tokens=False).ids)
    return prompt, got


def rate(tokens: int, ns: int) -> float | None:
    if ns and tokens > 0:
        return tokens / (ns / 1e9)
    return None


def summarize(runs: list[dict]) -> dict:
    def med(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    return {
        "n": len(runs),
        "prefill_tps_median": med("prefill_tps"),
        "decode_tps_median": med("decode_tps"),
        "mtp_effective_tps_median": med("mtp_effective_tps"),
        "mtp_acceptance_pct_median": med("mtp_acceptance_pct"),
        "ttft_s_median": med("ttft_s"),
        "peak_rss_bytes_max": max(r["peak_rss_bytes"] for r in runs),
        "prompt_tokens": [r["prompt_tokens"] for r in runs],
        "cached_tokens": [r["cached_tokens"] for r in runs],
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--out", default=str(OUT / "p0-longctx.json"))
    ap.add_argument("--pid", type=int, default=PID)
    args = ap.parse_args()
    sizes = tuple(int(x) for x in args.sizes.split(",") if x.strip())
    out_path = Path(args.out)
    pid = args.pid
    globals()["PID"] = pid

    if not HF_TOK.is_file():
        raise SystemExit(f"missing tokenizer {HF_TOK}")
    if not CORPUS.is_file():
        raise SystemExit(f"missing corpus {CORPUS}")
    health = get_json("/health")
    if health.get("status") != "ok":
        raise SystemExit(f"server not ok: {health}")
    tok = Tokenizer.from_file(str(HF_TOK))
    freeze = json.loads((OUT / "freeze.json").read_text())
    suite = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ds4_sha": freeze["ds4"]["commit"],
        "model_sha256": freeze["main_gguf"]["sha256"],
        "ple_sha256": freeze["ple_gguf"]["sha256"],
        "model_path": freeze["main_gguf"]["path"],
        "server_pid": pid,
        "swap_before": swap(),
        "health": health,
        "sizes": {},
    }
    partial = out_path.with_name(out_path.stem + "-partial.json")
    partial.write_text(json.dumps(suite, indent=2) + "\n")

    for n in sizes:
        size_runs = []
        swap_size_before = swap()
        for rep in range(REPS):
            salt = f"P0-{n}-{rep}"
            prompt, planned = make_prompt(tok, n, salt)
            peak = {"v": rss_bytes(PID)}
            stop = threading.Event()

            def sample():
                while not stop.wait(0.25):
                    try:
                        peak["v"] = max(peak["v"], rss_bytes(PID))
                    except (subprocess.CalledProcessError, ValueError):
                        return

            t = threading.Thread(target=sample, daemon=True)
            t.start()
            t0 = time.perf_counter()
            try:
                resp = post_completion(prompt)
            except urllib.error.URLError as exc:
                stop.set()
                raise SystemExit(f"{salt} request failed: {exc}") from exc
            wall = time.perf_counter() - t0
            stop.set()
            t.join(timeout=1)
            stats = get_json("/stats")
            rec = (stats.get("recent") or [None])[0]
            if not rec:
                raise SystemExit(f"{salt}: /stats.recent empty")
            usage = resp.get("usage") or {}
            prefill_tps = rate(rec["fresh_tokens"], rec["prefill_ns"])
            decode_tps = rate(rec["decode_tokens"], rec["decode_ns"])
            mtp_cyc = rec.get("mtp_cycles") or 0
            mtp_com = rec.get("mtp_committed") or 0
            mtp_acc = (100.0 * mtp_com / mtp_cyc) if mtp_cyc else None
            mem = stats.get("mem") or {}
            row = {
                "salt": salt,
                "planned_prompt_tokens": planned,
                "prompt_tokens": rec["prompt_tokens"],
                "cached_tokens": rec["cached_tokens"],
                "fresh_tokens": rec["fresh_tokens"],
                "decode_tokens": rec["decode_tokens"],
                "usage": usage,
                "wall_s": round(wall, 4),
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
                "mtp_effective_tps": decode_tps if mtp_cyc else None,
                "mtp_cycles": mtp_cyc,
                "mtp_committed": mtp_com,
                "mtp_acceptance_pct": mtp_acc,
                "ttft_s": rec["first_token_ns"] / 1e9 if rec.get("first_token_ns") else None,
                "prefill_ns": rec["prefill_ns"],
                "decode_ns": rec["decode_ns"],
                "first_token_ns": rec["first_token_ns"],
                "finish": rec.get("finish"),
                "source": rec.get("source"),
                "peak_rss_bytes": peak["v"],
                "metal_allocated_mb": mem.get("metal_allocated_mb"),
                "footprint_mb": mem.get("footprint_mb"),
                "swap_used_mb": mem.get("swap_used_mb"),
                "pressure": mem.get("pressure"),
                "thermal": mem.get("thermal"),
            }
            size_runs.append(row)
            def fmt(v, spec):
                return format(v, spec) if v is not None else "na"

            print(
                f"{salt} prompt={row['prompt_tokens']} cached={row['cached_tokens']} "
                f"prefill={fmt(prefill_tps, '.2f')} decode={fmt(decode_tps, '.2f')} "
                f"mtp_acc={fmt(mtp_acc, '.1f')} ttft={fmt(row['ttft_s'], '.4f')}s "
                f"thermal={row['thermal']}",
                flush=True,
            )
            suite["sizes"][str(n)] = {"runs": size_runs, "swap_before": swap_size_before}
            partial.write_text(json.dumps(suite, indent=2) + "\n")
        suite["sizes"][str(n)] = {
            "runs": size_runs,
            "median": summarize(size_runs),
            "swap_before": swap_size_before,
            "swap_after": swap(),
        }

    suite["swap_after"] = swap()
    suite["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(suite, indent=2) + "\n")
    print("WROTE", out_path, flush=True)
    for n in sizes:
        m = suite["sizes"][str(n)]["median"]
        print(
            f"median {n}: prefill={m['prefill_tps_median']} decode={m['decode_tps_median']} "
            f"mtp={m['mtp_effective_tps_median']} acc={m['mtp_acceptance_pct_median']} "
            f"ttft={m['ttft_s_median']}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED", exc, file=sys.stderr)
        sys.exit(1)
