#!/usr/bin/env python3
"""Observe O2b startup through shutdown, then stream at the full context ceiling.

Uses the existing O2-only cache. Run after verify-payloads.py has finished.
"""
import json
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
G = Path.home() / "models/gguf"
MODEL = G / "DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-Layers33-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
URL = "http://127.0.0.1:8009"
assert json.loads((OUT / "payload-verification.json").read_text())["pass"]
assert json.loads((OUT / "fill-followup-fixture.json").read_text())["prompt_tokens"] == 261888
assert subprocess.run(["pgrep", "-x", "ds4-server"], capture_output=True).returncode == 1
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 8009))

def command(args):
    return subprocess.check_output(args, text=True, timeout=10).strip()

def get(path):
    with urllib.request.urlopen(URL + path, timeout=5) as response:
        return json.load(response)

processes = command(["ps", "-axo", "pid,ppid,state,rss,comm"])
(OUT / "fill-followup-processes.txt").write_text(processes + "\n")
rows = []
monitor_errors = []
stop = threading.Event()
srv = None
phase = "baseline"

def sample():
    vm = command(["vm_stat"])
    row = {"time": time.time(), "phase": phase,
           "swapouts": int(re.search(r"^Swapouts:\s+(\d+)", vm, re.M).group(1)),
           "pressure": int(command(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])),
           "swap_usage": command(["sysctl", "-n", "vm.swapusage"])}
    if srv is not None and srv.poll() is None:
        row["server_rss_gib"] = int(command(["ps", "-p", str(srv.pid), "-o", "rss="])) / 1048576
    rows.append(row)
    with (OUT / "fill-followup-memory.jsonl").open("a") as log:
        log.write(json.dumps(row) + "\n")

def monitor():
    while not stop.wait(1):
        try:
            sample()
        except Exception as exc:
            monitor_errors.append(str(exc))
            return

(OUT / "fill-followup-memory.jsonl").write_text("")
sample()
thread = threading.Thread(target=monitor)
thread.start()
events = []
usage = None
finish = None
done = False
start = time.monotonic()
try:
    phase = "startup"
    with (OUT / "fill-followup.server.log").open("w") as log:
        srv = subprocess.Popen([
            str(ROOT / "ds4-server"), "--chdir", str(ROOT), "--metal", "--model", str(MODEL),
            "--vision", str(G / "DeepSeek-V4-Flash-Vision-Encoder.gguf"),
            "--ctx", "262144", "--tokens", "32768", "--warm-weights", "--power", "100",
            "--host", "127.0.0.1", "--port", "8009", "--kv-disk-dir",
            str(Path.home() / ".ds4/o2-gates/kv-fill-o2"), "--kv-disk-space-mb", "8192",
            "--kv-cache-min-tokens", "2048", "--kv-cache-reject-different-quant",
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(f"SERVER_PID {srv.pid}", flush=True)
    for _ in range(120):
        if srv.poll() is not None:
            raise RuntimeError(f"server exited {srv.returncode}")
        try:
            get("/health")
            break
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    else:
        raise RuntimeError("server readiness timeout")
    phase = "prefill"
    body = {"model": "deepseek-v4-flash", "temperature": 0, "max_tokens": 256,
            "ignore_eos": True, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": (OUT / "fill-followup-prompt.txt").read_text()}]}
    request = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response, (OUT / "fill-followup-stream.jsonl").open("w") as log:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            now = time.monotonic()
            log.write(json.dumps({"seconds": now - start, "event": event}) + "\n")
            log.flush()
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    phase = "decode"
                    events.append(now)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    stats = get("/stats")
    assert done and finish == "length", (done, finish)
    assert usage and usage["prompt_tokens"] == 261888 and usage["completion_tokens"] == 256, usage
finally:
    phase = "shutdown"
    if srv is not None and srv.poll() is None:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.terminate()
            srv.wait(timeout=15)
    stop.set()
    thread.join(timeout=15)
    sample()

gaps = [b - a for a, b in zip(events, events[1:])]
assert gaps and not monitor_errors, monitor_errors
result = {"usage": usage, "finish": finish, "swapouts_delta": rows[-1]["swapouts"] - rows[0]["swapouts"],
          "pressure_levels": sorted({r["pressure"] for r in rows}), "samples": len(rows),
          "rss_peak_gib": max(r.get("server_rss_gib", 0) for r in rows),
          "stream_content_events": len(events), "median_gap_ms": statistics.median(gaps) * 1000,
          "p99_gap_ms": sorted(gaps)[int((len(gaps)-1)*.99)] * 1000, "max_gap_ms": max(gaps) * 1000,
          "decode_tps": stats["last_decode_tps"], "prefill_tps": stats["last_prefill_tps"],
          "server_exit": srv.returncode, "wall_seconds": time.monotonic() - start}
result["pass"] = result["swapouts_delta"] == 0 and result["pressure_levels"] == [1] and max(gaps) < 1 and srv.returncode == 0
(OUT / "fill-followup-result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
assert result["pass"], "Lifecycle memory/stall check failed; retain base and donor"
