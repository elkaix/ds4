#!/usr/bin/env bash
# Start the GLM 5.3 Flash server (this checkout, branch m5-prod;
# branch/commit printed at startup) and print one-line health/resource snapshots
# while it runs. Server logs remain attached to this terminal.
#
# Usage:
#   ./run-glm-ds4.sh
#   MONITOR_INTERVAL_SECONDS=30 ./run-glm-ds4.sh
# Fans follow the ThermalForge profile while ds4-server runs, then return to Apple auto.

set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
GLM_DIR="${GLM_DS4_DIR:-$ROOT_DIR}"
SERVER_BIN="$GLM_DIR/ds4-server"
MODEL="${GLM_DS4_MODEL:-$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-23Q4KExperts-KDAvoQ4K-Q2.gguf}"
HOST="127.0.0.1"
PORT=8000
# 363K (363 x 1024) tokens; the model's native limit is 1,048,576.
CTX="${GLM_DS4_CTX:-371712}"
TOKENS=32768
# Clean-engine KV cache, per quant. Checkpoints are only valid for the model
# that wrote them, so a model change needs a new directory.
KV_DIR="${GLM_DS4_KV_DIR:-$HOME/.ds4/server-kv/glm53-m5-clean-l17-23q4k-kdavo}"
KV_BUDGET_MB=131072
KV_MIN_TOKENS=2048
KV_COLD_MAX_TOKENS=65536
# Continued-frontier snapshot interval. Every frontier is a full prefix file
# (26.8-27.7 KiB/token, measured from the checkpoint headers in KV_DIR), and
# older frontiers are only evicted when the budget fills. ds4 rounds the interval
# up to a multiple of --kv-cache-boundary-align-tokens (2048); its default 10000
# becomes 10240, which writes 21 files (~61 GiB) during one cold 218k prefill.
# 20480 writes 10 (~29 GiB) at the cost of re-prefilling up to 20480 tokens on a
# cold resume. 0 disables continued frontiers. The 2026-09-19 session diagnosis
# also flagged synchronous continued writes stalling decode: keep this coarse,
# and expect ~0.5-0.9 s pauses at each frontier during decode.
KV_CONTINUED_INTERVAL="${GLM_DS4_KV_CONTINUED_INTERVAL:-20480}"
KV_ALIGN_TOKENS=2048
# Idle TTL for disk checkpoints. ds4 evicts only when the budget is full and its
# score favours large files, so finished sessions otherwise pin the whole budget
# (2026-09-22: 73 files / 107.5 GiB idle >6h, crowding out small system-prompt
# anchors). kv-cache-prune.py deletes checkpoints idle longer than this at start,
# every KV_PRUNE_INTERVAL_SECONDS while running, and after the server stops.
# The shutdown checkpoint is fresh, so restart-resume survives. 0 disables.
KV_TTL_HOURS="${GLM_DS4_KV_TTL_HOURS:-6}"
KV_PRUNE_INTERVAL_SECONDS=600
KV_PRUNE="$ROOT_DIR/kv-cache-prune.py"
# Model-embedded GLM 5.3 MTP speculation, OFF by default: the clean baseline
# (PR #1090 lineage, glm53-m5-prod) is plain decode while the long-context
# campaign re-measures where speculation pays. Set GLM_DS4_MTP=1 to enable.
MTP="${GLM_DS4_MTP:-0}"
# Metal 4 tensor/MPP matmul route, OFF by default: it FAILS the
# --metal-tensor-equivalence gate on this M5 Max with GLM 5.3 (2026-09-19:
# logits rms drift 0.41-0.48, max_abs 3.14, greedy mismatches from step 7;
# rerun: ds4_test --metal-tensor-equivalence with DS4_TEST_MODEL/GLM vectors).
# #1090 defaults it ON for M5 via compile probe only — no equivalence check.
# GLM53 fused kernels (gate lift 0e1d68e) are unaffected: --metal-kernels and
# --glm53-continued-prefill pass with the route off. Set GLM_DS4_ALLOW_TENSOR_ROUTE=1
# to run it deliberately.
if [[ "${GLM_DS4_ALLOW_TENSOR_ROUTE:-0}" == 1 ]]; then
    unset DS4_METAL_DISABLE_TENSOR_API
else
    export DS4_METAL_DISABLE_TENSOR_API=1
fi
# Per-step MTP acceptance/verify timing. Diagnostic only: the extra logging
# perturbs the decode rate it measures, so leave it off for benchmarks.
MTP_TIMING="${GLM_DS4_MTP_TIMING:-0}"
# Context past which MTP speculation turns off (DS4_GLM_MTP_MAX_CTX). The
# 2026-09-12 agent session on this machine decoded 22-32 t/s with MTP below
# ~32K, 14-17 t/s with MTP at 47-63K, and 24-26 t/s plain right after the old
# 65536 ceiling, so speculation stops paying near 32K. 0 removes the ceiling.
# Only meaningful when GLM_DS4_MTP=1.
MTP_MAX_CTX="${GLM_DS4_MTP_MAX_CTX:-32768}"
# Optional ds4-server --trace file: per-request prompt/cache diagnostics,
# including the first mismatching tokens on a live KV cache miss.
TRACE_PATH="${GLM_DS4_TRACE:-}"
WIRED_LIMIT_MIN_MB=114688
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-15}
THERMALFORGE="${THERMALFORGE:-$(command -v thermalforge || echo /opt/homebrew/bin/thermalforge)}"
# ThermalForge.app drives the fans through the same daemon when it is running.
# Defer to it: no watch profile, no reset to Apple auto when the server stops.
FAN_APP_OWNED=false
if pgrep -xq ThermalForgeApp; then FAN_APP_OWNED=true; fi
FAN_COMMAND_TIMEOUT_SECONDS=5
FAN_RESTORE_TIMEOUT_SECONDS=5
FAN_PROFILE="${FAN_PROFILE:-balanced}"   # thermalforge watch profile: silent|balanced|performance|max
FAN_WATCH_INTERVAL_SECONDS=2
fan_max_owned=false
fan_watch_pid=""

usage() {
    cat <<EOF
Usage: ./run-glm-ds4.sh

Starts this checkout's ds4-server (ds4, m5-prod: PR #1090 clean GLM Metal
engine + #1093 tool-turn checkpoint + dashboard + M5 gate lift) with GLM 5.3
Flash Uncensored stage 2 (dealignai d21b, Q2 body, KDA v/output and layers
17-23 experts Q4_K, ctx $CTX) and prints health,
throughput, process memory, KV disk-cache use, and free disk space. Fans follow
the ThermalForge "$FAN_PROFILE" profile (temperature-driven) while the server
runs and return to Apple auto when it stops.

Environment:
  MONITOR_INTERVAL_SECONDS=N  Monitoring interval in seconds (default: 15)
  GLM_DS4_MODEL=PATH          Override the GGUF (default: GLM-5.3-Flash-UNCEN-d21b-L17-23Q4KExperts-KDAvoQ4K-Q2.gguf)
                              (pair it with GLM_DS4_KV_DIR -- the KV cache is per-quant)
  GLM_DS4_KV_DIR=PATH         Override the KV disk-cache directory
  GLM_DS4_KV_CONTINUED_INTERVAL=N
                              Continued KV snapshot interval in tokens (default:
                              20480; rounded up to a multiple of 2048; 0 disables)
  GLM_DS4_KV_TTL_HOURS=N      Delete KV checkpoints idle longer than N hours at
                              start, every 10 min, and on stop (default: 6; 0 disables)
  FAN_PROFILE=name            thermalforge watch profile (default: balanced)
  GLM_DS4_MTP=1               Enable model-embedded MTP speculation (default off)
  GLM_DS4_MTP_TIMING=1        Print MTP acceptance/verify timing (diagnostic)
  GLM_DS4_CTX=N               Context tokens (default: 371712 = 363K; model max 1048576)
  GLM_DS4_MTP_MAX_CTX=N       Turn MTP off past N context tokens (default 32768; 0 = never)
  GLM_DS4_TRACE=path          Write ds4-server request/cache trace to path
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi
if [[ ! $MONITOR_INTERVAL_SECONDS =~ ^[1-9][0-9]*$ || ${#MONITOR_INTERVAL_SECONDS} -gt 4 ]] ||
    (( 10#$MONITOR_INTERVAL_SECONDS > 3600 )); then
    echo "MONITOR_INTERVAL_SECONDS must be an integer between 1 and 3600" >&2
    exit 2
fi
if [[ ! $CTX =~ ^[1-9][0-9]*$ ]] || (( CTX > 1048576 )); then
    echo "GLM_DS4_CTX must be a positive integer <= 1048576, got: $CTX" >&2
    exit 2
fi
if [[ ! $MTP_MAX_CTX =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "GLM_DS4_MTP_MAX_CTX must be a non-negative integer, got: $MTP_MAX_CTX" >&2
    exit 2
fi
export DS4_GLM_MTP_MAX_CTX="$MTP_MAX_CTX"

if [[ ! $KV_CONTINUED_INTERVAL =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "GLM_DS4_KV_CONTINUED_INTERVAL must be a non-negative integer; got: $KV_CONTINUED_INTERVAL" >&2
    exit 2
fi
if [[ ! $KV_TTL_HOURS =~ ^(0|[1-9][0-9]{0,3})$ ]]; then
    echo "GLM_DS4_KV_TTL_HOURS must be an integer between 0 and 9999; got: $KV_TTL_HOURS" >&2
    exit 2
fi
if (( KV_TTL_HOURS > 0 )) && [[ ! -r $KV_PRUNE ]]; then
    echo "KV prune helper not found: $KV_PRUNE (set GLM_DS4_KV_TTL_HOURS=0 to run without it)" >&2
    exit 1
fi

for command in lsof macmon python3 ps sudo sysctl; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done
MACMON=$(command -v macmon)
if [[ ! -x $SERVER_BIN ]]; then
    echo "Server binary not found or not executable: $SERVER_BIN" >&2
    echo "Build it first with: make -C \"$GLM_DIR\" ds4-server" >&2
    exit 1
fi
if [[ ! -r $MODEL ]]; then
    echo "Model not found or unreadable: $MODEL" >&2
    exit 1
fi
if [[ ! -x $THERMALFORGE ]]; then
    echo "Fan controller not found or not executable: $THERMALFORGE" >&2
    echo "Install it with: brew install producerguy/tap/thermalforge" >&2
    exit 1
fi
# The root daemon (com.thermalforge.daemon) executes its own copy under
# /usr/local/bin, so a `brew upgrade` alone leaves the daemon on the old
# binary. Warn rather than fail: fan control still works, just at the older
# version. Deliberately not symlinked -- /opt/homebrew/bin is user-writable
# and the daemon runs as root.
DAEMON_THERMALFORGE=/usr/local/bin/thermalforge
if [[ -x $DAEMON_THERMALFORGE && $DAEMON_THERMALFORGE != "$THERMALFORGE" ]] &&
    ! cmp -s "$THERMALFORGE" "$DAEMON_THERMALFORGE"; then
    echo "Warning: fan controller drift -- the ThermalForge daemon runs a different build." >&2
    echo "  CLI:    $THERMALFORGE ($("$THERMALFORGE" --version 2>/dev/null || echo unknown))" >&2
    echo "  daemon: $DAEMON_THERMALFORGE ($("$DAEMON_THERMALFORGE" --version 2>/dev/null || echo unknown))" >&2
    echo "  Realign: sudo thermalforge install" >&2
fi

fan_control() {
    local action=$1

    python3 - "$THERMALFORGE" "$action" "$FAN_COMMAND_TIMEOUT_SECONDS" <<'PY'
import json
import subprocess
import sys

path, action = sys.argv[1:3]
timeout = int(sys.argv[3])
valid_actions = {"probe", "auto", "status", "verify-auto"}
if action not in valid_actions:
    raise SystemExit(f"unsupported fan action: {action}")

command_action = "auto" if action == "auto" else "status"
# ThermalForge's privileged daemon (com.thermalforge.daemon, installed by
# `sudo thermalforge install`) owns the SMC writes, so the CLI needs no sudo.
command = [path, command_action]

try:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
except subprocess.TimeoutExpired:
    print(f"fan command timed out after {timeout}s: {command_action}", file=sys.stderr)
    raise SystemExit(1)

if result.returncode != 0:
    detail = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
    print(f"fan command failed ({command_action}): {detail}", file=sys.stderr)
    raise SystemExit(1)
if action == "auto":
    raise SystemExit(0)

try:
    payload = json.loads(result.stdout)
    fans = payload["fans"]
    if not isinstance(fans, list) or not fans or not all(isinstance(fan, dict) for fan in fans):
        raise ValueError("no fan readings")
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    print(f"invalid fan status: {exc}", file=sys.stderr)
    raise SystemExit(1)

if action == "status":
    print(json.dumps(payload, separators=(",", ":")))
    raise SystemExit(0)

modes = {str(fan.get("mode", "")).lower() for fan in fans}
if action == "probe":
    if not modes <= {"auto", "automatic", "default"}:
        print(f"fans must begin in Apple auto mode; found: {sorted(modes)}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)
# Only verify-auto reaches here; probe and status exit above.
raise SystemExit(0 if modes <= {"auto", "automatic", "default"} else 1)
PY
}

set_fans_max() {
    # Name kept for the call site; runs `thermalforge watch --profile $FAN_PROFILE`,
    # which drives the fans from CPU/GPU temperature instead of pinning them at max.
    # The controller may apply a write before we can check, so cleanup owns
    # restoration from the moment the watcher starts.
    fan_max_owned=true
    # A watcher from a launcher that is still tearing down can make the new one
    # exit at once (observed on a back-to-back restart), so try twice.
    local attempt
    for attempt in 1 2; do
        "$THERMALFORGE" watch --profile "$FAN_PROFILE" --interval "$FAN_WATCH_INTERVAL_SECONDS" \
            >/dev/null 2>&1 &
        fan_watch_pid=$!
        sleep 2
        if kill -0 "$fan_watch_pid" 2>/dev/null; then
            echo "Fans under ThermalForge '$FAN_PROFILE' profile (watch pid $fan_watch_pid)."
            return 0
        fi
        fan_watch_pid=""
        echo "thermalforge watch --profile $FAN_PROFILE exited immediately (attempt $attempt)." >&2
        sleep 3
    done
    return 1
}

stop_fan_watch() {
    local deadline
    [[ -n $fan_watch_pid ]] || return 0
    if kill -0 "$fan_watch_pid" 2>/dev/null; then
        kill -INT "$fan_watch_pid" 2>/dev/null || true
        deadline=$((SECONDS + FAN_RESTORE_TIMEOUT_SECONDS))
        while ((SECONDS < deadline)) && kill -0 "$fan_watch_pid" 2>/dev/null; do sleep 1; done
        kill -0 "$fan_watch_pid" 2>/dev/null && kill -KILL "$fan_watch_pid" 2>/dev/null || true
    fi
    wait "$fan_watch_pid" 2>/dev/null || true
    fan_watch_pid=""
}

restore_fans() {
    local deadline attempt

    [[ $fan_max_owned == true ]] || return 0
    stop_fan_watch
    echo "Restoring fans to Apple automatic control..." >&2
    for ((attempt = 1; attempt <= 3; attempt++)); do
        if fan_control auto; then
            deadline=$((SECONDS + FAN_RESTORE_TIMEOUT_SECONDS))
            while ((SECONDS < deadline)); do
                if fan_control verify-auto 2>/dev/null; then
                    fan_max_owned=false
                    echo "Fans restored to Apple automatic control." >&2
                    return 0
                fi
                sleep 1
            done
        fi
        ((attempt < 3)) && sleep 2
    done
    echo "WARNING: automatic fan control was requested but not verified." >&2
    fan_control status >&2 || true
    return 1
}

if [[ $FAN_APP_OWNED == true ]]; then fan_probe=status; else fan_probe=probe; fi
if ! fan_control "$fan_probe" >/dev/null; then
    echo "Non-interactive verified fan control is unavailable; refusing to start ds4-server." >&2
    echo "Repair it with: sudo thermalforge install" >&2
    echo "If fans are in manual mode, open ThermalForge.app (this script then defers to it) or run: thermalforge auto" >&2
    exit 1
fi

if ! wired_limit_mb=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null); then
    echo "Could not read iogpu.wired_limit_mb; refusing to start the model." >&2
    exit 1
fi
if [[ ! $wired_limit_mb =~ ^[0-9]+$ ]]; then
    echo "Unexpected iogpu.wired_limit_mb value: $wired_limit_mb" >&2
    exit 1
fi
if (( wired_limit_mb < WIRED_LIMIT_MIN_MB )); then
    cat >&2 <<EOF
Metal wired-memory limit is ${wired_limit_mb} MiB; this model needs at least
${WIRED_LIMIT_MIN_MB} MiB. Run this after reboot, then retry:

  sudo sysctl iogpu.wired_limit_mb=118000
EOF
    exit 1
fi

mkdir -p "$KV_DIR"

# Best effort: a prune failure is reported but never blocks start or shutdown.
prune_kv_cache() {
    (( KV_TTL_HOURS > 0 )) || return 0
    python3 "$KV_PRUNE" --dir "$KV_DIR" --ttl-hours "$KV_TTL_HOURS" --label "$1" >&2 ||
        echo "Warning: KV cache prune ($1) reported errors." >&2
}

# Launcher instance lock: closes the lsof→bind→start TOCTOU window between
# concurrent launcher starts, and stops two wrappers from racing the same
# port/KV dir. mkdir is atomic on APFS; a stale lock (crashed launcher) is
# detected by PID liveness and reclaimed.
LOCK_DIR="$HOME/.ds4/locks/ds4-$PORT"
mkdir -p "$HOME/.ds4/locks"
acquire_launcher_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        printf '%s\n' "$$" > "$LOCK_DIR/pid"
        return 0
    fi
    local lock_pid
    lock_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
        # Stale lock from a dead launcher: reclaim it once.
        rm -rf "$LOCK_DIR"
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            printf '%s\n' "$$" > "$LOCK_DIR/pid"
            return 0
        fi
    fi
    echo "Another launcher instance holds $LOCK_DIR (pid ${lock_pid:-unknown})." >&2
    echo "Stop it first, or remove the lock directory if it is stale." >&2
    return 1
}
acquire_launcher_lock
release_launcher_lock() {
    [[ -f "$LOCK_DIR/pid" ]] || return 0
    local lock_pid
    lock_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    if [[ "$lock_pid" == "$$" ]]; then
        rm -rf "$LOCK_DIR"
    fi
}

set +e
listener=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>&1)
listener_status=$?
set -e
if (( listener_status == 0 )); then
    echo "Port $PORT is already covered by a listening socket:" >&2
    echo "$listener" >&2
    echo "Stop the existing service first, then run this script again." >&2
    exit 1
fi
if (( listener_status != 1 )) || [[ -n $listener ]]; then
    echo "Could not safely inspect TCP port $PORT; refusing to start the model." >&2
    [[ -n $listener ]] && echo "$listener" >&2
    exit 1
fi

if ! port_error=$(python3 - "$HOST" "$PORT" 2>&1 <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((sys.argv[1], int(sys.argv[2])))
PY
); then
    echo "Cannot reserve $HOST:$PORT; no model process was started." >&2
    [[ -n $port_error ]] && echo "$port_error" >&2
    exit 1
fi

# Also covers checkpoints left behind by a launcher that was hard-killed.
prune_kv_cache start

server_pid=""
server_started=false
monitor_pid=""

stop_monitor() {
    local pid=${monitor_pid:-}
    local i

    [[ -n $pid ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        for ((i = 0; i < 3; i++)); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    wait "$pid" 2>/dev/null || true
    monitor_pid=""
}

# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2329
cleanup() {
    local status=$?
    local i

    trap - EXIT
    trap '' HUP INT TERM

    release_launcher_lock

    # Restore fans first: stopping the server can take tens of seconds, and if
    # this shell is killed mid-shutdown the fans must already be back to auto.
    restore_fans || true

    # The server runs in its own session, so this is its first and only SIGINT.
    # Stop it before waiting on monitoring so a stuck metric cannot delay shutdown.
    if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
        echo "Stopping ds4-server (PID $server_pid)..." >&2
        kill -INT "$server_pid" 2>/dev/null || true
        for ((i = 0; i < 30; i++)); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "Server did not stop after 30s; sending SIGTERM." >&2
            kill -TERM "$server_pid" 2>/dev/null || true
            for ((i = 0; i < 5; i++)); do
                kill -0 "$server_pid" 2>/dev/null || break
                sleep 1
            done
        fi
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "Server did not stop after SIGTERM; sending SIGKILL." >&2
            kill -KILL "$server_pid" 2>/dev/null || true
        fi
    fi
    if [[ -n ${server_pid:-} ]]; then
        wait "$server_pid" 2>/dev/null || true
    fi

    restore_fans || true
    stop_monitor
    # The server has exited, so its shutdown checkpoint is on disk and fresh.
    if [[ $server_started == true ]]; then
        prune_kv_cache stop
    fi
    exit "$status"
}

trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

start_monitor() {
    local watched_pid=$1

    python3 -u -c '
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

os.setsid()

pid = int(sys.argv[1])
host = sys.argv[2]
port = int(sys.argv[3])
kv_dir = sys.argv[4]
budget_bytes = int(sys.argv[5]) * 1024 * 1024
interval = int(sys.argv[6])
fan_controller = sys.argv[7]
fan_command_timeout = int(sys.argv[8])
macmon = sys.argv[9]
kv_prune = sys.argv[10]
kv_ttl_hours = int(sys.argv[11])
kv_prune_interval = int(sys.argv[12])
base_url = f"http://{host}:{port}"
ever_healthy = False
log_path = os.path.expanduser("~/.ds4/monitor.log")


def emit(message):
    """Print to the terminal and append to the log, surviving a dead pty."""
    try:
        print(message, flush=True)
    except OSError:
        pass
    try:
        with open(log_path, "a") as log_file:
            log_file.write(message + "\n")
    except OSError:
        pass


def server_alive():
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def fetch_json(path):
    try:
        with urllib.request.urlopen(base_url + path, timeout=3) as response:
            if response.status != 200:
                return None
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            return None
        value = json.loads(payload)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def process_stats():
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu=,%mem=,rss="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        fields = result.stdout.split()
        if len(fields) == 3:
            return fields[0], fields[1], int(fields[2])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "-", "-", 0


def macmon_snapshot():
    """One macmon sample -> (fans string, gpu telemetry string).

    GPU freq/power/temp feed the slow-window discriminator (route vs
    thermal): slow decode + stable clock -> MoE/kernel, slow decode +
    lower clock -> thermal/power.
    """
    try:
        result = subprocess.run(
            [macmon, "pipe", "--samples", "1", "--interval", "250"],
            check=False,
            capture_output=True,
            text=True,
            timeout=fan_command_timeout,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if result.returncode == 0 and lines else {}
        if not isinstance(payload, dict):
            return "-", "-"
        fans = payload.get("fans")
        fan_str = "-"
        if isinstance(fans, list) and fans:
            values = []
            for fan in fans:
                actual = int(fan["rpm"])
                maximum = int(fan["max_rpm"])
                if maximum <= 0:
                    values = []
                    break
                values.append("{}={}/{}".format(fan.get("name", "fan?"), actual, maximum))
            if values:
                fan_str = ",".join(values) + "RPM"
        temp = payload.get("temp", {})
        gpu_temp = temp.get("gpu_temp_avg") if isinstance(temp, dict) else None
        gpu_parts = []
        for key, fmt in (("gpu_freq_mhz", "{:.0f}MHz"), ("gpu_power", "{:.1f}W")):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                gpu_parts.append(fmt.format(value))
        if isinstance(gpu_temp, (int, float)):
            gpu_parts.append("{:.1f}C".format(gpu_temp))
        return fan_str, "/".join(gpu_parts) if gpu_parts else "-"
    except (IndexError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return "-", "-"


def cache_size():
    total = 0
    try:
        with os.scandir(kv_dir) as entries:
            for entry in entries:
                try:
                    if entry.name.endswith(".kv") and entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def disk_free():
    try:
        return shutil.disk_usage(kv_dir).free
    except OSError:
        return 0


def rate(value):
    try:
        return f"{float(value):.1f}t/s"
    except (TypeError, ValueError):
        return "-"


def restore_fans_after_server():
    """Hard-kill fallback: this monitor survives independently of the wrapper."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emit(f"[monitor {timestamp}] fans=restore-attempt reason=server-stopped")
    # Stop any thermalforge watch profile the wrapper left behind before resetting.
    subprocess.run(["pkill", "-INT", "-f", "thermalforge watch"], check=False, capture_output=True)
    try:
        reset = subprocess.run(
            [fan_controller, "auto"],
            check=False,
            capture_output=True,
            text=True,
            timeout=fan_command_timeout,
        )
        if reset.returncode != 0:
            detail = (reset.stderr or reset.stdout).strip() or f"exit {reset.returncode}"
            emit(f"[monitor {timestamp}] fans=restore-failed detail={detail}")
            return
        status = subprocess.run(
            [fan_controller, "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=fan_command_timeout,
        )
        payload = json.loads(status.stdout) if status.returncode == 0 else {}
        fans = payload.get("fans") if isinstance(payload, dict) else None
        restored = isinstance(fans, list) and bool(fans) and all(
            isinstance(fan, dict)
            and str(fan.get("mode", "")).lower() in {"auto", "automatic", "default"}
            for fan in fans
        )
        state = "auto" if restored else "restore-unverified"
        emit(f"[monitor {timestamp}] fans={state} reason=server-stopped")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        emit(
            f"[monitor {timestamp}] fans=restore-failed detail={type(exc).__name__}"
        )


def prune_kv_cache(label):
    """Idle-TTL prune; logs only when it removed something or failed."""
    try:
        result = subprocess.run(
            [sys.executable, kv_prune, "--dir", kv_dir,
             "--ttl-hours", str(kv_ttl_hours), "--label", label],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        emit(f"[monitor] kv-prune failed: {type(exc).__name__}")
        return
    summary = result.stdout.strip()
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        emit(f"[monitor] {summary} errors={len(detail)}")
    elif summary and " removed=0 " not in summary:
        emit(f"[monitor] {summary}")


next_prune = time.monotonic() + kv_prune_interval

while server_alive():
    # /v1/models is the liveness probe; /stats carries the serving counters.
    health = fetch_json("/v1/models")
    stats = fetch_json("/stats")

    if health is not None and health.get("object") == "list":
        health_status = "ok"
        ever_healthy = True
    elif health is not None:
        health_status = "unexpected"
    else:
        health_status = "unreachable" if ever_healthy else "starting"

    # A transient /stats failure must not look like the parser regressed, so the
    # line always carries an explicit stats= state next to the dashes.
    if stats is None:
        stats_status = "unreachable"
        summary = "state=- queue=- clients=- live=- requests=- hits=- cold=- cached=- prefill=- decode=-"
    elif "queue_depth" not in stats or "slot_count" not in stats:
        stats_status = "unexpected"
        summary = "state=- queue=- clients=- live=- requests=- hits=- cold=- cached=- prefill=- decode=-"
    else:
        stats_status = "ok"
        busy = stats.get("busy")
        state = "busy" if busy is True else "idle" if busy is False else "-"
        cache = stats.get("cache") if isinstance(stats.get("cache"), dict) else {}
        values = (
            state,
            stats.get("queue_depth", "-"),
            stats.get("clients", "-"),
            stats.get("live_tokens", "-"),
            stats.get("requests", "-"),
            cache.get("hits", "-"),
            cache.get("cold", "-"),
            stats.get("cached_tokens", "-"),
            rate(stats.get("last_prefill_tps")),
            rate(stats.get("last_decode_tps")),
        )
        summary = (
            "state={} queue={} clients={} live={} requests={} hits={} cold={} "
            "cached={} prefill={} decode={}".format(*values)
        )

    cpu_percent, memory_percent, rss_kib = process_stats()
    footprint_gib = float(stats.get("footprint_mb", 0.0)) / 1024 if isinstance(stats, dict) else 0.0
    swap_gib = float(stats.get("swap_used_mb", 0.0)) / 1024 if isinstance(stats, dict) else 0.0
    fan_status, gpu_status = macmon_snapshot()
    used_bytes = cache_size()
    used_gib = used_bytes / (1024 ** 3)
    budget_gib = budget_bytes / (1024 ** 3)
    cache_percent = used_bytes * 100 / budget_bytes
    free_gib = disk_free() / (1024 ** 3)
    warnings = []
    if cache_percent >= 90:
        warnings.append("kv-cache-near-full")
    warning = "".join(f" warning={value}" for value in warnings)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    emit(
        f"[monitor {timestamp}] health={health_status} stats={stats_status} {summary} "
        f"cpu={cpu_percent}% footprint={footprint_gib:.1f}GiB swap={swap_gib:.2f}GiB rss={rss_kib / 1048576:.1f}GiB "
        f"fans={fan_status} gpu={gpu_status} "
        f"kv={used_gib:.1f}/{budget_gib:.1f}GiB({cache_percent:.1f}%) "
        f"disk_free={free_gib:.1f}GiB{warning}"
    )

    if kv_ttl_hours > 0 and time.monotonic() >= next_prune:
        prune_kv_cache("running")
        next_prune = time.monotonic() + kv_prune_interval

    deadline = time.monotonic() + interval
    while server_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1, remaining))
if fan_controller:
    restore_fans_after_server()
# Hard-kill fallback for the launcher-side stop prune.
if kv_ttl_hours > 0:
    prune_kv_cache("server-stopped")
# Neutralize std streams so the interpreter shutdown flush cannot fail
# against a dead terminal (CPython would otherwise exit with status 120).
try:
    sys.stdout = sys.stderr = open(os.devnull, "w")
except OSError:
    pass
' "$watched_pid" "$HOST" "$PORT" "$KV_DIR" "$KV_BUDGET_MB" \
        "$MONITOR_INTERVAL_SECONDS" "$([[ $FAN_APP_OWNED == true ]] || echo "$THERMALFORGE")" \
        "$FAN_COMMAND_TIMEOUT_SECONDS" "$MACMON" \
        "$KV_PRUNE" "$KV_TTL_HOURS" "$KV_PRUNE_INTERVAL_SECONDS" &
    monitor_pid=$!
}

# Report the interval ds4 actually uses after its alignment rounding.
if (( KV_CONTINUED_INTERVAL == 0 )); then
    kv_continued_state="disabled"
else
    kv_continued_state="every $(( (KV_CONTINUED_INTERVAL + KV_ALIGN_TOKENS - 1) / KV_ALIGN_TOKENS * KV_ALIGN_TOKENS )) tokens"
fi

TRACE_ARGS=()
if [[ -n $TRACE_PATH ]]; then
    TRACE_ARGS+=(--trace "$TRACE_PATH")
fi

MTP_ARGS=()
if [[ $MTP != 0 ]]; then
    MTP_ARGS+=(--mtp)
    if [[ $MTP_TIMING != 0 ]]; then
        MTP_ARGS+=(--mtp-timing)
    fi
fi

ds4_branch=$(git -C "$GLM_DIR" branch --show-current 2>/dev/null || true)
ds4_commit=$(git -C "$GLM_DIR" rev-parse --short=12 HEAD 2>/dev/null || true)
BUILD_INPUTS=('*.c' '*.h' '*.m' '*.metal' 'Makefile')
if [[ -n $(git -C "$GLM_DIR" status --porcelain 2>/dev/null) ]]; then
    ds4_tree="dirty"
else
    ds4_tree="clean"
fi
if [[ -n $(git -C "$GLM_DIR" status --porcelain -- "${BUILD_INPUTS[@]}" 2>/dev/null) ]]; then
    ds4_code="dirty"
else
    ds4_code="clean"
fi
model_stamp=$(stat -f '%z bytes, mtime %Sm' -t '%Y-%m-%dT%H:%M:%S' "$MODEL" 2>/dev/null || echo unknown)
# A binary older than the newest source commit means the tree was edited but not
# rebuilt: the number you measure then belongs to code that is not running.
binary_stamp=$(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$SERVER_BIN" 2>/dev/null || echo unknown)
binary_epoch=$(stat -f '%m' "$SERVER_BIN" 2>/dev/null || echo 0)
# When the build inputs are clean their content is HEAD's, so the commit date is
# the honest comparison; checkouts and merges rewrite mtimes without changing a
# byte and would otherwise report a false STALE. Only once something is actually
# edited does the mtime of the edit become the thing to compare against.
if [[ $ds4_code == "clean" ]]; then
    source_epoch=$(git -C "$GLM_DIR" log -1 --format=%ct -- "${BUILD_INPUTS[@]}" 2>/dev/null || echo 0)
    stale_reason="older than the newest source commit"
else
    source_epoch=$( (cd "$GLM_DIR" && git ls-files -z -- "${BUILD_INPUTS[@]}" \
        | xargs -0 stat -f '%m' 2>/dev/null | sort -rn | head -1) || echo 0)
    stale_reason="older than an uncommitted source edit"
fi
source_epoch=${source_epoch:-0}
if (( binary_epoch > 0 && source_epoch > 0 && binary_epoch < source_epoch )); then
    binary_state="STALE: $stale_reason; rebuild before benchmarking"
elif [[ $ds4_code != "clean" ]]; then
    binary_state="current, but uncommitted source edits are present"
else
    binary_state="current"
fi

cat <<EOF
Starting monitored ds4-server (GLM 5.3 Flash Q2, clean engine)
  model:      $MODEL
  endpoint:   http://$HOST:$PORT
  context:    $CTX
  max tokens: $TOKENS
  KV cache:   $KV_DIR (${KV_BUDGET_MB} MiB budget, min ${KV_MIN_TOKENS}, cold max ${KV_COLD_MAX_TOKENS} tokens, continued ${kv_continued_state}, idle TTL $(if (( KV_TTL_HOURS > 0 )); then echo "${KV_TTL_HOURS}h"; else echo off; fi))
  build:      ${ds4_branch:-unknown} @ ${ds4_commit:-unknown}
  repo:       $ds4_tree (code: $ds4_code)
  binary:     $binary_state, built $binary_stamp
  model file: $model_stamp
  MTP:        $(if [[ $MTP != 0 ]]; then echo "enabled (--mtp, width 2, off past $(if [[ $MTP_MAX_CTX == 0 ]]; then echo "no ceiling"; else echo "$MTP_MAX_CTX tokens"; fi))"; else echo "disabled (clean baseline; GLM_DS4_MTP=1 to enable)"; fi)$(if [[ $MTP != 0 && $MTP_TIMING != 0 ]]; then echo " + timing (--mtp-timing)"; fi)
  trace:      ${TRACE_PATH:-off}
  monitor:    every ${MONITOR_INTERVAL_SECONDS}s
  fans:       ThermalForge '$FAN_PROFILE' profile (temperature-driven); Apple auto on stop
  wired limit: ${wired_limit_mb:-unknown} MiB

Press Ctrl-C once to stop the server, restore automatic fans, and stop the monitor.
EOF

python3 -c '
import os
import sys

os.setsid()
os.chdir(sys.argv[1])
os.execv(sys.argv[2], sys.argv[2:])
' "$GLM_DIR" "$SERVER_BIN" --metal \
    --model "$MODEL" \
    ${MTP_ARGS[@]+"${MTP_ARGS[@]}"} \
    ${TRACE_ARGS[@]+"${TRACE_ARGS[@]}"} \
    --ctx "$CTX" --tokens "$TOKENS" \
    --power 100 \
    --host "$HOST" --port "$PORT" \
    --kv-disk-dir "$KV_DIR" --kv-disk-space-mb "$KV_BUDGET_MB" \
    --kv-cache-reject-different-quant \
    --kv-cache-min-tokens "$KV_MIN_TOKENS" \
    --kv-cache-cold-max-tokens "$KV_COLD_MAX_TOKENS" \
    --kv-cache-continued-interval-tokens "$KV_CONTINUED_INTERVAL" &
server_pid=$!
server_started=true

start_monitor "$server_pid"
if [[ $FAN_APP_OWNED == true ]]; then
    echo "Fans managed by ThermalForge.app; leaving them as they are."
elif ! set_fans_max; then
    echo "ThermalForge fan profile could not be started; stopping ds4-server." >&2
    exit 1
fi

set +e
wait "$server_pid"
server_status=$?
set -e
server_pid=""
restore_fans || true
stop_monitor

if (( server_status != 0 )); then
    echo "ds4-server exited with status $server_status" >&2
else
    echo "ds4-server stopped." >&2
fi
exit "$server_status"
