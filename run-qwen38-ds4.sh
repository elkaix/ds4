#!/usr/bin/env bash
# Start the Qwen3.8-Flash-Next Uncensored server on antirez main (this worktree;
# branch/commit printed at startup) and print one-line health/resource snapshots
# while it runs. Server logs remain attached to this terminal.
#
# Usage:
#   ./run-qwen38-ds4.sh                 # full 262k ctx, agentic defaults
#   MONITOR_INTERVAL_SECONDS=30 ./run-qwen38-ds4.sh
#   QWEN_DS4_CTX=8192 ./run-qwen38-ds4.sh          # safer first smoke
#   QWEN_DS4_BATCHED_SESSION=4 ./run-qwen38-ds4.sh # more concurrent agent sessions
#   QWEN_DS4_MODEL=~/models/gguf/Qwen3.8-Flash-Next-Q2.gguf ./run-qwen38-ds4.sh
# Fans follow the ThermalForge profile while ds4-server runs, then return to Apple auto.
#
# This launcher targets CURRENT main: one self-contained qwen4exp GGUF with
# original BF16 n-grams (per_layer_token_embd.weight). No --ple sidecar.
# The ~95 GiB BF16 table stays on disk; the runtime preads selected rows.
#
# Agentic profile (defaults): full native ctx 262144, high max output tokens,
# batched multi-session decode, large tool-call ID memory, CORS for local UIs,
# long-prompt KV cold saves, continued frontiers for multi-turn tool loops.

set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
QWEN_DIR="${QWEN_DS4_DIR:-$ROOT_DIR}"
SERVER_BIN="$QWEN_DIR/ds4-server"
MODEL="${QWEN_DS4_MODEL:-$HOME/models/gguf/Qwen3.8-Flash-Next-Uncensored-Q4KQ8-NativeBF16Ngrams.gguf}"
HOST="127.0.0.1"
PORT=8000
# Full native context. The checkpoint declares 262144 tokens. Default still
# allocates that window, but memory/swap behavior is UNMEASURED on this main
# engine + native BF16 n-gram pack (old 99.20/115.23 GiB and zero-swapout
# numbers were for qwen38-ivan + Q4_1 sidecar and do not transfer).
# Prefer QWEN_DS4_CTX=8192 for first bring-up; re-measure resident, Metal
# working set, and swapouts before treating 262144 (or YaRN 393216) as safe.
CTX="${QWEN_DS4_CTX:-262144}"
NATIVE_CTX=262144
# Set QWEN_DS4_YARN=0 to allocate the long context without rescaling rope
# (ds4 then warns, and prompts past 262144 tokens degrade).
YARN="${QWEN_DS4_YARN:-auto}"
TOKENS="${QWEN_DS4_TOKENS:-65536}"  # agentic default max output
KV_DIR="${QWEN_DS4_KV_DIR:-$HOME/.ds4/server-kv/qwen38-flash-next-uncen-q4kq8-native-bf16}"
KV_BUDGET_MB=131072
KV_MIN_TOKENS="${QWEN_DS4_KV_MIN_TOKENS:-512}"  # cache short tool turns
KV_COLD_MAX_TOKENS="${QWEN_DS4_KV_COLD_MAX_TOKENS:-131072}"  # long agent system+tools prompts
# Continued-frontier snapshot interval. ds4 rounds the interval up to a
# multiple of --kv-cache-boundary-align-tokens (2048). Defaults keep the
# prior policy (20480). Frontier disk cost and resume latency are pack/engine
# dependent — re-measure on this native BF16 path before tuning. 0 disables.
KV_CONTINUED_INTERVAL="${QWEN_DS4_KV_CONTINUED_INTERVAL:-20480}"
KV_ALIGN_TOKENS=2048
# Model-embedded Qwen3.8 MTP speculation (the pack ships the MTP block).
# Set QWEN_DS4_MTP=0 to fall back to plain decode.
MTP="${QWEN_DS4_MTP:-1}"
# Per-step MTP acceptance/verify timing. Diagnostic only: the extra logging
# perturbs the decode rate it measures, so leave it off for benchmarks.
MTP_TIMING="${QWEN_DS4_MTP_TIMING:-0}"
# Multi-token MTP draft width. Default 1. Throughput/acceptance for draft>1
# is UNMEASURED on current main (batched MTP / concurrent n-gram reads landed
# upstream). Prior qwen38-ivan+sidecar figures (draft 2/4/8 ≈ 0.986x/0.953x/
# 0.960x of draft-1) are historical only — re-measure before changing this.
# ds4 caps the width at 16.
MTP_DRAFT="${QWEN_DS4_MTP_DRAFT:-1}"
MTP_DRAFT_MAX=16
PREFILL_CHUNK="${QWEN_DS4_PREFILL_CHUNK:-1024}"
# Concurrent resident sessions for agentic multi-turn / parallel tool clients.
# main batches decode-ready sessions when N>1. 1 = single stream.
BATCHED_SESSION="${QWEN_DS4_BATCHED_SESSION:-1}"
# Exact tool-call IDs kept in RAM (agent loops generate many IDs).
TOOL_MEMORY_MAX_IDS="${QWEN_DS4_TOOL_MEMORY_MAX_IDS:-200000}"  # optional; upstream docs this mainly for DSML/DeepSeek tool history
# Browser / local agent UIs often need CORS on 127.0.0.1.
CORS="${QWEN_DS4_CORS:-1}"
# Optional request trace for debugging agent tool loops (empty = off).
TRACE="${QWEN_DS4_TRACE:-}"
WIRED_LIMIT_MIN_MB=114688
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-15}
THERMALFORGE="${THERMALFORGE:-$(command -v thermalforge || echo /opt/homebrew/bin/thermalforge)}"
FAN_COMMAND_TIMEOUT_SECONDS=5
FAN_RESTORE_TIMEOUT_SECONDS=5
FAN_PROFILE="${FAN_PROFILE:-performance}"  # agentic sustained load; override silent|balanced|max
FAN_WATCH_INTERVAL_SECONDS=2
fan_max_owned=false
fan_watch_pid=""

usage() {
    cat <<EOF
Usage: ./run-qwen38-ds4.sh

Full-context agentic launcher for antirez main + Qwen3.8 Uncensored
native BF16 n-grams (Q4_K gate/up + Q8_0 down + embedded MTP). No --ple.

Defaults tuned for agent / tool-loop workloads:
  ctx 262144 (full native) · max tokens 65536 · batched-session 1
  CORS on · KV cold-max 131072 · MTP on · tool-memory optional
  fan profile performance · prefill-chunk 1024

Profiles:
  # First validation
  QWEN_DS4_CTX=8192 ./run-qwen38-ds4.sh

  # Production: maximum native context, single agent/session (default)
  ./run-qwen38-ds4.sh

  # Parallel agents: start conservatively (do NOT pair C4 with 262K unmeasured)
  QWEN_DS4_CTX=32768 QWEN_DS4_BATCHED_SESSION=4 ./run-qwen38-ds4.sh


Environment:
  MONITOR_INTERVAL_SECONDS=N     Monitor interval (default: 15)
  QWEN_DS4_MODEL=PATH            GGUF path (default: Uncensored native BF16 pack)
  QWEN_DS4_CTX=N                 Context tokens (default: 262144)
  QWEN_DS4_TOKENS=N              Default max output tokens (default: 65536)
  QWEN_DS4_BATCHED_SESSION=N     Resident sessions to batch (default: 1; use 4+ only with smaller CTX)
  QWEN_DS4_TOOL_MEMORY_MAX_IDS=N Tool-call ID RAM cache (default: 200000; DSML-oriented, optional for Qwen)
  QWEN_DS4_CORS=0                Disable CORS (default: on)
  QWEN_DS4_TRACE=FILE            Write prompt/tool trace log (default: off)
  QWEN_DS4_KV_DIR=PATH           KV disk-cache directory
  QWEN_DS4_KV_MIN_TOKENS=N       Min tokens to save/load (default: 512)
  QWEN_DS4_KV_COLD_MAX_TOKENS=N  Cold first-prompt save cap (default: 131072)
  QWEN_DS4_KV_CONTINUED_INTERVAL=N  Continued frontier interval (default: 20480)
  QWEN_DS4_PREFILL_CHUNK=N       Prefill chunk (default: 1024)
  QWEN_DS4_YARN=F                YaRN factor auto|0|N (default: auto)
  QWEN_DS4_MTP=0                 Disable MTP
  QWEN_DS4_MTP_DRAFT=N           MTP draft width 1..16 (default: 1)
  QWEN_DS4_MTP_TIMING=1          MTP timing logs
  FAN_PROFILE=name               silent|balanced|performance|max (default: performance)
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

if [[ ! $CTX =~ ^[1-9][0-9]*$ ]]; then
    echo "QWEN_DS4_CTX must be a positive integer; got: $CTX" >&2
    exit 2
fi
if [[ ! $KV_CONTINUED_INTERVAL =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "QWEN_DS4_KV_CONTINUED_INTERVAL must be a non-negative integer; got: $KV_CONTINUED_INTERVAL" >&2
    exit 2
fi
if [[ ! $MTP_DRAFT =~ ^[1-9][0-9]*$ ]] || (( MTP_DRAFT > MTP_DRAFT_MAX )); then
    echo "QWEN_DS4_MTP_DRAFT must be an integer between 1 and $MTP_DRAFT_MAX; got: $MTP_DRAFT" >&2
    exit 2
fi
if [[ ! $TOKENS =~ ^[1-9][0-9]*$ ]]; then
    echo "QWEN_DS4_TOKENS must be a positive integer; got: $TOKENS" >&2
    exit 2
fi
if [[ ! $BATCHED_SESSION =~ ^[1-9][0-9]*$ ]] || (( BATCHED_SESSION > 64 )); then
    echo "QWEN_DS4_BATCHED_SESSION must be 1..64; got: $BATCHED_SESSION" >&2
    exit 2
fi
if [[ ! $TOOL_MEMORY_MAX_IDS =~ ^[1-9][0-9]*$ ]]; then
    echo "QWEN_DS4_TOOL_MEMORY_MAX_IDS must be a positive integer; got: $TOOL_MEMORY_MAX_IDS" >&2
    exit 2
fi
if [[ ! $KV_MIN_TOKENS =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "QWEN_DS4_KV_MIN_TOKENS must be a non-negative integer; got: $KV_MIN_TOKENS" >&2
    exit 2
fi
if [[ ! $KV_COLD_MAX_TOKENS =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "QWEN_DS4_KV_COLD_MAX_TOKENS must be a non-negative integer; got: $KV_COLD_MAX_TOKENS" >&2
    exit 2
fi
# ds4 reads the factor from the environment, so compute it here and export it.
if [[ $YARN == "auto" ]]; then
    if (( CTX > NATIVE_CTX )); then
        YARN_FACTOR=$(python3 -c 'import sys; print(f"{int(sys.argv[1])/int(sys.argv[2]):.6g}")' "$CTX" "$NATIVE_CTX")
    else
        YARN_FACTOR=0
    fi
elif [[ $YARN == "0" ]]; then
    YARN_FACTOR=0
else
    YARN_FACTOR=$YARN
fi
if [[ $YARN_FACTOR != 0 ]]; then
    export DS4_QWEN4_YARN_FACTOR="$YARN_FACTOR"
    rope_state="YaRN factor $YARN_FACTOR over $NATIVE_CTX native tokens"
elif (( CTX > NATIVE_CTX )); then
    rope_state="native rope, NOT rescaled -- prompts past $NATIVE_CTX tokens will degrade"
else
    rope_state="native rope ($NATIVE_CTX tokens, no rescaling needed)"
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
    echo "Build it first with: make -C \"$QWEN_DIR\" ds4-server" >&2
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

    python3 - "$THERMALFORGE" "$action" \
        "$FAN_COMMAND_TIMEOUT_SECONDS" <<'PY'
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
    "$THERMALFORGE" watch --profile "$FAN_PROFILE" --interval "$FAN_WATCH_INTERVAL_SECONDS" \
        >/dev/null 2>&1 &
    fan_watch_pid=$!
    sleep 2
    if ! kill -0 "$fan_watch_pid" 2>/dev/null; then
        echo "thermalforge watch --profile $FAN_PROFILE exited immediately." >&2
        fan_watch_pid=""
        return 1
    fi
    echo "Fans under ThermalForge '$FAN_PROFILE' profile (watch pid $fan_watch_pid)."
    return 0
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

if ! fan_control probe; then
    echo "Non-interactive verified fan control is unavailable; refusing to start ds4-server." >&2
    echo "Repair it with: sudo thermalforge install" >&2
    exit 1
fi

if ! wired_limit_mb=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null); then
    echo "Could not read iogpu.wired_limit_mb; refusing to start the 90 GB model." >&2
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

server_pid=""
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


def fan_snapshot():
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
        fans = payload.get("fans") if isinstance(payload, dict) else None
        if not isinstance(fans, list) or not fans:
            return "-"
        values = []
        for fan in fans:
            actual = int(fan["rpm"])
            maximum = int(fan["max_rpm"])
            if maximum <= 0:
                return "-"
            values.append("{}={}/{}".format(fan.get("name", "fan?"), actual, maximum))
        return ",".join(values) + "RPM"
    except (IndexError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return "-"


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
    fan_status = fan_snapshot()
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
        f"cpu={cpu_percent}% mem={memory_percent}% rss={rss_kib / 1048576:.1f}GiB "
        f"fans={fan_status} "
        f"kv={used_gib:.1f}/{budget_gib:.1f}GiB({cache_percent:.1f}%) "
        f"disk_free={free_gib:.1f}GiB{warning}"
    )

    deadline = time.monotonic() + interval
    while server_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1, remaining))
restore_fans_after_server()
# Neutralize std streams so the interpreter shutdown flush cannot fail
# against a dead terminal (CPython would otherwise exit with status 120).
try:
    sys.stdout = sys.stderr = open(os.devnull, "w")
except OSError:
    pass
' "$watched_pid" "$HOST" "$PORT" "$KV_DIR" "$KV_BUDGET_MB" \
        "$MONITOR_INTERVAL_SECONDS" "$THERMALFORGE" \
        "$FAN_COMMAND_TIMEOUT_SECONDS" "$MACMON" &
    monitor_pid=$!
}

# Report the interval ds4 actually uses after its alignment rounding.
if (( KV_CONTINUED_INTERVAL == 0 )); then
    kv_continued_state="disabled"
else
    kv_continued_state="every $(( (KV_CONTINUED_INTERVAL + KV_ALIGN_TOKENS - 1) / KV_ALIGN_TOKENS * KV_ALIGN_TOKENS )) tokens"
fi

MTP_ARGS=()
if [[ $MTP != 0 ]]; then
    MTP_ARGS+=(--mtp --mtp-draft "$MTP_DRAFT")
    if [[ $MTP_TIMING != 0 ]]; then
        MTP_ARGS+=(--mtp-timing)
    fi
fi

# The old banner claimed "width 2", a width ds4 was never given: the width comes
# from --mtp-draft, not from --mtp. Report the width actually passed, and make no
# claim about what it does internally.
if [[ $MTP == 0 ]]; then
    mtp_state="disabled"
else
    mtp_state="enabled (--mtp-draft $MTP_DRAFT)"
fi
if [[ $MTP != 0 && $MTP_TIMING != 0 ]]; then
    mtp_state+=" + timing (--mtp-timing)"
fi

ds4_branch=$(git -C "$QWEN_DIR" branch --show-current 2>/dev/null || true)
ds4_commit=$(git -C "$QWEN_DIR" rev-parse --short=12 HEAD 2>/dev/null || true)
BUILD_INPUTS=('*.c' '*.h' '*.m' '*.metal' 'Makefile')
if [[ -n $(git -C "$QWEN_DIR" status --porcelain 2>/dev/null) ]]; then
    ds4_tree="dirty"
else
    ds4_tree="clean"
fi
if [[ -n $(git -C "$QWEN_DIR" status --porcelain -- "${BUILD_INPUTS[@]}" 2>/dev/null) ]]; then
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
    source_epoch=$(git -C "$QWEN_DIR" log -1 --format=%ct -- "${BUILD_INPUTS[@]}" 2>/dev/null || echo 0)
    stale_reason="older than the newest source commit"
else
    source_epoch=$( (cd "$QWEN_DIR" && git ls-files -z -- "${BUILD_INPUTS[@]}" \
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
Starting monitored ds4-server (Qwen3.8 Uncensored native BF16 n-grams, Q4_K/Q8_0, main)
  model:      $MODEL
  n-grams:    native BF16 (embedded; disk-backed pread)
  endpoint:   http://$HOST:$PORT
  context:    $CTX
  rope:       $rope_state
  max tokens: $TOKENS
  prefill:    $PREFILL_CHUNK tokens/chunk
  KV cache:   $KV_DIR (${KV_BUDGET_MB} MiB budget, min ${KV_MIN_TOKENS}, cold max ${KV_COLD_MAX_TOKENS} tokens, continued ${kv_continued_state})
  build:      ${ds4_branch:-unknown} @ ${ds4_commit:-unknown}
  repo:       $ds4_tree (code: $ds4_code)
  binary:     $binary_state, built $binary_stamp
  model file: $model_stamp
  MTP:        $mtp_state
  batched:    $BATCHED_SESSION session(s)
  tools RAM:  $TOOL_MEMORY_MAX_IDS tool-call IDs
  CORS:       $([ "$CORS" = 0 ] && echo off || echo on)
  trace:      ${TRACE:-off}
  profile:    agentic full-ctx
  monitor:    every ${MONITOR_INTERVAL_SECONDS}s
  fans:       ThermalForge '$FAN_PROFILE' profile (temperature-driven); Apple auto on stop
  wired limit: ${wired_limit_mb:-unknown} MiB

API:        http://$HOST:$PORT/v1/chat/completions
Models:     http://$HOST:$PORT/v1/models
Dashboard:  http://$HOST:$PORT/dashboard   # this branch LOCAL PATCH
Health:     http://$HOST:$PORT/health      # this branch LOCAL PATCH; pure main: /v1/models
Stats:      http://$HOST:$PORT/stats       # this branch LOCAL PATCH

Press Ctrl-C once to stop the server, restore automatic fans, and stop the monitor.
EOF

AGENT_ARGS=()
if (( BATCHED_SESSION > 1 )); then
    AGENT_ARGS+=(--batched-session "$BATCHED_SESSION")
fi
AGENT_ARGS+=(--tool-memory-max-ids "$TOOL_MEMORY_MAX_IDS")
if [[ $CORS != 0 ]]; then
    AGENT_ARGS+=(--cors)
fi
if [[ -n $TRACE ]]; then
    AGENT_ARGS+=(--trace "$TRACE")
fi

python3 -c '
import os
import sys

os.setsid()
os.chdir(sys.argv[1])
os.execv(sys.argv[2], sys.argv[2:])
' "$QWEN_DIR" "$SERVER_BIN" --metal \
    --model "$MODEL" \
    ${MTP_ARGS[@]+"${MTP_ARGS[@]}"} \
    ${AGENT_ARGS[@]+"${AGENT_ARGS[@]}"} \
    --ctx "$CTX" --tokens "$TOKENS" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --power 100 \
    --host "$HOST" --port "$PORT" \
    --kv-disk-dir "$KV_DIR" --kv-disk-space-mb "$KV_BUDGET_MB" \
    --kv-cache-min-tokens "$KV_MIN_TOKENS" \
    --kv-cache-cold-max-tokens "$KV_COLD_MAX_TOKENS" \
    --kv-cache-continued-interval-tokens "$KV_CONTINUED_INTERVAL" \
    --kv-cache-reject-different-quant &
server_pid=$!

start_monitor "$server_pid"
if ! set_fans_max; then
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
