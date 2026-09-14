#!/usr/bin/env bash
# Start the GLM 5.3 Flash server from this checkout and print one-line health/resource
# snapshots while it runs. Server logs remain attached to this terminal.
#
# Usage:
#   ./run-glm-ds4.sh
#   MONITOR_INTERVAL_SECONDS=30 ./run-glm-ds4.sh
# Fans run at verified maximum while ds4-server runs, then return to Apple auto.

set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
GLM_DIR=$(cd -- "${GLM_DS4_DIR:-$ROOT_DIR}" && pwd -P)
SERVER_BIN="$GLM_DIR/ds4-server"
MODEL="$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-Q2.gguf"
HOST="127.0.0.1"
PORT=8000
CTX=262144
TOKENS=32768
KV_DIR="$HOME/.ds4/server-kv/glm-5.3-flash-uncen-q2"
KV_BUDGET_MB=131072
KV_MIN_TOKENS=2048
KV_COLD_MAX_TOKENS=65536
# Model-embedded GLM 5.3 MTP speculation. Required for the Metal width-2
# verify fast path; set GLM_DS4_MTP=0 to fall back to plain decode.
MTP="${GLM_DS4_MTP:-1}"
# Per-cycle MTP logging is diagnostic and must stay off for scored decode.
MTP_TIMING="${GLM_DS4_MTP_TIMING:-0}"
S55_MODE="${GLM_DS4_S55:-0}"
MTP_COUNTERS="${GLM_DS4_MTP_COUNTERS:-$S55_MODE}"
S55_M0_SEGMENT_TOKENS="${GLM_DS4_S55_M0_SEGMENT_TOKENS:-0}"
S55_M0_REPEATS="${GLM_DS4_S55_M0_REPEATS:-0}"
S55_H25_PAIR_AB="${GLM_DS4_S55_H25_PAIR_AB:-0}"
INDEXER_PAIR_EXACT="${DS4_METAL_GLM53_INDEXER_SCORE_PAIR_EXACT:-0}"
WIRED_LIMIT_MIN_MB=114688
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-15}
THERMALFORGE=$(command -v thermalforge || true)
FAN_COMMAND_TIMEOUT_SECONDS=5
FAN_RAMP_TIMEOUT_SECONDS=20
FAN_RESTORE_TIMEOUT_SECONDS=5
FAN_RAMP_MIN_PERCENT=98
fan_max_owned=false

usage() {
    cat <<'EOF'
Usage: ./run-glm-ds4.sh

Starts the glm-5.3-flash branch ds4-server with GLM 5.3 Flash Q2 (ctx 262144) and prints
health, throughput, process memory, KV disk-cache use, and free disk space.
Fans run at maximum while the server runs and return to Apple auto when it stops.

Environment:
  MONITOR_INTERVAL_SECONDS=N  Monitoring interval in seconds (default: 15)
  GLM_DS4_MTP=0               Disable model-embedded MTP speculation
  GLM_DS4_MTP_TIMING=1        Enable diagnostic --mtp-timing logging
  GLM_DS4_MTP_COUNTERS=0|1    Toggle cycle counters (Phase 2 default: 1; M0: 0)
  GLM_DS4_S55=1               Require a clean Phase 2 measurement environment
  GLM_DS4_S55_M0_SEGMENT_TOKENS=64
                              Enable wash-separated MATCHED-STATE NOMTP M0
  GLM_DS4_S55_M0_REPEATS=2    ABBA/BAAB repetitions (minimum contract: 2)
  GLM_DS4_S55_H25_PAIR_AB=1   Compare H25 baseline/candidate in the M0 schedule
  DS4_METAL_GLM53_INDEXER_SCORE_PAIR_EXACT=1
                              Enable the measured exact width-2 indexer kernel
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

if [[ $S55_MODE != 0 ]]; then
    if [[ $MONITOR_INTERVAL_SECONDS != 15 ]]; then
        echo "S55 Phase 2 mode requires MONITOR_INTERVAL_SECONDS=15" >&2
        exit 2
    fi
    forbidden_env=()
    while IFS='=' read -r name _; do
        case $name in
            DS4_METAL_GLM53_INDEXER_SCORE_PAIR_EXACT) ;;
            DS4_*|MTL_*|METAL_*|DYLD_*) forbidden_env+=("$name") ;;
        esac
    done < <(env)
    if (( ${#forbidden_env[@]} != 0 )); then
        printf 'S55 Phase 2 mode refuses environment override: %s\n' \
            "${forbidden_env[*]}" >&2
        exit 2
    fi
fi
if [[ $INDEXER_PAIR_EXACT != 0 && $INDEXER_PAIR_EXACT != 1 ]]; then
    echo "DS4_METAL_GLM53_INDEXER_SCORE_PAIR_EXACT must be 0 or 1" >&2
    exit 2
fi
if [[ $S55_H25_PAIR_AB != 0 && $S55_H25_PAIR_AB != 1 ]]; then
    echo "GLM_DS4_S55_H25_PAIR_AB must be 0 or 1" >&2
    exit 2
fi
if [[ $S55_M0_SEGMENT_TOKENS != 0 || $S55_M0_REPEATS != 0 ]]; then
    if [[ $S55_MODE == 0 || $MTP == 0 || $MTP_TIMING != 0 ||
          ! $S55_M0_SEGMENT_TOKENS =~ ^[0-9]+$ ||
          ! $S55_M0_REPEATS =~ ^[0-9]+$ ]] ||
       (( 10#$S55_M0_SEGMENT_TOKENS < 32 ||
          10#$S55_M0_SEGMENT_TOKENS > 4096 ||
          10#$S55_M0_REPEATS < 2 || 10#$S55_M0_REPEATS > 64 ||
          10#$S55_M0_SEGMENT_TOKENS * 4 * 10#$S55_M0_REPEATS < 512 )); then
        echo "S55 M0 requires S55=1, MTP on, timing off, and >=512 scored tokens per arm" >&2
        exit 2
    fi
    MTP_COUNTERS=0
fi
if [[ $S55_H25_PAIR_AB == 1 &&
      ($S55_M0_SEGMENT_TOKENS == 0 ||
       -n "${DS4_METAL_GLM53_INDEXER_SCORE_PAIR_EXACT+x}") ]]; then
    echo "S55 H25 pair A/B requires M0 and refuses a global H25 override" >&2
    exit 2
fi

for command in lsof macmon python3 ps sysctl; do
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
    echo "Install it with: thermalforge install" >&2
    exit 1
fi

fan_control() {
    local action=$1

    python3 - "$THERMALFORGE" "$MACMON" "$action" \
        "$FAN_COMMAND_TIMEOUT_SECONDS" "$FAN_RAMP_MIN_PERCENT" <<'PY'
import json
import subprocess
import sys

path, macmon, action = sys.argv[1:4]
timeout = int(sys.argv[4])
minimum_fraction = int(sys.argv[5]) / 100
valid_actions = {"probe", "max", "auto", "status", "verify-max", "verify-auto"}
if action not in valid_actions:
    raise SystemExit(f"unsupported fan action: {action}")

command_action = action if action in {"max", "auto"} else "status"
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
if action in {"max", "auto"}:
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
if action == "verify-auto":
    raise SystemExit(0 if modes <= {"auto", "automatic", "default"} else 1)

for fan in fans:
    try:
        target = int(fan["target_rpm"])
        maximum = int(fan["max_rpm"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(1)
    if (
        str(fan.get("mode", "")).lower() != "manual"
        or maximum <= 0
        or target < maximum * 0.98
    ):
        raise SystemExit(1)

try:
    sample = subprocess.run(
        [macmon, "pipe", "--samples", "1", "--interval", "250"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
except subprocess.TimeoutExpired:
    raise SystemExit(1)
if sample.returncode != 0:
    raise SystemExit(1)
try:
    samples = [line for line in sample.stdout.splitlines() if line.strip()]
    hardware_fans = json.loads(samples[-1])["fans"]
    if not isinstance(hardware_fans, list) or len(hardware_fans) != len(fans):
        raise ValueError("fan count mismatch")
except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
for fan in hardware_fans:
    try:
        actual = int(fan["rpm"])
        maximum = int(fan["max_rpm"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(1)
    if maximum <= 0 or actual < maximum * minimum_fraction:
        raise SystemExit(1)

summary = " ".join(
    "fan{}={}/{}RPM".format(index, int(fan["rpm"]), int(fan["max_rpm"]))
    for index, fan in enumerate(hardware_fans)
)
print(summary)
PY
}

set_fans_max() {
    local deadline fan_status

    # The controller may apply the write before reporting failure, so cleanup
    # owns restoration from the moment the privileged command starts.
    fan_max_owned=true
    fan_control max || return 1
    deadline=$((SECONDS + FAN_RAMP_TIMEOUT_SECONDS))
    while ((SECONDS < deadline)); do
        if [[ -n ${server_pid:-} ]] && ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ds4-server stopped while fans were ramping." >&2
            return 1
        fi
        if fan_status=$(fan_control verify-max 2>/dev/null); then
            echo "Fans at maximum: $fan_status"
            return 0
        fi
        sleep 1
    done
    echo "Fans did not reach ${FAN_RAMP_MIN_PERCENT}% of maximum within ${FAN_RAMP_TIMEOUT_SECONDS}s." >&2
    fan_control status >&2 || true
    return 1
}

restore_fans() {
    local deadline attempt

    [[ $fan_max_owned == true ]] || return 0
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
    echo "Repair it with: thermalforge install" >&2
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
import math
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
fan_minimum_fraction = int(sys.argv[10]) / 100
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
            return "-", None, "-", "-", "-", "-", "-", "-"
        values = []
        at_max = True
        for index, fan in enumerate(fans):
            actual = int(fan["rpm"])
            maximum = int(fan["max_rpm"])
            if actual < 0 or maximum <= 0:
                return "-", None, "-", "-", "-", "-", "-", "-"
            values.append("fan{}={}/{}".format(index, actual, maximum))
            at_max = at_max and actual >= maximum * fan_minimum_fraction
        temp = payload.get("temp") if isinstance(payload.get("temp"), dict) else {}
        memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}

        def metric(source, key, allow_zero=False):
            if key not in source:
                raise ValueError("missing metric " + key)
            value = float(source[key])
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError("invalid metric " + key)
            return value

        gpu_power = "{:.1f}W".format(metric(payload, "gpu_power"))
        sys_power = "{:.1f}W".format(metric(payload, "sys_power"))
        gpu_temp = "{:.1f}C".format(metric(temp, "gpu_temp_avg"))
        cpu_temp = "{:.1f}C".format(metric(temp, "cpu_temp_avg"))
        ram_used = "{:.1f}GiB".format(metric(memory, "ram_usage") / (1024 ** 3))
        swap_used = "{:.1f}GiB".format(
            metric(memory, "swap_usage", allow_zero=True) / (1024 ** 3)
        )
        return (",".join(values) + "RPM", at_max, gpu_power, sys_power,
                gpu_temp, cpu_temp, ram_used, swap_used)
    except (IndexError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return "-", None, "-", "-", "-", "-", "-", "-"


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
    # /v1/models proves liveness; the local /stats endpoint supplies telemetry.
    health = fetch_json("/v1/models")
    stats = fetch_json("/stats")

    if health is not None and health.get("object") == "list":
        health_status = "ok"
        ever_healthy = True
    elif health is not None:
        health_status = "unexpected"
    else:
        health_status = "unreachable" if ever_healthy else "starting"

    if stats is None:
        summary = "state=- queue=- clients=- live=- requests=- hits=- cold=- cached=- prefill=- decode=-"
    else:
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
    (fan_status, fans_at_max, gpu_power, sys_power, gpu_temp, cpu_temp,
     ram_used, swap_used) = fan_snapshot()
    used_bytes = cache_size()
    used_gib = used_bytes / (1024 ** 3)
    budget_gib = budget_bytes / (1024 ** 3)
    cache_percent = used_bytes * 100 / budget_bytes
    free_gib = disk_free() / (1024 ** 3)
    warnings = []
    if cache_percent >= 90:
        warnings.append("kv-cache-near-full")
    if ever_healthy and fans_at_max is False:
        warnings.append("fans-below-max")
    warning = "".join(f" warning={value}" for value in warnings)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    emit(
        f"[monitor {timestamp}] health={health_status} {summary} "
        f"cpu={cpu_percent}% mem={memory_percent}% rss={rss_kib / 1048576:.1f}GiB "
        f"fans={fan_status} gpu_power={gpu_power} sys_power={sys_power} "
        f"gpu_temp={gpu_temp} cpu_temp={cpu_temp} "
        f"ram_used={ram_used} swap_used={swap_used} "
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
        "$FAN_COMMAND_TIMEOUT_SECONDS" "$MACMON" \
        "$FAN_RAMP_MIN_PERCENT" &
    monitor_pid=$!
}

MTP_ARGS=()
if [[ $MTP != 0 ]]; then
    MTP_ARGS+=(--mtp)
    if [[ $MTP_COUNTERS != 0 ]]; then
        MTP_ARGS+=(--mtp-counters)
    fi
    if [[ $MTP_TIMING != 0 ]]; then
        MTP_ARGS+=(--mtp-timing)
    fi
fi
M0_ARGS=()
KV_ARGS=(
    --kv-disk-dir "$KV_DIR" --kv-disk-space-mb "$KV_BUDGET_MB"
    --kv-cache-min-tokens "$KV_MIN_TOKENS"
    --kv-cache-cold-max-tokens "$KV_COLD_MAX_TOKENS"
    --kv-cache-reject-different-quant
)
if [[ $S55_M0_SEGMENT_TOKENS != 0 ]]; then
    M0_ARGS+=(--s55-m0-segment-tokens "$S55_M0_SEGMENT_TOKENS")
    M0_ARGS+=(--s55-m0-repeats "$S55_M0_REPEATS")
    if [[ $S55_H25_PAIR_AB == 1 ]]; then
        M0_ARGS+=(--s55-h25-pair-ab)
    fi
    KV_ARGS=()
fi

cat <<EOF
Starting monitored ds4-server (GLM 5.3 Flash, branch glm-5.3-flash)
  binary:     $SERVER_BIN
  model:      $MODEL
  endpoint:   http://$HOST:$PORT
  context:    $CTX
  max tokens: $TOKENS
  KV cache:   $(if [[ $S55_M0_SEGMENT_TOKENS != 0 ]]; then echo -n "disabled for checkpoint-bound M0"; else echo -n "$KV_DIR (${KV_BUDGET_MB} MiB budget, min ${KV_MIN_TOKENS}, cold max ${KV_COLD_MAX_TOKENS} tokens)"; fi)
  MTP:        $(if [[ $MTP != 0 ]]; then echo -n "enabled (--mtp, width 2)"; else echo -n "disabled"; fi)$(if [[ $MTP != 0 && $MTP_COUNTERS != 0 ]]; then echo -n " + counters (--mtp-counters)"; fi)$(if [[ $MTP != 0 && $MTP_TIMING != 0 ]]; then echo -n " + timing (--mtp-timing)"; fi)
  H25 pair:   $(if [[ $S55_H25_PAIR_AB == 1 ]]; then echo -n "balanced baseline/candidate"; elif [[ $INDEXER_PAIR_EXACT == 1 ]]; then echo -n "enabled (exact width-2 indexer)"; else echo -n "disabled"; fi)
  S55:        $(if [[ $S55_H25_PAIR_AB == 1 ]]; then echo -n "H25 contract-locked (ABBA/BAAB + wash)"; elif [[ $S55_M0_SEGMENT_TOKENS != 0 ]]; then echo -n "M0 contract-locked (ABBA/BAAB + wash; only declared H25 override permitted)"; elif [[ $S55_MODE != 0 ]]; then echo -n "Phase 2 contract-locked (no DS4/Metal/DYLD overrides)"; else echo -n "off"; fi)
  monitor:    every ${MONITOR_INTERVAL_SECONDS}s
  fans:       ThermalForge max + macmon RPM verification; Apple auto on stop
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
    ${M0_ARGS[@]+"${M0_ARGS[@]}"} \
    --ctx "$CTX" --tokens "$TOKENS" \
    --power 100 \
    --host "$HOST" --port "$PORT" \
    ${KV_ARGS[@]+"${KV_ARGS[@]}"} &
server_pid=$!

start_monitor "$server_pid"
if ! set_fans_max; then
    echo "Verified maximum fan speed is required; stopping ds4-server." >&2
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
