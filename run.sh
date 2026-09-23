#!/usr/bin/env bash
# Pick one of the local models and start it on :8000 through its own launcher.
#
# Usage:
#   ./run.sh                 # menu
#   ./run.sh glm             # GLM 5.3 Flash stage 2    -> run-glm-ds4.sh
#   ./run.sh qwen            # Qwen3.8 Flash Next       -> run-qwen38-ds4.sh
#   ./run.sh deepseek        # DeepSeek V4 Flash O2b    -> run-ds4-monitored.sh
# Per-model settings stay environment variables, e.g.
#   GLM_DS4_CTX=262144 ./run.sh glm
#   DS4_ARM=o1 ./run.sh deepseek

set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PORT=8000

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
}

launcher_for() {
    case "$1" in
        glm) echo run-glm-ds4.sh ;;
        qwen) echo run-qwen38-ds4.sh ;;
        deepseek | ds | dsv4) echo run-ds4-monitored.sh ;;
        *) return 1 ;;
    esac
}

if (( $# > 1 )); then
    usage
    exit 2
fi

case "${1:-}" in
    -h | --help)
        usage
        exit 0
        ;;
esac

# Only one model fits in memory, so say what already holds the port before
# asking which model to start.
if listener=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null) && [[ -n $listener ]]; then
    echo "Port $PORT is in use; stop that server first:" >&2
    echo "$listener" >&2
    exit 1
fi

choice="${1:-}"
if [[ -z $choice ]]; then
    PS3="Model to start on :$PORT: "
    select picked in "glm       GLM 5.3 Flash stage 2 (363K ctx)" \
                     "qwen      Qwen3.8 Flash Next (262K ctx)" \
                     "deepseek  DeepSeek V4 Flash O2b (363K ctx)"; do
        [[ -n $picked ]] && break
    done
    [[ -n ${picked:-} ]] || exit 1
    choice=${picked%% *}
fi

if ! launcher=$(launcher_for "$choice"); then
    echo "Unknown model: $choice (expected glm, qwen or deepseek)" >&2
    exit 2
fi

exec "$ROOT_DIR/$launcher"
