#!/bin/bash
# Clear the DS4 disk KV cache. Only ever the .kv files inside the model's own
# cache dir -- never the dir itself, never anything else under ~/.ds4.
# Regenerable by definition: cold prefill rebuilds any entry that is needed.
set -euo pipefail
DIR="${1:-$HOME/.ds4/server-kv/glm-5.3-flash-uncen-q2}"
[ -d "$DIR" ] || { echo "no such kv dir: $DIR"; exit 0; }
case "$DIR" in *"/.ds4/server-kv/"*) ;; *) echo "refusing: $DIR is not a ds4 kv dir"; exit 1;; esac
before=$(du -sk "$DIR" | cut -f1)
n=$(find "$DIR" -maxdepth 1 -name '*.kv' | wc -l | tr -d ' ')
find "$DIR" -maxdepth 1 -name '*.kv' -delete
after=$(du -sk "$DIR" | cut -f1)
echo "kv cache: removed $n files, $((before/1024/1024)) GiB -> $((after/1024/1024)) GiB"
