#!/usr/bin/env python3
"""Delete ds4 disk KV checkpoints that no session has used for longer than a TTL.

ds4-server evicts only when its disk budget is full, and its eviction score
favours large files, so checkpoints of finished sessions can hold the whole
budget indefinitely. This prunes by age instead. It is safe while the server
runs: the server rescans the directory on every lookup and treats a vanished
file as a cache miss.

Only regular files named like the server's own are touched:
  <40 hex>.kv             deleted when idle longer than the TTL
  <40 hex>.kv.tmp.<pid>   deleted when the writing process is gone (crash orphan)
"""

import argparse
import os
import re
import struct
import sys
import time

CHECKPOINT = re.compile(r"^[0-9a-fA-F]{40}\.kv$")
ORPHAN_TMP = re.compile(r"^[0-9a-fA-F]{40}\.kv\.tmp\.(\d+)$")
HEADER_BYTES = 48  # ds4_kvstore.c DS4_KVSTORE_FIXED_HEADER


def last_activity(path, st):
    """Newest of the header's created_at/last_used and the file mtime.

    The server stamps last_used in the header on every hit; mtime covers
    headers that cannot be parsed. Taking the newest keeps a file whenever
    either source says it is recent.
    """
    newest = st.st_mtime
    try:
        with open(path, "rb") as fp:
            header = fp.read(HEADER_BYTES)
        if len(header) == HEADER_BYTES and header[:3] == b"KVC":
            created_at, last_used = struct.unpack_from("<QQ", header, 24)
            newest = max(newest, created_at, last_used)
    except OSError:
        pass
    return newest


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prune(kv_dir, ttl_seconds, now, dry_run):
    removed = kept = 0
    removed_bytes = kept_bytes = 0
    errors = []
    with os.scandir(kv_dir) as entries:
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
                orphan = ORPHAN_TMP.match(entry.name)
                if orphan:
                    stale = not pid_alive(int(orphan.group(1)))
                elif CHECKPOINT.match(entry.name):
                    stale = now - last_activity(entry.path, st) > ttl_seconds
                else:
                    continue
                if not stale:
                    kept += 1
                    kept_bytes += st.st_size
                    continue
                if not dry_run:
                    os.unlink(entry.path)
                removed += 1
                removed_bytes += st.st_size
            except FileNotFoundError:
                continue  # the server evicted or renamed it concurrently
            except OSError as exc:
                errors.append(f"{entry.name}: {exc.strerror}")
    return removed, removed_bytes, kept, kept_bytes, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True, help="KV checkpoint directory")
    parser.add_argument("--ttl-hours", type=int, required=True,
                        help="delete checkpoints idle longer than this (> 0)")
    parser.add_argument("--label", default="prune", help="tag for the summary line")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ttl_hours <= 0:
        parser.error("--ttl-hours must be > 0")
    if not os.path.isdir(args.dir):
        return 0

    removed, removed_bytes, kept, kept_bytes, errors = prune(
        args.dir, args.ttl_hours * 3600, time.time(), args.dry_run)
    gib = 1024 ** 3
    verb = "would remove" if args.dry_run else "removed"
    print(f"kv-prune[{args.label}] ttl={args.ttl_hours}h {verb}={removed} "
          f"({removed_bytes / gib:.1f}GiB) kept={kept} ({kept_bytes / gib:.1f}GiB)")
    for error in errors:
        print(f"kv-prune[{args.label}] error {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
