#!/usr/bin/env python3
"""Replay a recorded ds4-server session and report per-turn latency.

Recording: run ds4-server with DS4_SERVER_RECORD_DIR=<dir>; every accepted
request writes <seq>.body (raw HTTP body) and <seq>.json (metadata incl. the
rendered-prompt FNV-1a fingerprint).

Replay:  tasks/replay.py <record-dir> [--host 127.0.0.1:8000] [--verify <new-record-dir>]
                          [--out results.json] [--limit N]

Sends the bodies in sequence order to the same endpoint kind, measures
time-to-first-byte, total wall time and generated tokens per turn, snapshots
/stats after each turn (last_prefill_tps, last_decode_tps, footprint, swap), and
prints p50/p95 for the T1/T2/T3 metrics of tasks/plan-long-ctx-decode.md.

With --verify, the server must be running with DS4_SERVER_RECORD_DIR=<new-dir>;
after the replay the rendered-prompt fingerprints of the new recording are
compared with the original, turn by turn.  A mismatch means the replay did not
reproduce the exact prompt stream and the timings are not comparable.
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import time

ENDPOINT = {
    "chat": "/v1/chat/completions",
    "anthropic": "/v1/messages",
    "responses": "/v1/responses",
    "completion": "/v1/completions",
}


def load_recording(d):
    turns = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        seq = name[:-5]
        with open(os.path.join(d, name)) as f:
            meta = json.load(f)
        body_path = os.path.join(d, seq + ".body")
        if not os.path.exists(body_path):
            continue
        with open(body_path, "rb") as f:
            body = f.read()
        turns.append((meta, body))
    turns.sort(key=lambda t: t[0]["seq"])
    return turns


def get_json(host, path):
    conn = http.client.HTTPConnection(host, timeout=30)
    conn.request("GET", path)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return json.loads(data)


def send(host, meta, body):
    """Returns (ttfb_s, total_s, completion_tokens or None, status)."""
    conn = http.client.HTTPConnection(host, timeout=3600)
    headers = {"Content-Type": "application/json",
               "Content-Length": str(len(body))}
    if meta["api"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    t0 = time.monotonic()
    conn.request("POST", ENDPOINT[meta["api"]], body=body, headers=headers)
    r = conn.getresponse()
    ttfb = None
    completion_tokens = None
    chunks = []
    while True:
        chunk = r.read(65536)
        if not chunk:
            break
        if ttfb is None:
            ttfb = time.monotonic() - t0
        chunks.append(chunk)
    total = time.monotonic() - t0
    conn.close()
    payload = b"".join(chunks)
    # usage: non-stream JSON, or the final SSE frame carrying usage
    try:
        if meta["stream"]:
            for line in payload.split(b"\n"):
                if line.startswith(b"data: ") and b"usage" in line:
                    obj = json.loads(line[6:])
                    u = obj.get("usage") or {}
                    completion_tokens = (u.get("completion_tokens")
                                         or u.get("output_tokens")
                                         or completion_tokens)
        else:
            obj = json.loads(payload)
            u = obj.get("usage") or {}
            completion_tokens = (u.get("completion_tokens")
                                 or u.get("output_tokens"))
    except (ValueError, AttributeError):
        pass
    return ttfb if ttfb is not None else total, total, completion_tokens, r.status


def pct(values, p):
    if not values:
        return float("nan")
    vs = sorted(values)
    k = (len(vs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("--host", default="127.0.0.1:8000")
    ap.add_argument("--verify", metavar="NEW_RECORD_DIR")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    turns = load_recording(args.record_dir)
    if args.limit:
        turns = turns[:args.limit]
    if not turns:
        sys.exit("no turns found in " + args.record_dir)
    health = get_json(args.host, "/health")
    route = get_json(args.host, "/stats").get("tensor_route")
    if route != "auto":
        sys.exit(f"refusing to benchmark: server tensor_route={route!r}, expected 'auto' "
                 "(DS4_METAL_ENABLE_TENSOR must be unset for comparable numbers)")
    print(f"server: {health.get('model')} uptime={health.get('uptime_s')}s turns={len(turns)}")

    rows = []
    stats_before = get_json(args.host, "/stats")
    for meta, body in turns:
        ttfb, total, gen, status = send(args.host, meta, body)
        st = get_json(args.host, "/stats")
        new_tokens = meta["prompt_tokens"] - meta["cached_tokens"]
        row = {
            "seq": meta["seq"],
            "status": status,
            "ttfb_s": ttfb,
            "total_s": total,
            "gen_tokens": gen,
            "recorded_new_tokens": new_tokens,
            "recorded_cached": meta["cached_tokens"],
            "prefill_tps": st.get("last_prefill_tps"),
            "decode_tps": st.get("last_decode_tps"),
            "live_tokens": st.get("live_tokens"),
            "footprint_mb": st.get("footprint_mb"),
            "swap_used_mb": st.get("swap_used_mb"),
        }
        rows.append(row)
        print(f"#{meta['seq']:06d} {status} ttfb={ttfb:6.2f}s total={total:7.2f}s "
              f"gen={gen if gen is not None else '-':>5} new={new_tokens:5d} "
              f"pf={row['prefill_tps']} dec={row['decode_tps']} live={row['live_tokens']}",
              flush=True)
    stats_after = get_json(args.host, "/stats")

    small = [r["ttfb_s"] for r in rows if r["recorded_new_tokens"] <= 64]
    ttfb_all = [r["ttfb_s"] for r in rows]
    dec = [r["decode_tps"] for r in rows if r["decode_tps"]]
    server_total = sum(r["total_s"] for r in rows)
    summary = {
        "turns": len(rows),
        "server_wall_s": server_total,
        "T2_small_append_ttfb_p50": pct(small, 0.5),
        "T2_small_append_ttfb_p95": pct(small, 0.95),
        "ttfb_p50": pct(ttfb_all, 0.5),
        "ttfb_p95": pct(ttfb_all, 0.95),
        "decode_tps_median": statistics.median(dec) if dec else None,
        "decode_tps_min": min(dec) if dec else None,
        "requests_delta": stats_after["requests"] - stats_before["requests"],
        "cache_hits_delta": stats_after["cache"]["hits"] - stats_before["cache"]["hits"],
        "cache_cold_delta": stats_after["cache"]["cold"] - stats_before["cache"]["cold"],
        "footprint_mb_end": stats_after.get("footprint_mb"),
        "swap_used_mb_end": stats_after.get("swap_used_mb"),
    }
    print(json.dumps(summary, indent=2))

    verify = None
    if args.verify:
        new = {m["seq"]: m for m, _ in load_recording(args.verify)}
        # the new recording's seq numbers continue the server's counter, so
        # align by order, not by seq
        new_sorted = [new[k] for k in sorted(new)][-len(turns):]
        mism = []
        for (m_old, _), m_new in zip(turns, new_sorted):
            if m_old["prompt_text_fnv1a64"] != m_new["prompt_text_fnv1a64"]:
                mism.append((m_old["seq"], m_new["seq"]))
        verify = {"compared": len(new_sorted), "prompt_mismatches": mism}
        print("verify:", json.dumps(verify))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows, "verify": verify,
                       "record_dir": args.record_dir, "host": args.host,
                       "time": time.time()}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
