#!/usr/bin/env python3
"""Control-referenced greedy quality on a live ds4-server.

Collects or scores /v1/chat/completions (temp 0). Thinking disabled via
chat_template_kwargs; falls back to a no-think completions prefix.
Not official NLL — HTTP has no logprobs. Metric is greedy agreement with
frozen control continuations from the same prompt set.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[2] / "gguf-tools/quality-testing/prompts.jsonl"
BASE = "http://127.0.0.1:8000"
N_CASES = 40
MAX_TOKENS = 32


def post_json(path: str, body: dict, timeout: int = 180) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_prompts(n: int) -> list[dict]:
    rows = []
    for line in PROMPTS.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if len(rows) >= n:
            break
    return rows


def complete_chat(prompt: str) -> tuple[str, dict]:
    body = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        d = post_json("/v1/chat/completions", body)
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return text, d
    except urllib.error.HTTPError:
        prefix = (
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        d = post_json("/v1/completions", {
            "model": "qwen3.8-flash-next",
            "prompt": prefix,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        })
        text = ((d.get("choices") or [{}])[0].get("text") or "")
        return text, d


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def collect(n: int) -> dict:
    cases = []
    for row in load_prompts(n):
        text, d = complete_chat(row["prompt"])
        choice = (d.get("choices") or [{}])[0]
        cases.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "text": text,
            "finish": choice.get("finish_reason"),
            "usage": d.get("usage"),
        })
        print(f"{row['id']} chars={len(text)} finish={choice.get('finish_reason')}", flush=True)
    return {
        "n": len(cases),
        "max_tokens": MAX_TOKENS,
        "cases": cases,
    }


def score(cand: dict, refs: dict) -> dict:
    by_id = {c["id"]: c for c in refs["cases"]}
    rows = []
    first = 0
    exact = 0
    lcp_sum = 0
    lcp_norm_sum = 0.0
    for c in cand["cases"]:
        ref = by_id[c["id"]]
        a, b = c["text"], ref["text"]
        lp = lcp(a, b)
        denom = max(len(a), len(b), 1)
        fm = int(a[:1] == b[:1] and a != "") if a and b else 0
        ex = int(a == b)
        first += fm
        exact += ex
        lcp_sum += lp
        lcp_norm_sum += lp / denom
        rows.append({
            "id": c["id"],
            "first_char_match": fm,
            "exact": ex,
            "lcp": lp,
            "lcp_norm": lp / denom,
            "cand_len": len(a),
            "ref_len": len(b),
        })
    n = len(rows)
    return {
        "n": n,
        "first_char_match": first,
        "exact": exact,
        "mean_lcp": lcp_sum / n if n else 0.0,
        "mean_lcp_norm": lcp_norm_sum / n if n else 0.0,
        "cases": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["collect", "score"], required=True)
    ap.add_argument("--n", type=int, default=N_CASES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", default="")
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == "collect":
        blob = collect(args.n)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        print("WROTE", out, "n", blob["n"])
        return
    refs = json.loads(Path(args.refs).read_text())
    cand = collect(args.n)
    metrics = score(cand, refs)
    payload = {"continuations": cand, "vs_control": metrics}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    v = metrics
    print(
        f"quality n={v['n']} exact={v['exact']}/{v['n']} "
        f"first={v['first_char_match']}/{v['n']} "
        f"mean_lcp_norm={v['mean_lcp_norm']:.4f}",
        flush=True,
    )
    print("WROTE", out)


if __name__ == "__main__":
    main()
