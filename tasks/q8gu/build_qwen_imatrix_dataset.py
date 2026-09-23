#!/usr/bin/env python3
"""Qwen-rendered imatrix prompts for the uncensored Flash-Next pack.

Not the DeepSeek DSML corpus. Markers match ds4_engine_collect_imatrix.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "imatrix"
MARKER = "===== DS4_IMATRIX_PROMPT"

SYSTEM = "You are a local coding and tools assistant. Be concise and exact."


def render(user: str, *, think: bool, extra_system: str = "") -> str:
    sys = SYSTEM if not extra_system else SYSTEM + "\n" + extra_system
    think_body = "<think>\n" if think else "<think>\n\n</think>\n\n"
    return (
        f"<|im_start|>system\n{sys}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{think_body}"
    )


def coding() -> list[tuple[str, str]]:
    rows = []
    qt = ROOT / "gguf-tools/quality-testing/prompts.jsonl"
    if qt.is_file():
        for i, line in enumerate(qt.read_text().splitlines()):
            if i >= 24:
                break
            p = json.loads(line)["prompt"]
            rows.append((f"code-{i:03d}", p))
    extras = [
        ("code-rust-own", "Write a Rust function that takes ownership of a Vec<u8> and returns a hex string without extra copies."),
        ("code-py-gil", "Show a ctypes call that releases the GIL while a C encoder runs over float32 blocks."),
        ("code-go-race", "Fix a Go data race on a map written from two goroutines. Small example."),
        ("code-sql", "Write a Postgres query that finds duplicate (user_id, day) rows and keeps the latest updated_at."),
        ("code-metal", "Explain why a Metal kernel might use threadgroup memory for a 256-wide Q4_K dequant."),
        ("code-review", "Review this: `for i in range(len(xs)):\n    if xs[i] in seen: return True` — name the bug and rewrite."),
    ]
    rows.extend(extras)
    srcs = [
        ROOT / "gguf-tools/qwen4_exp_convert.py",
        ROOT / "tasks/q8gu/p2_reconstruction.py",
        ROOT / "run-qwen38-ds4.sh",
    ]
    for p in srcs:
        if p.is_file():
            text = p.read_text(errors="replace")[:6000]
            rows.append((f"src-{p.stem}", f"Review this file for correctness. Quote the risky lines.\n\n```\n{text}\n```"))
    return rows


def agent() -> list[tuple[str, str]]:
    schema = (
        "Tools (JSON):\n"
        '{"name":"read_file","parameters":{"path":"string"}}\n'
        '{"name":"bash","parameters":{"command":"string"}}\n'
        "Call tools as:\n"
        "<tool_call>\n<function=NAME>\n<parameter=path>VALUE</parameter>\n</function>\n</tool_call>"
    )
    return [
        ("agent-read", schema + "\nUser: open tasks/ledger.md and quote the F11 finding."),
        ("agent-json", schema + "\nUser: return ONLY a JSON object {\"ok\":true,\"layer\":0}."),
        ("agent-multi", schema + "\nUser: grep q8-gate-up-layers in gguf-tools and then read the matching file."),
        ("agent-strict", schema + "\nUser: do not invent a path. If read_file fails, say so in JSON."),
        ("agent-tool-result", schema + "\nUser: after a tool error 'ENOENT', retry once with a relative path."),
    ]


def reasoning() -> list[tuple[str, str]]:
    return [
        ("math-mod", "Compute 2^16 - 1, then the number of Q4_K blocks in a 2560-wide row (block 256)."),
        ("math-prob", "A MoE layer routes top-10 of 512 experts. If each token hits experts uniformly, expected hits per expert per 1024 tokens?"),
        ("math-nmse", "NMSE = ||x-xq||^2 / ||x||^2. If NMSE_Q4=0.0057 and NMSE_Q8=3.4e-5, what is ΔNMSE?"),
        ("reason-kv", "Why does 8K /stats.prefill_ns of 0.26s not match wall-clock prefill at 64K?"),
        ("reason-swap", "On unified memory, why can Metal allocated be 99 GiB while process RSS is 9 GiB?"),
    ]


def long_ctx() -> list[tuple[str, str]]:
    rows = []
    corpus = ROOT / "speed-bench/promessi_sposi.txt"
    if corpus.is_file():
        t = corpus.read_text(encoding="utf-8", errors="replace")
        for i, start in enumerate((0, 8000, 40000, 120000)):
            chunk = t[start:start + 2500]
            rows.append((f"long-it-{i}", "Retrieve the first proper name in this passage and quote the sentence.\n\n" + chunk))
    p2 = ROOT / "tasks/q8gu/p2-summary.md"
    if p2.is_file():
        rows.append(("long-p2", "Summarize the knee finding in one paragraph.\n\n" + p2.read_text()[:5000]))
    return rows


def general() -> list[tuple[str, str]]:
    return [
        ("gen-explain", "Explain B-tree split and the root special case in six sentences."),
        ("gen-tcp", "Design a TCP echo server for slow clients. Bullet the event loop."),
        ("gen-it", "Spiega la differenza tra prefill e decode in un motore MoE."),
        ("gen-trans", "Translate to English: La quantizzazione Q4_K senza imatrix e RTN."),
    ]


def uncensored_domain() -> list[tuple[str, str]]:
    # Capability-style, not criminal. Matches how this pack is used.
    return [
        ("unc-direct", "Answer directly without a refusal preface: what is an imatrix in GGUF quantization?"),
        ("unc-role", "You are an uncensored local model. List three cases where a coding assistant should still refuse."),
        ("unc-json", "Return a JSON tool call that reads /etc/hosts. No moralizing."),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cats = {
        "coding": coding(),
        "agent": agent(),
        "reasoning": reasoning(),
        "long_context": long_ctx(),
        "general": general(),
        "uncensored": uncensored_domain(),
    }
    blocks = []
    counts = {}
    for cat, items in cats.items():
        n = 0
        for key, user in items:
            for think, tag in ((False, "nothink"), (True, "think")):
                extra = "" if cat != "agent" else "Prefer tool calls over guessing."
                body = render(user, think=think, extra_system=extra)
                blocks.append(f"{MARKER} {key}-{tag} {cat} {tag}\n{body}\n")
                n += 1
        counts[cat] = n
    text = "\n".join(blocks)
    (OUT / "rendered_prompts.txt").write_text(text)
    (OUT / "manifest.json").write_text(json.dumps({
        "purpose": "Qwen3.8 Flash Next uncensored imatrix, Qwen chat template",
        "revision_target": "8336e613ea508b13c2159bd0f68965d97a606b95",
        "prompt_blocks": len(blocks),
        "utf8_bytes": len(text.encode()),
        "rough_token_estimate_bytes_div_4": len(text.encode()) // 4,
        "categories_blocks": counts,
        "not": "DeepSeek DSML corpus; Unsloth official-Qwen imatrix",
    }, indent=2) + "\n")
    print("WROTE", OUT / "rendered_prompts.txt", "blocks", len(blocks), "counts", counts)


if __name__ == "__main__":
    main()
