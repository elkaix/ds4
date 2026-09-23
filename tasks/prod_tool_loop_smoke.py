#!/usr/bin/env python3
"""Multi-turn tool-loop smoke test for a running ds4-server on :8000.

Replays the full transcript every turn, like an agent client, so the server
must reuse its live KV (memory-token / memory-text / tool continuation) rather
than re-prefill. Prints per-turn usage and decode speed; pair with the server
log to confirm cache source, 0 replayed tokens and no token-mismatch misses.
Usage: tasks/prod_tool_loop_smoke.py [base_url] [turns]
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TOOLS = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a text file from the project and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]
FILES = {
    "README.md": "# demo\nA tiny service. Entry point: src/main.py. Config: config.toml.\n",
    "src/main.py": "import tomllib\ncfg = tomllib.load(open('config.toml','rb'))\nprint(cfg['port'])\n",
    "config.toml": "port = 8123\nworkers = 4\n",
}
QUESTIONS = [
    "Read README.md with the tool and tell me the entry point.",
    "Now read the entry point file and tell me which config key it prints.",
    "Read the config file and tell me the value of that key.",
    "How many workers are configured? Use the tool if you need to.",
]


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def main():
    model = json.loads(urllib.request.urlopen(BASE + "/v1/models").read())["data"][0]["id"]
    msgs = [{"role": "system", "content": "You are a concise coding assistant. Use tools to read files."}]
    calls = 0
    for turn in range(TURNS):
        msgs.append({"role": "user", "content": QUESTIONS[turn % len(QUESTIONS)]})
        for _hop in range(4):
            t0 = time.time()
            out = post("/v1/chat/completions", {"model": model, "messages": msgs,
                       "tools": TOOLS, "max_tokens": 1024, "temperature": 0})
            dt = time.time() - t0
            msg = out["choices"][0]["message"]
            u = out.get("usage", {})
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            print(f"turn {turn + 1} hop {_hop + 1}: finish={out['choices'][0]['finish_reason']} "
                  f"prompt={u.get('prompt_tokens')} cached={cached} "
                  f"completion={u.get('completion_tokens')} wall={dt:.1f}s", flush=True)
            msgs.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            tcs = msg.get("tool_calls") or []
            if not tcs:
                print(f"  answer: {(msg.get('content') or '').strip()[:160]!r}")
                break
            for tc in tcs:
                calls += 1
                path = json.loads(tc["function"]["arguments"] or "{}").get("path", "")
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": FILES.get(path.lstrip("./"), f"error: no such file {path}")})
    stats = json.loads(urllib.request.urlopen(BASE + "/stats").read())
    print("tool_calls:", calls)
    print("stats:", json.dumps({k: stats[k] for k in sorted(stats) if any(
        s in k for s in ("replay", "hit", "miss", "recover", "decode", "prefill", "fresh"))})[:1500])


if __name__ == "__main__":
    main()
