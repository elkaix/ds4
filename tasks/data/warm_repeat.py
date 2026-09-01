#!/usr/bin/env python3
"""Warm-repeat high-context control (roadmap sec.5-6, redesigned).

The originally prescribed design -- run the same decode twice and compare --
is n=1 per arm. Against this machine's 14-25% cross-run drift that can only
detect a Case-1 collapse, and the existing three-pass sweep already argues
against Case 1: between pass0 and pass2, page-ins at 49K fell 437K -> 319K
and at 65K fell 420K -> 220K while throughput went DOWN (26.81 -> 24.23,
26.38 -> 25.30). So the effect to resolve is small, and the informative
object is the trajectory across many repeats, not a two-point difference.

Design: one server process, one fixed prompt, N identical back-to-back
decodes. Prefill is paid once (every later repeat is a cached prefix), so
each repeat differs only in how warm memory is. Deterministic sampling makes
the decoded token stream identical across repeats, so the compute is
literally the same work every time and any change is environmental.

The discriminating instrument is per-class residency, not the global page-in
counter. vm_stat Pageins is machine-wide and cannot say whether DS4 faulted
or Brave did -- exactly the D-vs-A/B/C question. So we also track:

  File-backed pages   the 97 GiB GGUF is mapped SM=SHM and is essentially
                      all of this; a drop here IS model eviction (hyp. B)
  Anonymous pages     KV/KDA/session state growth (hyp. A)
  Compressor pages    system-level pressure (hyp. C)
  phys_footprint      the server's own dirty+GPU-owned bytes

If throughput decays while file-backed pages fall, the model working set is
being displaced and multi-hot-session KV residency would make it worse.
"""
import json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep as S

VM_KEYS = {
    "Pages free": "free", "Pages active": "active", "Pages inactive": "inactive",
    "Pages wired down": "wired", "Pages purgeable": "purgeable",
    "File-backed pages": "filebacked", "Anonymous pages": "anon",
    "Pages occupied by compressor": "compressor",
    "Pageins": "pageins", "Pageouts": "pageouts",
    "Swapins": "swapins", "Swapouts": "swapouts",
    "Pages reactivated": "reactivated",
}
PAGE = 16384

def vmstat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=15).stdout
    d = {}
    for line in out.splitlines():
        if ":" not in line: continue
        k, v = line.split(":", 1)
        k = k.strip().strip('"')
        if k in VM_KEYS:
            try: d[VM_KEYS[k]] = int(v.strip().rstrip("."))
            except ValueError: pass
    return d

def server_pid():
    out = subprocess.run(["pgrep", "-f", "ds4-server --metal"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None

def footprint_mb(pid):
    """phys_footprint counts dirty + GPU-owned bytes. It deliberately excludes
    the file-backed GGUF mapping, which is why it must be read alongside the
    machine-wide File-backed page count rather than instead of it."""
    if pid is None: return None
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True,
                             text=True, timeout=30).stdout
        m = re.search(r"phys_footprint:\s+(\d+)\s+MB", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def main():
    target = int(os.environ.get("WR_TARGET", "65536"))
    reps = int(os.environ.get("WR_REPS", "6"))
    gen = int(os.environ.get("WR_GEN", "256"))
    label = os.environ.get("WR_LABEL", "warm")
    corpus = open(S.CORPUS, encoding="utf-8", errors="replace").read()

    # Calibrate the prompt once, by the same secant probe the sweep uses, so
    # the context is a known token count rather than a chars/token guess.
    lo_c = int(target * float(os.environ.get("WR_CPT", "3.647")))
    r, _, _ = S.chat(S.INSTRUCTION + corpus[:lo_c], max_tokens=1)
    ptok = r["usage"]["prompt_tokens"]
    for _ in range(6):
        if abs(ptok - target) / target <= 0.01: break
        lo_c = int(lo_c * target / ptok)
        r, _, _ = S.chat(S.INSTRUCTION + corpus[:lo_c], max_tokens=1)
        ptok = r["usage"]["prompt_tokens"]
    prompt = S.INSTRUCTION + corpus[:lo_c]
    print(f"# {label}: target {target} -> {lo_c} chars = {ptok} tok", flush=True)

    pid = server_pid()
    out = open(os.path.join(HERE, f"warm-{label}.jsonl"), "w")
    prev_tokens = None
    for i in range(reps):
        # The wrapper's own monitor polls /stats, so `clients` is not a clean
        # count of inference clients. What actually invalidates a repeat is
        # another request occupying the slot, so gate on that and record
        # `clients` for the record rather than skipping on it.
        st = S.stats()
        if st["busy"] or st["queue_depth"]:
            S.wait_for_idle()
            st = S.stats()
        v0, f0, log0 = vmstat(), footprint_mb(pid), S.server_log_size()
        r, t0, t1 = S.chat(prompt, max_tokens=gen)
        gw, gc = S.power_sample()
        v1, f1 = vmstat(), footprint_mb(pid)
        kv_n, kv_ms = S.kv_stores_since(log0)
        after = S.stats()
        comp = r["usage"]["completion_tokens"]
        text = r["choices"][0]["message"]["content"]
        # Fail closed: if the decoded stream is not identical across repeats the
        # repeats are not the same work and the comparison is void.
        ident = None if prev_tokens is None else (text == prev_tokens)
        prev_tokens = text
        wall = t1 - t0
        # KV disk stores run inside the decode loop, so subtract them before
        # calling anything a decode rate.
        corrected = (wall * 1000.0 - (kv_ms or 0.0)) / comp if comp else None
        rec = {"label": label, "rep": i, "ptok": ptok, "clients": st["clients"], "completion": comp,
               "identical_to_prev": ident,
               "wall_s": round(wall, 2),
               "ms_per_tok_corrected": round(corrected, 3) if corrected else None,
               "tps_corrected": round(1000.0 / corrected, 2) if corrected else None,
               "dash_tps": after["last_decode_tps"],
               "kv_stores": kv_n, "kv_store_ms": kv_ms,
               "gpu_w": gw, "gpu_c": gc,
               "footprint_mb_before": f0, "footprint_mb_after": f1,
               "vm_before": v0,
               "d_pageins": v1.get("pageins", 0) - v0.get("pageins", 0),
               "d_swapins": v1.get("swapins", 0) - v0.get("swapins", 0),
               "d_swapouts": v1.get("swapouts", 0) - v0.get("swapouts", 0),
               "d_filebacked_mb": (v1.get("filebacked", 0) - v0.get("filebacked", 0)) * PAGE // 1048576,
               "d_anon_mb": (v1.get("anon", 0) - v0.get("anon", 0)) * PAGE // 1048576,
               "d_compressor_mb": (v1.get("compressor", 0) - v0.get("compressor", 0)) * PAGE // 1048576,
               "filebacked_mb": v1.get("filebacked", 0) * PAGE // 1048576,
               "free_mb": v1.get("free", 0) * PAGE // 1048576,
               "compressor_mb": v1.get("compressor", 0) * PAGE // 1048576}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"  rep{i} {rec['tps_corrected']:6.2f} t/s corr "
              f"({rec['ms_per_tok_corrected']:.2f} ms/tok) wall={wall:6.2f}s "
              f"comp={comp} same={ident} "
              f"pagein={rec['d_pageins']:>8} swapin={rec['d_swapins']} "
              f"dFB={rec['d_filebacked_mb']:>6}MB dANON={rec['d_anon_mb']:>5}MB "
              f"dCOMP={rec['d_compressor_mb']:>5}MB "
              f"FB={rec['filebacked_mb']}MB free={rec['free_mb']}MB "
              f"fp={f1}MB kv={kv_n}/{kv_ms}ms gpu={gw}W/{gc}C", flush=True)
    out.close()

if __name__ == "__main__":
    main()
