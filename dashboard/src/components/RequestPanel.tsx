import { useState } from 'react';
import type { RequestRecord } from '../api/stats.ts';
import { decodeTokS, freshPrefillTokS, reusePct, waterfallReconcile } from '../lib/metrics.ts';
import { displayNumber } from '../lib/availability.ts';
import { fmtClock, fmtPct, fmtSeconds, fmtTokens } from '../lib/format.ts';

function tokS(r: RequestRecord): number | undefined {
  return decodeTokS(r.outputTokens, r.decodeNs);
}
function prefillS(r: RequestRecord): number | undefined {
  const ns = r.prefillComputeNs ?? r.promptWallNs;
  return ns != null ? freshPrefillTokS(r.freshPromptTokens, ns) : undefined;
}

export function RequestPanel({ requests }: { requests: readonly RequestRecord[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const [cmp, setCmp] = useState<string[]>([]);
  const recent = [...requests].slice(-40).reverse();
  const selected = recent.filter((r) => cmp.includes(r.requestId));
  const active = recent.find((r) => r.requestId === open);

  const toggleCmp = (id: string) => {
    setCmp((cur) => {
      if (cur.includes(id)) return cur.filter((x) => x !== id);
      const second = cur[1];
      if (cur.length >= 2 && second != null) return [second, id];
      return [...cur, id];
    });
  };

  if (recent.length === 0) {
    return <p className="tone-muted">No completed requests this process. History is a bounded ring on the server.</p>;
  }

  return (
    <div>
      <div className="req-wrap">
        <table className="req">
          <thead>
            <tr>
              <th scope="col"> </th>
              <th scope="col">Time</th>
              <th scope="col">Context</th>
              <th scope="col">Fresh</th>
              <th scope="col">Reuse</th>
              <th scope="col">Prompt</th>
              <th scope="col">TTFT</th>
              <th scope="col">Output</th>
              <th scope="col">Decode</th>
              <th scope="col">MTP</th>
              <th scope="col">Finish</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((r) => {
              const reuse = reusePct(r.cachedPromptTokens, r.logicalPromptTokens);
              return (
                <tr
                  key={r.requestId}
                  className={open === r.requestId ? 'is-on' : undefined}
                  onClick={() => { setOpen(r.requestId === open ? null : r.requestId); }}
                >
                  <td>
                    <input
                      type="checkbox"
                      checked={cmp.includes(r.requestId)}
                      onClick={(e) => { e.stopPropagation(); }}
                      onChange={() => { toggleCmp(r.requestId); }}
                      aria-label={`Compare request ${r.requestId}`}
                    />
                  </td>
                  <td>{r.completedAt ? fmtClock(r.completedAt) : 'Not instrumented'}</td>
                  <td>{fmtTokens(r.contextAfter)}</td>
                  <td>{fmtTokens(r.freshPromptTokens)}</td>
                  <td>{fmtPct(reuse, 1)}</td>
                  <td>{r.promptWallNs != null ? fmtSeconds(r.promptWallNs / 1e9) : 'Not instrumented'}</td>
                  <td>{r.firstTokenNs != null ? fmtSeconds(r.firstTokenNs / 1e9) : 'Not instrumented'}</td>
                  <td>{fmtTokens(r.outputTokens)}</td>
                  <td>{displayNumber(tokS(r), (n) => `${n.toFixed(1)} t/s`)}</td>
                  <td>{r.mtpState ?? 'Not instrumented'}</td>
                  <td>{r.finishReason ?? 'Not instrumented'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {active && <RequestDetail r={active} />}
      {selected.length === 2 && selected[0] && selected[1] && (
        <Compare a={selected[0]} b={selected[1]} />
      )}
    </div>
  );
}

function RequestDetail({ r }: { r: RequestRecord }) {
  const totalNs = r.completedAt > r.startedAt ? (r.completedAt - r.startedAt) * 1e6 : 0;
  const stages = [];
  if (r.prefillComputeNs && r.promptWallNs && r.promptWallNs > r.prefillComputeNs) {
    stages.push({ name: 'prefill compute', ns: r.prefillComputeNs });
    stages.push({ name: 'prompt overhead', ns: r.promptWallNs - r.prefillComputeNs });
  } else if (r.promptWallNs && r.promptWallNs > 0) {
    stages.push({ name: 'prompt wall', ns: r.promptWallNs });
  }
  if (r.decodeNs > 0) stages.push({ name: 'decode', ns: r.decodeNs });
  const rec = waterfallReconcile(stages, totalNs > 0 ? totalNs : stages.reduce((s, x) => s + x.ns, 0));
  const denom = Math.max(rec.totalNs, rec.measuredNs, 1);
  const copy = () => {
    void navigator.clipboard.writeText(JSON.stringify(r, null, 2));
  };
  return (
    <div className="drawer">
      <div className="tile__head">
        <h3>Request {r.requestId}</h3>
        <button type="button" onClick={copy}>Copy JSON</button>
      </div>
      <div className="drawer__grid">
        <div>
          <h3>Prompt — last request</h3>
          <p>logical {fmtTokens(r.logicalPromptTokens)}</p>
          <p>cached {fmtTokens(r.cachedPromptTokens)}</p>
          <p>fresh {fmtTokens(r.freshPromptTokens)}</p>
          <p>source {r.cacheSource}</p>
        </div>
        <div>
          <h3>Generation — last request</h3>
          <p>output {fmtTokens(r.outputTokens)}</p>
          <p>decode {displayNumber(tokS(r), (n) => `${n.toFixed(1)} tok/s`)}</p>
          <p>TTFT {r.firstTokenNs != null ? fmtSeconds(r.firstTokenNs / 1e9) : 'Not instrumented'}</p>
          <p>finish {r.finishReason ?? 'Not instrumented'}</p>
        </div>
        <div>
          <h3>Prefill — last request</h3>
          <p>wall {r.promptWallNs != null ? fmtSeconds(r.promptWallNs / 1e9) : 'Not instrumented'}</p>
          <p>compute {r.prefillComputeNs != null ? fmtSeconds(r.prefillComputeNs / 1e9) : 'Not instrumented'}</p>
          <p>fresh tok/s {displayNumber(prefillS(r), (n) => n.toFixed(1))}</p>
        </div>
      </div>
      {stages.length > 0 && (
        <table className="waterfall" style={{ marginTop: 10 }}>
          <tbody>
            {stages.map((st) => (
              <tr key={st.name}>
                <td>{st.name}</td>
                <td className="num">{fmtSeconds(st.ns / 1e9)}</td>
                <td className="num">{((st.ns / denom) * 100).toFixed(1)}%</td>
                <td style={{ width: '40%' }}>
                  <div className="waterfall__bar" aria-hidden="true">
                    <i style={{ width: `${String((st.ns / denom) * 100)}%` }} />
                  </div>
                </td>
              </tr>
            ))}
            {rec.unclassifiedNs > 1e6 && (
              <tr>
                <td>unclassified</td>
                <td className="num">{fmtSeconds(rec.unclassifiedNs / 1e9)}</td>
                <td className="num">{((rec.unclassifiedNs / denom) * 100).toFixed(1)}%</td>
                <td />
              </tr>
            )}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Compare({ a, b }: { a: RequestRecord; b: RequestRecord }) {
  const row = (k: string, av: string, bv: string) => (
    <tr><th scope="row">{k}</th><td>{av}</td><td>{bv}</td></tr>
  );
  return (
    <div className="drawer">
      <h3>Compare</h3>
      <table className="compare">
        <thead><tr><th> </th><th>A {a.requestId}</th><th>B {b.requestId}</th></tr></thead>
        <tbody>
          {row('Context', fmtTokens(a.contextAfter), fmtTokens(b.contextAfter))}
          {row('Fresh', fmtTokens(a.freshPromptTokens), fmtTokens(b.freshPromptTokens))}
          {row('Reuse', fmtPct(reusePct(a.cachedPromptTokens, a.logicalPromptTokens), 1), fmtPct(reusePct(b.cachedPromptTokens, b.logicalPromptTokens), 1))}
          {row('Decode', displayNumber(tokS(a), (n) => n.toFixed(1)), displayNumber(tokS(b), (n) => n.toFixed(1)))}
          {row('Prefill', displayNumber(prefillS(a), (n) => n.toFixed(1)), displayNumber(prefillS(b), (n) => n.toFixed(1)))}
          {row('TTFT', a.firstTokenNs != null ? fmtSeconds(a.firstTokenNs / 1e9) : 'Not instrumented', b.firstTokenNs != null ? fmtSeconds(b.firstTokenNs / 1e9) : 'Not instrumented')}
          {row('Cache', a.cacheSource, b.cacheSource)}
        </tbody>
      </table>
    </div>
  );
}
