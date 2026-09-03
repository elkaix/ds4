/** Shape of `GET /stats` as emitted by `send_stats()` in ds4_server.c. */
export interface CacheStats {
  hits: number;
  cold: number;
  memory_token: number;
  memory_text: number;
  responses_visible: number;
  responses_tool_output: number;
  anthropic_tool_output: number;
  thinking_visible: number;
  tool_visible: number;
  disk_text: number;
}

export interface Stats {
  uptime_s: number;
  busy: boolean;
  queue_depth: number;
  clients: number;
  live_tokens: number;
  ctx_size: number;
  requests: number;
  queue_rejected: number;
  queue_dropped_disconnected: number;
  prefill_cancelled: number;
  prompt_tokens: number;
  cached_tokens: number;
  generated_tokens: number;
  last_prefill_tps: number;
  last_decode_tps: number;
  cache: CacheStats;
}

/** Cache paths in the order the server counts them, with reader-facing labels. */
export const CACHE_PATHS: readonly { key: keyof CacheStats; label: string }[] = [
  { key: 'memory_token', label: 'Memory · token' },
  { key: 'memory_text', label: 'Memory · text' },
  { key: 'thinking_visible', label: 'Thinking visible' },
  { key: 'tool_visible', label: 'Tool visible' },
  { key: 'responses_visible', label: 'Responses visible' },
  { key: 'responses_tool_output', label: 'Responses tool output' },
  { key: 'anthropic_tool_output', label: 'Anthropic tool output' },
  { key: 'disk_text', label: 'Disk · text' },
];

const num = (v: unknown, name: string): number => {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(`/stats: "${name}" is not a number`);
  return v;
};

/** Narrow an unknown JSON value to `Stats`, failing loudly on a shape drift. */
export function parseStats(raw: unknown): Stats {
  if (typeof raw !== 'object' || raw === null) throw new TypeError('/stats: not an object');
  const o = raw as Record<string, unknown>;
  const c = o.cache;
  if (typeof c !== 'object' || c === null) throw new TypeError('/stats: "cache" missing');
  const cc = c as Record<string, unknown>;
  const cache = {} as Record<keyof CacheStats, number>;
  for (const k of ['hits', 'cold', ...CACHE_PATHS.map((p) => p.key)] as const) cache[k] = num(cc[k], `cache.${k}`);
  return {
    uptime_s: num(o.uptime_s, 'uptime_s'),
    busy: o.busy === true,
    queue_depth: num(o.queue_depth, 'queue_depth'),
    clients: num(o.clients, 'clients'),
    live_tokens: num(o.live_tokens, 'live_tokens'),
    ctx_size: num(o.ctx_size, 'ctx_size'),
    requests: num(o.requests, 'requests'),
    queue_rejected: num(o.queue_rejected, 'queue_rejected'),
    queue_dropped_disconnected: num(o.queue_dropped_disconnected, 'queue_dropped_disconnected'),
    prefill_cancelled: num(o.prefill_cancelled, 'prefill_cancelled'),
    prompt_tokens: num(o.prompt_tokens, 'prompt_tokens'),
    cached_tokens: num(o.cached_tokens, 'cached_tokens'),
    generated_tokens: num(o.generated_tokens, 'generated_tokens'),
    last_prefill_tps: num(o.last_prefill_tps, 'last_prefill_tps'),
    last_decode_tps: num(o.last_decode_tps, 'last_decode_tps'),
    cache,
  };
}

export async function fetchStats(base: string, signal: AbortSignal): Promise<Stats> {
  const r = await fetch(`${base}/stats`, { cache: 'no-store', signal });
  if (!r.ok) throw new Error(`HTTP ${String(r.status)}`);
  return parseStats(await r.json());
}

export async function fetchModelIds(base: string, signal: AbortSignal): Promise<string[]> {
  const r = await fetch(`${base}/v1/models`, { signal });
  if (!r.ok) throw new Error(`HTTP ${String(r.status)}`);
  const j = (await r.json()) as { data?: { id?: unknown }[] };
  return (j.data ?? []).map((m) => m.id).filter((id): id is string => typeof id === 'string');
}

/**
 * Where to poll. Served from ds4-server at /dashboard the page is same-origin,
 * so no --cors is needed. Opened as a file or from the dev server it falls back
 * to the daily port, which then needs --cors. `?url=` overrides both.
 */
export function resolveBase(): string {
  const p = new URLSearchParams(location.search);
  const fromParam = p.get('url');
  const base = fromParam ?? (location.protocol.startsWith('http') && !location.port.startsWith('517')
    ? location.origin
    : 'http://127.0.0.1:8000');
  return base.replace(/\/$/, '');
}

export function resolveInterval(): number {
  const ms = Number(new URLSearchParams(location.search).get('ms'));
  return Number.isFinite(ms) && ms >= 200 ? ms : 1000;
}
