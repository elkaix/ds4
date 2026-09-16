/** Shape of `GET /stats` as emitted by `send_stats()` in ds4_server.c.
 *  Extra nested objects are optional: absent means not instrumented. */

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

export interface HostStats {
  chip?: string;
  cpu_cores?: number;
  gpu_cores?: number;
  gpu_util_pct?: number;
  unified_memory_bytes?: number;
  macos_version?: string;
  memory_pressure?: string;
  swap_bytes?: number;
  swap_delta_bytes?: number;
  pageins_per_s?: number;
  disk_read_bytes_per_s?: number;
  disk_write_bytes_per_s?: number;
  process_cpu_pct?: number;
  thermal_state?: string;
}

export interface MetalStats {
  allocated_bytes?: number;
  recommended_max_bytes?: number;
}

export interface ProcessStats {
  physical_footprint_bytes?: number;
  peak_physical_footprint_bytes?: number;
  rss_bytes?: number;
  pid?: number;
}

export interface KvDiskStats {
  enabled: boolean;
  used_bytes?: number;
  budget_bytes?: number;
  entries?: number;
  last_write_s_ago?: number;
  dir?: string;
}

export interface IdentityStats {
  model_name?: string;
  model_bytes?: number;
  architecture?: string;
  routed_quant_bits?: number;
  backend?: string;
  tensor_route?: string;
  layers?: number;
  slot_count?: number;
  vision?: boolean;
  build_timestamp?: string;
  git_sha?: string;
}

export interface MtpStats {
  state: 'unsupported' | 'disabled' | 'gated' | 'active';
  draft_tokens?: number;
  cycles?: number;
  accepted_tokens?: number;
  drafted_tokens?: number;
  cycle_ns?: number;
  plain_step_ns?: number;
}

export interface LastRequestStats {
  id?: number;
  started_at_s?: number;
  completed_at_s?: number;
  context_before?: number;
  context_after?: number;
  context_capacity?: number;
  logical_prompt_tokens?: number;
  cached_prompt_tokens?: number;
  fresh_prompt_tokens?: number;
  cache_source?: string;
  prompt_wall_s?: number;
  prefill_compute_s?: number;
  decode_s?: number;
  output_tokens?: number;
  first_token_s?: number;
  finish_reason?: string;
}

export interface CheckpointStats {
  saves?: number;
  restores?: number;
  last_write_s?: number;
  last_blocking_s?: number;
}

export interface Stats {
  uptime_s: number;
  busy: boolean;
  queue_depth: number;
  clients: number;
  live_tokens: number;
  live_tokens_peak?: number;
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
  identity?: IdentityStats;
  host?: HostStats;
  metal?: MetalStats;
  process?: ProcessStats;
  kv?: KvDiskStats;
  mtp?: MtpStats;
  last_request?: LastRequestStats;
  checkpoint?: CheckpointStats;
}

export interface RequestRecord {
  requestId: string;
  startedAt: number;
  completedAt: number;
  contextBefore: number;
  contextAfter: number;
  contextCapacity: number;
  logicalPromptTokens: number;
  cachedPromptTokens: number;
  freshPromptTokens: number;
  cacheSource: string;
  promptWallNs?: number;
  prefillComputeNs?: number;
  firstTokenNs?: number;
  outputTokens: number;
  decodeNs: number;
  finishReason?: string;
  mtpState?: MtpStats['state'];
}

/** Cache paths in the order the server counts them, with reader-facing labels. */
export const CACHE_PATHS: readonly { key: keyof CacheStats; label: string }[] = [
  { key: 'memory_token', label: 'memory-token' },
  { key: 'memory_text', label: 'memory-text' },
  { key: 'thinking_visible', label: 'thinking-visible' },
  { key: 'tool_visible', label: 'tool-visible' },
  { key: 'responses_visible', label: 'responses-visible' },
  { key: 'responses_tool_output', label: 'responses-tool-output' },
  { key: 'anthropic_tool_output', label: 'anthropic-tool-output' },
  { key: 'disk_text', label: 'disk-text' },
];

const num = (v: unknown, name: string): number => {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(`/stats: "${name}" is not a number`);
  return v;
};

const optNum = (v: unknown): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) ? v : undefined;

const optStr = (v: unknown): string | undefined =>
  typeof v === 'string' && v.length > 0 ? v : undefined;

const optBool = (v: unknown): boolean | undefined =>
  typeof v === 'boolean' ? v : undefined;

const obj = (v: unknown): Record<string, unknown> | undefined =>
  typeof v === 'object' && v !== null ? v as Record<string, unknown> : undefined;

function put<T, K extends keyof T>(dst: T, key: K, value: T[K] | undefined): void {
  if (value !== undefined) dst[key] = value;
}

function parseHost(raw: unknown): HostStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const h: HostStats = {};
  put(h, 'chip', optStr(o.chip));
  put(h, 'cpu_cores', optNum(o.cpu_cores));
  put(h, 'gpu_cores', optNum(o.gpu_cores));
  put(h, 'gpu_util_pct', optNum(o.gpu_util_pct));
  put(h, 'unified_memory_bytes', optNum(o.unified_memory_bytes));
  put(h, 'macos_version', optStr(o.macos_version));
  put(h, 'memory_pressure', optStr(o.memory_pressure));
  put(h, 'swap_bytes', optNum(o.swap_bytes));
  put(h, 'swap_delta_bytes', optNum(o.swap_delta_bytes));
  put(h, 'pageins_per_s', optNum(o.pageins_per_s));
  put(h, 'disk_read_bytes_per_s', optNum(o.disk_read_bytes_per_s));
  put(h, 'disk_write_bytes_per_s', optNum(o.disk_write_bytes_per_s));
  put(h, 'process_cpu_pct', optNum(o.process_cpu_pct));
  put(h, 'thermal_state', optStr(o.thermal_state));
  return h;
}

function parseMetal(raw: unknown): MetalStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const m: MetalStats = {};
  put(m, 'allocated_bytes', optNum(o.allocated_bytes));
  put(m, 'recommended_max_bytes', optNum(o.recommended_max_bytes));
  return m;
}

function parseProcess(raw: unknown): ProcessStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const p: ProcessStats = {};
  put(p, 'physical_footprint_bytes', optNum(o.physical_footprint_bytes));
  put(p, 'peak_physical_footprint_bytes', optNum(o.peak_physical_footprint_bytes));
  put(p, 'rss_bytes', optNum(o.rss_bytes));
  put(p, 'pid', optNum(o.pid));
  return p;
}

function parseKv(raw: unknown): KvDiskStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const k: KvDiskStats = { enabled: o.enabled === true };
  put(k, 'used_bytes', optNum(o.used_bytes));
  put(k, 'budget_bytes', optNum(o.budget_bytes));
  put(k, 'entries', optNum(o.entries));
  put(k, 'last_write_s_ago', optNum(o.last_write_s_ago));
  put(k, 'dir', optStr(o.dir));
  return k;
}

function parseIdentity(raw: unknown): IdentityStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const i: IdentityStats = {};
  put(i, 'model_name', optStr(o.model_name));
  put(i, 'model_bytes', optNum(o.model_bytes));
  put(i, 'architecture', optStr(o.architecture));
  put(i, 'routed_quant_bits', optNum(o.routed_quant_bits));
  put(i, 'backend', optStr(o.backend));
  put(i, 'tensor_route', optStr(o.tensor_route));
  put(i, 'layers', optNum(o.layers));
  put(i, 'slot_count', optNum(o.slot_count));
  put(i, 'vision', optBool(o.vision));
  put(i, 'build_timestamp', optStr(o.build_timestamp));
  put(i, 'git_sha', optStr(o.git_sha));
  return i;
}

function parseMtp(raw: unknown): MtpStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const state = optStr(o.state);
  if (state !== 'unsupported' && state !== 'disabled' && state !== 'gated' && state !== 'active') {
    return undefined;
  }
  const m: MtpStats = { state };
  put(m, 'draft_tokens', optNum(o.draft_tokens));
  put(m, 'cycles', optNum(o.cycles));
  put(m, 'accepted_tokens', optNum(o.accepted_tokens));
  put(m, 'drafted_tokens', optNum(o.drafted_tokens));
  put(m, 'cycle_ns', optNum(o.cycle_ns));
  put(m, 'plain_step_ns', optNum(o.plain_step_ns));
  return m;
}

function parseLastRequest(raw: unknown): LastRequestStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const r: LastRequestStats = {};
  put(r, 'id', optNum(o.id));
  put(r, 'started_at_s', optNum(o.started_at_s));
  put(r, 'completed_at_s', optNum(o.completed_at_s));
  put(r, 'context_before', optNum(o.context_before));
  put(r, 'context_after', optNum(o.context_after));
  put(r, 'context_capacity', optNum(o.context_capacity));
  put(r, 'logical_prompt_tokens', optNum(o.logical_prompt_tokens));
  put(r, 'cached_prompt_tokens', optNum(o.cached_prompt_tokens));
  put(r, 'fresh_prompt_tokens', optNum(o.fresh_prompt_tokens));
  put(r, 'cache_source', optStr(o.cache_source));
  put(r, 'prompt_wall_s', optNum(o.prompt_wall_s));
  put(r, 'prefill_compute_s', optNum(o.prefill_compute_s));
  put(r, 'decode_s', optNum(o.decode_s));
  put(r, 'output_tokens', optNum(o.output_tokens));
  put(r, 'first_token_s', optNum(o.first_token_s));
  put(r, 'finish_reason', optStr(o.finish_reason));
  return r;
}

function parseCheckpoint(raw: unknown): CheckpointStats | undefined {
  const o = obj(raw);
  if (!o) return undefined;
  const c: CheckpointStats = {};
  put(c, 'saves', optNum(o.saves));
  put(c, 'restores', optNum(o.restores));
  put(c, 'last_write_s', optNum(o.last_write_s));
  put(c, 'last_blocking_s', optNum(o.last_blocking_s));
  return c;
}

/** Narrow an unknown JSON value to `Stats`, failing loudly on a shape drift of required fields. */
export function parseStats(raw: unknown): Stats {
  if (typeof raw !== 'object' || raw === null) throw new TypeError('/stats: not an object');
  const o = raw as Record<string, unknown>;
  const c = o.cache;
  if (typeof c !== 'object' || c === null) throw new TypeError('/stats: "cache" missing');
  const cc = c as Record<string, unknown>;
  const cache = {} as Record<keyof CacheStats, number>;
  for (const k of ['hits', 'cold', ...CACHE_PATHS.map((p) => p.key)] as const) cache[k] = num(cc[k], `cache.${k}`);
  const identity = parseIdentity(o.identity);
  const host = parseHost(o.host);
  const metal = parseMetal(o.metal);
  const process = parseProcess(o.process);
  const kv = parseKv(o.kv);
  const mtp = parseMtp(o.mtp);
  const last_request = parseLastRequest(o.last_request);
  const checkpoint = parseCheckpoint(o.checkpoint);
  const live_tokens_peak = optNum(o.live_tokens_peak);
  const stats: Stats = {
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
  if (live_tokens_peak != null) stats.live_tokens_peak = live_tokens_peak;
  if (identity) stats.identity = identity;
  if (host) stats.host = host;
  if (metal) stats.metal = metal;
  if (process) stats.process = process;
  if (kv) stats.kv = kv;
  if (mtp) stats.mtp = mtp;
  if (last_request) stats.last_request = last_request;
  if (checkpoint) stats.checkpoint = checkpoint;
  return stats;
}

export function parseRequestList(raw: unknown): RequestRecord[] {
  if (typeof raw !== 'object' || raw === null) return [];
  const o = raw as Record<string, unknown>;
  const arr = o.requests;
  if (!Array.isArray(arr)) return [];
  const out: RequestRecord[] = [];
  for (const item of arr) {
    const r = obj(item);
    if (!r) continue;
    const logical = optNum(r.logical_prompt_tokens) ?? 0;
    const cached = optNum(r.cached_prompt_tokens) ?? 0;
    const wallS = optNum(r.prompt_wall_s);
    const computeS = optNum(r.prefill_compute_s);
    const decodeS = optNum(r.decode_s);
    const firstS = optNum(r.first_token_s);
    const rec: RequestRecord = {
      requestId: String(optNum(r.id) ?? out.length),
      startedAt: (optNum(r.started_at_s) ?? 0) * 1000,
      completedAt: (optNum(r.completed_at_s) ?? 0) * 1000,
      contextBefore: optNum(r.context_before) ?? 0,
      contextAfter: optNum(r.context_after) ?? 0,
      contextCapacity: optNum(r.context_capacity) ?? 0,
      logicalPromptTokens: logical,
      cachedPromptTokens: cached,
      freshPromptTokens: optNum(r.fresh_prompt_tokens) ?? Math.max(0, logical - cached),
      cacheSource: optStr(r.cache_source) ?? 'unknown',
      outputTokens: optNum(r.output_tokens) ?? 0,
      decodeNs: decodeS != null ? decodeS * 1e9 : 0,
    };
    if (wallS != null) rec.promptWallNs = wallS * 1e9;
    if (computeS != null) rec.prefillComputeNs = computeS * 1e9;
    if (firstS != null) rec.firstTokenNs = firstS * 1e9;
    const finish = optStr(r.finish_reason);
    if (finish) rec.finishReason = finish;
    const mtpState = optStr(r.mtp_state);
    if (mtpState === 'unsupported' || mtpState === 'disabled' || mtpState === 'gated' || mtpState === 'active') {
      rec.mtpState = mtpState;
    }
    out.push(rec);
  }
  return out;
}

export async function fetchStats(base: string, signal: AbortSignal): Promise<Stats> {
  const r = await fetch(`${base}/stats`, { cache: 'no-store', signal });
  if (!r.ok) throw new Error(`HTTP ${String(r.status)}`);
  return parseStats(await r.json());
}

export async function fetchRequests(base: string, signal: AbortSignal): Promise<RequestRecord[]> {
  const r = await fetch(`${base}/stats/requests`, { cache: 'no-store', signal });
  if (r.status === 404) return [];
  if (!r.ok) throw new Error(`HTTP ${String(r.status)}`);
  return parseRequestList(await r.json());
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
  const base = fromParam ?? (location.protocol.startsWith('http')
    ? location.origin
    : 'http://127.0.0.1:8000');
  return base.replace(/\/$/, '');
}

export function resolveInterval(): number {
  const ms = Number(new URLSearchParams(location.search).get('ms'));
  return Number.isFinite(ms) && ms >= 200 ? ms : 1000;
}
