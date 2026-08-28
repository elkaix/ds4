export interface Slot { id: number; busy: boolean; live_tokens: number; ctx: number }
export interface Stats {
  model: string;
  model_path: string;
  kv_disk: { enabled: boolean; dir: string; used_mb: number; budget_mb: number; files: number };
  uptime_s: number;
  busy: boolean;
  queue_depth: number;
  clients: number;
  live_tokens: number;
  ctx_size: number;
  slot_count: number;
  rss_mb: number;
  requests: number;
  prefill_cancelled: number;
  prompt_tokens: number;
  cached_tokens: number;
  generated_tokens: number;
  last_prefill_tps: number;
  last_decode_tps: number;
  cache: { hits: number; cold: number };
  slots: Slot[];
}
