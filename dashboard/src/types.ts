export interface Slot { id: number; busy: boolean; live_tokens: number; ctx: number }

export type Pressure = "normal" | "warn" | "critical" | "unknown";
export type Thermal = "nominal" | "fair" | "serious" | "critical" | "unknown";

/** Memory / thermal snapshot (instantaneous — never baseline-subtract these). */
export interface Mem {
  footprint_mb: number;
  peak_footprint_mb: number;
  rss_mb: number;
  metal_allocated_mb: number;
  metal_working_set_mb: number;
  swap_used_mb: number;
  pressure: Pressure;
  thermal: Thermal;
}

/** Process counters, cumulative since server start. Consumed as rates. */
export interface Proc {
  cpu_user_ns: number;
  cpu_sys_ns: number;
  disk_read_bytes: number;
  disk_write_bytes: number;
  pageins: number;
}

/** Uniformly cumulative since server start — safe to baseline-subtract wholesale. */
export interface Totals {
  prefill_fresh_tokens: number;
  prefill_compute_ns: number;
  prompt_total_ns: number;
  decode_tokens: number;
  decode_ns: number;
  first_token_ns: number;
  checkpoint_saves: number;
  checkpoint_save_ns: number;
  checkpoint_save_bytes: number;
  checkpoint_restores: number;
  checkpoint_restore_ns: number;
}

/** Speculative-decode (multi-token prediction) state + cumulative cycle counters. */
export interface Mtp {
  /** instantaneous */
  enabled: boolean;
  active: boolean;
  max_ctx: number;
  pos: number;
  /** cumulative */
  cycles: number;
  accepted: number;
  committed: number;
  rows_cycles: number;
  batch_cycles: number;
  setup_ns: number;
  verify_ns: number;
  rollback_ns: number;
  draft_ns: number;
  total_ns: number;
}

/** The cumulative half of Mtp, which the Session baseline subtracts. */
export type MtpCounters = Omit<Mtp, "enabled" | "active" | "max_ctx" | "pos">;

/** One completed request, newest first, at most 64 of them. */
export interface RecentRequest {
  seq: number;
  /** server uptime seconds at the moment the request finished */
  at_s: number;
  kind: string;
  /** context position at prompt start */
  pos: number;
  prompt_tokens: number;
  cached_tokens: number;
  fresh_tokens: number;
  lookup_ns: number;
  restore_ns: number;
  cold_ns: number;
  /** GPU compute time for the fresh tokens (the "sync" prefill) */
  prefill_ns: number;
  store_ns: number;
  /** complete prompt-processing wall time */
  prompt_ns: number;
  /** end of prompt processing → first generated token */
  first_token_ns: number;
  decode_tokens: number;
  decode_ns: number;
  mtp_cycles: number;
  mtp_committed: number;
  source: string;
  finish: string;
}

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
  footprint_mb?: number;
  swap_used_mb?: number;
  tensor_route?: string;
  requests: number;
  prefill_cancelled: number;
  prompt_tokens: number;
  cached_tokens: number;
  generated_tokens: number;
  last_prefill_tps: number;
  last_decode_tps: number;
  cache: { hits: number; cold: number };
  slots: Slot[];
  /** Added by newer servers — always guard with optional chaining. */
  mem?: Mem;
  proc?: Proc;
  totals?: Totals;
  mtp?: Mtp;
  recent?: RecentRequest[];
}
