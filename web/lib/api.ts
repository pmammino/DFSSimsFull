// Client-side typed API. Talks to THIS app's /api proxy routes (same origin),
// which forward to the worker. Types mirror the worker's pydantic models in
// service/main.py — in a later phase these can be auto-generated from the
// worker's OpenAPI schema instead of hand-kept.

export interface PlayerRow {
  Player: string;
  Type: "Hitter" | "Pitcher";
  Proj: number;
  "Floor (p10)": number;
  Median: number;
  "Ceiling (p90)": number;
  p99: number;
  Min: number;
  Max: number;
  Std: number;
  "Bust% (<=0)": number;
  "2x%": number;
  "30+%": number;
}

export interface PlayersResponse {
  count: number;
  players: PlayerRow[];
}

export interface DistBin {
  x: number;
  count: number;
}

export interface PlayerDist {
  player: string;
  n_sim: number;
  mean: number;
  p10: number;
  median: number;
  p90: number;
  bins: DistBin[];
}

export interface WorkerStatus {
  n_sim: number | null;
  hitters: number;
  pitchers: number;
  remote_store: boolean;
  build_stamp: Record<string, unknown>;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function fetchStatus(): Promise<WorkerStatus> {
  return getJson<WorkerStatus>("/api/status");
}

export function fetchPlayers(
  kind: "all" | "hitters" | "pitchers",
  search: string,
): Promise<PlayersResponse> {
  const q = new URLSearchParams({ kind, search });
  return getJson<PlayersResponse>(`/api/players?${q.toString()}`);
}

export function fetchPlayerDist(
  name: string,
  nbins = 40,
): Promise<PlayerDist> {
  const q = new URLSearchParams({ nbins: String(nbins) });
  return getJson<PlayerDist>(
    `/api/players/${encodeURIComponent(name)}/distribution?${q.toString()}`,
  );
}

// ---- Setup / Run (Phase 1) ----

export interface RunParams {
  contest_size: number;
  sim_runs: number;
  num_candidates: number;
  medium: number;
  chalk: number;
  tilt: number;
  seed_field: number;
  seed_cand: number;
  talent_tilt: number;
  team_tilt: number;
  cand_jitter: number;
  stack_boost: number;
  stack_aggr: number;
  bringback: number;
  game_stack: number;
  order_tilt: number;
  ace_pitcher: number;
}

export interface ParamsDefaults {
  defaults: RunParams;
  size_presets: number[];
  sim_runs_max: number | null;
}

export interface SlateSummary {
  slate_token: string;
  n_players: number;
  teams: number;
  has_ownership: boolean;
  preview: Record<string, unknown>[];
}

export interface RunMetrics {
  best_win_pct: number;
  best_top10_pct: number;
  best_top100_pct: number;
  candidates_with_a_win: number;
}

export interface RunSummary {
  run_id: string;
  K: number;
  contest_size: number;
  field_n: number;
  field_short?: boolean;
  beta: number;
  n_candidates: number;
  elapsed_s?: number;
  pool?: {
    hitters: number;
    starters: number;
    teams: number;
    matched: number;
    slate_players: number;
  };
  metrics: RunMetrics;
  results: Record<string, number | string>[];
  columns: string[];
  log?: string[];
}

export interface PlaceDistBin {
  lo: number;
  hi: number;
  pct: number;
  sims: number;
}

export interface PlaceDist {
  run_id: string;
  candidate: number;
  field_n: number;
  n_sim: number;
  mean_place: number;
  best_place: number;
  worst_place: number;
  bins: PlaceDistBin[];
}

export function fetchParamsDefaults(): Promise<ParamsDefaults> {
  return getJson<ParamsDefaults>("/api/params/defaults");
}

export function fetchSampleSlate(): Promise<SlateSummary> {
  return getJson<SlateSummary>("/api/slate/sample");
}

export async function uploadSlate(file: File): Promise<SlateSummary> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/slate/upload", { method: "POST", body: form });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<SlateSummary>;
}

export async function postRun(body: {
  slate_token?: string;
  slate_id?: string;
  params: Partial<RunParams>;
}): Promise<RunSummary> {
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<RunSummary>;
}

export function fetchPlaceDist(
  runId: string,
  candidate: number,
): Promise<PlaceDist> {
  return getJson<PlaceDist>(
    `/api/run/${encodeURIComponent(runId)}/candidate/${candidate}/place-distribution`,
  );
}
