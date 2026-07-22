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
