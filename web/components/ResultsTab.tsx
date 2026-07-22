"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  candidatesCsvUrl,
  fetchFacets,
  fetchFilteredResults,
  fetchPlaceDist,
  fieldCsvUrl,
  type FilteredResults,
  type PlaceDist,
  type ResultsFilter,
  type RunFacets,
  type RunSummary,
} from "@/lib/api";
import MultiSelect from "./MultiSelect";
import PlaceDistChart from "./PlaceDistChart";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-card border border-rw-line bg-rw-surface px-4 py-3">
      <div className="font-mono text-[10px] uppercase tracking-widest text-rw-mut">
        {label}
      </div>
      <div className="mt-1 font-display text-2xl text-white">{value}</div>
    </div>
  );
}

interface Col {
  key: string;
  label: string;
  num?: boolean;
  money?: boolean;
}
const CLASSIC_COLS: Col[] = [
  { key: "Rank", label: "#" },
  { key: "Stack", label: "Stack" },
  { key: "PrimaryTeam", label: "Team" },
  { key: "Salary", label: "Salary", money: true },
  { key: "OwnSum", label: "Own∑", num: true },
  { key: "Win%", label: "Win%", num: true },
  { key: "Top10%", label: "Top10%", num: true },
  { key: "Top100%", label: "Top100%", num: true },
  { key: "AvgPlace", label: "Avg", num: true },
];
const SHOWDOWN_COLS: Col[] = [
  { key: "Rank", label: "#" },
  { key: "Captain", label: "Captain" },
  { key: "CptTeam", label: "CPT tm" },
  { key: "Split", label: "Split" },
  { key: "Salary", label: "Salary", money: true },
  { key: "OwnSum", label: "Own∑", num: true },
  { key: "Win%", label: "Win%", num: true },
  { key: "Top10%", label: "Top10%", num: true },
  { key: "Top100%", label: "Top100%", num: true },
  { key: "AvgPlace", label: "Avg", num: true },
];

const EMPTY: ResultsFilter = { match_mode: "all", limit: 5000 };

export default function ResultsTab({
  run,
  marked,
  onMarked,
}: {
  run: RunSummary | null;
  marked: Set<number>;
  onMarked: (next: Set<number>) => void;
}) {
  const [facets, setFacets] = useState<RunFacets | null>(null);
  const [filter, setFilter] = useState<ResultsFilter>(EMPTY);
  const [data, setData] = useState<FilteredResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);

  const [dist, setDist] = useState<PlaceDist | null>(null);
  const [distFor, setDistFor] = useState<number | null>(null);

  const runId = run?.run_id ?? null;

  useEffect(() => {
    setFilter(EMPTY);
    setData(null);
    setDistFor(null);
    setDist(null);
    if (!runId) return;
    fetchFacets(runId).then(setFacets).catch(() => setFacets(null));
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    const id = setTimeout(() => {
      setLoading(true);
      setErr(null);
      fetchFilteredResults(runId, filter)
        .then(setData)
        .catch((e) => setErr(e.message))
        .finally(() => setLoading(false));
    }, 200);
    return () => clearTimeout(id);
  }, [runId, filter]);

  const set = useCallback(
    <K extends keyof ResultsFilter>(k: K, v: ResultsFilter[K]) =>
      setFilter((f) => ({ ...f, [k]: v })),
    [],
  );

  function inspect(candidate: number) {
    if (!runId) return;
    setDistFor(candidate);
    setDist(null);
    fetchPlaceDist(runId, candidate).then(setDist).catch(() => setDist(null));
  }

  function toggleMark(candidate: number) {
    const next = new Set(marked);
    if (next.has(candidate)) next.delete(candidate);
    else next.add(candidate);
    onMarked(next);
  }
  function markAll() {
    if (!data) return;
    onMarked(new Set([...marked, ...data.all_ids]));
  }

  function downloadFiltered() {
    if (!data) return;
    const cols = data.columns;
    const head = cols.join(",");
    const body = data.results
      .map((r) => cols.map((c) => JSON.stringify(r[c] ?? "")).join(","))
      .join("\n");
    const blob = new Blob([head + "\n" + body], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `filtered_results_${run?.contest_size ?? ""}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const activeFilters = useMemo(() => {
    let n = 0;
    if (filter.players?.length) n++;
    if (filter.exclude?.length) n++;
    if (filter.stacks?.length) n++;
    if (filter.teams?.length) n++;
    if (filter.sizes?.length) n++;
    if (filter.captains?.length) n++;
    if (filter.splits?.length) n++;
    if (filter.own_min != null || filter.own_max != null) n++;
    if (filter.sal_min != null || filter.sal_max != null) n++;
    if (filter.min_win || filter.min_top10 || filter.min_top100) n++;
    return n;
  }, [filter]);

  if (!run) {
    return (
      <div className="rounded-card border border-rw-line bg-rw-surface p-6 text-sm text-rw-mut">
        No run yet. Configure a slate and parameters on the{" "}
        <span className="text-white">Setup</span> tab, then hit{" "}
        <span className="text-white">Run simulation</span>.
      </div>
    );
  }

  const m = run.metrics;
  const showdown = run.format === "showdown";
  const COLS = showdown ? SHOWDOWN_COLS : CLASSIC_COLS;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Best Win%" value={`${m.best_win_pct}%`} />
        <Metric label="Best Top10%" value={`${m.best_top10_pct}%`} />
        <Metric label="Best Top100%" value={`${m.best_top100_pct}%`} />
        <Metric label="Cands w/ a win" value={m.candidates_with_a_win.toLocaleString()} />
      </div>

      <div className="font-mono text-[11px] uppercase tracking-wider text-rw-mut">
        {run.n_candidates.toLocaleString()} candidates · field{" "}
        {run.field_n.toLocaleString()}
        {run.field_short ? " (pool-constrained)" : ""} · {run.K.toLocaleString()} sims
        · β {run.beta}
        {run.elapsed_s ? ` · ${run.elapsed_s}s` : ""}
      </div>

      {/* Filters */}
      <div className="rounded-card border border-rw-line bg-rw-surface">
        <button
          onClick={() => setShowFilters((s) => !s)}
          className="flex w-full items-center justify-between px-4 py-3 text-sm"
        >
          <span className="font-display uppercase tracking-wide">
            🔎 Filter &amp; search
            {activeFilters > 0 && (
              <span className="ml-2 rounded-full bg-rw-red px-2 py-0.5 text-[11px]">
                {activeFilters}
              </span>
            )}
          </span>
          <span className="text-rw-mut">{showFilters ? "▾" : "▸"}</span>
        </button>
        {showFilters && facets && (
          <div className="space-y-4 border-t border-rw-line p-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <MultiSelect
                label="Must include player(s)"
                options={facets.pool_players}
                selected={filter.players ?? []}
                onChange={(v) => set("players", v)}
              />
              <MultiSelect
                label="Exclude player(s)"
                options={facets.pool_players}
                selected={filter.exclude ?? []}
                onChange={(v) => set("exclude", v)}
              />
              <div>
                <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
                  Player match
                </label>
                <div className="flex gap-1">
                  {(["all", "any"] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => set("match_mode", mode)}
                      className={
                        "rounded-lg border px-3 py-1.5 text-sm capitalize " +
                        ((filter.match_mode ?? "all") === mode
                          ? "border-rw-red bg-rw-red text-white"
                          : "border-rw-line bg-rw-raised text-rw-mut hover:text-white")
                      }
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {showdown ? (
                <>
                  <MultiSelect
                    label="Captain is"
                    options={facets.captains ?? []}
                    selected={filter.captains ?? []}
                    onChange={(v) => set("captains", v)}
                  />
                  <MultiSelect
                    label="Team split"
                    options={facets.splits ?? []}
                    selected={filter.splits ?? []}
                    onChange={(v) => set("splits", v)}
                  />
                </>
              ) : (
                <>
                  <MultiSelect
                    label="Stack shape"
                    options={facets.stacks ?? []}
                    selected={filter.stacks ?? []}
                    onChange={(v) => set("stacks", v)}
                  />
                  <MultiSelect
                    label="Primary stack team"
                    options={facets.teams ?? []}
                    selected={filter.teams ?? []}
                    onChange={(v) => set("teams", v)}
                  />
                  <MultiSelect
                    label="Primary stack size"
                    options={(facets.sizes ?? []).map(String)}
                    selected={(filter.sizes ?? []).map(String)}
                    onChange={(v) => set("sizes", v.map(Number))}
                  />
                </>
              )}
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Range
                label="Own∑ min/max"
                lo={filter.own_min ?? null}
                hi={filter.own_max ?? null}
                onLo={(v) => set("own_min", v)}
                onHi={(v) => set("own_max", v)}
              />
              <Range
                label="Salary min/max"
                lo={filter.sal_min ?? null}
                hi={filter.sal_max ?? null}
                onLo={(v) => set("sal_min", v)}
                onHi={(v) => set("sal_max", v)}
              />
              <NumMin label="Min Win%" onChange={(v) => set("min_win", v)} />
              <NumMin label="Min Top100%" onChange={(v) => set("min_top100", v)} />
            </div>
            <button
              onClick={() => setFilter(EMPTY)}
              className="text-xs text-rw-red-400 hover:text-white"
            >
              Reset filters
            </button>
          </div>
        )}
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="font-mono text-[11px] uppercase tracking-wider text-rw-mut">
          {loading
            ? "filtering…"
            : data
              ? `${data.total.toLocaleString()} of ${run.n_candidates.toLocaleString()} match`
              : ""}
        </span>
        <button
          onClick={markAll}
          className="rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 hover:border-rw-red"
        >
          Mark all shown
        </button>
        <button
          onClick={() => onMarked(new Set())}
          className="rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-rw-mut hover:text-white"
        >
          Clear marks
        </button>
        <span className="font-mono text-[11px] uppercase tracking-wider text-rw-turf">
          ☑ {marked.size.toLocaleString()} marked
        </span>
        <div className="ml-auto flex gap-2">
          <button
            onClick={downloadFiltered}
            className="rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-xs hover:border-rw-red"
          >
            ⬇ Filtered CSV
          </button>
          <a
            href={candidatesCsvUrl(run.run_id)}
            className="rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-xs hover:border-rw-red"
          >
            ⬇ All candidates
          </a>
          <a
            href={fieldCsvUrl(run.run_id)}
            className="rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-xs hover:border-rw-red"
          >
            ⬇ Field
          </a>
        </div>
      </div>

      {err && (
        <div className="rounded-card border border-rw-red-700 bg-rw-red-700/20 p-3 text-sm">
          {err}
        </div>
      )}

      {distFor != null && (
        <div className="rounded-card border border-rw-line bg-rw-surface p-4">
          <div className="mb-1 flex items-center justify-between">
            <h3 className="text-base">Finishing place — candidate #{distFor}</h3>
            <button
              onClick={() => {
                setDistFor(null);
                setDist(null);
              }}
              className="text-xs text-rw-mut hover:text-white"
            >
              ✕ close
            </button>
          </div>
          {dist && dist.candidate === distFor ? (
            <PlaceDistChart dist={dist} />
          ) : (
            <div className="text-sm text-rw-mut">loading…</div>
          )}
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded-card border border-rw-line">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-rw-raised text-left font-mono text-[10px] uppercase tracking-wider text-rw-mut">
              <th className="px-3 py-2">✓</th>
              {COLS.map((c) => (
                <th
                  key={c.key}
                  className={"px-3 py-2 " + (c.num || c.money ? "text-right" : "")}
                >
                  {c.label}
                </th>
              ))}
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {(data?.results ?? []).map((r) => {
              const cand = Number(r["Candidate"]);
              const on = cand === distFor;
              const isMarked = marked.has(cand);
              return (
                <tr
                  key={cand}
                  className={
                    "border-t border-rw-line/60 " +
                    (on ? "bg-rw-red/15" : "hover:bg-rw-raised/60")
                  }
                >
                  <td className="px-3 py-1.5">
                    <input
                      type="checkbox"
                      checked={isMarked}
                      onChange={() => toggleMark(cand)}
                      className="accent-rw-red"
                    />
                  </td>
                  {COLS.map((c) => {
                    const v = r[c.key];
                    const text = c.money
                      ? `$${Number(v).toLocaleString()}`
                      : String(v);
                    return (
                      <td
                        key={c.key}
                        className={
                          "px-3 py-1.5 " +
                          (c.num || c.money ? "text-right tabular-nums" : "")
                        }
                      >
                        {text}
                      </td>
                    );
                  })}
                  <td className="px-3 py-1.5 text-right">
                    <button
                      onClick={() => inspect(cand)}
                      className="rounded border border-rw-line px-2 py-0.5 text-xs text-rw-mut hover:border-rw-red hover:text-white"
                    >
                      distribution
                    </button>
                  </td>
                </tr>
              );
            })}
            {!loading && data && data.results.length === 0 && (
              <tr>
                <td
                  colSpan={COLS.length + 2}
                  className="px-3 py-6 text-center text-rw-mut"
                >
                  No lineups match these filters — loosen them.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {data && data.total > data.count && (
        <p className="text-xs text-rw-mut">
          Showing the first {data.count.toLocaleString()} of{" "}
          {data.total.toLocaleString()} matches (ranked). Tighten filters to
          narrow; “Mark all shown” marks every match.
        </p>
      )}
    </div>
  );
}

function Range({
  label,
  lo,
  hi,
  onLo,
  onHi,
}: {
  label: string;
  lo: number | null;
  hi: number | null;
  onLo: (v: number | null) => void;
  onHi: (v: number | null) => void;
}) {
  const parse = (s: string) => (s === "" ? null : Number(s));
  return (
    <div>
      <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
        {label}
      </label>
      <div className="flex gap-1">
        <input
          type="number"
          value={lo ?? ""}
          placeholder="min"
          onChange={(e) => onLo(parse(e.target.value))}
          className="w-full rounded-lg border border-rw-line bg-rw-raised px-2 py-1.5 text-sm text-white outline-none focus:border-rw-red"
        />
        <input
          type="number"
          value={hi ?? ""}
          placeholder="max"
          onChange={(e) => onHi(parse(e.target.value))}
          className="w-full rounded-lg border border-rw-line bg-rw-raised px-2 py-1.5 text-sm text-white outline-none focus:border-rw-red"
        />
      </div>
    </div>
  );
}

function NumMin({
  label,
  onChange,
}: {
  label: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
        {label}
      </label>
      <input
        type="number"
        min={0}
        step={0.1}
        placeholder="0"
        onChange={(e) => onChange(Number(e.target.value) || 0)}
        className="w-full rounded-lg border border-rw-line bg-rw-raised px-2 py-1.5 text-sm text-white outline-none focus:border-rw-red"
      />
    </div>
  );
}
