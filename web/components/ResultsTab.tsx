"use client";

import { useState } from "react";
import { fetchPlaceDist, type PlaceDist, type RunSummary } from "@/lib/api";
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

const COLS: Col[] = [
  { key: "Candidate", label: "#" },
  { key: "Stack", label: "Stack" },
  { key: "PrimaryTeam", label: "Team" },
  { key: "Salary", label: "Salary", money: true },
  { key: "OwnSum", label: "Own∑", num: true },
  { key: "Win%", label: "Win%", num: true },
  { key: "Top10%", label: "Top10%", num: true },
  { key: "Top100%", label: "Top100%", num: true },
  { key: "AvgPlace", label: "Avg", num: true },
];

export default function ResultsTab({ run }: { run: RunSummary | null }) {
  const [dist, setDist] = useState<PlaceDist | null>(null);
  const [distFor, setDistFor] = useState<number | null>(null);
  const [distErr, setDistErr] = useState<string | null>(null);

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
  function inspect(candidate: number) {
    setDistFor(candidate);
    setDistErr(null);
    fetchPlaceDist(run!.run_id, candidate)
      .then(setDist)
      .catch((e) => setDistErr(e.message));
  }

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
          ) : distErr ? (
            <div className="text-sm text-rw-mut">{distErr}</div>
          ) : (
            <div className="text-sm text-rw-mut">loading…</div>
          )}
        </div>
      )}

      <div className="overflow-x-auto rounded-card border border-rw-line">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-rw-raised text-left font-mono text-[10px] uppercase tracking-wider text-rw-mut">
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
            {run.results.map((r) => {
              const cand = Number(r["Candidate"]);
              const on = cand === distFor;
              return (
                <tr
                  key={cand}
                  className={
                    "border-t border-rw-line/60 " +
                    (on ? "bg-rw-red/15" : "hover:bg-rw-raised/60")
                  }
                >
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
          </tbody>
        </table>
      </div>
      <p className="text-xs text-rw-mut">
        Showing the top {run.results.length.toLocaleString()} candidates by
        Win/Top10/Top100. Filtering, marking, and export land in Phase 2–3.
      </p>
    </div>
  );
}
