"use client";

import { useEffect, useState } from "react";
import {
  fetchExportOptions,
  postExport,
  type ExportOptions,
  type ExportRequest,
  type ExportResult,
  type RunSummary,
} from "@/lib/api";
import ExportReturnChart from "./ExportReturnChart";

const DEFAULTS: ExportRequest = {
  mode: "ranked",
  n_select: 20,
  sort_by: "Top100 Rate",
  hitter_cap: 1,
  pitcher_cap: 1,
  team_cap: 1,
  max_overlap: 1,
  use_value_groups: false,
  entry_fee: 20,
  pct_paid: 0.2,
  rake: 0.15,
  top_heaviness: 0.9,
  risk: "Balanced",
  shortlist: 1000,
};

export default function ExportTab({
  run,
  marked,
}: {
  run: RunSummary | null;
  marked: Set<number>;
}) {
  const [opts, setOpts] = useState<ExportOptions | null>(null);
  const [form, setForm] = useState<ExportRequest>(DEFAULTS);
  const [useMarked, setUseMarked] = useState(false);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchExportOptions().then(setOpts).catch(() => setOpts(null));
  }, []);
  useEffect(() => {
    setResult(null);
  }, [run?.run_id]);

  function set<K extends keyof ExportRequest>(k: K, v: ExportRequest[K]) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function generate() {
    if (!run) return;
    setBusy(true);
    setErr(null);
    try {
      const body: ExportRequest = { ...form };
      if (useMarked) body.candidate_ids = [...marked];
      setResult(await postExport(run.run_id, body));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function downloadCsv() {
    if (!result?.csv) return;
    const blob = new Blob([result.csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `DK_upload_${result.n_written}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  if (!run) {
    return (
      <div className="rounded-card border border-rw-line bg-rw-surface p-6 text-sm text-rw-mut">
        No run yet. Run a simulation on the <span className="text-white">Setup</span>{" "}
        tab first, then build your DraftKings upload here.
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Config */}
      <section className="rounded-card border border-rw-line bg-rw-surface p-5 space-y-4">
        <h2 className="text-lg">Build upload</h2>

        {/* source + count */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
              Which lineups
            </label>
            <div className="flex gap-1">
              <button
                onClick={() => setUseMarked(false)}
                className={btn(!useMarked)}
              >
                Top-ranked
              </button>
              <button
                onClick={() => setUseMarked(true)}
                className={btn(useMarked)}
                disabled={marked.size === 0}
                title={marked.size === 0 ? "Mark lineups on Results first" : ""}
              >
                Marked ({marked.size})
              </button>
            </div>
          </div>
          <Num label="How many lineups" value={form.n_select} min={1}
               onChange={(v) => set("n_select", v)} />
          <div>
            <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
              Selection
            </label>
            <div className="flex gap-1">
              <button onClick={() => set("mode", "ranked")} className={btn(form.mode === "ranked")}>
                Ranked
              </button>
              <button onClick={() => set("mode", "ev")} className={btn(form.mode === "ev")}>
                Portfolio EV
              </button>
            </div>
          </div>
        </div>

        {form.mode === "ranked" ? (
          <div>
            <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
              Rank by
            </label>
            <div className="flex gap-1">
              {(opts?.sort_by ?? ["Top100 Rate"]).map((s) => (
                <button key={s} onClick={() => set("sort_by", s)} className={btn(form.sort_by === s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-rw-line bg-rw-raised/40 p-4">
            <div className="mb-3 font-mono text-[10px] uppercase tracking-widest text-rw-mut">
              Payout structure &amp; risk (field {run.field_n.toLocaleString()})
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Num label="Entry fee $" value={form.entry_fee!} step={1} min={0.25}
                   onChange={(v) => set("entry_fee", v)} />
              <Slider label="% paid" value={form.pct_paid!} min={0.05} max={0.3} step={0.01}
                      onChange={(v) => set("pct_paid", v)} />
              <Slider label="Rake" value={form.rake!} min={0} max={0.3} step={0.01}
                      onChange={(v) => set("rake", v)} />
              <Slider label="Top-heaviness" value={form.top_heaviness!} min={0.3} max={1.5} step={0.1}
                      onChange={(v) => set("top_heaviness", v)} />
            </div>
            <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
                  Risk posture
                </label>
                <select
                  value={form.risk}
                  onChange={(e) => set("risk", e.target.value)}
                  className="w-full rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-sm text-white outline-none focus:border-rw-red"
                >
                  {(opts?.risk_postures ?? ["Balanced"]).map((r) => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
                {opts?.risk_help?.[form.risk ?? ""] && (
                  <p className="mt-1 text-xs text-rw-mut">{opts.risk_help[form.risk!]}</p>
                )}
              </div>
              <Num label="Candidate pool size" value={form.shortlist!} min={50} step={100}
                   onChange={(v) => set("shortlist", v)} />
            </div>
          </div>
        )}

        {/* exposure caps */}
        <div className="rounded-lg border border-rw-line bg-rw-raised/40 p-4">
          <div className="mb-3 font-mono text-[10px] uppercase tracking-widest text-rw-mut">
            Exposure &amp; diversity caps (100% = no cap)
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
            <Pct
              label={run.format === "showdown" ? "Max player %" : "Max hitter %"}
              value={form.hitter_cap!}
              onChange={(v) => set("hitter_cap", v)}
            />
            <Pct
              label={run.format === "showdown" ? "Max captain %" : "Max pitcher %"}
              value={form.pitcher_cap!}
              onChange={(v) => set("pitcher_cap", v)}
            />
            <Pct label="Max stack-team %" value={form.team_cap!} onChange={(v) => set("team_cap", v)} />
            <Pct label="Max lineup overlap" value={form.max_overlap!} onChange={(v) => set("max_overlap", v)} />
          </div>
          <label className="mt-3 flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={form.use_value_groups}
              onChange={(e) => set("use_value_groups", e.target.checked)}
              className="accent-rw-red"
            />
            Spread exposure across near-twin value players
          </label>
        </div>

        {err && (
          <div className="rounded-lg border border-rw-red-700 bg-rw-red-700/20 p-2 text-sm">
            {err}
          </div>
        )}
        <button
          onClick={generate}
          disabled={busy}
          className="w-full rounded-lg bg-rw-red px-4 py-3 font-display text-sm uppercase tracking-wide text-white hover:bg-rw-red-400 disabled:opacity-40"
        >
          {busy ? "Building…" : "⬇ Build DraftKings upload"}
        </button>
      </section>

      {result && <ExportResults result={result} onDownload={downloadCsv} />}
    </div>
  );
}

function ExportResults({
  result,
  onDownload,
}: {
  result: ExportResult;
  onDownload: () => void;
}) {
  return (
    <>
      <section className="rounded-card border border-rw-line bg-rw-surface p-5">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="text-base">
            {result.n_chosen} lineups selected
            {result.mode === "ev" ? " (payout-aware EV)" : " (ranked)"}
          </h3>
          {result.ev && (
            <span className="font-mono text-[11px] uppercase tracking-wider text-rw-mut">
              cost ${result.ev.cost.toLocaleString()} · pool {result.ev.shortlist}
            </span>
          )}
          <div className="ml-auto">
            {result.csv ? (
              <button
                onClick={onDownload}
                className="rounded-lg bg-rw-red px-4 py-2 text-sm text-white hover:bg-rw-red-400"
              >
                ⬇ Download DK upload CSV ({result.n_written})
              </button>
            ) : (
              <span className="text-xs text-rw-mut">CSV needs DK ids</span>
            )}
          </div>
        </div>
        {result.note && (
          <p className="mt-2 rounded-lg border border-rw-line bg-rw-raised/40 p-2 text-xs text-rw-mut">
            {result.note}
          </p>
        )}
      </section>

      {result.ev?.returns && (
        <section className="rounded-card border border-rw-line bg-rw-surface p-5">
          <h3 className="mb-2 text-base">Payout coverage</h3>
          <ExportReturnChart hist={result.ev.returns} />
        </section>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ExposureTable
          title="Player exposure"
          rows={result.player_exposure}
          nameKey="player"
        />
        <ExposureTable
          title="Stack-team exposure"
          rows={result.team_exposure}
          nameKey="team"
        />
      </div>

      <section className="overflow-x-auto rounded-card border border-rw-line">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-rw-raised text-left font-mono text-[10px] uppercase tracking-wider text-rw-mut">
              <th className="px-3 py-2">Stack</th>
              <th className="px-3 py-2">Lineup</th>
              <th className="px-3 py-2 text-right">Salary</th>
            </tr>
          </thead>
          <tbody>
            {result.lineups.map((lu) => (
              <tr key={lu.candidate} className="border-t border-rw-line/60">
                <td className="px-3 py-2 align-top font-mono text-xs text-rw-mut">
                  {lu.stack}
                  <br />#{lu.candidate}
                </td>
                <td className="px-3 py-2">
                  <div className="flex flex-wrap gap-1">
                    {lu.players.map((p, i) => (
                      <span
                        key={i}
                        className="rounded bg-rw-raised px-1.5 py-0.5 text-xs"
                        title={p.slot}
                      >
                        <span className="text-rw-mut">{p.slot}</span> {p.player}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  ${lu.salary.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function ExposureTable({
  title,
  rows,
  nameKey,
}: {
  title: string;
  rows: { lineups: number; exposure: number; team: string; player?: string; pos?: string }[];
  nameKey: "player" | "team";
}) {
  return (
    <div className="rounded-card border border-rw-line bg-rw-surface p-4">
      <h4 className="mb-2 font-display text-sm uppercase tracking-wide">{title}</h4>
      <div className="max-h-72 space-y-1 overflow-auto">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center gap-2 text-sm">
            <div className="w-40 shrink-0 truncate">
              {(r[nameKey] as string) || "—"}
              {nameKey === "player" && r.team && (
                <span className="text-rw-mut"> · {r.team}</span>
              )}
            </div>
            <div className="h-2 grow rounded bg-rw-raised">
              <div
                className="h-2 rounded bg-rw-red"
                style={{ width: `${Math.round(r.exposure * 100)}%` }}
              />
            </div>
            <div className="w-14 shrink-0 text-right tabular-nums text-rw-mut">
              {Math.round(r.exposure * 100)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- small controls ----
const btn = (on: boolean) =>
  "rounded-lg border px-3 py-1.5 text-sm " +
  (on
    ? "border-rw-red bg-rw-red text-white"
    : "border-rw-line bg-rw-raised text-rw-mut hover:text-white disabled:opacity-40");

function Num({
  label,
  value,
  min,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
        {label}
      </label>
      <input
        type="number"
        value={value}
        min={min}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-sm text-white outline-none focus:border-rw-red"
      />
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <label className="font-mono text-[10px] uppercase tracking-widest text-rw-mut">
          {label}
        </label>
        <span className="font-mono text-xs tabular-nums text-white">{value}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-rw-red"
      />
    </div>
  );
}

// A percentage cap slider that maps 0–100% UI to the 0–1 fraction the API wants.
function Pct({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <label className="font-mono text-[10px] uppercase tracking-widest text-rw-mut">
          {label}
        </label>
        <span className="font-mono text-xs tabular-nums text-white">
          {Math.round(value * 100)}%
        </span>
      </div>
      <input
        type="range"
        value={Math.round(value * 100)}
        min={0}
        max={100}
        step={5}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="w-full accent-rw-red"
      />
    </div>
  );
}
