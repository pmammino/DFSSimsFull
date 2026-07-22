"use client";

import { useEffect, useRef, useState } from "react";
import {
  fetchParamsDefaults,
  fetchSampleSlate,
  fetchShowdownSample,
  postRun,
  uploadSlate,
  type ParamsDefaults,
  type RunParams,
  type RunSummary,
  type SlateSummary,
} from "@/lib/api";
import RefreshPanel from "./RefreshPanel";

// Slider metadata mirrors app.py's Advanced field-model controls exactly.
const SLIDERS: {
  key: keyof RunParams;
  label: string;
  min: number;
  max: number;
  step: number;
}[] = [
  { key: "talent_tilt", label: "Talent tilt (players)", min: 0, max: 2, step: 0.1 },
  { key: "team_tilt", label: "Stack-team tilt", min: 0, max: 2, step: 0.1 },
  { key: "cand_jitter", label: "Diversity jitter", min: 0, max: 1.5, step: 0.1 },
  { key: "stack_boost", label: "Stack-ownership boost", min: 0, max: 0.25, step: 0.01 },
  { key: "stack_aggr", label: "Stack aggressiveness", min: 0, max: 2, step: 0.1 },
  { key: "bringback", label: "Bring-back rate", min: 0, max: 1, step: 0.05 },
  { key: "game_stack", label: "Game-stack rate", min: 0.05, max: 0.95, step: 0.05 },
  { key: "order_tilt", label: "Batting-order tilt", min: 0, max: 0.6, step: 0.05 },
  { key: "ace_pitcher", label: "Ace-pitcher rate", min: 0, max: 1, step: 0.05 },
];

type SlateSource = "sample" | "showdown" | "upload";

export default function SetupTab({
  onRun,
}: {
  onRun: (summary: RunSummary) => void;
}) {
  const [defaults, setDefaults] = useState<ParamsDefaults | null>(null);
  const [params, setParams] = useState<RunParams | null>(null);
  const [source, setSource] = useState<SlateSource>("sample");
  const [slate, setSlate] = useState<SlateSummary | null>(null);
  const [slateBusy, setSlateBusy] = useState(false);
  const [slateErr, setSlateErr] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runErr, setRunErr] = useState<string | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchParamsDefaults()
      .then((d) => {
        setDefaults(d);
        // Sensible defaults for a quick first run (smaller than production).
        setParams({ ...d.defaults, contest_size: 1000, sim_runs: 2000, num_candidates: 500 });
      })
      .catch(() => {
        /* status badge surfaces worker-offline */
      });
  }, []);

  function set<K extends keyof RunParams>(k: K, v: RunParams[K]) {
    setParams((p) => (p ? { ...p, [k]: v } : p));
  }

  async function loadSample(showdown = false) {
    setSlateBusy(true);
    setSlateErr(null);
    try {
      setSlate(await (showdown ? fetchShowdownSample() : fetchSampleSlate()));
    } catch (e) {
      setSlateErr((e as Error).message);
    } finally {
      setSlateBusy(false);
    }
  }

  async function onUpload(file: File) {
    setSlateBusy(true);
    setSlateErr(null);
    try {
      setSlate(await uploadSlate(file));
    } catch (e) {
      setSlateErr((e as Error).message);
    } finally {
      setSlateBusy(false);
    }
  }

  async function run() {
    if (!slate || !params) return;
    setRunning(true);
    setRunErr(null);
    try {
      const summary = await postRun({
        slate_token: slate.slate_token,
        params,
      });
      onRun(summary);
    } catch (e) {
      setRunErr((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  const canRun = !!slate && !!params && !running;

  return (
    <div className="space-y-6">
      {/* 1) Slate */}
      <section className="rounded-card border border-rw-line bg-rw-surface p-5">
        <h2 className="mb-3 text-lg">1 · Slate</h2>
        <div className="mb-4 flex gap-1">
          {(
            [
              ["sample", "Sample (classic)"],
              ["showdown", "Sample (showdown)"],
              ["upload", "Upload CSV"],
            ] as [SlateSource, string][]
          ).map(([s, label]) => (
            <button
              key={s}
              onClick={() => setSource(s)}
              className={
                "rounded-lg border px-3 py-1.5 text-sm " +
                (s === source
                  ? "border-rw-red bg-rw-red text-white"
                  : "border-rw-line bg-rw-raised text-rw-mut hover:text-white")
              }
            >
              {label}
            </button>
          ))}
        </div>

        {source === "sample" || source === "showdown" ? (
          <button
            onClick={() => loadSample(source === "showdown")}
            disabled={slateBusy}
            className="rounded-lg border border-rw-line bg-rw-raised px-4 py-2 text-sm text-white hover:border-rw-red disabled:opacity-50"
          >
            {slateBusy
              ? "Loading…"
              : source === "showdown"
                ? "Load bundled showdown slate"
                : "Load bundled sample slate"}
          </button>
        ) : (
          <div>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUpload(f);
              }}
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={slateBusy}
              className="rounded-lg border border-rw-line bg-rw-raised px-4 py-2 text-sm text-white hover:border-rw-red disabled:opacity-50"
            >
              {slateBusy ? "Parsing…" : "Choose DraftKings CSV…"}
            </button>
            <p className="mt-2 text-xs text-rw-mut">
              Raw DKSalaries export or a clean CSV (FullName, Team, Position,
              Salary, Ownership).
            </p>
          </div>
        )}

        {slateErr && (
          <div className="mt-3 rounded-lg border border-rw-red-700 bg-rw-red-700/20 p-2 text-sm">
            {slateErr}
          </div>
        )}
        {slate && (
          <div className="mt-3 font-mono text-[11px] uppercase tracking-wider text-rw-turf">
            ✓ {slate.n_players} players · {slate.teams} teams ·{" "}
            {slate.has_ownership ? "ownership present" : "no ownership (uniform field)"}
            {slate.format === "showdown" &&
              ` · showdown${slate.matchup ? ` (${slate.matchup})` : ""}`}
          </div>
        )}
      </section>

      {/* 2) Parameters */}
      <section className="rounded-card border border-rw-line bg-rw-surface p-5">
        <h2 className="mb-3 text-lg">2 · Parameters</h2>
        {!params || !defaults ? (
          <div className="text-sm text-rw-mut">Loading defaults…</div>
        ) : (
          <>
            <div className="mb-4">
              <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
                Contest size (field entries)
              </label>
              <div className="flex flex-wrap gap-1">
                {defaults.size_presets.map((n) => (
                  <button
                    key={n}
                    onClick={() => set("contest_size", n)}
                    className={
                      "rounded-lg border px-3 py-1.5 text-sm tabular-nums " +
                      (params.contest_size === n
                        ? "border-rw-red bg-rw-red text-white"
                        : "border-rw-line bg-rw-raised text-rw-mut hover:text-white")
                    }
                  >
                    {n.toLocaleString()}
                  </button>
                ))}
                <input
                  type="number"
                  value={params.contest_size}
                  min={2}
                  onChange={(e) => set("contest_size", Number(e.target.value))}
                  className="w-28 rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-sm text-white outline-none focus:border-rw-red"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <NumField
                label={`Sim runs${defaults.sim_runs_max ? ` (max ${defaults.sim_runs_max.toLocaleString()})` : ""}`}
                value={params.sim_runs}
                min={100}
                max={defaults.sim_runs_max ?? undefined}
                step={500}
                onChange={(v) => set("sim_runs", v)}
              />
              <NumField
                label="Candidate lineups"
                value={params.num_candidates}
                min={10}
                step={100}
                onChange={(v) => set("num_candidates", v)}
              />
              <NumField
                label="Chalk sensitivity"
                value={params.chalk}
                min={0}
                max={2}
                step={0.05}
                onChange={(v) => set("chalk", v)}
              />
            </div>

            <button
              onClick={() => setAdvanced((a) => !a)}
              className="mt-4 text-sm text-rw-red-400 hover:text-white"
            >
              {advanced ? "▾ Hide" : "▸ Show"} advanced field model
            </button>
            {advanced && (
              <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
                {SLIDERS.map((s) => (
                  <Slider
                    key={s.key}
                    label={s.label}
                    value={params[s.key] as number}
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    onChange={(v) => set(s.key, v as RunParams[typeof s.key])}
                  />
                ))}
                <NumField
                  label="Stack-shape tilt (field)"
                  value={params.tilt}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(v) => set("tilt", v)}
                />
                <NumField
                  label="Medium baseline size"
                  value={params.medium}
                  min={100}
                  step={500}
                  onChange={(v) => set("medium", v)}
                />
              </div>
            )}
          </>
        )}
      </section>

      {/* 3) Run */}
      <section className="rounded-card border border-rw-line bg-rw-surface p-5">
        <h2 className="mb-3 text-lg">3 · Run</h2>
        {runErr && (
          <div className="mb-3 rounded-lg border border-rw-red-700 bg-rw-red-700/20 p-2 text-sm">
            {runErr}
          </div>
        )}
        <button
          onClick={run}
          disabled={!canRun}
          className="w-full rounded-lg bg-rw-red px-4 py-3 font-display text-sm uppercase tracking-wide text-white transition-colors hover:bg-rw-red-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {running ? "Simulating…" : "▶ Run simulation"}
        </button>
        {!slate && (
          <p className="mt-2 text-center text-xs text-rw-mut">
            Load or upload a slate first.
          </p>
        )}
      </section>

      <RefreshPanel />
    </div>
  );
}

function NumField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
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
        max={max}
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
