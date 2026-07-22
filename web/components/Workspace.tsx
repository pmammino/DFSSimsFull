"use client";

import { useState } from "react";
import Header from "./Header";
import PlayersTab from "./PlayersTab";
import SetupTab from "./SetupTab";
import ResultsTab from "./ResultsTab";
import Placeholder from "./Placeholder";
import type { RunSummary } from "@/lib/api";

const TABS = [
  { key: "setup", label: "⚙ Setup" },
  { key: "players", label: "📊 Players" },
  { key: "results", label: "🏆 Results" },
  { key: "export", label: "⬇ Export" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export default function Workspace() {
  const [active, setActive] = useState<TabKey>("setup");
  const [run, setRun] = useState<RunSummary | null>(null);
  // Marked candidate ids live here so the Export tab (Phase 3) can consume the
  // Results tab's selection — the app's st.session_state["picked"].
  const [marked, setMarked] = useState<Set<number>>(new Set());

  function handleRun(summary: RunSummary) {
    setRun(summary);
    setMarked(new Set()); // fresh run clears prior marks (mirrors app.py)
    setActive("results"); // jump to Results as the "done" signal (like the app)
  }

  return (
    <>
      <Header />

      <nav className="mb-5 flex gap-1 border-b border-rw-line">
        {TABS.map((t) => {
          const on = t.key === active;
          return (
            <button
              key={t.key}
              onClick={() => setActive(t.key)}
              className={
                "relative px-4 py-2 font-display text-[13px] uppercase tracking-wide transition-colors " +
                (on ? "text-white" : "text-rw-mut hover:text-white")
              }
            >
              {t.label}
              {t.key === "results" && run && (
                <span className="ml-1 rounded-full bg-rw-red px-1.5 text-[10px] text-white">
                  •
                </span>
              )}
              {on && (
                <span className="absolute inset-x-0 -bottom-px h-[3px] bg-rw-red" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Keep tabs mounted so state (Setup form, last run) survives switching. */}
      <div className={active === "setup" ? "" : "hidden"}>
        <SetupTab onRun={handleRun} />
      </div>
      <div className={active === "players" ? "" : "hidden"}>
        <PlayersTab />
      </div>
      <div className={active === "results" ? "" : "hidden"}>
        <ResultsTab run={run} marked={marked} onMarked={setMarked} />
      </div>
      {active === "export" && (
        <Placeholder
          title="Export"
          phase="Phase 3"
          items={[
            "DraftKings upload CSV (ranked + Portfolio EV)",
            "Payout structure & risk posture",
            "Exposure caps (global + per-player / per-team)",
            "Portfolio diversity controls + exposure breakdown",
          ]}
        />
      )}
    </>
  );
}
