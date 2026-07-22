"use client";

import { useState } from "react";
import Header from "./Header";
import PlayersTab from "./PlayersTab";
import Placeholder from "./Placeholder";

const TABS = [
  { key: "setup", label: "⚙ Setup" },
  { key: "players", label: "📊 Players" },
  { key: "results", label: "🏆 Results" },
  { key: "export", label: "⬇ Export" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export default function Workspace() {
  const [active, setActive] = useState<TabKey>("players");

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
              {on && (
                <span className="absolute inset-x-0 -bottom-px h-[3px] bg-rw-red" />
              )}
            </button>
          );
        })}
      </nav>

      {active === "players" && <PlayersTab />}
      {active === "setup" && (
        <Placeholder
          title="Setup"
          phase="Phase 1"
          items={[
            "Slate picker (RotoWire feed) + CSV upload",
            "Ownership upload fallback",
            "Team-totals (Vegas) editor",
            "Field-model params form (contest size, sims, 11 tuning sliders)",
            "Run → POST /run · Refresh → async POST /refresh with progress",
          ]}
        />
      )}
      {active === "results" && (
        <Placeholder
          title="Results"
          phase="Phase 2"
          items={[
            "Win% / Top10% / Top100% metrics from the last run",
            "Client-side filters (players, stack shape, team, ownership, salary)",
            "Marked-lineup selection",
            "Finishing-position distribution chart",
          ]}
        />
      )}
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
