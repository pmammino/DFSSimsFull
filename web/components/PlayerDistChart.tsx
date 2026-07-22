"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PlayerDist } from "@/lib/api";

// Histogram of one player's simulated DK scores — the JS analogue of
// app.py:player_score_chart. The worker pre-bins, so this just renders.
export default function PlayerDistChart({ dist }: { dist: PlayerDist }) {
  return (
    <div className="rounded-card border border-rw-line bg-rw-surface p-4">
      <div className="mb-3 flex flex-wrap items-baseline gap-x-5 gap-y-1">
        <h3 className="text-base">{dist.player}</h3>
        <span className="font-mono text-[11px] uppercase tracking-wider text-rw-mut">
          {dist.n_sim.toLocaleString()} sims · mean {dist.mean} · p10 {dist.p10}{" "}
          · median {dist.median} · p90 {dist.p90}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={dist.bins} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="#1c4a7a" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="x"
            tick={{ fill: "#8ba0ba", fontSize: 11 }}
            stroke="#1c4a7a"
          />
          <YAxis tick={{ fill: "#8ba0ba", fontSize: 11 }} stroke="#1c4a7a" />
          <Tooltip
            cursor={{ fill: "rgba(242,46,69,0.12)" }}
            contentStyle={{
              background: "#001428",
              border: "1px solid #1c4a7a",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
            labelFormatter={(v) => `${v} DK pts`}
            formatter={(v: number) => [v.toLocaleString(), "sims"]}
          />
          <ReferenceLine x={dist.mean} stroke="#f22e45" strokeWidth={2} />
          <Bar dataKey="count" fill="#083363" stroke="#1c4a7a" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
