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
import type { PlaceDist } from "@/lib/api";

// Finishing-place histogram for one candidate — the JS analogue of
// app.py:place_distribution_chart. The worker pre-bins into place ranges.
export default function PlaceDistChart({ dist }: { dist: PlaceDist }) {
  const data = dist.bins.map((b) => ({ place: b.lo, pct: b.pct, sims: b.sims }));
  return (
    <div>
      <div className="mb-2 font-mono text-[11px] uppercase tracking-wider text-rw-mut">
        Best {dist.best_place.toLocaleString()} · mean{" "}
        {Math.round(dist.mean_place).toLocaleString()} · worst{" "}
        {dist.worst_place.toLocaleString()} of {dist.field_n.toLocaleString()}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
          <CartesianGrid stroke="#1c4a7a" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="place"
            type="number"
            domain={[1, dist.field_n]}
            tick={{ fill: "#8ba0ba", fontSize: 11 }}
            stroke="#1c4a7a"
          />
          <YAxis
            tick={{ fill: "#8ba0ba", fontSize: 11 }}
            stroke="#1c4a7a"
            unit="%"
          />
          <Tooltip
            cursor={{ fill: "rgba(242,46,69,0.12)" }}
            contentStyle={{
              background: "#001428",
              border: "1px solid #1c4a7a",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
            labelFormatter={(v) => `place ≥ ${Number(v).toLocaleString()}`}
            formatter={(v: number, n) =>
              n === "pct" ? [`${v}%`, "% of sims"] : [v, n]
            }
          />
          {/* 1st / Top-10 / Top-100 markers, like the Streamlit chart */}
          {[
            { x: 1, c: "#ffffff" },
            { x: 10, c: "#f5566a" },
            { x: 100, c: "#c21e31" },
          ]
            .filter((m) => m.x <= dist.field_n)
            .map((m) => (
              <ReferenceLine
                key={m.x}
                x={m.x}
                stroke={m.c}
                strokeDasharray="4 3"
                strokeOpacity={0.85}
              />
            ))}
          <ReferenceLine x={dist.mean_place} stroke="#00e657" strokeWidth={2} />
          <Bar dataKey="pct" fill="#f22e45" fillOpacity={0.9} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
