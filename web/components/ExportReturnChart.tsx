"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReturnHist } from "@/lib/api";

// Overlaid per-slate $-return distributions: payout-aware EV set vs a
// rank-selected set of the same size — the JS analogue of
// app.py:portfolio_return_chart.
export default function ExportReturnChart({ hist }: { hist: ReturnHist }) {
  return (
    <div>
      <div className="mb-2 font-mono text-[11px] uppercase tracking-wider text-rw-mut">
        Mean per-slate return — EV{" "}
        <span className="text-rw-turf">${hist.mean_ev.toLocaleString()}</span> vs
        ranked{" "}
        <span className="text-rw-mut">${hist.mean_ranked.toLocaleString()}</span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={hist.bins} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#1c4a7a" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="x"
            tick={{ fill: "#8ba0ba", fontSize: 11 }}
            stroke="#1c4a7a"
            tickFormatter={(v) => `$${v}`}
          />
          <YAxis tick={{ fill: "#8ba0ba", fontSize: 11 }} stroke="#1c4a7a" />
          <Tooltip
            contentStyle={{
              background: "#001428",
              border: "1px solid #1c4a7a",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
            labelFormatter={(v) => `$${v} / slate`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Area
            type="monotone"
            dataKey="ranked"
            name="Ranked"
            stroke="#8ba0ba"
            fill="#8ba0ba"
            fillOpacity={0.25}
          />
          <Area
            type="monotone"
            dataKey="ev"
            name="Payout-aware EV"
            stroke="#f22e45"
            fill="#f22e45"
            fillOpacity={0.35}
          />
          <ReferenceLine x={0} stroke="#ffffff" strokeOpacity={0.4} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
