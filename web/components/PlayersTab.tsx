"use client";

import { useEffect, useMemo, useState } from "react";
import {
  fetchPlayers,
  fetchPlayerDist,
  type PlayerDist,
  type PlayerRow,
} from "@/lib/api";
import PlayerDistChart from "./PlayerDistChart";

type Kind = "all" | "hitters" | "pitchers";

// Column order mirrors app.py:cached_player_table.
const COLUMNS: { key: keyof PlayerRow; label: string; num?: boolean }[] = [
  { key: "Player", label: "Player" },
  { key: "Type", label: "Type" },
  { key: "Proj", label: "Proj", num: true },
  { key: "Floor (p10)", label: "Floor", num: true },
  { key: "Median", label: "Median", num: true },
  { key: "Ceiling (p90)", label: "Ceiling", num: true },
  { key: "p99", label: "p99", num: true },
  { key: "Std", label: "Std", num: true },
  { key: "Bust% (<=0)", label: "Bust%", num: true },
  { key: "2x%", label: "2x%", num: true },
  { key: "30+%", label: "30+%", num: true },
];

export default function PlayersTab() {
  const [kind, setKind] = useState<Kind>("all");
  const [search, setSearch] = useState("");
  const [rows, setRows] = useState<PlayerRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [dist, setDist] = useState<PlayerDist | null>(null);
  const [distErr, setDistErr] = useState<string | null>(null);

  // Debounced fetch of the table on filter/search change.
  useEffect(() => {
    const id = setTimeout(() => {
      setLoading(true);
      setErr(null);
      fetchPlayers(kind, search)
        .then((r) => setRows(r.players))
        .catch((e) => setErr(e.message))
        .finally(() => setLoading(false));
    }, 200);
    return () => clearTimeout(id);
  }, [kind, search]);

  // Default the inspected player to the top projection once rows load.
  useEffect(() => {
    if (!selected && rows.length) setSelected(rows[0].Player);
  }, [rows, selected]);

  useEffect(() => {
    if (!selected) return;
    setDistErr(null);
    fetchPlayerDist(selected, 40)
      .then(setDist)
      .catch((e) => setDistErr(e.message));
  }, [selected]);

  const count = rows.length;
  const kinds: Kind[] = useMemo(() => ["all", "hitters", "pitchers"], []);

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
            Show
          </label>
          <div className="flex gap-1">
            {kinds.map((k) => (
              <button
                key={k}
                onClick={() => setKind(k)}
                className={
                  "rounded-lg border px-3 py-1.5 text-sm capitalize " +
                  (k === kind
                    ? "border-rw-red bg-rw-red text-white"
                    : "border-rw-line bg-rw-raised text-rw-mut hover:text-white")
                }
              >
                {k}
              </button>
            ))}
          </div>
        </div>
        <div className="grow">
          <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
            Search player
          </label>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="e.g. Ohtani"
            className="w-full max-w-xs rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-sm text-white outline-none placeholder:text-rw-mut focus:border-rw-red"
          />
        </div>
        <div className="pb-1 font-mono text-[11px] uppercase tracking-wider text-rw-mut">
          {loading ? "loading…" : `${count} players`}
        </div>
      </div>

      {err && (
        <div className="rounded-card border border-rw-red-700 bg-rw-red-700/20 p-3 text-sm">
          {err}
        </div>
      )}

      {/* Distribution chart for the selected player */}
      {dist && <PlayerDistChart dist={dist} />}
      {distErr && <div className="text-sm text-rw-mut">{distErr}</div>}

      {/* Player table */}
      <div className="overflow-x-auto rounded-card border border-rw-line">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-rw-raised text-left font-mono text-[10px] uppercase tracking-wider text-rw-mut">
              {COLUMNS.map((c) => (
                <th
                  key={String(c.key)}
                  className={"px-3 py-2 " + (c.num ? "text-right" : "")}
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const on = r.Player === selected;
              return (
                <tr
                  key={r.Player + r.Type}
                  onClick={() => setSelected(r.Player)}
                  className={
                    "cursor-pointer border-t border-rw-line/60 " +
                    (on ? "bg-rw-red/15" : "hover:bg-rw-raised/60")
                  }
                >
                  {COLUMNS.map((c) => (
                    <td
                      key={String(c.key)}
                      className={
                        "px-3 py-1.5 " +
                        (c.num ? "text-right tabular-nums" : "font-medium")
                      }
                    >
                      {String(r[c.key])}
                    </td>
                  ))}
                </tr>
              );
            })}
            {!loading && !rows.length && (
              <tr>
                <td
                  colSpan={COLUMNS.length}
                  className="px-3 py-6 text-center text-rw-mut"
                >
                  No players match.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
