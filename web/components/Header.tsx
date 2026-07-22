"use client";

import { useEffect, useState } from "react";
import { fetchStatus, type WorkerStatus } from "@/lib/api";

export default function Header() {
  const [status, setStatus] = useState<WorkerStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus().then(setStatus).catch((e) => setErr(e.message));
  }, []);

  const badge = err
    ? "worker offline"
    : status
      ? `${status.n_sim?.toLocaleString() ?? "?"} sims · ${status.hitters + status.pitchers} players`
      : "loading…";

  return (
    <header className="rw-header">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/logo.svg" alt="" className="h-11 w-11 shrink-0" />
      <div className="rw-divider" />
      <div>
        <div className="rw-title">DFS Contest Simulator</div>
        <div className="rw-eyebrow">
          MLB · Correlated DraftKings Sims · Portfolio EV
        </div>
      </div>
      <span
        className="rw-badge"
        style={err ? { background: "#c21e31" } : undefined}
        title={
          status?.remote_store
            ? "Reading shared artifacts from object storage"
            : "Reading local artifacts"
        }
      >
        <span
          className="inline-block h-[7px] w-[7px] rounded-full bg-white"
          aria-hidden
        />
        {badge}
      </span>
    </header>
  );
}
