"use client";

import { useEffect, useRef, useState } from "react";

interface JobStatus {
  id: string;
  kind: string;
  state: string;
  returncode: number | null;
  error: string | null;
  log_tail: string[];
}

// Kicks the heavy sim rebuild (POST /refresh) and polls its status — the
// non-blocking replacement for the Streamlit app's minutes-long spinner.
export default function RefreshPanel() {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  const terminal = job && ["succeeded", "failed", "skipped"].includes(job.state);

  async function poll(id: string) {
    try {
      const res = await fetch(`/api/refresh/status/${id}`);
      const st: JobStatus = await res.json();
      setJob(st);
      if (!["succeeded", "failed", "skipped"].includes(st.state)) {
        timer.current = setTimeout(() => poll(id), 2000);
      }
    } catch {
      timer.current = setTimeout(() => poll(id), 3000);
    }
  }

  async function start() {
    setStarting(true);
    setErr(null);
    setJob(null);
    try {
      const res = await fetch("/api/refresh", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ full: false }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail ?? "refresh failed to start");
      poll(j.job_id);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  return (
    <section className="rounded-card border border-rw-line bg-rw-surface p-5">
      <h2 className="mb-1 text-lg">Rebuild sims</h2>
      <p className="mb-3 text-xs text-rw-mut">
        Regenerates today’s correlated sims (Stage C) on the worker as a
        background job. Needs live feeds + the worker’s pipeline environment;
        harmless to trigger — progress streams below.
      </p>
      <button
        onClick={start}
        disabled={starting || (job != null && !terminal)}
        className="rounded-lg border border-rw-line bg-rw-raised px-4 py-2 text-sm text-white hover:border-rw-red disabled:opacity-50"
      >
        {job && !terminal ? "Rebuilding…" : starting ? "Starting…" : "↻ Rebuild sims"}
      </button>

      {err && <div className="mt-3 text-sm text-rw-red-400">{err}</div>}
      {job && (
        <div className="mt-3">
          <div className="font-mono text-[11px] uppercase tracking-wider">
            <span
              className={
                job.state === "succeeded"
                  ? "text-rw-turf"
                  : job.state === "failed" || job.state === "skipped"
                    ? "text-rw-red-400"
                    : "text-rw-mut"
              }
            >
              {job.state}
            </span>
            {job.error ? ` · ${job.error}` : ""}
          </div>
          {job.log_tail.length > 0 && (
            <pre className="mt-2 max-h-40 overflow-auto rounded-lg border border-rw-line bg-rw-ink p-2 text-[11px] leading-snug text-rw-mut">
              {job.log_tail.join("\n")}
            </pre>
          )}
        </div>
      )}
    </section>
  );
}
