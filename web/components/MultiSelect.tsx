"use client";

import { useEffect, useRef, useState } from "react";

// Compact multi-select: a button showing the selection count that opens a
// searchable checkbox panel. Works for small enums (stacks/teams/sizes) and the
// ~300-name player pool alike.
export default function MultiSelect({
  label,
  options,
  selected,
  onChange,
  placeholder = "Any",
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const sel = new Set(selected);
  const shown = q
    ? options.filter((o) => o.toLowerCase().includes(q.toLowerCase()))
    : options;

  function toggle(o: string) {
    const next = new Set(sel);
    if (next.has(o)) next.delete(o);
    else next.add(o);
    onChange([...next]);
  }

  return (
    <div ref={ref} className="relative">
      <label className="mb-1 block font-mono text-[10px] uppercase tracking-widest text-rw-mut">
        {label}
      </label>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between rounded-lg border border-rw-line bg-rw-raised px-3 py-1.5 text-left text-sm text-white hover:border-rw-red"
      >
        <span className={selected.length ? "" : "text-rw-mut"}>
          {selected.length ? `${selected.length} selected` : placeholder}
        </span>
        <span className="text-rw-mut">▾</span>
      </button>

      {selected.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {selected.map((s) => (
            <button
              key={s}
              onClick={() => toggle(s)}
              className="rounded bg-rw-red/20 px-1.5 py-0.5 text-[11px] text-white hover:bg-rw-red/40"
              title="remove"
            >
              {s} ✕
            </button>
          ))}
        </div>
      )}

      {open && (
        <div className="absolute z-20 mt-1 max-h-64 w-full overflow-auto rounded-lg border border-rw-line bg-rw-ink shadow-xl">
          {options.length > 8 && (
            <input
              autoFocus
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="search…"
              className="sticky top-0 w-full border-b border-rw-line bg-rw-raised px-3 py-1.5 text-sm text-white outline-none"
            />
          )}
          {shown.map((o) => (
            <label
              key={o}
              className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-rw-raised/60"
            >
              <input
                type="checkbox"
                checked={sel.has(o)}
                onChange={() => toggle(o)}
                className="accent-rw-red"
              />
              {o}
            </label>
          ))}
          {shown.length === 0 && (
            <div className="px-3 py-2 text-sm text-rw-mut">no matches</div>
          )}
        </div>
      )}
    </div>
  );
}
