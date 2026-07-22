export default function Placeholder({
  title,
  phase,
  items,
}: {
  title: string;
  phase: string;
  items: string[];
}) {
  return (
    <div className="rounded-card border border-rw-line bg-rw-surface p-6">
      <div className="mb-1 flex items-baseline gap-3">
        <h2 className="text-xl">{title}</h2>
        <span className="font-mono text-[10px] uppercase tracking-widest text-rw-mut">
          {phase} — port in progress
        </span>
      </div>
      <p className="mb-4 text-sm text-rw-mut">
        Wired to the same warm worker API as the Players tab. Controls below are
        the planned surface, ported from the Streamlit app.
      </p>
      <ul className="space-y-1.5">
        {items.map((it) => (
          <li key={it} className="flex items-start gap-2 text-sm">
            <span className="mt-[6px] h-[6px] w-[6px] shrink-0 rounded-full bg-rw-red" />
            {it}
          </li>
        ))}
      </ul>
    </div>
  );
}
