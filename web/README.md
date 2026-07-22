# dfs-web

Next.js + React + TypeScript front end for the DFS contest simulator — the
Vercel-hosted replacement for the Streamlit `app.py`. It renders the four-tab
workspace (Setup · Players · Results · Export) in the RotoWire full-dark theme
and talks to the Python worker (`../service`) through same-origin `/api` proxy
routes.

See `../ARCHITECTURE.md` for how this fits with the worker and object store.

## Develop

```bash
npm install
cp .env.example .env.local     # set WORKER_API_URL (default http://localhost:8000)
npm run dev                     # http://localhost:3000
```

The worker must be running (see `../service/README.md`). The **Players** tab is
wired end-to-end; Setup/Results/Export show the planned control surface and are
ported in later phases.

## Deploy to Vercel

1. Import the repo in Vercel and set **Root Directory** to `web/`.
2. Set env vars: `WORKER_API_URL` (your deployed worker URL) and optionally
   `WORKER_API_KEY`.
3. Deploy. Vercel auto-detects Next.js (`vercel.json` pins the framework).

When ready, `web/` can be extracted into its own repo verbatim — nothing here
imports from the parent directory except brand assets already copied into
`public/`.

## Layout

```
app/
  layout.tsx            root layout + theme
  page.tsx              renders <Workspace/>
  globals.css           RotoWire theme (fonts, tokens)
  api/                  server-side proxy routes → worker
    status/route.ts
    players/route.ts
    players/[name]/distribution/route.ts
components/
  Workspace.tsx         tab shell
  Header.tsx            branded header + live status badge
  PlayersTab.tsx        table + search + type filter (wired)
  PlayerDistChart.tsx   Recharts histogram
  Placeholder.tsx       planned-surface cards for unported tabs
lib/
  worker.ts             server-side worker fetch/proxy helper
  api.ts                client-side typed API + response types
public/                 fonts + logos (copied from ../static, ../assets)
```

## Theme

Brand tokens live in `tailwind.config.ts` (colors `rw.*`, fonts) and
`app/globals.css` (`@font-face`, header styles), ported from
`../.streamlit/config.toml` and the `:root` block in `../app.py`.
