# Vercel migration architecture

This repo is migrating from a single Streamlit app (`app.py`) to a
Vercel-hosted web app backed by an off-Vercel Python worker. This document
explains the split, why it's shaped this way, and how the pieces talk.

## Why not "Streamlit on Vercel"?

Streamlit needs a long-lived WebSocket server; Vercel runs **stateless
serverless/edge functions** (≤300s execution, 250MB function size, ephemeral
disk). The Streamlit app's **Run** button spawns multi-minute subprocesses —
`run_pipeline.py` (Stage B: XGBoost + scikit-learn + pybaseball/Statcast) and
`run_slate.py` (Stage C: 10,000 correlated numpy sims) — whose combined
dependencies alone exceed Vercel's function-size limit. That compute cannot run
on Vercel.

But `app.py` is, in practice, a **thin UI over pre-computed artifacts plus a
subprocess launcher**. At UI time it imports only `numpy/scipy/pandas` (never
xgboost/sklearn/pybaseball) and reads small artifacts (24 MB of `.npy` sims +
~1 MB of projection CSVs). So the app splits cleanly.

## The three pieces

```
┌─────────────────────┐   HTTPS/JSON    ┌──────────────────────────────┐
│  Vercel: web/        │ ─────────────▶ │  Worker: service/             │
│  Next.js + React +   │                │  FastAPI, always-on           │
│  TypeScript UI       │ ◀───────────── │  • warm numeric API           │
│  (4-tab workspace)   │                │  • heavy pipeline as jobs      │
│  + /api proxy routes │                │    (later phase)               │
└─────────────────────┘                └───────────────┬──────────────┘
                                                        │ pull / push
                                                        ▼
                                        ┌────────────────────────────┐
                                        │  Object store (R2 / S3)     │
                                        │  hitter/pitcher_dk_sims.npy │
                                        │  *_pa_projections_*.csv     │
                                        │  .build_stamp.json (version)│
                                        │  slate.json                 │
                                        └────────────────────────────┘
```

### 1. `web/` — Next.js app (deploys to Vercel)
Pure presentation + a thin server-side proxy. It never talks to the worker
directly from the browser: client code calls this app's own `/api/*` route
handlers (`web/app/api/**`), which forward to the worker via
`web/lib/worker.ts`. That hides the worker origin and is the single place to
add auth later. Brand theme (RotoWire full-dark) is ported from
`.streamlit/config.toml` + `app.py` into Tailwind tokens
(`web/tailwind.config.ts`) and `web/app/globals.css`; fonts/logos live in
`web/public/`.

**Deploy note:** point Vercel's *Root Directory* at `web/`. When you're ready,
`web/` lifts out into its own repo unchanged.

### 2. `service/` — worker FastAPI (deploys to Fly.io / Render / Railway)
A long-lived process that reuses the repo's numeric modules directly (no
rewrite): `stage_d`, `mlb_lineup_builder`, `portfolio`, `portfolio_ev`,
`field_simulator`, `stack_signal`, `dk_ids`, `dk_slate_feed`, `slate_ingest`,
`showdown_*`. Being warm, it loads the ~24 MB sim arrays **once** and keeps them
resident (`service/sims.py`), so contest sims / EV are fast — no per-request
reload. In a later phase it also launches the heavy Stage B/C pipeline as
background jobs (reusing `ensure_fresh`/`run_script` logic from `app.py`) and
pushes fresh artifacts to the object store.

### 3. Object store (Cloudflare R2 or S3)
The durable hand-off between the heavy pipeline and the web tier. This reuses
the existing `shared_store.py` (artifact list, `pull`/`push`, and the
`.build_stamp.json` "version" marker). The pipeline pushes; the worker pulls
the newest build (gated cheaply on the stamp timestamp).

## Data flow

- **Read (implemented):** browser → `web/app/api/players/route.ts` →
  worker `GET /players` → warm sims → JSON table. Same for
  `/players/{name}/distribution` and `/status`.
- **Refresh (planned):** UI → worker `POST /refresh` → background Stage B/C →
  `shared_store.push()` → worker reloads → UI polls `GET /refresh/status`.
- **Run a slate (planned):** UI params → worker `POST /run` (field build +
  `run_contest_dist`) → run summary + `run_id`; heavy arrays stay server-side,
  fetched by `run_id` for Results/Export.

## Environment variables

**Worker (`service/`)** — same schema as the old Streamlit secrets:

| var | purpose |
|---|---|
| `SHARED_STORE_BUCKET` | R2/S3 bucket (enables remote mode; unset ⇒ local files) |
| `SHARED_STORE_ENDPOINT` | S3-compatible endpoint (R2/MinIO/B2) |
| `SHARED_STORE_PREFIX` | optional key prefix |
| `AWS_REGION` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | credentials |
| `CORS_ALLOW_ORIGINS` | comma-separated allowed origins (default `*`) |
| `WORKER_API_KEY` | optional shared secret checked against `x-api-key` |

**Web (`web/`):**

| var | purpose |
|---|---|
| `WORKER_API_URL` | worker base URL (server-side only) |
| `WORKER_API_KEY` | optional; sent to the worker as `x-api-key` |

## Object-store layout

Keys mirror the repo paths (optionally under `SHARED_STORE_PREFIX`):

```
<prefix>/deliverables/hitter_dk_sims.npy
<prefix>/deliverables/pitcher_dk_sims.npy
<prefix>/out/hitter_pa_projections_2027.csv
<prefix>/out/pitcher_pa_projections_2027.csv
<prefix>/out/.build_stamp.json      # written LAST = atomic version marker
<prefix>/data/slate.json
<prefix>/_refresh.lock              # cross-instance rebuild lock (TTL)
```

Seed it from a local pipeline run with `python scripts/push_artifacts.py`.

## Local development

```bash
# 1. Worker (from repo root) — serves the artifacts already in deliverables/
pip install -r service/requirements.txt
uvicorn service.main:app --reload --port 8000
#    ...or: docker compose up

# 2. Web (in web/)
cd web && npm install
cp .env.example .env.local        # WORKER_API_URL=http://localhost:8000
npm run dev                        # http://localhost:3000
```

## Migration status

| Area | State |
|---|---|
| Worker: `/status`, `/players`, `/players/{name}/distribution` | ✅ done |
| Web: theme, 4-tab shell, **Players tab** end-to-end | ✅ done |
| Object-store layer + `scripts/push_artifacts.py` | ✅ done |
| Worker: `POST /run`, `/run/{id}`, place-distribution, `/slate/*` | ✅ Phase 1 |
| Worker: async `POST /refresh` + `/refresh/status/{id}` job | ✅ Phase 1 |
| Web: **Setup tab** (slate/upload, params, Run) + **Results** render | ✅ Phase 1 |
| Results tab: filters, marks, richer detail | ⏳ Phase 2 |
| Export tab (DK upload, EV, exposure caps) | ⏳ Phase 3 |
| Showdown branch | ⏳ Phase 4 |
| Worker deploy hardening (Docker, cron, auth) | ⏳ Phase 5 |

### Run path (Phase 1) at a glance

`Setup` picks a slate (bundled sample / uploaded CSV / RotoWire catalog) →
`POST /run` with the tuning params → the worker builds candidates + an
ownership-weighted field, scores both against the correlated sims, runs the
contest (`runner.run_slate`, a faithful port of app.py's classic Run handler),
caches the full payload by `run_id` (`runstore`), and returns a JSON summary →
`Results` renders metrics, the candidate table, and each lineup's
finishing-place distribution (`GET /run/{id}/candidate/{c}/place-distribution`).
Heavy rebuilds go through the async job (`jobs.start_refresh`).

## Backend improvements this unlocks

- **True multi-user** off one shared artifact set — no in-process rebuild
  stampede.
- **Non-blocking refresh** with progress (async job + polling) instead of a
  minutes-long blocking spinner.
- **Persistent, shareable runs** keyed by `run_id` (impossible with ephemeral
  `st.session_state`).
- **CDN/edge caching** for artifacts and the slate catalog.
- **Auth & access control** at the Vercel/worker edge.
- **Clean dependency split:** heavy ML stays on the worker; the web tier is lean.
