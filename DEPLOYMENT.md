# Deployment runbook — going live

A step-by-step to get the migrated app running in production. Three things get
stood up, in this order:

1. **Object store** (Cloudflare R2) — holds the sim artifacts.
2. **Worker** (Fly.io) — the Python API that reads those artifacts and runs sims.
3. **Web app** (Vercel) — the UI, pointed at the worker.

You'll also wire an optional **daily refresh** (GitHub Actions). Budget ~30–45
minutes the first time. Everything below uses the free/low tiers.

> Prerequisites: a machine that can already run the pipeline today (the repo as
> you use it now), plus the `fly` CLI and a Vercel + Cloudflare account. Install
> the Fly CLI from https://fly.io/docs/hurry/ .

---

## Step 1 — Create the object store (Cloudflare R2)

R2 is S3-compatible and has no egress fees. (Plain AWS S3 works too — skip the
`ENDPOINT` bits if you use S3.)

1. Cloudflare dashboard → **R2** → **Create bucket**. Name it e.g. `dfs-sims`.
   Note your **Account ID** (top of the R2 page).
2. R2 → **Manage R2 API Tokens** → **Create API token**:
   - Permission: **Object Read & Write**, scoped to the `dfs-sims` bucket.
   - Save the **Access Key ID** and **Secret Access Key** (shown once).
3. Your S3 endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.

You now have five values you'll reuse everywhere:

| Name | Example |
|---|---|
| `SHARED_STORE_BUCKET` | `dfs-sims` |
| `SHARED_STORE_ENDPOINT` | `https://<account-id>.r2.cloudflarestorage.com` |
| `AWS_ACCESS_KEY_ID` | `…` |
| `AWS_SECRET_ACCESS_KEY` | `…` |
| `AWS_REGION` | `auto` (R2 ignores it; any value is fine) |

---

## Step 2 — Seed the store with today's artifacts

Run the pipeline once locally as you do now, then publish. From the repo root:

```bash
# produce fresh artifacts (whatever you normally run), e.g.:
python refresh_and_run.py            # or: python run_slate.py

# point at the store and upload
export SHARED_STORE_BUCKET=dfs-sims
export SHARED_STORE_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
export AWS_ACCESS_KEY_ID=…
export AWS_SECRET_ACCESS_KEY=…
python scripts/push_artifacts.py            # add --check first to preview
```

You should see the six artifacts upload (two `.npy` sims, two projection CSVs,
the build stamp, the slate JSON). This is the data the worker will serve.

---

## Step 3 — Deploy the worker (Fly.io)

From the repo root (`fly.toml` is already here):

```bash
fly auth login
fly launch --no-deploy          # creates the app; keep the generated name or
                                # edit `app = "…"` in fly.toml to something unique

# give the worker its secrets (never commit these):
fly secrets set \
  SHARED_STORE_BUCKET=dfs-sims \
  SHARED_STORE_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com \
  AWS_ACCESS_KEY_ID=… \
  AWS_SECRET_ACCESS_KEY=… \
  WORKER_API_KEY=$(openssl rand -hex 24) \
  CORS_ALLOW_ORIGINS=https://<your-vercel-app>.vercel.app

fly deploy
```

- Save the **`WORKER_API_KEY`** you generated — Vercel needs the same value.
- Note the worker URL Fly prints, e.g. `https://dfs-worker.fly.dev`.
- Check it: `curl https://dfs-worker.fly.dev/health` → `{"status":"ok",…}`.
  (`/health` is intentionally open; every other route needs the key.)

> Don't yet know your Vercel URL for `CORS_ALLOW_ORIGINS`? Set it to `*` for
> now and tighten it after Step 4 with another `fly secrets set`.

---

## Step 4 — Deploy the web app (Vercel)

1. https://vercel.com → **Add New… → Project** → import this GitHub repo.
2. **Root Directory**: set to **`web`** (click Edit, pick the `web` folder).
   Framework auto-detects as Next.js.
3. **Environment Variables** (Project → Settings → Environment Variables):
   | Key | Value |
   |---|---|
   | `WORKER_API_URL` | `https://dfs-worker.fly.dev` (your worker URL) |
   | `WORKER_API_KEY` | the same random string from Step 3 |
4. **Deploy**. Open the resulting URL. The header badge should read
   `10,000 sims · N players` — that means the browser → Vercel → worker → R2
   path is live.
5. If you set CORS to `*` earlier, now run
   `fly secrets set CORS_ALLOW_ORIGINS=https://<your-app>.vercel.app`.

Smoke test in the browser: **Setup → Load bundled sample slate → Run
simulation** → **Results** shows metrics; **Export** builds a portfolio.

---

## Step 5 — Automate the daily refresh (optional but recommended)

So you don't have to run the pipeline by hand each day:

1. GitHub → repo **Settings → Secrets and variables → Actions → New repository
   secret**, add: `SHARED_STORE_BUCKET`, `SHARED_STORE_ENDPOINT`,
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
   (and `SHARED_STORE_PREFIX` if you used one).
2. The workflow `.github/workflows/refresh.yml` already runs daily at 13:00 UTC
   (~9am ET) and can be run on demand: **Actions → refresh-sims → Run
   workflow**. It rebuilds and publishes to R2; the worker picks up the new
   build automatically on its next request.
3. Adjust the cron time by editing the `schedule:` line in that file.

The in-app **Rebuild sims** button (Setup tab) triggers the worker's own
on-demand rebuild job for the same effect between scheduled runs.

---

## How it fits together (recap)

```
 daily cron / Rebuild button ─▶ pipeline ─▶ R2 (artifacts + build stamp)
                                              │
 browser ─▶ Vercel (web/, /api proxy + key) ─▶ Fly worker ─▶ reads R2, runs sims
```

- The browser never sees the worker URL or key — the Vercel `/api` routes add
  the key server-side.
- The worker keeps the sims warm in memory and re-pulls from R2 whenever a
  newer build stamp appears.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Header badge says "worker offline" | `curl <worker>/health`; check `WORKER_API_URL` in Vercel and that Fly app is running (`fly status`). |
| Everything 401s | `WORKER_API_KEY` differs between Fly and Vercel — set them to the same value. |
| `/status` → 503 "artifacts not found" | Step 2 didn't publish; re-run `scripts/push_artifacts.py` and check the bucket. |
| CORS error in browser console | `CORS_ALLOW_ORIGINS` on the worker must include your exact Vercel origin (or `*`). |
| Slate catalog / team-totals empty | Those hit live RotoWire/Vegas feeds; use the sample slate or an uploaded CSV if a feed is down. |
| Showdown upload CSV missing | Expected — showdown needs a DK showdown DKSalaries template for captain ids (a follow-up); the portfolio + exposure still render. |

## What is NOT covered yet

- **Showdown DK-upload CSV**: needs a DKSalaries showdown template upload for
  captain ids (classic upload CSVs work when the slate carries ids).
- **Auth for humans**: the worker uses a shared API key for the Vercel↔worker
  hop; add Vercel Authentication (Settings → Deployment Protection) if you want
  to gate who can open the site.
