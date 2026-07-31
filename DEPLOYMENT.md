# Deployment runbook — Streamlit app

How to run the DFS contest simulator (the Streamlit app in `app.py`) in
production. There are two pieces, and only the first is required:

1. **The app itself** — `streamlit run app.py`.
2. **An optional object store** (Cloudflare R2 / S3) — so multiple users share
   one refreshed build and it survives restarts, plus a **scheduled morning
   refresh** so the slow projection rebuild (Stage B) is done ahead of time.

If you run the app on a single always-on machine, you can skip the object store
entirely and use a local cron for the refresh (see Step 4).

---

## Step 1 — Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the sims in `deliverables/` and the projections in `out/`. On a
single machine those files on disk are shared across every browser session, so
one refresh is seen by everyone hitting that instance. To host it, any platform
that runs a long-lived Python process works (Streamlit Community Cloud, a VM,
a container). Streamlit needs a persistent WebSocket connection, so a stateless
serverless host is not suitable.

Theme tokens live in `.streamlit/config.toml`; brand fonts are served from
`static/fonts/`. The app is Windows-portable.

---

## Step 2 — (Optional) Shared object store for multi-user / persistence

On an ephemeral or multi-replica host the local filesystem doesn't persist, so
configure an **S3-compatible bucket** to share one build across all users and
survive restarts. R2 is S3-compatible with no egress fees (plain AWS S3 works
too — skip the `ENDPOINT` bits).

1. Cloudflare dashboard → **R2** → **Create bucket** (e.g. `dfs-sims`). Note your
   **Account ID**.
2. R2 → **Manage R2 API Tokens** → **Create API token** (Object Read & Write,
   scoped to the bucket). Save the **Access Key ID** and **Secret Access Key**.
3. Your endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.

Give the app the config via **Streamlit secrets** (`.streamlit/secrets.toml`, or
the app's Settings → Secrets on Streamlit Cloud):

```toml
[shared_store]
bucket = "dfs-sims"
prefix = "dfs"                                          # optional
region = "auto"                                         # R2 ignores it
endpoint_url = "https://<account-id>.r2.cloudflarestorage.com"
access_key_id = "..."
secret_access_key = "..."
```

Requires `boto3` (already in `requirements.txt`). With no `[shared_store]` config
the app runs purely on the local filesystem, unchanged. See
`.streamlit/secrets.toml.example` and the *Sharing across users* section of
`README_app.md` for how the app pulls/pushes.

---

## Step 3 — Seed the store with today's artifacts

Run the pipeline once, then publish. The same env-var schema also drives the
scheduled Action in Step 4:

```bash
python refresh_and_run.py            # produce fresh projections + sims

export SHARED_STORE_BUCKET=dfs-sims
export SHARED_STORE_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
export AWS_ACCESS_KEY_ID=…
export AWS_SECRET_ACCESS_KEY=…
python scripts/push_artifacts.py     # add --check first to preview
```

You should see the shared artifact set upload (two `.npy` sims, two projection
CSVs, the build stamp, the slate JSON).

Each publish also **archives a dated snapshot** of that day's prediction set under
`history/<slate-date>/` in the same bucket, and prunes to a rolling window (see
below) — so the latest keys always hold today's build and the history folder
holds the last few days for accuracy review.

---

## Retention — reviewing past slates for sim accuracy

Every publish (the scheduled refresh, `scripts/push_artifacts.py`, or an
in-app rebuild) snapshots the build's **slate-specific prediction set** — the two
DK sim arrays, the slate, the build stamp, and the projection summaries + manifest
— into `history/<slate-date>/`. Retention is enforced automatically:

- **Keeps the last 4 days** by default. A day of sims is ~25 MB, so 4 days
  (~100 MB) sits far inside the R2/B2 ~10 GB **free tier**.
- A **storage-budget guard** (default 2 GB) is applied on top: if a window ever
  wouldn't fit the budget, only the most-recent days that fit are kept (the build
  just published is always kept). So it can only ever *shrink* below 4 days, never
  blow the free tier.
- Snapshots are **deduped by build timestamp**, so re-publishing the same build
  uploads nothing extra; a genuinely newer build for the same date (e.g. a forced
  rebuild once final lineups post) overwrites that date's snapshot with the better
  one.

Tune the window with repo secrets / env vars (both optional):

```
SHARED_STORE_HISTORY_DAYS=4        # rolling window length
SHARED_STORE_HISTORY_MAX_MB=2048   # hard storage cap for the window
```

Review the retained snapshots:

```bash
python scripts/push_artifacts.py --list-history            # dates + sizes retained
python scripts/push_artifacts.py --pull-history 2026-07-24 # download that day into review/
```

`--pull-history` writes `hitter_dk_sims.npy`, `pitcher_dk_sims.npy`,
`hitter_projections.csv`, `pitcher_projections.csv`, `sim_manifest.json`, and
`slate.json` for the date, so you can score that slate's sims against the real box
scores and feed what you learn back into the projection/sim logic.

---

## Step 4 — Schedule the morning refresh (so Stage B never blocks a Run)

The projection rebuild (Stage B) is by far the slowest part of a cold Run. A
scheduled job rebuilds it every morning and publishes, so an interactive Run
finds today's projections already dated today and **skips Stage B entirely** —
at most it does a fast Stage C re-sim if lineups moved.

**With the object store (GitHub Actions).** Add the store config as repo secrets
(**Settings → Secrets and variables → Actions**): `SHARED_STORE_BUCKET`,
`SHARED_STORE_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION` (and `SHARED_STORE_PREFIX` if used). The workflow
`.github/workflows/refresh.yml` then runs **daily at 11:00 UTC (~7am ET)**:

- It rebuilds **projections + sims** (`refresh_and_run.py --skip-bip`, reusing the
  committed `bip_inputs/`), stamps the build (`scripts/stamp_build.py
  --projections`), and publishes to the store.
- **Run workflow** (manual) offers three modes: **projections** (default, what
  the schedule runs), **sims** (fast Stage C only), and **full** (Stage A+B+C
  including the heavy Statcast **BIP scrape** — run occasionally, e.g. weekly, to
  refresh the underlying data).
- Change the time by editing the `schedule:` cron; projections don't need posted
  lineups, so an earlier slot (e.g. `0 11 * * *` ≈ 7am ET) gives users a bigger
  head start.

**Single server (no object store).** Schedule the same commands with cron so they
write straight to the app's `deliverables/` and `out/`:

```cron
# 7am daily — rebuild projections + sims for the app on this box
0 7 * * *  cd /path/to/DFSSimsFull && /usr/bin/python refresh_and_run.py --skip-bip \
             && /usr/bin/python scripts/stamp_build.py --projections >> refresh.log 2>&1
```

The app reads the same `out/.build_stamp.json`, so it skips Stage B on Runs the
same way — no object store required. The app's build banner shows when the
baselines (Stage B) were last refreshed.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Banner: baselines not from today | The morning refresh didn't run/publish — trigger **Actions → refresh-sims → Run workflow** (or your cron). |
| Shared build not appearing for a user | Hit **↻ Refresh** on the Setup tab (pulls the latest build); confirm the `[shared_store]` secrets match the store the Action publishes to. |
| Slate catalog / team-totals empty | Those hit live RotoWire/Vegas feeds; retry when the feed is back. |
| Stage B rebuild fails in the app | Needs `scikit-learn`/`xgboost`/`pybaseball`/`pyarrow` and reachable `statsapi.mlb.com`; on Python 3.14 those wheels may be missing — run on Python 3.11–3.12. The sims still rebuild on the existing projections. |
