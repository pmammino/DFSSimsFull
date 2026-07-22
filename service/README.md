# DFS worker service

FastAPI service that powers the Vercel web app (`../web`). It reuses this
repo's numeric modules directly and serves the warm, interactive API; in a
later phase it also runs the heavy Stage B/C pipeline as background jobs. See
`../ARCHITECTURE.md` for the full picture.

## Run locally

From the **repo root** (so the service can import the repo modules):

```bash
pip install -r service/requirements.txt
uvicorn service.main:app --reload --port 8000
```

Or containerized (build context is the repo root):

```bash
docker compose up            # uses ./deliverables + ./out via bind mounts
```

Open http://localhost:8000/docs for the interactive OpenAPI UI.

## Endpoints

| Method / path | Status | Wraps |
|---|---|---|
| `GET /health` | ✅ | liveness (no array load) |
| `GET /status` | ✅ | sim inventory + build stamp |
| `GET /players?kind=&search=` | ✅ | `app.py:cached_player_table` |
| `GET /players/{name}/distribution?nbins=` | ✅ | `app.py:player_score_chart` data |
| `GET /params/defaults` | ✅ | RunParams defaults + size presets |
| `POST /run` | ✅ | field build + `run_contest_dist` (→ run_id) |
| `GET /run/{id}` | ✅ | re-fetch a run summary |
| `GET /run/{id}/candidate/{c}/place-distribution` | ✅ | `place_distribution_chart` data |
| `GET /slate/catalog` | ✅ | `dk_slate_feed.build_catalog` (live, cached) |
| `POST /slate/upload` | ✅ | parse DK export / clean CSV |
| `GET /slate/sample` | ✅ | bundled sample slate (offline dev) |
| `GET /slate/team-totals` | ✅ | Vegas totals (live) |
| `POST /refresh` (+ `/refresh/status/{id}`) | ✅ | Stage C (or full) as a background job |
| `POST /export/dk-upload`, `/export/ev` | ⏳ | `build_dk_upload`, `build_dk_upload_ev` |

## Artifacts

The service reads `deliverables/*.npy` and `out/*projections*.csv`. With
`SHARED_STORE_BUCKET` set it pulls the newest build from R2/S3 (via
`shared_store`); unset, it uses whatever is on local disk. Seed R2 with
`python ../scripts/push_artifacts.py`.

## Dependencies

`service/requirements.txt` is intentionally light (`numpy/scipy/pandas` +
FastAPI + boto3) — **no** xgboost/sklearn/pybaseball. Those are only needed by
the heavy pipeline, which the worker launches as a subprocess using the repo's
root `requirements.txt`.
