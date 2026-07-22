"""main.py — the DFS worker FastAPI application.

Read path (Phase 0): /status, /players, /players/{name}/distribution.
Run path (Phase 1): /slate/catalog, /slate/upload, /slate/sample,
/slate/team-totals, POST /run (+ /run/{id} and place-distribution), and the
async refresh job (POST /refresh, GET /refresh/status/{id}).

Run locally (from repo root):
    pip install -r service/requirements.txt
    uvicorn service.main:app --reload --port 8000
"""
import json
import os
import sys
import time

# Make the repo root importable so the service reuses the existing numeric
# modules (stage_d, mlb_lineup_builder, portfolio, field_simulator, …).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, UploadFile, File  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import portfolio_ev as pev  # noqa: E402
from service import sims, runstore, jobs  # noqa: E402
from service.runner import RunParams, run_slate, RunError  # noqa: E402

app = FastAPI(
    title="DFS Sims Worker",
    version="0.2.0",
    description="Warm numeric API + heavy-pipeline jobs for the DFS simulator.",
)

_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PARAMS_PATH = os.path.join(_REPO_ROOT, "field_params.json")
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
SIZE_PRESETS = [150, 1000, 6000, 20000, 50000, 150000]

# Small in-process caches (catalog is a live fetch; uploads keep id_map server-side).
_catalog_cache = {"ts": 0.0, "data": None}
_uploaded_slates: dict[str, dict] = {}
_upload_seq = 0


def _stack_params():
    with open(_PARAMS_PATH) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Health / status
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    from service import artifacts
    return {"status": "ok", "remote_store": artifacts.remote_enabled()}


@app.get("/status")
def status():
    try:
        return sims.status()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/params/defaults")
def params_defaults():
    """The Setup-tab form defaults + presets, so the UI stays in sync with the
    worker's RunParams (single source of truth)."""
    from dataclasses import asdict
    return {"defaults": asdict(RunParams()), "size_presets": SIZE_PRESETS,
            "sim_runs_max": (sims.status() or {}).get("n_sim")}


# --------------------------------------------------------------------------- #
# Players (Phase 0)
# --------------------------------------------------------------------------- #
@app.get("/players")
def players(kind: str = Query("all", pattern="^(all|hitters|pitchers)$"),
            search: str = Query("")):
    try:
        rows = sims.player_table()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if kind == "hitters":
        rows = [r for r in rows if r["Type"] == "Hitter"]
    elif kind == "pitchers":
        rows = [r for r in rows if r["Type"] == "Pitcher"]
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r["Player"].lower()]
    return {"count": len(rows), "players": rows}


@app.get("/players/{name}/distribution")
def player_distribution(name: str, nbins: int = Query(40, ge=5, le=200)):
    try:
        dist = sims.player_distribution(name, nbins=nbins)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if dist is None:
        raise HTTPException(status_code=404, detail=f"No sims for player '{name}'")
    return dist


# --------------------------------------------------------------------------- #
# Slate sources
# --------------------------------------------------------------------------- #
@app.get("/slate/catalog")
def slate_catalog(refresh: bool = False):
    """Today's pickable DraftKings slates from the RotoWire feeds (live, cached
    10 min). Mirrors app.py:_load_slate_catalog."""
    now = time.time()
    if not refresh and _catalog_cache["data"] and now - _catalog_cache["ts"] < 600:
        return _catalog_cache["data"]
    try:
        import dk_slate_feed
        cat = dk_slate_feed.build_catalog()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch the live slate feed ({type(e).__name__}: {e}). "
                   "Upload a slate CSV or use /slate/sample instead.")
    # Trim player lists from the catalog listing (kept server-side for /run).
    slates = [{k: v for k, v in s.items() if k != "players"} for s in cat["slates"]]
    for s, full in zip(slates, cat["slates"]):
        _uploaded_slates[f"cat:{full['slate_id']}"] = {
            "kind": "catalog", "slate": full}
    out = {"date": cat["date"], "slates": slates}
    _catalog_cache.update(ts=now, data=out)
    return out


@app.post("/slate/upload")
async def slate_upload(file: UploadFile = File(...)):
    """Parse an uploaded slate CSV (raw DK export or clean CSV) and stash it
    server-side (keeping the id_map for later export). Returns a slate_token to
    pass to /run, plus a preview."""
    global _upload_seq
    raw = (await file.read()).decode("utf-8", "replace")
    from service import slate_parse
    try:
        df, idmap = slate_parse.parse_slate_csv(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _upload_seq += 1
    token = f"up_{_upload_seq:04d}"
    _uploaded_slates[token] = {"kind": "upload", "df": df, "id_map": idmap}
    return {"slate_token": token, "n_players": int(len(df)),
            "teams": int(df["Team"].nunique()),
            "has_ownership": bool(df["Ownership"].abs().sum() > 0),
            "preview": df.head(10).to_dict("records")}


@app.get("/slate/sample")
def slate_sample():
    """A bundled sample slate (sample_ownership.csv) so the UI is runnable
    without the live feed — handy for dev/demo. Returns a slate_token."""
    global _upload_seq
    path = os.path.join(_REPO_ROOT, "sample_ownership.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="sample_ownership.csv not found")
    df = pd.read_csv(path)
    _upload_seq += 1
    token = f"sample_{_upload_seq:04d}"
    _uploaded_slates[token] = {"kind": "sample", "df": df, "id_map": {}}
    return {"slate_token": token, "n_players": int(len(df)),
            "teams": int(df["Team"].nunique()), "has_ownership": True,
            "preview": df.head(10).to_dict("records")}


@app.get("/slate/team-totals")
def slate_team_totals():
    """Vegas-implied team totals for today's slate (live). Mirrors
    app.py:slate_team_totals; used by the Setup team-totals editor."""
    try:
        import slate_ingest
        slate = slate_ingest.build_slate(write=False)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch the live slate/Vegas feed ({type(e).__name__}).")
    totals = {}
    for g in slate.get("games", []):
        for side in ("home", "away"):
            t = g.get(side)
            if t and t.get("team"):
                totals[t["team"]] = t.get("team_total")
    return {"date": slate.get("date"), "totals": totals}


def _resolve_slate(body: "RunIn"):
    """Turn a run request's slate reference into (dk_df, id_map)."""
    if body.slate_token:
        rec = _uploaded_slates.get(body.slate_token)
        if not rec:
            raise HTTPException(status_code=404,
                                detail="Unknown slate_token (expired or wrong id).")
        if rec["kind"] == "catalog":
            import dk_slate_feed
            return dk_slate_feed.to_dk_df(rec["slate"])
        return rec["df"], rec.get("id_map", {})
    if body.slate_id:
        rec = _uploaded_slates.get(f"cat:{body.slate_id}")
        if not rec:
            raise HTTPException(
                status_code=404,
                detail="Unknown slate_id — call /slate/catalog first.")
        import dk_slate_feed
        return dk_slate_feed.to_dk_df(rec["slate"])
    if body.players:
        df = pd.DataFrame([p.model_dump() for p in body.players])
        return df, {}
    raise HTTPException(status_code=422,
                        detail="Provide slate_token, slate_id, or players.")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
class SlatePlayer(BaseModel):
    FullName: str
    Team: str
    Position: str
    Salary: int = 0
    Ownership: float = 0.0


class RunIn(BaseModel):
    slate_token: str | None = None
    slate_id: str | None = None
    players: list[SlatePlayer] | None = None
    params: dict = {}


@app.post("/run")
def run(body: RunIn):
    """Build candidates + an ownership-weighted field and simulate the contest
    over the correlated sims. Returns a run summary + run_id; the full payload
    (arrays) is kept server-side for Results/Export. Port of the classic Run
    handler in app.py."""
    try:
        bundle = sims._load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    dk_df, id_map = _resolve_slate(body)
    if "Ownership" not in dk_df.columns:
        dk_df["Ownership"] = 0.0

    params = RunParams.from_dict(body.params)
    log_lines: list[str] = []
    t0 = time.time()
    try:
        summary, payload = run_slate(
            dk_df,
            {"H": bundle["H"], "P": bundle["P"],
             "score": bundle["score"], "n_sim": bundle["n_sim"]},
            params, _stack_params(), id_map=id_map,
            log=log_lines.append)
    except RunError as e:
        raise HTTPException(status_code=422, detail=str(e))
    run_id = runstore.put(payload)
    summary["run_id"] = run_id
    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["log"] = log_lines
    summary["params"] = {**RunParams.from_dict(body.params).__dict__}
    return summary


@app.get("/run/{run_id}")
def get_run(run_id: str, limit: int = Query(500, ge=1, le=5000)):
    """Re-fetch a stored run's summary + result rows (for the Results tab)."""
    from service import runner
    payload = runstore.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found (expired).")
    res = payload["res"]
    return {
        "run_id": run_id, "K": payload["K"], "contest_size": payload["contest_size"],
        "field_n": payload["field_n"], "beta": round(float(payload["beta"]), 3),
        "n_candidates": len(payload["cands"]),
        "metrics": runner._headline_metrics(res, payload["K"]),
        "results": runner._results_rows(res, limit=limit),
        "columns": list(res.columns),
    }


@app.get("/run/{run_id}/facets")
def run_facets(run_id: str):
    """Filter-control options for the Results tab (players, stacks, teams,
    sizes, ownership/salary ranges)."""
    from service import runner
    payload = runstore.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found (expired).")
    return runner.facets(payload)


class ResultsFilter(BaseModel):
    players: list[str] = []
    match_mode: str = "all"           # "all" | "any"
    exclude: list[str] = []
    stacks: list[str] = []
    teams: list[str] = []
    sizes: list[int] = []
    own_min: float | None = None
    own_max: float | None = None
    sal_min: float | None = None
    sal_max: float | None = None
    min_win: float = 0.0
    min_top10: float = 0.0
    min_top100: float = 0.0
    limit: int = 1000


@app.post("/run/{run_id}/results")
def run_results(run_id: str, f: ResultsFilter):
    """Filtered candidate results (server-side, mirrors app.py's Results mask)."""
    from service import runner
    payload = runstore.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found (expired).")
    limit = max(1, min(int(f.limit), 5000))
    out = runner.filter_results(payload, f.model_dump(), limit=limit)
    out["run_id"] = run_id
    return out


def _csv_response(df, filename: str):
    from fastapi.responses import Response
    return Response(
        content=df.to_csv(index=False),
        media_type="text/csv",
        headers={"content-disposition": f'attachment; filename="{filename}"'})


@app.get("/run/{run_id}/candidates.csv")
def run_candidates_csv(run_id: str):
    """All candidate lineups (the app's 'Download all candidate lineups')."""
    payload = runstore.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found (expired).")
    return _csv_response(payload["res"], f"candidates_{len(payload['res'])}.csv")


@app.get("/run/{run_id}/field.csv")
def run_field_csv(run_id: str):
    """The simulated field (the app's 'Download field')."""
    payload = runstore.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found (expired).")
    return _csv_response(payload["field_df"], f"field_{payload['field_n']}.csv")


@app.get("/run/{run_id}/candidate/{candidate}/place-distribution")
def candidate_place_distribution(run_id: str, candidate: int):
    """Finishing-place histogram for one candidate (data behind
    app.py:place_distribution_chart). `candidate` is the 1-based Candidate id."""
    payload = runstore.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found (expired).")
    dist = payload["dist"]
    # dist arrays are in candidate BUILD order (candidate id 1..N -> index 0..N-1),
    # not the sorted-results order — mirrors app.py `cand_idx = chosen_cand - 1`.
    i = int(candidate) - 1
    if i < 0 or i >= len(dist["mean"]):
        raise HTTPException(status_code=404, detail="Candidate not in this run.")
    edges = dist["edges"].astype(float)
    counts = dist["counts"][i].astype(float)
    n = int(payload["K"])
    bins = [
        {"lo": float(edges[j]), "hi": float(edges[j + 1]),
         "pct": round(100 * counts[j] / max(1, n), 3), "sims": int(counts[j])}
        for j in range(len(counts))
    ]
    return {
        "run_id": run_id, "candidate": candidate, "field_n": payload["field_n"],
        "n_sim": n,
        "mean_place": float(dist["mean"][i]),
        "best_place": int(dist["best"][i]), "worst_place": int(dist["worst"][i]),
        "bins": bins,
    }


# --------------------------------------------------------------------------- #
# Export (Phase 3) — portfolio selection + DK upload + exposure
# --------------------------------------------------------------------------- #
class ExportIn(BaseModel):
    mode: str = "ranked"                  # "ranked" | "ev"
    n_select: int = 20
    candidate_ids: list[int] | None = None  # restrict to marked lineups
    sort_by: str = "Top100 Rate"          # ranked objective
    # global exposure / diversity caps (1.0 = no cap)
    hitter_cap: float = 1.0
    pitcher_cap: float = 1.0
    team_cap: float = 1.0
    pair_cap: float = 1.0
    core_cap: float = 1.0
    max_overlap: float = 1.0
    group_cap: float = 1.0
    use_value_groups: bool = False
    group_salary_tol: int = 300
    group_proj_tol: float = 1.5
    # per-entity caps/mins (name/team -> fraction)
    player_caps: dict[str, float] | None = None
    team_caps: dict[str, float] | None = None
    player_mins: dict[str, float] | None = None
    team_mins: dict[str, float] | None = None
    # EV params
    entry_fee: float = 20.0
    pct_paid: float = 0.20
    rake: float = 0.15
    top_heaviness: float = 0.9
    risk: str = "Balanced"
    shortlist: int = 1000


@app.get("/export/options")
def export_options():
    """Risk postures + ranked objectives for the Export UI."""
    return {"risk_postures": list(pev.UTILITIES.keys()),
            "risk_help": {k: v[1] for k, v in pev.UTILITIES.items()},
            "sort_by": ["Win%", "Top10 Rate", "Top100 Rate"]}


@app.post("/run/{run_id}/export")
def run_export(run_id: str, body: ExportIn):
    """Select a portfolio and package the DK upload + exposure (+ EV returns).
    Port of app.py's build_dk_upload / build_dk_upload_ev + exposure breakdown."""
    from service import exporter
    payload = runstore.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found (expired).")
    out = exporter.run_export(payload, body.model_dump())
    if "error" in out:
        raise HTTPException(status_code=422, detail=out["error"])
    out["run_id"] = run_id
    return out


# --------------------------------------------------------------------------- #
# Refresh (heavy pipeline as a background job)
# --------------------------------------------------------------------------- #
class RefreshIn(BaseModel):
    team_totals: dict[str, float] | None = None   # TEAM -> implied total override
    full: bool = False                            # True = Stage B+C (needs full deps)


@app.post("/refresh")
def refresh(body: RefreshIn):
    """Kick a background rebuild of the correlated sims (Stage C), or the full
    pipeline (Stage B+C) when `full`. Returns a job_id to poll. Team-total
    overrides are written for the sim rebuild to consume."""
    tt_path = None
    if body.team_totals:
        os.makedirs(_DATA_DIR, exist_ok=True)
        tt_path = os.path.join(_DATA_DIR, "team_totals_override.json")
        with open(tt_path, "w") as fh:
            json.dump(body.team_totals, fh)
    job_id = jobs.start_refresh(team_totals_path=tt_path, full=body.full)
    return {"job_id": job_id, "state": "queued", "full": body.full}


@app.get("/refresh/status/{job_id}")
def refresh_status(job_id: str):
    st = jobs.status(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return st
