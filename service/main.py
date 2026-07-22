"""main.py — the DFS worker FastAPI application.

Scaffold scope (Phase 0): the read path is wired end-to-end —
``GET /players`` and ``GET /players/{name}/distribution`` serve the warm sim
artifacts. The heavy pipeline job endpoints (``POST /refresh`` …) and the
remaining light endpoints (``/run``, ``/export/*``, showdown) are stubbed with
501 responses and a documented contract, to be filled in during the phased
port (see ARCHITECTURE.md).

Run locally:
    cd <repo root>
    pip install -r service/requirements.txt
    uvicorn service.main:app --reload --port 8000
"""
import os
import sys

# Make the repo root importable so the service can reuse the existing numeric
# modules (stage_d, mlb_lineup_builder, portfolio, field_simulator, …).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from service import sims  # noqa: E402

app = FastAPI(
    title="DFS Sims Worker",
    version="0.1.0",
    description="Warm numeric API + heavy-pipeline jobs for the DFS simulator.",
)

# CORS: the Next.js app calls this either directly (dev) or via its server-side
# proxy (prod). Allowed origins are env-driven; default permissive for local dev.
_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Response models (become the source of truth for generated TS types)
# --------------------------------------------------------------------------- #
class HealthOut(BaseModel):
    status: str
    remote_store: bool


class StatusOut(BaseModel):
    n_sim: int | None = None
    hitters: int = 0
    pitchers: int = 0
    remote_store: bool = False
    build_stamp: dict = {}


class PlayerRow(BaseModel):
    Player: str
    Type: str
    Proj: float
    # Threshold columns carry spaces/symbols; expose the raw dict downstream.

    model_config = {"extra": "allow"}


class DistBin(BaseModel):
    x: float
    count: int


class PlayerDistOut(BaseModel):
    player: str
    n_sim: int
    mean: float
    p10: float
    median: float
    p90: float
    bins: list[DistBin]


# --------------------------------------------------------------------------- #
# Health / status
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthOut)
def health():
    """Liveness probe — does not touch the (possibly large) sim arrays."""
    from service import artifacts
    return {"status": "ok", "remote_store": artifacts.remote_enabled()}


@app.get("/status", response_model=StatusOut)
def status():
    """Freshness + inventory of the currently loaded artifacts."""
    try:
        return sims.status()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# --------------------------------------------------------------------------- #
# Players (wired end-to-end)
# --------------------------------------------------------------------------- #
@app.get("/players")
def players(
    kind: str = Query("all", pattern="^(all|hitters|pitchers)$"),
    search: str = Query("", description="case-insensitive substring filter"),
):
    """Per-player DK-point threshold table from the current sims.

    Mirrors app.py:cached_player_table. ``kind`` and ``search`` apply the same
    filters the Streamlit Players tab exposes (Show selector + search box)."""
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


@app.get("/players/{name}/distribution", response_model=PlayerDistOut)
def player_distribution(name: str, nbins: int = Query(40, ge=5, le=200)):
    """Histogram + summary of one player's simulated DK scores
    (data behind app.py:player_score_chart)."""
    try:
        dist = sims.player_distribution(name, nbins=nbins)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if dist is None:
        raise HTTPException(status_code=404, detail=f"No sims for player '{name}'")
    return dist


# --------------------------------------------------------------------------- #
# Stubs — contracts for the phased port (return 501 until implemented)
# --------------------------------------------------------------------------- #
_STUBS = {
    "POST /run": "Field build + run_contest_dist -> run summary + run_id (Phase 1/2).",
    "GET /slate/catalog": "dk_slate_feed.build_catalog (Phase 1).",
    "GET /slate/team-totals": "slate_team_totals (Phase 1).",
    "POST /results/filter": "Filtered candidate results by run_id (Phase 2).",
    "POST /export/dk-upload": "build_dk_upload / rows_to_upload_csv (Phase 3).",
    "POST /export/ev": "build_dk_upload_ev + portfolio EV (Phase 3).",
    "POST /refresh": "Launch Stage B/C as a background job; poll /refresh/status (Phase 5).",
}


@app.get("/roadmap")
def roadmap():
    """The not-yet-implemented endpoints and what they will wrap."""
    return {"implemented": ["/health", "/status", "/players",
                            "/players/{name}/distribution"],
            "planned": _STUBS}


class RunIn(BaseModel):
    # Placeholder shape for the Setup-tab params form; fleshed out in Phase 1.
    contest_size: int = 6000
    n_sim: int = 10000
    num_candidates: int = 10000


@app.post("/run")
def run(_: RunIn):
    raise HTTPException(status_code=501, detail=_STUBS["POST /run"])
