"""runner.py — the classic "Run simulation" path, ported out of app.py as a
pure function the worker can call.

This is a faithful port of the Streamlit Run handler (app.py:2271-2635, classic
branch): build the pool from the slate ∩ sims, tilt candidates to projected
value / better offenses, optionally apply the stack-ownership ceiling boost,
build an ownership-weighted field, score both against the correlated sims, and
run the contest to get each candidate's Win% / Top10% / Top100% / place
distribution.

Only two helpers live in app.py (not an importable module), so they are ported
here: ``run_contest_dist`` (verbatim) and a headless ``build_many`` (the
Streamlit progress-bar version minus the UI). Everything else is imported from
the existing modules with no change to the numeric logic.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from stage_d import (build_pool, score_matrix, lineups_to_df,
                     norm as normname)
from mlb_lineup_builder import Pool, Builder, candidate_stack_structures
from field_simulator import (normalize_to_slots, adjust_ownership,
                             beta_for_size, tilt_structures)
from stack_signal import team_stack_ownership, apply_stack_ownership_boost
import portfolio_ev as pev


# --------------------------------------------------------------------------- #
# Tuning params (mirror the Setup-tab form; defaults match app.py exactly)
# --------------------------------------------------------------------------- #
@dataclass
class RunParams:
    contest_size: int = 6000
    sim_runs: int = 10000          # K — clipped to available n_sim
    num_candidates: int = 5000
    # Advanced field model
    medium: int = 6000
    chalk: float = 0.35            # chalk sensitivity
    tilt: float = 0.15             # stack-shape tilt (field)
    seed_field: int = 101
    seed_cand: int = 2025
    talent_tilt: float = 0.7
    team_tilt: float = 0.6
    cand_jitter: float = 0.0
    stack_boost: float = 0.05
    stack_aggr: float = 0.8
    bringback: float = 0.3
    game_stack: float = 0.5
    order_tilt: float = 0.15
    ace_pitcher: float = 0.6

    @classmethod
    def from_dict(cls, d: dict) -> "RunParams":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# --------------------------------------------------------------------------- #
# app.py-only helpers, ported
# --------------------------------------------------------------------------- #
def run_contest_dist(field_mat, cand_mat, n_sim, n_field, nbins=24,
                     cut_places=None):
    """Verbatim port of app.py:run_contest_dist. Scores each candidate against
    the field per sim; returns (wins, t10, t100, avg, dist) with a compact
    finishing-place histogram and, if cut_places is given, the field's per-sim
    placement ladder for payout-aware export."""
    N = cand_mat.shape[1]
    wins = np.zeros(N, np.int64); t10 = np.zeros(N, np.int64)
    t100 = np.zeros(N, np.int64); ps = np.zeros(N, np.int64)
    best = np.full(N, n_field + 1, np.int64); worst = np.zeros(N, np.int64)

    nb_target = max(6, min(int(nbins), int(n_field)))
    edges = np.unique(np.linspace(1, n_field + 1, nb_target + 1).astype(np.int64))
    nb = len(edges) - 1
    counts = np.zeros((N, nb), np.int32)
    idx = np.arange(N)

    cut_scores = None
    if cut_places is not None and len(cut_places):
        cut_places = np.asarray(cut_places, np.int64)
        cut_scores = np.empty((n_sim, len(cut_places)), np.float32)
        take = n_field - cut_places

    for s in range(n_sim):
        fs = np.sort(field_mat[s]); cv = cand_mat[s]
        pl = (n_field - np.searchsorted(fs, cv, side="right")) + 1
        wins += (pl == 1); t10 += (pl <= 10); t100 += (pl <= 100); ps += pl
        best = np.minimum(best, pl); worst = np.maximum(worst, pl)
        b = np.clip(np.searchsorted(edges, pl, side="right") - 1, 0, nb - 1)
        np.add.at(counts, (idx, b), 1)
        if cut_scores is not None:
            cut_scores[s] = fs[take]
    dist = {"edges": edges, "counts": counts, "best": best, "worst": worst,
            "mean": ps / n_sim}
    if cut_scores is not None:
        dist["field_cut_scores"] = cut_scores
        dist["cut_places"] = cut_places
    return wins, t10, t100, ps / n_sim, dist


def build_many(builder, target, hard_cap_mult=60):
    """Headless port of app.py:build_many — build `target` lineups from a
    Builder (no Streamlit progress bar). Returns (lineups, attempts)."""
    out, attempts = [], 0
    cap = target * hard_cap_mult + 500
    while len(out) < target and attempts < cap:
        lu = builder.build_one()
        attempts += 1
        if lu is not None:
            out.append(lu)
    return out, attempts


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
class RunError(Exception):
    """Raised for user-actionable run failures (bad slate, tiny pool, …)."""


def run_slate(dk_df: pd.DataFrame, sims: dict, params: RunParams,
              stack_params: dict, id_map: dict | None = None,
              order_map: dict | None = None, log=lambda m: None):
    """Execute the classic contest build+sim for a slate.

    Args:
        dk_df: FullName, Team, Position, Salary, Ownership (the slate ∩ universe).
        sims: the warm bundle {"H","P","score","n_sim"} from service.sims.
        params: RunParams (the tuning form).
        stack_params: field_params.json contents (stack_structures, etc).
        id_map: norm(name)->{TEAM->DK id} for later export (stored, not used here).
        order_map: norm(name)->batting order slot (0/absent => order-blind).

    Returns (summary, payload). `summary` is JSON-safe (metrics + result rows);
    `payload` holds the numpy arrays kept server-side for Results/Export.
    """
    order_map = order_map or {}
    H, P = sims["H"], sims["P"]
    score, n_sim = sims["score"], sims["n_sim"]

    K = min(int(params.sim_runs), int(n_sim))
    # sim index is aligned across players — a prefix slice preserves correlation.
    score_k = {k: v[:K] for k, v in score.items()}
    simnames = set(score_k)
    matched = int(dk_df["FullName"].map(lambda n: normname(n) in simnames).sum())
    if matched == 0:
        raise RunError("None of the slate players matched the sim universe.")

    # ---- pool (build_pool reads a CSV; write the slate to a temp file) ----
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     newline="") as tf:
        dk_df.to_csv(tf.name, index=False)
        tmp_csv = tf.name
    try:
        pool = build_pool(tmp_csv, H, P, score_k)
    finally:
        os.unlink(tmp_csv)

    nh = pool[pool.Pos != "P"].Name.nunique()
    npi = pool[pool.Pos == "P"].Name.nunique()
    nt = pool.Team.nunique()
    if npi < 2 or nh < 8:
        raise RunError(
            f"Pool too small ({npi} starters, {nh} hitters; need >=2 and >=8). "
            f"Slate matched {matched} of {len(dk_df)} players to the sims — the "
            "sims may be stale for this slate (Force refresh) or names/teams "
            "may not match.")
    log(f"Pool: {nh} hitters + {npi} starters across {nt} teams "
        f"({matched}/{len(dk_df)} slate players matched).")

    # ---- stack-ownership ceiling boost (same boosted sims score field+cands) ----
    score_b = score_k
    if float(params.stack_boost) > 0:
        hit_pool = pool[pool.Pos != "P"]
        names_by_team, own_by_name = {}, {}
        for r in hit_pool.itertuples():
            nn = normname(r.Name)
            names_by_team.setdefault(r.Team, set()).add(nn)
            own_by_name[nn] = float(r.Ownership)
        names_by_team = {t: sorted(ns) for t, ns in names_by_team.items()}
        stack_own = team_stack_ownership(names_by_team, own_by_name)
        score_b = apply_stack_ownership_boost(
            score_k, names_by_team, stack_own, K,
            strength=float(params.stack_boost), quantile=0.80)

    # ---- candidate value tilts (z-score softmax of talent / ceiling) ----
    cdf = pool[(pool.Pos != "P") | (pool.Role == "SP")].copy()
    tal, cel = {}, {}
    for nm in cdf["Name"].unique():
        a = score_k.get(normname(nm))
        if a is not None and len(a):
            tal[nm] = 0.5 * float(np.mean(a)) + 0.5 * float(np.percentile(a, 90))
            cel[nm] = float(np.percentile(a, 90))
    base = float(np.median(list(tal.values()))) if tal else 1.0

    def zmap(names, vmap=None):
        vmap = tal if vmap is None else vmap
        vals = np.array([vmap[n] for n in names if n in vmap], float)
        if len(vals) == 0:
            return {}
        mu, sd = float(vals.mean()), float(vals.std()) + 1e-9
        return {n: (vmap.get(n, mu) - mu) / sd for n in names}

    hset = set(cdf[cdf["Pos"] != "P"]["Name"])
    tt = float(params.talent_tilt)
    if tt > 0:
        zh = zmap(hset); zp = zmap(set(cdf[cdf["Pos"] == "P"]["Name"]))
        cdf["Ownership"] = [
            float(np.exp(tt * (zp if r.Pos == "P" else zh).get(r.Name, 0.0)))
            for r in cdf.itertuples()]
        zc = zmap(hset, cel); zcp = zmap(set(cdf[cdf["Pos"] == "P"]["Name"]), cel)
        cdf["Upside"] = [
            float(np.exp(tt * (zcp if r.Pos == "P" else zc).get(r.Name, 0.0)))
            for r in cdf.itertuples()]
    else:
        cdf["Ownership"] = 1.0
        cdf["Upside"] = 1.0

    cdf["Order"] = [float(order_map.get(normname(n), 0)) for n in cdf["Name"]]

    team_weights = None
    if float(params.team_tilt) > 0:
        hit = cdf[cdf["Pos"] != "P"]
        tteam = hit.groupby("Team")["Name"].apply(
            lambda s: sum(tal.get(n, base) for n in s)).to_dict()
        vals = np.array(list(tteam.values()), float)
        mu, sd = float(vals.mean()), float(vals.std()) + 1e-9
        team_weights = {t: float(np.exp(float(params.team_tilt) * (v - mu) / sd))
                        for t, v in tteam.items()}

    cand_params = dict(stack_params)
    cand_params["stack_structures"] = candidate_stack_structures(
        stack_params["stack_structures"], float(params.stack_aggr))
    cb = Builder(Pool(cdf), cand_params, seed=int(params.seed_cand), uniform=True,
                 team_weights=team_weights, jitter=float(params.cand_jitter),
                 upside_attr="Upside", bringback_prob=float(params.bringback),
                 game_stack_prob=float(params.game_stack),
                 order_weight=float(params.order_tilt),
                 ace_pitcher_prob=float(params.ace_pitcher))
    log(f"Developing {int(params.num_candidates):,} candidate lineups…")
    cands, _ = build_many(cb, int(params.num_candidates))
    if not cands:
        raise RunError("Failed to construct any valid candidate lineup from this pool.")
    cand_mat = score_matrix(cands, score_b, K)

    # ---- ownership-weighted field ----
    contest_size = int(params.contest_size)
    log(f"Building an ownership-weighted field of {contest_size:,}…")
    beta = beta_for_size(contest_size, int(params.medium), float(params.chalk))
    fdf = adjust_ownership(normalize_to_slots(pool, 0.15), beta=beta)
    tilted = tilt_structures(
        [(tuple(s), w) for s, w in stack_params["stack_structures"]],
        contest_size, int(params.medium), float(params.tilt))
    fp = dict(stack_params)
    fp["stack_structures"] = [(list(s), w) for s, w in tilted]
    fb = Builder(Pool(fdf), fp, seed=int(params.seed_field), uniform=False)
    field, _ = build_many(fb, contest_size)
    field_short = len(field) < contest_size
    field_mat = score_matrix(field, score_b, K)

    # ---- the contest ----
    log(f"Simulating the contest over {K:,} runs…")
    cut_places = pev.field_place_cutpoints(len(field))
    wins, t10, t100, avg, dist = run_contest_dist(
        field_mat, cand_mat, K, len(field), cut_places=cut_places)

    # ---- per-lineup attributes ----
    own_map = {normname(rr.FullName): float(rr.Ownership) for rr in dk_df.itertuples()}
    cand_players = [frozenset(pl.Name for pl in lu["players"]) for lu in cands]
    prim_team, prim_size, own_sum = [], [], []
    for lu in cands:
        if lu["teams"]:
            pt, ps_ = max(lu["teams"].items(), key=lambda kv: kv[1])
        else:
            pt, ps_ = "", 0
        prim_team.append(pt); prim_size.append(int(ps_))
        own_sum.append(round(sum(own_map.get(normname(pl.Name), 0.0)
                                 for pl in lu["players"]), 1))

    res = lineups_to_df(cands)
    res.insert(0, "Candidate", np.arange(1, len(cands) + 1))
    res["Wins"] = wins
    res["Win%"] = np.round(100 * wins / K, 3)
    res["Top10"] = t10
    res["Top10%"] = np.round(100 * t10 / K, 2)
    res["Top100"] = t100
    res["Top100%"] = np.round(100 * t100 / K, 2)
    res["AvgPlace"] = np.round(avg, 1)
    res["BestPlace"] = dist["best"]
    res["WorstPlace"] = dist["worst"]
    res["OwnSum"] = own_sum
    res["PrimaryTeam"] = prim_team
    res["PrimaryStack"] = prim_size
    res = res.sort_values(["Wins", "Top10", "Top100", "AvgPlace"],
                          ascending=[False, False, False, True]).reset_index(drop=True)
    res["Rank"] = np.arange(1, len(res) + 1)   # global rank in the sorted order

    players_meta = {}
    for r in pool.itertuples():
        nm = r.Name
        if nm in players_meta:
            continue
        players_meta[nm] = {
            "pos": "P" if r.Pos == "P" else r.Pos,
            "salary": int(r.Salary), "team": r.Team,
            "proj": float(tal[nm]) if nm in tal else None}

    pool_norm = {normname(n) for n in pool["Name"].unique()}
    score_pool = {k: np.asarray(v, np.float32)
                  for k, v in score_b.items() if k in pool_norm}

    payload = {
        "res": res, "cands": cands, "field_df": lineups_to_df(field),
        "K": K, "contest_size": contest_size, "field_n": len(field), "beta": beta,
        "dist": dist, "id_map": id_map or {}, "score_pool": score_pool,
        "cand_to_players": {i + 1: cand_players[i] for i in range(len(cands))},
        "pool_players": sorted({pl.Name for lu in cands for pl in lu["players"]}),
        "players_meta": players_meta,
    }
    summary = {
        "K": K, "contest_size": contest_size, "field_n": len(field),
        "field_short": field_short, "beta": round(float(beta), 3),
        "n_candidates": len(cands),
        "pool": {"hitters": int(nh), "starters": int(npi), "teams": int(nt),
                 "matched": matched, "slate_players": int(len(dk_df))},
        "metrics": _headline_metrics(res, K),
        "results": _results_rows(res),
        "columns": list(res.columns),
    }
    return summary, payload


def _headline_metrics(res: pd.DataFrame, K: int) -> dict:
    """The four Results-tab headline numbers (best candidate by each objective)."""
    if res.empty:
        return {}
    return {
        "best_win_pct": float(res["Win%"].max()),
        "best_top10_pct": float(res["Top10%"].max()),
        "best_top100_pct": float(res["Top100%"].max()),
        "candidates_with_a_win": int((res["Wins"] > 0).sum()),
    }


# Columns returned to the client for the results table (JSON-safe subset).
_RESULT_COLS = ["Rank", "Candidate", "Stack", "Salary", "PrimaryTeam",
                "PrimaryStack", "OwnSum", "Win%", "Top10%", "Top100%",
                "AvgPlace", "BestPlace", "WorstPlace"]
# Showdown result rows key on Captain / team Split instead of Stack columns.
_RESULT_COLS_SHOWDOWN = ["Rank", "Candidate", "Captain", "CptTeam", "Split",
                         "Salary", "OwnSum", "Win%", "Top10%", "Top100%",
                         "AvgPlace", "BestPlace", "WorstPlace"]


def _result_cols_for(res: pd.DataFrame) -> list[str]:
    base = _RESULT_COLS_SHOWDOWN if "Captain" in res.columns else _RESULT_COLS
    return [c for c in base if c in res.columns]


def _results_rows(res: pd.DataFrame, limit: int = 500) -> list[dict]:
    cols = _result_cols_for(res)
    out = res[cols].head(limit)
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in out.to_dict("records")
    ]


# --------------------------------------------------------------------------- #
# Results filtering (Phase 2) — server-side over the cached payload, so the
# include/exclude-player filters can use each lineup's membership set.
# Mirrors the mask logic in app.py's Results tab.
# --------------------------------------------------------------------------- #
def facets(payload: dict) -> dict:
    """Filter-control options for a run: player pool, ownership/salary ranges,
    and the format-specific dimensions (classic: stack shape / primary team /
    size; showdown: captain / team split)."""
    res = payload["res"]
    out = {
        "format": payload.get("format", "classic"),
        "pool_players": payload["pool_players"],
        "own_sum": {"min": float(res["OwnSum"].min()),
                    "max": float(res["OwnSum"].max())},
        "salary": {"min": int(res["Salary"].min()),
                   "max": int(res["Salary"].max())},
        "n_candidates": int(len(res)),
    }
    if payload.get("format") == "showdown":
        out["captains"] = payload.get("captains") or sorted(res["Captain"].unique().tolist())
        out["splits"] = sorted(res["Split"].unique().tolist())
    else:
        out["stacks"] = sorted(res["Stack"].unique().tolist())
        out["teams"] = sorted(t for t in res["PrimaryTeam"].unique().tolist() if t)
        out["sizes"] = sorted(res["PrimaryStack"].unique().tolist(), reverse=True)
    return out


def filter_results(payload: dict, f: dict, limit: int = 1000) -> dict:
    """Apply the Results-tab filters to a run's candidates. Returns matching
    rows (up to `limit`), the total match count, and every matching Candidate id
    (so the UI's 'mark all' covers matches beyond the returned page)."""
    import numpy as _np
    res = payload["res"]
    c2p = payload["cand_to_players"]
    mask = _np.ones(len(res), dtype=bool)
    cand = res["Candidate"]

    players = set(f.get("players") or [])
    if players:
        mode = f.get("match_mode", "all")
        mask &= cand.map(
            lambda c: (players.issubset(c2p[int(c)]) if mode == "all"
                       else bool(players & c2p[int(c)]))).to_numpy()
    exclude = set(f.get("exclude") or [])
    if exclude:
        mask &= cand.map(lambda c: not (exclude & c2p[int(c)])).to_numpy()
    if f.get("stacks") and "Stack" in res.columns:
        mask &= res["Stack"].isin(f["stacks"]).to_numpy()
    if f.get("teams") and "PrimaryTeam" in res.columns:
        mask &= res["PrimaryTeam"].isin(f["teams"]).to_numpy()
    if f.get("sizes") and "PrimaryStack" in res.columns:
        mask &= res["PrimaryStack"].isin(f["sizes"]).to_numpy()
    # showdown dimensions
    if f.get("captains") and "Captain" in res.columns:
        mask &= res["Captain"].isin(f["captains"]).to_numpy()
    if f.get("splits") and "Split" in res.columns:
        mask &= res["Split"].isin(f["splits"]).to_numpy()
    if f.get("own_min") is not None:
        mask &= (res["OwnSum"] >= float(f["own_min"])).to_numpy()
    if f.get("own_max") is not None:
        mask &= (res["OwnSum"] <= float(f["own_max"])).to_numpy()
    if f.get("sal_min") is not None:
        mask &= (res["Salary"] >= float(f["sal_min"])).to_numpy()
    if f.get("sal_max") is not None:
        mask &= (res["Salary"] <= float(f["sal_max"])).to_numpy()
    if f.get("min_win"):
        mask &= (res["Win%"] >= float(f["min_win"])).to_numpy()
    if f.get("min_top10"):
        mask &= (res["Top10%"] >= float(f["min_top10"])).to_numpy()
    if f.get("min_top100"):
        mask &= (res["Top100%"] >= float(f["min_top100"])).to_numpy()

    fres = res[mask]
    all_ids = [int(x) for x in fres["Candidate"].tolist()]
    return {"total": int(len(fres)), "count": min(len(fres), limit),
            "all_ids": all_ids, "results": _results_rows(fres, limit=limit),
            "columns": _result_cols_for(res)}
