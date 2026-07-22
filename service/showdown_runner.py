"""showdown_runner.py — the showdown (1 CPT + 5 UTIL, single game) contest
build+sim, ported from app.py:run_showdown_sim.

Reuses the showdown backend modules (showdown_builder / showdown_contest /
showdown_portfolio / showdown_upload) unchanged, plus the shared headless
helpers in service.runner (build_many, run_contest_dist).
"""
import numpy as np

import showdown_builder as sb
import showdown_contest as sc
import portfolio_ev as pev
from stage_d import norm as normname
from field_simulator import beta_for_size

from service.runner import build_many, run_contest_dist, RunError, RunParams


def run_showdown(dk_df, sims: dict, params: RunParams, id_map=None,
                 log=lambda m: None):
    """Build a showdown contest and return (summary, payload). Payload carries
    format='showdown' and the arrays the Results/Export endpoints need."""
    score, n_sim = sims["score"], sims["n_sim"]
    K = min(int(params.sim_runs), int(n_sim))
    score_k = {k: v[:K] for k, v in score.items()}

    try:
        pool = sc.build_pool(dk_df, score_k, normfn=normname)
    except ValueError as e:
        raise RunError(str(e))
    teams = sorted(pool["Team"].unique())
    if len(pool) < sb.ROSTER_SIZE:
        raise RunError(f"Need at least {sb.ROSTER_SIZE} simmed players for a "
                       f"showdown roster (have {len(pool)}).")
    log(f"Showdown pool: {len(pool)} players — {teams[0]} vs {teams[1]}.")

    cb = sb.Builder(sb.Pool(pool), seed=int(params.seed_cand), uniform=True,
                    jitter=float(params.cand_jitter))
    cands, _ = build_many(cb, int(params.num_candidates))
    if not cands:
        raise RunError("Could not build any showdown candidate lineup.")

    contest_size = int(params.contest_size)
    beta = beta_for_size(contest_size, int(params.medium), float(params.chalk))
    fp = sc._field_pool(pool, score_k, normname, sc.DEFAULT_CPT_CEILING_TILT)
    fb = sb.Builder(sb.Pool(fp), {"cpt_chalk": beta, "util_chalk": beta},
                    seed=int(params.seed_field), uniform=False)
    field, _ = build_many(fb, contest_size)
    if not field:
        raise RunError("Could not build a showdown field.")
    field_short = len(field) < contest_size

    cand_mat = sb.score_matrix(cands, score_k, K, norm=normname)
    field_mat = sb.score_matrix(field, score_k, K, norm=normname)

    log(f"Simulating the showdown contest over {K:,} runs…")
    cut_places = pev.field_place_cutpoints(len(field))
    wins, t10, t100, avg, dist = run_contest_dist(
        field_mat, cand_mat, K, len(field), cut_places=cut_places)

    own_map = {normname(r.FullName): float(r.Ownership) for r in dk_df.itertuples()}
    res = sb.lineups_to_df(cands)
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
    res["Captain"] = [lu["captain"].Name for lu in cands]
    res["CptTeam"] = [lu["captain"].Team for lu in cands]
    res["OwnSum"] = [round(sum(own_map.get(normname(pl.Name), 0.0)
                               for pl in lu["players"]), 1) for lu in cands]
    res = res.sort_values(["Wins", "Top10", "Top100", "AvgPlace"],
                          ascending=[False, False, False, True]).reset_index(drop=True)
    res["Rank"] = np.arange(1, len(res) + 1)

    pool_norm = {normname(n) for n in pool["Name"].unique()}
    score_pool = {k: np.asarray(v, np.float32)
                  for k, v in score_k.items() if k in pool_norm}
    tal, players_meta = {}, {}
    for nm in pool["Name"]:
        a = score_k.get(normname(nm))
        if a is not None and len(a):
            tal[nm] = 0.5 * float(np.mean(a)) + 0.5 * float(np.percentile(a, 90))
    for r in pool.itertuples():
        if r.Name not in players_meta:
            players_meta[r.Name] = {"pos": r.Pos, "salary": int(r.Salary),
                                    "team": r.Team, "proj": tal.get(r.Name)}

    payload = {
        "format": "showdown",
        "res": res, "cands": cands, "field": field,
        "field_df": sb.lineups_to_df(field),
        "K": K, "contest_size": contest_size, "field_n": len(field), "beta": beta,
        "dist": dist, "id_map": id_map or {}, "score_pool": score_pool,
        "cand_to_players": {i + 1: frozenset(pl.Name for pl in lu["players"])
                            for i, lu in enumerate(cands)},
        "pool_players": sorted({pl.Name for lu in cands for pl in lu["players"]}),
        "players_meta": players_meta,
        "captains": sorted({lu["captain"].Name for lu in cands}),
        "teams": teams,
    }
    from service.runner import _headline_metrics, _results_rows
    summary = {
        "format": "showdown",
        "K": K, "contest_size": contest_size, "field_n": len(field),
        "field_short": field_short, "beta": round(float(beta), 3),
        "n_candidates": len(cands),
        "teams": teams,
        "pool": {"players": int(len(pool)), "teams": len(teams),
                 "captains": int(res["Captain"].nunique())},
        "metrics": _headline_metrics(res, K),
        "results": _results_rows(res, limit=500),
        "columns": list(res.columns),
    }
    return summary, payload
