"""exporter.py — the Export tab, ported out of app.py.

Selects a portfolio of N lineups from a cached run — either ranked
(``select_portfolio``) or payout-aware EV (``select_portfolio_ev``) — under the
same exposure / diversity caps as the Streamlit app, then produces the
DraftKings upload CSV (when the slate carries DK ids), the player/team exposure
breakdown, and (for EV) the per-slate portfolio return distribution.

Numeric selection reuses portfolio.py / portfolio_ev.py unchanged; only the
app.py-local id-resolution helpers (``_lineup_ids`` &c.) are ported here.
"""
import csv
import io
from collections import Counter

import numpy as np

import dk_ids
import portfolio_ev as pev
from portfolio import select_portfolio, select_portfolio_ev, detect_value_groups
from stage_d import COLS, HITC, SLOT, norm as normname
import showdown_builder as sb
from showdown_portfolio import (select_showdown_portfolio,
                                select_showdown_portfolio_ev)


# --------------------------------------------------------------------------- #
# id resolution (ported from app.py)
# --------------------------------------------------------------------------- #
def _split_cell(cell):
    s = str(cell)
    if s.endswith(")") and " (" in s:
        nm, tm = s.rsplit(" (", 1)
        return nm, tm[:-1]
    return s, ""


def _players_to_ids(players, dkid):
    ids = []
    for pl in players:
        cid = dk_ids.lookup(dkid, pl.Name, getattr(pl, "Team", ""),
                            pos=getattr(pl, "Pos", ""),
                            salary=getattr(pl, "Salary", None))
        if cid is None:
            return None
        ids.append(cid)
    return ids


def _row_players(row, cands):
    try:
        return cands[int(row["Candidate"]) - 1]["players"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _lineup_ids(row, dkid, cands=None):
    players = _row_players(row, cands) if cands is not None else None
    if players is not None:
        return _players_to_ids(players, dkid)
    ids = []
    for c in COLS:
        nm, tm = _split_cell(row[c])
        cid = dk_ids.lookup(dkid, nm, tm)
        if cid is None:
            return None
        ids.append(cid)
    return ids


def _row_names(row, cands):
    """Player names for a chosen row (via objects when available)."""
    players = _row_players(row, cands)
    if players is not None:
        return [pl.Name for pl in players]
    return [_split_cell(row[c])[0] for c in COLS]


def _lineup_display(row, cands):
    """Slot -> {slot, player, team} for the chosen-lineup view."""
    players = _row_players(row, cands)
    out = []
    if players is not None:
        for slot, pl in zip(SLOT, players):
            out.append({"slot": slot, "player": pl.Name,
                        "team": getattr(pl, "Team", "")})
    else:
        for slot, c in zip(SLOT, COLS):
            nm, tm = _split_cell(row[c])
            out.append({"slot": slot, "player": nm, "team": tm})
    return out


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
_SORT_KEYMAP = {"Win%": ["Wins", "Top10", "Top100"],
                "Top10 Rate": ["Top10", "Top100", "Wins"],
                "Top100 Rate": ["Top100", "Top10", "Wins"]}


def _caps(o):
    """Common exposure/diversity kwargs from the request dict."""
    return dict(
        hitter_cap=float(o.get("hitter_cap", 1.0)),
        pitcher_cap=float(o.get("pitcher_cap", 1.0)),
        team_cap=float(o.get("team_cap", 1.0)),
        pair_cap=float(o.get("pair_cap", 1.0)),
        core_cap=float(o.get("core_cap", 1.0)),
        max_overlap=float(o.get("max_overlap", 1.0)),
        group_cap=float(o.get("group_cap", 1.0)),
        player_caps=o.get("player_caps") or None,
        team_caps=o.get("team_caps") or None,
        player_mins=o.get("player_mins") or None,
        team_mins=o.get("team_mins") or None,
    )


def run_export(payload: dict, o: dict) -> dict:
    """Select and package the portfolio export. `o` is the request dict."""
    if payload.get("format") == "showdown":
        return _run_export_showdown(payload, o)
    res = payload["res"]
    cands = payload["cands"]
    dkid = payload.get("id_map") or {}
    has_ids = bool(dkid)

    # optional restriction to a marked subset (preserves ranked order)
    ids = o.get("candidate_ids")
    src = res
    if ids:
        keep = set(int(x) for x in ids)
        src = res[res["Candidate"].isin(keep)].reset_index(drop=True)
    if len(src) == 0:
        return {"error": "No candidates to export (empty selection)."}

    n_select = int(o.get("n_select", 20))

    # value groups (spread exposure across near-twin players)
    group_of = None
    if o.get("use_value_groups"):
        group_of, _ = detect_value_groups(
            payload["players_meta"],
            salary_tol=int(o.get("group_salary_tol", 300)),
            proj_tol=float(o.get("group_proj_tol", 1.5)))

    eligible = (lambda nms: all(dk_ids.has_name(dkid, n) for n in nms)) if has_ids else None
    caps = _caps(o)
    caps["group_of"] = group_of

    mode = o.get("mode", "ranked")
    extra: dict = {}

    if mode == "ev":
        ev = _export_ev(payload, src, n_select, dkid, eligible, caps, o)
        if ev is None:
            return {"error": "This run predates the payout ladder; use ranked "
                             "export (re-run to enable EV)."}
        chosen, info, W, W_naive, extra = ev
    else:
        sort_by = o.get("sort_by", "Top100 Rate")
        keymap = _SORT_KEYMAP.get(sort_by, _SORT_KEYMAP["Top100 Rate"])
        chosen, info = select_portfolio(
            src, n_select, keymap, cols=COLS, hitc=HITC,
            eligible=eligible, **caps)
        W = W_naive = None

    # DK upload CSV (only when ids resolve)
    csv_text, written = None, len(chosen)
    if has_ids:
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(SLOT)
        written = 0
        for row in chosen:
            lid = _lineup_ids(row, dkid, cands)
            if lid is None:
                continue
            w.writerow(lid)
            written += 1
        csv_text = out.getvalue()
        info["skipped_unmapped"] = info.get("skipped_unmapped", 0) + (len(chosen) - written)
        info["chosen"] = written

    lineups = [{
        "rank": int(row["Rank"]) if "Rank" in row else None,
        "candidate": int(row["Candidate"]),
        "stack": row["Stack"], "salary": int(row["Salary"]),
        "team": row.get("PrimaryTeam", ""),
        "win_pct": float(row["Win%"]), "top100_pct": float(row["Top100%"]),
        "players": _lineup_display(row, cands),
    } for row in chosen]

    player_expo, team_expo = _exposure(chosen, cands, payload["players_meta"],
                                       payload["pool_players"])
    n_lu = max(1, len(chosen))
    result = {
        "mode": mode,
        "has_ids": has_ids,
        "n_chosen": len(chosen),
        "n_written": written,
        "csv": csv_text,
        "lineups": lineups,
        "player_exposure": [
            {**p, "exposure": round(p["lineups"] / n_lu, 4)} for p in player_expo],
        "team_exposure": [
            {**t, "exposure": round(t["lineups"] / n_lu, 4)} for t in team_expo],
        "note": None if has_ids else
                "Slate carries no DraftKings ids — the portfolio is selected and "
                "shown, but the upload CSV needs a slate with ids (RotoWire feed) "
                "or a DKSalaries template.",
    }
    if mode == "ev":
        result["ev"] = {**extra, "returns": _return_hist(W, W_naive)}
    return result


def _export_ev(payload, src, n_select, dkid, eligible, caps, o):
    dist = payload.get("dist") or {}
    score_pool = payload.get("score_pool")
    if "field_cut_scores" not in dist or not score_pool:
        return None
    K = int(payload["K"]); field_n = int(payload["field_n"])
    cands = payload["cands"]
    cut_scores = dist["field_cut_scores"]; cut_places = dist["cut_places"]

    shortlist = int(o.get("shortlist", min(1000, len(src))))
    short = src.head(shortlist).reset_index(drop=True)
    M = len(short)
    if M == 0:
        return [], {"chosen": 0, "skipped_unmapped": 0}, None, None, {}

    cand_scores = np.zeros((K, M), np.float32)
    for j, cid in enumerate(short["Candidate"].to_numpy()):
        lu = cands[int(cid) - 1]
        for pl in lu["players"]:
            arr = score_pool.get(normname(pl.Name))
            if arr is not None:
                cand_scores[:, j] += arr

    prize = pev.make_payout_curve(
        field_n, float(o.get("entry_fee", 20.0)),
        top_heaviness=float(o.get("top_heaviness", 0.9)),
        pct_paid=float(o.get("pct_paid", 0.20)),
        rake=float(o.get("rake", 0.15)))
    pay = pev.candidate_payout_matrix(cand_scores, cut_scores, cut_places, prize)

    risk = o.get("risk", "Balanced")
    chosen, info, W = select_portfolio_ev(
        short, n_select, pay, pev.utility(risk), cols=COLS, hitc=HITC,
        eligible=eligible, eval_sims=int(o.get("eval_sims", 4000)), **caps)

    naive_pos = []
    for i in range(M):
        nms = [str(short.iloc[i][c]).rsplit(" (", 1)[0] for c in COLS]
        if eligible is None or eligible(nms):
            naive_pos.append(i)
        if len(naive_pos) >= info["chosen"]:
            break
    W_naive = (pay[:, naive_pos].sum(axis=1) if naive_pos
               else np.zeros(K, np.float64))

    extra = {"prize_summary": pev.payout_curve_summary(prize, float(o.get("entry_fee", 20.0))),
             "cost": info["chosen"] * float(o.get("entry_fee", 20.0)),
             "shortlist": M, "field_n": field_n, "risk": risk}
    return chosen, info, W, W_naive, extra


def _exposure(chosen, cands, meta, pool_players):
    pset = {m for m in pool_players}  # not used for pos; kept for parity
    pexpo, texpo = Counter(), Counter()
    for row in chosen:
        for nm in _row_names(row, cands):
            pexpo[nm] += 1
        t = row.get("PrimaryTeam", "")
        if t:
            texpo[t] += 1
    players = []
    for nm, ct in pexpo.most_common():
        m = meta.get(nm, {})
        players.append({"player": nm, "pos": m.get("pos", ""),
                        "team": m.get("team", ""), "lineups": int(ct)})
    teams = [{"team": t, "lineups": int(ct)} for t, ct in texpo.most_common()]
    return players, teams


def _return_hist(W, W_naive, nbins=40):
    """Overlaid per-slate $-return histograms for the EV vs ranked sets."""
    if W is None or W_naive is None or len(W) == 0:
        return None
    lo = float(min(W.min(), W_naive.min()))
    hi = float(max(W.max(), W_naive.max()))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cw, _ = np.histogram(W, bins=edges)
    cn, _ = np.histogram(W_naive, bins=edges)
    return {
        "mean_ev": round(float(W.mean()), 2),
        "mean_ranked": round(float(W_naive.mean()), 2),
        "bins": [{"x": round(float(x), 2), "ev": int(a), "ranked": int(b)}
                 for x, a, b in zip(centers, cw, cn)],
    }


# --------------------------------------------------------------------------- #
# Showdown export (1 CPT + 5 UTIL). Uses showdown_portfolio; the captain scores
# 1.5x in the EV per-sim rebuild. The DK upload CSV needs a DKSalaries showdown
# template (captain ids aren't in the slate feed), so it is deferred — the
# portfolio + exposure are still selected and shown.
# --------------------------------------------------------------------------- #
def _sd_cand_scores(short, payload):
    score_pool = payload["score_pool"]; cands = payload["cands"]
    K = int(payload["K"]); M = len(short)
    mat = np.zeros((K, M), np.float32)
    for j, cid in enumerate(short["Candidate"].to_numpy()):
        lu = cands[int(cid) - 1]
        for i, pl in enumerate(lu["players"]):
            arr = score_pool.get(normname(pl.Name))
            if arr is not None:
                mat[:, j] += (sb.CPT_MULT * arr) if i == 0 else arr
    return mat


def _sd_caps(o):
    # UI reuses hitter_cap/pitcher_cap -> player_cap/captain_cap for showdown.
    return dict(
        player_cap=float(o.get("hitter_cap", 1.0)),
        captain_cap=float(o.get("pitcher_cap", 1.0)),
        team_cap=float(o.get("team_cap", 1.0)),
        max_overlap=float(o.get("max_overlap", 1.0)),
        group_cap=float(o.get("group_cap", 1.0)),
    )


def _run_export_showdown(payload: dict, o: dict) -> dict:
    res = payload["res"]
    cands = payload["cands"]
    ids = o.get("candidate_ids")
    src = res
    if ids:
        keep = set(int(x) for x in ids)
        src = res[res["Candidate"].isin(keep)].reset_index(drop=True)
    if len(src) == 0:
        return {"error": "No candidates to export (empty selection)."}

    n_select = int(o.get("n_select", 20))
    caps = _sd_caps(o)
    mode = o.get("mode", "ranked")
    extra: dict = {}

    if mode == "ev":
        dist = payload.get("dist") or {}
        if "field_cut_scores" not in dist or not payload.get("score_pool"):
            return {"error": "This run predates the payout ladder; use ranked export."}
        shortlist = int(o.get("shortlist", min(1000, len(src))))
        short = src.head(shortlist).reset_index(drop=True)
        cand_scores = _sd_cand_scores(short, payload)
        prize = pev.make_payout_curve(
            int(payload["field_n"]), float(o.get("entry_fee", 20.0)),
            top_heaviness=float(o.get("top_heaviness", 0.9)),
            pct_paid=float(o.get("pct_paid", 0.20)), rake=float(o.get("rake", 0.15)))
        pay = pev.candidate_payout_matrix(
            cand_scores, dist["field_cut_scores"], dist["cut_places"], prize)
        chosen, info, W = select_showdown_portfolio_ev(
            short, n_select, pay, pev.utility(o.get("risk", "Balanced")),
            eval_sims=int(o.get("eval_sims", 4000)), **caps)
        naive = W_naive_showdown(short, pay, info["chosen"])
        extra = {"prize_summary": pev.payout_curve_summary(prize, float(o.get("entry_fee", 20.0))),
                 "cost": info["chosen"] * float(o.get("entry_fee", 20.0)),
                 "shortlist": len(short), "field_n": int(payload["field_n"]),
                 "risk": o.get("risk", "Balanced")}
        W_out = W; W_naive_out = naive
    else:
        sort_by = o.get("sort_by", "Top100 Rate")
        keymap = _SORT_KEYMAP.get(sort_by, _SORT_KEYMAP["Top100 Rate"])
        chosen, info = select_showdown_portfolio(src, n_select, keymap, **caps)
        W_out = W_naive_out = None

    lineups = [{
        "rank": int(row["Rank"]) if "Rank" in row else None,
        "candidate": int(row["Candidate"]),
        "stack": row.get("Split", ""), "salary": int(row["Salary"]),
        "team": row.get("CptTeam", ""), "captain": row.get("Captain", ""),
        "win_pct": float(row["Win%"]), "top100_pct": float(row["Top100%"]),
        "players": [{"slot": sb.SD_COLS[i], "player": _split_cell(row[c])[0],
                     "team": _split_cell(row[c])[1]}
                    for i, c in enumerate(sb.SD_COLS)],
    } for row in chosen]

    # exposure by player + by captain team
    from collections import Counter as _C
    pexpo, texpo = _C(), _C()
    for row in chosen:
        for c in sb.SD_COLS:
            pexpo[_split_cell(row[c])[0]] += 1
        if row.get("CptTeam"):
            texpo[row["CptTeam"]] += 1
    meta = payload["players_meta"]
    n_lu = max(1, len(chosen))
    player_expo = [{"player": nm, "pos": meta.get(nm, {}).get("pos", ""),
                    "team": meta.get(nm, {}).get("team", ""), "lineups": int(ct),
                    "exposure": round(ct / n_lu, 4)} for nm, ct in pexpo.most_common()]
    team_expo = [{"team": t, "lineups": int(ct), "exposure": round(ct / n_lu, 4)}
                 for t, ct in texpo.most_common()]

    result = {
        "mode": mode, "format": "showdown", "has_ids": False,
        "n_chosen": len(chosen), "n_written": len(chosen), "csv": None,
        "lineups": lineups, "player_exposure": player_expo, "team_exposure": team_expo,
        "note": "Showdown upload CSV needs a DraftKings showdown DKSalaries "
                "template (captain ids aren't in the slate feed) — the portfolio "
                "and exposure are selected and shown here.",
    }
    if mode == "ev":
        result["ev"] = {**extra, "returns": _return_hist(W_out, W_naive_out)}
    return result


def W_naive_showdown(short, pay, n):
    """Rank-selected baseline returns of the same size (all eligible)."""
    pos = list(range(min(int(n), pay.shape[1])))
    return pay[:, pos].sum(axis=1) if pos else np.zeros(pay.shape[0], np.float64)
