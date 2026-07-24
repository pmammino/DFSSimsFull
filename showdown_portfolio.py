#!/usr/bin/env python3
"""
showdown_portfolio.py
=====================
Diversity-aware portfolio selection for DraftKings **Showdown** lineups — the
showdown counterpart to ``portfolio.py`` (which is built around classic hitter
stacks). It keeps the same "rank / EV-greedy then fill under caps" shape and
reuses the format-agnostic primitives from ``portfolio.py`` (``_split``,
``_jaccard``, ``_unmet_mins``, ``detect_value_groups``), but the exposure axes
are the ones that matter in showdown:

  * per-player exposure          (player_cap)                    — any of 6 slots
  * per CAPTAIN exposure          (captain_cap)  — the defining showdown decision
  * per (majority) TEAM exposure  (team_cap)     — the lineup's dominant side
  * pairwise lineup OVERLAP       (max_overlap)  — no near-duplicate lineups
  * per VALUE-GROUP of near-twins (group_cap)

Per-entity overrides (player_caps / captain_caps / team_caps) and minimum-exposure
floors (player_mins / captain_mins / team_mins) work exactly like the classic
selector. Every control is OFF by default (caps = 1.0), so an uncapped call is a
plain rank-then-take-N.

Both selectors read lineups from a results DataFrame whose player cells are
``"Name (TEAM)"`` in the six showdown columns (CPT first) — see ``SD_COLS``.
"""
import math
from collections import Counter

import numpy as np

from portfolio import (_split, _jaccard, _unmet_mins, detect_value_groups,  # noqa: F401
                       _tie_banded_sort, value_group_member_caps,
                       shrink_value_group_means)

SD_COLS = ['CPT', 'UTIL1', 'UTIL2', 'UTIL3', 'UTIL4', 'UTIL5']


def lineup_features(row, cols=SD_COLS):
    """Pull the showdown selector's reasoning bits out of one result row: the six
    player names (CPT first), the player set, the captain, the captain's team, and
    the lineup's majority ('primary') team."""
    names, teams = [], []
    for c in cols:
        nm, tm = _split(row[c])
        names.append(nm)
        teams.append(tm)
    tc = Counter(t for t in teams if t)
    primary = tc.most_common(1)[0][0] if tc else ""
    return {
        "names": names,
        "playerset": frozenset(names),
        "captain": names[0],
        "cpt_team": teams[0],
        "primary": primary,
    }


def _cap_n(frac, N):
    f = float(frac)
    return 0 if f <= 0 else max(1, int(round(f * N)))


def _min_n(frac, N):
    f = float(frac)
    return 0 if f <= 0 else min(N, int(math.ceil(f * N)))


def _prep_caps(N, player_cap, captain_cap, team_cap, group_cap,
               player_caps, captain_caps, team_caps,
               player_mins, captain_mins, team_mins):
    pcap, ccap, tcap, gcap = (_cap_n(player_cap, N), _cap_n(captain_cap, N),
                              _cap_n(team_cap, N), _cap_n(group_cap, N))
    player_capn = {k: _cap_n(v, N) for k, v in (player_caps or {}).items()}
    captain_capn = {k: _cap_n(v, N) for k, v in (captain_caps or {}).items()}
    team_capn = {k: _cap_n(v, N) for k, v in (team_caps or {}).items()}

    player_minn = {k: _min_n(v, N) for k, v in (player_mins or {}).items()}
    captain_minn = {k: _min_n(v, N) for k, v in (captain_mins or {}).items()}
    team_minn = {k: _min_n(v, N) for k, v in (team_mins or {}).items()}
    # clamp a floor to the entity's own max so mins never conflict with caps
    player_minn = {k: min(v, player_capn.get(k, N)) for k, v in player_minn.items() if v > 0}
    captain_minn = {k: min(v, captain_capn.get(k, N)) for k, v in captain_minn.items() if v > 0}
    team_minn = {k: min(v, team_capn.get(k, N)) for k, v in team_minn.items() if v > 0}
    return (pcap, ccap, tcap, gcap, player_capn, captain_capn, team_capn,
            player_minn, captain_minn, team_minn)


def _info(chosen, N, skipped, expo, capexpo, teamc, groupc, player_minn,
          captain_minn, team_minn):
    return {
        "chosen": len(chosen), "requested": N, "skipped_unmapped": skipped,
        "max_player": max(expo.values()) if expo else 0,
        "max_captain": max(capexpo.values()) if capexpo else 0,
        "max_team": max(teamc.values()) if teamc else 0,
        "distinct_captains": len(capexpo),
        "distinct_teams": len(teamc),
        "player_expo": dict(expo),
        "captain_expo": dict(capexpo),
        "team_expo": dict(teamc),
        "unmet_mins": (_unmet_mins(player_minn, team_minn, expo, teamc)
                       + [{"kind": "captain", "name": nm, "have": int(capexpo[nm]),
                           "need": int(need)}
                          for nm, need in captain_minn.items() if capexpo[nm] < need]),
    }


# --------------------------------------------------------------------------- #
# Rank-based selection
# --------------------------------------------------------------------------- #
def select_showdown_portfolio(res_df, n_select, sort_cols, *, cols=SD_COLS,
                              eligible=None, player_cap=1.0, captain_cap=1.0,
                              team_cap=1.0, max_overlap=1.0, group_of=None,
                              group_cap=1.0, player_caps=None, captain_caps=None,
                              team_caps=None, player_mins=None, captain_mins=None,
                              team_mins=None, tie_sims=None, tie_seed=None):
    """Rank `res_df` by `sort_cols` (descending) then greedily accept lineups that
    keep every exposure cap and the overlap ceiling satisfied. Returns
    (chosen_rows, info).

    `tie_sims` / `tie_seed`, when both given, tie-band the ranking (see
    ``portfolio._tie_banded_sort``): a difference in the primary sort column below
    its Monte-Carlo standard error is treated as a tie and broken by a seeded
    shuffle, so a sub-noise projection edge stops funnelling every near-clone onto
    the same captain/player."""
    N = int(n_select)
    if tie_sims and tie_seed is not None and len(res_df):
        rdf = _tie_banded_sort(res_df, list(sort_cols), int(tie_sims), int(tie_seed))
    else:
        rdf = res_df.sort_values(list(sort_cols), ascending=False).reset_index(drop=True)
    (pcap, ccap, tcap, gcap, player_capn, captain_capn, team_capn,
     player_minn, captain_minn, team_minn) = _prep_caps(
        N, player_cap, captain_cap, team_cap, group_cap,
        player_caps, captain_caps, team_caps, player_mins, captain_mins, team_mins)
    group_of = group_of or {}

    expo = Counter(); capexpo = Counter(); teamc = Counter(); groupc = Counter()
    chosen, chosen_sets, chosen_idx, skipped = [], [], set(), 0
    feats = [lineup_features(rdf.iloc[i], cols) for i in range(len(rdf))]

    def gids(names):
        return {group_of[n] for n in names if n in group_of} if group_of else set()

    def fits(f):
        names = f["names"]
        if any(expo[n] >= player_capn.get(n, pcap) for n in names):
            return False
        if capexpo[f["captain"]] >= captain_capn.get(f["captain"], ccap):
            return False
        if teamc[f["primary"]] >= team_capn.get(f["primary"], tcap):
            return False
        if any(groupc[g] >= gcap for g in gids(names)):
            return False
        if max_overlap < 1.0 and chosen_sets:
            if max(_jaccard(f["playerset"], s) for s in chosen_sets) > max_overlap:
                return False
        return True

    def accept(pos, row, f):
        chosen.append(row); chosen_sets.append(f["playerset"]); chosen_idx.add(pos)
        for n in f["names"]:
            expo[n] += 1
        capexpo[f["captain"]] += 1
        teamc[f["primary"]] += 1
        for g in gids(f["names"]):
            groupc[g] += 1

    def deficits():
        return (any(expo[k] < v for k, v in player_minn.items())
                or any(capexpo[k] < v for k, v in captain_minn.items())
                or any(teamc[k] < v for k, v in team_minn.items()))

    def helps(f):
        if any(n in player_minn and expo[n] < player_minn[n] for n in f["names"]):
            return True
        if f["captain"] in captain_minn and capexpo[f["captain"]] < captain_minn[f["captain"]]:
            return True
        return f["primary"] in team_minn and teamc[f["primary"]] < team_minn[f["primary"]]

    # Phase 1: seed minimum-exposure targets in rank order
    if player_minn or captain_minn or team_minn:
        for pos, row in rdf.iterrows():
            if len(chosen) >= N or not deficits():
                break
            f = feats[pos]
            if eligible is not None and not eligible(f["names"]):
                continue
            if helps(f) and fits(f):
                accept(pos, row, f)

    # Phase 2: fill by rank
    for pos, row in rdf.iterrows():
        if len(chosen) >= N:
            break
        if pos in chosen_idx:
            continue
        f = feats[pos]
        if eligible is not None and not eligible(f["names"]):
            skipped += 1
            continue
        if fits(f):
            accept(pos, row, f)

    return chosen, _info(chosen, N, skipped, expo, capexpo, teamc, groupc,
                         player_minn, captain_minn, team_minn)


# --------------------------------------------------------------------------- #
# Payout-aware (EV) selection
# --------------------------------------------------------------------------- #
def select_showdown_portfolio_ev(res_df, n_select, pay, util, *, cols=SD_COLS,
                                 eligible=None, player_cap=1.0, captain_cap=1.0,
                                 team_cap=1.0, max_overlap=1.0, group_of=None,
                                 group_cap=1.0, player_caps=None, captain_caps=None,
                                 team_caps=None, player_mins=None, captain_mins=None,
                                 team_mins=None, eval_sims=None, tie_seed=None,
                                 pay_report=None):
    """Greedily build the export set that maximizes the expected *utility* of the
    portfolio's per-simulation dollar return, subject to the showdown exposure /
    diversity caps. Mirrors ``portfolio.select_portfolio_ev``; row order of
    `res_df` MUST align with the columns of `pay` (row i <-> pay[:, i]).
    Returns (chosen_rows, info, W).

    `pay_report`, when given, is an independent (held-out) payout matrix used only
    to compute the reported outcome stats and returned ``W`` — so the headline EV
    is out-of-sample, not measured on the sims the set was optimized against.
    `tie_seed`, when set, perturbs each step's marginal gains by their
    Monte-Carlo SE so near-twin lineups with indistinguishable gains alternate."""
    N = int(n_select)
    rdf = res_df.reset_index(drop=True)
    n_row = len(rdf)
    pay = np.asarray(pay, dtype=np.float32)
    if pay.shape[1] != n_row:
        raise ValueError(f"pay has {pay.shape[1]} cols but res_df has {n_row} rows")
    n_sim = pay.shape[0]

    if eval_sims and int(eval_sims) < n_sim:
        step = max(1, n_sim // int(eval_sims))
        sel_idx = np.arange(0, n_sim, step)[:int(eval_sims)]
    else:
        sel_idx = np.arange(n_sim)
    pay_sel = pay[sel_idx]

    (pcap, ccap, tcap, gcap, player_capn, captain_capn, team_capn,
     player_minn, captain_minn, team_minn) = _prep_caps(
        N, player_cap, captain_cap, team_cap, group_cap,
        player_caps, captain_caps, team_caps, player_mins, captain_mins, team_mins)
    group_of = group_of or {}

    feats = [lineup_features(rdf.iloc[i], cols) for i in range(n_row)]
    elig = np.ones(n_row, dtype=bool)
    if eligible is not None:
        for i in range(n_row):
            if not eligible(feats[i]["names"]):
                elig[i] = False
    skipped = int((~elig).sum())

    expo = Counter(); capexpo = Counter(); teamc = Counter(); groupc = Counter()
    chosen_pos, chosen_sets = [], []
    taken = np.zeros(n_row, dtype=bool)

    def gids(names):
        return {group_of[n] for n in names if n in group_of} if group_of else set()

    def fits(i):
        f = feats[i]; names = f["names"]
        if any(expo[n] >= player_capn.get(n, pcap) for n in names):
            return False
        if capexpo[f["captain"]] >= captain_capn.get(f["captain"], ccap):
            return False
        if teamc[f["primary"]] >= team_capn.get(f["primary"], tcap):
            return False
        if any(groupc[g] >= gcap for g in gids(names)):
            return False
        if max_overlap < 1.0 and chosen_sets:
            if max(_jaccard(f["playerset"], s) for s in chosen_sets) > max_overlap:
                return False
        return True

    def deficits():
        return (any(expo[k] < v for k, v in player_minn.items())
                or any(capexpo[k] < v for k, v in captain_minn.items())
                or any(teamc[k] < v for k, v in team_minn.items()))

    def helps(f):
        if any(n in player_minn and expo[n] < player_minn[n] for n in f["names"]):
            return True
        if f["captain"] in captain_minn and capexpo[f["captain"]] < captain_minn[f["captain"]]:
            return True
        return f["primary"] in team_minn and teamc[f["primary"]] < team_minn[f["primary"]]

    W_sel = np.zeros(len(sel_idx), dtype=np.float64)
    cur_u = float(np.mean(util(W_sel)))
    n_sel = len(sel_idx)
    for step in range(N):
        avail = elig & ~taken
        if not avail.any():
            break
        avail_idx = np.where(avail)[0]
        u_new = util(W_sel[:, None] + pay_sel[:, avail_idx])
        gains = u_new.mean(axis=0) - cur_u
        if tie_seed is not None and n_sel > 1:
            se = u_new.std(axis=0) / np.sqrt(n_sel)
            rng = np.random.default_rng(int(tie_seed) + step)
            gains = gains + rng.normal(0.0, np.where(se < 1e-12, 1e-12, se))
        order = np.argsort(-gains)
        picked = -1
        if deficits():
            for li in order:
                i = int(avail_idx[li])
                if fits(i) and helps(feats[i]):
                    picked = i
                    break
        if picked < 0:
            for li in order:
                i = int(avail_idx[li])
                if fits(i):
                    picked = i
                    break
        if picked < 0:
            break
        i = picked; f = feats[i]
        chosen_pos.append(i); chosen_sets.append(f["playerset"]); taken[i] = True
        W_sel += pay_sel[:, i]
        cur_u = float(np.mean(util(W_sel)))
        for n in f["names"]:
            expo[n] += 1
        capexpo[f["captain"]] += 1
        teamc[f["primary"]] += 1
        for g in gids(f["names"]):
            groupc[g] += 1

    chosen = [rdf.iloc[i] for i in chosen_pos]
    # report on the held-out payouts when supplied (out-of-sample EV)
    report = pay if pay_report is None else np.asarray(pay_report, dtype=np.float32)
    if report.shape[1] != n_row:
        raise ValueError(f"pay_report has {report.shape[1]} cols but res_df has {n_row}")
    W = (report[:, chosen_pos].sum(axis=1) if chosen_pos
         else np.zeros(report.shape[0], dtype=np.float64))
    info = _info(chosen, N, skipped, expo, capexpo, teamc, groupc,
                 player_minn, captain_minn, team_minn)
    info.update({
        "exp_return": float(W.mean()),
        "floor_p10": float(np.percentile(W, 10)),
        "median": float(np.percentile(W, 50)),
        "ceiling_p90": float(np.percentile(W, 90)),
        "cash_rate": float(np.mean(W > 0)),
    })
    return chosen, info, W


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pandas as pd
    from portfolio_ev import utility

    def mk(cpt, cpt_tm, utils, ranks):
        row = {'CPT': f"{cpt} ({cpt_tm})"}
        for i, (nm, tm) in enumerate(utils, 1):
            row[f'UTIL{i}'] = f"{nm} ({tm})"
        row.update(ranks)
        return row

    # six lineups all captained by the same stud -> captain_cap must break them up
    rows = []
    for i in range(6):
        rows.append(mk('Stud', 'AAA',
                       [('u1', 'AAA'), ('u2', 'AAA'), ('u3', 'BBB'),
                        ('u4', 'BBB'), ('u5', 'BBB')],
                       {'Wins': 100 - i, 'Top10': 0}))
    # one alternate captained by someone else
    rows.append(mk('Other', 'BBB',
                   [('u1', 'AAA'), ('u6', 'AAA'), ('u3', 'BBB'),
                    ('u4', 'BBB'), ('u7', 'BBB')], {'Wins': 40, 'Top10': 0}))
    df = pd.DataFrame(rows)

    # no caps -> top 5 all share the 'Stud' captain
    ch, info = select_showdown_portfolio(df, 5, ['Wins', 'Top10'])
    assert info["max_captain"] == 5, info

    # captain cap forces the alternate captain in
    ch, info = select_showdown_portfolio(df, 5, ['Wins', 'Top10'], captain_cap=0.8)
    assert info["max_captain"] <= 4, info
    assert info["distinct_captains"] >= 2, info

    # per-entity captain cap of 0 excludes a captain entirely
    ch, info = select_showdown_portfolio(df, 6, ['Wins', 'Top10'],
                                         captain_caps={'Stud': 0.0})
    assert info["captain_expo"].get('Stud', 0) == 0, info

    # captain minimum floor pulls in a captain rank alone would skip
    ch, info = select_showdown_portfolio(df, 5, ['Wins', 'Top10'],
                                         captain_mins={'Other': 0.2})
    assert info["captain_expo"].get('Other', 0) >= 1 and not info["unmet_mins"], info

    # overlap ceiling breaks up near-identical lineups
    ch, info = select_showdown_portfolio(df, 5, ['Wins', 'Top10'], max_overlap=0.8)
    assert info["distinct_captains"] >= 2, info

    # ---- EV selection: spread across sims via concave utility ----
    ev_rows = [
        mk('A', 'AAA', [('a1', 'AAA')] * 1 + [('x1', 'BBB'), ('x2', 'BBB'),
                        ('x3', 'BBB'), ('x4', 'BBB')], {}),
        mk('B', 'AAA', [('b1', 'AAA')] + [('y1', 'BBB'), ('y2', 'BBB'),
                        ('y3', 'BBB'), ('y4', 'BBB')], {}),
        mk('C', 'BBB', [('c1', 'BBB')] + [('z1', 'AAA'), ('z2', 'AAA'),
                        ('z3', 'AAA'), ('z4', 'AAA')], {}),
        mk('D', 'BBB', [('d1', 'BBB')] + [('w1', 'AAA'), ('w2', 'AAA'),
                        ('w3', 'AAA'), ('w4', 'AAA')], {}),
    ]
    ev_df = pd.DataFrame(ev_rows)
    ev_pay = np.array([[100., 100., 0., 0.],
                       [0.,   0.,  100., 0.]])
    chC, iC, WC = select_showdown_portfolio_ev(
        ev_df, 2, ev_pay, utility("Conservative (consistent cashing)"))
    assert iC["cash_rate"] == 1.0 and list(WC) == [100.0, 100.0], (iC, WC)
    chL, iL, WL = select_showdown_portfolio_ev(
        ev_df, 2, ev_pay, utility("Aggressive (max ceiling)"))
    assert iL["cash_rate"] == 0.5, iL

    # ---- held-out EV reporting: report on a disjoint matrix, not `pay` ----
    ev_report = np.zeros_like(ev_pay)
    _, iR, WR = select_showdown_portfolio_ev(
        ev_df, 2, ev_pay, utility("Aggressive (max ceiling)"), pay_report=ev_report)
    assert iR["exp_return"] == 0.0 and float(np.sum(WR)) == 0.0, (iR, WR)

    # ---- tie-banded ranking is reproducible ----
    tie_rows = [mk('Stud', 'AAA',
                   [('u1', 'AAA'), ('u2', 'AAA'), ('u3', 'BBB'),
                    ('u4', 'BBB'), ('u5', 'BBB')], {'Wins': 100, 'Top10': 0})
                for _ in range(6)]
    tie_df = pd.DataFrame(tie_rows)
    c1, _ = select_showdown_portfolio(tie_df, 3, ['Wins', 'Top10'],
                                      tie_sims=1000, tie_seed=2)
    c2, _ = select_showdown_portfolio(tie_df, 3, ['Wins', 'Top10'],
                                      tie_sims=1000, tie_seed=2)
    assert [r['CPT'] for r in c1] == [r['CPT'] for r in c2]

    print("showdown_portfolio.py self-test passed")
