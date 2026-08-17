#!/usr/bin/env python3
"""
stage_d.py  —  STAGE D: sims -> fields + candidates -> contest sims -> lineups
==============================================================================
Consumes Stage C output (the correlated DK sims) plus a DraftKings salary/
ownership file, and produces, for any set of contest sizes:

  * an ownership-weighted FIELD per size (contest-size chalk model),
  * a uniform, ownership-blind CANDIDATE pool,
  * contest results (Wins / Top10 / Top100 / AvgPlace per candidate),
  * optionally a DK upload file of the best N lineups under exposure caps.

It needs only two inputs: the sim .npy files and a DK file (FullName, Team,
Position, Salary, Ownership). Game matchups are inferred from the sims
themselves (a starter's scores anti-correlate hardest with the team he faces),
so no team-code reconciliation between feeds is required.

Example
-------
  python3 stage_d.py \
      --hitter-sims deliverables/hitter_dk_sims.npy \
      --pitcher-sims deliverables/pitcher_dk_sims.npy \
      --dk slateplayers.csv \
      --contest-sizes 1000 6000 20000 --num-candidates 10000 \
      --outdir stage_d_out \
      --select 20 --objective top100 --player-cap 0.60 --team-cap 0.50 \
      --dk-template DKSalaries.csv
"""
import argparse, csv, json, os, unicodedata, itertools
import numpy as np
import pandas as pd
from collections import Counter

from slate_config import canonical_team
from mlb_lineup_builder import Pool, Builder
from field_simulator import (normalize_to_slots, adjust_ownership,
                             beta_for_size, tilt_structures)
from stack_signal import team_stack_ownership, apply_stack_ownership_boost

COLS = ['P1','P2','C','1B','2B','3B','SS','OF1','OF2','OF3']
HITC = ['C','1B','2B','3B','SS','OF1','OF2','OF3']
SLOT = ['P','P','C','1B','2B','3B','SS','OF','OF','OF']

def norm(n):
    n = unicodedata.normalize('NFKD', str(n)).encode('ascii','ignore').decode()
    n = n.lower().replace('.','').replace(',','').replace("'","")
    for s in [' jr',' sr',' ii',' iii',' iv']:
        if n.endswith(s): n = n[:-len(s)]
    return n.strip()

def load_sims(hpath, ppath):
    H = np.load(hpath, allow_pickle=True).item()
    P = np.load(ppath, allow_pickle=True).item()
    score, n_sim = {}, None
    for d in (H, P):
        for k, v in d.items():
            a = np.asarray(v, np.float32); score[norm(k)] = a; n_sim = len(a)
    return H, P, score, n_sim

def derive_opponents(dk_df, H, P):
    """Infer team->opponent from sims: a pitcher's scores anti-correlate most
    with the team he faces."""
    team = {norm(r.FullName): r.Team for r in dk_df.itertuples()}
    Hm = {norm(k): np.asarray(v, float) for k, v in H.items() if norm(k) in team}
    Pm = {norm(k): np.asarray(v, float) for k, v in P.items() if norm(k) in team}
    teams = sorted({team[k] for k in Hm})
    tvec = {t: np.mean([Hm[k] for k in Hm if team[k] == t], axis=0) for t in teams}
    opp = {}
    for pk in Pm:
        pt = team[pk]
        cs = {t: np.corrcoef(Pm[pk], tvec[t])[0,1] for t in teams if t != pt}
        if cs: opp[pt] = min(cs, key=cs.get)
    return opp

def _sim_key_for(full_name, team, simset):
    """The score-dict key for a DK player, honouring same-name collisions.

    A colliding player is keyed by ``"<name> (<CANON_TEAM>)"`` on BOTH the sim
    side (from the slate team) and here (from the DK team), canonicalised so the
    two reconcile. Try the team-qualified key first; fall back to the plain name.
    Returns (display_name, key) or (None, None) when neither is simmed — which
    fail-safe-drops a collision whose team didn't match any sim, so a player is
    never scored with a different player's array. `display_name` is what the pool
    row is keyed by so it stays unique per real player; dk_ids strips the suffix
    back off for upload."""
    disamb_disp = f"{full_name} ({canonical_team(team) or str(team or '').upper()})"
    if norm(disamb_disp) in simset:
        return disamb_disp, norm(disamb_disp)
    if norm(full_name) in simset:
        return full_name, norm(full_name)
    return None, None


def build_pool(dk_path, H, P, score):
    """DK file + sims -> pool restricted to simmed players. Multi-pos expanded,
    SP/RP->P, sim-confirmed pitchers flagged starters, opponents from sims.

    Same-name collisions (two real players, e.g. the Dodgers' vs the Athletics'
    Max Muncy) are matched to their OWN sim by a team-qualified key, so each pool
    row carries a distinct name and the field can no longer roster one wearing
    the other's projection. A duplicate whose team matches no sim is dropped."""
    dk = pd.read_csv(dk_path, encoding='latin-1'); dk['FullName'] = dk['FullName'].str.strip()
    opp = derive_opponents(dk, H, P)
    simset = set(score)
    rows = []
    for r in dk.itertuples():
        disp, _key = _sim_key_for(r.FullName, r.Team, simset)
        if disp is None:
            continue
        for p in str(r.Position).split('/'):
            p = p.strip(); pos = 'P' if p in ('SP','RP') else p
            role = 'SP' if pos == 'P' else pos      # sim pitchers are starters
            rows.append({'Name': disp, 'Pos': pos, 'Role': role,
                         'Team': r.Team, 'Opp': opp.get(r.Team, ''),
                         'Salary': int(r.Salary), 'Ownership': float(r.Ownership)})
    pool = pd.DataFrame(rows).drop_duplicates(['Name','Pos'])
    return pool

def lineups_to_df(lineups):
    out = []
    for i, lu in enumerate(lineups, 1):
        r = {'Lineup': i, 'Salary': lu['salary'],
             'Stack': '-'.join(map(str, sorted(lu['teams'].values(), reverse=True)))}
        for j, pl in enumerate(lu['players']): r[COLS[j]] = f"{pl.Name} ({pl.Team})"
        out.append(r)
    return pd.DataFrame(out)

def _score_matrix_loop(lineups, score, n_sim):
    # Fallback: fill the (n_sim, n_lineups) result in place (no column_stack copy).
    out = np.zeros((n_sim, len(lineups)), np.float32)
    for j, lu in enumerate(lineups):
        col = out[:, j]
        for pl in lu['players']:
            col += score[norm(pl.Name)]
    return out


def score_matrix(lineups, score, n_sim):
    """Total DK points per (sim, lineup): out[s, j] = sum over lineup j's players
    of their per-sim score. Vectorized as a sparse (lineups x players) incidence
    matrix times the (players x sims) score stack — one BLAS-backed matmul, much
    faster than a Python loop over every lineup on large candidate/field sets, and
    without the column_stack peak-memory spike. Falls back to the explicit loop if
    SciPy isn't importable. Result matches the loop to float32 rounding."""
    if not lineups:
        return np.zeros((n_sim, 0), np.float32)
    try:
        from scipy import sparse
    except Exception:
        return _score_matrix_loop(lineups, score, n_sim)
    name_idx = {nm: i for i, nm in enumerate(score)}
    S = np.empty((len(name_idx), n_sim), np.float32)      # (players x sims)
    for nm, i in name_idx.items():
        S[i] = score[nm]
    rows, cols = [], []
    for j, lu in enumerate(lineups):
        for pl in lu['players']:
            rows.append(j)
            cols.append(name_idx[norm(pl.Name)])          # KeyError if unscored, as before
    M = sparse.csr_matrix(
        (np.ones(len(rows), np.float32),
         (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
        shape=(len(lineups), len(name_idx)))
    return (M @ S).T                                      # (sims x lineups)


def incidence(lineups, name_index):
    """Sparse (n_lineups x n_players) 0/1 incidence matrix under `name_index`
    (norm(name) -> column). Build it ONCE per lineup set and reuse it across sim
    slices via :func:`score_from_incidence`, so the field/candidates are only
    scored on the sims each held-out slice needs — never the full (K x N) matrix.
    Returns None if SciPy is unavailable (callers fall back to the dense path)."""
    try:
        from scipy import sparse
    except Exception:
        return None
    rows, cols = [], []
    for j, lu in enumerate(lineups):
        for pl in lu['players']:
            rows.append(j)
            cols.append(name_index[norm(pl.Name)])
    return sparse.csr_matrix(
        (np.ones(len(rows), np.float32),
         (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
        shape=(len(lineups), len(name_index)))


def score_from_incidence(M, lineups, name_index, score, sim_index):
    """(len(sim_index) x n_lineups) point totals for just the sims in `sim_index`.
    Equivalent to ``score_matrix(lineups, score, K)[sim_index]`` but it never
    materializes the full-K matrix — the memory driver on large contests. `M` is
    the incidence from :func:`incidence` (or None → dense fallback)."""
    n = len(sim_index)
    if M is None:
        out = np.zeros((n, len(lineups)), np.float32)
        for j, lu in enumerate(lineups):
            col = out[:, j]
            for pl in lu['players']:
                col += score[norm(pl.Name)][sim_index]
        return out
    S = np.empty((len(name_index), n), np.float32)
    for nm, i in name_index.items():
        S[i] = score[nm][sim_index]
    return (M @ S).T

def run_contest(field_mat, cand_mat, n_sim, N_FIELD):
    N = cand_mat.shape[1]
    wins=np.zeros(N,np.int64); t10=np.zeros(N,np.int64); t100=np.zeros(N,np.int64); ps=np.zeros(N,np.int64)
    for s in range(n_sim):
        fs = np.sort(field_mat[s]); cv = cand_mat[s]
        place = (N_FIELD - np.searchsorted(fs, cv, side='right')) + 1
        wins += (place==1); t10 += (place<=10); t100 += (place<=100); ps += place
    return wins, t10, t100, ps / n_sim

def main():
    ap = argparse.ArgumentParser(description="Stage D: sims -> fields/candidates -> contests.")
    ap.add_argument('--hitter-sims', required=True)
    ap.add_argument('--pitcher-sims', required=True)
    ap.add_argument('--dk', required=True, help='DK file: FullName,Team,Position,Salary,Ownership')
    ap.add_argument('--contest-sizes', nargs='+', type=int, default=[1000, 6000, 20000])
    ap.add_argument('--num-candidates', type=int, default=10000)
    ap.add_argument('--medium', type=int, default=6000)
    ap.add_argument('--chalk-sensitivity', type=float, default=0.35)
    ap.add_argument('--stack-tilt', type=float, default=0.15)
    ap.add_argument('--stack-boost', type=float, default=0.05,
                    help='Stack-ownership ceiling boost: in each team\'s high-end '
                         'sims, scale its hitters\' DK points up by a factor that '
                         'grows with projected stack ownership (0 = off).')
    ap.add_argument('--params', default='field_params.json')
    ap.add_argument('--seed-field', type=int, default=101)
    ap.add_argument('--seed-candidates', type=int, default=2025)
    ap.add_argument('--outdir', default='stage_d_out')
    # optional DK upload selection
    ap.add_argument('--select', type=int, default=0, help='select N lineups for a DK upload (0=skip)')
    ap.add_argument('--objective', choices=['top100','top10','win'], default='top100')
    ap.add_argument('--from-size', type=int, default=None, help='which contest size to select from')
    ap.add_argument('--player-cap', type=float, default=1.0)
    ap.add_argument('--team-cap', type=float, default=1.0)
    ap.add_argument('--dk-template', default=None, help='DKSalaries template for upload IDs')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    params = json.load(open(a.params))
    H, P, score, n_sim = load_sims(a.hitter_sims, a.pitcher_sims)
    pool = build_pool(a.dk, H, P, score)
    nh = pool[pool.Pos!='P'].Name.nunique(); npi = pool[pool.Pos=='P'].Name.nunique()
    print(f"pool: {nh} hitters + {npi} starters, {pool.Team.nunique()} teams, {n_sim} sims")

    # stack-ownership upside signal: nudge popular stacks' high-end outcomes up.
    # The same boosted sims score both the field and the candidates.
    if a.stack_boost > 0:
        hp = pool[pool.Pos != 'P']
        names_by_team, own_by_name = {}, {}
        for r in hp.itertuples():
            nn = norm(r.Name)
            names_by_team.setdefault(r.Team, set()).add(nn)
            own_by_name[nn] = float(r.Ownership)
        names_by_team = {t: sorted(ns) for t, ns in names_by_team.items()}
        stack_own = team_stack_ownership(names_by_team, own_by_name)
        score = apply_stack_ownership_boost(score, names_by_team, stack_own, n_sim,
                                            strength=a.stack_boost, quantile=0.80)
        print(f"stack-ownership ceiling boost={a.stack_boost:g} applied to "
              f"{len(names_by_team)} teams")

    # candidates once (uniform, starters only)
    cdf = pool[(pool.Pos!='P') | (pool.Role=='SP')].copy(); cdf['Ownership'] = 1.0
    cb = Builder(Pool(cdf), params, seed=a.seed_candidates, uniform=True)
    cands = []
    while len(cands) < a.num_candidates:
        lu = cb.build_one()
        if lu: cands.append(lu)
    cand_df = lineups_to_df(cands); cand_df.to_csv(f"{a.outdir}/candidates_{a.num_candidates}.csv", index=False)
    cand_mat = score_matrix(cands, score, n_sim)
    print(f"candidates: {len(cands)} built and scored")

    results_by_size = {}
    for N in a.contest_sizes:
        beta = beta_for_size(N, a.medium, a.chalk_sensitivity)
        fdf = adjust_ownership(normalize_to_slots(pool, 0.15), beta=beta)
        tilted = tilt_structures([(tuple(s),w) for s,w in params['stack_structures']], N, a.medium, a.stack_tilt)
        fp = dict(params); fp['stack_structures'] = [(list(s),w) for s,w in tilted]
        fb = Builder(Pool(fdf), fp, seed=a.seed_field, uniform=False)
        field = []
        while len(field) < N:
            lu = fb.build_one()
            if lu: field.append(lu)
        lineups_to_df(field).to_csv(f"{a.outdir}/field_{N}.csv", index=False)
        field_mat = score_matrix(field, score, n_sim)
        wins, t10, t100, avg = run_contest(field_mat, cand_mat, n_sim, N)
        res = cand_df.copy(); res.insert(0, 'Candidate', np.arange(1, len(cands)+1))
        res['Wins']=wins; res['Win%']=np.round(100*wins/n_sim,3)
        res['Top10']=t10; res['Top10%']=np.round(100*t10/n_sim,2)
        res['Top100']=t100; res['Top100%']=np.round(100*t100/n_sim,2)
        res['AvgPlace']=np.round(avg,1)
        res = res.sort_values(['Wins','Top10','Top100','AvgPlace'], ascending=[False,False,False,True])
        res.to_csv(f"{a.outdir}/candidate_results_{N}.csv", index=False)
        results_by_size[N] = res
        print(f"[field {N:>6}] beta={beta:.2f}  best win {int(wins.max())}  "
              f"win>=1 {int((wins>0).sum())}  best top100% {res['Top100%'].max():.2f}")

    # optional DK upload
    if a.select > 0:
        size = a.from_size or a.medium
        if size not in results_by_size: size = a.contest_sizes[0]
        make_upload(results_by_size[size], a, size)

def _split_cell(cell):
    """A result cell is 'Name (TEAM)'; return (name, team)."""
    s = str(cell)
    if s.endswith(')') and ' (' in s:
        nm, tm = s.rsplit(' (', 1)
        return nm, tm[:-1]
    return s, ''


def make_upload(res, a, size):
    keymap = {'top100':['Top100','Top10','Wins'], 'top10':['Top10','Top100','Wins'],
              'win':['Wins','Top10','Top100']}[a.objective]
    res = res.sort_values(keymap, ascending=False).reset_index(drop=True)
    import dk_ids
    dkid = {}
    if a.dk_template:
        rows = list(csv.reader(open(a.dk_template, encoding='utf-8', errors='replace')))
        hdr = next(i for i,r in enumerate(rows) if len(r)>=20 and r[11]=='Position')
        for r in rows[hdr+1:]:
            # col 11=Position, 13=Name, 14=ID, 16=Salary, 18=TeamAbbrev — keyed by
            # salary/team/pos so same-named players keep distinct ids
            if len(r)>=20 and r[14].strip():
                dk_ids.add_id(dkid, r[13], r[18], r[14], pos=r[11], salary=r[16])
    N = a.select; pcap = int(a.player_cap*N); tcap = int(a.team_cap*N)
    def cells_of(row): return [_split_cell(row[c]) for c in COLS]
    def names_of(row): return [nm for nm,_ in cells_of(row)]
    def prim(row):
        c = Counter(str(row[x]).rsplit(' (',1)[1][:-1] for x in HITC if ' (' in str(row[x]))
        return c.most_common(1)[0][0]
    expo=Counter(); teamc=Counter(); chosen=[]
    for _, row in res.iterrows():
        nms = names_of(row)
        if dkid and any(not dk_ids.has_name(dkid, n) for n in nms): continue
        if all(expo[n] < pcap for n in nms) and teamc[prim(row)] < tcap:
            chosen.append(row)
            for n in nms: expo[n]+=1
            teamc[prim(row)] += 1
        if len(chosen) == N: break
    if dkid:
        with open(f"{a.outdir}/DK_upload_{N}.csv",'w',newline='') as f:
            w=csv.writer(f); w.writerow(SLOT)
            for row in chosen:
                ids = [dk_ids.lookup(dkid, nm, tm, pos=SLOT[i])
                       for i,(nm,tm) in enumerate(cells_of(row))]
                if all(ids): w.writerow(ids)
    pd.DataFrame(chosen).to_csv(f"{a.outdir}/selected_{N}.csv", index=False)
    print(f"selected {len(chosen)} by {a.objective} | max player {max(expo.values())}/{N} "
          f"| max stack-team {max(teamc.values())}/{N}"
          + (f" | wrote DK_upload_{N}.csv" if dkid else " (no template -> ids skipped)"))

if __name__ == '__main__':
    main()
